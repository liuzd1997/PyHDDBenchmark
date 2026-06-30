"""
HDD Notch Filter Auto Design Tool - Simple Version
Automated notch filter design for HDD control systems based on reinforcement learning (Iterative & Randomized)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt
import control.matlab as matlab
from control import freqresp
import scipy.signal as signal
from scipy.optimize import differential_evolution
import sys
import os
import copy
import argparse
from datetime import datetime

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
import plant
import utils
import config


class NotchFilterParams:
    """Notch filter parameters"""
    def __init__(self, center_freq, bandwidth, depth):
        self.center_freq = center_freq
        self.bandwidth = bandwidth
        self.depth = depth
        # Q factor is calculated from center_freq and bandwidth
        self.q_factor = center_freq / bandwidth


class SimpleHDDNotchDesignEnv:
    """Simple HDD Notch Filter Design Environment (Iterative & Randomized)"""
    
    def __init__(self, env_config):
        self.config = env_config
        
        # System parameters
        try:
            self.Ts = float(plant.Ts)
        except Exception:
            self.Ts = 1.9841e-05
        try:
            self.Mr_f = int(plant.Mr_f)
        except Exception:
            self.Mr_f = 2
        
        # Plant cases (reuse original set for diversity)
        self.plant_cases = [
            ('c1', plant.Sys_Pc_vcm_c1, plant.Sys_Pc_pzt_c1),
            ('c2', plant.Sys_Pc_vcm_c2, plant.Sys_Pc_pzt_c2),
            ('c3', plant.Sys_Pc_vcm_c3, plant.Sys_Pc_pzt_c3),
            ('c4', plant.Sys_Pc_vcm_c4, plant.Sys_Pc_pzt_c4),
            ('c5', plant.Sys_Pc_vcm_c5, plant.Sys_Pc_pzt_c5),
            ('c6', plant.Sys_Pc_vcm_c6, plant.Sys_Pc_pzt_c6),
            ('c7', plant.Sys_Pc_vcm_c7, plant.Sys_Pc_pzt_c7),
            ('c8', plant.Sys_Pc_vcm_c8, plant.Sys_Pc_pzt_c8),
            ('c9', plant.Sys_Pc_vcm_c9, plant.Sys_Pc_pzt_c9),
        ]
        
        # Current Plant (will be randomized per episode)
        self.current_case = None
        self.current_vcm_plant = None
        self.current_pzt_plant = None
        
        # Controllers
        if hasattr(utils, 'Sys_Cd_vcm'):
            self.Sys_Cd_vcm = utils.Sys_Cd_vcm
        elif hasattr(utils, 'get_Sys_Cd_vcm'):
            self.Sys_Cd_vcm = utils.get_Sys_Cd_vcm()
        else:
            raise AttributeError('No VCM controller found in utils')

        if hasattr(utils, 'Sys_Cd_pzt'):
            self.Sys_Cd_pzt = utils.Sys_Cd_pzt
        elif hasattr(utils, 'get_Sys_Cd_pzt'):
            self.Sys_Cd_pzt = utils.get_Sys_Cd_pzt()
        else:
            raise AttributeError('No PZT controller found in utils')
        
        # Multi-rate filters
        if hasattr(utils, 'Sys_Fm_vcm'):
            self.Sys_Fm_vcm = utils.Sys_Fm_vcm
        elif hasattr(utils, 'get_Sys_Fm_vcm'):
            self.Sys_Fm_vcm = utils.get_Sys_Fm_vcm()
        else:
            raise AttributeError('No VCM multirate filter found in utils')

        if hasattr(utils, 'Sys_Fm_pzt'):
            self.Sys_Fm_pzt = utils.Sys_Fm_pzt
        elif hasattr(utils, 'get_Sys_Fm_pzt'):
            self.Sys_Fm_pzt = utils.get_Sys_Fm_pzt()
        else:
            raise AttributeError('No PZT multirate filter found in utils')
        
        # Reward weights and targets
        self.weights = env_config['weights']
        self.targets = env_config['targets']
        self.reward_scale = float(env_config.get('reward_scale', 100.0))
        self.frequency_delta_decades = float(env_config.get('frequency_delta_decades', 0.12))
        self.delta_max = float(env_config.get('delta_max', 0.12))
        self.notches_per_channel = int(env_config.get('notches_per_channel', 2))
        self.params_per_notch = 3
        self.param_dim = 2 * self.notches_per_channel * self.params_per_notch
        
        # Closed-loop metrics include the slow-rate controller, so they must stay
        # below the slow-rate Nyquist frequency.
        slow_nyquist_freq = 1.0 / self.Ts / 2.0
        max_freq = min(float(env_config.get('max_closed_loop_freq', 25000.0)), slow_nyquist_freq * 0.98)
        num_freq_points = int(env_config.get('num_freq_points', 800))
        self.freq_range = np.logspace(1, np.log10(max_freq), num_freq_points)
        self.omega = 2 * np.pi * self.freq_range
        
        # Precompute Controller Frequency Responses (Fixed)
        self.Cd_vcm_fr = self._ensure_1d_fr(utils.freqresp(self.Sys_Cd_vcm, self.omega))
        self.Cd_pzt_fr = self._ensure_1d_fr(utils.freqresp(self.Sys_Cd_pzt, self.omega))
        
        self.Fm_vcm_fr = self._ensure_1d_fr(utils.freqresp(self.Sys_Fm_vcm, self.omega))
        self.Fm_pzt_fr = self._ensure_1d_fr(utils.freqresp(self.Sys_Fm_pzt, self.omega))
        
        self.param_low, self.param_high = self._load_action_bounds(env_config.get('action_bounds'))
        self.param_range = np.clip(self.param_high - self.param_low, 1e-6, None)
        # State and action spaces
        self.observation_space = self._create_observation_space()
        self.action_space = self._create_action_space()
        
        # Precompute no-notch plant responses and fast-rate chains for reuse.
        self.plant_fr_cache = {}
        self.plant_chain_cache = {}
        self._precompute_plant_responses()

        # Optional DE warm-start: maps case_name → notch_params array.
        # Populated by load_initial_params(); None means start from midpoint.
        self.initial_params_by_case = None
        self.init_noise = float(env_config.get('init_noise', 0.15))

        # Initialize state variables
        self.current_notch_params = self._midpoint_params()
        self.plant_features = np.zeros(30)  # Base plant features (static)
        self.current_system_features = np.zeros(15)  # Current sensitivity features (dynamic: 5 peaks * 3 values)
        self.current_performance = np.array([0.0])  # Only sensitivity_peak
        self.current_vcm_peak_refs = []
        self.current_pzt_peak_refs = []
        
        # Placeholders for current system FRs
        self.current_vcm_fr_with_notch = None
        self.current_pzt_fr_with_notch = None
        self.current_sensitivity_fr = None
        
    def _create_observation_space(self):
        """Create observation space"""
        # State: [Base_Plant_Features(30), Current_Sensitivity_Features(15), Current_Notch_Params, Sensitivity_Peak(1)]
        # Base_Plant_Features: Static features from P*Fm (for reference - tells agent what peaks to suppress)
        # Current_Sensitivity_Features: Dynamic sensitivity peaks (tells agent current system state and notch effectiveness)
        # Sensitivity_Peak: Current sensitivity peak value (directly related to reward)
        return {
            'shape': (30 + 15 + self.param_dim + 1,),
            'low': -np.inf,
            'high': np.inf
        }
    
    def _ensure_1d_fr(self, fr):
        """Ensure frequency response arrays are 1D (nfreq,)"""
        fr = np.asarray(fr)
        fr = np.squeeze(fr)
        if fr.ndim == 0:
            return np.array([fr], dtype=complex)
        if fr.ndim > 1:
            # Take the first response if multiple exist (we only expect SISO)
            fr = fr.reshape(fr.shape[0], -1)[0]
        return fr
    
    def _freqresp_1d(self, system):
        """Frequency response helper that returns 1D complex array"""
        response = utils.freqresp(system, self.omega)
        return self._ensure_1d_fr(response)

    def _ss_tuple(self, system):
        """Return dense SISO state-space matrices as a tuple."""
        return (
            np.asarray(system.A, dtype=np.float64),
            np.asarray(system.B, dtype=np.float64),
            np.asarray(system.C, dtype=np.float64),
            np.asarray(system.D, dtype=np.float64),
        )

    def _series_ss(self, sys1, sys2):
        """State-space realization for sys1 * sys2, matching python-control series order."""
        if sys2 is None:
            return sys1
        if sys1 is None:
            return sys2

        A1, B1, C1, D1 = sys1
        A2, B2, C2, D2 = sys2
        n1 = A1.shape[0]
        n2 = A2.shape[0]

        A = np.block([
            [A1, B1 @ C2],
            [np.zeros((n2, n1)), A2],
        ])
        B = np.vstack([B1 @ D2, B2])
        C = np.hstack([C1, D1 @ C2])
        D = D1 @ D2
        return A, B, C, D

    def _resample_ss(self, sys_ss):
        """Match utils.dts_resampling for cached dense state-space matrices."""
        A, B, C, D = sys_ss
        Az = A.copy()
        Bz = B.copy()
        for _ in range(1, self.Mr_f):
            Bz = Bz + Az @ B
            Az = Az @ A
        return Az, Bz, C, D

    def _freqresp_ss(self, sys_ss):
        """Fast SISO discrete state-space frequency response on self.omega."""
        A, B, C, D = sys_ss
        sys = signal.dlti(A, B, C, D, dt=self.Ts)
        _, response = signal.dfreqresp(sys, w=self.omega * self.Ts)
        return np.asarray(response, dtype=np.complex128)

    def _notch_ss(self, f0, bw, depth_db):
        """Create a discrete notch state-space tuple at the fast sample rate."""
        if depth_db >= 0:
            return None

        ts = self.Ts / self.Mr_f
        w0 = 2 * np.pi * f0
        zeta = bw / (2 * f0)
        depth_lin = max(10 ** (depth_db / 20.0), 1e-4)
        num = [1.0, 2.0 * zeta * w0 * depth_lin, w0**2]
        den = [1.0, 2.0 * zeta * w0, w0**2]
        b, a, _ = signal.cont2discrete((num, den), ts, method='zoh')
        A, B, C, D = signal.tf2ss(b.ravel(), a)
        return (
            np.asarray(A, dtype=np.float64),
            np.asarray(B, dtype=np.float64),
            np.asarray(C, dtype=np.float64),
            np.asarray(D, dtype=np.float64),
        )
    
    def _build_notch_filter_tf(self, f0, bw, depth_db):
        """Create discrete-time notch filter using original multirate pipeline"""
        sample_time = self.Ts / self.Mr_f
        if depth_db >= 0:
            return matlab.tf([1.0], [1.0], sample_time)
        
        w0 = 2 * np.pi * f0
        zeta = bw / (2 * f0)
        depth_lin = max(10 ** (depth_db / 20.0), 1e-4)
        
        # Standard Notch: Num damping < Den damping for attenuation
        num = [1, 2 * zeta * w0 * depth_lin, w0**2]
        den = [1, 2 * zeta * w0, w0**2]
        notch_ct = matlab.tf(num, den)
        return matlab.c2d(notch_ct, sample_time, 'zoh')
    
    def _create_digital_path(self, sys_pc, sys_fm, notch_tf=None):
        """Create discrete-time plant path with multirate filters and optional notch"""
        sys_pdm0 = matlab.c2d(sys_pc, self.Ts / self.Mr_f, 'zoh')
        sys_chain = sys_pdm0 * sys_fm
        if notch_tf is not None:
            sys_chain = sys_chain * notch_tf
        sys_pd = utils.dts_resampling(sys_chain, self.Mr_f)
        return sys_pd

    def _create_digital_path_from_chain(self, sys_chain, notch_tf=None):
        """Resample a cached fast-rate plant/filter chain to the controller rate."""
        if notch_tf is not None:
            sys_chain = sys_chain * notch_tf
        return utils.dts_resampling(sys_chain, self.Mr_f)
    
    def _load_action_bounds(self, action_bounds):
        """Load physical notch parameter bounds"""
        if action_bounds is None:
            # Default bounds matching NotchFilterConfig ranges
            # VCM: freq(5k-45k), bw(100-5k), depth(-60-0)
            # PZT: freq(10k-47k), bw(100-5k), depth(-60-0)
            low_6 = np.array([5000.0, 100.0, -60.0, 10000.0, 100.0, -60.0], dtype=np.float32)
            high_6 = np.array([45000.0, 5000.0, 0.0, 47000.0, 5000.0, 0.0], dtype=np.float32)
            low, high = self._expand_base_bounds(low_6, high_6)
            return low, high
        low = np.array(action_bounds['low'], dtype=np.float32)
        high = np.array(action_bounds['high'], dtype=np.float32)
        if low.shape == (6,) and high.shape == (6,):
            low, high = self._expand_base_bounds(low, high)
        if low.shape != (self.param_dim,) or high.shape != (self.param_dim,):
            raise ValueError(f"Action bounds must provide 6 or {self.param_dim} low/high values.")
        return low, high

    def _expand_base_bounds(self, low_6, high_6):
        """Expand [VCM notch, PZT notch] bounds to multiple notches per channel."""
        vcm_low, pzt_low = low_6[:3], low_6[3:]
        vcm_high, pzt_high = high_6[:3], high_6[3:]
        low = np.concatenate(
            [np.tile(vcm_low, self.notches_per_channel), np.tile(pzt_low, self.notches_per_channel)]
        ).astype(np.float32)
        high = np.concatenate(
            [np.tile(vcm_high, self.notches_per_channel), np.tile(pzt_high, self.notches_per_channel)]
        ).astype(np.float32)
        return low, high
    
    def _midpoint_params(self):
        """Return mid-point parameters within physical bounds (Geometric mean for Log indices)"""
        params = np.zeros(self.param_dim)
        
        for i in range(self.param_dim):
            low = self.param_low[i]
            high = self.param_high[i]
            
            if i % 3 in (0, 1):
                # Geometric mean
                params[i] = np.sqrt(low * high)
            else:
                # Arithmetic mean
                params[i] = (low + high) / 2.0
                
        return params
    
    def _create_action_space(self):
        """Create action space"""
        # Action: normalized parameters mapped to physical ranges
        return {
            'shape': (self.param_dim,),
            'low': np.array([-1.0] * self.param_dim),
            'high': np.array([1.0] * self.param_dim)
        }
    
    def _action_to_params(self, action):
        """Map normalized action [-1,1] to physical notch parameters.

        Frequency actions are local offsets around the current plant's main
        no-notch peak. Bandwidth and depth still use the global physical ranges.
        """
        action = np.clip(action, self.action_space['low'], self.action_space['high'])
        normalized = (action + 1.0) / 2.0  # [0,1]
        
        params = np.zeros_like(normalized, dtype=np.float64)

        for i in range(self.param_dim):
            low = self.param_low[i]
            high = self.param_high[i]
            field = i % 3
            channel_offset = 0 if i < self.notches_per_channel * 3 else self.notches_per_channel * 3
            notch_idx = (i - channel_offset) // 3

            if field == 0:
                refs = self.current_vcm_peak_refs if channel_offset == 0 else self.current_pzt_peak_refs
                fallback = np.sqrt(low * high)
                ref = refs[notch_idx] if notch_idx < len(refs) and refs[notch_idx] > 0 else fallback
                params[i] = np.clip(
                    ref * (10.0 ** (action[i] * self.frequency_delta_decades)),
                    low,
                    high,
                )
            elif field == 1:
                log_low = np.log10(low)
                log_high = np.log10(high)
                params[i] = 10**(log_low + normalized[i] * (log_high - log_low))
            else:
                params[i] = low + normalized[i] * (high - low)
                
        return params

    def _params_to_action(self, params):
        """Inverse of _action_to_params for supervised warm-start targets."""
        params = np.asarray(params, dtype=np.float64)
        action = np.zeros(self.param_dim, dtype=np.float32)

        for i in range(self.param_dim):
            low = self.param_low[i]
            high = self.param_high[i]
            field = i % 3
            channel_offset = 0 if i < self.notches_per_channel * 3 else self.notches_per_channel * 3
            notch_idx = (i - channel_offset) // 3
            if field == 0:
                refs = self.current_vcm_peak_refs if channel_offset == 0 else self.current_pzt_peak_refs
                fallback = np.sqrt(low * high)
                ref = refs[notch_idx] if notch_idx < len(refs) and refs[notch_idx] > 0 else fallback
                action[i] = np.log10(np.clip(params[i], low, high) / ref) / self.frequency_delta_decades
            elif field == 1:
                log_low = np.log10(low)
                log_high = np.log10(high)
                n = (np.log10(np.clip(params[i], low, high)) - log_low) / (log_high - log_low)
                action[i] = 2.0 * n - 1.0
            else:
                n = (np.clip(params[i], low, high) - low) / (high - low)
                action[i] = 2.0 * n - 1.0

        return np.clip(action, -1.0, 1.0)
    
    def _precompute_plant_responses(self):
        """Precompute no-notch full-pipeline responses for all plant cases."""
        print("Precomputing full-pipeline plant responses for optimization...")
        fast_ts = self.Ts / self.Mr_f
        
        for case_name, base_vcm, base_pzt in self.plant_cases:
            sys_pdm0_vcm = matlab.c2d(base_vcm, fast_ts, 'zoh')
            sys_chain_vcm = sys_pdm0_vcm * self.Sys_Fm_vcm
            chain_vcm_ss = self._ss_tuple(sys_chain_vcm)
            fr_vcm = self._freqresp_ss(self._resample_ss(chain_vcm_ss))

            sys_pdm0_pzt = matlab.c2d(base_pzt, fast_ts, 'zoh')
            sys_chain_pzt = sys_pdm0_pzt * self.Sys_Fm_pzt
            chain_pzt_ss = self._ss_tuple(sys_chain_pzt)
            fr_pzt = self._freqresp_ss(self._resample_ss(chain_pzt_ss))
            
            self.plant_fr_cache[case_name] = (fr_vcm, fr_pzt)
            self.plant_chain_cache[case_name] = (chain_vcm_ss, chain_pzt_ss)

    def load_initial_params(self, npz_path):
        """Warm-start each episode from DE-optimized notch params.

        Loads a .npz produced by --mode optimize and stores the notch_params
        as the per-case starting point for reset().  A small random perturbation
        (controlled by init_noise) is applied each reset so the agent still
        explores different refinements rather than always refining the same point.

        npz plant_case == 'all'  → same params used for every case.
        npz plant_case == 'c2'   → only case c2 is warm-started; others midpoint.
        """
        data = np.load(npz_path)
        params = np.asarray(data['notch_params'], dtype=np.float64)
        plant_case = str(data.get('plant_case', 'all'))

        self.initial_params_by_case = {}
        if plant_case == 'all':
            for case_name, _, _ in self.plant_cases:
                self.initial_params_by_case[case_name] = params.copy()
        else:
            self.initial_params_by_case[plant_case] = params.copy()

        print(f"Loaded DE warm-start from: {os.path.abspath(npz_path)}")
        print(f"  plant_case={plant_case}, "
              f"params={format_notch_params(params, self.notches_per_channel)}")

    def _compute_notch_fr(self, f0, bw, depth_db):
        """Compute Notch Filter Frequency Response analytically (Fast)"""
        ts = self.Ts / self.Mr_f
        w0 = 2 * np.pi * f0
        zeta = bw / (2 * f0)
        
        # Clip depth to avoid issues
        depth_lin = max(10 ** (depth_db / 20.0), 1e-6)
        
        # Coefficients for Continuous Notch
        # To achieve attenuation (Notch), we want reduced gain at resonance.
        # Standard Notch: (s^2 + 2*zeta*w0*depth_lin*s + w0^2) / (s^2 + 2*zeta*w0*s + w0^2)
        # where depth_lin < 1.
        
        # Numerator has reduced damping (creating the null)
        num = [1.0, 2.0 * zeta * w0 * depth_lin, w0**2]
        # Denominator has standard damping (defining the recovery bandwidth)
        den = [1.0, 2.0 * zeta * w0, w0**2]
        
        # Use Scipy to get discrete coefficients (ZOH)
        # Returns (num, den, dt) when input is (num, den)
        res = signal.cont2discrete((num, den), ts, method='zoh')
        b = res[0].ravel() # Ensure 1D
        a = res[1]
        
        # Compute frequency response at self.omega
        # freqz expects normalized frequency in [0, pi)
        # w_digital = w_physical * ts
        w_digital = self.omega * ts
        
        # signal.freqz returns (w, h)
        _, h = signal.freqz(b, a, worN=w_digital)
        
        return h

    def _select_plant_case(self, case_name):
        """Select a plant case and refresh cached feature state."""
        self.current_case = case_name
        
        # Removed random gain scaling
        self.current_gain_scale = 1.0
        self.current_base_fr = self.plant_fr_cache[case_name]
        self.current_fast_chain = self.plant_chain_cache[case_name]
        
        # Calculate plant features from Base FR (No Notch)
        # Base_FR is the slow-rate, resampled plant path without notch.
        self.base_vcm_fr = self.current_base_fr[0]
        self.base_pzt_fr = self.current_base_fr[1]
        
        vcm_peaks = self._find_peaks(self.base_vcm_fr)
        pzt_peaks = self._find_peaks(self.base_pzt_fr)
        self.current_vcm_peak_refs = self._peak_refs(vcm_peaks, self.param_low[:3], self.param_high[:3])
        pzt_start = self.notches_per_channel * 3
        self.current_pzt_peak_refs = self._peak_refs(pzt_peaks, self.param_low[pzt_start:pzt_start + 3], self.param_high[pzt_start:pzt_start + 3])

        self.plant_features = self._extract_plant_features_from_fr(self.base_vcm_fr, self.base_pzt_fr)
        
        # Legacy placeholders not used in optimized evaluation
        self.current_vcm_plant = None
        self.current_pzt_plant = None

    def _peak_refs(self, peaks, low_3, high_3):
        f_low, f_high = float(low_3[0]), float(high_3[0])
        refs = [p['freq'] for p in peaks if f_low <= p['freq'] <= f_high]
        fallback = float(np.sqrt(low_3[0] * high_3[0]))
        while len(refs) < self.notches_per_channel:
            refs.append(fallback)
        return refs[:self.notches_per_channel]

    def _randomize_plant(self):
        """Randomize plant by selecting a case (Gain scaling removed)"""
        case_idx = np.random.randint(0, len(self.plant_cases))
        case_name = self.plant_cases[case_idx][0]
        self._select_plant_case(case_name)

    def reset(self):
        """Reset environment"""
        # 1. Randomize Plant
        self._randomize_plant()

        # 2. Set starting notch params.
        # If DE warm-start is loaded, begin near the optimizer solution for the
        # current case (with small noise so the agent explores refinements).
        # Otherwise fall back to the mid-range geometric mean.
        if (self.initial_params_by_case is not None
                and self.current_case in self.initial_params_by_case):
            base_params = self.initial_params_by_case[self.current_case]
            base_norm = self._normalize_notch_params(base_params)
            noise = np.random.uniform(-self.init_noise, self.init_noise, self.param_dim)
            start_norm = np.clip(base_norm + noise, 0.0, 1.0)
            self.current_notch_params = self._denormalize_params(start_norm)
        else:
            self.current_notch_params = self._midpoint_params()
        
        # 3. Calculate Initial Performance
        performance = self._evaluate_current_system()
        self.current_performance = np.array([performance['sensitivity_peak']])
        
        # 4. Extract current system features (dynamic, reflects notch effect)
        self.current_system_features = self._extract_current_system_features(
            self.current_vcm_fr_with_notch,
            self.current_pzt_fr_with_notch,
            self.current_sensitivity_fr
        )
        
        # 5. Construct State
        state = np.concatenate([
            self.plant_features,  # Base plant features (static reference)
            self.current_system_features,  # Current system features (dynamic)
            self._normalize_notch_params(self.current_notch_params),
            self._normalize_performance(self.current_performance)
        ])
        
        return state
    
    def step(self, action):
        """Execute action as a delta move in normalized param space.

        Each call nudges the current notch parameters by action * delta_max
        in the [0,1] normalized space, then converts back to physical units.
        This enables iterative RL: the agent sees the effect of each move
        before deciding the next one.
        """
        action = np.clip(action, self.action_space['low'], self.action_space['high'])
        prev_sp = float(self.current_performance[0])  # sensitivity peak before this move
        current_norm = self._normalize_notch_params(self.current_notch_params)
        new_norm = np.clip(current_norm + action * self.delta_max, 0.0, 1.0)
        self.current_notch_params = self._denormalize_params(new_norm)

        # 2. Evaluate System
        performance = self._evaluate_current_system()

        # 3. Calculate Reward + per-step improvement shaping.
        # The base reward captures absolute quality; the shaping term rewards
        # the agent for reducing sensitivity peak this specific step, giving a
        # dense gradient signal for iterative delta-action RL.
        reward = self._calculate_reward(performance)
        new_sp = float(performance['sensitivity_peak'])
        improvement_bonus = max(0.0, prev_sp - new_sp) * 2.0 / self.reward_scale
        reward = float(np.clip(reward + improvement_bonus, -100.0, 10.0))
        
        # 4. Update State
        self.current_performance = np.array([performance['sensitivity_peak']])
        
        # Extract current system features (dynamic, reflects notch effect)
        self.current_system_features = self._extract_current_system_features(
            self.current_vcm_fr_with_notch,
            self.current_pzt_fr_with_notch,
            self.current_sensitivity_fr
        )
        
        next_state = np.concatenate([
            self.plant_features,  # Base plant features (static reference)
            self.current_system_features,  # Current system features (dynamic)
            self._normalize_notch_params(self.current_notch_params),
            self._normalize_performance(self.current_performance)
        ])
        
        # 5. Check Done
        done = self._is_done(performance)
        
        info = {
            'performance': performance,
            'notch_params': self.current_notch_params,
            'case': self.current_case
        }
        
        return next_state, reward, done, info

    def _evaluate_current_system(self):
        """Evaluate the current notch using the full multirate plant path."""
        if not hasattr(self, 'current_fast_chain') or self.current_fast_chain is None:
            raise RuntimeError("Plant must be randomized before evaluation.")

        vcm_params = self._channel_params(self.current_notch_params, 'vcm')
        pzt_params = self._channel_params(self.current_notch_params, 'pzt')
        sys_pd_vcm = self._resample_ss(self._series_ss(self.current_fast_chain[0], self._cascade_notch_ss(vcm_params)))
        sys_pd_pzt = self._resample_ss(self._series_ss(self.current_fast_chain[1], self._cascade_notch_ss(pzt_params)))

        Fr_Pd_vcm = self._freqresp_ss(sys_pd_vcm) * self.current_gain_scale
        Fr_Pd_pzt = self._freqresp_ss(sys_pd_pzt) * self.current_gain_scale
        
        L_vcm = Fr_Pd_vcm * self.Cd_vcm_fr
        L_pzt = Fr_Pd_pzt * self.Cd_pzt_fr
        L_total = L_vcm + L_pzt
        
        S = 1.0 / (1.0 + L_total)
        
        # Store current system FRs for state extraction
        self.current_vcm_fr_with_notch = Fr_Pd_vcm
        self.current_pzt_fr_with_notch = Fr_Pd_pzt
        self.current_sensitivity_fr = S
        
        return self._calculate_performance_metrics(L_total, S)

    def _channel_params(self, params, channel):
        params = np.asarray(params, dtype=np.float64)
        if channel == 'vcm':
            start = 0
        elif channel == 'pzt':
            start = self.notches_per_channel * 3
        else:
            raise ValueError(f"Unknown channel: {channel}")
        return params[start:start + self.notches_per_channel * 3].reshape(self.notches_per_channel, 3)

    def _cascade_notch_ss(self, notch_params):
        cascade = None
        for f0, bw, depth_db in notch_params:
            notch = self._notch_ss(f0, bw, depth_db)
            cascade = self._series_ss(cascade, notch)
        return cascade

    def _extract_plant_features_from_fr(self, vcm_fr, pzt_fr):
        """Extract features from frequency response"""
        vcm_peaks = self._find_peaks(vcm_fr)
        pzt_peaks = self._find_peaks(pzt_fr)
        
        features = []
        for p in vcm_peaks:
            features.extend([p['freq']/50000.0, p['mag']/60.0, p['phase']/180.0])
        for p in pzt_peaks:
            features.extend([p['freq']/50000.0, p['mag']/60.0, p['phase']/180.0])
            
        return np.array(features, dtype=np.float32)
    
    def _extract_sensitivity_features(self, sensitivity_fr):
        """Extract features from sensitivity function (current system state)"""
        """This captures the current system's sensitivity peaks, which is more informative than static plant features"""
        mag_db = 20 * np.log10(np.abs(sensitivity_fr) + 1e-12)
        
        # Find sensitivity peaks (peaks in magnitude)
        peaks = []
        for i in range(1, len(mag_db)-1):
            if mag_db[i] > mag_db[i-1] and mag_db[i] > mag_db[i+1] and mag_db[i] > -40:  # Threshold
                peaks.append({
                    'freq': self.freq_range[i],
                    'mag': mag_db[i],
                    'phase': np.angle(sensitivity_fr[i]) * 180 / np.pi
                })
        
        peaks.sort(key=lambda x: x['mag'], reverse=True)
        peaks = peaks[:5]  # Top 5 peaks
        while len(peaks) < 5:
            peaks.append({'freq': 0, 'mag': -60, 'phase': 0})
        
        # Extract features: [freq1, mag1, phase1, freq2, mag2, phase2, ...]
        features = []
        for p in peaks:
            features.extend([
                p['freq']/50000.0,  # Normalized frequency
                (p['mag'] + 40) / 40.0,  # Normalized magnitude: -40 to 0 dB -> 0 to 1
                p['phase']/180.0  # Normalized phase
            ])
        
        return np.array(features, dtype=np.float32)
    
    def _extract_current_system_features(self, vcm_fr_with_notch, pzt_fr_with_notch, sensitivity_fr):
        """Extract features from current system (plant + notch)"""
        """This provides dynamic state information that reflects the effect of current notch"""
        # Option 1: Full features (45 dim): VCM + PZT + Sensitivity peaks
        # Option 2: Sensitivity only (15 dim): Only sensitivity peaks (most relevant to reward)
        # Option 3: Reduced peaks (27 dim): 3 peaks each instead of 5
        
        # Using Option 2: Only sensitivity peaks (most directly related to reward)
        # This reduces state from 86 to 56 dimensions while keeping the most important dynamic info
        sens_features = self._extract_sensitivity_features(sensitivity_fr)
        
        return sens_features  # 15 dimensions (5 peaks * 3 values)

    def _find_peaks(self, fr):
        """Find top 5 peaks"""
        mag = 20 * np.log10(np.abs(fr) + 1e-12)
        phase = np.angle(fr) * 180 / np.pi
        
        peaks = []
        # Simple peak finding
        for i in range(1, len(mag)-1):
            if mag[i] > mag[i-1] and mag[i] > mag[i+1] and mag[i] > -20: # Threshold
                peaks.append({'freq': self.freq_range[i], 'mag': mag[i], 'phase': phase[i]})
        
        peaks.sort(key=lambda x: x['mag'], reverse=True)
        peaks = peaks[:5]
        while len(peaks) < 5:
            peaks.append({'freq': 0, 'mag': 0, 'phase': 0})
            
        return peaks

    def _normalize_notch_params(self, params):
        """Normalize params to [0, 1] for state (Log scale for Freq/BW)"""
        norm = np.zeros_like(params)
        
        for i in range(len(params)):
            low = self.param_low[i]
            high = self.param_high[i]
            val = params[i]
            
            if i % 3 in (0, 1):
                # Inverse of log mapping
                log_low = np.log10(low)
                log_high = np.log10(high)
                n = (np.log10(val) - log_low) / (log_high - log_low)
                norm[i] = n
            else:
                n = (val - low) / (high - low)
                norm[i] = n
                
        return np.clip(norm, 0.0, 1.0)

    def _denormalize_params(self, norm):
        """Inverse of _normalize_notch_params: [0,1] → physical space."""
        params = np.zeros(self.param_dim, dtype=np.float64)
        for i in range(self.param_dim):
            low = float(self.param_low[i])
            high = float(self.param_high[i])
            n = float(np.clip(norm[i], 0.0, 1.0))
            if i % 3 in (0, 1):  # freq or bw: log scale
                log_low = np.log10(low)
                log_high = np.log10(high)
                params[i] = 10.0 ** (log_low + n * (log_high - log_low))
            else:  # depth: linear
                params[i] = low + n * (high - low)
        return params

    def _normalize_performance(self, perf):
        """Normalize performance metrics - only sensitivity_peak"""
        # Sensitivity peak: typically ranges from -20 to 20 dB, normalize to [0, 1]
        # Using range -20 to 20 dB -> normalized to [0, 1]
        norm = (perf[0] + 20.0) / 40.0
        return np.array([np.clip(norm, 0.0, 1.0)])

    # Reuse existing calculation methods
    def _calculate_performance_metrics(self, Fr_L, Fr_S):
        # Same as before, but ensure inputs are 1D
        Fr_L = np.squeeze(Fr_L)
        Fr_S = np.squeeze(Fr_S)
        if Fr_L.ndim > 1: Fr_L = Fr_L[0]
        if Fr_S.ndim > 1: Fr_S = Fr_S[0]
        
        pm = self._calculate_phase_margin(Fr_L)
        gm = self._calculate_gain_margin(Fr_L)
        sp = np.max(20*np.log10(np.abs(Fr_S) + 1e-12))
        te = np.mean(20*np.log10(np.abs(Fr_S) + 1e-12)) # Simple proxy
        return_difference_min = float(np.min(np.abs(1.0 + Fr_L)))
        
        # This frequency-domain proxy is more reliable for this workflow than
        # using PM/GM alone, which can be misleading with multiple crossovers.
        finite = np.all(np.isfinite(Fr_L)) and np.all(np.isfinite(Fr_S))
        stable = 1.0 if finite and sp < 12.0 and return_difference_min > 0.15 else 0.0
        
        return {
            'phase_margin': float(pm),
            'gain_margin': float(gm),
            'sensitivity_peak': float(sp),
            'stability': float(stable),
            'tracking_error': float(te),
            'return_difference_min': float(return_difference_min)
        }

    def _calculate_phase_margin(self, Fr_L):
        mag = np.abs(Fr_L)
        phase = np.unwrap(np.angle(Fr_L)) * 180 / np.pi
        phase_margins = []
        for i in range(len(mag) - 1):
            if (mag[i] - 1.0) * (mag[i + 1] - 1.0) <= 0 and mag[i] != mag[i + 1]:
                frac = (1.0 - mag[i]) / (mag[i + 1] - mag[i])
                p = phase[i] + frac * (phase[i + 1] - phase[i])
                pm = 180.0 + p
                # Fold the unwrapped value into [-180, 180] to remove the
                # multi-rotation ambiguity that plagued the earlier version.
                pm = ((pm + 180.0) % 360.0) - 180.0
                phase_margins.append(pm)
        if phase_margins:
            return float(min(phase_margins))
        return float('inf')

    def _calculate_gain_margin(self, Fr_L):
        mag = np.abs(Fr_L)
        phase = np.unwrap(np.angle(Fr_L)) * 180 / np.pi
        gain_margins = []
        for i in range(len(phase) - 1):
            if (phase[i] + 180.0) * (phase[i + 1] + 180.0) <= 0 and phase[i] != phase[i + 1]:
                frac = (phase[i] - (-180)) / (phase[i] - phase[i+1] + 1e-12)
                m = mag[i] + frac*(mag[i+1]-mag[i])
                gain_margins.append(-20*np.log10(m + 1e-12))
        if gain_margins:
            return float(min(gain_margins))
        return float('inf')

    def _calculate_reward(self, performance):
        """Reward as the negative of the constrained notch-design loss."""
        reward = -self._objective_from_performance(performance, self.current_notch_params) / self.reward_scale
        return float(np.clip(reward, -100.0, 10.0))

    def _objective_from_performance(self, performance, notch_params):
        """Lower is better: sensitivity peak plus control and notch-cost penalties.

        Phase margin is excluded because the unwrapped-phase estimate is
        unreliable for this multi-rate dual-loop system.  Stability is
        instead captured by return_difference_min and gain_margin.
        """
        sp = float(performance['sensitivity_peak'])
        gm = float(performance['gain_margin'])
        return_difference_min = float(performance.get('return_difference_min', 0.0))

        gm_min = float(self.targets.get('gain_margin', 6.0))
        sp_target = float(self.targets.get('sensitivity_peak', 3.0))

        loss = sp
        loss += 10.0 * max(0.0, sp - sp_target) ** 2
        if np.isfinite(gm):
            loss += 0.5 * max(0.0, gm_min - gm) ** 2
        loss += 40.0 * max(0.0, 0.30 - return_difference_min) ** 2

        notch_params = np.asarray(notch_params, dtype=np.float64).reshape(-1, 3)
        total_bw = float(np.sum(notch_params[:, 1]))
        total_depth = float(np.sum(np.abs(notch_params[:, 2])))
        loss += 0.15 * (total_bw / (5000.0 * len(notch_params)))
        loss += 0.02 * (total_depth / (60.0 * len(notch_params)))

        if performance['stability'] < 0.5:
            # Soft cliff: large enough to penalize instability but not so large
            # that 30-step episodes diverge uncontrollably.
            loss += 20.0
        return float(loss)
    

    def _is_done(self, performance):
        # PM is excluded (unreliable estimate for this system); use GM + RDM.
        gm = performance['gain_margin']
        gm_ok = (not np.isfinite(gm)) or gm >= self.targets['gain_margin']
        sp_ok = performance['sensitivity_peak'] <= self.targets['sensitivity_peak']
        stable = performance['stability'] > 0.8
        rdm_ok = performance.get('return_difference_min', 0.0) > 0.25
        return gm_ok and sp_ok and stable and rdm_ok


# PPO Policy Network (Updated for new dimensions)
class PPOPolicy(nn.Module):
    """PPO Policy Network"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Shared network
        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Policy network (mean)
        self.policy_mean = nn.Linear(hidden_dim, action_dim)
        
        # Policy network (std)
        self.policy_std = nn.Linear(hidden_dim, action_dim)
        
        # Value network
        self.value_net = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use different gains for different layers
                if m == self.policy_mean:
                    # Policy mean: initialize to output near zero (will be scaled to action space)
                    nn.init.orthogonal_(m.weight, gain=0.01)
                    nn.init.constant_(m.bias, 0)
                elif m == self.policy_std:
                    # Policy std: initialize to produce reasonable std
                    nn.init.orthogonal_(m.weight, gain=0.01)
                    # Initialize bias to produce log_std around -1 (std ~0.37)
                    nn.init.constant_(m.bias, -1.0)
                elif m == self.value_net:
                    # Value network: standard initialization
                    nn.init.orthogonal_(m.weight, gain=1.0)
                    nn.init.constant_(m.bias, 0)
                else:
                    # Shared network: standard initialization
                    nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, state, action_low=None, action_high=None):
        """Forward pass"""
        shared_features = self.shared_net(state)
        
        # Policy output - use tanh to map to [-1, 1], then scale to action space
        mean_raw = self.policy_mean(shared_features)
        mean_tanh = torch.tanh(mean_raw)
        
        # Scale to action space if bounds provided, otherwise keep in [-1, 1]
        if action_low is not None and action_high is not None:
            # Convert to tensors if numpy arrays
            if isinstance(action_low, np.ndarray):
                action_low_t = torch.tensor(action_low, dtype=torch.float32, device=mean_tanh.device)
                action_high_t = torch.tensor(action_high, dtype=torch.float32, device=mean_tanh.device)
            else:
                action_low_t = torch.tensor(action_low, dtype=torch.float32, device=mean_tanh.device)
                action_high_t = torch.tensor(action_high, dtype=torch.float32, device=mean_tanh.device)
            
            action_mid = (action_low_t + action_high_t) / 2.0
            action_range = (action_high_t - action_low_t) / 2.0
            mean = action_mid + mean_tanh * action_range
        else:
            mean = mean_tanh
        
        # Policy std produced as a log-std; clamp for numerical stability
        log_std = self.policy_std(shared_features)
        log_std = torch.clamp(log_std, -20.0, 2.0)
        std = torch.exp(log_std)
        
        # Scale std proportionally to action range
        if action_low is not None and action_high is not None:
            # Convert to tensors if numpy arrays
            if isinstance(action_low, np.ndarray):
                action_low_t = torch.tensor(action_low, dtype=torch.float32, device=std.device)
                action_high_t = torch.tensor(action_high, dtype=torch.float32, device=std.device)
            else:
                action_low_t = torch.tensor(action_low, dtype=torch.float32, device=std.device)
                action_high_t = torch.tensor(action_high, dtype=torch.float32, device=std.device)
            
            action_range = (action_high_t - action_low_t) / 2.0
            std = std * action_range
        
        # Value output
        value = self.value_net(shared_features)
        
        return mean, std, value
    
    def get_action(self, state, action_low=None, action_high=None, deterministic=False):
        """Get action"""
        mean, std, value = self.forward(state, action_low, action_high)
        
        if deterministic:
            return mean, value
        
        dist = Normal(mean, std)
        action = dist.sample()
        if action_low is not None and action_high is not None:
            action_low_t = torch.tensor(action_low, dtype=torch.float32, device=action.device)
            action_high_t = torch.tensor(action_high, dtype=torch.float32, device=action.device)
            action = torch.max(torch.min(action, action_high_t), action_low_t)
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob, value


class PPOAgent:
    """PPO Agent"""
    
    def __init__(self, state_dim, action_dim, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Policy network
        self.policy = PPOPolicy(state_dim, action_dim, config.hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.learning_rate)
        
        # PPO parameters
        self.clip_ratio = config.clip_ratio
        self.value_loss_coef = config.value_loss_coef
        self.entropy_coef = config.entropy_coef
        self.max_grad_norm = config.max_grad_norm
        
        # Experience buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
    def get_action(self, state, deterministic=False):
        """Get action"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get action bounds if available
        action_low = None
        action_high = None
        if hasattr(self, 'action_low') and hasattr(self, 'action_high'):
            action_low = self.action_low
            action_high = self.action_high
        
        with torch.no_grad():
            if deterministic:
                action, value = self.policy.get_action(state_tensor, action_low, action_high, deterministic=True)
                act_np = action.cpu().numpy()[0]
                # Final clip to ensure within bounds (shouldn't be needed but safety check)
                if action_low is not None and action_high is not None:
                    act_np = np.clip(act_np, np.array(action_low), np.array(action_high))
                return act_np, value.cpu().numpy()[0]
            else:
                action, log_prob, value = self.policy.get_action(state_tensor, action_low, action_high)
                act_np = action.cpu().numpy()[0]
                # Final clip to ensure within bounds (shouldn't be needed but safety check)
                if action_low is not None and action_high is not None:
                    act_np = np.clip(act_np, np.array(action_low), np.array(action_high))
                return act_np, log_prob.cpu().numpy()[0], value.cpu().numpy()[0]
    
    def store_transition(self, state, action, reward, value, log_prob, done):
        """Store transition"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def update(self):
        """Update policy"""
        if len(self.states) < self.config.batch_size:
            return
        
        # Convert to tensors
        states = torch.as_tensor(np.asarray(self.states, dtype=np.float32), device=self.device)
        actions = torch.as_tensor(np.asarray(self.actions, dtype=np.float32), device=self.device)
        old_log_probs = torch.as_tensor(np.asarray(self.log_probs, dtype=np.float32), device=self.device)
        old_values = torch.as_tensor(np.asarray(self.values, dtype=np.float32), device=self.device)
        
        # Calculate discounted rewards
        rewards = self._compute_discounted_rewards()
        
        # Calculate advantages
        advantages = rewards - old_values.squeeze()
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Get action bounds if available
        action_low = None
        action_high = None
        if hasattr(self, 'action_low') and hasattr(self, 'action_high'):
            action_low = self.action_low
            action_high = self.action_high
        
        # PPO update
        for _ in range(self.config.ppo_epochs):
            # Forward pass
            mean, std, values = self.policy(states, action_low, action_high)
            
            # Calculate new log probabilities
            dist = Normal(mean, std)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            
            # Calculate ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Calculate policy loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Calculate value loss
            value_loss = nn.MSELoss()(values.squeeze(), rewards)
            
            # Calculate entropy loss
            entropy_loss = -dist.entropy().mean()
            
            # Total loss
            total_loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
        
        # Clear buffer
        self.clear_buffer()
    
    def _compute_discounted_rewards(self):
        """Compute discounted rewards"""
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)
        
        discounted_rewards = np.zeros_like(rewards)
        running_reward = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                running_reward = 0
            running_reward = rewards[t] + self.config.gamma * running_reward
            discounted_rewards[t] = running_reward
        
        return torch.FloatTensor(discounted_rewards).to(self.device)
    
    def clear_buffer(self):
        """Clear experience buffer"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
    
    def save_model(self, filepath):
        """Save model"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
    
    def load_model(self, filepath):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


# Simple Training Function
def train_simple_notch_designer(pretrain_model=None, max_episodes=None, max_steps=None, init_params=None):
    """Train simple notch filter designer"""
    
    print("=== Training Simple HDD Notch Filter Designer (Iterative & Randomized) ===")
    
    # Create environment
    env_config = config.get_simple_config()
    env = SimpleHDDNotchDesignEnv(env_config)
    if init_params is not None:
        env.load_initial_params(init_params)

    # Create agent - use TrainingConfig from config.py
    train_config = config.TrainingConfig()
    if max_episodes is not None:
        train_config.max_episodes = int(max_episodes)
    if max_steps is not None:
        train_config.max_steps_per_episode = int(max_steps)
    
    state_dim = env.observation_space['shape'][0]
    action_dim = env.action_space['shape'][0]
    agent = PPOAgent(state_dim, action_dim, train_config)
    # Provide action bounds to agent so it can clip outputs
    agent.action_low = env.action_space['low']
    agent.action_high = env.action_space['high']
    if pretrain_model is not None:
        agent.load_model(pretrain_model)
        print(f"Loaded pretrained model: {os.path.abspath(pretrain_model)}")
    
    print(f"Environment info:")
    print(f"  State dimension: {state_dim}")
    print(f"  Action dimension: {action_dim}")
    print(f"  Plant case: RT (Randomized)")
    print(f"  Episodes: {train_config.max_episodes}")
    print(f"  Steps per episode: {train_config.max_steps_per_episode}")
    
    # Training loop
    episode_rewards = []
    best_reward = float('-inf')
    
    # Ensure models directory exists
    models_dir = os.path.join(os.getcwd(), 'models')
    os.makedirs(models_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for episode in range(train_config.max_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(train_config.max_steps_per_episode):
            # Get action
            action, log_prob, value = agent.get_action(state)
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            agent.store_transition(state, action, reward, value, log_prob, done)
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # Update policy
        agent.update()
        
        episode_rewards.append(episode_reward)
        
        # Record best performance
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_path = os.path.join(models_dir, f"simple_notch_designer_best_{timestamp}.pth")
            agent.save_model(best_path)
            print(f"Saved best model to: {os.path.abspath(best_path)}")
        
        # Print progress
        if episode % train_config.log_interval == 0:
            avg_reward = np.mean(episode_rewards[-train_config.log_interval:])
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Best: {best_reward:.2f}")
    
    # Persist reward history
    rewards_array = np.array(episode_rewards, dtype=np.float32)
    rewards_npy_path = os.path.join(models_dir, f"simple_notch_rewards_{timestamp}.npy")
    rewards_csv_path = os.path.join(models_dir, f"simple_notch_rewards_{timestamp}.csv")
    np.save(rewards_npy_path, rewards_array)
    np.savetxt(rewards_csv_path, rewards_array, fmt='%.6f', delimiter=',', header='episode_reward', comments='')
    print(f"Saved reward history to: {rewards_npy_path} and {rewards_csv_path}")
    
    # Plot reward curve
    plt.figure(figsize=(10, 4))
    plt.plot(episode_rewards, label='Episode Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Simple Notch Designer Training Rewards')
    plt.grid(True, alpha=0.3)
    plt.legend()
    reward_plot_path = os.path.join(models_dir, f"simple_notch_rewards_{timestamp}.png")
    plt.tight_layout()
    plt.savefig(reward_plot_path)
    plt.close()
    print(f"Saved reward plot to: {reward_plot_path}")
    
    # Save final model
    final_path = os.path.join(models_dir, f"final_simple_notch_designer_{timestamp}.pth")
    agent.save_model(final_path)
    print(f"Saved final model to: {os.path.abspath(final_path)}")

def evaluate_notch_params(env, notch_params, plant_case=None):
    """Evaluate a physical notch-parameter vector in the environment."""
    if plant_case is not None:
        env._select_plant_case(plant_case)
    elif env.current_case is None:
        env._randomize_plant()

    env.current_notch_params = np.array(notch_params, dtype=np.float64)
    performance = env._evaluate_current_system()
    reward = env._calculate_reward(performance)
    return performance, reward


def format_notch_params(params, notches_per_channel=2):
    params = np.asarray(params, dtype=np.float64)
    vcm = params[:notches_per_channel * 3].reshape(notches_per_channel, 3)
    pzt = params[notches_per_channel * 3:].reshape(notches_per_channel, 3)
    parts = []
    for idx, (f0, bw, depth) in enumerate(vcm, start=1):
        parts.append(f"VCM{idx}(f0={f0:.2f} Hz, bw={bw:.2f} Hz, depth={depth:.2f} dB)")
    for idx, (f0, bw, depth) in enumerate(pzt, start=1):
        parts.append(f"PZT{idx}(f0={f0:.2f} Hz, bw={bw:.2f} Hz, depth={depth:.2f} dB)")
    return ", ".join(parts)


def optimize_notch_designer(plant_case='c2', maxiter=25, popsize=8, seed=1, workers=1):
    """Find notch parameters with deterministic constrained optimization."""
    print("=== Optimizing HDD Notch Filter with Full Multirate Evaluation ===")
    if workers != 1:
        print("Parallel workers are disabled for this optimizer entrypoint on Windows; using workers=1.")
        workers = 1
    env = SimpleHDDNotchDesignEnv(config.get_simple_config())

    if plant_case == 'all':
        case_names = [case[0] for case in env.plant_cases]
    else:
        case_name = plant_case if str(plant_case).startswith('c') else f'c{plant_case}'
        known_cases = {case[0] for case in env.plant_cases}
        if case_name not in known_cases:
            raise ValueError(f"Unknown plant case: {plant_case}. Available: {sorted(known_cases)} or 'all'.")
        case_names = [case_name]

    bounds = list(zip(env.param_low, env.param_high))

    def objective(params):
        losses = []
        for case_name in case_names:
            performance, _ = evaluate_notch_params(env, params, case_name)
            losses.append(env._objective_from_performance(performance, params))
        return float(np.mean(losses))

    result = differential_evolution(
        objective,
        bounds,
        maxiter=maxiter,
        popsize=popsize,
        seed=seed,
        polish=True,
        workers=workers,
        updating='immediate' if workers == 1 else 'deferred',
        tol=1e-3,
    )

    best_params = np.array(result.x, dtype=np.float64)
    print(f"Best objective: {result.fun:.4f}")
    print(f"Best notch params: {format_notch_params(best_params, env.notches_per_channel)}")

    for case_name in case_names:
        performance, reward = evaluate_notch_params(env, best_params, case_name)
        print(
            f"{case_name}: reward={reward:.4f}, "
            f"S_peak={performance['sensitivity_peak']:.2f} dB, "
            f"PM={performance['phase_margin']:.2f} deg, "
            f"GM={performance['gain_margin']:.2f} dB, "
            f"stable={performance['stability']:.0f}"
        )

    models_dir = os.path.join(os.getcwd(), 'models')
    os.makedirs(models_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(models_dir, f"optimized_notch_{plant_case}_{timestamp}.npz")
    np.savez(
        output_path,
        notch_params=best_params,
        objective=float(result.fun),
        plant_case=str(plant_case),
        maxiter=int(maxiter),
        popsize=int(popsize),
        seed=int(seed),
    )
    print(f"Saved optimized notch to: {os.path.abspath(output_path)}")
    return best_params, result


def build_state_for_case(env, case_name):
    """Build the policy input state for a fixed plant case."""
    env._select_plant_case(case_name)
    env.current_notch_params = env._midpoint_params()
    performance = env._evaluate_current_system()
    env.current_performance = np.array([performance['sensitivity_peak']])
    env.current_system_features = env._extract_current_system_features(
        env.current_vcm_fr_with_notch,
        env.current_pzt_fr_with_notch,
        env.current_sensitivity_fr,
    )
    return np.concatenate([
        env.plant_features,
        env.current_system_features,
        env._normalize_notch_params(env.current_notch_params),
        env._normalize_performance(env.current_performance),
    ]).astype(np.float32)


def optimize_action_for_case(env, case_name, maxiter=8, popsize=5, seed=1):
    """Optimize directly in the policy's local normalized action space."""
    build_state_for_case(env, case_name)

    def objective(action):
        env._select_plant_case(case_name)
        params = env._action_to_params(np.asarray(action, dtype=np.float64))
        performance, _ = evaluate_notch_params(env, params, case_name)
        return env._objective_from_performance(performance, params)

    result = differential_evolution(
        objective,
        [(-1.0, 1.0)] * env.action_space['shape'][0],
        maxiter=maxiter,
        popsize=popsize,
        seed=seed,
        polish=True,
        tol=1e-3,
    )
    action = np.clip(np.asarray(result.x, dtype=np.float32), -1.0, 1.0)
    env._select_plant_case(case_name)
    params = env._action_to_params(action)
    performance, reward = evaluate_notch_params(env, params, case_name)
    actual_objective = env._objective_from_performance(performance, params)
    return action, params, performance, reward, actual_objective, result


def pretrain_policy_with_optimizer(maxiter=8, popsize=5, epochs=600, seed=1):
    """Warm-start the PPO policy by fitting optimizer-generated actions."""
    print("=== Supervised Pretrain from Local Optimizer Baselines ===")
    env = SimpleHDDNotchDesignEnv(config.get_simple_config())
    train_config = config.TrainingConfig()
    state_dim = env.observation_space['shape'][0]
    action_dim = env.action_space['shape'][0]
    agent = PPOAgent(state_dim, action_dim, train_config)
    agent.action_low = env.action_space['low']
    agent.action_high = env.action_space['high']

    states = []
    actions = []
    rows = []
    for idx, (case_name, _, _) in enumerate(env.plant_cases):
        action, params, performance, reward, actual_objective, result = optimize_action_for_case(
            env,
            case_name,
            maxiter=maxiter,
            popsize=popsize,
            seed=seed + idx,
        )
        state = build_state_for_case(env, case_name)
        states.append(state)
        actions.append(action)
        rows.append((case_name, params, performance, reward, actual_objective))
        print(
            f"{case_name}: objective={actual_objective:.4f}, reward={reward:.4f}, "
            f"S_peak={performance['sensitivity_peak']:.2f} dB, "
            f"PM={performance['phase_margin']:.2f} deg, "
            f"GM={performance['gain_margin']:.2f} dB"
        )

    states_t = torch.as_tensor(np.asarray(states, dtype=np.float32), device=agent.device)
    actions_t = torch.as_tensor(np.asarray(actions, dtype=np.float32), device=agent.device)

    for epoch in range(int(epochs)):
        mean, _, _ = agent.policy(states_t)
        loss = nn.MSELoss()(mean, actions_t)
        agent.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), agent.max_grad_norm)
        agent.optimizer.step()
        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Pretrain epoch {epoch}: mse={loss.item():.6f}")

    models_dir = os.path.join(os.getcwd(), 'models')
    os.makedirs(models_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(models_dir, f"pretrained_simple_notch_policy_{timestamp}.pth")
    data_path = os.path.join(models_dir, f"pretrained_simple_notch_dataset_{timestamp}.npz")
    agent.save_model(model_path)
    np.savez(
        data_path,
        states=np.asarray(states, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        cases=np.asarray([row[0] for row in rows]),
        notch_params=np.asarray([row[1] for row in rows], dtype=np.float64),
        rewards=np.asarray([row[3] for row in rows], dtype=np.float64),
    )
    print(f"Saved pretrained policy to: {os.path.abspath(model_path)}")
    print(f"Saved pretrain dataset to: {os.path.abspath(data_path)}")
    return model_path


def load_model_and_predict(model_path, env=None, deterministic=True):
    """Compatibility helper used by evaluate_model.py."""
    if env is None:
        env = SimpleHDDNotchDesignEnv(config.get_simple_config())

    train_config = config.TrainingConfig()
    state_dim = env.observation_space['shape'][0]
    action_dim = env.action_space['shape'][0]
    agent = PPOAgent(state_dim, action_dim, train_config)
    agent.action_low = env.action_space['low']
    agent.action_high = env.action_space['high']
    agent.load_model(model_path)

    state = env.reset()
    if deterministic:
        action, value = agent.get_action(state, deterministic=True)
    else:
        action, _, value = agent.get_action(state, deterministic=False)
    _, reward, done, info = env.step(action)

    return {
        'action': action,
        'value': value,
        'reward': reward,
        'done': done,
        'performance': info['performance'],
        'notch_params': info['notch_params'],
        'case': info['case'],
    }


def main():
    parser = argparse.ArgumentParser(description="HDD notch filter design")
    parser.add_argument('--mode', choices=['train', 'optimize', 'pretrain'], default='train')
    parser.add_argument('--plant-case', default='c2', help="Plant case for optimization, e.g. c2, or all.")
    parser.add_argument('--maxiter', type=int, default=25)
    parser.add_argument('--popsize', type=int, default=8)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--pretrain-model', default=None, help="Optional pretrained .pth model for PPO training.")
    parser.add_argument('--max-episodes', type=int, default=None, help="Override PPO episode count.")
    parser.add_argument('--max-steps', type=int, default=None, help="Override PPO steps per episode.")
    parser.add_argument('--pretrain-epochs', type=int, default=600, help="Supervised pretrain epochs.")
    parser.add_argument('--init-params', default=None,
                        help="Path to optimized_notch_*.npz to warm-start each episode from DE solution.")
    args = parser.parse_args()

    if args.mode == 'train':
        train_simple_notch_designer(
            pretrain_model=args.pretrain_model,
            max_episodes=args.max_episodes,
            max_steps=args.max_steps,
            init_params=args.init_params,
        )
    elif args.mode == 'optimize':
        optimize_notch_designer(
            plant_case=args.plant_case,
            maxiter=args.maxiter,
            popsize=args.popsize,
            seed=args.seed,
            workers=args.workers,
        )
    else:
        pretrain_policy_with_optimizer(
            maxiter=args.maxiter,
            popsize=args.popsize,
            epochs=args.pretrain_epochs,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
