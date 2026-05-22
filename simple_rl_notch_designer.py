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
import sys
import os
import copy
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
        
        # Frequency range
        # Nyquist frequency for the multirate system
        nyquist_freq = 1.0 / (self.Ts / self.Mr_f) / 2.0
        # Limit max frequency to Nyquist to avoid aliasing and numerical issues
        max_freq = min(100000, nyquist_freq * 0.99)
        self.freq_range = np.logspace(1, np.log10(max_freq), 1000)
        self.omega = 2 * np.pi * self.freq_range
        
        # Precompute Controller Frequency Responses (Fixed)
        self.Cd_vcm_fr = self._ensure_1d_fr(utils.freqresp(self.Sys_Cd_vcm, self.omega))
        self.Cd_pzt_fr = self._ensure_1d_fr(utils.freqresp(self.Sys_Cd_pzt, self.omega))
        
        # Note: Multirate filters are part of the plant path in the simplified model
        # We approximate the digital plant as P_d = P_c * F_m * Notch
        # So we need F_m frequency response
        # However, strictly speaking, F_m is at high rate, P_c is continuous.
        # Let's approximate the "Plant Path without Notch" frequency response.
        # P_path = P_c(s) * F_m(z) * Hold(s)
        # For simplicity and speed, we will use the precomputed P_d from the original code logic
        # but we need to be able to randomize it.
        # Strategy: Randomize P_c, then multiply by fixed F_m.
        
        self.Fm_vcm_fr = self._ensure_1d_fr(utils.freqresp(self.Sys_Fm_vcm, self.omega))
        self.Fm_pzt_fr = self._ensure_1d_fr(utils.freqresp(self.Sys_Fm_pzt, self.omega))
        
        # State and action spaces
        self.observation_space = self._create_observation_space()
        self.param_low, self.param_high = self._load_action_bounds(env_config.get('action_bounds'))
        self.param_range = np.clip(self.param_high - self.param_low, 1e-6, None)
        self.action_space = self._create_action_space()
        
        # Precompute Plant Frequency Responses for speed
        self.plant_fr_cache = {}
        self._precompute_plant_responses()

        # Initialize state variables
        self.current_notch_params = self._midpoint_params()
        self.plant_features = np.zeros(30)  # Base plant features (static)
        self.current_system_features = np.zeros(15)  # Current sensitivity features (dynamic: 5 peaks * 3 values)
        self.current_performance = np.array([0.0])  # Only sensitivity_peak
        
        # Placeholders for current system FRs
        self.current_vcm_fr_with_notch = None
        self.current_pzt_fr_with_notch = None
        self.current_sensitivity_fr = None
        
    def _create_observation_space(self):
        """Create observation space"""
        # State: [Base_Plant_Features(30), Current_Sensitivity_Features(15), Current_Notch_Params(6), Sensitivity_Peak(1)]
        # Total: 52 dimensions
        # Base_Plant_Features: Static features from P*Fm (for reference - tells agent what peaks to suppress)
        # Current_Sensitivity_Features: Dynamic sensitivity peaks (tells agent current system state and notch effectiveness)
        # Sensitivity_Peak: Current sensitivity peak value (directly related to reward)
        return {
            'shape': (52,),
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
    
    def _load_action_bounds(self, action_bounds):
        """Load physical notch parameter bounds"""
        if action_bounds is None:
            # Default bounds matching NotchFilterConfig ranges
            # VCM: freq(5k-45k), bw(100-5k), depth(-60-0)
            # PZT: freq(10k-47k), bw(100-5k), depth(-60-0)
            low = np.array([5000.0, 100.0, -60.0, 10000.0, 100.0, -60.0], dtype=np.float32)
            high = np.array([45000.0, 5000.0, 0.0, 47000.0, 5000.0, 0.0], dtype=np.float32)
            return low, high
        low = np.array(action_bounds['low'], dtype=np.float32)
        high = np.array(action_bounds['high'], dtype=np.float32)
        if low.shape != (6,) or high.shape != (6,):
            raise ValueError("Action bounds must provide 6 low and 6 high values.")
        return low, high
    
    def _midpoint_params(self):
        """Return mid-point parameters within physical bounds (Geometric mean for Log indices)"""
        params = np.zeros(6)
        log_indices = [0, 1, 3, 4]
        
        for i in range(6):
            low = self.param_low[i]
            high = self.param_high[i]
            
            if i in log_indices:
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
            'shape': (6,),
            'low': np.array([-1.0] * 6),
            'high': np.array([1.0] * 6)
        }
    
    def _action_to_params(self, action):
        """Map normalized action [-1,1] to physical notch parameters (Log scale for Freq/BW)"""
        action = np.clip(action, self.action_space['low'], self.action_space['high'])
        normalized = (action + 1.0) / 2.0  # [0,1]
        
        params = np.zeros_like(normalized)
        
        # Log-scale indices: 0 (VCM Freq), 1 (VCM BW), 3 (PZT Freq), 4 (PZT BW)
        log_indices = [0, 1, 3, 4]
        
        for i in range(6):
            low = self.param_low[i]
            high = self.param_high[i]
            
            if i in log_indices:
                # Log mapping
                log_low = np.log10(low)
                log_high = np.log10(high)
                params[i] = 10**(log_low + normalized[i] * (log_high - log_low))
            else:
                # Linear mapping
                params[i] = low + normalized[i] * (high - low)
                
        return params
    
    def _precompute_plant_responses(self):
        """Precompute frequency responses for all plant cases combined with Fm"""
        print("Precomputing plant responses for optimization...")
        fast_ts = self.Ts / self.Mr_f
        
        for case_name, base_vcm, base_pzt in self.plant_cases:
            # 1. VCM Path
            # Discretize Plant (ZOH)
            sys_pdm0_vcm = matlab.c2d(base_vcm, fast_ts, 'zoh')
            # Combine with Fm (Fast Rate)
            # Note: Series connection. In freq domain: product.
            sys_chain_vcm = sys_pdm0_vcm * self.Sys_Fm_vcm
            
            # Compute FR
            fr_vcm = self._freqresp_1d(sys_chain_vcm)
            
            # 2. PZT Path
            sys_pdm0_pzt = matlab.c2d(base_pzt, fast_ts, 'zoh')
            sys_chain_pzt = sys_pdm0_pzt * self.Sys_Fm_pzt
            fr_pzt = self._freqresp_1d(sys_chain_pzt)
            
            self.plant_fr_cache[case_name] = (fr_vcm, fr_pzt)
            
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

    def _randomize_plant(self):
        """Randomize plant by selecting a case (Gain scaling removed)"""
        case_idx = np.random.randint(0, len(self.plant_cases))
        case_name, base_vcm, base_pzt = self.plant_cases[case_idx]
        self.current_case = case_name
        
        # Removed random gain scaling
        self.current_gain_scale = 1.0
        self.current_base_fr = self.plant_fr_cache[case_name]
        
        # Calculate plant features from Base FR (No Notch)
        # Base_FR is (VCM_Chain, PZT_Chain) where Chain = P * Fm
        self.base_vcm_fr = self.current_base_fr[0]
        self.base_pzt_fr = self.current_base_fr[1]
        
        self.plant_features = self._extract_plant_features_from_fr(self.base_vcm_fr, self.base_pzt_fr)
        
        # Legacy placeholders not used in optimized evaluation
        self.current_vcm_plant = None
        self.current_pzt_plant = None

    def reset(self):
        """Reset environment"""
        # 1. Randomize Plant
        self._randomize_plant()
        
        # 2. Reset Notch Params to mid-range values
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
        """Execute action"""
        # 1. Map normalized action to physical notch parameters
        self.current_notch_params = self._action_to_params(action)
        
        # 2. Evaluate System
        performance = self._evaluate_current_system()
        
        # 3. Calculate Reward
        reward = self._calculate_reward(performance)
        
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
        """Evaluate system with current notch params using frequency response multiplication (Optimized)"""
        if not hasattr(self, 'current_base_fr') or self.current_base_fr is None:
            raise RuntimeError("Plant must be randomized before evaluation.")
        
        # Get Notch FRs (Vectorized, Analytical)
        notch_vcm_fr = self._compute_notch_fr(
            self.current_notch_params[0],
            self.current_notch_params[1],
            self.current_notch_params[2]
        )
        notch_pzt_fr = self._compute_notch_fr(
            self.current_notch_params[3],
            self.current_notch_params[4],
            self.current_notch_params[5]
        )
        
        # Combine: Base * Notch * Gain
        # Base_FR is precomputed (P * Fm)
        Fr_Pd_vcm = self.current_base_fr[0] * notch_vcm_fr * self.current_gain_scale
        Fr_Pd_pzt = self.current_base_fr[1] * notch_pzt_fr * self.current_gain_scale
        
        L_vcm = Fr_Pd_vcm * self.Cd_vcm_fr
        L_pzt = Fr_Pd_pzt * self.Cd_pzt_fr
        L_total = L_vcm + L_pzt
        
        S = 1.0 / (1.0 + L_total)
        
        # Store current system FRs for state extraction
        self.current_vcm_fr_with_notch = Fr_Pd_vcm
        self.current_pzt_fr_with_notch = Fr_Pd_pzt
        self.current_sensitivity_fr = S
        
        return self._calculate_performance_metrics(L_total, S)

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
        log_indices = [0, 1, 3, 4]
        
        for i in range(6):
            low = self.param_low[i]
            high = self.param_high[i]
            val = params[i]
            
            if i in log_indices:
                # Inverse of log mapping
                log_low = np.log10(low)
                log_high = np.log10(high)
                n = (np.log10(val) - log_low) / (log_high - log_low)
                norm[i] = n
            else:
                n = (val - low) / (high - low)
                norm[i] = n
                
        return np.clip(norm, 0.0, 1.0)

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
        
        stable = 1.0 if pm > 0 and gm > 0 else 0.0
        
        return {
            'phase_margin': float(pm),
            'gain_margin': float(gm),
            'sensitivity_peak': float(sp),
            'stability': float(stable),
            'tracking_error': float(te)
        }

    def _calculate_phase_margin(self, Fr_L):
        # Same as before
        mag = np.abs(Fr_L)
        phase = np.angle(Fr_L) * 180/np.pi
        for i in range(len(mag)-1):
            if mag[i] >= 1.0 and mag[i+1] <= 1.0:
                # Interp
                frac = (mag[i] - 1.0) / (mag[i] - mag[i+1] + 1e-12)
                p = phase[i] + frac*(phase[i+1]-phase[i])
                return 180 + p
        return 0.0

    def _calculate_gain_margin(self, Fr_L):
        # Same as before
        mag = np.abs(Fr_L)
        phase = np.angle(Fr_L) * 180/np.pi
        for i in range(len(phase)-1):
            if phase[i] >= -180 and phase[i+1] <= -180:
                frac = (phase[i] - (-180)) / (phase[i] - phase[i+1] + 1e-12)
                m = mag[i] + frac*(mag[i+1]-mag[i])
                return -20*np.log10(m + 1e-12)
        return 0.0

    def _calculate_reward(self, performance):
        """Calculate reward with improved sensitivity_peak reward shaping"""
        reward = 0
        
        # Phase margin reward
        pm_diff = abs(performance['phase_margin'] - self.targets['phase_margin'])
        reward += self.weights['phase_margin'] * np.exp(-pm_diff / 10)
        
        # Gain margin reward
        gm_diff = abs(performance['gain_margin'] - self.targets['gain_margin'])
        reward += self.weights['gain_margin'] * np.exp(-gm_diff / 2)
        
        # Sensitivity peak reward - improved shaping
        # Use both absolute difference and relative improvement
        sp_diff = abs(performance['sensitivity_peak'] - self.targets['sensitivity_peak'])
        # Reward for being below target (good), penalize for being above (bad)
        if performance['sensitivity_peak'] <= self.targets['sensitivity_peak']:
            # Bonus for achieving target or better
            reward += self.weights['sensitivity_peak'] * (1.0 + np.exp(-sp_diff / 1.0))
        else:
            # Penalty for exceeding target, with exponential decay
            reward += self.weights['sensitivity_peak'] * np.exp(-sp_diff / 2.0)
        
        # Stability reward
        reward += self.weights['stability'] * performance['stability']
        
        # Tracking error reward
        te_diff = abs(performance['tracking_error'] - self.targets['tracking_error'])
        reward += self.weights['tracking_error'] * np.exp(-te_diff / 2.0)
        
        # Strong penalty for unstable systems
        if performance['stability'] < 0.5:
            reward -= 10
            
        return reward
    

    def _is_done(self, performance):
        # Same as before
        pm_ok = abs(performance['phase_margin'] - self.targets['phase_margin']) < 2
        gm_ok = abs(performance['gain_margin'] - self.targets['gain_margin']) < 1
        sp_ok = performance['sensitivity_peak'] < (self.targets['sensitivity_peak'] + 1.0)
        stable = performance['stability'] > 0.8
        return pm_ok and gm_ok and sp_ok and stable


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
        states = torch.FloatTensor(self.states).to(self.device)
        actions = torch.FloatTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        old_values = torch.FloatTensor(self.values).to(self.device)
        
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
def train_simple_notch_designer():
    """Train simple notch filter designer"""
    
    print("=== Training Simple HDD Notch Filter Designer (Iterative & Randomized) ===")
    
    # Create environment
    env_config = config.get_simple_config()
    env = SimpleHDDNotchDesignEnv(env_config)
    
    # Create agent - use TrainingConfig from config.py
    train_config = config.TrainingConfig()
    
    state_dim = env.observation_space['shape'][0]
    action_dim = env.action_space['shape'][0]
    agent = PPOAgent(state_dim, action_dim, train_config)
    # Provide action bounds to agent so it can clip outputs
    agent.action_low = env.action_space['low']
    agent.action_high = env.action_space['high']
    
    print(f"Environment info:")
    print(f"  State dimension: {state_dim}")
    print(f"  Action dimension: {action_dim}")
    print(f"  Plant case: RT (Randomized)")
    
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

if __name__ == "__main__":
    train_simple_notch_designer()