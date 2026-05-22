"""
HDD Notch Filter Auto Design Tool - Peak-Focused Version

基于 simple_rl_notch_designer.py 的复制版本，在此基础上做两件事：
1) 使用 P*Fm 频响中的主峰来引导 notch 中心频率（动作只在主峰附近微调）
2) 奖励函数更直接地评价“敏感度峰是否被压低，以及相对改善”

注意：不修改原始 simple_rl_notch_designer.py，本文件是一个独立的实验版本。
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


class PeakFocusedHDDNotchDesignEnv:
    """
    Peak-Focused HDD Notch Filter Design Environment

    与 SimpleHDDNotchDesignEnv 的主要差异：
    - 仍然预计算 P*Fm 的频响，但提取各 case 的主峰频率，用来引导 notch 中心频率
    - 动作空间维度仍为 6（VCM/PZT 各 freq, bw, depth），但 freq 解释为“围绕主峰的小范围偏移”
    - 奖励函数更专注于敏感度峰值的绝对水平 + 相对改善，并对过宽/过深 notch 给予轻微惩罚
    """

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
            ("c1", plant.Sys_Pc_vcm_c1, plant.Sys_Pc_pzt_c1),
            ("c2", plant.Sys_Pc_vcm_c2, plant.Sys_Pc_pzt_c2),
            ("c3", plant.Sys_Pc_vcm_c3, plant.Sys_Pc_pzt_c3),
            ("c4", plant.Sys_Pc_vcm_c4, plant.Sys_Pc_pzt_c4),
            ("c5", plant.Sys_Pc_vcm_c5, plant.Sys_Pc_pzt_c5),
            ("c6", plant.Sys_Pc_vcm_c6, plant.Sys_Pc_pzt_c6),
            ("c7", plant.Sys_Pc_vcm_c7, plant.Sys_Pc_pzt_c7),
            ("c8", plant.Sys_Pc_vcm_c8, plant.Sys_Pc_pzt_c8),
            ("c9", plant.Sys_Pc_vcm_c9, plant.Sys_Pc_pzt_c9),
        ]

        # Current Plant (will be randomized per episode)
        self.current_case = None
        self.current_vcm_plant = None
        self.current_pzt_plant = None

        # Controllers
        if hasattr(utils, "Sys_Cd_vcm"):
            self.Sys_Cd_vcm = utils.Sys_Cd_vcm
        elif hasattr(utils, "get_Sys_Cd_vcm"):
            self.Sys_Cd_vcm = utils.get_Sys_Cd_vcm()
        else:
            raise AttributeError("No VCM controller found in utils")

        if hasattr(utils, "Sys_Cd_pzt"):
            self.Sys_Cd_pzt = utils.Sys_Cd_pzt
        elif hasattr(utils, "get_Sys_Cd_pzt"):
            self.Sys_Cd_pzt = utils.get_Sys_Cd_pzt()
        else:
            raise AttributeError("No PZT controller found in utils")

        # Multi-rate filters
        if hasattr(utils, "Sys_Fm_vcm"):
            self.Sys_Fm_vcm = utils.Sys_Fm_vcm
        elif hasattr(utils, "get_Sys_Fm_vcm"):
            self.Sys_Fm_vcm = utils.get_Sys_Fm_vcm()
        else:
            raise AttributeError("No VCM multirate filter found in utils")

        if hasattr(utils, "Sys_Fm_pzt"):
            self.Sys_Fm_pzt = utils.Sys_Fm_pzt
        elif hasattr(utils, "get_Sys_Fm_pzt"):
            self.Sys_Fm_pzt = utils.get_Sys_Fm_pzt()
        else:
            raise AttributeError("No PZT multirate filter found in utils")

        # Reward weights and targets
        self.weights = env_config["weights"]
        self.targets = env_config["targets"]

        # Frequency range
        nyquist_freq = 1.0 / (self.Ts / self.Mr_f) / 2.0
        max_freq = min(100000, nyquist_freq * 0.99)
        self.freq_range = np.logspace(1, np.log10(max_freq), 1000)
        self.omega = 2 * np.pi * self.freq_range

        # Precompute Controller Frequency Responses (Fixed)
        self.Cd_vcm_fr = self._ensure_1d_fr(utils.freqresp(self.Sys_Cd_vcm, self.omega))
        self.Cd_pzt_fr = self._ensure_1d_fr(utils.freqresp(self.Sys_Cd_pzt, self.omega))

        # Multirate filters frequency responses
        self.Fm_vcm_fr = self._ensure_1d_fr(utils.freqresp(self.Sys_Fm_vcm, self.omega))
        self.Fm_pzt_fr = self._ensure_1d_fr(utils.freqresp(self.Sys_Fm_pzt, self.omega))

        # State and action spaces
        self.observation_space = self._create_observation_space()
        self.param_low, self.param_high = self._load_action_bounds(env_config.get("action_bounds"))
        self.param_range = np.clip(self.param_high - self.param_low, 1e-6, None)
        self.action_space = self._create_action_space()

        # Precompute Plant Frequency Responses for speed
        self.plant_fr_cache = {}
        self._precompute_plant_responses()

        # 初始化：各 case 的主峰频率（由 _precompute_plant_responses 填充）
        self.vcm_main_peak_freq = {}  # case_name -> float
        self.pzt_main_peak_freq = {}  # case_name -> float

        # Initialize state variables
        self.current_notch_params = self._midpoint_params()
        self.plant_features = np.zeros(30)  # Base plant features (static)
        self.current_system_features = np.zeros(15)  # Current sensitivity features
        self.current_performance = np.array([0.0])  # Only sensitivity_peak
        self.prev_sensitivity_peak = None  # for relative improvement reward

        # Placeholders for current system FRs
        self.current_vcm_fr_with_notch = None
        self.current_pzt_fr_with_notch = None
        self.current_sensitivity_fr = None

        # 当前 episode 所用的“主峰参考频率”（VCM/PZT）
        self.current_vcm_peak_ref = None
        self.current_pzt_peak_ref = None

    def _create_observation_space(self):
        """Create observation space"""
        # State: [Base_Plant_Features(30), Current_Sensitivity_Features(15),
        #         Current_Notch_Params(6), Sensitivity_Peak(1),
        #         Peak_Ref_Freqs(2)]
        # Total: 54 dimensions
        return {
            "shape": (54,),
            "low": -np.inf,
            "high": np.inf,
        }

    def _ensure_1d_fr(self, fr):
        """Ensure frequency response arrays are 1D (nfreq,)"""
        fr = np.asarray(fr)
        fr = np.squeeze(fr)
        if fr.ndim == 0:
            return np.array([fr], dtype=complex)
        if fr.ndim > 1:
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

        num = [1, 2 * zeta * w0 * depth_lin, w0**2]
        den = [1, 2 * zeta * w0, w0**2]
        notch_ct = matlab.tf(num, den)
        return matlab.c2d(notch_ct, sample_time, "zoh")

    def _create_digital_path(self, sys_pc, sys_fm, notch_tf=None):
        """Create discrete-time plant path with multirate filters and optional notch"""
        sys_pdm0 = matlab.c2d(sys_pc, self.Ts / self.Mr_f, "zoh")
        sys_chain = sys_pdm0 * sys_fm
        if notch_tf is not None:
            sys_chain = sys_chain * notch_tf
        sys_pd = utils.dts_resampling(sys_chain, self.Mr_f)
        return sys_pd

    def _load_action_bounds(self, action_bounds):
        """Load physical notch parameter bounds"""
        if action_bounds is None:
            low = np.array([5000.0, 100.0, -60.0, 10000.0, 100.0, -60.0], dtype=np.float32)
            high = np.array([45000.0, 5000.0, 0.0, 47000.0, 5000.0, 0.0], dtype=np.float32)
            return low, high
        low = np.array(action_bounds["low"], dtype=np.float32)
        high = np.array(action_bounds["high"], dtype=np.float32)
        if low.shape != (6,) or high.shape != (6,):
            raise ValueError("Action bounds must provide 6 low and 6 high values.")
        return low, high

    def _midpoint_params(self):
        """
        返回初始 notch 参数：
        - 频率直接用当前 case 的主峰作为初始值（在 reset 之后再更新）
        - BW/Depth 用物理范围的中点/几何中点
        """
        params = np.zeros(6, dtype=np.float32)
        log_indices = [1, 4]  # 仅对 BW 做 log/geo 处理；freq 在 reset 时用主峰重置

        for i in range(6):
            low = self.param_low[i]
            high = self.param_high[i]
            if i in log_indices:
                params[i] = np.sqrt(low * high)
            else:
                params[i] = (low + high) / 2.0
        return params

    def _create_action_space(self):
        """Create action space"""
        # Action: 6 维，仍在 [-1, 1]，但 freq 解释为“围绕主峰的小范围偏移”
        return {
            "shape": (6,),
            "low": np.array([-1.0] * 6),
            "high": np.array([1.0] * 6),
        }

    def _action_to_params(self, action):
        """
        Map normalized action [-1,1] to physical notch parameters.

        与原实现的区别：
        - 频率：不再直接映射到 [low, high]，而是围绕当前 case 的主峰频率做对数尺度的小范围偏移
        - BW/Depth：仍然使用原来的物理范围映射
        """
        action = np.clip(action, self.action_space["low"], self.action_space["high"])
        normalized = (action + 1.0) / 2.0  # [0,1]

        params = np.zeros(6, dtype=np.float32)

        # 1) 频率：围绕主峰做对数偏移（±delta_decades）
        delta_decades = 0.25  # 大约 10^(±0.25) ≈ 0.56~1.78 倍，限制在主峰附近

        # VCM
        if self.current_vcm_peak_ref is None:
            vcm_ref = (self.param_low[0] + self.param_high[0]) / 2.0
        else:
            vcm_ref = float(self.current_vcm_peak_ref)
        delta_vcm = action[0] * delta_decades
        params[0] = np.clip(
            vcm_ref * (10.0 ** delta_vcm),
            self.param_low[0],
            self.param_high[0],
        )

        # PZT
        if self.current_pzt_peak_ref is None:
            pzt_ref = (self.param_low[3] + self.param_high[3]) / 2.0
        else:
            pzt_ref = float(self.current_pzt_peak_ref)
        delta_pzt = action[3] * delta_decades
        params[3] = np.clip(
            pzt_ref * (10.0 ** delta_pzt),
            self.param_low[3],
            self.param_high[3],
        )

        # 2) BW/Depth：和原始 simple_rl_notch_designer 相同的线性/对数映射
        # Log-scale indices for BW
        log_indices = [1, 4]

        for i in [1, 2, 4, 5]:
            low = self.param_low[i]
            high = self.param_high[i]
            if i in log_indices:
                log_low = np.log10(low)
                log_high = np.log10(high)
                params[i] = 10 ** (log_low + normalized[i] * (log_high - log_low))
            else:
                params[i] = low + normalized[i] * (high - low)

        return params

    def _precompute_plant_responses(self):
        """Precompute frequency responses for all plant cases combined with Fm"""
        print("Precomputing plant responses for peak-focused optimization...")
        fast_ts = self.Ts / self.Mr_f

        for case_name, base_vcm, base_pzt in self.plant_cases:
            # 1. VCM Path
            sys_pdm0_vcm = matlab.c2d(base_vcm, fast_ts, "zoh")
            sys_chain_vcm = sys_pdm0_vcm * self.Sys_Fm_vcm
            fr_vcm = self._freqresp_1d(sys_chain_vcm)

            # 2. PZT Path
            sys_pdm0_pzt = matlab.c2d(base_pzt, fast_ts, "zoh")
            sys_chain_pzt = sys_pdm0_pzt * self.Sys_Fm_pzt
            fr_pzt = self._freqresp_1d(sys_chain_pzt)

            self.plant_fr_cache[case_name] = (fr_vcm, fr_pzt)

    def _compute_notch_fr(self, f0, bw, depth_db):
        """Compute Notch Filter Frequency Response analytically (Fast)"""
        ts = self.Ts / self.Mr_f
        w0 = 2 * np.pi * f0
        zeta = bw / (2 * f0)

        depth_lin = max(10 ** (depth_db / 20.0), 1e-6)

        num = [1.0, 2.0 * zeta * w0 * depth_lin, w0**2]
        den = [1.0, 2.0 * zeta * w0, w0**2]

        res = signal.cont2discrete((num, den), ts, method="zoh")
        b = res[0].ravel()
        a = res[1]

        w_digital = self.omega * ts
        _, h = signal.freqz(b, a, worN=w_digital)
        return h

    def _find_peaks(self, fr, threshold_db=-20.0, top_k=5):
        """Find top-k peaks in given frequency response"""
        mag = 20 * np.log10(np.abs(fr) + 1e-12)
        phase = np.angle(fr) * 180 / np.pi

        peaks = []
        for i in range(1, len(mag) - 1):
            if mag[i] > mag[i - 1] and mag[i] > mag[i + 1] and mag[i] > threshold_db:
                peaks.append(
                    {
                        "freq": self.freq_range[i],
                        "mag": mag[i],
                        "phase": phase[i],
                        "index": i,
                    }
                )

        peaks.sort(key=lambda x: x["mag"], reverse=True)
        peaks = peaks[:top_k]
        while len(peaks) < top_k:
            peaks.append({"freq": 0.0, "mag": 0.0, "phase": 0.0, "index": 0})
        return peaks

    def _extract_plant_features_from_fr(self, vcm_fr, pzt_fr):
        """Extract features from frequency response"""
        vcm_peaks = self._find_peaks(vcm_fr)
        pzt_peaks = self._find_peaks(pzt_fr)

        features = []
        for p in vcm_peaks:
            features.extend(
                [p["freq"] / 50000.0, p["mag"] / 60.0, p["phase"] / 180.0]
            )
        for p in pzt_peaks:
            features.extend(
                [p["freq"] / 50000.0, p["mag"] / 60.0, p["phase"] / 180.0]
            )
        return np.array(features, dtype=np.float32)

    def _extract_sensitivity_features(self, sensitivity_fr):
        """Extract features from sensitivity function (current system state)"""
        mag_db = 20 * np.log10(np.abs(sensitivity_fr) + 1e-12)

        peaks = []
        for i in range(1, len(mag_db) - 1):
            if (
                mag_db[i] > mag_db[i - 1]
                and mag_db[i] > mag_db[i + 1]
                and mag_db[i] > -40
            ):
                peaks.append(
                    {
                        "freq": self.freq_range[i],
                        "mag": mag_db[i],
                        "phase": np.angle(sensitivity_fr[i]) * 180 / np.pi,
                    }
                )

        peaks.sort(key=lambda x: x["mag"], reverse=True)
        peaks = peaks[:5]
        while len(peaks) < 5:
            peaks.append({"freq": 0, "mag": -60, "phase": 0})

        features = []
        for p in peaks:
            features.extend(
                [
                    p["freq"] / 50000.0,
                    (p["mag"] + 40) / 40.0,
                    p["phase"] / 180.0,
                ]
            )
        return np.array(features, dtype=np.float32)

    def _extract_current_system_features(
        self, vcm_fr_with_notch, pzt_fr_with_notch, sensitivity_fr
    ):
        """Extract features from current system (plant + notch)"""
        sens_features = self._extract_sensitivity_features(sensitivity_fr)
        return sens_features

    def _randomize_plant(self):
        """Randomize plant by selecting a case and记录该 case 的主峰频率"""
        case_idx = np.random.randint(0, len(self.plant_cases))
        case_name, base_vcm, base_pzt = self.plant_cases[case_idx]
        self.current_case = case_name

        # Gain scaling removed
        self.current_gain_scale = 1.0
        self.current_base_fr = self.plant_fr_cache[case_name]

        # Base_FR is (VCM_Chain, PZT_Chain) where Chain = P * Fm
        self.base_vcm_fr = self.current_base_fr[0]
        self.base_pzt_fr = self.current_base_fr[1]

        self.plant_features = self._extract_plant_features_from_fr(
            self.base_vcm_fr, self.base_pzt_fr
        )

        # 提取该 case 的主峰（幅值最大的一个），并存为本 episode 内的参考频率
        vcm_peaks = self._find_peaks(self.base_vcm_fr, threshold_db=-20.0, top_k=1)
        pzt_peaks = self._find_peaks(self.base_pzt_fr, threshold_db=-20.0, top_k=1)

        self.current_vcm_peak_ref = float(vcm_peaks[0]["freq"]) if vcm_peaks else None
        self.current_pzt_peak_ref = float(pzt_peaks[0]["freq"]) if pzt_peaks else None

        # Legacy placeholders not used in optimized evaluation
        self.current_vcm_plant = None
        self.current_pzt_plant = None

    def _normalize_notch_params(self, params):
        """Normalize params to [0, 1] for state (Log scale for Freq/BW where appropriate)"""
        norm = np.zeros_like(params, dtype=np.float32)
        log_indices = [0, 1, 3, 4]

        for i in range(6):
            low = self.param_low[i]
            high = self.param_high[i]
            val = params[i]
            if i in log_indices:
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
        norm = (perf[0] + 20.0) / 40.0
        return np.array([np.clip(norm, 0.0, 1.0)], dtype=np.float32)

    def reset(self):
        """Reset environment"""
        # 1. Randomize Plant
        self._randomize_plant()

        # 2. Reset Notch Params：频率直接锁到主峰，BW/Depth 用中值
        self.current_notch_params = self._midpoint_params()
        if self.current_vcm_peak_ref is not None:
            self.current_notch_params[0] = float(self.current_vcm_peak_ref)
        if self.current_pzt_peak_ref is not None:
            self.current_notch_params[3] = float(self.current_pzt_peak_ref)

        # 3. Calculate Initial Performance
        performance = self._evaluate_current_system()
        self.current_performance = np.array(
            [performance["sensitivity_peak"]], dtype=np.float32
        )
        self.prev_sensitivity_peak = float(performance["sensitivity_peak"])

        # 4. Extract current system features
        self.current_system_features = self._extract_current_system_features(
            self.current_vcm_fr_with_notch,
            self.current_pzt_fr_with_notch,
            self.current_sensitivity_fr,
        )

        # 5. Construct State
        state = np.concatenate(
            [
                self.plant_features,  # Base plant features
                self.current_system_features,  # Current system features
                self._normalize_notch_params(self.current_notch_params),
                self._normalize_performance(self.current_performance),
                np.array(  # 参考峰频率（归一化）
                    [
                        (self.current_vcm_peak_ref or 0.0) / 50000.0,
                        (self.current_pzt_peak_ref or 0.0) / 50000.0,
                    ],
                    dtype=np.float32,
                ),
            ]
        )
        return state

    def step(self, action):
        """Execute action"""
        # 1. Map normalized action to physical notch parameters (peak-guided)
        self.current_notch_params = self._action_to_params(action)

        # 2. Evaluate System
        performance = self._evaluate_current_system()

        # 3. Calculate Reward（包含绝对水平 + 相对改善 + notch 成本）
        reward = self._calculate_reward(performance)

        # 4. Update State
        self.current_performance = np.array(
            [performance["sensitivity_peak"]], dtype=np.float32
        )

        self.current_system_features = self._extract_current_system_features(
            self.current_vcm_fr_with_notch,
            self.current_pzt_fr_with_notch,
            self.current_sensitivity_fr,
        )

        next_state = np.concatenate(
            [
                self.plant_features,
                self.current_system_features,
                self._normalize_notch_params(self.current_notch_params),
                self._normalize_performance(self.current_performance),
                np.array(
                    [
                        (self.current_vcm_peak_ref or 0.0) / 50000.0,
                        (self.current_pzt_peak_ref or 0.0) / 50000.0,
                    ],
                    dtype=np.float32,
                ),
            ]
        )

        # 5. Check Done（沿用原来的判据）
        done = self._is_done(performance)

        info = {
            "performance": performance,
            "notch_params": self.current_notch_params,
            "case": self.current_case,
        }

        return next_state, reward, done, info

    def _evaluate_current_system(self):
        """Evaluate system with current notch params using frequency response multiplication"""
        if not hasattr(self, "current_base_fr") or self.current_base_fr is None:
            raise RuntimeError("Plant must be randomized before evaluation.")

        notch_vcm_fr = self._compute_notch_fr(
            self.current_notch_params[0],
            self.current_notch_params[1],
            self.current_notch_params[2],
        )
        notch_pzt_fr = self._compute_notch_fr(
            self.current_notch_params[3],
            self.current_notch_params[4],
            self.current_notch_params[5],
        )

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

    def _calculate_performance_metrics(self, Fr_L, Fr_S):
        """Calculate basic loop metrics (PM/GM + sensitivity stats)"""
        Fr_L = np.squeeze(Fr_L)
        Fr_S = np.squeeze(Fr_S)
        if Fr_L.ndim > 1:
            Fr_L = Fr_L[0]
        if Fr_S.ndim > 1:
            Fr_S = Fr_S[0]

        pm = self._calculate_phase_margin(Fr_L)
        gm = self._calculate_gain_margin(Fr_L)
        sp = float(np.max(20 * np.log10(np.abs(Fr_S) + 1e-12)))
        te = float(np.mean(20 * np.log10(np.abs(Fr_S) + 1e-12)))

        stable = 1.0 if pm > 0 and gm > 0 else 0.0

        return {
            "phase_margin": float(pm),
            "gain_margin": float(gm),
            "sensitivity_peak": sp,
            "stability": float(stable),
            "tracking_error": te,
        }

    def _calculate_phase_margin(self, Fr_L):
        mag = np.abs(Fr_L)
        phase = np.angle(Fr_L) * 180 / np.pi
        for i in range(len(mag) - 1):
            if mag[i] >= 1.0 and mag[i + 1] <= 1.0:
                frac = (mag[i] - 1.0) / (mag[i] - mag[i + 1] + 1e-12)
                p = phase[i] + frac * (phase[i + 1] - phase[i])
                return 180 + p
        return 0.0

    def _calculate_gain_margin(self, Fr_L):
        mag = np.abs(Fr_L)
        phase = np.angle(Fr_L) * 180 / np.pi
        for i in range(len(phase) - 1):
            if phase[i] >= -180 and phase[i + 1] <= -180:
                frac = (phase[i] - (-180)) / (phase[i] - phase[i + 1] + 1e-12)
                m = mag[i] + frac * (mag[i + 1] - mag[i])
                return -20 * np.log10(m + 1e-12)
        return 0.0

    def _calculate_reward(self, performance):
        """
        奖励设计（peak-focused）：
        - 绝对项：当前敏感度峰与目标的距离（越接近越好）
        - 相对项：相对于上一步敏感度峰的改善（只要在往下压就有正奖励）
        - 成本项：notch 的深度和带宽越大惩罚越多，避免过度补偿
        - 稳定性：不稳定时给予大负奖励
        """
        sp_cur = float(performance["sensitivity_peak"])
        sp_tar = float(self.targets["sensitivity_peak"])

        # 1) 绝对项：距离目标越近越好
        abs_diff = abs(sp_cur - sp_tar)
        abs_scale = 2.0  # 控制曲线陡峭程度
        r_abs = self.weights["sensitivity_peak"] * np.exp(-abs_diff / abs_scale)

        # 2) 相对改善项：比上一步更好就加分
        if self.prev_sensitivity_peak is None:
            r_rel = 0.0
        else:
            delta = self.prev_sensitivity_peak - sp_cur  # 正值表示改善
            rel_scale = 1.0
            r_rel = self.weights["sensitivity_peak"] * (delta / rel_scale)

        # 更新 prev_sensitivity_peak
        self.prev_sensitivity_peak = sp_cur

        # 3) notch 成本项：避免过宽 / 过深
        vcm_depth = float(self.current_notch_params[2])
        pzt_depth = float(self.current_notch_params[5])
        vcm_bw = float(self.current_notch_params[1])
        pzt_bw = float(self.current_notch_params[4])

        k_depth = 0.02
        k_bw = 0.02
        bw_ref = 5000.0

        cost_depth = k_depth * (abs(vcm_depth) + abs(pzt_depth))
        cost_bw = k_bw * (vcm_bw / bw_ref + pzt_bw / bw_ref)
        r_cost = -(cost_depth + cost_bw)

        # 4) 稳定性惩罚
        if performance["stability"] < 0.5:
            r_stab = -10.0
        else:
            r_stab = 0.0

        reward = r_abs + r_rel + r_cost + r_stab
        return float(reward)

    def _is_done(self, performance):
        pm_ok = abs(performance["phase_margin"] - self.targets["phase_margin"]) < 2
        gm_ok = abs(performance["gain_margin"] - self.targets["gain_margin"]) < 1
        sp_ok = performance["sensitivity_peak"] < (self.targets["sensitivity_peak"] + 1.0)
        stable = performance["stability"] > 0.8
        return pm_ok and gm_ok and sp_ok and stable


class PPOPolicy(nn.Module):
    """PPO Policy Network"""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.policy_mean = nn.Linear(hidden_dim, action_dim)
        self.policy_std = nn.Linear(hidden_dim, action_dim)
        self.value_net = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m == self.policy_mean:
                    nn.init.orthogonal_(m.weight, gain=0.01)
                    nn.init.constant_(m.bias, 0)
                elif m == self.policy_std:
                    nn.init.orthogonal_(m.weight, gain=0.01)
                    nn.init.constant_(m.bias, -1.0)
                elif m == self.value_net:
                    nn.init.orthogonal_(m.weight, gain=1.0)
                    nn.init.constant_(m.bias, 0)
                else:
                    nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                    nn.init.constant_(m.bias, 0)

    def forward(self, state, action_low=None, action_high=None):
        """Forward pass"""
        shared_features = self.shared_net(state)

        mean_raw = self.policy_mean(shared_features)
        mean_tanh = torch.tanh(mean_raw)

        if action_low is not None and action_high is not None:
            if isinstance(action_low, np.ndarray):
                action_low_t = torch.tensor(
                    action_low, dtype=torch.float32, device=mean_tanh.device
                )
                action_high_t = torch.tensor(
                    action_high, dtype=torch.float32, device=mean_tanh.device
                )
            else:
                action_low_t = torch.tensor(
                    action_low, dtype=torch.float32, device=mean_tanh.device
                )
                action_high_t = torch.tensor(
                    action_high, dtype=torch.float32, device=mean_tanh.device
                )

            action_mid = (action_low_t + action_high_t) / 2.0
            action_range = (action_high_t - action_low_t) / 2.0
            mean = action_mid + mean_tanh * action_range
        else:
            mean = mean_tanh

        log_std = self.policy_std(shared_features)
        log_std = torch.clamp(log_std, -20.0, 2.0)
        std = torch.exp(log_std)

        if action_low is not None and action_high is not None:
            if isinstance(action_low, np.ndarray):
                action_low_t = torch.tensor(
                    action_low, dtype=torch.float32, device=std.device
                )
                action_high_t = torch.tensor(
                    action_high, dtype=torch.float32, device=std.device
                )
            else:
                action_low_t = torch.tensor(
                    action_low, dtype=torch.float32, device=std.device
                )
                action_high_t = torch.tensor(
                    action_high, dtype=torch.float32, device=std.device
                )

            action_range = (action_high_t - action_low_t) / 2.0
            std = std * action_range

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

        self.policy = PPOPolicy(state_dim, action_dim, config.hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.learning_rate)

        self.clip_ratio = config.clip_ratio
        self.value_loss_coef = config.value_loss_coef
        self.entropy_coef = config.entropy_coef
        self.max_grad_norm = config.max_grad_norm

        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def get_action(self, state, deterministic=False):
        """Get action"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        action_low = None
        action_high = None
        if hasattr(self, "action_low") and hasattr(self, "action_high"):
            action_low = self.action_low
            action_high = self.action_high

        with torch.no_grad():
            if deterministic:
                action, value = self.policy.get_action(
                    state_tensor, action_low, action_high, deterministic=True
                )
                act_np = action.cpu().numpy()[0]
                if action_low is not None and action_high is not None:
                    act_np = np.clip(act_np, np.array(action_low), np.array(action_high))
                return act_np, value.cpu().numpy()[0]
            else:
                action, log_prob, value = self.policy.get_action(
                    state_tensor, action_low, action_high
                )
                act_np = action.cpu().numpy()[0]
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

        states = torch.FloatTensor(self.states).to(self.device)
        actions = torch.FloatTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        old_values = torch.FloatTensor(self.values).to(self.device)

        rewards = self._compute_discounted_rewards()

        advantages = rewards - old_values.squeeze()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        action_low = None
        action_high = None
        if hasattr(self, "action_low") and hasattr(self, "action_high"):
            action_low = self.action_low
            action_high = self.action_high

        for _ in range(self.config.ppo_epochs):
            mean, std, values = self.policy(states, action_low, action_high)

            dist = Normal(mean, std)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = nn.MSELoss()(values.squeeze(), rewards)
            entropy_loss = -dist.entropy().mean()

            total_loss = (
                policy_loss
                + self.value_loss_coef * value_loss
                + self.entropy_coef * entropy_loss
            )

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

        self.clear_buffer()

    def _compute_discounted_rewards(self):
        """Compute discounted rewards"""
        rewards = np.array(self.rewards, dtype=np.float32)
        dones = np.array(self.dones, dtype=bool)

        discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
        running_reward = 0.0

        for t in reversed(range(len(rewards))):
            if dones[t]:
                running_reward = 0.0
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
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            filepath,
        )

    def load_model(self, filepath):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


def train_peak_focused_notch_designer():
    """Train peak-focused notch filter designer"""
    print("=== Training Peak-Focused HDD Notch Filter Designer (PPO) ===")

    # Create environment
    env_config = config.get_simple_config()
    env = PeakFocusedHDDNotchDesignEnv(env_config)

    # Create agent
    train_config = config.TrainingConfig()

    state_dim = env.observation_space["shape"][0]
    action_dim = env.action_space["shape"][0]
    agent = PPOAgent(state_dim, action_dim, train_config)
    agent.action_low = env.action_space["low"]
    agent.action_high = env.action_space["high"]

    print("Environment info:")
    print(f"  State dimension: {state_dim}")
    print(f"  Action dimension: {action_dim}")
    print(f"  Plant case: RT (Randomized)")
    print("  Strategy: notch freq guided by dominant plant peaks (P*Fm)")

    episode_rewards = []
    best_reward = float("-inf")

    models_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(models_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for episode in range(train_config.max_episodes):
        state = env.reset()
        episode_reward = 0.0

        for step in range(train_config.max_steps_per_episode):
            action, log_prob, value = agent.get_action(state)

            next_state, reward, done, info = env.step(action)

            agent.store_transition(state, action, reward, value, log_prob, done)

            state = next_state
            episode_reward += reward

            if done:
                break

        agent.update()

        episode_rewards.append(episode_reward)

        if episode_reward > best_reward:
            best_reward = episode_reward
            best_path = os.path.join(
                models_dir, f"peak_focused_notch_designer_best_{timestamp}.pth"
            )
            agent.save_model(best_path)
            print(f"Saved best model to: {os.path.abspath(best_path)}")

        if episode % train_config.log_interval == 0:
            avg_reward = np.mean(episode_rewards[-train_config.log_interval :])
            print(
                f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Best: {best_reward:.2f}"
            )

    # Persist reward history
    rewards_array = np.array(episode_rewards, dtype=np.float32)
    rewards_npy_path = os.path.join(
        models_dir, f"peak_focused_notch_rewards_{timestamp}.npy"
    )
    rewards_csv_path = os.path.join(
        models_dir, f"peak_focused_notch_rewards_{timestamp}.csv"
    )
    np.save(rewards_npy_path, rewards_array)
    np.savetxt(
        rewards_csv_path,
        rewards_array,
        fmt="%.6f",
        delimiter=",",
        header="episode_reward",
        comments="",
    )
    print(f"Saved reward history to: {rewards_npy_path} and {rewards_csv_path}")

    # Plot reward curve
    plt.figure(figsize=(10, 4))
    plt.plot(episode_rewards, label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Peak-Focused Notch Designer Training Rewards")
    plt.grid(True, alpha=0.3)
    plt.legend()
    reward_plot_path = os.path.join(
        models_dir, f"peak_focused_notch_rewards_{timestamp}.png"
    )
    plt.tight_layout()
    plt.savefig(reward_plot_path)
    plt.close()
    print(f"Saved reward plot to: {reward_plot_path}")

    # Save final model
    final_path = os.path.join(
        models_dir, f"final_peak_focused_notch_designer_{timestamp}.pth"
    )
    agent.save_model(final_path)
    print(f"Saved final model to: {os.path.abspath(final_path)}")


if __name__ == "__main__":
    train_peak_focused_notch_designer()


