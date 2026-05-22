import argparse
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle
from control import matlab

import config
import plant
import utils
from simple_rl_notch_designer import SimpleHDDNotchDesignEnv, PPOAgent

FREQ_VECTOR = np.logspace(1, np.log10(60000), 3000)


def build_agent(env_config):
    """Create env and agent for inference."""
    env = SimpleHDDNotchDesignEnv(env_config)
    train_cfg = config.TrainingConfig()
    state_dim = env.observation_space['shape'][0]
    action_dim = env.action_space['shape'][0]
    agent = PPOAgent(state_dim, action_dim, train_cfg)
    agent.action_low = env.action_space['low']
    agent.action_high = env.action_space['high']
    return env, agent


def _freqresp_1d(system, freq_hz):
    """Utility to get 1D complex frequency response."""
    resp = utils.freqresp(system, 2 * np.pi * freq_hz)
    resp = np.array(resp)
    if resp.ndim == 1:
        return resp
    return resp[:, 0]


def _build_notch_filter(f0, bw, depth_db, Ts, Mr_f):
    """Create discrete-time notch filter using original multirate sampling."""
    sample_time = Ts / Mr_f
    if depth_db >= 0:
        return matlab.tf([1.0], [1.0], sample_time)
    wn = 2 * np.pi * f0
    zeta = bw / (2 * f0)
    depth_lin = 10 ** (depth_db / 20.0)
    # Correct Notch implementation: reduced damping in numerator
    num = [1, 2 * zeta * wn * max(depth_lin, 1e-4), wn**2]
    den = [1, 2 * zeta * wn, wn**2]
    notch_ct = matlab.tf(num, den)
    return matlab.c2d(notch_ct, sample_time, 'zoh')


def _create_digital_path(sys_pc, sys_fm, notch_tf, Ts, Mr_f):
    """Construct discrete-time plant path with multirate filter and notch."""
    sys_pdm0 = matlab.c2d(sys_pc, Ts / Mr_f, 'zoh')
    sys_pdm = sys_pdm0 * sys_fm * notch_tf
    return utils.dts_resampling(sys_pdm, Mr_f)


def compute_full_response(env, notch_params, freq_hz, plant_case):
    """Evaluate multirate filter & sensitivity using full pipeline."""
    Ts = env.Ts
    Mr_f = env.Mr_f
    sys_fm_vcm = env.Sys_Fm_vcm
    sys_fm_pzt = env.Sys_Fm_pzt
    sys_cd_vcm = env.Sys_Cd_vcm
    sys_cd_pzt = env.Sys_Cd_pzt
    case_suffix = plant_case if plant_case.startswith('c') else f'c{plant_case}'
    sys_pc_vcm = getattr(plant, f"Sys_Pc_vcm_{case_suffix}")
    sys_pc_pzt = getattr(plant, f"Sys_Pc_pzt_{case_suffix}")

    notch_vcm_tf = _build_notch_filter(notch_params[0], notch_params[1], notch_params[2], Ts, Mr_f)
    notch_pzt_tf = _build_notch_filter(notch_params[3], notch_params[4], notch_params[5], Ts, Mr_f)

    sys_pd_vcm = _create_digital_path(sys_pc_vcm, sys_fm_vcm, notch_vcm_tf, Ts, Mr_f)
    sys_pd_pzt = _create_digital_path(sys_pc_pzt, sys_fm_pzt, notch_pzt_tf, Ts, Mr_f)

    Fr_Pd_vcm = _freqresp_1d(sys_pd_vcm, freq_hz)
    Fr_Pd_pzt = _freqresp_1d(sys_pd_pzt, freq_hz)
    Fr_Cd_vcm = _freqresp_1d(sys_cd_vcm, freq_hz)
    Fr_Cd_pzt = _freqresp_1d(sys_cd_pzt, freq_hz)

    Fr_L_vcm = Fr_Pd_vcm * Fr_Cd_vcm
    Fr_L_pzt = Fr_Pd_pzt * Fr_Cd_pzt
    Fr_L = Fr_L_vcm + Fr_L_pzt
    Fr_S = 1.0 / (1.0 + Fr_L)

    Fr_Fm_vcm = _freqresp_1d(sys_fm_vcm * notch_vcm_tf, freq_hz)
    Fr_Fm_pzt = _freqresp_1d(sys_fm_pzt * notch_pzt_tf, freq_hz)

    performance = env._calculate_performance_metrics(Fr_L, Fr_S)

    return {
        'Fm_vcm': Fr_Fm_vcm,
        'Fm_pzt': Fr_Fm_pzt,
        'L': Fr_L,
        'S': Fr_S,
        'performance': performance
    }


def plot_case(freq, base_data, notch_data, output_path, case_idx, reward, performance, notch_params=None):
    """Generate comparison plots for multirate filters and sensitivity."""
    mag = lambda x: 20 * np.log10(np.abs(x) + 1e-12)

    # Set larger font sizes for better readability (especially when zoomed)
    plt.rcParams.update({
        'font.size': 16,         # Base font size
        'axes.titlesize': 18,    # Subplot titles
        'axes.labelsize': 16,    # Axis labels
        'xtick.labelsize': 14,   # X tick labels
        'ytick.labelsize': 14,   # Y tick labels
        'legend.fontsize': 14    # Legend
    })

    # Increase figure size and DPI for better clarity when zoomed
    fig, axes = plt.subplots(4, 1, figsize=(14, 18), sharex=True, dpi=300)

    # Extract notch parameters if provided
    vcm_notch_freq = None
    pzt_notch_freq = None
    vcm_notch_bw = None
    pzt_notch_bw = None
    vcm_notch_depth = None
    pzt_notch_depth = None
    if notch_params is not None:
        vcm_notch_freq = notch_params[0]  # VCM center frequency
        vcm_notch_bw = notch_params[1]    # VCM bandwidth
        vcm_notch_depth = notch_params[2] # VCM depth
        pzt_notch_freq = notch_params[3]  # PZT center frequency
        pzt_notch_bw = notch_params[4]    # PZT bandwidth
        pzt_notch_depth = notch_params[5] # PZT depth

    # Plot 1: VCM Multirate Filter
    axes[0].semilogx(freq, mag(base_data['Fm_vcm']), label='VCM multi-rate filter', linewidth=2)
    axes[0].semilogx(freq, mag(notch_data['Fm_vcm']), label='VCM multi-rate filter (with new notch)', linewidth=2)
    if vcm_notch_freq is not None:
        axes[0].axvline(vcm_notch_freq, color='red', linestyle='--', linewidth=1.0, alpha=0.8)
    axes[0].set_ylabel('Magnitude (dB)')
    # Add VCM notch parameters to title
    if vcm_notch_freq is not None:
        vcm_title = f'VCM Multirate Filter (New Notch: freq={vcm_notch_freq:.2f} Hz, width={vcm_notch_bw:.2f} Hz, depth={vcm_notch_depth:.2f} dB)'
    else:
        vcm_title = 'VCM Multirate Filter'
    axes[0].set_title(vcm_title)
    axes[0].grid(True, which='both', alpha=0.3)
    axes[0].legend()

    # Plot 2: PZT Multirate Filter
    axes[1].semilogx(freq, mag(base_data['Fm_pzt']), label='PZT multi-rate filter', linewidth=2)
    axes[1].semilogx(freq, mag(notch_data['Fm_pzt']), label='PZT multi-rate filter (with new notch)', linewidth=2)
    if pzt_notch_freq is not None:
        axes[1].axvline(pzt_notch_freq, color='green', linestyle='--', linewidth=1.0, alpha=0.8)
    axes[1].set_ylabel('Magnitude (dB)')
    # Add PZT notch parameters to title
    if pzt_notch_freq is not None:
        pzt_title = f'PZT Multirate Filter (New Notch: freq={pzt_notch_freq:.2f} Hz, width={pzt_notch_bw:.2f} Hz, depth={pzt_notch_depth:.2f} dB)'
    else:
        pzt_title = 'PZT Multirate Filter'
    axes[1].set_title(pzt_title)
    axes[1].grid(True, which='both', alpha=0.3)
    axes[1].legend()

    # Plot 3: Open Loop
    axes[2].semilogx(freq, mag(base_data['L']), label='Open Loop L', linewidth=2)
    axes[2].semilogx(freq, mag(notch_data['L']), label='Open Loop L (with new notch)', linewidth=2)
    if vcm_notch_freq is not None:
        axes[2].axvline(vcm_notch_freq, color='red', linestyle='--', linewidth=1.0, alpha=0.8)
    if pzt_notch_freq is not None:
        axes[2].axvline(pzt_notch_freq, color='green', linestyle='--', linewidth=1.0, alpha=0.8)
    axes[2].set_ylabel('Magnitude (dB)')
    axes[2].set_title('Open Loop (L)')
    axes[2].grid(True, which='both', alpha=0.3)
    axes[2].legend()

    # Plot 4: Sensitivity
    axes[3].semilogx(freq, mag(base_data['S']), label='Sensitivity', linewidth=2)
    axes[3].semilogx(freq, mag(notch_data['S']), label='Sensitivity (with newnotch)', linewidth=2)
    if vcm_notch_freq is not None:
        axes[3].axvline(vcm_notch_freq, color='red', linestyle='--', linewidth=1.0, alpha=0.8)
    if pzt_notch_freq is not None:
        axes[3].axvline(pzt_notch_freq, color='green', linestyle='--', linewidth=1.0, alpha=0.8)
    axes[3].set_xlabel('Frequency (Hz)')
    axes[3].set_ylabel('Magnitude (dB)')
    axes[3].set_title(f'Sensitivity Comparison')
    axes[3].grid(True, which='both', alpha=0.3)
    axes[3].legend()

    # Add a zoomed-in inset around the notch frequency region on the Open Loop (L) plot
    notch_freqs = [f for f in (vcm_notch_freq, pzt_notch_freq) if f is not None]
    if len(notch_freqs) > 0:
        # Determine zoom region in frequency
        f_min = min(notch_freqs)
        f_max = max(notch_freqs)
        # Give some margin around notches
        f_min *= 0.8
        f_max *= 1.2

        # Ensure the zoom window lies within frequency vector range
        f_min = max(f_min, freq[0])
        f_max = min(f_max, freq[-1])

        zoom_mask = (freq >= f_min) & (freq <= f_max)
        if np.any(zoom_mask):
            # Use Open Loop L magnitude for inset (consistent with subplot 3)
            base_L_db = mag(base_data['L'])[zoom_mask]
            notch_L_db = mag(notch_data['L'])[zoom_mask]

            # Determine y-limits with small margins
            y_min = min(base_L_db.min(), notch_L_db.min())
            y_max = max(base_L_db.max(), notch_L_db.max())
            y_margin = 0.05 * (y_max - y_min + 1e-6)
            y_min -= y_margin
            y_max += y_margin

            # Create inset axes in upper-right corner of open-loop plot
            ax_ins = inset_axes(axes[2], width="50%", height="40%", loc='upper right')
            ax_ins.semilogx(freq[zoom_mask], base_L_db, label='Open Loop L', linewidth=2)
            ax_ins.semilogx(freq[zoom_mask], notch_L_db, label='Open Loop L (with new notch)', linewidth=2)
            if vcm_notch_freq is not None:
                ax_ins.axvline(vcm_notch_freq, color='red', linestyle='--', linewidth=0.8, alpha=0.8)
            if pzt_notch_freq is not None:
                ax_ins.axvline(pzt_notch_freq, color='green', linestyle='--', linewidth=0.8, alpha=0.8)
            ax_ins.set_xlim(f_min, f_max)
            ax_ins.set_ylim(y_min, y_max)
            ax_ins.grid(True, which='both', alpha=0.3)

            # Draw a rectangle on the main axes that exactly matches the inset view
            # so the zoomed region and框位置一一对应
            rect = Rectangle(
                (f_min, y_min),
                f_max - f_min,
                y_max - y_min,
                linewidth=1.0,
                edgecolor='0.5',
                facecolor='none',
                linestyle='--',
                alpha=0.9,
            )
            axes[2].add_patch(rect)

    # Notch parameters are now shown in subplot titles instead of at the top

    # fig.suptitle(f'Notch Analysis Case {case_idx+1}\nReward={reward:.2f}, PM={performance["phase_margin"]:.1f}°, GM={performance["gain_margin"]:.1f}dB')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    # Save with high DPI for clarity when zoomed
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def save_case_summary(output_dir, timestamp, case_idx, performance, notch_params, reward):
    """Persist notch parameters and metrics for quick review."""
    summary_path = os.path.join(output_dir, f'notch_case_{case_idx+1}_{timestamp}.txt')
    with open(summary_path, 'w', encoding='utf-8') as fp:
        fp.write(f"Case {case_idx+1} Summary - {timestamp}\n")
        fp.write(f"Reward: {reward:.4f}\n")
        perf = performance
        fp.write(
            f"Performance -> PM: {perf['phase_margin']:.2f} deg, "
            f"GM: {perf['gain_margin']:.2f} dB, "
            f"SensPeak: {perf['sensitivity_peak']:.2f} dB, "
            f"TrackingErr: {perf['tracking_error']:.4f}, "
            f"Stability: {perf['stability']}\n"
        )
        vcm = notch_params[:3]
        pzt = notch_params[3:]
        fp.write(f"VCM Notch -> f0={vcm[0]:.2f} Hz, bw={vcm[1]:.2f} Hz, depth={vcm[2]:.2f} dB\n")
        fp.write(f"PZT Notch -> f0={pzt[0]:.2f} Hz, bw={pzt[1]:.2f} Hz, depth={pzt[2]:.2f} dB\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze trained notch designer output with full multirate filters.")
    parser.add_argument('--model', required=True, help='Path to trained PPO model (.pth)')
    parser.add_argument('--cases', type=int, default=1, help='Number of analyses to run')
    parser.add_argument('--output-dir', default='analysis_outputs', help='Directory for plots and summaries')
    parser.add_argument('--stochastic', action='store_true', help='Sample stochastic actions instead of deterministic mean')
    parser.add_argument('--plant-case', default='c2', help="Plant case to evaluate (e.g., c2). Must match entries in plant.py")
    args = parser.parse_args()

    env_config = config.get_simple_config()
    env, agent = build_agent(env_config)
    agent.load_model(args.model)

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for case_idx in range(args.cases):
        state = env.reset()
        action, value = agent.get_action(state, deterministic=not args.stochastic)
        _, reward, _, info = env.step(action)

        notch_params = np.array(info['notch_params'])
        base_params = notch_params.copy()
        base_params[2] = 0.0
        base_params[5] = 0.0

        base_data = compute_full_response(env, base_params, FREQ_VECTOR, args.plant_case)
        notch_data = compute_full_response(env, notch_params, FREQ_VECTOR, args.plant_case)

        plot_path = os.path.join(
            args.output_dir,
            f"notch_analysis_case{case_idx+1}_{timestamp}.png"
        )
        plot_case(
            FREQ_VECTOR,
            base_data,
            notch_data,
            plot_path,
            case_idx,
            reward,
            notch_data['performance'],
            notch_params  # Pass notch parameters to plot
        )
        save_case_summary(args.output_dir, timestamp, case_idx, notch_data['performance'], notch_params, reward)
        vcm = notch_params[:3]
        pzt = notch_params[3:]
        print(f"[Case {case_idx+1}] Reward={reward:.2f}, plot saved to {plot_path}")
        print(f"  VCM Notch: freq={vcm[0]:.2f} Hz, width={vcm[1]:.2f} Hz, depth={vcm[2]:.2f} dB")
        print(f"  PZT Notch: freq={pzt[0]:.2f} Hz, width={pzt[1]:.2f} Hz, depth={pzt[2]:.2f} dB")


if __name__ == "__main__":
    main()

