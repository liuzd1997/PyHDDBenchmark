"""
Evaluate saved PPO notch-design model(s).

Usage from PowerShell:
    python evaluate_model.py --model models/final_simple_notch_designer.pth --runs 3 --deterministic

This script imports the project's helper `load_model_and_predict` from
`simple_rl_notch_designer.py` and prints per-run notch parameters and performance.
"""
import os
import sys
import argparse
import json
from statistics import mean

# Ensure repo root is on sys.path
repo_root = os.path.dirname(os.path.abspath(__file__))
if repo_root not in sys.path:
    sys.path.append(repo_root)

from simple_rl_notch_designer import load_model_and_predict


def pretty_print_run(i, res):
    v = res['vcm_params']
    p = res['pzt_params']
    perf = res['performance']
    print(f"Run #{i+1}")
    print(f"  Action: {res['action']}")
    print(f"  VCM notch: center={v.center_freq:.1f} Hz, bw={v.bandwidth:.1f} Hz, depth={v.depth:.1f} dB, Q={v.q_factor:.2f}")
    print(f"  PZT notch: center={p.center_freq:.1f} Hz, bw={p.bandwidth:.1f} Hz, depth={p.depth:.1f} dB, Q={p.q_factor:.2f}")
    print('  Performance:')
    for k, val in perf.items():
        try:
            print(f"    {k}: {float(val):.4g}")
        except Exception:
            print(f"    {k}: {val}")
    print("")


def aggregate_metrics(results):
    # Collect numeric metrics into lists
    keys = results[0]['performance'].keys()
    agg = {}
    for k in keys:
        vals = []
        for r in results:
            try:
                vals.append(float(r['performance'][k]))
            except Exception:
                pass
        if vals:
            agg[k] = { 'mean': mean(vals), 'min': min(vals), 'max': max(vals) }
    return agg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate saved notch-design PPO model')
    parser.add_argument('--model', '-m', type=str, default=os.path.join(repo_root, 'models', 'final_simple_notch_designer.pth'), help='Path to saved model checkpoint (.pth)')
    parser.add_argument('--runs', '-n', type=int, default=1, help='Number of deterministic runs to average')
    parser.add_argument('--deterministic', action='store_true', help='Use deterministic action (policy mean)')
    parser.add_argument('--out', '-o', type=str, default=None, help='Optional JSON output file to save results')

    args = parser.parse_args()

    model_path = args.model
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        sys.exit(2)

    results = []
    for i in range(args.runs):
        res = load_model_and_predict(model_path, env=None, deterministic=args.deterministic)
        pretty_print_run(i, res)
        results.append(res)

    if results:
        agg = aggregate_metrics(results)
        print("Aggregate performance:")
        for k, stats in agg.items():
            print(f"  {k}: mean={stats['mean']:.4g}, min={stats['min']:.4g}, max={stats['max']:.4g}")

    if args.out:
        with open(args.out, 'w') as f:
            json.dump({'model': os.path.abspath(model_path), 'results': [
                {
                    'action': r['action'] if hasattr(r['action'], 'tolist') else list(r['action']),
                    'vcm_params': {'center_freq': r['vcm_params'].center_freq, 'bandwidth': r['vcm_params'].bandwidth, 'depth': r['vcm_params'].depth},
                    'pzt_params': {'center_freq': r['pzt_params'].center_freq, 'bandwidth': r['pzt_params'].bandwidth, 'depth': r['pzt_params'].depth},
                    'performance': r['performance']
                } for r in results
            ]}, f, indent=2)
        print(f"Saved results to {args.out}")

    print('Done')
