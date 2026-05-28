"""
Fig. 11 — Pareto frontier of PSNR vs model size.

Inputs needed
-------------
* results/ablation/ablation_results.json (variant key) — PAID-SnowNet variants
* Optional: --competitor_data path/to/competitors.json with format
      [{"name": "DesnowNet", "PSNR": 27.31, "Params_M": 15.6}, ...]
  If supplied, the competitors are overlaid on the same axes. This file is
  produced by your *real* runs against other methods; this script does NOT
  invent or fill in competitor numbers.

Usage
-----
    python visualization/plot_pareto.py \
        --results ./results/ablation/ablation_results.json \
        --competitor_data path/to/competitors.json \
        --output ./figs/fig11.png
"""

import os
import json
import argparse

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str,
                        default='./results/ablation/ablation_results.json',
                        help='Ablation JSON containing the network variants.')
    parser.add_argument('--competitor_data', type=str, default=None,
                        help='Optional JSON file of competitor (name,PSNR,Params_M) entries.')
    parser.add_argument('--output', type=str, default='./figs/fig11.png')
    args = parser.parse_args()

    with open(args.results) as f:
        ablation = json.load(f)
    variants = ablation.get('variant', [])
    if not variants:
        raise ValueError(
            f'`variant` key missing in {args.results}. '
            f'Run experiments/ablation_study.py --only variant first.'
        )

    fig, ax = plt.subplots(figsize=(8, 6))

    xs = [r['Params_M'] for r in variants]
    ys = [r['PSNR'] for r in variants]
    ax.scatter(xs, ys, s=140, marker='*', color='#d62728', zorder=3,
               label='PAID-SnowNet variants')
    for r in variants:
        ax.annotate(r['config'], (r['Params_M'], r['PSNR']),
                    textcoords='offset points', xytext=(6, 6), fontsize=9)

    if args.competitor_data and os.path.exists(args.competitor_data):
        with open(args.competitor_data) as f:
            comps = json.load(f)
        cx = [c['Params_M'] for c in comps]
        cy = [c['PSNR'] for c in comps]
        ax.scatter(cx, cy, s=80, marker='o', color='#1f77b4',
                   label='Competitor methods')
        for c in comps:
            ax.annotate(c['name'], (c['Params_M'], c['PSNR']),
                        textcoords='offset points', xytext=(6, -10), fontsize=9)
    else:
        ax.text(0.02, 0.02,
                'Competitor data not supplied — pass --competitor_data\n'
                'with a JSON list of {name,PSNR,Params_M} measured on your runs.',
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle='round', alpha=0.15, facecolor='#cccccc'),
                verticalalignment='bottom')

    ax.set_xlabel('Parameter count (M)')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('Pareto frontier: PSNR vs model size')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f'Saved -> {args.output}')


if __name__ == '__main__':
    main()
