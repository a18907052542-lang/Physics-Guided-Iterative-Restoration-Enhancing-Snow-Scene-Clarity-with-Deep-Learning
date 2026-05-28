"""
Fig. 7 — module-level ablation bar chart.

Reads results/ablation/ablation_results.json (produced by
experiments/ablation_study.py) and plots PSNR / SSIM bars for the
configurations in Table 12.

Usage
-----
    python visualization/plot_ablation_modules.py \
        --results ./results/ablation/ablation_results.json \
        --output ./figs/fig7.png
"""

import os
import json
import argparse

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str,
                        default='./results/ablation/ablation_results.json')
    parser.add_argument('--output', type=str, default='./figs/fig7.png')
    args = parser.parse_args()

    with open(args.results) as f:
        data = json.load(f)
    if 'module' not in data:
        raise ValueError('Ablation file lacks the "module" key. '
                         'Run experiments/ablation_study.py with --only module first.')

    rows = data['module']
    names = [r['config'] for r in rows]
    psnr = [r['PSNR'] for r in rows]
    ssim = [r['SSIM'] for r in rows]

    x = np.arange(len(names))
    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    w = 0.38

    b1 = ax1.bar(x - w / 2, psnr, w, label='PSNR (dB)', color='#1f77b4')
    ax1.set_ylabel('PSNR (dB)', color='#1f77b4')
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    if psnr:
        ax1.set_ylim(0, max(psnr) * 1.15 if max(psnr) > 0 else 1.0)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=20, ha='right')

    ax2 = ax1.twinx()
    b2 = ax2.bar(x + w / 2, ssim, w, label='SSIM', color='#2ca02c')
    ax2.set_ylabel('SSIM', color='#2ca02c')
    ax2.tick_params(axis='y', labelcolor='#2ca02c')
    if ssim:
        ax2.set_ylim(0, max(ssim) * 1.15 if max(ssim) > 0 else 1.0)

    for b, v in zip(b1, psnr):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                 f'{v:.2f}', ha='center', va='bottom', fontsize=9, color='#1f77b4')
    for b, v in zip(b2, ssim):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                 f'{v:.3f}', ha='center', va='bottom', fontsize=9, color='#2ca02c')

    plt.title('Module-level ablation (Table 12)')
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f'Saved -> {args.output}')


if __name__ == '__main__':
    main()
