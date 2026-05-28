"""
Fig. 12 — correlation between restoration quality and downstream usefulness.

Reads results/downstream/downstream_results.json (must contain
`per_sample_restoration_metrics`). The plot shows a scatter of
SnowDenoiseMetric vs PSNR for each test image, with a horizontal banner
indicating proxy vs real downstream mode.

Usage
-----
    python visualization/plot_downstream_correlation.py \
        --results ./results/downstream/downstream_results.json \
        --output ./figs/fig12.png
"""

import os
import json
import argparse

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str,
                        default='./results/downstream/downstream_results.json')
    parser.add_argument('--output', type=str, default='./figs/fig12.png')
    args = parser.parse_args()

    with open(args.results) as f:
        data = json.load(f)
    per_sample = data.get('per_sample_restoration_metrics', [])
    if not per_sample:
        raise ValueError(
            f'`per_sample_restoration_metrics` missing in {args.results}. '
            f'Run experiments/downstream_eval.py first.'
        )

    psnr = [r['PSNR'] for r in per_sample]
    sdm = [r['SnowDenoiseMetric'] for r in per_sample]
    ssim = [r['SSIM'] for r in per_sample]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].scatter(psnr, sdm, alpha=0.7, s=18, c='#1f77b4')
    if len(psnr) >= 2:
        p = np.polyfit(psnr, sdm, 1)
        xs = np.linspace(min(psnr), max(psnr), 50)
        axes[0].plot(xs, np.polyval(p, xs), 'r--', linewidth=2,
                     label=f'fit: y = {p[0]:.3f}x + {p[1]:.3f}')
        axes[0].legend()
    axes[0].set_xlabel('PSNR (dB)'); axes[0].set_ylabel('SnowDenoiseMetric')
    axes[0].set_title('PSNR vs composite SnowDenoiseMetric')
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(ssim, sdm, alpha=0.7, s=18, c='#2ca02c')
    if len(ssim) >= 2:
        p = np.polyfit(ssim, sdm, 1)
        xs = np.linspace(min(ssim), max(ssim), 50)
        axes[1].plot(xs, np.polyval(p, xs), 'r--', linewidth=2,
                     label=f'fit: y = {p[0]:.3f}x + {p[1]:.3f}')
        axes[1].legend()
    axes[1].set_xlabel('SSIM'); axes[1].set_ylabel('SnowDenoiseMetric')
    axes[1].set_title('SSIM vs composite SnowDenoiseMetric')
    axes[1].grid(True, alpha=0.3)

    banner = 'mode: ' + data.get('mode', 'unknown')
    if data.get('proxy'):
        banner += '  (downstream metrics are FEATURE-DISTANCE PROXIES, ' \
                  'NOT real YOLOv8 mAP / DeepLabV3+ mIoU)'
    fig.suptitle(banner, fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f'Saved -> {args.output}')


if __name__ == '__main__':
    main()
