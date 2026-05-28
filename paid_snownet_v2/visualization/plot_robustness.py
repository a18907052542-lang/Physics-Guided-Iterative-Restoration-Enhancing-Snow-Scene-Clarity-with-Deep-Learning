"""
Fig. 9 — robustness curves.

Plots PSNR vs perturbation strength for each perturbation family covered by
experiments/robustness_eval.py. Reads results/robustness/robustness_results.json.

Usage
-----
    python visualization/plot_robustness.py \
        --results ./results/robustness/robustness_results.json \
        --output ./figs/fig9.png
"""

import os
import json
import argparse

import matplotlib.pyplot as plt


_PANEL_SPECS = [
    # key in JSON, x-field, x-label, panel title
    ('gaussian_noise',     'sigma',      'σ (out of 255)',            'Gaussian noise'),
    ('salt_pepper',        'density',    'density',                   'Salt & pepper noise'),
    ('motion_blur',        'kernel_size', 'kernel size',               'Motion blur'),
    ('snow_density',       'coverage',   'snow coverage',             'Snowflake density'),
    ('transparency',       'alpha_max',  'max opacity',               'Snowflake transparency'),
    ('jpeg_compression',   'quality',    'JPEG quality',              'JPEG-like compression'),
    ('low_resolution',     'scale',      'downsample factor',         'Low resolution'),
    ('bit_depth',          'bits',       'bits/channel',              'Bit-depth reduction'),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str,
                        default='./results/robustness/robustness_results.json')
    parser.add_argument('--output', type=str, default='./figs/fig9.png')
    args = parser.parse_args()

    with open(args.results) as f:
        data = json.load(f)

    panels = [s for s in _PANEL_SPECS if s[0] in data]
    if not panels:
        raise ValueError(f'No expected keys in {args.results}.')

    cols = 4
    rows = (len(panels) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows))
    axes = axes.flatten() if rows > 1 else list(axes)

    for ax, (key, xfield, xlabel, title) in zip(axes, panels):
        rows_data = data[key]
        xs = [r[xfield] for r in rows_data]
        psnr = [r['PSNR'] for r in rows_data]
        ssim = [r['SSIM'] for r in rows_data]
        ax.plot(xs, psnr, 'o-', color='#1f77b4', label='PSNR (dB)', linewidth=2)
        ax.set_xlabel(xlabel); ax.set_ylabel('PSNR (dB)', color='#1f77b4')
        ax.tick_params(axis='y', labelcolor='#1f77b4'); ax.grid(True, alpha=0.3)
        ax.set_title(title)
        twin = ax.twinx()
        twin.plot(xs, ssim, 's--', color='#2ca02c', label='SSIM', linewidth=2)
        twin.set_ylabel('SSIM', color='#2ca02c')
        twin.tick_params(axis='y', labelcolor='#2ca02c')

    for ax in axes[len(panels):]:
        ax.axis('off')

    plt.suptitle('PAID-SnowNet robustness across perturbation types')
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f'Saved -> {args.output}')


if __name__ == '__main__':
    main()
