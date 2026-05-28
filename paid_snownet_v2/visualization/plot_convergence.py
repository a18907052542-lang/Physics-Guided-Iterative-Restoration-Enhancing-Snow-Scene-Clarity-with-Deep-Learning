"""
Fig. 5 & Fig. 8 — convergence visualization.

Fig. 5 shows PSNR / SSIM / LPIPS as a function of the iteration count T.
Fig. 8 shows the Theorem-1 bound vs the observed min ||grad||^2.

Both panels read from results/convergence/convergence_results.json
which is produced by analysis/convergence_analysis.py.

Usage
-----
    python visualization/plot_convergence.py \
        --results ./results/convergence/convergence_results.json \
        --output_fig5 ./figs/fig5.png \
        --output_fig8 ./figs/fig8.png
"""

import os
import json
import argparse

import matplotlib.pyplot as plt


def _abort_if_empty(path, key):
    with open(path) as f:
        data = json.load(f)
    if key not in data or not data[key]:
        raise ValueError(f'`{key}` is missing or empty in {path}. '
                         f'Run analysis/convergence_analysis.py first.')
    return data


def plot_fig5(data, output):
    rows = data['iteration_convergence']
    Ts = sorted([int(k) for k in rows.keys()])
    psnr = [rows[str(T)]['PSNR'] for T in Ts]
    ssim = [rows[str(T)]['SSIM'] for T in Ts]
    lpips_v = [rows[str(T)]['LPIPS'] for T in Ts]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    color_psnr = '#1f77b4'
    ax1.plot(Ts, psnr, 'o-', color=color_psnr, label='PSNR (dB)', linewidth=2, markersize=8)
    ax1.set_xlabel('Iteration count T')
    ax1.set_ylabel('PSNR (dB)', color=color_psnr)
    ax1.tick_params(axis='y', labelcolor=color_psnr)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color_ssim = '#2ca02c'
    ax2.plot(Ts, ssim, 's--', color=color_ssim, label='SSIM', linewidth=2, markersize=8)
    ax2.set_ylabel('SSIM', color=color_ssim)
    ax2.tick_params(axis='y', labelcolor=color_ssim)

    # LPIPS on a third visual axis via inset
    fig.subplots_adjust(right=0.78)
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('axes', 1.18))
    color_lp = '#d62728'
    ax3.plot(Ts, lpips_v, '^:', color=color_lp, label='LPIPS', linewidth=2, markersize=8)
    ax3.set_ylabel('LPIPS', color=color_lp)
    ax3.tick_params(axis='y', labelcolor=color_lp)

    lines = [ax1.get_lines()[0], ax2.get_lines()[0], ax3.get_lines()[0]]
    ax1.legend(lines, [l.get_label() for l in lines], loc='lower right')
    plt.title('Restoration quality vs iteration count T')

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    plt.savefig(output, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved -> {output}')


def plot_fig8(data, output):
    rows = data['theorem_verification']
    Ts = [r['T'] for r in rows]
    obs = [r['min_grad_norm_sq'] for r in rows]
    bnd = [r['theoretical_bound'] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(Ts, obs, 'o-', linewidth=2, label='Observed min ||grad||²')
    ax.semilogy(Ts, bnd, 's--', linewidth=2, label='Theoretical bound C·(L₀-L*)/√T')
    ax.set_xlabel('Iteration count T')
    ax.set_ylabel('Squared gradient norm (log scale)')
    ax.legend(loc='best')
    ax.grid(True, which='both', alpha=0.3)
    ax.set_title('Theorem 1 verification (synthetic quadratic problem)')

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    plt.savefig(output, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved -> {output}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str,
                        default='./results/convergence/convergence_results.json')
    parser.add_argument('--output_fig5', type=str, default='./figs/fig5.png')
    parser.add_argument('--output_fig8', type=str, default='./figs/fig8.png')
    args = parser.parse_args()
    data = _abort_if_empty(args.results, 'iteration_convergence')
    plot_fig5(data, args.output_fig5)
    plot_fig8(data, args.output_fig8)


if __name__ == '__main__':
    main()
