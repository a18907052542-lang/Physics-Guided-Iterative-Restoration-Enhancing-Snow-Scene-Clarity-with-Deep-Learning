"""
Fig. 3 — Snow physics decomposition visualization.

Reads physics maps saved by test.py (results/physics/) and produces a
single composite figure with three panels:
    (a) scattering coefficient
    (b) transmission map
    (c) occlusion mask

Usage
-----
    python visualization/plot_physics_decomposition.py \
        --physics_dir ./results/physics --output ./figs/fig3.png
"""

import os
import argparse
import glob

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--physics_dir', type=str, default='./results/physics')
    parser.add_argument('--index', type=str, default=None,
                        help='4-digit image index (e.g. 0007). Default: first found.')
    parser.add_argument('--input_image', type=str, default=None,
                        help='Optional path to the snowy input image for context.')
    parser.add_argument('--output', type=str, default='./figs/fig3.png')
    args = parser.parse_args()

    if args.index is None:
        files = sorted(glob.glob(os.path.join(args.physics_dir, '*_scattering.png')))
        if not files:
            raise FileNotFoundError(
                f'No physics maps found under {args.physics_dir}. '
                f'Run test.py first to generate them.')
        idx = os.path.basename(files[0]).split('_')[0]
    else:
        idx = args.index

    scattering = mpimg.imread(os.path.join(args.physics_dir, f'{idx}_scattering.png'))
    transmission = mpimg.imread(os.path.join(args.physics_dir, f'{idx}_transmission.png'))
    occlusion = mpimg.imread(os.path.join(args.physics_dir, f'{idx}_occlusion.png'))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(scattering, cmap='viridis'); axes[0].set_title('(a) Scattering coefficient β(x,y)')
    axes[1].imshow(transmission, cmap='magma');   axes[1].set_title('(b) Transmission rate t(x,y)')
    axes[2].imshow(occlusion, cmap='gray');       axes[2].set_title('(c) Occlusion mask M(x,y)')
    for ax in axes:
        ax.axis('off')

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f'Saved -> {args.output}')


if __name__ == '__main__':
    main()
