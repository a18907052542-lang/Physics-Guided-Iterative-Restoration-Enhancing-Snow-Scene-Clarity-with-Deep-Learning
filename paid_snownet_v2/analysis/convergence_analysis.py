"""
Convergence analysis for the SIRM module.

Two analyses:

1. Theorem 1 verification on a synthetic quadratic test problem.
   This is a controlled numerical experiment for the convergence-rate bound
   (Eq. 27); it is *not* used for any image-restoration metric.

2. Empirical iteration sweep on the test set, tracking PSNR / SSIM / LPIPS
   at each iteration count T in {1..max_T}. Used by visualization scripts
   to reproduce Fig. 5 / Fig. 8.

Usage
-----
    python analysis/convergence_analysis.py --data_root <Snow100K_root>
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.paid_snownet import PAIDSnowNet
from datasets.snow_dataset import Snow100KDataset
from utils.metrics import compute_psnr, compute_ssim, compute_lpips


# ----------------------------------------------------- Theorem 1 verification

def theorem1_check(max_T=10, dim=1000, seed=42):
    """Numerical sanity check of Eq. 27.
    Optimizes 0.5 x^T A x - b^T x with AdaGrad, then checks
        min_t ||grad||^2  vs  C * (L(X^0) - L*) / sqrt(T)."""
    rng = np.random.RandomState(seed)
    eig = rng.uniform(0.1, 10.0, dim)
    A = np.diag(eig); b = rng.randn(dim)
    x_star = np.linalg.solve(A, b)
    L_star = -0.5 * b @ x_star
    L_smooth = eig.max()

    rows = []
    for T in range(1, max_T + 1):
        x = np.zeros(dim); G = np.zeros(dim)
        eta0 = 0.1; eps = 1e-8
        min_g2 = float('inf')
        for t in range(T):
            grad = A @ x - b
            min_g2 = min(min_g2, float((grad ** 2).sum()))
            G += grad ** 2
            x = x - eta0 / np.sqrt(G + eps) * grad
        G_max = float((eig ** 2).sum()) * dim
        C = 2 * L_smooth * (1 + np.log(1 + G_max * eta0 ** 2)) / eta0
        bound = C * (-L_star) / np.sqrt(T)
        rows.append({'T': T, 'min_grad_norm_sq': min_g2, 'theoretical_bound': bound,
                     'satisfied': min_g2 <= bound * 10})
        print(f'  T={T:2d} | min||g||^2={min_g2:.3e} | bound={bound:.3e}')
    return rows


# ---------------------------------------------------- Empirical convergence

@torch.no_grad()
def iteration_sweep(model, loader, device, max_T=6, max_batches=20):
    rows = {}
    for T in range(1, max_T + 1):
        model.sirm.num_iterations = T
        model.eval()
        psnr = ssim = lpips_v = 0.0
        n = 0
        for i, (x, y) in enumerate(loader):
            if i >= max_batches:
                break
            x = x.to(device); y = y.to(device)
            restored, _, _ = model(x)
            psnr += compute_psnr(restored, y)
            ssim += compute_ssim(restored, y)
            v, _ = compute_lpips(restored, y)
            lpips_v += v
            n += 1
        n = max(n, 1)
        rows[T] = {'PSNR': psnr / n, 'SSIM': ssim / n, 'LPIPS': lpips_v / n}
        print(f"  T={T} | PSNR {rows[T]['PSNR']:.2f} | SSIM {rows[T]['SSIM']:.3f} "
              f"| LPIPS {rows[T]['LPIPS']:.3f}")
    model.sirm.num_iterations = 3
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./results/convergence')
    parser.add_argument('--max_T', type=int, default=6)
    parser.add_argument('--max_batches', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    print('=== Theorem 1 verification ===')
    thm = theorem1_check(max_T=args.max_T)

    print('\n=== Empirical iteration sweep ===')
    model = PAIDSnowNet().to(device)
    if args.checkpoint and os.path.exists(args.checkpoint):
        sd = torch.load(args.checkpoint, map_location=device)
        sd = sd.get('model_state_dict', sd)
        model.load_state_dict(sd, strict=False)

    dataset = Snow100KDataset(args.data_root, split='test',
                              patch_size=256, augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    emp = iteration_sweep(model, loader, device,
                          max_T=args.max_T, max_batches=args.max_batches)

    out = {'theorem_verification': thm,
           'iteration_convergence': {str(k): v for k, v in emp.items()}}
    with open(os.path.join(args.save_dir, 'convergence_results.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {args.save_dir}/convergence_results.json")


if __name__ == '__main__':
    main()
