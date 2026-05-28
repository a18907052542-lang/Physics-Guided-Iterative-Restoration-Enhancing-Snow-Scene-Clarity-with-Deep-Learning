"""
Robustness evaluation for PAID-SnowNet.

Covers:
    Table 17 — Gaussian noise (sigma=10/20/30),
               salt-and-pepper (density=1/3/5%),
               motion blur (kernel=3/5/7)
    Table 18 — snowflake density (10/30/60/80%),
               morphology (flake/granular/mixed/irregular),
               optical transparency (alpha=0.3/0.5/0.8),
               temperature regime (wet/dry snow)
    Table 19 — JPEG compression (Q=70/50/30),
               low resolution (2x/4x downsample-then-upsample),
               bit-depth reduction (6-bit / 4-bit)

Usage
-----
    python experiments/robustness_eval.py --data_root <Snow100K_root> \
                                          [--checkpoint best.pth]
"""

import os
import sys
import json
import math
import argparse
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np

from torch.utils.data import DataLoader

from models.paid_snownet import PAIDSnowNet
from datasets.snow_dataset import Snow100KDataset
from utils.metrics import compute_psnr, compute_ssim, compute_lpips


# ===================================================== perturbation operators
# These functions DO NOT generate training data: they apply controlled
# perturbations to existing real test inputs so that robustness margins can
# be measured. The point is to perturb, not to fabricate.

def add_gaussian_noise(images, sigma):
    noise = torch.randn_like(images) * (sigma / 255.0)
    return (images + noise).clamp(0, 1)


def add_salt_pepper_noise(images, density):
    noisy = images.clone()
    B, C, H, W = images.shape
    num = int(density * H * W)
    half = num // 2
    for b in range(B):
        cy = torch.randint(0, H, (num,))
        cx = torch.randint(0, W, (num,))
        noisy[b, :, cy[:half], cx[:half]] = 1.0
        noisy[b, :, cy[half:], cx[half:]] = 0.0
    return noisy


def add_motion_blur(images, kernel_size):
    k = torch.zeros(kernel_size, kernel_size, device=images.device)
    k[kernel_size // 2, :] = 1.0 / kernel_size
    k = k.view(1, 1, kernel_size, kernel_size).repeat(3, 1, 1, 1)
    pad = kernel_size // 2
    padded = F.pad(images, [pad] * 4, mode='reflect')
    return F.conv2d(padded, k, groups=3).clamp(0, 1)


def add_low_resolution(images, scale):
    """Bicubic down then up."""
    B, C, H, W = images.shape
    h2, w2 = max(1, H // scale), max(1, W // scale)
    down = F.interpolate(images, size=(h2, w2), mode='bicubic', align_corners=False)
    up = F.interpolate(down, size=(H, W), mode='bicubic', align_corners=False)
    return up.clamp(0, 1)


def add_bit_depth(images, bits):
    levels = 2 ** bits
    q = torch.round(images * (levels - 1)) / (levels - 1)
    return q.clamp(0, 1)


def add_jpeg_like_compression(images, quality):
    """Block-DCT-like degradation. Not bit-exact JPEG, but yields the same
    block-artifact + high-frequency loss pattern that the paper reports."""
    factor = (100 - quality) / 100.0
    block = 8
    out = images.clone()
    B, C, H, W = out.shape
    for y in range(0, H - block + 1, block):
        for x in range(0, W - block + 1, block):
            patch = out[:, :, y:y + block, x:x + block]
            mu = patch.mean(dim=(-2, -1), keepdim=True)
            out[:, :, y:y + block, x:x + block] = patch * (1 - factor) + mu * factor
    return (out + torch.randn_like(out) * factor * 0.05).clamp(0, 1)


def add_snowflake_layer(images, coverage, morphology='flake', alpha_max=1.0):
    """Add an extra snowflake layer onto already-snowy inputs.
    morphology in {flake, granular, mixed, irregular}.
    alpha_max is the maximum opacity (per-pixel uniform in [0.3, alpha_max])."""
    B, C, H, W = images.shape
    mask = torch.zeros(B, 1, H, W, device=images.device)
    snowflakes = max(1, int(coverage * H * W / 25))
    for b in range(B):
        for _ in range(snowflakes):
            cy = random.randint(0, H - 1)
            cx = random.randint(0, W - 1)
            if morphology == 'flake':
                r = random.randint(1, 3)
                shape = 'disc'
            elif morphology == 'granular':
                r = 1
                shape = 'disc'
            elif morphology == 'mixed':
                r = random.choice([1, 2, 3, 4])
                shape = 'disc'
            elif morphology == 'irregular':
                r = random.randint(1, 4)
                shape = 'streak'
            else:
                r, shape = 2, 'disc'

            opacity = random.uniform(0.3, alpha_max)
            if shape == 'disc':
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        if dy * dy + dx * dx <= r * r:
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < H and 0 <= nx < W:
                                mask[b, 0, ny, nx] = max(mask[b, 0, ny, nx].item(), opacity)
            else:  # streak
                length = r * 3
                ang = random.uniform(-0.4, 0.4)  # nearly vertical
                for i in range(length):
                    ny = cy + i
                    nx = cx + int(i * math.tan(ang))
                    if 0 <= ny < H and 0 <= nx < W:
                        mask[b, 0, ny, nx] = max(mask[b, 0, ny, nx].item(), opacity)

    snowy = images * (1 - mask) + 0.95 * mask
    return snowy.clamp(0, 1)


# ============================================================== evaluator

@torch.no_grad()
def evaluate_with(perturb_fn, kwargs, model, loader, device, max_batches=50):
    model.eval()
    psnr = ssim = lpips_v = 0.0
    n = 0
    lpips_mode = None
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(device); y = y.to(device)
        xp = perturb_fn(x, **kwargs)
        restored, _, _ = model(xp)
        psnr += compute_psnr(restored, y)
        ssim += compute_ssim(restored, y)
        v, mode = compute_lpips(restored, y); lpips_v += v; lpips_mode = mode
        n += 1
    n = max(n, 1)
    return {'PSNR': psnr / n, 'SSIM': ssim / n,
            'LPIPS': lpips_v / n, 'LPIPS_mode': lpips_mode}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./results/robustness')
    parser.add_argument('--max_batches', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    random.seed(0)
    torch.manual_seed(0)

    model = PAIDSnowNet().to(device)
    if args.checkpoint and os.path.exists(args.checkpoint):
        sd = torch.load(args.checkpoint, map_location=device)
        sd = sd.get('model_state_dict', sd)
        model.load_state_dict(sd, strict=False)
        print(f'Loaded {args.checkpoint}')
    else:
        print('No checkpoint loaded — output reflects current weights only.')

    dataset = Snow100KDataset(args.data_root, split='test',
                              patch_size=256, augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=2)

    results = {}

    print('\n[Table 17] Gaussian noise / salt-pepper / motion blur')
    results['gaussian_noise'] = []
    for sigma in [10, 20, 30]:
        m = evaluate_with(add_gaussian_noise, {'sigma': sigma}, model, loader, device, args.max_batches)
        results['gaussian_noise'].append({'sigma': sigma, **m})
        print(f"  sigma={sigma:3d} | PSNR {m['PSNR']:.2f}")
    results['salt_pepper'] = []
    for d in [0.01, 0.03, 0.05]:
        m = evaluate_with(add_salt_pepper_noise, {'density': d}, model, loader, device, args.max_batches)
        results['salt_pepper'].append({'density': d, **m})
        print(f"  s&p {d*100:.0f}% | PSNR {m['PSNR']:.2f}")
    results['motion_blur'] = []
    for k in [3, 5, 7]:
        m = evaluate_with(add_motion_blur, {'kernel_size': k}, model, loader, device, args.max_batches)
        results['motion_blur'].append({'kernel_size': k, **m})
        print(f"  motion k={k} | PSNR {m['PSNR']:.2f}")

    print('\n[Table 18] Snowflake density / morphology / transparency')
    results['snow_density'] = []
    for cov in [0.10, 0.30, 0.60, 0.80]:
        m = evaluate_with(add_snowflake_layer, {'coverage': cov, 'morphology': 'mixed'},
                          model, loader, device, args.max_batches)
        results['snow_density'].append({'coverage': cov, **m})
        print(f"  density {cov*100:.0f}% | PSNR {m['PSNR']:.2f}")
    results['morphology'] = []
    for morph in ['flake', 'granular', 'mixed', 'irregular']:
        m = evaluate_with(add_snowflake_layer, {'coverage': 0.30, 'morphology': morph},
                          model, loader, device, args.max_batches)
        results['morphology'].append({'morphology': morph, **m})
        print(f"  morph {morph:9s} | PSNR {m['PSNR']:.2f}")
    results['transparency'] = []
    for alpha in [0.3, 0.5, 0.8]:
        m = evaluate_with(add_snowflake_layer,
                          {'coverage': 0.30, 'morphology': 'mixed', 'alpha_max': alpha},
                          model, loader, device, args.max_batches)
        results['transparency'].append({'alpha_max': alpha, **m})
        print(f"  alpha {alpha} | PSNR {m['PSNR']:.2f}")
    # Wet vs dry: we model wet snow as higher transparency / softer mask,
    # dry snow as low transparency. This is a proxy for temperature regimes.
    results['temperature_regime'] = []
    for label, alpha in [('wet_-2C_to_0C', 0.6), ('dry_-10C_to_-5C', 0.9)]:
        m = evaluate_with(add_snowflake_layer,
                          {'coverage': 0.30, 'morphology': 'mixed', 'alpha_max': alpha},
                          model, loader, device, args.max_batches)
        results['temperature_regime'].append({'regime': label, 'alpha_proxy': alpha, **m})
        print(f"  {label} | PSNR {m['PSNR']:.2f}")

    print('\n[Table 19] JPEG / low-resolution / bit-depth')
    results['jpeg_compression'] = []
    for q in [70, 50, 30]:
        m = evaluate_with(add_jpeg_like_compression, {'quality': q},
                          model, loader, device, args.max_batches)
        results['jpeg_compression'].append({'quality': q, **m})
        print(f"  JPEG Q={q} | PSNR {m['PSNR']:.2f}")
    results['low_resolution'] = []
    for s in [2, 4]:
        m = evaluate_with(add_low_resolution, {'scale': s},
                          model, loader, device, args.max_batches)
        results['low_resolution'].append({'scale': s, **m})
        print(f"  LR x{s}  | PSNR {m['PSNR']:.2f}")
    results['bit_depth'] = []
    for b in [6, 4]:
        m = evaluate_with(add_bit_depth, {'bits': b},
                          model, loader, device, args.max_batches)
        results['bit_depth'].append({'bits': b, **m})
        print(f"  bits {b} | PSNR {m['PSNR']:.2f}")

    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, 'robustness_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved -> {args.save_dir}/robustness_results.json')


if __name__ == '__main__':
    main()
