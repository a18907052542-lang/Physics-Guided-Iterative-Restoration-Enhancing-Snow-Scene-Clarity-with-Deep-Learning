"""
Table 21 — runtime analysis at different resolutions.

Records wall-clock inference time for PAID-SnowNet (and, if you point it
at adapter scripts for competitor models, those too) at
256x256, 512x512, 1024x1024, 2048x2048.

Usage
-----
    python experiments/runtime_analysis.py [--checkpoint best.pth]
"""

import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models.paid_snownet import PAIDSnowNet


@torch.no_grad()
def time_model(model, device, H, W, num_runs=10, warmup=2):
    model.eval()
    dummy = torch.randn(1, 3, H, W, device=device)
    for _ in range(warmup):
        _ = model(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(num_runs):
        _ = model(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    return (time.time() - t0) * 1000.0 / num_runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./results/runtime')
    parser.add_argument('--resolutions', type=str, default='256,512,1024,2048',
                        help='Comma-separated edge sizes.')
    parser.add_argument('--num_runs', type=int, default=10)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    model = PAIDSnowNet().to(device)
    if args.checkpoint and os.path.exists(args.checkpoint):
        sd = torch.load(args.checkpoint, map_location=device)
        sd = sd.get('model_state_dict', sd)
        model.load_state_dict(sd, strict=False)

    sizes = [int(s) for s in args.resolutions.split(',')]
    rows = []
    for s in sizes:
        try:
            t = time_model(model, device, s, s, num_runs=args.num_runs)
        except RuntimeError as e:  # OOM
            t = None
            print(f'  {s}x{s} | OOM ({e})')
        else:
            print(f'  {s}x{s} | {t:.0f} ms')
        rows.append({'resolution': f'{s}x{s}', 'time_ms': t})

    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, 'runtime_results.json'), 'w') as f:
        json.dump({'device': str(device), 'PAID-SnowNet': rows}, f, indent=2)
    print(f"\nSaved -> {args.save_dir}/runtime_results.json")


if __name__ == '__main__':
    main()
