"""
Evaluate PAID-SnowNet on Snow100K and save per-image results.

Produces:
    results/test_results.json
    results/restored/restored_<idx>.png        (restored output)
    results/restored/input_<idx>.png           (degraded input)
    results/restored/gt_<idx>.png              (ground truth)
    results/physics/<idx>_scattering.png       (Fig. 3 components)
    results/physics/<idx>_transmission.png
    results/physics/<idx>_occlusion.png
"""

import os
import sys
import json
import time
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from models.paid_snownet import PAIDSnowNet
from datasets.snow_dataset import Snow100KDataset
from utils.metrics import snow_denoise_metric


@torch.no_grad()
def test(model, loader, device, save_dir=None, save_every=None,
         max_save=100):
    model.eval()
    metrics = defaultdict(list)
    total_time = 0.0
    n_imgs = 0
    saved = 0

    if save_dir:
        os.makedirs(os.path.join(save_dir, 'restored'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'physics'), exist_ok=True)

    for idx, (inp, tgt) in enumerate(loader):
        inp = inp.to(device); tgt = tgt.to(device)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        restored, physics, _ = model(inp)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        total_time += time.time() - t0

        m = snow_denoise_metric(restored, tgt)
        for k, v in m.items():
            if isinstance(v, (int, float)):
                metrics[k].append(v)
        n_imgs += inp.shape[0]

        if save_dir and saved < max_save:
            save_image(restored, os.path.join(save_dir, 'restored', f'restored_{idx:04d}.png'))
            save_image(inp, os.path.join(save_dir, 'restored', f'input_{idx:04d}.png'))
            save_image(tgt, os.path.join(save_dir, 'restored', f'gt_{idx:04d}.png'))
            save_image(physics['occlusion_mask'],
                       os.path.join(save_dir, 'physics', f'{idx:04d}_occlusion.png'))
            save_image(physics['transmission'],
                       os.path.join(save_dir, 'physics', f'{idx:04d}_transmission.png'))
            save_image(physics['scattering_coeff'],
                       os.path.join(save_dir, 'physics', f'{idx:04d}_scattering.png'))
            saved += 1

        if (idx + 1) % 100 == 0:
            print(f'  {idx+1}/{len(loader)} | PSNR {np.mean(metrics["PSNR"]):.2f}')

    avg = {k: float(np.mean(v)) for k, v in metrics.items()}
    avg['avg_inference_time_ms'] = (total_time / max(n_imgs, 1)) * 1000
    avg['total_samples'] = n_imgs
    return avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_save', type=int, default=100)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    model = PAIDSnowNet().to(device)
    if args.checkpoint and os.path.exists(args.checkpoint):
        ck = torch.load(args.checkpoint, map_location=device)
        sd = ck.get('model_state_dict', ck)
        model.load_state_dict(sd, strict=False)
        if isinstance(ck, dict) and 'best_psnr' in ck:
            print(f"Loaded {args.checkpoint} (best PSNR in ckpt: {ck['best_psnr']:.2f}dB)")
        else:
            print(f'Loaded {args.checkpoint}')

    print(f'Parameters: {model.get_parameter_count() / 1e6:.2f}M')

    dataset = Snow100KDataset(args.data_root, split='test',
                              patch_size=256, augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=4)
    print(f'Test samples: {len(dataset)}')

    avg = test(model, loader, device, save_dir=args.save_dir, max_save=args.max_save)
    print('\n----- summary -----')
    for k, v in avg.items():
        if isinstance(v, float):
            print(f'  {k}: {v:.4f}')
        else:
            print(f'  {k}: {v}')

    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, 'test_results.json'), 'w') as f:
        json.dump(avg, f, indent=2)
    print(f"\nSaved -> {args.save_dir}/test_results.json")


if __name__ == '__main__':
    main()
