"""
Table 16 — loss function component ablation.

Unlike the other ablations, this one needs (re-)training because L1/L2/+perceptual
/+edge change the supervision signal. The script runs a short fine-tuning
loop for each configuration; for full reproduction use --epochs 200.

Usage
-----
    python experiments/ablation_loss.py --data_root <Snow100K_root> \
                                        --epochs 5 --max_batches 200
"""

import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from models.paid_snownet import PAIDSnowNet
from datasets.snow_dataset import Snow100KDataset
from losses.multi_scale_loss import MultiScaleLoss, PhysicsConsistencyLoss
from utils.metrics import compute_psnr, compute_ssim, compute_lpips


def train_short(model, train_loader, criterion, physics_loss, device,
                epochs, lr, max_batches_per_epoch, use_physics):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for ep in range(epochs):
        for i, (x, y) in enumerate(train_loader):
            if i >= max_batches_per_epoch:
                break
            x = x.to(device); y = y.to(device)
            restored, phys, inters = model(x)
            loss, _ = criterion(inters, y)
            if use_physics:
                loss = loss + 0.1 * physics_loss(restored, x, phys)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()


@torch.no_grad()
def evaluate(model, dataloader, device, max_batches):
    model.eval()
    psnr = ssim = lpips_v = 0.0
    n = 0
    for i, (x, y) in enumerate(dataloader):
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
    return {'PSNR': psnr / n, 'SSIM': ssim / n, 'LPIPS': lpips_v / n}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_train_batches', type=int, default=200)
    parser.add_argument('--max_eval_batches', type=int, default=50)
    parser.add_argument('--save_dir', type=str, default='./results/ablation')
    parser.add_argument('--init_checkpoint', type=str, default=None,
                        help='Optional checkpoint to initialise each variant from.')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    train_set = Snow100KDataset(args.data_root, split='train', patch_size=256, augment=True)
    test_set = Snow100KDataset(args.data_root, split='test', patch_size=256, augment=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    configs = [
        # (label, recon, use_perceptual, use_edge, use_physics)
        ('L2 only',                          'l2', False, False, False),
        ('L1 only',                          'l1', False, False, False),
        ('L2 + Perceptual',                  'l2', True,  False, False),
        ('L2 + Edge',                        'l2', False, True,  False),
        ('L1 + Perceptual',                  'l1', True,  False, False),
        ('L1 + Perceptual + Edge',           'l1', True,  True,  False),
        ('Complete (L1 + Perc + Edge + Physics)', 'l1', True, True, True),
    ]

    os.makedirs(args.save_dir, exist_ok=True)
    results = []
    for label, recon, use_p, use_e, use_phys in configs:
        print(f'\n--- {label} ---')
        model = PAIDSnowNet().to(device)
        if args.init_checkpoint and os.path.exists(args.init_checkpoint):
            sd = torch.load(args.init_checkpoint, map_location=device)
            sd = sd.get('model_state_dict', sd)
            model.load_state_dict(sd, strict=False)

        criterion = MultiScaleLoss(num_iterations=3,
                                   recon_type=recon,
                                   use_perceptual=use_p,
                                   use_edge=use_e).to(device)
        physics_loss = PhysicsConsistencyLoss().to(device)

        train_short(model, train_loader, criterion, physics_loss, device,
                    epochs=args.epochs, lr=args.lr,
                    max_batches_per_epoch=args.max_train_batches,
                    use_physics=use_phys)
        m = evaluate(model, test_loader, device, max_batches=args.max_eval_batches)
        row = {'config': label, 'recon': recon,
               'use_perceptual': use_p, 'use_edge': use_e,
               'use_physics_consistency': use_phys, **m,
               'epochs': args.epochs}
        results.append(row)
        print(f"  PSNR {m['PSNR']:.2f} | SSIM {m['SSIM']:.3f} | LPIPS {m['LPIPS']:.3f}")

    out = {'table16_loss_ablation': results,
           'epochs_per_config': args.epochs,
           'note': 'Numbers reflect the configured training duration; use '
                   '--epochs 200 to match the paper protocol.'}
    with open(os.path.join(args.save_dir, 'ablation_loss_results.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved -> {args.save_dir}/ablation_loss_results.json')


if __name__ == '__main__':
    main()
