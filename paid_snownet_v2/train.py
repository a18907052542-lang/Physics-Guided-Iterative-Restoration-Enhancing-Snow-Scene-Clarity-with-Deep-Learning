"""
Train PAID-SnowNet on Snow100K. Hyperparameters follow Table 7 / 8 of the paper.

Usage
-----
    python train.py --data_root <Snow100K_root> --epochs 200 --batch_size 32
"""

import os
import sys
import time
import random
import logging
import argparse

import numpy as np
import torch
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from models.paid_snownet import PAIDSnowNet
from losses.multi_scale_loss import MultiScaleLoss, PhysicsConsistencyLoss
from datasets.snow_dataset import get_dataloaders
from utils.metrics import compute_psnr, compute_ssim


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'train.log')),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def train_one_epoch(model, loader, criterion, physics_loss, optimizer, device, epoch, logger):
    model.train()
    total_loss = total_psnr = 0.0
    nb = 0
    for i, (x, y) in enumerate(loader):
        x = x.to(device); y = y.to(device)
        restored, phys, inters = model(x)
        loss, ld = criterion(inters, y)
        loss = loss + 0.1 * physics_loss(restored, x, phys)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item()
        total_psnr += compute_psnr(restored, y)
        nb += 1
        if i % 50 == 0:
            logger.info(f'  ep {epoch} [{i}/{len(loader)}] loss={loss.item():.4f} '
                        f'rec={ld["reconstruction"]:.4f} '
                        f'perc={ld["perceptual"]:.4f} edge={ld["edge"]:.4f}')
    return total_loss / max(nb, 1), total_psnr / max(nb, 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    tl = tp = ts = 0.0; nb = 0
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        restored, _, inters = model(x)
        loss, _ = criterion(inters, y)
        tl += loss.item(); tp += compute_psnr(restored, y); ts += compute_ssim(restored, y); nb += 1
    nb = max(nb, 1)
    return tl / nb, tp / nb, ts / nb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    args = parser.parse_args()

    set_seed(args.seed)
    logger = setup_logging(args.log_dir)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    cfg = Config(); cfg.data_root = args.data_root; cfg.batch_size = args.batch_size
    cfg.num_epochs = args.epochs; cfg.learning_rate = args.lr

    model = PAIDSnowNet().to(device)
    logger.info(f'Parameters: {model.get_parameter_count() / 1e6:.2f}M')

    criterion = MultiScaleLoss(num_iterations=3, alpha=0.5, beta=0.1).to(device)
    physics_loss = PhysicsConsistencyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    train_loader, val_loader, _ = get_dataloaders(cfg)
    logger.info(f'Train: {len(train_loader.dataset)}  Val: {len(val_loader.dataset)}')

    start_epoch = 0; best_psnr = 0.0
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck['model_state_dict'])
        optimizer.load_state_dict(ck['optimizer_state_dict'])
        scheduler.load_state_dict(ck['scheduler_state_dict'])
        start_epoch = ck['epoch'] + 1
        best_psnr = ck.get('best_psnr', 0.0)
        logger.info(f'Resumed @ {start_epoch}, best={best_psnr:.2f}')

    for ep in range(start_epoch, args.epochs):
        t0 = time.time()
        tl, tp = train_one_epoch(model, train_loader, criterion, physics_loss,
                                 optimizer, device, ep, logger)
        vl, vp, vs = validate(model, val_loader, criterion, device)
        scheduler.step()
        logger.info(f'[ep {ep}/{args.epochs}] {time.time() - t0:.1f}s '
                    f'train: loss={tl:.4f} psnr={tp:.2f} | '
                    f'val: loss={vl:.4f} psnr={vp:.2f} ssim={vs:.4f} '
                    f'lr={scheduler.get_last_lr()[0]:.2e}')

        ck = {
            'epoch': ep,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_psnr': max(best_psnr, vp),
            'val_ssim': vs,
        }
        torch.save(ck, os.path.join(args.checkpoint_dir, 'latest.pth'))
        if vp > best_psnr:
            best_psnr = vp
            torch.save(ck, os.path.join(args.checkpoint_dir, 'best.pth'))
            logger.info(f'  -> new best {best_psnr:.2f}')

    logger.info(f'Done. Best PSNR: {best_psnr:.2f}')


if __name__ == '__main__':
    main()
