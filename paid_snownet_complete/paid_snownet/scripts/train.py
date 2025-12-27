"""
PAID-SnowNet Training Script
=============================

Training pipeline for physics-aware snow removal network.
Supports multi-GPU training, mixed precision, and comprehensive logging.
"""

import os
import sys
import time
import argparse
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import Config, get_config
from data.dataset import create_dataloader, denormalize_tensor
from models.paid_snownet import create_model
from losses.multi_scale_loss import MultiScaleLoss
from analysis.convergence_analysis import ConvergenceAnalyzer


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_optimizer(model: nn.Module, config: Config) -> optim.Optimizer:
    """Create optimizer"""
    params = filter(lambda p: p.requires_grad, model.parameters())
    
    if config.training.optimizer == 'adamw':
        return optim.AdamW(
            params,
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
    elif config.training.optimizer == 'adam':
        return optim.Adam(
            params,
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
    elif config.training.optimizer == 'sgd':
        return optim.SGD(
            params,
            lr=config.training.learning_rate,
            momentum=0.9,
            weight_decay=config.training.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.training.optimizer}")


def get_scheduler(optimizer: optim.Optimizer, config: Config):
    """Create learning rate scheduler"""
    if config.training.scheduler == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.training.num_epochs - config.training.warmup_epochs,
            eta_min=config.training.min_lr
        )
    elif config.training.scheduler == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=50,
            gamma=0.5
        )
    elif config.training.scheduler == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=10,
            min_lr=config.training.min_lr
        )
    else:
        raise ValueError(f"Unknown scheduler: {config.training.scheduler}")


def warmup_lr(optimizer: optim.Optimizer, epoch: int, config: Config):
    """Linear warmup learning rate"""
    if epoch < config.training.warmup_epochs:
        lr = config.training.learning_rate * (epoch + 1) / config.training.warmup_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


class Trainer:
    """PAID-SnowNet trainer"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
        # Setup output directories
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.log_dir = self.output_dir / 'logs'
        self.vis_dir = self.output_dir / 'visualizations'
        
        for d in [self.output_dir, self.checkpoint_dir, self.log_dir, self.vis_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self._init_model()
        self._init_dataloaders()
        self._init_loss()
        self._init_optimizer()
        self._init_logging()
        
        # Training state
        self.start_epoch = 0
        self.best_psnr = 0.0
        self.global_step = 0
        
        # Mixed precision
        self.scaler = GradScaler() if config.training.use_amp else None
        
        # Convergence analyzer
        self.convergence_analyzer = ConvergenceAnalyzer(
            max_iterations=config.model.sirm.num_iterations
        )
    
    def _init_model(self):
        """Initialize model"""
        print(f"Creating {self.config.model.model_type} model...")
        self.model = create_model(
            model_type=self.config.model.model_type,
            in_channels=self.config.model.in_channels,
            out_channels=self.config.model.out_channels
        ).to(self.device)
        
        # Multi-GPU support
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    
    def _init_dataloaders(self):
        """Initialize data loaders"""
        print("Creating dataloaders...")
        
        self.train_loader = create_dataloader(
            data_dir=self.config.training.train_dir,
            dataset_type=self.config.training.dataset_name,
            mode='train',
            batch_size=self.config.training.batch_size,
            patch_size=self.config.training.patch_size,
            num_workers=self.config.training.num_workers
        )
        
        self.val_loader = create_dataloader(
            data_dir=self.config.training.val_dir,
            dataset_type=self.config.training.dataset_name,
            mode='val',
            batch_size=1,  # Full resolution for validation
            patch_size=self.config.training.patch_size,
            num_workers=self.config.training.num_workers
        )
        
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
    
    def _init_loss(self):
        """Initialize loss function"""
        self.criterion = MultiScaleLoss(
            use_perceptual=True,
            use_edge=True,
            perceptual_weight=self.config.model.loss.perceptual_weight,
            edge_weight=self.config.model.loss.edge_weight,
            num_iterations=self.config.model.sirm.num_iterations
        ).to(self.device)
    
    def _init_optimizer(self):
        """Initialize optimizer and scheduler"""
        self.optimizer = get_optimizer(self.model, self.config)
        self.scheduler = get_scheduler(self.optimizer, self.config)
    
    def _init_logging(self):
        """Initialize logging"""
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # Log config
        config_str = str(self.config)
        self.writer.add_text('config', config_str)
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        model_state = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_psnr': self.best_psnr,
            'global_step': self.global_step,
            'config': self.config
        }
        
        # Save periodic checkpoint
        if (epoch + 1) % self.config.training.save_freq == 0:
            path = self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
            torch.save(checkpoint, path)
            print(f"Saved checkpoint: {path}")
        
        # Save latest checkpoint
        torch.save(checkpoint, self.checkpoint_dir / 'checkpoint_latest.pth')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'checkpoint_best.pth')
            print(f"New best model! PSNR: {self.best_psnr:.2f} dB")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_psnr = checkpoint.get('best_psnr', 0.0)
        self.global_step = checkpoint.get('global_step', 0)
        
        print(f"Resumed from epoch {self.start_epoch}, best PSNR: {self.best_psnr:.2f} dB")
    
    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch"""
        self.model.train()
        
        # Warmup learning rate
        warmup_lr(self.optimizer, epoch, self.config)
        
        epoch_loss = 0.0
        epoch_losses = {'total': 0.0, 'reconstruction': 0.0, 'perceptual': 0.0, 'edge': 0.0}
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config.training.num_epochs}')
        
        for batch_idx, batch in enumerate(pbar):
            inputs = batch['input'].to(self.device)
            targets = batch['gt'].to(self.device)
            
            # Forward pass with mixed precision
            self.optimizer.zero_grad()
            
            if self.config.training.use_amp:
                with autocast():
                    outputs, intermediates = self.model(inputs, return_intermediates=True)
                    loss, loss_dict = self.criterion(
                        outputs, targets,
                        intermediates=intermediates
                    )
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.training.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.gradient_clip
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs, intermediates = self.model(inputs, return_intermediates=True)
                loss, loss_dict = self.criterion(
                    outputs, targets,
                    intermediates=intermediates
                )
                
                loss.backward()
                
                if self.config.training.gradient_clip > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.gradient_clip
                    )
                
                self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            for k, v in loss_dict.items():
                if k in epoch_losses:
                    epoch_losses[k] += v
            
            # Logging
            if self.global_step % self.config.training.log_freq == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                for k, v in loss_dict.items():
                    self.writer.add_scalar(f'train/{k}', v, self.global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
            
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        # Average losses
        num_batches = len(self.train_loader)
        epoch_loss /= num_batches
        for k in epoch_losses:
            epoch_losses[k] /= num_batches
        
        return epoch_losses
    
    @torch.no_grad()
    def validate(self, epoch: int) -> dict:
        """Validation loop"""
        self.model.eval()
        
        total_psnr = 0.0
        total_ssim = 0.0
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(tqdm(self.val_loader, desc='Validation')):
            inputs = batch['input'].to(self.device)
            targets = batch['gt'].to(self.device)
            
            outputs, _ = self.model(inputs, return_intermediates=True)
            loss, _ = self.criterion(outputs, targets)
            
            # Denormalize for metric computation
            outputs_denorm = denormalize_tensor(outputs)
            targets_denorm = denormalize_tensor(targets)
            
            # Compute PSNR
            mse = torch.mean((outputs_denorm - targets_denorm) ** 2)
            psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
            
            # Compute SSIM (simplified)
            ssim = self._compute_ssim(outputs_denorm, targets_denorm)
            
            total_psnr += psnr.item()
            total_ssim += ssim.item()
            total_loss += loss.item()
            
            # Save visualization
            if batch_idx == 0:
                self._save_visualization(inputs, outputs, targets, epoch)
        
        num_samples = len(self.val_loader)
        avg_psnr = total_psnr / num_samples
        avg_ssim = total_ssim / num_samples
        avg_loss = total_loss / num_samples
        
        # Log validation metrics
        self.writer.add_scalar('val/psnr', avg_psnr, epoch)
        self.writer.add_scalar('val/ssim', avg_ssim, epoch)
        self.writer.add_scalar('val/loss', avg_loss, epoch)
        
        return {
            'psnr': avg_psnr,
            'ssim': avg_ssim,
            'loss': avg_loss
        }
    
    def _compute_ssim(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Simplified SSIM computation"""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        mu_pred = torch.mean(pred)
        mu_target = torch.mean(target)
        
        sigma_pred = torch.var(pred)
        sigma_target = torch.var(target)
        sigma_cross = torch.mean((pred - mu_pred) * (target - mu_target))
        
        ssim = ((2 * mu_pred * mu_target + C1) * (2 * sigma_cross + C2)) / \
               ((mu_pred ** 2 + mu_target ** 2 + C1) * (sigma_pred + sigma_target + C2))
        
        return ssim
    
    def _save_visualization(
        self,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        epoch: int
    ):
        """Save visualization images"""
        from torchvision.utils import save_image, make_grid
        
        # Denormalize
        inputs = denormalize_tensor(inputs)
        outputs = denormalize_tensor(outputs)
        targets = denormalize_tensor(targets)
        
        # Clamp to valid range
        inputs = torch.clamp(inputs, 0, 1)
        outputs = torch.clamp(outputs, 0, 1)
        targets = torch.clamp(targets, 0, 1)
        
        # Create comparison grid
        comparison = torch.cat([inputs[:4], outputs[:4], targets[:4]], dim=0)
        grid = make_grid(comparison, nrow=4, padding=2)
        
        save_path = self.vis_dir / f'epoch_{epoch+1}.png'
        save_image(grid, save_path)
        
        # Log to tensorboard
        self.writer.add_image('val/comparison', grid, epoch)
    
    def train(self):
        """Main training loop"""
        print(f"\n{'='*60}")
        print(f"Starting training: {self.config.experiment_name}")
        print(f"{'='*60}\n")
        
        for epoch in range(self.start_epoch, self.config.training.num_epochs):
            epoch_start = time.time()
            
            # Train
            train_losses = self.train_epoch(epoch)
            
            # Update scheduler
            if epoch >= self.config.training.warmup_epochs:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    pass  # Updated after validation
                else:
                    self.scheduler.step()
            
            # Validate
            if (epoch + 1) % self.config.training.val_freq == 0:
                val_metrics = self.validate(epoch)
                
                # Update scheduler if plateau
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['psnr'])
                
                # Check for best model
                is_best = val_metrics['psnr'] > self.best_psnr
                if is_best:
                    self.best_psnr = val_metrics['psnr']
                
                # Save checkpoint
                self.save_checkpoint(epoch, is_best)
                
                # Print epoch summary
                epoch_time = time.time() - epoch_start
                print(f"\nEpoch {epoch+1}/{self.config.training.num_epochs}")
                print(f"  Train Loss: {train_losses['total']:.4f}")
                print(f"  Val PSNR: {val_metrics['psnr']:.2f} dB | SSIM: {val_metrics['ssim']:.4f}")
                print(f"  Best PSNR: {self.best_psnr:.2f} dB")
                print(f"  Time: {epoch_time:.1f}s")
            else:
                self.save_checkpoint(epoch, False)
        
        print(f"\n{'='*60}")
        print(f"Training complete! Best PSNR: {self.best_psnr:.2f} dB")
        print(f"{'='*60}")
        
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train PAID-SnowNet')
    parser.add_argument('--config', type=str, default='base',
                        choices=['base', 'lightweight', 'deep'],
                        help='Configuration name')
    parser.add_argument('--train_dir', type=str, default=None,
                        help='Training data directory')
    parser.add_argument('--val_dir', type=str, default=None,
                        help='Validation data directory')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Output directory')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Load config
    config = get_config(args.config)
    
    # Override with command line arguments
    if args.train_dir:
        config.training.train_dir = args.train_dir
    if args.val_dir:
        config.training.val_dir = args.val_dir
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.lr:
        config.training.learning_rate = args.lr
    
    config.seed = args.seed
    config.device = args.device
    
    # Set random seed
    set_seed(config.seed)
    
    # Create trainer
    trainer = Trainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Start training
    trainer.train()


if __name__ == '__main__':
    main()
