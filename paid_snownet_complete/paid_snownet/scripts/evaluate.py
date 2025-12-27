"""
PAID-SnowNet Evaluation Script
===============================

Comprehensive evaluation with multiple metrics and benchmark comparisons.
Reproduces results from paper Tables 1-2.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import Config, get_config
from data.dataset import create_dataloader, denormalize_tensor, tensor_to_image
from models.paid_snownet import create_model


class MetricsCalculator:
    """Calculate image quality metrics"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        
        # Try to load LPIPS if available
        self.lpips_fn = None
        try:
            import lpips
            self.lpips_fn = lpips.LPIPS(net='alex').to(device)
            self.lpips_fn.eval()
        except ImportError:
            print("Warning: LPIPS not available. Install with: pip install lpips")
    
    def compute_psnr(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute PSNR (Peak Signal-to-Noise Ratio)"""
        mse = torch.mean((pred - target) ** 2)
        if mse == 0:
            return float('inf')
        psnr = 10 * torch.log10(1.0 / mse)
        return psnr.item()
    
    def compute_ssim(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        window_size: int = 11,
        sigma: float = 1.5
    ) -> float:
        """Compute SSIM (Structural Similarity Index)"""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # Create Gaussian window
        gaussian = torch.exp(-torch.arange(window_size).float().sub(window_size // 2).pow(2) / (2 * sigma ** 2))
        gaussian = gaussian / gaussian.sum()
        window = gaussian.unsqueeze(1) * gaussian.unsqueeze(0)
        window = window.unsqueeze(0).unsqueeze(0).to(pred.device)
        window = window.expand(pred.size(1), 1, window_size, window_size)
        
        # Compute means
        mu_pred = F.conv2d(pred, window, padding=window_size // 2, groups=pred.size(1))
        mu_target = F.conv2d(target, window, padding=window_size // 2, groups=target.size(1))
        
        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_cross = mu_pred * mu_target
        
        # Compute variances
        sigma_pred_sq = F.conv2d(pred ** 2, window, padding=window_size // 2, groups=pred.size(1)) - mu_pred_sq
        sigma_target_sq = F.conv2d(target ** 2, window, padding=window_size // 2, groups=target.size(1)) - mu_target_sq
        sigma_cross = F.conv2d(pred * target, window, padding=window_size // 2, groups=pred.size(1)) - mu_cross
        
        # SSIM formula
        ssim_map = ((2 * mu_cross + C1) * (2 * sigma_cross + C2)) / \
                   ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))
        
        return ssim_map.mean().item()
    
    def compute_lpips(self, pred: torch.Tensor, target: torch.Tensor) -> Optional[float]:
        """Compute LPIPS (Learned Perceptual Image Patch Similarity)"""
        if self.lpips_fn is None:
            return None
        
        # LPIPS expects [-1, 1] range
        pred_scaled = pred * 2 - 1
        target_scaled = target * 2 - 1
        
        with torch.no_grad():
            lpips_value = self.lpips_fn(pred_scaled, target_scaled)
        
        return lpips_value.item()
    
    def compute_all(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, float]:
        """Compute all metrics"""
        metrics = {
            'psnr': self.compute_psnr(pred, target),
            'ssim': self.compute_ssim(pred, target)
        }
        
        lpips_val = self.compute_lpips(pred, target)
        if lpips_val is not None:
            metrics['lpips'] = lpips_val
        
        return metrics


class Evaluator:
    """PAID-SnowNet evaluator"""
    
    def __init__(
        self,
        model_path: str,
        config: Config,
        device: str = 'cuda'
    ):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self._load_model(model_path)
        
        # Initialize metrics calculator
        self.metrics = MetricsCalculator(device=str(self.device))
    
    def _load_model(self, model_path: str):
        """Load model from checkpoint"""
        print(f"Loading model from: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Determine model type from checkpoint
        if 'config' in checkpoint:
            model_type = checkpoint['config'].model.model_type
        else:
            model_type = self.config.model.model_type
        
        self.model = create_model(
            model_type=model_type,
            in_channels=3,
            out_channels=3
        ).to(self.device)
        
        # Load weights
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Handle DataParallel wrapped models
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        self.model.load_state_dict(new_state_dict)
        self.model.eval()
        
        print(f"Model loaded successfully ({model_type})")
    
    @torch.no_grad()
    def evaluate_dataset(
        self,
        data_dir: str,
        dataset_type: str = 'snow100k',
        save_results: bool = True,
        save_images: bool = False,
        output_dir: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate on a dataset
        
        Args:
            data_dir: Path to test dataset
            dataset_type: Dataset type
            save_results: Save results to JSON
            save_images: Save output images
            output_dir: Output directory for results
        
        Returns:
            Dictionary of average metrics
        """
        print(f"\nEvaluating on {data_dir}...")
        
        # Create dataloader
        dataloader = create_dataloader(
            data_dir=data_dir,
            dataset_type=dataset_type,
            mode='test',
            batch_size=1,
            patch_size=256,
            num_workers=4
        )
        
        # Setup output directory
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            if save_images:
                (output_path / 'images').mkdir(exist_ok=True)
        
        # Collect metrics
        all_metrics = []
        
        for batch in tqdm(dataloader, desc='Evaluating'):
            inputs = batch['input'].to(self.device)
            targets = batch['gt'].to(self.device)
            name = batch['name'][0]
            
            # Forward pass
            outputs = self.model(inputs)
            
            # Denormalize
            outputs = denormalize_tensor(outputs)
            targets = denormalize_tensor(targets)
            
            # Clamp to valid range
            outputs = torch.clamp(outputs, 0, 1)
            targets = torch.clamp(targets, 0, 1)
            
            # Compute metrics
            metrics = self.metrics.compute_all(outputs, targets)
            metrics['name'] = name
            all_metrics.append(metrics)
            
            # Save output image
            if save_images and output_dir:
                img = tensor_to_image(outputs)
                Image.fromarray(img).save(output_path / 'images' / f'{name}_restored.png')
        
        # Compute averages
        avg_metrics = {}
        metric_keys = [k for k in all_metrics[0].keys() if k != 'name']
        
        for key in metric_keys:
            values = [m[key] for m in all_metrics]
            avg_metrics[key] = np.mean(values)
            avg_metrics[f'{key}_std'] = np.std(values)
        
        # Print results
        print(f"\n{'='*50}")
        print(f"Results on {Path(data_dir).name}")
        print(f"{'='*50}")
        print(f"PSNR:  {avg_metrics['psnr']:.2f} ± {avg_metrics['psnr_std']:.2f} dB")
        print(f"SSIM:  {avg_metrics['ssim']:.4f} ± {avg_metrics['ssim_std']:.4f}")
        if 'lpips' in avg_metrics:
            print(f"LPIPS: {avg_metrics['lpips']:.4f} ± {avg_metrics['lpips_std']:.4f}")
        print(f"{'='*50}")
        
        # Save results
        if save_results and output_dir:
            results = {
                'dataset': str(data_dir),
                'num_samples': len(all_metrics),
                'average_metrics': avg_metrics,
                'per_image_metrics': all_metrics
            }
            
            with open(output_path / 'results.json', 'w') as f:
                json.dump(results, f, indent=2)
            
            # Save as CSV
            df = pd.DataFrame(all_metrics)
            df.to_csv(output_path / 'results.csv', index=False)
        
        return avg_metrics
    
    @torch.no_grad()
    def evaluate_single_image(
        self,
        image_path: str,
        gt_path: Optional[str] = None,
        output_path: Optional[str] = None
    ) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
        """
        Evaluate on a single image
        
        Args:
            image_path: Path to input image
            gt_path: Optional path to ground truth
            output_path: Optional path to save output
        
        Returns:
            Tuple of (output tensor, metrics dict)
        """
        from data.dataset import load_single_image
        
        # Load input
        inputs = load_single_image(image_path, normalize=True, device=str(self.device))
        
        # Forward pass
        outputs = self.model(inputs)
        
        # Denormalize
        outputs = denormalize_tensor(outputs)
        outputs = torch.clamp(outputs, 0, 1)
        
        # Compute metrics if GT available
        metrics = None
        if gt_path:
            targets = load_single_image(gt_path, normalize=False, device=str(self.device))
            metrics = self.metrics.compute_all(outputs, targets)
            print(f"PSNR: {metrics['psnr']:.2f} dB | SSIM: {metrics['ssim']:.4f}")
        
        # Save output
        if output_path:
            img = tensor_to_image(outputs)
            Image.fromarray(img).save(output_path)
            print(f"Saved: {output_path}")
        
        return outputs, metrics
    
    def benchmark_comparison(
        self,
        data_dir: str,
        methods: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        Compare with other methods
        
        Args:
            data_dir: Test data directory
            methods: Dict of method_name -> results_dir
        
        Returns:
            DataFrame with comparison results
        """
        results = []
        
        # Evaluate our method
        our_metrics = self.evaluate_dataset(data_dir, save_results=False)
        results.append({
            'Method': 'PAID-SnowNet (Ours)',
            'PSNR': our_metrics['psnr'],
            'SSIM': our_metrics['ssim'],
            'LPIPS': our_metrics.get('lpips', '-')
        })
        
        # Add comparison methods if provided
        if methods:
            for name, results_dir in methods.items():
                results_file = Path(results_dir) / 'results.json'
                if results_file.exists():
                    with open(results_file) as f:
                        method_results = json.load(f)
                    results.append({
                        'Method': name,
                        'PSNR': method_results['average_metrics']['psnr'],
                        'SSIM': method_results['average_metrics']['ssim'],
                        'LPIPS': method_results['average_metrics'].get('lpips', '-')
                    })
        
        # Create DataFrame
        df = pd.DataFrame(results)
        df = df.sort_values('PSNR', ascending=False)
        
        print("\n" + "="*60)
        print("Benchmark Comparison")
        print("="*60)
        print(df.to_string(index=False))
        print("="*60)
        
        return df


def evaluate_ablation(
    base_dir: str,
    data_dir: str,
    config_name: str = 'base'
) -> pd.DataFrame:
    """
    Evaluate ablation study checkpoints
    
    Args:
        base_dir: Directory containing ablation checkpoints
        data_dir: Test data directory
    
    Returns:
        DataFrame with ablation results
    """
    from configs.config import get_ablation_configs
    
    results = []
    ablation_configs = get_ablation_configs()
    
    for name, config in ablation_configs.items():
        checkpoint_path = Path(base_dir) / f'ablation_{name}' / 'checkpoints' / 'checkpoint_best.pth'
        
        if not checkpoint_path.exists():
            print(f"Skipping {name}: checkpoint not found")
            continue
        
        print(f"\nEvaluating ablation: {name}")
        
        evaluator = Evaluator(
            model_path=str(checkpoint_path),
            config=config
        )
        
        metrics = evaluator.evaluate_dataset(data_dir, save_results=False)
        
        results.append({
            'Configuration': name,
            'PSNR': metrics['psnr'],
            'SSIM': metrics['ssim']
        })
    
    df = pd.DataFrame(results)
    df = df.sort_values('PSNR', ascending=False)
    
    print("\n" + "="*60)
    print("Ablation Study Results")
    print("="*60)
    print(df.to_string(index=False))
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Evaluate PAID-SnowNet')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Test data directory')
    parser.add_argument('--dataset_type', type=str, default='snow100k',
                        choices=['snow100k', 'srrs', 'csd', 'generic'],
                        help='Dataset type')
    parser.add_argument('--output_dir', type=str, default='./eval_results',
                        help='Output directory for results')
    parser.add_argument('--save_images', action='store_true',
                        help='Save restored images')
    parser.add_argument('--config', type=str, default='base',
                        choices=['base', 'lightweight', 'deep'],
                        help='Model configuration')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Load config
    config = get_config(args.config)
    
    # Create evaluator
    evaluator = Evaluator(
        model_path=args.model,
        config=config,
        device=args.device
    )
    
    # Evaluate
    evaluator.evaluate_dataset(
        data_dir=args.data_dir,
        dataset_type=args.dataset_type,
        save_results=True,
        save_images=args.save_images,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
