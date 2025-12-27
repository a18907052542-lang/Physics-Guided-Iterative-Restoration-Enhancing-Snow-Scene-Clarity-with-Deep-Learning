"""
Visualization Tools for PAID-SnowNet

Provides comprehensive visualization utilities for:
1. Physical parameter maps (scattering, transmission, occlusion, depth)
2. Attention weight visualization
3. Iterative restoration process visualization
4. Comparison with ground truth and baselines
5. Convergence curve plotting

Reference: Visual Computer Journal paper visualizations
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Optional, Tuple, Union
import os
from pathlib import Path


class PhysicsParameterVisualizer:
    """
    Visualizes physical parameters estimated by SPM module.
    
    Supports visualization of:
    - Scattering coefficient β(x,y) - Eq(10)
    - Transmission map t(x,y) - Eq(11)
    - Occlusion mask M(x,y) - Eq(12)
    - Depth map d(x,y)
    - PSF parameters σ(x,y)
    """
    
    def __init__(self, save_dir: str = './visualizations'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Custom colormaps for different parameters
        self.colormaps = {
            'scattering': self._create_scattering_cmap(),
            'transmission': 'viridis',
            'occlusion': 'gray',
            'depth': 'plasma',
            'psf': 'coolwarm'
        }
    
    def _create_scattering_cmap(self):
        """Create custom colormap for scattering visualization."""
        colors = ['#000080', '#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF0000']
        return LinearSegmentedColormap.from_list('scattering', colors, N=256)
    
    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy array for visualization."""
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return tensor
    
    def visualize_all_parameters(
        self,
        physics_params: Dict[str, torch.Tensor],
        input_image: Optional[torch.Tensor] = None,
        save_name: str = 'physics_params',
        show: bool = False
    ) -> str:
        """
        Visualize all physical parameters in a single figure.
        
        Args:
            physics_params: Dictionary containing parameter tensors
            input_image: Optional input snow image for reference
            save_name: Filename for saving
            show: Whether to display the figure
            
        Returns:
            Path to saved figure
        """
        n_params = len(physics_params)
        n_cols = min(3, n_params + (1 if input_image is not None else 0))
        n_rows = (n_params + (1 if input_image is not None else 0) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_rows * n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        idx = 0
        
        # Plot input image if provided
        if input_image is not None:
            img = self._to_numpy(input_image)
            if img.ndim == 4:
                img = img[0]
            if img.shape[0] in [1, 3]:
                img = np.transpose(img, (1, 2, 0))
            if img.shape[-1] == 1:
                img = img.squeeze(-1)
            axes[idx].imshow(np.clip(img, 0, 1))
            axes[idx].set_title('Input Snow Image', fontsize=12)
            axes[idx].axis('off')
            idx += 1
        
        # Plot each parameter
        param_titles = {
            'scattering_coefficient': 'Scattering β(x,y)',
            'transmission': 'Transmission t(x,y)',
            'occlusion_mask': 'Occlusion M(x,y)',
            'depth': 'Depth d(x,y)',
            'psf_sigma': 'PSF σ(x,y)'
        }
        
        for key, value in physics_params.items():
            if idx >= len(axes):
                break
            
            param = self._to_numpy(value)
            if param.ndim == 4:
                param = param[0]
            if param.ndim == 3:
                param = param[0] if param.shape[0] == 1 else param.mean(axis=0)
            
            cmap = self.colormaps.get(key.split('_')[0], 'viridis')
            im = axes[idx].imshow(param, cmap=cmap)
            axes[idx].set_title(param_titles.get(key, key), fontsize=12)
            axes[idx].axis('off')
            plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
            idx += 1
        
        # Hide unused axes
        for i in range(idx, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        save_path = self.save_dir / f'{save_name}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        plt.close()
        
        return str(save_path)
    
    def visualize_scattering_decomposition(
        self,
        rayleigh_component: torch.Tensor,
        mie_component: torch.Tensor,
        combined: torch.Tensor,
        save_name: str = 'scattering_decomposition',
        show: bool = False
    ) -> str:
        """
        Visualize Rayleigh vs Mie scattering decomposition.
        
        Reference: Eq(1-7) - Scattering physics
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        components = [
            (rayleigh_component, 'Rayleigh Scattering\n(d << λ)', 'Blues'),
            (mie_component, 'Mie Scattering\n(d ≈ λ)', 'Reds'),
            (combined, 'Combined Scattering\nβ(x,y)', self._create_scattering_cmap())
        ]
        
        for ax, (comp, title, cmap) in zip(axes, components):
            data = self._to_numpy(comp)
            if data.ndim == 4:
                data = data[0, 0]
            elif data.ndim == 3:
                data = data[0]
            
            im = ax.imshow(data, cmap=cmap)
            ax.set_title(title, fontsize=12)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.suptitle('Scattering Decomposition Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.save_dir / f'{save_name}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        plt.close()
        
        return str(save_path)


class AttentionVisualizer:
    """
    Visualizes attention weights from multi-scale fusion.
    
    Reference: Eq(8-9) - Adaptive fusion mechanism
    """
    
    def __init__(self, save_dir: str = './visualizations'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return tensor
    
    def visualize_fusion_weights(
        self,
        weights: torch.Tensor,
        scale_names: List[str] = ['3×3', '5×5', '7×7'],
        save_name: str = 'fusion_weights',
        show: bool = False
    ) -> str:
        """
        Visualize channel attention weights for multi-scale fusion.
        
        Args:
            weights: Tensor of shape [B, num_scales] or [num_scales]
            scale_names: Names for each scale branch
            save_name: Filename for saving
            show: Whether to display
            
        Returns:
            Path to saved figure
        """
        weights_np = self._to_numpy(weights)
        if weights_np.ndim == 2:
            weights_np = weights_np[0]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(scale_names)))
        bars = ax.bar(scale_names, weights_np, color=colors, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, weight in zip(bars, weights_np):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{weight:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_ylabel('Attention Weight', fontsize=12)
        ax.set_xlabel('Scale Branch', fontsize=12)
        ax.set_title('Multi-Scale Fusion Weights (Eq. 8-9)', fontsize=14, fontweight='bold')
        ax.set_ylim(0, max(weights_np) * 1.2)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        save_path = self.save_dir / f'{save_name}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        plt.close()
        
        return str(save_path)
    
    def visualize_spatial_attention(
        self,
        attention_map: torch.Tensor,
        input_image: Optional[torch.Tensor] = None,
        save_name: str = 'spatial_attention',
        show: bool = False
    ) -> str:
        """
        Visualize spatial attention map overlaid on input image.
        """
        fig, axes = plt.subplots(1, 3 if input_image is not None else 1, 
                                 figsize=(15 if input_image is not None else 5, 4))
        
        if input_image is None:
            axes = [axes]
        
        attn = self._to_numpy(attention_map)
        if attn.ndim == 4:
            attn = attn[0, 0]
        elif attn.ndim == 3:
            attn = attn[0]
        
        if input_image is not None:
            img = self._to_numpy(input_image)
            if img.ndim == 4:
                img = img[0]
            if img.shape[0] in [1, 3]:
                img = np.transpose(img, (1, 2, 0))
            if img.shape[-1] == 1:
                img = img.squeeze(-1)
            
            axes[0].imshow(np.clip(img, 0, 1))
            axes[0].set_title('Input Image', fontsize=12)
            axes[0].axis('off')
            
            axes[1].imshow(attn, cmap='hot')
            axes[1].set_title('Attention Map', fontsize=12)
            axes[1].axis('off')
            
            # Overlay
            axes[2].imshow(np.clip(img, 0, 1))
            attn_resized = np.array(
                F.interpolate(
                    torch.from_numpy(attn).unsqueeze(0).unsqueeze(0),
                    size=img.shape[:2],
                    mode='bilinear',
                    align_corners=False
                ).squeeze().numpy()
            ) if attn.shape != img.shape[:2] else attn
            axes[2].imshow(attn_resized, cmap='hot', alpha=0.5)
            axes[2].set_title('Attention Overlay', fontsize=12)
            axes[2].axis('off')
        else:
            axes[0].imshow(attn, cmap='hot')
            axes[0].set_title('Spatial Attention Map', fontsize=12)
            axes[0].axis('off')
        
        plt.tight_layout()
        save_path = self.save_dir / f'{save_name}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        plt.close()
        
        return str(save_path)


class IterationVisualizer:
    """
    Visualizes the iterative restoration process of SIRM module.
    
    Reference: Algorithm 1, Eq(24-26) - Iterative restoration
    """
    
    def __init__(self, save_dir: str = './visualizations'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return tensor
    
    def _prepare_image(self, img: np.ndarray) -> np.ndarray:
        """Prepare image for display."""
        if img.ndim == 4:
            img = img[0]
        if img.shape[0] in [1, 3]:
            img = np.transpose(img, (1, 2, 0))
        if img.shape[-1] == 1:
            img = img.squeeze(-1)
        return np.clip(img, 0, 1)
    
    def visualize_iteration_progress(
        self,
        iterations: List[torch.Tensor],
        ground_truth: Optional[torch.Tensor] = None,
        psnr_values: Optional[List[float]] = None,
        save_name: str = 'iteration_progress',
        show: bool = False
    ) -> str:
        """
        Visualize the progression through iterations.
        
        Args:
            iterations: List of restored images at each iteration [X^(0), X^(1), ..., X^(T)]
            ground_truth: Optional ground truth for comparison
            psnr_values: Optional PSNR at each iteration
            save_name: Filename for saving
            show: Whether to display
            
        Returns:
            Path to saved figure
        """
        n_iter = len(iterations)
        n_cols = min(4, n_iter + (1 if ground_truth is not None else 0))
        n_rows = (n_iter + (1 if ground_truth is not None else 0) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        if n_rows * n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, img_tensor in enumerate(iterations):
            img = self._prepare_image(self._to_numpy(img_tensor))
            axes[idx].imshow(img)
            
            title = f'Iteration {idx}'
            if idx == 0:
                title = 'Input (t=0)'
            elif idx == len(iterations) - 1:
                title = f'Final (t={idx})'
            
            if psnr_values is not None and idx < len(psnr_values):
                title += f'\nPSNR: {psnr_values[idx]:.2f}dB'
            
            axes[idx].set_title(title, fontsize=11)
            axes[idx].axis('off')
        
        if ground_truth is not None:
            gt_img = self._prepare_image(self._to_numpy(ground_truth))
            axes[n_iter].imshow(gt_img)
            axes[n_iter].set_title('Ground Truth', fontsize=11)
            axes[n_iter].axis('off')
        
        # Hide unused axes
        for i in range(n_iter + (1 if ground_truth is not None else 0), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('SIRM Iterative Restoration Progress (Eq. 24-26)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.save_dir / f'{save_name}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        plt.close()
        
        return str(save_path)
    
    def visualize_gradient_and_update(
        self,
        current_estimate: torch.Tensor,
        gradient: torch.Tensor,
        step_size: torch.Tensor,
        next_estimate: torch.Tensor,
        iteration: int,
        save_name: str = 'gradient_update',
        show: bool = False
    ) -> str:
        """
        Visualize gradient descent step: X^(t+1) = X^(t) - η^(t) * ∇L
        
        Reference: Eq(21-24)
        """
        fig = plt.figure(figsize=(16, 4))
        gs = gridspec.GridSpec(1, 5, width_ratios=[1, 0.3, 1, 0.3, 1])
        
        # Current estimate
        ax0 = fig.add_subplot(gs[0])
        img = self._prepare_image(self._to_numpy(current_estimate))
        ax0.imshow(img)
        ax0.set_title(f'X^({iteration})\nCurrent Estimate', fontsize=11)
        ax0.axis('off')
        
        # Minus sign
        ax1 = fig.add_subplot(gs[1])
        ax1.text(0.5, 0.5, '−', fontsize=40, ha='center', va='center')
        ax1.axis('off')
        
        # Gradient (scaled for visibility)
        ax2 = fig.add_subplot(gs[2])
        grad = self._to_numpy(gradient)
        if grad.ndim == 4:
            grad = grad[0]
        if grad.shape[0] in [1, 3]:
            grad = np.transpose(grad, (1, 2, 0))
        if grad.shape[-1] == 1:
            grad = grad.squeeze(-1)
        grad_vis = (grad - grad.min()) / (grad.max() - grad.min() + 1e-8)
        ax2.imshow(grad_vis if grad_vis.ndim == 2 else grad_vis)
        
        # Get step size value
        eta = self._to_numpy(step_size)
        if isinstance(eta, np.ndarray):
            eta = eta.mean()
        ax2.set_title(f'η^({iteration}) · ∇L\nStep size: {eta:.4f}', fontsize=11)
        ax2.axis('off')
        
        # Equals sign
        ax3 = fig.add_subplot(gs[3])
        ax3.text(0.5, 0.5, '=', fontsize=40, ha='center', va='center')
        ax3.axis('off')
        
        # Next estimate
        ax4 = fig.add_subplot(gs[4])
        next_img = self._prepare_image(self._to_numpy(next_estimate))
        ax4.imshow(next_img)
        ax4.set_title(f'X^({iteration+1})\nUpdated Estimate', fontsize=11)
        ax4.axis('off')
        
        plt.suptitle(f'Gradient Descent Update (Iteration {iteration})', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.save_dir / f'{save_name}_iter{iteration}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        plt.close()
        
        return str(save_path)


class ComparisonVisualizer:
    """
    Visualizes comparisons between methods and with ground truth.
    
    Supports:
    - Side-by-side comparisons
    - Difference maps
    - Zoomed regions
    - Metric annotations
    """
    
    def __init__(self, save_dir: str = './visualizations'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return tensor
    
    def _prepare_image(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 4:
            img = img[0]
        if img.shape[0] in [1, 3]:
            img = np.transpose(img, (1, 2, 0))
        if img.shape[-1] == 1:
            img = img.squeeze(-1)
        return np.clip(img, 0, 1)
    
    def compare_methods(
        self,
        input_image: torch.Tensor,
        results: Dict[str, torch.Tensor],
        ground_truth: Optional[torch.Tensor] = None,
        metrics: Optional[Dict[str, Dict[str, float]]] = None,
        save_name: str = 'method_comparison',
        show: bool = False
    ) -> str:
        """
        Compare results from different methods.
        
        Args:
            input_image: Input snow image
            results: Dictionary mapping method names to result tensors
            ground_truth: Optional ground truth image
            metrics: Optional dictionary of metrics per method
            save_name: Filename for saving
            show: Whether to display
            
        Returns:
            Path to saved figure
        """
        n_methods = len(results) + 1 + (1 if ground_truth is not None else 0)
        n_cols = min(4, n_methods)
        n_rows = (n_methods + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4.5*n_rows))
        if n_rows * n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        idx = 0
        
        # Input image
        input_img = self._prepare_image(self._to_numpy(input_image))
        axes[idx].imshow(input_img)
        axes[idx].set_title('Input (Snow)', fontsize=11, fontweight='bold')
        axes[idx].axis('off')
        idx += 1
        
        # Method results
        for method_name, result in results.items():
            result_img = self._prepare_image(self._to_numpy(result))
            axes[idx].imshow(result_img)
            
            title = method_name
            if metrics is not None and method_name in metrics:
                m = metrics[method_name]
                title += f"\nPSNR: {m.get('psnr', 0):.2f}dB"
                if 'ssim' in m:
                    title += f" | SSIM: {m['ssim']:.4f}"
            
            axes[idx].set_title(title, fontsize=10)
            axes[idx].axis('off')
            idx += 1
        
        # Ground truth
        if ground_truth is not None:
            gt_img = self._prepare_image(self._to_numpy(ground_truth))
            axes[idx].imshow(gt_img)
            axes[idx].set_title('Ground Truth', fontsize=11, fontweight='bold')
            axes[idx].axis('off')
            idx += 1
        
        # Hide unused axes
        for i in range(idx, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('Snow Removal Method Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.save_dir / f'{save_name}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        plt.close()
        
        return str(save_path)
    
    def visualize_difference_map(
        self,
        result: torch.Tensor,
        ground_truth: torch.Tensor,
        method_name: str = 'PAID-SnowNet',
        save_name: str = 'difference_map',
        show: bool = False
    ) -> str:
        """
        Visualize pixel-wise difference between result and ground truth.
        """
        result_np = self._prepare_image(self._to_numpy(result))
        gt_np = self._prepare_image(self._to_numpy(ground_truth))
        
        # Compute difference
        diff = np.abs(result_np - gt_np)
        if diff.ndim == 3:
            diff_gray = diff.mean(axis=2)
        else:
            diff_gray = diff
        
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        axes[0].imshow(result_np)
        axes[0].set_title(f'{method_name} Result', fontsize=11)
        axes[0].axis('off')
        
        axes[1].imshow(gt_np)
        axes[1].set_title('Ground Truth', fontsize=11)
        axes[1].axis('off')
        
        im = axes[2].imshow(diff_gray, cmap='hot', vmin=0, vmax=0.5)
        axes[2].set_title('Absolute Difference', fontsize=11)
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        
        # Error histogram
        axes[3].hist(diff_gray.flatten(), bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        axes[3].set_xlabel('Pixel Error', fontsize=10)
        axes[3].set_ylabel('Frequency', fontsize=10)
        axes[3].set_title('Error Distribution', fontsize=11)
        axes[3].axvline(diff_gray.mean(), color='red', linestyle='--', 
                       label=f'Mean: {diff_gray.mean():.4f}')
        axes[3].legend()
        
        plt.suptitle(f'Difference Analysis: {method_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.save_dir / f'{save_name}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        plt.close()
        
        return str(save_path)
    
    def visualize_zoomed_regions(
        self,
        images: Dict[str, torch.Tensor],
        regions: List[Tuple[int, int, int, int]],
        save_name: str = 'zoomed_comparison',
        show: bool = False
    ) -> str:
        """
        Visualize zoomed-in regions for detail comparison.
        
        Args:
            images: Dictionary of images to compare
            regions: List of (x, y, width, height) tuples defining zoom regions
            save_name: Filename for saving
            show: Whether to display
            
        Returns:
            Path to saved figure
        """
        n_images = len(images)
        n_regions = len(regions)
        
        fig, axes = plt.subplots(n_regions + 1, n_images, 
                                 figsize=(3*n_images, 3*(n_regions + 1)))
        
        if n_images == 1:
            axes = axes.reshape(-1, 1)
        if n_regions + 1 == 1:
            axes = axes.reshape(1, -1)
        
        # Full images in first row
        for col, (name, img_tensor) in enumerate(images.items()):
            img = self._prepare_image(self._to_numpy(img_tensor))
            axes[0, col].imshow(img)
            axes[0, col].set_title(name, fontsize=10)
            axes[0, col].axis('off')
            
            # Draw region boxes
            for i, (x, y, w, h) in enumerate(regions):
                rect = plt.Rectangle((x, y), w, h, fill=False, 
                                     edgecolor=plt.cm.tab10(i), linewidth=2)
                axes[0, col].add_patch(rect)
        
        # Zoomed regions in subsequent rows
        for row, (x, y, w, h) in enumerate(regions, start=1):
            for col, (name, img_tensor) in enumerate(images.items()):
                img = self._prepare_image(self._to_numpy(img_tensor))
                
                # Extract region (handle bounds)
                y_end = min(y + h, img.shape[0])
                x_end = min(x + w, img.shape[1])
                region = img[y:y_end, x:x_end]
                
                axes[row, col].imshow(region)
                if col == 0:
                    axes[row, col].set_ylabel(f'Region {row}', fontsize=10)
                axes[row, col].axis('off')
        
        plt.suptitle('Zoomed Region Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.save_dir / f'{save_name}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        plt.close()
        
        return str(save_path)


class ConvergencePlotter:
    """
    Plots convergence curves and training statistics.
    
    Reference: Theorem 1, Eq(27) - Convergence analysis
    """
    
    def __init__(self, save_dir: str = './visualizations'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_training_curves(
        self,
        train_losses: List[float],
        val_losses: Optional[List[float]] = None,
        val_psnrs: Optional[List[float]] = None,
        val_ssims: Optional[List[float]] = None,
        save_name: str = 'training_curves',
        show: bool = False
    ) -> str:
        """
        Plot training and validation curves.
        """
        n_plots = 1 + (1 if val_psnrs is not None else 0) + (1 if val_ssims is not None else 0)
        
        fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 4))
        if n_plots == 1:
            axes = [axes]
        
        epochs = range(1, len(train_losses) + 1)
        
        # Loss curves
        axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
        if val_losses is not None:
            axes[0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=11)
        axes[0].set_ylabel('Loss', fontsize=11)
        axes[0].set_title('Training Progress', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        axes[0].set_yscale('log')
        
        plot_idx = 1
        
        # PSNR curve
        if val_psnrs is not None:
            axes[plot_idx].plot(epochs, val_psnrs, 'g-', linewidth=2)
            axes[plot_idx].set_xlabel('Epoch', fontsize=11)
            axes[plot_idx].set_ylabel('PSNR (dB)', fontsize=11)
            axes[plot_idx].set_title('Validation PSNR', fontsize=12, fontweight='bold')
            axes[plot_idx].grid(alpha=0.3)
            
            # Mark best
            best_idx = np.argmax(val_psnrs)
            axes[plot_idx].scatter([best_idx + 1], [val_psnrs[best_idx]], 
                                  color='red', s=100, zorder=5, marker='*')
            axes[plot_idx].annotate(f'Best: {val_psnrs[best_idx]:.2f}dB',
                                   (best_idx + 1, val_psnrs[best_idx]),
                                   textcoords="offset points", xytext=(10, 10),
                                   fontsize=10)
            plot_idx += 1
        
        # SSIM curve
        if val_ssims is not None:
            axes[plot_idx].plot(epochs, val_ssims, 'm-', linewidth=2)
            axes[plot_idx].set_xlabel('Epoch', fontsize=11)
            axes[plot_idx].set_ylabel('SSIM', fontsize=11)
            axes[plot_idx].set_title('Validation SSIM', fontsize=12, fontweight='bold')
            axes[plot_idx].grid(alpha=0.3)
        
        plt.tight_layout()
        
        save_path = self.save_dir / f'{save_name}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        plt.close()
        
        return str(save_path)
    
    def plot_iteration_convergence(
        self,
        iteration_metrics: Dict[str, List[float]],
        theoretical_bound: Optional[List[float]] = None,
        save_name: str = 'iteration_convergence',
        show: bool = False
    ) -> str:
        """
        Plot convergence across SIRM iterations.
        
        Reference: Theorem 1 - Convergence guarantee
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        iterations = range(len(iteration_metrics.get('loss', [])))
        
        # Loss convergence
        if 'loss' in iteration_metrics:
            axes[0].plot(iterations, iteration_metrics['loss'], 'b-o', 
                        label='Empirical Loss', linewidth=2, markersize=8)
        if theoretical_bound is not None:
            axes[0].plot(iterations, theoretical_bound, 'r--', 
                        label='Theoretical Bound (Eq. 27)', linewidth=2)
        axes[0].set_xlabel('Iteration t', fontsize=11)
        axes[0].set_ylabel('Loss L(X^(t))', fontsize=11)
        axes[0].set_title('Loss Convergence', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Gradient norm
        if 'gradient_norm' in iteration_metrics:
            axes[1].plot(iterations, iteration_metrics['gradient_norm'], 
                        'g-s', linewidth=2, markersize=8)
            axes[1].set_xlabel('Iteration t', fontsize=11)
            axes[1].set_ylabel('||∇L(X^(t))||²', fontsize=11)
            axes[1].set_title('Gradient Norm Convergence', fontsize=12, fontweight='bold')
            axes[1].grid(alpha=0.3)
            axes[1].set_yscale('log')
        
        plt.suptitle('SIRM Convergence Analysis (Theorem 1)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.save_dir / f'{save_name}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        plt.close()
        
        return str(save_path)
    
    def plot_ablation_results(
        self,
        ablation_data: Dict[str, Dict[str, float]],
        metric: str = 'psnr',
        save_name: str = 'ablation_study',
        show: bool = False
    ) -> str:
        """
        Plot ablation study results.
        
        Reference: Tables 3-6 in paper
        """
        configs = list(ablation_data.keys())
        values = [ablation_data[c].get(metric, 0) for c in configs]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(configs)))
        bars = ax.bar(range(len(configs)), values, color=colors, edgecolor='black')
        
        # Value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(configs, rotation=45, ha='right', fontsize=10)
        ax.set_ylabel(f'{metric.upper()} (dB)' if metric == 'psnr' else metric.upper(), fontsize=11)
        ax.set_title(f'Ablation Study: {metric.upper()}', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Highlight best
        best_idx = np.argmax(values)
        bars[best_idx].set_edgecolor('red')
        bars[best_idx].set_linewidth(3)
        
        plt.tight_layout()
        
        save_path = self.save_dir / f'{save_name}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        plt.close()
        
        return str(save_path)


def create_visualization_suite(save_dir: str = './visualizations') -> Dict:
    """
    Create a complete visualization suite.
    
    Returns:
        Dictionary containing all visualizer instances
    """
    return {
        'physics': PhysicsParameterVisualizer(save_dir),
        'attention': AttentionVisualizer(save_dir),
        'iteration': IterationVisualizer(save_dir),
        'comparison': ComparisonVisualizer(save_dir),
        'convergence': ConvergencePlotter(save_dir)
    }


if __name__ == '__main__':
    # Demo visualization with synthetic data
    print("PAID-SnowNet Visualization Tools Demo")
    print("=" * 50)
    
    # Create visualizers
    vis_suite = create_visualization_suite('./demo_visualizations')
    
    # Generate synthetic data
    H, W = 128, 128
    
    # Synthetic physics parameters
    physics_params = {
        'scattering_coefficient': torch.rand(1, 1, H, W) * 0.5,
        'transmission': torch.sigmoid(torch.randn(1, 1, H, W)),
        'occlusion_mask': torch.sigmoid(torch.randn(1, 1, H, W) * 2),
        'depth': torch.rand(1, 1, H, W)
    }
    
    # Synthetic input image
    input_image = torch.rand(1, 3, H, W) * 0.7 + 0.1
    
    # Visualize physics parameters
    path = vis_suite['physics'].visualize_all_parameters(
        physics_params, input_image, 'demo_physics'
    )
    print(f"Physics parameters visualization saved to: {path}")
    
    # Synthetic attention weights
    weights = torch.softmax(torch.randn(3), dim=0)
    path = vis_suite['attention'].visualize_fusion_weights(
        weights, save_name='demo_attention'
    )
    print(f"Attention weights visualization saved to: {path}")
    
    # Synthetic iteration progress
    iterations = [torch.rand(1, 3, H, W) * (0.5 + 0.1*i) for i in range(4)]
    psnr_values = [25.0, 28.5, 30.2, 30.8]
    path = vis_suite['iteration'].visualize_iteration_progress(
        iterations, ground_truth=torch.rand(1, 3, H, W),
        psnr_values=psnr_values, save_name='demo_iterations'
    )
    print(f"Iteration progress visualization saved to: {path}")
    
    # Synthetic training curves
    train_losses = [1.0 * (0.95 ** i) for i in range(100)]
    val_psnrs = [25 + 5 * (1 - 0.98**i) for i in range(100)]
    path = vis_suite['convergence'].plot_training_curves(
        train_losses, val_psnrs=val_psnrs, save_name='demo_training'
    )
    print(f"Training curves saved to: {path}")
    
    print("\nVisualization demo complete!")
