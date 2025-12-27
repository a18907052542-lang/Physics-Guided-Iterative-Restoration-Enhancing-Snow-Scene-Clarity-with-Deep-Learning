"""
Convergence Analysis for PAID-SnowNet
Implements theoretical analysis from Section 4 of the paper

This module provides tools for analyzing convergence properties:
- Theorem 1 verification: O(1/√T) convergence rate
- Iteration effect analysis (Table 4)
- Gradient norm monitoring
- L-smoothness verification
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt


class ConvergenceAnalyzer:
    """
    Analyzer for tracking and visualizing convergence behavior.
    
    Monitors:
    - Loss trajectory
    - Gradient norms
    - Relative change between iterations
    - PSNR/SSIM progression
    """
    
    def __init__(self, convergence_threshold: float = 1e-4):
        """
        Args:
            convergence_threshold: δ for convergence criterion
                ||X^{t+1} - X^{t}||_2 / ||X^{t}||_2 < δ
        """
        self.threshold = convergence_threshold
        self.reset()
        
    def reset(self):
        """Reset tracking history."""
        self.loss_history = []
        self.gradient_norms = []
        self.relative_changes = []
        self.psnr_history = []
        self.ssim_history = []
        
    def update(self, loss: float, gradient_norm: float,
               current: torch.Tensor, previous: torch.Tensor,
               psnr: Optional[float] = None, ssim: Optional[float] = None):
        """
        Update tracking with new iteration results.
        
        Args:
            loss: Current loss value
            gradient_norm: Current gradient L2 norm
            current: Current estimate X^{t+1}
            previous: Previous estimate X^{t}
            psnr: Optional PSNR value
            ssim: Optional SSIM value
        """
        self.loss_history.append(loss)
        self.gradient_norms.append(gradient_norm)
        
        # Compute relative change
        if previous is not None:
            rel_change = (current - previous).norm() / (previous.norm() + 1e-8)
            self.relative_changes.append(rel_change.item())
        
        if psnr is not None:
            self.psnr_history.append(psnr)
        if ssim is not None:
            self.ssim_history.append(ssim)
            
    def check_convergence(self) -> bool:
        """
        Check if convergence criterion is met.
        
        Returns:
            True if converged (relative change < threshold)
        """
        if len(self.relative_changes) == 0:
            return False
        return self.relative_changes[-1] < self.threshold
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Get convergence statistics.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'num_iterations': len(self.loss_history),
            'final_loss': self.loss_history[-1] if self.loss_history else 0,
            'loss_reduction': (self.loss_history[0] - self.loss_history[-1]) / \
                             (self.loss_history[0] + 1e-8) if self.loss_history else 0,
            'final_gradient_norm': self.gradient_norms[-1] if self.gradient_norms else 0,
            'final_relative_change': self.relative_changes[-1] if self.relative_changes else 0,
            'converged': self.check_convergence()
        }
        
        if self.psnr_history:
            stats['final_psnr'] = self.psnr_history[-1]
            stats['psnr_improvement'] = self.psnr_history[-1] - self.psnr_history[0]
            
        if self.ssim_history:
            stats['final_ssim'] = self.ssim_history[-1]
            
        return stats
    
    def plot_convergence(self, save_path: Optional[str] = None):
        """
        Plot convergence curves.
        
        Args:
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss curve
        axes[0, 0].plot(self.loss_history, 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss Convergence')
        axes[0, 0].grid(True)
        
        # Gradient norm
        axes[0, 1].semilogy(self.gradient_norms, 'r-', linewidth=2)
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Gradient Norm (log)')
        axes[0, 1].set_title('Gradient Norm Decay')
        axes[0, 1].grid(True)
        
        # Relative change
        if self.relative_changes:
            axes[1, 0].semilogy(self.relative_changes, 'g-', linewidth=2)
            axes[1, 0].axhline(y=self.threshold, color='k', linestyle='--',
                              label=f'Threshold ({self.threshold})')
            axes[1, 0].set_xlabel('Iteration')
            axes[1, 0].set_ylabel('Relative Change (log)')
            axes[1, 0].set_title('Convergence Criterion')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # PSNR progression
        if self.psnr_history:
            axes[1, 1].plot(self.psnr_history, 'm-', linewidth=2)
            axes[1, 1].set_xlabel('Iteration')
            axes[1, 1].set_ylabel('PSNR (dB)')
            axes[1, 1].set_title('Quality Improvement')
            axes[1, 1].grid(True)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
            
        plt.close()


class TheoreticalAnalysis:
    """
    Theoretical convergence analysis tools.
    
    Verifies conditions for Theorem 1:
    - L-smoothness of loss function
    - Boundedness from below
    - Non-expansive property of U-Net
    """
    
    @staticmethod
    def estimate_lipschitz_constant(model: nn.Module,
                                    input_tensor: torch.Tensor,
                                    num_samples: int = 100,
                                    epsilon: float = 1e-3) -> float:
        """
        Estimate Lipschitz constant of gradient (L-smoothness parameter).
        
        For L-smooth functions:
        ||∇f(x) - ∇f(y)|| ≤ L||x - y||
        
        Args:
            model: Neural network model
            input_tensor: Reference input
            num_samples: Number of random samples
            epsilon: Perturbation magnitude
            
        Returns:
            Estimated Lipschitz constant L
        """
        model.eval()
        lipschitz_estimates = []
        
        x = input_tensor.clone().requires_grad_(True)
        
        # Get base gradient
        output = model(x)
        if isinstance(output, dict):
            output = output['output']
        loss_base = output.mean()
        loss_base.backward()
        grad_base = x.grad.clone()
        
        for _ in range(num_samples):
            # Random perturbation
            delta = epsilon * torch.randn_like(x)
            x_perturbed = x.detach() + delta
            x_perturbed.requires_grad_(True)
            
            # Compute perturbed gradient
            output_p = model(x_perturbed)
            if isinstance(output_p, dict):
                output_p = output_p['output']
            loss_p = output_p.mean()
            loss_p.backward()
            grad_p = x_perturbed.grad.clone()
            
            # Estimate L
            grad_diff = (grad_p - grad_base).norm()
            input_diff = delta.norm()
            
            if input_diff > 1e-10:
                L_estimate = grad_diff / input_diff
                lipschitz_estimates.append(L_estimate.item())
                
        return np.percentile(lipschitz_estimates, 95)  # Conservative estimate
    
    @staticmethod
    def verify_non_expansive(unet: nn.Module,
                            input1: torch.Tensor,
                            input2: torch.Tensor) -> Tuple[bool, float]:
        """
        Verify non-expansive property of U-Net.
        
        Non-expansive: ||U(x) - U(y)|| ≤ ||x - y||
        
        Args:
            unet: U-Net module
            input1: First input
            input2: Second input
            
        Returns:
            Tuple of (is_non_expansive, expansion_factor)
        """
        unet.eval()
        with torch.no_grad():
            out1 = unet(input1)
            out2 = unet(input2)
            
            input_dist = (input1 - input2).norm()
            output_dist = (out1 - out2).norm()
            
            expansion = output_dist / (input_dist + 1e-8)
            
        return expansion.item() <= 1.0 + 1e-6, expansion.item()
    
    @staticmethod
    def theoretical_convergence_rate(L: float, f_0: float, f_star: float,
                                    T: int) -> float:
        """
        Compute theoretical convergence bound from Theorem 1.
        
        Eq(27): min_t ||∇L(X^{t})||² ≤ C(L(X^{0}) - L*) / √T
        
        Args:
            L: Lipschitz constant
            f_0: Initial loss value
            f_star: Optimal loss value
            T: Number of iterations
            
        Returns:
            Upper bound on minimum gradient norm squared
        """
        C = 2 * L  # Constant from analysis
        return C * (f_0 - f_star) / np.sqrt(T)


def run_convergence_experiment(model: nn.Module,
                              dataloader,
                              num_iterations: int = 10,
                              device: str = 'cuda') -> Dict:
    """
    Run convergence experiment on dataset.
    
    Args:
        model: PAID-SnowNet model
        dataloader: Data loader with (degraded, clean) pairs
        num_iterations: Maximum iterations to test
        device: Computation device
        
    Returns:
        Dictionary with experiment results
    """
    model = model.to(device)
    model.eval()
    
    results = {
        'iterations': list(range(1, num_iterations + 1)),
        'avg_psnr': [],
        'avg_ssim': [],
        'avg_time': [],
        'convergence_rate': []
    }
    
    from time import time
    
    for T in range(1, num_iterations + 1):
        # Modify model iterations
        if hasattr(model, 'sirm'):
            model.sirm.num_iterations = T
        
        psnr_list = []
        time_list = []
        
        with torch.no_grad():
            for degraded, clean in dataloader:
                degraded = degraded.to(device)
                clean = clean.to(device)
                
                start = time()
                output = model(degraded)
                elapsed = time() - start
                
                if isinstance(output, dict):
                    output = output['output']
                
                # Compute PSNR
                mse = ((output - clean) ** 2).mean()
                psnr = 10 * np.log10(1.0 / (mse.item() + 1e-8))
                
                psnr_list.append(psnr)
                time_list.append(elapsed)
        
        results['avg_psnr'].append(np.mean(psnr_list))
        results['avg_time'].append(np.mean(time_list) * 1000)  # ms
        
        # Convergence rate (PSNR improvement per iteration)
        if T > 1:
            rate = (results['avg_psnr'][-1] - results['avg_psnr'][-2])
            results['convergence_rate'].append(rate)
        else:
            results['convergence_rate'].append(0)
    
    return results


def analyze_iteration_effect(results: Dict, save_path: Optional[str] = None):
    """
    Analyze and visualize iteration effect (reproduces Table 4).
    
    Args:
        results: Results from run_convergence_experiment
        save_path: Optional path to save plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    iterations = results['iterations']
    
    # PSNR vs Iterations
    axes[0].plot(iterations, results['avg_psnr'], 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Number of Iterations (T)')
    axes[0].set_ylabel('PSNR (dB)')
    axes[0].set_title('Quality vs Iterations')
    axes[0].grid(True)
    
    # Highlight optimal point (T=3 from paper)
    if 3 in iterations:
        idx = iterations.index(3)
        axes[0].axvline(x=3, color='r', linestyle='--', alpha=0.5)
        axes[0].annotate(f'T=3: {results["avg_psnr"][idx]:.2f}dB',
                        xy=(3, results['avg_psnr'][idx]),
                        xytext=(4, results['avg_psnr'][idx] - 0.5),
                        arrowprops=dict(arrowstyle='->', color='red'))
    
    # Time vs Iterations
    axes[1].plot(iterations, results['avg_time'], 'rs-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Number of Iterations (T)')
    axes[1].set_ylabel('Inference Time (ms)')
    axes[1].set_title('Speed vs Iterations')
    axes[1].grid(True)
    
    # PSNR improvement rate
    axes[2].bar(iterations[1:], results['convergence_rate'][1:], color='green', alpha=0.7)
    axes[2].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    axes[2].set_xlabel('Number of Iterations (T)')
    axes[2].set_ylabel('PSNR Improvement (dB)')
    axes[2].set_title('Marginal Quality Gain')
    axes[2].grid(True, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
        
    plt.close()
    
    # Print Table 4 style summary
    print("\nIteration Effect Analysis (Table 4 reproduction):")
    print("-" * 50)
    print(f"{'T':^5} | {'PSNR (dB)':^12} | {'Time (ms)':^12} | {'Gain (dB)':^12}")
    print("-" * 50)
    for i, T in enumerate(iterations):
        gain = results['convergence_rate'][i] if i > 0 else '-'
        gain_str = f"{gain:.2f}" if isinstance(gain, float) else gain
        print(f"{T:^5} | {results['avg_psnr'][i]:^12.2f} | {results['avg_time'][i]:^12.1f} | {gain_str:^12}")
    print("-" * 50)


class ConvergenceTheorem:
    """
    Implementation of Theorem 1 convergence analysis.
    
    Theorem 1: Under assumptions:
    1. L is L-smooth
    2. L is bounded below by L*
    3. U is non-expansive
    
    Then: min_{t∈[T]} ||∇L(X^{t})||² ≤ C(L(X^{0}) - L*) / √T
    
    This implies O(1/√T) convergence to stationary point.
    """
    
    def __init__(self, L_smooth: float, L_star: float = 0):
        """
        Args:
            L_smooth: Lipschitz constant of gradient
            L_star: Lower bound on loss (usually 0)
        """
        self.L = L_smooth
        self.L_star = L_star
        self.C = 2 * L_smooth  # Constant from analysis
        
    def convergence_bound(self, L_0: float, T: int) -> float:
        """
        Compute convergence bound for T iterations.
        
        Eq(27): bound = C(L(X^{0}) - L*) / √T
        
        Args:
            L_0: Initial loss value
            T: Number of iterations
            
        Returns:
            Upper bound on min gradient norm squared
        """
        return self.C * (L_0 - self.L_star) / np.sqrt(T)
    
    def iterations_for_epsilon(self, L_0: float, epsilon: float) -> int:
        """
        Compute iterations needed for ε-accuracy.
        
        T ≥ (C(L_0 - L*) / ε)²
        
        Args:
            L_0: Initial loss
            epsilon: Target gradient norm squared
            
        Returns:
            Required number of iterations
        """
        return int(np.ceil((self.C * (L_0 - self.L_star) / epsilon) ** 2))
    
    def verify_empirical(self, gradient_norms: List[float], 
                        L_0: float) -> Dict[str, float]:
        """
        Verify empirical convergence against theoretical bound.
        
        Args:
            gradient_norms: List of gradient norms from training
            L_0: Initial loss
            
        Returns:
            Dictionary with verification results
        """
        T = len(gradient_norms)
        theoretical_bound = self.convergence_bound(L_0, T)
        empirical_min = min([g ** 2 for g in gradient_norms])
        
        return {
            'theoretical_bound': theoretical_bound,
            'empirical_minimum': empirical_min,
            'bound_satisfied': empirical_min <= theoretical_bound * 1.1,  # 10% margin
            'tightness': empirical_min / (theoretical_bound + 1e-8)
        }


# Initialize package
__all__ = [
    'ConvergenceAnalyzer',
    'TheoreticalAnalysis',
    'run_convergence_experiment',
    'analyze_iteration_effect',
    'ConvergenceTheorem'
]
