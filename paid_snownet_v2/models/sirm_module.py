"""
Snow Iterative Restoration Module (SIRM).

Combines an inner adaptive optimizer (default AdaGrad, Eq. 21-22) with a
completion network (default U-Net, Eq. 25) inside a T-iteration loop (Eq. 26).

Ablation flags (Tables 12, 14):
    - inner_optimizer:        'adagrad' | 'sgd' | 'sgd+momentum' | 'adam' | 'rmsprop'
    - completion_type:        'unet' | 'simple_cnn' | 'resnet'
    - use_physics_weight:     enable / disable Eq. 23 weighting
    - num_iterations:         T (set to 1 to disable iteration)
"""

import torch
import torch.nn as nn

from .degradation_ops import DegradationOperator
from .physics_weight import get_inner_optimizer
from .unet import CompletionUNet
from .simple_cnn import SimpleCNNCompletion
from .resnet_completion import ResNetCompletion


def _make_completion(completion_type, base_channels, num_layers):
    if completion_type == 'unet':
        return CompletionUNet(in_channels=3, out_channels=3,
                              base_channels=base_channels, num_layers=num_layers)
    if completion_type == 'simple_cnn':
        return SimpleCNNCompletion(in_channels=3, out_channels=3,
                                   base_channels=base_channels, num_blocks=num_layers + 1)
    if completion_type == 'resnet':
        return ResNetCompletion(in_channels=3, out_channels=3,
                                base_channels=base_channels, num_blocks=num_layers + 1)
    raise ValueError(f"Unknown completion_type: {completion_type}")


class SIRMModule(nn.Module):

    def __init__(self,
                 num_iterations=3,
                 eta0=0.1,
                 epsilon=1e-8,
                 alpha_w=0.5, beta_w=0.3, gamma_w=0.2,
                 lambda1=0.01, lambda2=0.001,
                 unet_layers=4,
                 unet_base_channels=64,
                 convergence_threshold=1e-4,
                 inner_optimizer='adagrad',
                 completion_type='unet',
                 use_physics_weight=True):
        super().__init__()
        self.num_iterations = num_iterations
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.convergence_threshold = convergence_threshold
        self.use_physics_weight = use_physics_weight

        self.degradation_op = DegradationOperator()

        self.optimizer = get_inner_optimizer(
            inner_optimizer,
            eta0=eta0, epsilon=epsilon,
            alpha_w=alpha_w, beta_w=beta_w, gamma_w=gamma_w,
            use_physics_weight=use_physics_weight,
        )

        self.completion = _make_completion(
            completion_type=completion_type,
            base_channels=unet_base_channels,
            num_layers=unet_layers,
        )

    def forward(self, y, physics_params):
        """Iterative restoration following Eq. 26.

        X^(0) = Y
        For t = 0..T-1:
            g_t = grad_L at X^(t)
            X^(t+1/2) = inner_opt.step(X^(t), g_t)
            X^(t+1)   = U(X^(t+1/2); theta)
        """
        x_t = y.clone()
        state = self.optimizer.init_state(y)
        intermediates = []

        for t in range(self.num_iterations):
            grad = self.degradation_op.compute_gradient(
                x_t, y, physics_params,
                lambda1=self.lambda1, lambda2=self.lambda2,
            )
            x_half, state = self.optimizer.step(x_t, grad, state, physics_params)
            x_t = self.completion(x_half)
            intermediates.append(x_t)

            # Early stopping during eval only
            if t > 0 and not self.training:
                prev = intermediates[-2]
                change = torch.norm(x_t - prev) / (torch.norm(prev) + 1e-8)
                if change.item() < self.convergence_threshold:
                    break

        return x_t, intermediates
