"""
Physics-aware weighting and inner optimizers for SIRM.

Implements Eq. 21-23 (AdaGrad + physics-aware weight), and additionally
provides honest implementations of SGD/Momentum/Adam/RMSprop so that
the Table 5 optimizer ablation is *actually* a comparison between
different optimization rules (rather than a relabelling of iteration counts).
"""

import torch
import torch.nn as nn


class PhysicsAwareWeight(nn.Module):
    """w(x,y) = alpha_w * M(x,y) + beta_w * t(x,y) + gamma_w  (Eq. 23)."""

    def __init__(self, alpha_w=0.5, beta_w=0.3, gamma_w=0.2, enabled=True):
        super().__init__()
        self.alpha_w = alpha_w
        self.beta_w = beta_w
        self.gamma_w = gamma_w
        self.enabled = enabled

    def forward(self, physics_params):
        if not self.enabled:
            # Constant weight 1 ⇒ AdaGrad with no physics modulation.
            ref = physics_params['occlusion_mask']
            return torch.ones_like(ref)
        M = physics_params['occlusion_mask']
        t = physics_params['transmission']
        return self.alpha_w * M + self.beta_w * t + self.gamma_w


class _BaseInner(nn.Module):
    """Common interface for inner optimizers used inside the SIRM iteration."""

    def __init__(self, eta0=0.1, epsilon=1e-8,
                 alpha_w=0.5, beta_w=0.3, gamma_w=0.2,
                 use_physics_weight=True):
        super().__init__()
        self.eta0 = eta0
        self.epsilon = epsilon
        self.physics_weight = PhysicsAwareWeight(
            alpha_w, beta_w, gamma_w, enabled=use_physics_weight)

    def init_state(self, like_tensor):
        """Return initial state dict for the optimizer."""
        raise NotImplementedError

    def step(self, x_k, grad, state, physics_params):
        raise NotImplementedError


class SGDOptimizer(_BaseInner):
    """Plain SGD with fixed (physics-modulated) step size."""

    def init_state(self, like_tensor):
        return {}

    def step(self, x_k, grad, state, physics_params):
        w = self.physics_weight(physics_params)
        eta = self.eta0 * w
        x_half = torch.clamp(x_k - eta * grad, 0.0, 1.0)
        return x_half, state


class MomentumOptimizer(_BaseInner):
    """SGD with momentum (Polyak heavy ball)."""

    def __init__(self, momentum=0.9, **kwargs):
        super().__init__(**kwargs)
        self.momentum = momentum

    def init_state(self, like_tensor):
        return {'v': torch.zeros_like(like_tensor)}

    def step(self, x_k, grad, state, physics_params):
        v = state['v']
        v = self.momentum * v + grad
        state['v'] = v
        w = self.physics_weight(physics_params)
        eta = self.eta0 * w
        x_half = torch.clamp(x_k - eta * v, 0.0, 1.0)
        return x_half, state


class AdamOptimizer(_BaseInner):
    """Adam with bias correction."""

    def __init__(self, beta1=0.9, beta2=0.999, **kwargs):
        super().__init__(**kwargs)
        self.beta1 = beta1
        self.beta2 = beta2

    def init_state(self, like_tensor):
        return {
            'm': torch.zeros_like(like_tensor),
            'v': torch.zeros_like(like_tensor),
            't': 0,
        }

    def step(self, x_k, grad, state, physics_params):
        state['t'] += 1
        t = state['t']
        m = self.beta1 * state['m'] + (1 - self.beta1) * grad
        v = self.beta2 * state['v'] + (1 - self.beta2) * grad ** 2
        state['m'] = m
        state['v'] = v
        m_hat = m / (1 - self.beta1 ** t)
        v_hat = v / (1 - self.beta2 ** t)
        w = self.physics_weight(physics_params)
        eta = self.eta0 * w / (torch.sqrt(v_hat) + self.epsilon)
        x_half = torch.clamp(x_k - eta * m_hat, 0.0, 1.0)
        return x_half, state


class RMSpropOptimizer(_BaseInner):
    """RMSprop with EMA of squared gradients."""

    def __init__(self, alpha=0.99, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def init_state(self, like_tensor):
        return {'v': torch.zeros_like(like_tensor)}

    def step(self, x_k, grad, state, physics_params):
        v = self.alpha * state['v'] + (1 - self.alpha) * grad ** 2
        state['v'] = v
        w = self.physics_weight(physics_params)
        eta = self.eta0 * w / (torch.sqrt(v) + self.epsilon)
        x_half = torch.clamp(x_k - eta * grad, 0.0, 1.0)
        return x_half, state


class AdaGradOptimizer(_BaseInner):
    """AdaGrad inner optimizer (default for PAID-SnowNet, Eq. 21-22)."""

    def init_state(self, like_tensor):
        return {'G': torch.zeros_like(like_tensor)}

    def step(self, x_k, grad, state, physics_params):
        G = state['G'] + grad ** 2
        state['G'] = G
        w = self.physics_weight(physics_params)
        eta = self.eta0 * w / (torch.sqrt(G + self.epsilon))
        x_half = torch.clamp(x_k - eta * grad, 0.0, 1.0)
        return x_half, state


def get_inner_optimizer(name='adagrad', **kwargs):
    """Factory used by SIRM and the optimizer ablation experiment."""
    name = name.lower()
    table = {
        'sgd': SGDOptimizer,
        'momentum': MomentumOptimizer,
        'sgd+momentum': MomentumOptimizer,
        'adam': AdamOptimizer,
        'rmsprop': RMSpropOptimizer,
        'adagrad': AdaGradOptimizer,
    }
    if name not in table:
        raise ValueError(f"Unknown inner optimizer: {name}")
    return table[name](**kwargs)
