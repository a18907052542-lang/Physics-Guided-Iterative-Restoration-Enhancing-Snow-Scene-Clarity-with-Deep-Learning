"""
Multi-scale loss functions for PAID-SnowNet training.

Implements Eq. 29-30 and the Table 16 loss-component ablation:
    L_total = sum_t w_t * L_rec(X^t, X_gt)
              + alpha * L_perc(X^T, X_gt)
              + beta  * L_edge(X^T, X_gt)
where L_rec is L1 or L2 (configurable).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchvision.models as models
    _HAS_TV = True
except Exception:
    _HAS_TV = False


class VGGPerceptualLoss(nn.Module):
    """VGG-19 perceptual loss: sum_l ||phi_l(X) - phi_l(X_gt)||_2^2."""

    def __init__(self, feature_layers=(2, 7, 12, 21, 30)):
        super().__init__()
        self.feature_layers = list(feature_layers)
        self.available = False
        if not _HAS_TV:
            return
        try:
            vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        except Exception:
            vgg = models.vgg19(weights=None).features
        # Disable in-place ReLU so that autograd can backprop through the
        # cached activations used by MSE losses at each chunk.
        for m in vgg.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = False
        self.blocks = nn.ModuleList()
        prev = 0
        for layer in self.feature_layers:
            self.blocks.append(nn.Sequential(*list(vgg.children())[prev:layer + 1]))
            prev = layer + 1
        for p in self.parameters():
            p.requires_grad = False
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.available = True

    def forward(self, pred, target):
        if not self.available:
            return F.l1_loss(pred, target)
        x = (pred - self.mean) / self.std
        y = (target - self.mean) / self.std
        loss = 0.0
        for blk in self.blocks:
            x = blk(x)
            y = blk(y)
            loss = loss + F.mse_loss(x, y)
        return loss


class EdgeLoss(nn.Module):
    """L1 difference of Sobel gradient maps."""

    def __init__(self):
        super().__init__()
        sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sx', sx)
        self.register_buffer('sy', sy)

    def forward(self, pred, target):
        loss = 0.0
        for c in range(pred.shape[1]):
            p = pred[:, c:c + 1]
            t = target[:, c:c + 1]
            pgx = F.conv2d(p, self.sx, padding=1)
            pgy = F.conv2d(p, self.sy, padding=1)
            tgx = F.conv2d(t, self.sx, padding=1)
            tgy = F.conv2d(t, self.sy, padding=1)
            loss = loss + F.l1_loss(pgx, tgx) + F.l1_loss(pgy, tgy)
        return loss / pred.shape[1]


class MultiScaleLoss(nn.Module):
    """Multi-scale supervision loss (Eq. 29-30).

    Ablation flags (Table 16):
        recon_type:        'l1' (default) | 'l2'
        use_perceptual:    enable perceptual term
        use_edge:          enable edge term
    """

    def __init__(self, num_iterations=3, alpha=0.5, beta=0.1,
                 recon_type='l1', use_perceptual=True, use_edge=True):
        super().__init__()
        assert recon_type in ('l1', 'l2')
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.recon_type = recon_type
        self.use_perceptual = use_perceptual
        self.use_edge = use_edge
        self.weights = [0.5 ** (num_iterations - t) for t in range(1, num_iterations + 1)]
        if use_perceptual:
            self.perceptual = VGGPerceptualLoss()
        if use_edge:
            self.edge = EdgeLoss()

    def _recon(self, x, target):
        if self.recon_type == 'l1':
            return F.l1_loss(x, target)
        return F.mse_loss(x, target)

    def forward(self, intermediates, target):
        rec_total = 0.0
        perc_v = 0.0
        edge_v = 0.0
        T = len(intermediates)
        for t in range(T):
            wt = self.weights[min(t, len(self.weights) - 1)]
            x_t = intermediates[t]
            rec_total = rec_total + wt * self._recon(x_t, target)
            if t == T - 1:
                if self.use_perceptual:
                    perc_v = self.perceptual(x_t, target)
                if self.use_edge:
                    edge_v = self.edge(x_t, target)

        total = rec_total
        if self.use_perceptual:
            total = total + self.alpha * perc_v
        if self.use_edge:
            total = total + self.beta * edge_v

        loss_dict = {
            'total': total.item() if isinstance(total, torch.Tensor) else float(total),
            'reconstruction': rec_total.item() if isinstance(rec_total, torch.Tensor) else float(rec_total),
            'perceptual': perc_v.item() if isinstance(perc_v, torch.Tensor) else 0.0,
            'edge': edge_v.item() if isinstance(edge_v, torch.Tensor) else 0.0,
        }
        return total, loss_dict


class PhysicsConsistencyLoss(nn.Module):
    """Physics consistency: restored image, when re-degraded, should match input."""

    def forward(self, restored, degraded, physics_params):
        M = physics_params['occlusion_mask']
        t = physics_params['transmission']
        return F.l1_loss(restored * M * t, degraded * M * t)
