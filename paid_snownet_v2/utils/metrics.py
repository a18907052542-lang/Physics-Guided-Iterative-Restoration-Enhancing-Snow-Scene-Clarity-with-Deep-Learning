"""
Evaluation metrics for snow scene image restoration.

Implements:
    - PSNR / SSIM (closed form)
    - LPIPS via the lpips package when installed, otherwise a VGG-feature
      L2 distance proxy (the proxy is clearly labelled in returned dict)
    - PixelSim / EdgeSim / ColorSim helpers
    - SnowDenoiseMetric (Eq. 31) with weights from Table 9.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import lpips for the real metric. If unavailable, we fall back to a
# VGG-based feature distance and tag the result accordingly.
try:
    import lpips as _lpips_pkg
    _HAS_LPIPS = True
except Exception:
    _HAS_LPIPS = False

try:
    import torchvision.models as tvm
    _HAS_TV = True
except Exception:
    _HAS_TV = False


# ---------------------------------------------------------------------- PSNR

def compute_psnr(pred, target, max_val=1.0):
    mse = torch.mean((pred - target) ** 2)
    if mse.item() == 0:
        return float('inf')
    return (10.0 * torch.log10(max_val ** 2 / mse)).item()


# ---------------------------------------------------------------------- SSIM

def compute_ssim(pred, target, window_size=11, max_val=1.0):
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32,
                          device=pred.device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = (g.unsqueeze(0) * g.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    channels = pred.shape[1]
    window = window.expand(channels, 1, window_size, window_size).contiguous()
    pad = window_size // 2

    mu1 = F.conv2d(pred, window, padding=pad, groups=channels)
    mu2 = F.conv2d(target, window, padding=pad, groups=channels)
    mu1_sq, mu2_sq, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = F.conv2d(pred * pred, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=pad, groups=channels) - mu12

    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()


# ---------------------------------------------------------------------- LPIPS

class _LPIPSWrapper(nn.Module):
    """Wrapper that uses the real `lpips` package when available, otherwise a
    VGG-feature L2 distance. The fallback is well-defined but is NOT the same
    quantity as LPIPS; callers should treat it as a proxy and report it as
    such (the dict returned by snow_denoise_metric tags it with `_proxy`)."""

    def __init__(self, net='alex'):
        super().__init__()
        self.mode = 'unavailable'
        if _HAS_LPIPS:
            try:
                self.lpips = _lpips_pkg.LPIPS(net=net, verbose=False)
                for p in self.lpips.parameters():
                    p.requires_grad = False
                self.lpips.eval()
                self.mode = 'lpips'
                return
            except Exception:
                pass
        if _HAS_TV:
            try:
                vgg = tvm.vgg16(weights=tvm.VGG16_Weights.DEFAULT).features[:16]
            except Exception:
                vgg = tvm.vgg16(weights=None).features[:16]
            for p in vgg.parameters():
                p.requires_grad = False
            vgg.eval()
            self.vgg = vgg
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
            self.mode = 'vgg_proxy'

    @torch.no_grad()
    def forward(self, pred, target):
        if self.mode == 'lpips':
            # lpips expects [-1, 1]
            p = pred * 2 - 1
            t = target * 2 - 1
            return self.lpips(p, t).mean().item()
        if self.mode == 'vgg_proxy':
            p = (pred - self.mean) / self.std
            t = (target - self.mean) / self.std
            return F.mse_loss(self.vgg(p), self.vgg(t)).item()
        return float('nan')


_LPIPS_SINGLETON = {}


def compute_lpips(pred, target, net='alex'):
    """Return (value, mode) where mode in {'lpips', 'vgg_proxy', 'unavailable'}."""
    key = (net, pred.device)
    if key not in _LPIPS_SINGLETON:
        m = _LPIPSWrapper(net=net).to(pred.device)
        _LPIPS_SINGLETON[key] = m
    metric = _LPIPS_SINGLETON[key]
    return metric(pred, target), metric.mode


# ---------------------------------------------------------------------- helpers

def compute_pixel_similarity(pred, target):
    return (1.0 - torch.mean(torch.abs(pred - target))).item()


def compute_edge_similarity(pred, target):
    sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                      dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
    sy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                      dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
    pg = pred.mean(dim=1, keepdim=True)
    tg = target.mean(dim=1, keepdim=True)
    pgx = F.conv2d(pg, sx, padding=1); pgy = F.conv2d(pg, sy, padding=1)
    tgx = F.conv2d(tg, sx, padding=1); tgy = F.conv2d(tg, sy, padding=1)
    pm = torch.sqrt(pgx ** 2 + pgy ** 2 + 1e-8).flatten()
    tm = torch.sqrt(tgx ** 2 + tgy ** 2 + 1e-8).flatten()
    pm = pm - pm.mean(); tm = tm - tm.mean()
    corr = (pm * tm).sum() / (torch.sqrt((pm ** 2).sum() * (tm ** 2).sum()) + 1e-8)
    return corr.item()


def compute_color_similarity(pred, target):
    s = 0.0
    for c in range(pred.shape[1]):
        p = pred[:, c].flatten()
        t = target[:, c].flatten()
        ph = torch.histc(p, bins=256, min=0.0, max=1.0); ph = ph / (ph.sum() + 1e-8)
        th = torch.histc(t, bins=256, min=0.0, max=1.0); th = th / (th.sum() + 1e-8)
        ph = ph - ph.mean(); th = th - th.mean()
        corr = (ph * th).sum() / (torch.sqrt((ph ** 2).sum() * (th ** 2).sum()) + 1e-8)
        s += corr.item()
    return s / pred.shape[1]


# ---------------------------------------------------------------------- composite

def snow_denoise_metric(pred, target):
    """Composite SnowDenoiseMetric (Eq. 31).
    Weights from Table 9: alpha=0.3, beta=0.25, gamma=0.15, delta=0.2, eta=0.1."""
    alpha, beta, gamma, delta, eta = 0.3, 0.25, 0.15, 0.2, 0.1

    pixel_sim = compute_pixel_similarity(pred, target)
    ssim_val = compute_ssim(pred, target)
    psnr_val = compute_psnr(pred, target)
    lpips_val, lpips_mode = compute_lpips(pred, target)
    # Map LPIPS distance -> similarity in [0,1] for the composite score.
    if lpips_mode in ('lpips', 'vgg_proxy'):
        perc_sim = max(0.0, 1.0 - lpips_val)
    else:
        perc_sim = min(psnr_val / 50.0, 1.0)
    edge_sim = compute_edge_similarity(pred, target)
    color_sim = compute_color_similarity(pred, target)

    metric = (alpha * pixel_sim + beta * ssim_val + gamma * perc_sim
              + delta * edge_sim + eta * color_sim)

    return {
        'SnowDenoiseMetric': metric,
        'PixelSim': pixel_sim,
        'SSIMSim': ssim_val,
        'PerceptualSim': perc_sim,
        'EdgeSim': edge_sim,
        'ColorSim': color_sim,
        'PSNR': psnr_val,
        'SSIM': ssim_val,
        'LPIPS': lpips_val,
        'LPIPS_mode': lpips_mode,
    }
