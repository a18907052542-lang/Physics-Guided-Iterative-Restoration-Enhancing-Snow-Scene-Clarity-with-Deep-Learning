"""
Structural smoke test for the codebase.

This script feeds *random* tensors through every model variant and ablation
configuration to verify shapes / wiring. The PSNR / SSIM numbers printed by
this script are MEANINGLESS — they are produced from random noise vs random
noise. This file exists solely to catch import / shape / API regressions
before you launch a real training run.

DO NOT cite or report any metric printed here.

Usage
-----
    python scripts/smoke_test.py
"""

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models.paid_snownet import PAIDSnowNet
from losses.multi_scale_loss import MultiScaleLoss, PhysicsConsistencyLoss
from utils.metrics import (
    compute_psnr, compute_ssim, compute_lpips, snow_denoise_metric
)


def green(s):
    return f'\033[92m{s}\033[0m'


def red(s):
    return f'\033[91m{s}\033[0m'


def check(name, fn):
    try:
        fn()
        print(green('[PASS]'), name)
        return True
    except Exception:
        print(red('[FAIL]'), name)
        traceback.print_exc()
        return False


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Torch:  {torch.__version__}')

    # Use a moderate spatial size to also exercise U-Net downsampling.
    x = torch.rand(2, 3, 64, 64, device=device)
    y = torch.rand(2, 3, 64, 64, device=device)

    ok = []

    def t_default():
        m = PAIDSnowNet().to(device)
        restored, phys, inters = m(x)
        assert restored.shape == x.shape
        assert len(inters) >= 1
        for key in ('scattering_coeff', 'transmission', 'occlusion_mask'):
            assert key in phys

    def t_no_spm():
        m = PAIDSnowNet(use_spm=False).to(device)
        restored, _, _ = m(x)
        assert restored.shape == x.shape

    def t_T1():
        m = PAIDSnowNet(sirm_iterations=1).to(device)
        restored, _, inters = m(x)
        assert len(inters) == 1

    def t_no_phys_w():
        m = PAIDSnowNet(sirm_use_physics_weight=False).to(device)
        m(x)

    def t_all_inner_optimizers():
        for opt in ['sgd', 'sgd+momentum', 'adam', 'rmsprop', 'adagrad']:
            m = PAIDSnowNet(sirm_inner_optimizer=opt).to(device)
            r, _, _ = m(x)
            assert r.shape == x.shape

    def t_all_completion_types():
        for c in ['unet', 'simple_cnn', 'resnet']:
            m = PAIDSnowNet(sirm_completion_type=c).to(device)
            r, _, _ = m(x)
            assert r.shape == x.shape

    def t_spm_ablations():
        for kw in [
            dict(spm_disable_mie=True),
            dict(spm_disable_rayleigh=True),
            dict(spm_disable_occlusion=True),
            dict(spm_fusion_mode='average'),
            dict(spm_fusion_mode='concat'),
            dict(spm_kernel_sizes=(3,)),
            dict(spm_kernel_sizes=(3, 5)),
        ]:
            m = PAIDSnowNet(**kw).to(device)
            r, _, _ = m(x)
            assert r.shape == x.shape

    def t_loss_variants():
        m = PAIDSnowNet().to(device)
        _, phys, inters = m(x)
        for recon in ('l1', 'l2'):
            for up in (True, False):
                for ue in (True, False):
                    crit = MultiScaleLoss(num_iterations=3, recon_type=recon,
                                          use_perceptual=up, use_edge=ue).to(device)
                    loss, ld = crit(inters, y)
                    assert torch.isfinite(loss).item()
        pcl = PhysicsConsistencyLoss().to(device)
        v = pcl(inters[-1], x, phys)
        assert torch.isfinite(v).item()

    def t_metrics():
        a = torch.rand(1, 3, 64, 64, device=device)
        b = torch.rand(1, 3, 64, 64, device=device)
        compute_psnr(a, b)
        compute_ssim(a, b)
        v, mode = compute_lpips(a, b)
        assert mode in ('lpips', 'vgg_proxy', 'unavailable')
        snow_denoise_metric(a, b)

    def t_train_step():
        m = PAIDSnowNet().to(device)
        opt = torch.optim.Adam(m.parameters(), lr=1e-4)
        crit = MultiScaleLoss(num_iterations=3).to(device)
        pcl = PhysicsConsistencyLoss().to(device)
        m.train()
        restored, phys, inters = m(x)
        loss, _ = crit(inters, y)
        loss = loss + 0.1 * pcl(restored, x, phys)
        opt.zero_grad(); loss.backward(); opt.step()

    print('\n=== running structural checks (random tensors only) ===\n')
    ok.append(check('default model forward', t_default))
    ok.append(check('without SPM',           t_no_spm))
    ok.append(check('T=1 iteration',         t_T1))
    ok.append(check('without physics weight', t_no_phys_w))
    ok.append(check('all inner optimizers (sgd/momentum/adam/rmsprop/adagrad)',
                    t_all_inner_optimizers))
    ok.append(check('all completion networks (unet/simple_cnn/resnet)',
                    t_all_completion_types))
    ok.append(check('SPM ablation flags',    t_spm_ablations))
    ok.append(check('loss variants (l1/l2 x perceptual x edge)', t_loss_variants))
    ok.append(check('metrics (psnr/ssim/lpips/SnowDenoiseMetric)', t_metrics))
    ok.append(check('one training step (backward + optimizer step)', t_train_step))

    print()
    if all(ok):
        print(green('All structural checks passed.'))
        print('Reminder: this script uses random tensors. The model has NOT been '
              'trained and any metric value printed here is meaningless.')
        sys.exit(0)
    else:
        print(red(f'{sum(not o for o in ok)}/{len(ok)} checks failed.'))
        sys.exit(1)


if __name__ == '__main__':
    main()
