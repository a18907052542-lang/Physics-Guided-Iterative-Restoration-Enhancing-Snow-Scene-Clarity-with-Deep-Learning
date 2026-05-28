"""
Comprehensive ablation studies for PAID-SnowNet.

Covers all module-level and component-level tables in the paper:

    Table 3  : convolutional kernel-size combinations  (ablation_kernel_sizes)
    Table 4  : iteration count T                       (ablation_iteration_count)
    Table 5  : inner optimizer SGD / Adam / AdaGrad …  (ablation_optimizer)
    Table 6  : U-Net depth / width                     (ablation_unet_architecture)
    Table 12 : module-level                            (ablation_modules)
    Table 13 : SPM components                          (ablation_spm_components)
    Table 14 : SIRM components                         (ablation_sirm_components)
    Table 15 : network variants                        (ablation_network_variants)

The Table-16 loss-component ablation lives in experiments/ablation_loss.py
because it requires a (short) training loop, not just an evaluation pass.

Usage
-----
    python experiments/ablation_study.py --data_root <Snow100K_root> \
                                         [--checkpoint best.pth] \
                                         [--max_batches 50]

All numbers in the resulting JSON are produced by running the model on the
provided test split; if no checkpoint is given the model uses its randomly
initialised weights, in which case the JSON is a sanity check of the
experimental *pipeline*, not a trained-model evaluation.
"""

import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.paid_snownet import PAIDSnowNet
from datasets.snow_dataset import Snow100KDataset
from utils.metrics import compute_psnr, compute_ssim, compute_lpips


# --------------------------------------------------------------------- helpers

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_weights_if_available(model, ckpt_path, device):
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        sd = ckpt.get('model_state_dict', ckpt)
        # Use strict=False so that ablation variants with different submodules
        # can still load the matching parameters.
        missing, unexpected = model.load_state_dict(sd, strict=False)
        return {'loaded': True, 'missing_keys': len(missing), 'unexpected_keys': len(unexpected)}
    return {'loaded': False}


@torch.no_grad()
def evaluate_model(model, dataloader, device, max_batches=50, with_lpips=True):
    model.eval()
    total_psnr = total_ssim = total_lpips = 0.0
    n = 0
    lpips_mode = None
    for i, (inputs, targets) in enumerate(dataloader):
        if i >= max_batches:
            break
        inputs = inputs.to(device); targets = targets.to(device)
        restored, _, _ = model(inputs)
        total_psnr += compute_psnr(restored, targets)
        total_ssim += compute_ssim(restored, targets)
        if with_lpips:
            v, m = compute_lpips(restored, targets)
            total_lpips += v
            lpips_mode = m
        n += 1
    n = max(n, 1)
    out = {'PSNR': total_psnr / n, 'SSIM': total_ssim / n}
    if with_lpips:
        out['LPIPS'] = total_lpips / n
        out['LPIPS_mode'] = lpips_mode
    return out


@torch.no_grad()
def measure_inference_ms(model, device, shape=(1, 3, 256, 256), num_runs=10, warmup=2):
    model.eval()
    dummy = torch.randn(*shape, device=device)
    for _ in range(warmup):
        _ = model(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_runs):
        _ = model(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    return (time.time() - start) * 1000.0 / num_runs


# ------------------------------------------------------------- Table 3: kernels

def ablation_kernel_sizes(dataloader, device, ckpt, max_batches):
    print('\n=== Table 3: kernel size combinations ===')
    configs = [
        ('Single 3x3', [3]),
        ('Single 5x5', [5]),
        ('Single 7x7', [7]),
        ('3x3 + 5x5', [3, 5]),
        ('3x3 + 7x7', [3, 7]),
        ('5x5 + 7x7', [5, 7]),
        ('3x3 + 5x5 + 7x7 (Ours)', [3, 5, 7]),
    ]
    results = []
    for name, kernels in configs:
        model = PAIDSnowNet(spm_kernel_sizes=kernels).to(device)
        load_info = load_weights_if_available(model, ckpt, device)
        params = count_parameters(model) / 1e6
        m = evaluate_model(model, dataloader, device, max_batches=max_batches)
        row = {'config': name, 'kernels': kernels,
               'PSNR': m['PSNR'], 'SSIM': m['SSIM'], 'LPIPS': m.get('LPIPS'),
               'Params_M': params, 'checkpoint': load_info}
        results.append(row)
        print(f"  {name:30s} | PSNR {m['PSNR']:.2f} | SSIM {m['SSIM']:.3f} "
              f"| LPIPS {m.get('LPIPS', float('nan')):.3f} | Params {params:.1f}M")
    return results


# ------------------------------------------------------------ Table 4: iters T

def ablation_iteration_count(dataloader, device, ckpt, max_batches):
    print('\n=== Table 4: iteration count T ===')
    results = []
    for T in [1, 2, 3, 4, 5]:
        model = PAIDSnowNet(sirm_iterations=T).to(device)
        load_info = load_weights_if_available(model, ckpt, device)
        t_ms = measure_inference_ms(model, device)
        m = evaluate_model(model, dataloader, device, max_batches=max_batches)
        row = {'T': T, 'PSNR': m['PSNR'], 'SSIM': m['SSIM'], 'LPIPS': m.get('LPIPS'),
               'time_ms': t_ms, 'checkpoint': load_info}
        results.append(row)
        print(f"  T={T} | PSNR {m['PSNR']:.2f} | SSIM {m['SSIM']:.3f} "
              f"| LPIPS {m.get('LPIPS', float('nan')):.3f} | {t_ms:.0f} ms")
    return results


# --------------------------------------------------------- Table 5: optimizers

def ablation_optimizer(dataloader, device, ckpt, max_batches):
    """Table 5 with real optimizer switching inside SIRM."""
    print('\n=== Table 5: inner optimizer ===')
    configs = [
        ('SGD',            'sgd'),
        ('SGD + Momentum', 'sgd+momentum'),
        ('Adam',           'adam'),
        ('RMSprop',        'rmsprop'),
        ('AdaGrad (Ours)', 'adagrad'),
    ]
    results = []
    for name, key in configs:
        model = PAIDSnowNet(sirm_inner_optimizer=key).to(device)
        load_info = load_weights_if_available(model, ckpt, device)
        m = evaluate_model(model, dataloader, device, max_batches=max_batches)
        row = {'optimizer': name, 'inner_optimizer': key,
               'PSNR': m['PSNR'], 'SSIM': m['SSIM'], 'LPIPS': m.get('LPIPS'),
               'checkpoint': load_info}
        results.append(row)
        print(f"  {name:18s} | PSNR {m['PSNR']:.2f} | SSIM {m['SSIM']:.3f} "
              f"| LPIPS {m.get('LPIPS', float('nan')):.3f}")
    return results


# -------------------------------------------------------- Table 6: U-Net arch

def ablation_unet_architecture(dataloader, device, ckpt, max_batches):
    print('\n=== Table 6: U-Net architecture ===')
    configs = [
        ('Shallow-Narrow', 3, 32),
        ('Shallow-Wide',   3, 64),
        ('Default (Ours)', 4, 64),
        ('Deep-Narrow',    5, 64),
        ('Deep-Wide',      5, 128),
    ]
    results = []
    for name, L, C in configs:
        model = PAIDSnowNet(unet_layers=L, unet_base_channels=C).to(device)
        load_info = load_weights_if_available(model, ckpt, device)
        params = count_parameters(model) / 1e6
        m = evaluate_model(model, dataloader, device, max_batches=max_batches)
        row = {'config': name, 'layers': L, 'base_channels': C,
               'PSNR': m['PSNR'], 'SSIM': m['SSIM'], 'LPIPS': m.get('LPIPS'),
               'Params_M': params, 'checkpoint': load_info}
        results.append(row)
        print(f"  {name:18s} | L={L} C={C:3d} | PSNR {m['PSNR']:.2f} "
              f"| SSIM {m['SSIM']:.3f} | Params {params:.1f}M")
    return results


# --------------------------------------------------- Table 12: module-level

def ablation_modules(dataloader, device, ckpt, max_batches):
    print('\n=== Table 12: module-level ablation ===')
    configs = [
        # name,                    use_spm, sirm_iters, use_phys_w
        ('Full PAID-SnowNet',        True, 3, True),
        ('w/o SPM Module',           False, 3, True),    # neutral physics params
        ('w/o SIRM Module (T=1)',    True, 1, True),     # single completion pass
        ('w/o Iteration (T=1)',      True, 1, True),     # alias of above
        ('w/o Physics-aware Weight', True, 3, False),
    ]
    results = []
    for name, use_spm, T, use_pw in configs:
        model = PAIDSnowNet(use_spm=use_spm, sirm_iterations=T,
                            sirm_use_physics_weight=use_pw).to(device)
        load_info = load_weights_if_available(model, ckpt, device)
        m = evaluate_model(model, dataloader, device, max_batches=max_batches)
        row = {'config': name, 'use_spm': use_spm, 'T': T, 'use_phys_w': use_pw,
               'PSNR': m['PSNR'], 'SSIM': m['SSIM'], 'LPIPS': m.get('LPIPS'),
               'checkpoint': load_info}
        results.append(row)
        print(f"  {name:30s} | PSNR {m['PSNR']:.2f} | SSIM {m['SSIM']:.3f} "
              f"| LPIPS {m.get('LPIPS', float('nan')):.3f}")
    return results


# ------------------------------------------ Table 13: SPM component ablation

def ablation_spm_components(dataloader, device, ckpt, max_batches):
    print('\n=== Table 13: SPM component ablation ===')
    configs = [
        # name,                              kernels,  mie, rayleigh, occ, fusion
        ('Full SPM (Ours)',                  (3, 5, 7), False, False, False, 'attention'),
        ('Single 3x3 only',                  (3,),      False, False, False, 'attention'),
        ('Single 5x5 only',                  (5,),      False, False, False, 'attention'),
        ('Single 7x7 only',                  (7,),      False, False, False, 'attention'),
        ('3x3 + 5x5',                        (3, 5),    False, False, False, 'attention'),
        ('3x3 + 7x7',                        (3, 7),    False, False, False, 'attention'),
        ('Triple branch + Average Fusion',   (3, 5, 7), False, False, False, 'average'),
        ('Triple branch + Concat Fusion',    (3, 5, 7), False, False, False, 'concat'),
        ('w/o Mie scattering branches',      (3, 5, 7), True,  False, False, 'attention'),
        ('w/o Rayleigh scattering branch',   (3, 5, 7), False, True,  False, 'attention'),
        ('w/o Occlusion mask estimation',    (3, 5, 7), False, False, True,  'attention'),
    ]
    results = []
    for name, k, dm, dr, do, fmode in configs:
        model = PAIDSnowNet(
            spm_kernel_sizes=k,
            spm_disable_mie=dm,
            spm_disable_rayleigh=dr,
            spm_disable_occlusion=do,
            spm_fusion_mode=fmode,
        ).to(device)
        load_info = load_weights_if_available(model, ckpt, device)
        m = evaluate_model(model, dataloader, device, max_batches=max_batches)
        row = {'config': name, 'kernels': list(k),
               'disable_mie': dm, 'disable_rayleigh': dr,
               'disable_occlusion': do, 'fusion_mode': fmode,
               'PSNR': m['PSNR'], 'SSIM': m['SSIM'], 'LPIPS': m.get('LPIPS'),
               'checkpoint': load_info}
        results.append(row)
        print(f"  {name:38s} | PSNR {m['PSNR']:.2f} | SSIM {m['SSIM']:.3f}")
    return results


# ----------------------------------------- Table 14: SIRM component ablation

def ablation_sirm_components(dataloader, device, ckpt, max_batches):
    print('\n=== Table 14: SIRM component ablation ===')
    configs = [
        # name,                       optimizer,     completion,  use_pw, layers
        ('Full SIRM (AdaGrad + PW + U-Net)', 'adagrad', 'unet',       True,  4),
        ('SGD optimizer',                    'sgd',      'unet',       True,  4),
        ('Adam optimizer',                   'adam',     'unet',       True,  4),
        ('RMSprop optimizer',                'rmsprop',  'unet',       True,  4),
        ('AdaGrad w/o Physics-aware Weight', 'adagrad',  'unet',       False, 4),
        ('U-Net replaced by ResNet',         'adagrad',  'resnet',     True,  4),
        ('U-Net replaced by Simple CNN',     'adagrad',  'simple_cnn', True,  4),
        ('Shallow U-Net (3 layers)',         'adagrad',  'unet',       True,  3),
        ('Deep U-Net (5 layers)',            'adagrad',  'unet',       True,  5),
    ]
    results = []
    for name, opt, comp, use_pw, L in configs:
        model = PAIDSnowNet(
            sirm_inner_optimizer=opt,
            sirm_completion_type=comp,
            sirm_use_physics_weight=use_pw,
            unet_layers=L,
        ).to(device)
        load_info = load_weights_if_available(model, ckpt, device)
        m = evaluate_model(model, dataloader, device, max_batches=max_batches)
        row = {'config': name, 'optimizer': opt, 'completion': comp,
               'use_phys_w': use_pw, 'unet_layers': L,
               'PSNR': m['PSNR'], 'SSIM': m['SSIM'], 'LPIPS': m.get('LPIPS'),
               'checkpoint': load_info}
        results.append(row)
        print(f"  {name:38s} | PSNR {m['PSNR']:.2f} | SSIM {m['SSIM']:.3f}")
    return results


# ------------------------------------------ Table 15: network architecture

def ablation_network_variants(dataloader, device, ckpt, max_batches):
    print('\n=== Table 15: network architecture variants ===')
    configs = [
        ('Base (Ours)',  4, 64),
        ('Lightweight',  3, 32),
        ('Deeper',       5, 64),
        ('Wider',        4, 80),
    ]
    results = []
    for name, L, C in configs:
        model = PAIDSnowNet(unet_layers=L, unet_base_channels=C).to(device)
        load_info = load_weights_if_available(model, ckpt, device)
        params = count_parameters(model) / 1e6
        t_ms = measure_inference_ms(model, device)
        m = evaluate_model(model, dataloader, device, max_batches=max_batches)
        row = {'config': name, 'layers': L, 'base_channels': C,
               'Params_M': params, 'PSNR': m['PSNR'], 'SSIM': m['SSIM'],
               'LPIPS': m.get('LPIPS'), 'time_ms': t_ms, 'checkpoint': load_info}
        results.append(row)
        print(f"  {name:14s} | Params {params:.1f}M | "
              f"PSNR {m['PSNR']:.2f} | SSIM {m['SSIM']:.3f} | {t_ms:.0f} ms")
    return results


# ------------------------------------------------------------------- driver

def main():
    parser = argparse.ArgumentParser(description='PAID-SnowNet ablation studies')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to Snow100K root directory.')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Optional path to trained weights (.pth).')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./results/ablation')
    parser.add_argument('--max_batches', type=int, default=50,
                        help='Number of test batches used per configuration.')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--only', type=str, default=None,
                        help='Comma-separated subset: kernel,iter,opt,unet,module,'
                             'spm,sirm,variant. Default = all.')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    dataset = Snow100KDataset(args.data_root, split='test',
                              patch_size=256, augment=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    selectors = {
        'kernel':  ablation_kernel_sizes,
        'iter':    ablation_iteration_count,
        'opt':     ablation_optimizer,
        'unet':    ablation_unet_architecture,
        'module':  ablation_modules,
        'spm':     ablation_spm_components,
        'sirm':    ablation_sirm_components,
        'variant': ablation_network_variants,
    }
    if args.only:
        wanted = [s.strip() for s in args.only.split(',') if s.strip()]
        for w in wanted:
            if w not in selectors:
                raise ValueError(f'Unknown ablation key: {w}')
    else:
        wanted = list(selectors.keys())

    os.makedirs(args.save_dir, exist_ok=True)
    all_results = {}
    for key in wanted:
        all_results[key] = selectors[key](dataloader, device,
                                          args.checkpoint, args.max_batches)

    with open(os.path.join(args.save_dir, 'ablation_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nResults saved to {args.save_dir}/ablation_results.json')


if __name__ == '__main__':
    main()
