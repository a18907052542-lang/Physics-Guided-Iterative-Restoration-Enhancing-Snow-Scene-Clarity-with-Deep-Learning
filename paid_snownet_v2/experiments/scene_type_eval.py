"""
Table 11 — Performance analysis across scene types.

Snow100K provides flat splits without scene-type labels. This script supports
*either* an explicit scene-type subfolder layout, *or* per-image scene labels
provided via a JSON manifest. Use whichever you can produce from your data
preparation pipeline; the script will simply report per-group metrics.

Two supported layouts
---------------------
Layout A (subfolders):
    <data_root>/test_by_scene/urban_streets/{input,gt}/
    <data_root>/test_by_scene/natural_landscapes/{input,gt}/
    <data_root>/test_by_scene/indoor_scenes/{input,gt}/

Layout B (manifest):
    --scene_manifest path/to/manifest.json with
        {"urban_streets": ["snow_00001.png", ...], "natural_landscapes": [...], ...}
    files are looked up under <data_root>/test/{input,gt}.

Usage
-----
    python experiments/scene_type_eval.py --data_root <root> --checkpoint best.pth
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from models.paid_snownet import PAIDSnowNet
from datasets.snow_dataset import Snow100KDataset
from utils.metrics import compute_psnr, compute_ssim, compute_lpips


class _SubfolderPairs(Dataset):
    def __init__(self, root):
        self.root = root
        self.to_tensor = transforms.ToTensor()
        input_dir = os.path.join(root, 'input')
        gt_dir = os.path.join(root, 'gt')
        if not (os.path.isdir(input_dir) and os.path.isdir(gt_dir)):
            raise FileNotFoundError(f'{root} must contain input/ and gt/ subdirs')
        self.files = []
        for f in sorted(os.listdir(input_dir)):
            tgt = os.path.join(gt_dir, f)
            if os.path.exists(tgt):
                self.files.append((os.path.join(input_dir, f), tgt))
        if not self.files:
            raise FileNotFoundError(f'No paired files in {root}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = self.to_tensor(Image.open(self.files[idx][0]).convert('RGB'))
        y = self.to_tensor(Image.open(self.files[idx][1]).convert('RGB'))
        return x, y


class _ManifestPairs(Dataset):
    def __init__(self, data_root, filenames):
        self.to_tensor = transforms.ToTensor()
        input_dir = os.path.join(data_root, 'test', 'input')
        gt_dir = os.path.join(data_root, 'test', 'gt')
        if not (os.path.isdir(input_dir) and os.path.isdir(gt_dir)):
            input_dir = os.path.join(data_root, 'input')
            gt_dir = os.path.join(data_root, 'gt')
        self.files = []
        for f in filenames:
            ip = os.path.join(input_dir, f)
            tp = os.path.join(gt_dir, f)
            if os.path.exists(ip) and os.path.exists(tp):
                self.files.append((ip, tp))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = self.to_tensor(Image.open(self.files[idx][0]).convert('RGB'))
        y = self.to_tensor(Image.open(self.files[idx][1]).convert('RGB'))
        return x, y


@torch.no_grad()
def evaluate(model, dataset, device, batch_size=1):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    psnr = ssim = lpips_v = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        restored, _, _ = model(x)
        psnr += compute_psnr(restored, y)
        ssim += compute_ssim(restored, y)
        v, _ = compute_lpips(restored, y); lpips_v += v
        n += 1
    n = max(n, 1)
    return {'PSNR': psnr / n, 'SSIM': ssim / n, 'LPIPS': lpips_v / n, 'num_samples': n}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--scene_manifest', type=str, default=None,
                        help='Optional JSON: {scene_name: [filenames]}')
    parser.add_argument('--scenes_root', type=str, default=None,
                        help='Optional subfolder root: <root>/<scene>/{input,gt}')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./results/scene_type')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    model = PAIDSnowNet().to(device)
    if args.checkpoint and os.path.exists(args.checkpoint):
        sd = torch.load(args.checkpoint, map_location=device)
        sd = sd.get('model_state_dict', sd)
        model.load_state_dict(sd, strict=False)

    scenes = {}
    if args.scene_manifest and os.path.exists(args.scene_manifest):
        with open(args.scene_manifest) as f:
            manifest = json.load(f)
        for name, files in manifest.items():
            scenes[name] = _ManifestPairs(args.data_root, files)
    elif args.scenes_root and os.path.isdir(args.scenes_root):
        for name in sorted(os.listdir(args.scenes_root)):
            sub = os.path.join(args.scenes_root, name)
            if os.path.isdir(sub):
                scenes[name] = _SubfolderPairs(sub)
    else:
        # Fall back to a single bucket on the regular test split.
        scenes['all_test_images'] = Snow100KDataset(args.data_root, split='test',
                                                    patch_size=256, augment=False)

    results = {}
    for name, ds in scenes.items():
        print(f'\n--- scene: {name} ({len(ds)} images) ---')
        results[name] = evaluate(model, ds, device)
        for k, v in results[name].items():
            print(f'  {k}: {v}')

    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, 'scene_type_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved -> {args.save_dir}/scene_type_results.json')


if __name__ == '__main__':
    main()
