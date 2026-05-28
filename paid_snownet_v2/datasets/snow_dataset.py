"""
Snow100K dataset loader for PAID-SnowNet.

This loader expects a real Snow100K dataset on disk. It does NOT generate
synthetic placeholder pairs when the data directory is missing — if the
paths cannot be resolved, it raises FileNotFoundError so that downstream
metrics are never computed against fake data.

Snow100K is publicly available from
    https://sites.google.com/view/yunfuliu/desnownet
Place the data under <data_root>/{train,val,test}/{input,gt}/.
"""

import os
import glob
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


_IMAGE_EXTS = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff')


def _find_dir(candidates):
    for d in candidates:
        if os.path.isdir(d):
            return d
    return None


class Snow100KDataset(Dataset):
    """Paired snow/clean image dataset.

    Directory layout searched (first match wins for each side):
        <data_root>/<split>/input,  <data_root>/<split>/snow,  <data_root>/<split>/degraded
        <data_root>/<split>/gt,     <data_root>/<split>/clean, <data_root>/<split>/target
        <data_root>/synthetic/<split>/snow, <data_root>/synthetic/<split>/gt
    """

    def __init__(self, data_root, split='train', patch_size=256,
                 augment=True, max_samples=None):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.patch_size = patch_size
        self.augment = augment and (split == 'train')

        input_dir = _find_dir([
            os.path.join(data_root, split, 'input'),
            os.path.join(data_root, split, 'snow'),
            os.path.join(data_root, split, 'degraded'),
            os.path.join(data_root, 'synthetic', split, 'snow'),
        ])
        gt_dir = _find_dir([
            os.path.join(data_root, split, 'gt'),
            os.path.join(data_root, split, 'clean'),
            os.path.join(data_root, split, 'target'),
            os.path.join(data_root, 'synthetic', split, 'gt'),
        ])

        if input_dir is None or gt_dir is None:
            raise FileNotFoundError(
                f"Could not locate paired input/gt directories under '{data_root}' "
                f"for split='{split}'. Expected one of "
                f"<data_root>/{split}/{{input,snow,degraded}} and "
                f"<data_root>/{split}/{{gt,clean,target}}. "
                f"Download Snow100K from https://sites.google.com/view/yunfuliu/desnownet "
                f"and arrange the data accordingly."
            )

        pairs = []
        for ext in _IMAGE_EXTS:
            for src in sorted(glob.glob(os.path.join(input_dir, ext))):
                tgt = os.path.join(gt_dir, os.path.basename(src))
                if os.path.exists(tgt):
                    pairs.append((src, tgt))

        if not pairs:
            raise FileNotFoundError(
                f"Found input dir '{input_dir}' and gt dir '{gt_dir}' but no "
                f"matching file pairs by basename. Check that filenames "
                f"correspond on both sides."
            )

        if max_samples is not None:
            pairs = pairs[:max_samples]

        self.input_paths = [p[0] for p in pairs]
        self.gt_paths = [p[1] for p in pairs]
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        x = self.to_tensor(Image.open(self.input_paths[idx]).convert('RGB'))
        y = self.to_tensor(Image.open(self.gt_paths[idx]).convert('RGB'))

        if self.split == 'train':
            _, H, W = x.shape
            if H >= self.patch_size and W >= self.patch_size:
                top = random.randint(0, H - self.patch_size)
                left = random.randint(0, W - self.patch_size)
                x = x[:, top:top + self.patch_size, left:left + self.patch_size]
                y = y[:, top:top + self.patch_size, left:left + self.patch_size]

        if self.augment:
            if random.random() > 0.5:
                x = torch.flip(x, [2])
                y = torch.flip(y, [2])
            k = random.randint(0, 3)
            if k > 0:
                x = torch.rot90(x, k, [1, 2])
                y = torch.rot90(y, k, [1, 2])

        return x, y


def get_dataloaders(config):
    train_set = Snow100KDataset(config.data_root, split='train',
                                patch_size=config.train_patch_size, augment=True)
    val_set = Snow100KDataset(config.data_root, split='val',
                              patch_size=config.train_patch_size, augment=False)
    test_set = Snow100KDataset(config.data_root, split='test',
                               patch_size=config.train_patch_size, augment=False)
    train_loader = DataLoader(train_set, batch_size=config.batch_size,
                              shuffle=True, num_workers=config.num_workers,
                              pin_memory=config.pin_memory, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size,
                            shuffle=False, num_workers=config.num_workers,
                            pin_memory=config.pin_memory)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False,
                             num_workers=config.num_workers,
                             pin_memory=config.pin_memory)
    return train_loader, val_loader, test_loader
