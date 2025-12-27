"""
PAID-SnowNet Data Loading Utilities
====================================

Dataset classes and data augmentation for snow image restoration.
Supports Snow100K, SRRS, and custom datasets.
"""

import os
import random
from typing import Tuple, List, Optional, Callable, Dict, Any
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np


class SnowDataset(Dataset):
    """
    Base dataset class for snow image restoration.
    
    Expects directory structure:
    data_dir/
        input/      # Snow-degraded images
        gt/         # Ground truth clean images
    
    Or paired format:
    data_dir/
        xxx_snow.png, xxx_gt.png
    """
    
    def __init__(
        self,
        data_dir: str,
        patch_size: int = 256,
        augment: bool = True,
        normalize: bool = True,
        mode: str = 'train'  # 'train', 'val', 'test'
    ):
        self.data_dir = Path(data_dir)
        self.patch_size = patch_size
        self.augment = augment and mode == 'train'
        self.normalize = normalize
        self.mode = mode
        
        # Find image pairs
        self.pairs = self._find_pairs()
        
        if len(self.pairs) == 0:
            raise ValueError(f"No image pairs found in {data_dir}")
        
        print(f"[{mode.upper()}] Found {len(self.pairs)} image pairs in {data_dir}")
        
        # Normalization parameters (ImageNet)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    def _find_pairs(self) -> List[Tuple[Path, Path]]:
        """Find input-gt image pairs"""
        pairs = []
        
        # Check for input/gt directory structure
        input_dir = self.data_dir / 'input'
        gt_dir = self.data_dir / 'gt'
        
        if input_dir.exists() and gt_dir.exists():
            for input_path in sorted(input_dir.glob('*')):
                if input_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']:
                    gt_path = gt_dir / input_path.name
                    if gt_path.exists():
                        pairs.append((input_path, gt_path))
        
        # Check for snow/synthetic directory structure (Snow100K)
        snow_dir = self.data_dir / 'snow'
        synthetic_dir = self.data_dir / 'synthetic'
        
        if snow_dir.exists() and synthetic_dir.exists():
            for snow_path in sorted(snow_dir.glob('*')):
                if snow_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    gt_path = synthetic_dir / snow_path.name
                    if gt_path.exists():
                        pairs.append((snow_path, gt_path))
        
        # Check for paired files (xxx_snow.png, xxx_gt.png)
        if len(pairs) == 0:
            for f in sorted(self.data_dir.glob('*_snow.*')):
                gt_name = f.name.replace('_snow', '_gt')
                gt_path = self.data_dir / gt_name
                if gt_path.exists():
                    pairs.append((f, gt_path))
        
        # Check for all directory structure
        all_dir = self.data_dir / 'all'
        if all_dir.exists() and len(pairs) == 0:
            gt_subdir = all_dir / 'gt'
            if gt_subdir.exists():
                for subdir in all_dir.iterdir():
                    if subdir.is_dir() and subdir.name != 'gt':
                        for input_path in sorted(subdir.glob('*')):
                            gt_path = gt_subdir / input_path.name
                            if gt_path.exists():
                                pairs.append((input_path, gt_path))
        
        return pairs
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        input_path, gt_path = self.pairs[idx]
        
        # Load images
        input_img = Image.open(input_path).convert('RGB')
        gt_img = Image.open(gt_path).convert('RGB')
        
        # Ensure same size
        if input_img.size != gt_img.size:
            gt_img = gt_img.resize(input_img.size, Image.BILINEAR)
        
        # Apply augmentations
        if self.augment:
            input_img, gt_img = self._augment(input_img, gt_img)
        
        # Random crop for training, center crop for validation
        if self.mode == 'train':
            input_img, gt_img = self._random_crop(input_img, gt_img)
        elif self.mode == 'val':
            input_img, gt_img = self._center_crop(input_img, gt_img)
        # For test mode, use full resolution
        
        # Convert to tensor
        input_tensor = TF.to_tensor(input_img)
        gt_tensor = TF.to_tensor(gt_img)
        
        # Normalize
        if self.normalize:
            input_tensor = (input_tensor - self.mean) / self.std
            gt_tensor = (gt_tensor - self.mean) / self.std
        
        return {
            'input': input_tensor,
            'gt': gt_tensor,
            'name': input_path.stem,
            'input_path': str(input_path),
            'gt_path': str(gt_path)
        }
    
    def _augment(
        self, 
        input_img: Image.Image, 
        gt_img: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        """Apply synchronized augmentations"""
        
        # Random horizontal flip
        if random.random() > 0.5:
            input_img = TF.hflip(input_img)
            gt_img = TF.hflip(gt_img)
        
        # Random vertical flip
        if random.random() > 0.5:
            input_img = TF.vflip(input_img)
            gt_img = TF.vflip(gt_img)
        
        # Random rotation (90, 180, 270)
        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            input_img = TF.rotate(input_img, angle)
            gt_img = TF.rotate(gt_img, angle)
        
        return input_img, gt_img
    
    def _random_crop(
        self, 
        input_img: Image.Image, 
        gt_img: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        """Random crop for training"""
        w, h = input_img.size
        
        if w < self.patch_size or h < self.patch_size:
            # Resize if image is smaller than patch size
            scale = max(self.patch_size / w, self.patch_size / h) * 1.1
            new_w, new_h = int(w * scale), int(h * scale)
            input_img = input_img.resize((new_w, new_h), Image.BILINEAR)
            gt_img = gt_img.resize((new_w, new_h), Image.BILINEAR)
            w, h = new_w, new_h
        
        # Random crop position
        x = random.randint(0, w - self.patch_size)
        y = random.randint(0, h - self.patch_size)
        
        input_img = TF.crop(input_img, y, x, self.patch_size, self.patch_size)
        gt_img = TF.crop(gt_img, y, x, self.patch_size, self.patch_size)
        
        return input_img, gt_img
    
    def _center_crop(
        self, 
        input_img: Image.Image, 
        gt_img: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        """Center crop for validation"""
        w, h = input_img.size
        
        # Ensure divisible by 32 for U-Net
        crop_h = (h // 32) * 32
        crop_w = (w // 32) * 32
        
        if crop_h < self.patch_size:
            crop_h = self.patch_size
        if crop_w < self.patch_size:
            crop_w = self.patch_size
        
        input_img = TF.center_crop(input_img, (crop_h, crop_w))
        gt_img = TF.center_crop(gt_img, (crop_h, crop_w))
        
        return input_img, gt_img
    
    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Denormalize tensor for visualization"""
        if self.normalize:
            return tensor * self.std.to(tensor.device) + self.mean.to(tensor.device)
        return tensor


class Snow100KDataset(SnowDataset):
    """
    Snow100K Dataset
    
    Reference: https://sites.google.com/view/yunfuliu/desnownet
    
    Structure:
    Snow100K/
        train/
            Snow100K-S/  # Small snow particles
            Snow100K-M/  # Medium snow particles  
            Snow100K-L/  # Large snow particles
        test/
            Snow100K-S/
            Snow100K-M/
            Snow100K-L/
        gt/
            train/
            test/
    """
    
    def __init__(
        self,
        data_dir: str,
        subset: str = 'all',  # 'S', 'M', 'L', 'all'
        **kwargs
    ):
        self.subset = subset
        super().__init__(data_dir, **kwargs)
    
    def _find_pairs(self) -> List[Tuple[Path, Path]]:
        """Find Snow100K image pairs"""
        pairs = []
        
        # Determine subsets to load
        if self.subset == 'all':
            subsets = ['Snow100K-S', 'Snow100K-M', 'Snow100K-L']
        else:
            subsets = [f'Snow100K-{self.subset}']
        
        # Check for standard Snow100K structure
        for subset_name in subsets:
            input_dir = self.data_dir / self.mode / subset_name
            gt_dir = self.data_dir / 'gt' / self.mode
            
            if input_dir.exists() and gt_dir.exists():
                for input_path in sorted(input_dir.glob('*')):
                    if input_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                        # GT filename may differ slightly
                        gt_path = gt_dir / input_path.name
                        if not gt_path.exists():
                            gt_path = gt_dir / input_path.stem.replace('snow', 'gt') + input_path.suffix
                        if gt_path.exists():
                            pairs.append((input_path, gt_path))
        
        # Fallback to base class method if structure doesn't match
        if len(pairs) == 0:
            pairs = super()._find_pairs()
        
        return pairs


class SRRSDataset(SnowDataset):
    """
    SRRS (Snow Removing in Realistic Scenario) Dataset
    
    Structure:
    SRRS/
        train/
            input/
            gt/
        test/
            input/
            gt/
    """
    
    def _find_pairs(self) -> List[Tuple[Path, Path]]:
        """Find SRRS image pairs"""
        pairs = []
        
        input_dir = self.data_dir / self.mode / 'input'
        gt_dir = self.data_dir / self.mode / 'gt'
        
        if input_dir.exists() and gt_dir.exists():
            for input_path in sorted(input_dir.glob('*')):
                if input_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    gt_path = gt_dir / input_path.name
                    if gt_path.exists():
                        pairs.append((input_path, gt_path))
        
        if len(pairs) == 0:
            pairs = super()._find_pairs()
        
        return pairs


class CSDDataset(SnowDataset):
    """
    CSD (Comprehensive Snow Dataset) for testing
    
    Structure:
    CSD/
        CSD-2000/
            Test/
                Snow/
                Gt/
    """
    
    def _find_pairs(self) -> List[Tuple[Path, Path]]:
        """Find CSD image pairs"""
        pairs = []
        
        # Try CSD-2000 structure
        for subdir in ['CSD-2000']:
            snow_dir = self.data_dir / subdir / 'Test' / 'Snow'
            gt_dir = self.data_dir / subdir / 'Test' / 'Gt'
            
            if snow_dir.exists() and gt_dir.exists():
                for snow_path in sorted(snow_dir.glob('*')):
                    if snow_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                        gt_path = gt_dir / snow_path.name
                        if gt_path.exists():
                            pairs.append((snow_path, gt_path))
        
        if len(pairs) == 0:
            pairs = super()._find_pairs()
        
        return pairs


class MixedSnowDataset(Dataset):
    """
    Mixed dataset combining multiple snow datasets
    """
    
    def __init__(
        self,
        datasets: List[Dataset],
        weights: Optional[List[float]] = None
    ):
        self.datasets = datasets
        self.weights = weights or [1.0] * len(datasets)
        
        # Compute cumulative lengths for indexing
        self.cumulative_lengths = []
        total = 0
        for ds in datasets:
            total += len(ds)
            self.cumulative_lengths.append(total)
        
        self.total_length = total
    
    def __len__(self) -> int:
        return self.total_length
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Find which dataset this index belongs to
        for i, cum_len in enumerate(self.cumulative_lengths):
            if idx < cum_len:
                if i == 0:
                    local_idx = idx
                else:
                    local_idx = idx - self.cumulative_lengths[i - 1]
                return self.datasets[i][local_idx]
        
        raise IndexError(f"Index {idx} out of range")


def create_dataloader(
    data_dir: str,
    dataset_type: str = 'snow100k',
    mode: str = 'train',
    batch_size: int = 8,
    patch_size: int = 256,
    num_workers: int = 4,
    **kwargs
) -> DataLoader:
    """
    Factory function to create dataloaders
    
    Args:
        data_dir: Path to dataset
        dataset_type: 'snow100k', 'srrs', 'csd', 'generic'
        mode: 'train', 'val', 'test'
        batch_size: Batch size
        patch_size: Crop size for training
        num_workers: Number of data loading workers
    
    Returns:
        DataLoader instance
    """
    
    dataset_classes = {
        'snow100k': Snow100KDataset,
        'srrs': SRRSDataset,
        'csd': CSDDataset,
        'generic': SnowDataset
    }
    
    if dataset_type not in dataset_classes:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    dataset_class = dataset_classes[dataset_type]
    
    dataset = dataset_class(
        data_dir=data_dir,
        patch_size=patch_size,
        augment=(mode == 'train'),
        mode=mode,
        **kwargs
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(mode == 'train'),
        persistent_workers=(num_workers > 0)
    )
    
    return dataloader


def get_test_transforms(normalize: bool = True) -> Callable:
    """Get transforms for test/inference"""
    transforms = [T.ToTensor()]
    
    if normalize:
        transforms.append(
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        )
    
    return T.Compose(transforms)


def load_single_image(
    image_path: str,
    normalize: bool = True,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Load a single image for inference
    
    Args:
        image_path: Path to image
        normalize: Whether to normalize
        device: Target device
    
    Returns:
        Tensor of shape (1, 3, H, W)
    """
    img = Image.open(image_path).convert('RGB')
    
    # Ensure dimensions are divisible by 32
    w, h = img.size
    new_w = (w // 32) * 32
    new_h = (h // 32) * 32
    
    if new_w != w or new_h != h:
        img = img.resize((new_w, new_h), Image.BILINEAR)
    
    transform = get_test_transforms(normalize)
    tensor = transform(img).unsqueeze(0).to(device)
    
    return tensor


def denormalize_tensor(
    tensor: torch.Tensor,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]
) -> torch.Tensor:
    """Denormalize tensor for visualization"""
    mean = torch.tensor(mean).view(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(1, 3, 1, 1).to(tensor.device)
    return tensor * std + mean


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy image (H, W, C) in range [0, 255]"""
    if tensor.dim() == 4:
        tensor = tensor[0]  # Remove batch dimension
    
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    
    return img


def save_image(tensor: torch.Tensor, path: str, denormalize: bool = True):
    """Save tensor as image file"""
    if denormalize:
        tensor = denormalize_tensor(tensor)
    
    img = tensor_to_image(tensor)
    Image.fromarray(img).save(path)
