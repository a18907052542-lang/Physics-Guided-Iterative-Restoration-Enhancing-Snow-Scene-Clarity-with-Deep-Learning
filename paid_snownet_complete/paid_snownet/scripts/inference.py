"""
PAID-SnowNet Inference Script
==============================

Fast inference for snow removal on single images or directories.
Supports batch processing, video frames, and real-time preview.
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Optional, List, Tuple
import glob

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.paid_snownet import create_model
from data.dataset import denormalize_tensor


class SnowRemover:
    """
    PAID-SnowNet inference wrapper
    
    Supports:
    - Single image processing
    - Batch processing
    - Directory processing
    - Test-time augmentation
    """
    
    def __init__(
        self,
        model_path: str,
        model_type: str = 'base',
        device: str = 'cuda',
        use_half: bool = False
    ):
        """
        Initialize snow remover
        
        Args:
            model_path: Path to trained checkpoint
            model_type: 'base', 'lightweight', or 'deep'
            device: 'cuda' or 'cpu'
            use_half: Use FP16 inference for faster speed
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.use_half = use_half and device == 'cuda'
        
        # Load model
        self._load_model(model_path, model_type)
        
        # Normalization parameters
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        
        if self.use_half:
            self.mean = self.mean.half()
            self.std = self.std.half()
    
    def _load_model(self, model_path: str, model_type: str):
        """Load model from checkpoint"""
        print(f"Loading model from: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Try to get model type from checkpoint
        if 'config' in checkpoint:
            try:
                model_type = checkpoint['config'].model.model_type
            except:
                pass
        
        self.model = create_model(
            model_type=model_type,
            in_channels=3,
            out_channels=3
        ).to(self.device)
        
        # Load weights
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Handle DataParallel wrapped models
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        self.model.load_state_dict(new_state_dict)
        self.model.eval()
        
        if self.use_half:
            self.model.half()
        
        print(f"Model loaded: {model_type}")
    
    def _preprocess(self, image: Image.Image) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Preprocess image for inference
        
        Args:
            image: PIL Image
        
        Returns:
            Tuple of (normalized tensor, original size)
        """
        # Store original size
        original_size = image.size  # (W, H)
        
        # Ensure RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to tensor [0, 1]
        img_np = np.array(image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device)
        
        if self.use_half:
            tensor = tensor.half()
        
        # Normalize
        tensor = (tensor - self.mean) / self.std
        
        # Pad to multiple of 32 for U-Net
        _, _, h, w = tensor.shape
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        
        if pad_h > 0 or pad_w > 0:
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')
        
        return tensor, original_size, (h, w)
    
    def _postprocess(
        self,
        tensor: torch.Tensor,
        original_size: Tuple[int, int],
        padded_size: Tuple[int, int]
    ) -> Image.Image:
        """
        Postprocess output tensor to image
        
        Args:
            tensor: Model output tensor
            original_size: Original image size (W, H)
            padded_size: Size before padding (H, W)
        
        Returns:
            PIL Image
        """
        # Remove padding
        h, w = padded_size
        tensor = tensor[:, :, :h, :w]
        
        # Denormalize
        tensor = tensor * self.std + self.mean
        
        # Clamp and convert
        tensor = torch.clamp(tensor, 0, 1)
        
        # Convert to numpy
        img_np = tensor[0].cpu().float().numpy()
        img_np = (img_np.transpose(1, 2, 0) * 255).astype(np.uint8)
        
        image = Image.fromarray(img_np)
        
        # Resize to original size if different
        if image.size != original_size:
            image = image.resize(original_size, Image.BILINEAR)
        
        return image
    
    @torch.no_grad()
    def remove_snow(
        self,
        image: Image.Image,
        use_tta: bool = False
    ) -> Image.Image:
        """
        Remove snow from a single image
        
        Args:
            image: Input PIL Image
            use_tta: Use test-time augmentation
        
        Returns:
            Restored PIL Image
        """
        # Preprocess
        tensor, original_size, padded_size = self._preprocess(image)
        
        if use_tta:
            # Test-time augmentation: average predictions from multiple augmentations
            outputs = []
            
            # Original
            out = self.model(tensor)
            outputs.append(out)
            
            # Horizontal flip
            out_hflip = self.model(torch.flip(tensor, dims=[3]))
            outputs.append(torch.flip(out_hflip, dims=[3]))
            
            # Vertical flip
            out_vflip = self.model(torch.flip(tensor, dims=[2]))
            outputs.append(torch.flip(out_vflip, dims=[2]))
            
            # Average
            output = torch.stack(outputs).mean(dim=0)
        else:
            output = self.model(tensor)
        
        # Postprocess
        result = self._postprocess(output, original_size, padded_size)
        
        return result
    
    def process_file(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        use_tta: bool = False
    ) -> Image.Image:
        """
        Process a single image file
        
        Args:
            input_path: Path to input image
            output_path: Optional path to save output
            use_tta: Use test-time augmentation
        
        Returns:
            Restored PIL Image
        """
        # Load image
        image = Image.open(input_path).convert('RGB')
        
        # Process
        start_time = time.time()
        result = self.remove_snow(image, use_tta=use_tta)
        elapsed = time.time() - start_time
        
        print(f"Processed {Path(input_path).name} in {elapsed:.2f}s")
        
        # Save if output path provided
        if output_path:
            result.save(output_path)
            print(f"Saved: {output_path}")
        
        return result
    
    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        extensions: List[str] = ['png', 'jpg', 'jpeg', 'bmp'],
        use_tta: bool = False
    ) -> List[str]:
        """
        Process all images in a directory
        
        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            extensions: List of valid extensions
            use_tta: Use test-time augmentation
        
        Returns:
            List of output paths
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all images
        image_files = []
        for ext in extensions:
            image_files.extend(input_path.glob(f'*.{ext}'))
            image_files.extend(input_path.glob(f'*.{ext.upper()}'))
        
        image_files = sorted(set(image_files))
        print(f"Found {len(image_files)} images to process")
        
        output_paths = []
        total_time = 0
        
        for img_path in tqdm(image_files, desc='Processing'):
            out_path = output_path / f'{img_path.stem}_restored{img_path.suffix}'
            
            start_time = time.time()
            self.process_file(str(img_path), str(out_path), use_tta=use_tta)
            total_time += time.time() - start_time
            
            output_paths.append(str(out_path))
        
        avg_time = total_time / len(image_files) if image_files else 0
        print(f"\nProcessed {len(image_files)} images")
        print(f"Average time per image: {avg_time:.2f}s")
        print(f"Total time: {total_time:.1f}s")
        
        return output_paths
    
    def process_video_frames(
        self,
        input_dir: str,
        output_dir: str,
        frame_pattern: str = 'frame_%05d.png'
    ):
        """
        Process video frames (extracted with ffmpeg)
        
        Args:
            input_dir: Directory with input frames
            output_dir: Directory for output frames
            frame_pattern: Frame naming pattern
        """
        return self.process_directory(input_dir, output_dir)
    
    def benchmark(
        self,
        image_sizes: List[Tuple[int, int]] = [(256, 256), (512, 512), (1024, 1024), (1920, 1080)],
        num_iterations: int = 10,
        warmup: int = 3
    ):
        """
        Benchmark inference speed at different resolutions
        
        Args:
            image_sizes: List of (H, W) sizes to test
            num_iterations: Number of iterations per size
            warmup: Warmup iterations
        """
        print("\nBenchmarking inference speed...")
        print("=" * 60)
        
        results = []
        
        for h, w in image_sizes:
            # Create dummy input
            dummy = torch.randn(1, 3, h, w).to(self.device)
            if self.use_half:
                dummy = dummy.half()
            dummy = (dummy - self.mean) / self.std
            
            # Pad to multiple of 32
            pad_h = (32 - h % 32) % 32
            pad_w = (32 - w % 32) % 32
            if pad_h > 0 or pad_w > 0:
                dummy = F.pad(dummy, (0, pad_w, 0, pad_h), mode='reflect')
            
            # Warmup
            for _ in range(warmup):
                with torch.no_grad():
                    _ = self.model(dummy)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Benchmark
            times = []
            for _ in range(num_iterations):
                start = time.time()
                with torch.no_grad():
                    _ = self.model(dummy)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                times.append(time.time() - start)
            
            avg_time = np.mean(times) * 1000  # ms
            std_time = np.std(times) * 1000
            fps = 1000 / avg_time
            
            results.append({
                'resolution': f'{w}x{h}',
                'time_ms': avg_time,
                'std_ms': std_time,
                'fps': fps
            })
            
            print(f"{w}x{h}: {avg_time:.1f} ± {std_time:.1f} ms ({fps:.1f} FPS)")
        
        print("=" * 60)
        
        return results


def create_side_by_side(
    input_path: str,
    output_path: str,
    save_path: str
):
    """Create side-by-side comparison image"""
    input_img = Image.open(input_path)
    output_img = Image.open(output_path)
    
    # Create combined image
    total_width = input_img.width + output_img.width + 10
    max_height = max(input_img.height, output_img.height)
    
    combined = Image.new('RGB', (total_width, max_height), (255, 255, 255))
    combined.paste(input_img, (0, 0))
    combined.paste(output_img, (input_img.width + 10, 0))
    
    combined.save(save_path)
    print(f"Comparison saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='PAID-SnowNet Inference')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                        help='Input image or directory')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path (file or directory)')
    parser.add_argument('--model_type', type=str, default='base',
                        choices=['base', 'lightweight', 'deep'],
                        help='Model type')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--half', action='store_true',
                        help='Use FP16 inference')
    parser.add_argument('--tta', action='store_true',
                        help='Use test-time augmentation')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run speed benchmark')
    parser.add_argument('--comparison', action='store_true',
                        help='Save side-by-side comparison')
    
    args = parser.parse_args()
    
    # Create snow remover
    remover = SnowRemover(
        model_path=args.model,
        model_type=args.model_type,
        device=args.device,
        use_half=args.half
    )
    
    # Run benchmark if requested
    if args.benchmark:
        remover.benchmark()
        return
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single image processing
        if args.output:
            output_path = args.output
        else:
            output_path = str(input_path.parent / f'{input_path.stem}_restored{input_path.suffix}')
        
        remover.process_file(str(input_path), output_path, use_tta=args.tta)
        
        # Create comparison if requested
        if args.comparison:
            comp_path = str(Path(output_path).parent / f'{Path(output_path).stem}_comparison.png')
            create_side_by_side(str(input_path), output_path, comp_path)
    
    elif input_path.is_dir():
        # Directory processing
        if args.output:
            output_dir = args.output
        else:
            output_dir = str(input_path.parent / f'{input_path.name}_restored')
        
        remover.process_directory(str(input_path), output_dir, use_tta=args.tta)
    
    else:
        print(f"Error: {args.input} does not exist")
        sys.exit(1)


if __name__ == '__main__':
    main()
