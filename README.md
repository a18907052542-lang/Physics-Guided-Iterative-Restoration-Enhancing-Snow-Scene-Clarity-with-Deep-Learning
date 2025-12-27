# PAID-SnowNet: Physics-Aware Iterative Denoising Network for Snow Removal

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.10+](https://img.shields.io/badge/pytorch-1.10+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of **PAID-SnowNet** (Physics-Aware Iterative Denoising Network) for snow scene restoration, published in *The Visual Computer* (SCI Q3).

## 📋 Overview

PAID-SnowNet is a deep learning framework that combines **Mie/Rayleigh scattering physics** with **iterative optimization** for high-quality snow removal. The method achieves state-of-the-art performance on benchmark datasets while maintaining physical interpretability.

### Key Features

- **Physics-Guided Design**: Incorporates Mie and Rayleigh scattering models (Eq. 1-7) to accurately model light interaction with snow particles
- **Multi-Scale Processing**: 3×3, 5×5, 7×7 parallel branches for comprehensive feature extraction (Table 3)
- **Iterative Restoration**: SIRM module with adaptive step sizes and U-Net refinement (Algorithm 1)
- **Convergence Guarantee**: Theoretical convergence analysis with O(1/√T) rate (Theorem 1)
- **Comprehensive Evaluation**: Extensive ablation studies and comparisons (Tables 3-9)

## 🏗️ Architecture

```
Input Image (Y)
      │
      ▼
┌─────────────────────────────────────────┐
│           SPM (Snow Physics Module)      │
│  ┌─────────┬─────────┬─────────┐        │
│  │  3×3    │   5×5   │   7×7   │        │
│  │ Branch  │  Branch │  Branch │        │
│  └────┬────┴────┬────┴────┬────┘        │
│       └─────────┼─────────┘              │
│                 ▼                        │
│       Adaptive Fusion (Eq. 8-9)          │
│                 │                        │
│    ┌────────────┼────────────┐          │
│    ▼            ▼            ▼          │
│    β          t(x,y)       M(x,y)       │
│ Scattering  Transmission  Occlusion     │
└─────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────┐
│          SIRM (Iterative Restoration)    │
│                                          │
│  for t = 0 to T-1:                       │
│    g_t = ∇L(X^t)           (Eq. 19)     │
│    η_t = AdaGrad(g_t, w)   (Eq. 21-22)  │
│    X^(t+1/2) = X^t - η_t·g_t (Eq. 24)   │
│    X^(t+1) = UNet(X^(t+1/2)) (Eq. 25)   │
│                                          │
└─────────────────────────────────────────┘
      │
      ▼
Output Image (X^T)
```

## 📊 Performance

### Quantitative Results (Snow100K Dataset)

| Method | PSNR (dB) | SSIM | Parameters | Inference Time |
|--------|-----------|------|------------|----------------|
| DesnowNet | 27.89 | 0.8834 | 12.3M | 45ms |
| JSTASR | 28.56 | 0.9012 | 28.7M | 123ms |
| DDMSNet | 29.34 | 0.9156 | 35.2M | 156ms |
| TransWeather | 29.78 | 0.9234 | 41.8M | 178ms |
| **PAID-SnowNet (Ours)** | **30.85** | **0.9412** | 45.2M | 138ms |

### Ablation Studies

**Iteration Count (Table 4)**:
| Iterations | PSNR | Time |
|------------|------|------|
| T=1 | 27.8dB | 45ms |
| T=2 | 29.6dB | 92ms |
| **T=3** | **30.85dB** | **138ms** |
| T=4 | 30.89dB | 185ms |

**Multi-Scale Branches (Table 3)**:
| Configuration | PSNR |
|---------------|------|
| 3×3 only | 29.12dB |
| 5×5 only | 29.45dB |
| 7×7 only | 28.89dB |
| **3×3 + 5×5 + 7×7** | **30.85dB** |

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/your-repo/paid-snownet.git
cd paid-snownet

# Create environment
conda create -n paid_snownet python=3.8
conda activate paid_snownet

# Install dependencies
pip install torch torchvision torchaudio
pip install numpy scipy matplotlib pillow tqdm tensorboard
pip install scikit-image opencv-python
```

### Inference

```python
from paid_snownet import create_model
import torch
from PIL import Image
import torchvision.transforms as T

# Load model
model = create_model('base')
model.load_state_dict(torch.load('checkpoints/paid_snownet_best.pth'))
model.eval()

# Load and preprocess image
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
image = transform(Image.open('snow_image.jpg')).unsqueeze(0)

# Inference
with torch.no_grad():
    output = model(image)
    
# Save result
T.ToPILImage()(output.squeeze().clamp(0, 1)).save('restored.jpg')
```

### Training

```bash
# Train base model
python scripts/train.py \
    --data_dir /path/to/snow100k \
    --output_dir ./experiments/base \
    --model_type base \
    --epochs 200 \
    --batch_size 8 \
    --lr 1e-4

# Train with custom config
python scripts/train.py --config configs/custom_config.yaml
```

### Evaluation

```bash
# Evaluate on test set
python scripts/evaluate.py \
    --checkpoint ./experiments/base/best_model.pth \
    --data_dir /path/to/snow100k/test \
    --output_dir ./results
```

## 📁 Project Structure

```
paid_snownet/
├── __init__.py              # Package initialization
├── models/
│   ├── __init__.py
│   ├── paid_snownet.py      # Main model (PAIDSnowNet)
│   ├── spm_module.py        # Snow Physics Module
│   ├── sirm_module.py       # Iterative Restoration Module
│   ├── degradation_ops.py   # Degradation operators
│   ├── attention_fusion.py  # Multi-scale attention
│   └── physics_weight.py    # Physics-aware weighting
├── losses/
│   ├── __init__.py
│   └── multi_scale_loss.py  # Multi-scale supervision
├── utils/
│   ├── __init__.py
│   ├── scattering_utils.py  # Mie/Rayleigh physics
│   └── visualization.py     # Visualization tools
├── analysis/
│   ├── __init__.py
│   └── convergence_analysis.py  # Convergence tools
├── configs/
│   ├── __init__.py
│   └── config.py            # Configuration classes
├── data/
│   ├── __init__.py
│   └── dataset.py           # Dataset utilities
├── scripts/
│   ├── train.py             # Training script
│   ├── evaluate.py          # Evaluation script
│   └── inference.py         # Inference script
└── README.md
```

## 🔧 Model Variants

| Variant | Parameters | PSNR | Speed | Use Case |
|---------|------------|------|-------|----------|
| **Base** | 45.2M | 30.85dB | 138ms | Balanced |
| **Lightweight** | 18.6M | 29.42dB | 87ms | Real-time |
| **Deep** | 78.4M | 30.91dB | 312ms | Quality-first |

```python
# Create different variants
from paid_snownet import create_model

model_base = create_model('base')        # Default
model_light = create_model('lightweight') # Fast
model_deep = create_model('deep')         # High quality
```

## 📐 Key Equations

### Scattering Physics (Eq. 1-7)

**Rayleigh Scattering** (d << λ):
```
I_s/I_0 = (8π⁴a⁶/λ⁴r²) · ((n²-1)/(n²+2))² · (1+cos²θ)/2
```

**Mie Scattering** (d ≈ λ):
```
S₁(θ) = Σ (2n+1)/(n(n+1)) · [aₙπₙ(cosθ) + bₙτₙ(cosθ)]
```

### Adaptive Fusion (Eq. 8-9)

```
[α, β, γ] = Softmax(Wₐ · GAP([F₁, F₂, F₃]) + bₐ)
F_fused = α·F₁ + β·F₂ + γ·F₃
```

### Iterative Restoration (Eq. 24-26)

```
X^(t+1/2) = X^t - η^t · ∇L(X^t)     # Physics constraint
X^(t+1) = U(X^(t+1/2); θ)            # U-Net refinement
```

### Physics-Aware Step Size (Eq. 21-23)

```
G_k = G_{k-1} + g_k²                 # Accumulated gradient
η_k(x,y) = η₀/√(G_k + ε) · w(x,y)   # AdaGrad step
w(x,y) = αw·M(x,y) + βw·t(x,y) + γw  # Physics weight
```

## 📈 Visualization

```python
from paid_snownet import create_visualization_suite

# Create visualizers
vis = create_visualization_suite('./visualizations')

# Visualize physics parameters
vis['physics'].visualize_all_parameters(params, input_img)

# Visualize iteration progress
vis['iteration'].visualize_iteration_progress(iterations, gt)

# Plot convergence curves
vis['convergence'].plot_training_curves(train_loss, val_psnr=psnrs)
```

## 📝 Citation

```bibtex
@article{paid_snownet2025,
  title={PAID-SnowNet: Physics-Aware Iterative Denoising Network for Snow Scene Restoration},
  author={Author Names},
  journal={The Visual Computer},
  year={2025},
  publisher={Springer}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Snow100K dataset creators
- PyTorch team
- VGG network for perceptual loss

## 📧 Contact

For questions or issues, please open an issue on GitHub
