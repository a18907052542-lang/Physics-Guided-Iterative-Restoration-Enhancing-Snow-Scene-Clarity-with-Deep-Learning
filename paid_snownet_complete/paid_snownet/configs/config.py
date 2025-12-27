"""
PAID-SnowNet Configuration
==========================

Centralized configuration for model architecture, training, and evaluation.
Based on paper specifications and Table 3-9 ablation studies.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import os


@dataclass
class ScatteringConfig:
    """Scattering physics parameters (Section 3.1)"""
    wavelength: float = 550e-9  # λ = 550nm (green light)
    refractive_index: complex = 1.31 + 0j  # Ice/snow refractive index
    snow_albedo: float = 0.9  # ρ_s high reflectivity
    particle_size_range: Tuple[float, float] = (50e-6, 5e-3)  # 50μm to 5mm
    mie_threshold: float = 0.1  # x = 2πa/λ threshold for Mie vs Rayleigh
    max_mie_terms: int = 50  # Maximum terms in Mie series


@dataclass
class SPMConfig:
    """Scattering Parameter Module configuration (Section 3.2, Figure 2)"""
    # Multi-scale branches (Table 3)
    kernel_sizes: List[int] = field(default_factory=lambda: [3, 5, 7])
    base_channels: int = 64
    num_layers: int = 4
    
    # Feature extraction
    use_batch_norm: bool = True
    activation: str = 'relu'
    dropout_rate: float = 0.0
    
    # Parameter estimation heads
    scattering_channels: int = 32
    transmission_channels: int = 32
    occlusion_channels: int = 32
    depth_channels: int = 32
    psf_channels: int = 16


@dataclass
class SIRMConfig:
    """Snow Image Restoration Module configuration (Section 3.3)"""
    # Iteration parameters (Table 4)
    num_iterations: int = 3  # T = 3 optimal
    
    # U-Net architecture (Table 6)
    unet_channels: int = 64  # Base channels
    unet_layers: int = 4  # 4-layer optimal
    use_skip_connections: bool = True
    
    # Regularization (Eq 17-18)
    lambda_sparse: float = 0.01  # λ₁ for sparsity
    lambda_smooth: float = 0.001  # λ₂ for smoothness
    charbonnier_epsilon: float = 1e-3


@dataclass
class OptimizerConfig:
    """AdaGrad optimizer configuration (Section 3.3, Table 5)"""
    optimizer_type: str = 'adagrad'  # 'adagrad', 'sgd', 'adam'
    initial_lr: float = 0.1  # η₀
    adagrad_epsilon: float = 1e-8  # ε for numerical stability
    
    # Physics-aware weighting (Eq 23)
    alpha_weight: float = 0.5  # α_w for occlusion
    beta_weight: float = 0.3  # β_w for transmission
    gamma_weight: float = 0.2  # γ_w base weight
    
    # Momentum (optional)
    use_momentum: bool = False
    momentum: float = 0.9


@dataclass
class LossConfig:
    """Loss function configuration (Section 3.4)"""
    # Multi-scale supervision (Eq 29-30)
    use_multi_scale: bool = True
    iteration_weight_decay: float = 0.5  # w_t = 0.5^(T-t)
    
    # Loss weights
    reconstruction_weight: float = 1.0
    perceptual_weight: float = 0.5  # α
    edge_weight: float = 0.1  # β
    ssim_weight: float = 0.2
    color_weight: float = 0.1
    
    # Perceptual loss (VGG-19)
    vgg_layers: List[str] = field(default_factory=lambda: [
        'relu1_2', 'relu2_2', 'relu3_4', 'relu4_4'
    ])
    vgg_weights: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.4, 0.3])
    
    # Charbonnier loss
    charbonnier_epsilon: float = 1e-3


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Dataset
    dataset_name: str = 'snow100k'
    train_dir: str = './data/Snow100K/train'
    val_dir: str = './data/Snow100K/val'
    test_dir: str = './data/Snow100K/test'
    
    # Data augmentation
    patch_size: int = 256
    use_flip: bool = True
    use_rotation: bool = True
    use_color_jitter: bool = False
    
    # Training parameters
    batch_size: int = 8
    num_epochs: int = 200
    num_workers: int = 4
    
    # Optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    optimizer: str = 'adamw'
    
    # Scheduler
    scheduler: str = 'cosine'  # 'cosine', 'step', 'plateau'
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # Checkpointing
    save_freq: int = 10
    val_freq: int = 5
    log_freq: int = 100
    
    # Mixed precision
    use_amp: bool = True
    gradient_clip: float = 1.0


@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    # Metrics
    compute_psnr: bool = True
    compute_ssim: bool = True
    compute_lpips: bool = True
    compute_fid: bool = False
    
    # Snow denoise metric weights (Eq 31, Table 9)
    metric_weights: dict = field(default_factory=lambda: {
        'pixel': 0.30,  # α
        'ssim': 0.25,   # β
        'perceptual': 0.15,  # γ
        'edge': 0.20,   # δ
        'color': 0.10   # η
    })
    
    # Inference
    use_tta: bool = False  # Test-time augmentation
    ensemble_count: int = 1


@dataclass
class ModelConfig:
    """Complete model configuration"""
    model_type: str = 'base'  # 'base', 'lightweight', 'deep'
    in_channels: int = 3
    out_channels: int = 3
    
    # Sub-module configs
    scattering: ScatteringConfig = field(default_factory=ScatteringConfig)
    spm: SPMConfig = field(default_factory=SPMConfig)
    sirm: SIRMConfig = field(default_factory=SIRMConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    loss: LossConfig = field(default_factory=LossConfig)


@dataclass
class Config:
    """Master configuration"""
    # Experiment
    experiment_name: str = 'paid_snownet_base'
    output_dir: str = './outputs'
    seed: int = 42
    device: str = 'cuda'
    
    # Sub-configs
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    def __post_init__(self):
        """Create output directories"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'logs'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'visualizations'), exist_ok=True)


def get_base_config() -> Config:
    """Get base model configuration (Table 3-6 optimal settings)"""
    return Config(
        experiment_name='paid_snownet_base',
        model=ModelConfig(
            model_type='base',
            spm=SPMConfig(
                kernel_sizes=[3, 5, 7],
                base_channels=64,
                num_layers=4
            ),
            sirm=SIRMConfig(
                num_iterations=3,
                unet_channels=64,
                unet_layers=4
            )
        )
    )


def get_lightweight_config() -> Config:
    """Get lightweight model configuration for real-time applications"""
    return Config(
        experiment_name='paid_snownet_lightweight',
        model=ModelConfig(
            model_type='lightweight',
            spm=SPMConfig(
                kernel_sizes=[3, 5],
                base_channels=32,
                num_layers=3
            ),
            sirm=SIRMConfig(
                num_iterations=2,
                unet_channels=32,
                unet_layers=3
            )
        ),
        training=TrainingConfig(
            batch_size=16,
            patch_size=192
        )
    )


def get_deep_config() -> Config:
    """Get deep model configuration for maximum quality"""
    return Config(
        experiment_name='paid_snownet_deep',
        model=ModelConfig(
            model_type='deep',
            spm=SPMConfig(
                kernel_sizes=[3, 5, 7],
                base_channels=128,
                num_layers=5
            ),
            sirm=SIRMConfig(
                num_iterations=4,
                unet_channels=128,
                unet_layers=5
            )
        ),
        training=TrainingConfig(
            batch_size=4,
            patch_size=256
        )
    )


def get_config(config_name: str = 'base') -> Config:
    """Factory function to get configuration by name"""
    configs = {
        'base': get_base_config,
        'lightweight': get_lightweight_config,
        'deep': get_deep_config
    }
    
    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(configs.keys())}")
    
    return configs[config_name]()


# Ablation study configurations (reproducing paper tables)
def get_ablation_configs() -> dict:
    """Get configurations for ablation studies"""
    
    ablations = {}
    
    # Table 3: Multi-scale branch ablation
    for kernels in [[3], [5], [7], [3, 5], [3, 7], [5, 7], [3, 5, 7]]:
        name = f"branch_{'_'.join(map(str, kernels))}"
        cfg = get_base_config()
        cfg.experiment_name = f"ablation_{name}"
        cfg.model.spm.kernel_sizes = kernels
        ablations[name] = cfg
    
    # Table 4: Iteration count ablation
    for T in [1, 2, 3, 4, 5]:
        name = f"iter_{T}"
        cfg = get_base_config()
        cfg.experiment_name = f"ablation_{name}"
        cfg.model.sirm.num_iterations = T
        ablations[name] = cfg
    
    # Table 5: Optimizer ablation
    for opt in ['sgd', 'adam', 'adagrad']:
        name = f"opt_{opt}"
        cfg = get_base_config()
        cfg.experiment_name = f"ablation_{name}"
        cfg.model.optimizer.optimizer_type = opt
        ablations[name] = cfg
    
    # Table 6: U-Net depth ablation
    for layers, channels in [(3, 32), (3, 64), (4, 64), (4, 128), (5, 128)]:
        name = f"unet_{layers}l_{channels}c"
        cfg = get_base_config()
        cfg.experiment_name = f"ablation_{name}"
        cfg.model.sirm.unet_layers = layers
        cfg.model.sirm.unet_channels = channels
        ablations[name] = cfg
    
    return ablations
