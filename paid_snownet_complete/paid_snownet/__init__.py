"""
PAID-SnowNet: Physics-Aware Iterative Denoising Network for Snow Removal

A deep learning framework that combines Mie/Rayleigh scattering physics with 
iterative optimization for high-quality snow scene restoration.

Reference: Visual Computer Journal (SCI Q3)

Modules:
    - models: Neural network architectures (SPM, SIRM, main model)
    - losses: Multi-scale loss functions
    - utils: Scattering physics, visualization tools
    - analysis: Convergence analysis tools
    - configs: Configuration management
    - data: Dataset utilities
    - scripts: Training, evaluation, inference

Example Usage:
    >>> from paid_snownet import create_model
    >>> model = create_model('base')
    >>> output = model(input_image)
    
    >>> # With physics parameters
    >>> output, params = model(input_image)
    >>> print(params.keys())  # scattering, transmission, occlusion, depth
"""

__version__ = '1.0.0'
__author__ = 'PAID-SnowNet Team'
__license__ = 'MIT'

from .models.paid_snownet import (
    PAIDSnowNet,
    PAIDSnowNetLightweight,
    PAIDSnowNetDeep,
    create_model,
    count_parameters,
    model_summary
)

from .models.spm_module import SPMModule, LightweightSPM
from .models.sirm_module import SIRMModule, LightweightSIRM, DeepSIRM
from .models.degradation_ops import DegradationOperator
from .models.attention_fusion import AdaptiveFusion, MultiScaleAttentionFusion
from .models.physics_weight import PhysicsAwareWeight, AdaGradOptimizer

from .losses.multi_scale_loss import (
    MultiScaleLoss,
    CharbonnierLoss,
    VGGPerceptualLoss,
    EdgeLoss,
    SSIMLoss
)

from .utils.scattering_utils import (
    RayleighScattering,
    MieScattering,
    CombinedScatteringModel
)

from .utils.visualization import (
    PhysicsParameterVisualizer,
    AttentionVisualizer,
    IterationVisualizer,
    ComparisonVisualizer,
    ConvergencePlotter,
    create_visualization_suite
)

from .analysis.convergence_analysis import (
    ConvergenceAnalyzer,
    TheoreticalAnalysis
)

from .configs.config import (
    PAIDSnowNetConfig,
    TrainingConfig,
    DataConfig,
    get_default_config
)

from .data.dataset import (
    Snow100KDataset,
    SnowDatasetSynthetic,
    get_train_transforms,
    get_val_transforms
)

__all__ = [
    # Main models
    'PAIDSnowNet',
    'PAIDSnowNetLightweight', 
    'PAIDSnowNetDeep',
    'create_model',
    'count_parameters',
    'model_summary',
    
    # Module components
    'SPMModule',
    'LightweightSPM',
    'SIRMModule',
    'LightweightSIRM',
    'DeepSIRM',
    'DegradationOperator',
    'AdaptiveFusion',
    'MultiScaleAttentionFusion',
    'PhysicsAwareWeight',
    'AdaGradOptimizer',
    
    # Losses
    'MultiScaleLoss',
    'CharbonnierLoss',
    'VGGPerceptualLoss',
    'EdgeLoss',
    'SSIMLoss',
    
    # Physics
    'RayleighScattering',
    'MieScattering',
    'CombinedScatteringModel',
    
    # Visualization
    'PhysicsParameterVisualizer',
    'AttentionVisualizer',
    'IterationVisualizer',
    'ComparisonVisualizer',
    'ConvergencePlotter',
    'create_visualization_suite',
    
    # Analysis
    'ConvergenceAnalyzer',
    'TheoreticalAnalysis',
    
    # Config
    'PAIDSnowNetConfig',
    'TrainingConfig',
    'DataConfig',
    'get_default_config',
    
    # Data
    'Snow100KDataset',
    'SnowDatasetSynthetic',
    'get_train_transforms',
    'get_val_transforms',
]
