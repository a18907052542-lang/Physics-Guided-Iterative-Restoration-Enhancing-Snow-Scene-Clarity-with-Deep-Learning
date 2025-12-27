"""
PAID-SnowNet Model Components

This module contains all neural network architectures:
- SPM (Snow Physics Module): Multi-scale feature extraction and physics estimation
- SIRM (Snow Iterative Restoration Module): Iterative optimization with U-Net refinement
- Degradation operators: Forward model implementation
- Attention fusion: Multi-scale feature fusion
- Physics-aware weighting: Adaptive step size computation
"""

from .paid_snownet import (
    PAIDSnowNet,
    PAIDSnowNetLightweight,
    PAIDSnowNetDeep,
    create_model,
    count_parameters,
    model_summary
)

from .spm_module import (
    SPMModule,
    LightweightSPM,
    MultiScaleFeatureExtractor,
    ScatteringEstimator,
    TransmissionEstimator,
    OcclusionMaskEstimator,
    DepthEstimator,
    PSFEstimator
)

from .sirm_module import (
    SIRMModule,
    LightweightSIRM,
    DeepSIRM,
    UNet,
    GradientComputation
)

from .degradation_ops import (
    DegradationOperator,
    BlurOperator,
    ScatteringOperator,
    OcclusionOperator,
    SnowReflectionModel,
    SpatiallyVaryingDegradation
)

from .attention_fusion import (
    AdaptiveFusion,
    MultiScaleAttentionFusion,
    ScaleAwareAttention,
    CrossScaleAttention,
    HierarchicalFusion
)

from .physics_weight import (
    PhysicsAwareWeight,
    AdaGradOptimizer,
    MomentumAdaGrad,
    SpatiallyVaryingStepSize,
    IterationAwareStepSize,
    GradientNormalization
)

__all__ = [
    # Main models
    'PAIDSnowNet',
    'PAIDSnowNetLightweight',
    'PAIDSnowNetDeep',
    'create_model',
    'count_parameters',
    'model_summary',
    
    # SPM
    'SPMModule',
    'LightweightSPM',
    'MultiScaleFeatureExtractor',
    'ScatteringEstimator',
    'TransmissionEstimator',
    'OcclusionMaskEstimator',
    'DepthEstimator',
    'PSFEstimator',
    
    # SIRM
    'SIRMModule',
    'LightweightSIRM',
    'DeepSIRM',
    'UNet',
    'GradientComputation',
    
    # Degradation
    'DegradationOperator',
    'BlurOperator',
    'ScatteringOperator',
    'OcclusionOperator',
    'SnowReflectionModel',
    'SpatiallyVaryingDegradation',
    
    # Attention
    'AdaptiveFusion',
    'MultiScaleAttentionFusion',
    'ScaleAwareAttention',
    'CrossScaleAttention',
    'HierarchicalFusion',
    
    # Physics weighting
    'PhysicsAwareWeight',
    'AdaGradOptimizer',
    'MomentumAdaGrad',
    'SpatiallyVaryingStepSize',
    'IterationAwareStepSize',
    'GradientNormalization',
]
