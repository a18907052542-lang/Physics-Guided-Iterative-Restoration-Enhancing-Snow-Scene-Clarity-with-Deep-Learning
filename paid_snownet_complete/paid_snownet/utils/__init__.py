"""
PAID-SnowNet Utilities

Utility modules:
- Scattering physics: Rayleigh and Mie scattering models
- Visualization: Comprehensive visualization tools for analysis
"""

from .scattering_utils import (
    RayleighScattering,
    MieScattering,
    CombinedScatteringModel,
    compute_scattering_coefficient_map,
    compute_transmittance_map,
    riccati_bessel_jn,
    riccati_bessel_yn
)

from .visualization import (
    PhysicsParameterVisualizer,
    AttentionVisualizer,
    IterationVisualizer,
    ComparisonVisualizer,
    ConvergencePlotter,
    create_visualization_suite
)

__all__ = [
    # Scattering
    'RayleighScattering',
    'MieScattering',
    'CombinedScatteringModel',
    'compute_scattering_coefficient_map',
    'compute_transmittance_map',
    'riccati_bessel_jn',
    'riccati_bessel_yn',
    
    # Visualization
    'PhysicsParameterVisualizer',
    'AttentionVisualizer',
    'IterationVisualizer',
    'ComparisonVisualizer',
    'ConvergencePlotter',
    'create_visualization_suite',
]
