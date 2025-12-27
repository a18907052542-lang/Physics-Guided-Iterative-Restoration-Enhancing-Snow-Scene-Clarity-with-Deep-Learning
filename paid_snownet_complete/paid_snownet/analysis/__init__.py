"""
PAID-SnowNet Convergence Analysis

Tools for analyzing convergence properties:
- Empirical convergence tracking
- Theoretical bound verification
- Iteration effect analysis

Reference: Theorem 1, Eq(27) in paper
"""

from .convergence_analysis import (
    ConvergenceAnalyzer,
    TheoreticalAnalysis,
    ConvergenceTheorem,
    run_convergence_experiment,
    analyze_iteration_effect
)

__all__ = [
    'ConvergenceAnalyzer',
    'TheoreticalAnalysis',
    'ConvergenceTheorem',
    'run_convergence_experiment',
    'analyze_iteration_effect',
]
