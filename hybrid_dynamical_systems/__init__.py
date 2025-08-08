"""
Hybrid Dynamical Systems Framework
==================================

A comprehensive implementation of hybrid dynamical systems inspired by 
Ryan David Oates' work on physics-informed neural networks and hybrid modeling.

The framework implements the core expression:
Ψ(x) = ∫[ α(t) S(x) + [1−α(t)] N(x) + w_cross(S(m₁)N(m₂)−S(m₂)N(m₁)) ]
       × exp[−(λ₁ R_cognitive + λ₂ R_efficiency)] × P(H|E, β) dt

Where:
- α(t): Time-dependent weight blending symbolic S(x) and neural N(x)
- λ₁(t): Penalty weight for cognitive implausibility  
- λ₂(t): Penalty weight for computational cost/efficiency
- S(x): Symbolic/physics-based prediction
- N(x): Neural/data-driven prediction
- R_cognitive: Cognitive plausibility regularizer
- R_efficiency: Computational efficiency regularizer
- P(H|E, β): Probabilistic bias with expert knowledge β
"""

__version__ = "1.0.0"
__author__ = "Inspired by Ryan David Oates' Work"

from .core.hybrid_system import (
    HybridDynamicalSystem,
    HybridSystemConfig,
    SymbolicPredictor,
    NeuralPredictor,
    Regularizer,
    ProbabilisticBias
)

from .visualization.phase_space_plotter import PhaseSpacePlotter

__all__ = [
    "HybridDynamicalSystem",
    "HybridSystemConfig", 
    "SymbolicPredictor",
    "NeuralPredictor",
    "Regularizer",
    "ProbabilisticBias",
    "PhaseSpacePlotter"
]