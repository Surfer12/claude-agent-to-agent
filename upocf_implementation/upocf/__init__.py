"""
UPOCF: Unified Onto-Phenomenological Consciousness Framework

A concrete implementation of consciousness detection based on Integrated Information 
Theory (IIT) and geometric analysis.

Core Components:
- ConsciousnessFunction: Ψ(x) := Φ_IIT(x) implementation
- RK4Integrator: 4th-order Runge-Kutta integration for consciousness evolution
- ConsciousnessDetector: Threshold-based consciousness detection with ROC analysis

Example Usage:
    from upocf import ConsciousnessDetector, RK4Integrator, psi
    
    # Direct consciousness evaluation
    consciousness_level = psi(network_state)
    
    # Consciousness detection
    detector = ConsciousnessDetector(threshold=0.5)
    is_conscious, psi_value = detector.detect(network_state)
    
    # Evolution simulation
    integrator = RK4Integrator(dt=0.001)
    results = integrator.simulate(initial_state, dynamics, steps=1000)
"""

from .core.consciousness import ConsciousnessFunction, psi
from .core.integrator import (
    RK4Integrator, 
    consciousness_dynamics, 
    hopf_bifurcation_dynamics,
    rk4_step
)
from .core.detector import ConsciousnessDetector

__version__ = "0.1.0"
__author__ = "UPOCF Implementation Team"
__license__ = "MIT"

__all__ = [
    # Core consciousness function
    'ConsciousnessFunction',
    'psi',
    
    # Integration and dynamics
    'RK4Integrator',
    'consciousness_dynamics',
    'hopf_bifurcation_dynamics', 
    'rk4_step',
    
    # Detection and analysis
    'ConsciousnessDetector',
    
    # Metadata
    '__version__',
    '__author__',
    '__license__'
]