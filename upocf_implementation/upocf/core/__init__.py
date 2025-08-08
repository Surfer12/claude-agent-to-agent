"""
UPOCF Core Components

Core implementations of the consciousness framework:
- consciousness: Ψ(x) := Φ_IIT(x) function with autodiff
- integrator: RK4 integration for consciousness evolution  
- detector: Threshold-based consciousness detection
"""

from .consciousness import ConsciousnessFunction, psi
from .integrator import RK4Integrator, consciousness_dynamics, hopf_bifurcation_dynamics, rk4_step
from .detector import ConsciousnessDetector

__all__ = [
    'ConsciousnessFunction',
    'psi', 
    'RK4Integrator',
    'consciousness_dynamics',
    'hopf_bifurcation_dynamics',
    'rk4_step',
    'ConsciousnessDetector'
]