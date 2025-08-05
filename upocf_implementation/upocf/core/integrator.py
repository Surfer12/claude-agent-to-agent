"""
RK4 integration for consciousness evolution dynamics.

Implements Algorithm 1 from UPOCF paper exactly as specified.
"""

import jax.numpy as jnp
from jax import jit
import numpy as np
from typing import Callable, Tuple, Optional
from .consciousness import ConsciousnessFunction


class RK4Integrator:
    """
    4th-order Runge-Kutta integrator for consciousness evolution.
    
    Implements the exact algorithm from UPOCF Algorithm 1:
    k₁ = f(tₙ, yₙ)
    k₂ = f(tₙ + h/2, yₙ + hk₁/2)  
    k₃ = f(tₙ + h/2, yₙ + hk₂/2)
    k₄ = f(tₙ + h, yₙ + hk₃)
    yₙ₊₁ = yₙ + (h/6)(k₁ + 2k₂ + 2k₃ + k₄)
    """
    
    def __init__(self, dt: float = 0.001):
        """
        Initialize RK4 integrator.
        
        Args:
            dt: Integration time step
        """
        self.dt = dt
        self.rk4_step = jit(self._rk4_step_impl)
    
    def _rk4_step_impl(self, f: Callable, y: jnp.ndarray, t: float, h: float) -> jnp.ndarray:
        """
        Single RK4 integration step matching Algorithm 1.
        
        Args:
            f: Dynamics function dy/dt = f(t, y)
            y: Current state vector
            t: Current time
            h: Step size
            
        Returns:
            Updated state vector y_{n+1}
        """
        k1 = f(t, y)
        k2 = f(t + h/2.0, y + h*k1/2.0)
        k3 = f(t + h/2.0, y + h*k2/2.0)
        k4 = f(t + h, y + h*k3)
        
        return y + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    def simulate(self, 
                 initial_state: jnp.ndarray,
                 dynamics: Callable,
                 steps: int,
                 save_trajectory: bool = True) -> dict:
        """
        Simulate consciousness evolution using RK4 integration.
        
        Args:
            initial_state: Initial network state x₀
            dynamics: Dynamics function dx/dt = f(t, x)
            steps: Number of integration steps
            save_trajectory: Whether to save full trajectory
            
        Returns:
            Dictionary containing:
            - final_state: Final network state
            - trajectory: Full state trajectory (if save_trajectory=True)
            - time_points: Time points
            - psi_values: Consciousness values Ψ(x) over time
        """
        # Initialize arrays
        if save_trajectory:
            trajectory = jnp.zeros((steps + 1, len(initial_state)))
            trajectory = trajectory.at[0].set(initial_state)
        
        time_points = jnp.linspace(0, steps * self.dt, steps + 1)
        psi_values = jnp.zeros(steps + 1)
        
        # Initialize consciousness function
        consciousness_fn = ConsciousnessFunction(len(initial_state))
        psi_values = psi_values.at[0].set(consciousness_fn(initial_state))
        
        # Integration loop
        current_state = initial_state
        current_time = 0.0
        
        for i in range(steps):
            # RK4 step
            current_state = self.rk4_step(dynamics, current_state, current_time, self.dt)
            current_time += self.dt
            
            # Store results
            if save_trajectory:
                trajectory = trajectory.at[i + 1].set(current_state)
            
            psi_values = psi_values.at[i + 1].set(consciousness_fn(current_state))
        
        results = {
            'final_state': current_state,
            'time_points': time_points,
            'psi_values': psi_values
        }
        
        if save_trajectory:
            results['trajectory'] = trajectory
            
        return results
    
    def adaptive_simulate(self,
                         initial_state: jnp.ndarray,
                         dynamics: Callable,
                         t_final: float,
                         tolerance: float = 1e-6) -> dict:
        """
        Adaptive step size RK4 integration.
        
        Uses step doubling for error estimation and adaptive control.
        """
        # Start with initial step size
        h = self.dt
        t = 0.0
        y = initial_state
        
        trajectory = [y]
        time_points = [t]
        consciousness_fn = ConsciousnessFunction(len(initial_state))
        psi_values = [consciousness_fn(y)]
        
        while t < t_final:
            # Adjust step size to not overshoot
            if t + h > t_final:
                h = t_final - t
            
            # Take one step of size h
            y1 = self._rk4_step_impl(dynamics, y, t, h)
            
            # Take two steps of size h/2
            y_half = self._rk4_step_impl(dynamics, y, t, h/2)
            y2 = self._rk4_step_impl(dynamics, y_half, t + h/2, h/2)
            
            # Estimate error
            error = jnp.linalg.norm(y2 - y1)
            
            if error < tolerance:
                # Accept step
                y = y2  # Use more accurate estimate
                t += h
                trajectory.append(y)
                time_points.append(t)
                psi_values.append(consciousness_fn(y))
                
                # Increase step size if error is very small
                if error < tolerance / 10:
                    h = min(h * 1.5, 2 * self.dt)
            else:
                # Reject step and reduce step size
                h = h * 0.5
                if h < 1e-8:  # Prevent infinite loops
                    break
        
        return {
            'final_state': y,
            'trajectory': jnp.array(trajectory),
            'time_points': jnp.array(time_points),
            'psi_values': jnp.array(psi_values)
        }


def consciousness_dynamics(t: float, x: jnp.ndarray, 
                         alpha: float = 1.0, 
                         beta: float = 0.1) -> jnp.ndarray:
    """
    Default consciousness evolution dynamics.
    
    Implements: dx/dt = α∇Ψ(x) - βx + noise
    
    Args:
        t: Current time
        x: Current network state
        alpha: Gradient flow strength
        beta: Damping coefficient
        
    Returns:
        State derivative dx/dt
    """
    consciousness_fn = ConsciousnessFunction(len(x))
    
    # Gradient ascent on consciousness function
    grad_psi = consciousness_fn.gradient(x)
    
    # Damping term to prevent unbounded growth
    damping = -beta * x
    
    # Small noise for exploration (deterministic for reproducibility)
    noise = 0.01 * jnp.sin(10 * t * jnp.arange(len(x)))
    
    return alpha * grad_psi + damping + noise


def hopf_bifurcation_dynamics(t: float, x: jnp.ndarray,
                             mu: float = 0.1,
                             omega: float = 1.0) -> jnp.ndarray:
    """
    Hopf bifurcation dynamics for consciousness transitions.
    
    Implements the corrected polar form from UPOCF:
    ṙ = μr - r³
    θ̇ = ω
    
    Args:
        t: Current time
        x: State in Cartesian coordinates [x₁, x₂, ...]
        mu: Bifurcation parameter
        omega: Oscillation frequency
        
    Returns:
        State derivative dx/dt
    """
    n = len(x)
    dxdt = jnp.zeros_like(x)
    
    # Apply Hopf dynamics to pairs of variables
    for i in range(0, n-1, 2):
        # Convert to polar coordinates
        r = jnp.sqrt(x[i]**2 + x[i+1]**2)
        theta = jnp.arctan2(x[i+1], x[i])
        
        # Hopf bifurcation equations
        drdt = mu * r - r**3
        dthetadt = omega
        
        # Convert back to Cartesian
        dxdt = dxdt.at[i].set(drdt * jnp.cos(theta) - r * dthetadt * jnp.sin(theta))
        dxdt = dxdt.at[i+1].set(drdt * jnp.sin(theta) + r * dthetadt * jnp.cos(theta))
    
    # Handle odd number of variables
    if n % 2 == 1:
        dxdt = dxdt.at[-1].set(mu * x[-1] - x[-1]**3)
    
    return dxdt


# Convenience function matching the action plan specification
def rk4_step(f: Callable, y: jnp.ndarray, t: float, h: float) -> jnp.ndarray:
    """
    Standalone RK4 step function exactly as specified in action plan.
    
    Args:
        f: Dynamics function
        y: Current state
        t: Current time
        h: Step size
        
    Returns:
        Next state
    """
    k1 = f(t, y)
    k2 = f(t + h/2.0, y + h*k1/2.0)
    k3 = f(t + h/2.0, y + h*k2/2.0)
    k4 = f(t + h, y + h*k3)
    return y + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)


# Export main interface
__all__ = ['RK4Integrator', 'consciousness_dynamics', 'hopf_bifurcation_dynamics', 'rk4_step']