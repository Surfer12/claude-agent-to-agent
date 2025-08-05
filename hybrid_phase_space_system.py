"""
Hybrid Dynamical Systems Framework
Phase-Space Trajectory Analysis in the Style of Ryan David Oates

This module implements the mathematical framework described in the walk-through:
Ψ(x) = ∫[ α(t) S(x) + [1−α(t)] N(x) + w_cross(S(m₁)N(m₂)−S(m₂)N(m₁)) ]
       × exp[−(λ₁ R_cognitive + λ₂ R_efficiency)] × P(H|E, β) dt

Where the trajectory (α(t), λ₁(t), λ₂(t)) evolves through 3D phase space
according to differential equations that can be learned via PINNs or Neural-ODEs.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp, quad
from scipy.optimize import minimize
import torch
import torch.nn as nn
from typing import Tuple, Callable, Optional, Dict, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class SystemParameters:
    """Core parameters for the hybrid dynamical system"""
    # Time domain
    t_span: Tuple[float, float] = (0.0, 10.0)
    dt: float = 0.01
    
    # Initial conditions [α(0), λ₁(0), λ₂(0)]
    initial_state: np.ndarray = np.array([2.0, 2.0, 0.0])
    
    # Cross-coupling strength
    w_cross: float = 0.1
    
    # Expert knowledge parameters
    beta: float = 1.4
    
    # Bounds for parameters
    alpha_bounds: Tuple[float, float] = (0.0, 2.0)
    lambda_bounds: Tuple[float, float] = (0.0, 2.0)


class SymbolicSolver(ABC):
    """Abstract base class for symbolic reasoning components"""
    
    @abstractmethod
    def evaluate(self, x: np.ndarray, t: float) -> float:
        """Evaluate symbolic solution S(x) at time t"""
        pass


class NeuralPredictor(ABC):
    """Abstract base class for neural network components"""
    
    @abstractmethod
    def evaluate(self, x: np.ndarray, t: float) -> float:
        """Evaluate neural prediction N(x) at time t"""
        pass


class PhysicsSymbolicSolver(SymbolicSolver):
    """RK4-based physics solver for symbolic reasoning"""
    
    def __init__(self, physics_model: Callable = None):
        self.physics_model = physics_model or self._default_physics_model
    
    def _default_physics_model(self, x: np.ndarray, t: float) -> float:
        """Default physics model - harmonic oscillator with damping"""
        return 0.6 * np.exp(-0.1 * t) * np.cos(2 * np.pi * t)
    
    def evaluate(self, x: np.ndarray, t: float) -> float:
        return self.physics_model(x, t)


class LSTMNeuralPredictor(NeuralPredictor):
    """LSTM-based neural predictor"""
    
    def __init__(self, model_params: Optional[Dict] = None):
        self.model_params = model_params or {}
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize a simple LSTM-like predictor"""
        # Simplified model for demonstration
        self.weights = np.random.randn(10) * 0.1
    
    def evaluate(self, x: np.ndarray, t: float) -> float:
        """Evaluate neural prediction with time-dependent behavior"""
        # Simplified neural response that learns over time
        base_response = 0.8
        time_adaptation = 0.1 * np.sin(0.5 * t)
        noise = 0.05 * np.random.randn()
        return base_response + time_adaptation + noise


class PenaltyFunctions:
    """Cognitive and efficiency penalty functions"""
    
    @staticmethod
    def cognitive_penalty(alpha: float, lambda1: float, t: float) -> float:
        """R_cognitive: penalizes cognitively implausible solutions"""
        # Higher penalty when symbolic weight is low but should be high
        physics_importance = 0.5 + 0.3 * np.sin(0.2 * t)  # Time-varying physics relevance
        cognitive_mismatch = abs(alpha/2.0 - physics_importance)
        return 0.25 * cognitive_mismatch**2
    
    @staticmethod
    def efficiency_penalty(alpha: float, lambda2: float, t: float) -> float:
        """R_efficiency: penalizes computationally expensive solutions"""
        # Penalty increases with neural complexity and symbolic overhead
        neural_complexity = (1 - alpha/2.0) * 0.3  # More neural = more complex
        symbolic_overhead = (alpha/2.0) * 0.2      # Symbolic has overhead too
        return neural_complexity + symbolic_overhead


class ProbabilisticBias:
    """Handles P(H|E, β) - expert knowledge integration"""
    
    def __init__(self, beta: float = 1.4):
        self.beta = beta
    
    def evaluate(self, evidence: float, hypothesis_prior: float = 0.7) -> float:
        """Compute P(H|E, β) with expert bias β"""
        # Bayesian update with expert bias
        likelihood = evidence
        prior = hypothesis_prior
        
        # Expert bias adjustment
        biased_likelihood = likelihood ** self.beta
        
        # Simplified Bayesian update
        posterior = (biased_likelihood * prior) / (
            biased_likelihood * prior + (1 - biased_likelihood) * (1 - prior)
        )
        
        return min(0.98, max(0.02, posterior))  # Clamp to reasonable bounds


class HybridDynamicalSystem:
    """Main class implementing the hybrid dynamical system"""
    
    def __init__(self, 
                 params: SystemParameters,
                 symbolic_solver: SymbolicSolver,
                 neural_predictor: NeuralPredictor):
        self.params = params
        self.symbolic_solver = symbolic_solver
        self.neural_predictor = neural_predictor
        self.penalty_functions = PenaltyFunctions()
        self.prob_bias = ProbabilisticBias(params.beta)
        
        # Storage for trajectory data
        self.trajectory_data = None
        self.psi_values = None
    
    def phase_space_dynamics(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Defines the differential equations governing (α(t), λ₁(t), λ₂(t))
        
        This is where the "smart thermostat" logic lives - how the system
        adapts its symbolic/neural balance and penalty weights over time.
        """
        alpha, lambda1, lambda2 = state
        
        # Ensure bounds
        alpha = np.clip(alpha, *self.params.alpha_bounds)
        lambda1 = np.clip(lambda1, *self.params.lambda_bounds)
        lambda2 = np.clip(lambda2, *self.params.lambda_bounds)
        
        # Example dynamics inspired by Oates' work:
        # - α decreases as system learns (neural takes over)
        # - λ₁ decreases as cognitive constraints are internalized
        # - λ₂ increases as efficiency becomes more important
        
        # Coupling terms (simplified Koopman-inspired dynamics)
        coupling_alpha_lambda1 = -0.1 * alpha * lambda1
        coupling_lambda1_lambda2 = 0.05 * lambda1 * (2.0 - lambda2)
        
        # Time-dependent forcing (could be learned via PINN)
        time_forcing = 0.1 * np.sin(0.3 * t)
        
        # Differential equations
        dalpha_dt = -0.2 * alpha + coupling_alpha_lambda1 + time_forcing
        dlambda1_dt = -0.15 * lambda1 + coupling_lambda1_lambda2
        dlambda2_dt = 0.1 * (2.0 - lambda2) + 0.05 * alpha
        
        return np.array([dalpha_dt, dlambda1_dt, dlambda2_dt])
    
    def generate_trajectory(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate the phase-space trajectory using ODE integration"""
        
        # Solve the differential equation
        solution = solve_ivp(
            self.phase_space_dynamics,
            self.params.t_span,
            self.params.initial_state,
            dense_output=True,
            rtol=1e-8,
            atol=1e-10
        )
        
        # Create dense time grid for smooth trajectory
        t_eval = np.linspace(*self.params.t_span, int((self.params.t_span[1] - self.params.t_span[0]) / self.params.dt))
        trajectory = solution.sol(t_eval).T
        
        # Ensure bounds are respected
        trajectory[:, 0] = np.clip(trajectory[:, 0], *self.params.alpha_bounds)
        trajectory[:, 1] = np.clip(trajectory[:, 1], *self.params.lambda_bounds)
        trajectory[:, 2] = np.clip(trajectory[:, 2], *self.params.lambda_bounds)
        
        self.trajectory_data = (t_eval, trajectory)
        return t_eval, trajectory
    
    def compute_psi_integrand(self, x: np.ndarray, t: float, alpha: float, lambda1: float, lambda2: float) -> float:
        """
        Compute the integrand of Ψ(x) at a specific time point
        
        Ψ_t(x) = [α(t) S(x) + [1−α(t)] N(x) + w_cross Δ_mix]
                 × exp[−(λ₁(t) R_cognitive + λ₂(t) R_efficiency)]
                 × P(H|E, β)
        """
        
        # Evaluate symbolic and neural components
        S_x = self.symbolic_solver.evaluate(x, t)
        N_x = self.neural_predictor.evaluate(x, t)
        
        # Normalize α to [0,1] for blending
        alpha_norm = alpha / 2.0
        
        # Hybrid output with cross-term
        # Cross-term: w_cross(S(m₁)N(m₂) - S(m₂)N(m₁))
        # Simplified as interaction between S and N
        cross_term = self.params.w_cross * (S_x * N_x - 0.5 * (S_x + N_x))
        
        hybrid_output = alpha_norm * S_x + (1 - alpha_norm) * N_x + cross_term
        
        # Compute penalty terms
        R_cog = self.penalty_functions.cognitive_penalty(alpha, lambda1, t)
        R_eff = self.penalty_functions.efficiency_penalty(alpha, lambda2, t)
        
        # Normalize λ values for penalty computation
        lambda1_norm = lambda1 / 2.0
        lambda2_norm = lambda2 / 2.0
        
        penalty_term = np.exp(-(lambda1_norm * R_cog + lambda2_norm * R_eff))
        
        # Probabilistic bias term
        evidence = abs(hybrid_output)  # Use hybrid output magnitude as evidence
        prob_term = self.prob_bias.evaluate(evidence)
        
        return hybrid_output * penalty_term * prob_term
    
    def compute_psi_integral(self, x: np.ndarray) -> float:
        """
        Compute the full Ψ(x) integral along the trajectory
        
        Ψ(x) = ∫ Ψ_t(x) dt over the trajectory
        """
        if self.trajectory_data is None:
            self.generate_trajectory()
        
        t_eval, trajectory = self.trajectory_data
        
        def integrand(t):
            # Interpolate trajectory values at time t
            alpha_t = np.interp(t, t_eval, trajectory[:, 0])
            lambda1_t = np.interp(t, t_eval, trajectory[:, 1])
            lambda2_t = np.interp(t, t_eval, trajectory[:, 2])
            
            return self.compute_psi_integrand(x, t, alpha_t, lambda1_t, lambda2_t)
        
        # Numerical integration
        result, _ = quad(integrand, *self.params.t_span, limit=100)
        return result
    
    def analyze_trajectory_point(self, t: float, x: np.ndarray) -> Dict[str, Any]:
        """
        Detailed analysis of a single point on the trajectory
        Reproduces the concrete example from the walk-through
        """
        if self.trajectory_data is None:
            self.generate_trajectory()
        
        t_eval, trajectory = self.trajectory_data
        
        # Interpolate trajectory values
        alpha_t = np.interp(t, t_eval, trajectory[:, 0])
        lambda1_t = np.interp(t, t_eval, trajectory[:, 1])
        lambda2_t = np.interp(t, t_eval, trajectory[:, 2])
        
        # Component evaluations
        S_x = self.symbolic_solver.evaluate(x, t)
        N_x = self.neural_predictor.evaluate(x, t)
        
        # Hybrid computation
        alpha_norm = alpha_t / 2.0
        hybrid_output = alpha_norm * S_x + (1 - alpha_norm) * N_x
        
        # Penalty computation
        R_cog = self.penalty_functions.cognitive_penalty(alpha_t, lambda1_t, t)
        R_eff = self.penalty_functions.efficiency_penalty(alpha_t, lambda2_t, t)
        
        lambda1_norm = lambda1_t / 2.0
        lambda2_norm = lambda2_t / 2.0
        penalty_term = np.exp(-(lambda1_norm * R_cog + lambda2_norm * R_eff))
        
        # Probabilistic bias
        evidence = abs(hybrid_output)
        prob_term = self.prob_bias.evaluate(evidence)
        
        # Final contribution
        psi_contribution = hybrid_output * penalty_term * prob_term
        
        return {
            'time': t,
            'trajectory_point': {
                'alpha': alpha_t,
                'lambda1': lambda1_t,
                'lambda2': lambda2_t
            },
            'components': {
                'symbolic': S_x,
                'neural': N_x,
                'hybrid': hybrid_output
            },
            'penalties': {
                'cognitive': R_cog,
                'efficiency': R_eff,
                'penalty_term': penalty_term
            },
            'probabilistic_bias': prob_term,
            'psi_contribution': psi_contribution
        }


def create_example_system() -> HybridDynamicalSystem:
    """Create an example system matching the walk-through"""
    
    params = SystemParameters(
        t_span=(0.0, 10.0),
        initial_state=np.array([2.0, 2.0, 0.0]),
        w_cross=0.1,
        beta=1.4
    )
    
    symbolic_solver = PhysicsSymbolicSolver()
    neural_predictor = LSTMNeuralPredictor()
    
    return HybridDynamicalSystem(params, symbolic_solver, neural_predictor)


if __name__ == "__main__":
    # Example usage
    system = create_example_system()
    
    # Generate trajectory
    t_eval, trajectory = system.generate_trajectory()
    
    # Analyze mid-trajectory point (matching walk-through example)
    mid_time = 5.0
    test_x = np.array([1.0])  # Example input
    
    analysis = system.analyze_trajectory_point(mid_time, test_x)
    
    print("=== Trajectory Analysis at t = 5.0 ===")
    print(f"α(t) = {analysis['trajectory_point']['alpha']:.3f}")
    print(f"λ₁(t) = {analysis['trajectory_point']['lambda1']:.3f}")
    print(f"λ₂(t) = {analysis['trajectory_point']['lambda2']:.3f}")
    print(f"\nS(x) = {analysis['components']['symbolic']:.3f}")
    print(f"N(x) = {analysis['components']['neural']:.3f}")
    print(f"Hybrid = {analysis['components']['hybrid']:.3f}")
    print(f"\nPenalty term = {analysis['penalties']['penalty_term']:.4f}")
    print(f"P(H|E,β) = {analysis['probabilistic_bias']:.3f}")
    print(f"Ψ_t(x) = {analysis['psi_contribution']:.4f}")
    
    # Compute full integral
    psi_integral = system.compute_psi_integral(test_x)
    print(f"\nFull Ψ(x) integral = {psi_integral:.4f}")