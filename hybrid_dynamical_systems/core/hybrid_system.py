"""
Hybrid Dynamical Systems Framework
Inspired by Ryan David Oates' work on physics-informed neural networks and hybrid modeling.

This module implements the core expression:
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

import numpy as np
import torch
import torch.nn as nn
from typing import Callable, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

@dataclass
class HybridSystemConfig:
    """Configuration for the hybrid dynamical system."""
    # Time parameters
    t_start: float = 0.0
    t_end: float = 10.0
    dt: float = 0.01
    
    # Parameter bounds
    alpha_bounds: Tuple[float, float] = (0.0, 2.0)
    lambda1_bounds: Tuple[float, float] = (0.0, 2.0)
    lambda2_bounds: Tuple[float, float] = (0.0, 2.0)
    
    # Cross-coupling weight
    w_cross: float = 0.1
    
    # Expert bias parameter
    beta: float = 1.4
    
    # Initial conditions for trajectory
    alpha_init: float = 2.0
    lambda1_init: float = 2.0
    lambda2_init: float = 0.0

class SymbolicPredictor:
    """Symbolic/physics-based predictor S(x)."""
    
    def __init__(self, physics_model: str = "rk4"):
        self.physics_model = physics_model
        
    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        """Evaluate symbolic prediction S(x) at time t."""
        if self.physics_model == "rk4":
            # Simulate RK4 physics solver
            return self._rk4_prediction(x, t)
        else:
            return np.zeros_like(x)
    
    def _rk4_prediction(self, x: np.ndarray, t: float) -> np.ndarray:
        """RK4-based physics prediction."""
        # Simplified physics model - could be extended to actual ODEs
        return 0.6 * np.ones_like(x)  # Example: constant physics prediction

class NeuralPredictor:
    """Neural/data-driven predictor N(x)."""
    
    def __init__(self, hidden_size: int = 64):
        self.hidden_size = hidden_size
        # Simplified neural network - could be extended to actual LSTM/Transformer
        self._setup_network()
        
    def _setup_network(self):
        """Setup neural network architecture."""
        # Placeholder for actual neural network
        pass
        
    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        """Evaluate neural prediction N(x) at time t."""
        # Simplified neural prediction
        return 0.8 * np.ones_like(x)  # Example: constant neural prediction

class Regularizer:
    """Regularization terms for cognitive plausibility and efficiency."""
    
    def __init__(self):
        pass
        
    def cognitive_plausibility(self, x: np.ndarray, alpha: float, lambda1: float) -> float:
        """Compute cognitive plausibility regularizer R_cognitive."""
        # Penalize solutions that violate basic physics or common sense
        physics_violation = np.mean(np.abs(x - 0.5))  # Simplified metric
        return lambda1 * physics_violation
        
    def computational_efficiency(self, x: np.ndarray, lambda2: float) -> float:
        """Compute computational efficiency regularizer R_efficiency."""
        # Penalize computationally expensive solutions
        complexity = np.mean(np.abs(x))  # Simplified complexity metric
        return lambda2 * complexity

class ProbabilisticBias:
    """Probabilistic bias incorporating expert knowledge P(H|E, β)."""
    
    def __init__(self, beta: float = 1.4):
        self.beta = beta
        
    def __call__(self, evidence: float, hypothesis: float) -> float:
        """Compute P(H|E, β) with expert bias β."""
        # Simplified Bayesian update with expert bias
        base_prob = 0.7  # P(H|E)
        expert_bias = np.power(base_prob, self.beta)
        return np.clip(expert_bias, 0.0, 1.0)

class HybridDynamicalSystem:
    """
    Core hybrid dynamical system implementing the Ψ(x) expression.
    
    The system evolves the parameters (α(t), λ₁(t), λ₂(t)) according to
    learned differential equations, producing a trajectory in 3D phase space.
    """
    
    def __init__(self, config: HybridSystemConfig):
        self.config = config
        self.symbolic_predictor = SymbolicPredictor()
        self.neural_predictor = NeuralPredictor()
        self.regularizer = Regularizer()
        self.probabilistic_bias = ProbabilisticBias(config.beta)
        
        # Initialize trajectory
        self.trajectory = None
        self.times = None
        
    def compute_hybrid_output(self, x: np.ndarray, t: float, alpha: float) -> np.ndarray:
        """Compute the hybrid output: α(t) S(x) + [1−α(t)] N(x)."""
        alpha_normalized = alpha / self.config.alpha_bounds[1]
        symbolic_pred = self.symbolic_predictor(x, t)
        neural_pred = self.neural_predictor(x, t)
        
        return (alpha_normalized * symbolic_pred + 
                (1 - alpha_normalized) * neural_pred)
    
    def compute_cross_coupling(self, x: np.ndarray, t: float) -> np.ndarray:
        """Compute cross-coupling term: w_cross(S(m₁)N(m₂)−S(m₂)N(m₁))."""
        # Simplified cross-coupling - could be extended to symplectic/Koopman terms
        m1, m2 = x[::2], x[1::2]  # Split input into two components
        s_m1 = self.symbolic_predictor(m1, t)
        s_m2 = self.symbolic_predictor(m2, t)
        n_m1 = self.neural_predictor(m1, t)
        n_m2 = self.neural_predictor(m2, t)
        
        cross_term = self.config.w_cross * (s_m1 * n_m2 - s_m2 * n_m1)
        return cross_term
    
    def compute_penalty_term(self, x: np.ndarray, lambda1: float, lambda2: float) -> float:
        """Compute penalty term: exp[−(λ₁ R_cognitive + λ₂ R_efficiency)]."""
        r_cognitive = self.regularizer.cognitive_plausibility(x, 1.0, lambda1)
        r_efficiency = self.regularizer.computational_efficiency(x, lambda2)
        
        penalty = np.exp(-(lambda1 * r_cognitive + lambda2 * r_efficiency))
        return penalty
    
    def evaluate_psi(self, x: np.ndarray, t: float, alpha: float, 
                    lambda1: float, lambda2: float) -> float:
        """
        Evaluate Ψ(x) at a specific time t with given parameters.
        
        Returns:
            float: The value of Ψ(x) at time t
        """
        # Hybrid output
        hybrid_output = self.compute_hybrid_output(x, t, alpha)
        
        # Cross-coupling term
        cross_term = self.compute_cross_coupling(x, t)
        
        # Combined prediction
        prediction = hybrid_output + cross_term
        
        # Penalty term
        penalty = self.compute_penalty_term(x, lambda1, lambda2)
        
        # Probabilistic bias
        evidence = np.mean(x)
        hypothesis = np.mean(prediction)
        prob_bias = self.probabilistic_bias(evidence, hypothesis)
        
        # Final Ψ(x) evaluation
        psi_value = np.mean(prediction * penalty * prob_bias)
        
        return psi_value
    
    def parameter_dynamics(self, t: float, params: np.ndarray) -> np.ndarray:
        """
        Define the dynamics for the parameters (α, λ₁, λ₂).
        
        This implements the differential equations that govern the evolution
        of the adaptive parameters, producing the 3D phase-space trajectory.
        """
        alpha, lambda1, lambda2 = params
        
        # Example dynamics - could be learned via PINN or Neural ODE
        # Linear descent from high symbolic/cognitive to high neural/efficiency
        d_alpha = -0.2  # α decreases over time
        d_lambda1 = -0.15  # λ₁ decreases over time  
        d_lambda2 = 0.2  # λ₂ increases over time
        
        return np.array([d_alpha, d_lambda1, d_lambda2])
    
    def integrate_trajectory(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Integrate the parameter dynamics to produce the 3D trajectory.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (times, trajectory)
        """
        t_span = (self.config.t_start, self.config.t_end)
        initial_conditions = np.array([
            self.config.alpha_init,
            self.config.lambda1_init, 
            self.config.lambda2_init
        ])
        
        # Solve the ODE system
        solution = solve_ivp(
            self.parameter_dynamics,
            t_span,
            initial_conditions,
            method='RK45',
            t_eval=np.arange(self.config.t_start, self.config.t_end, self.config.dt)
        )
        
        self.times = solution.t
        self.trajectory = solution.y.T  # Shape: (n_times, 3)
        
        return self.times, self.trajectory
    
    def evaluate_psi_trajectory(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate Ψ(x) along the entire trajectory.
        
        Args:
            x: Input data
            
        Returns:
            np.ndarray: Ψ(x) values at each time step
        """
        if self.trajectory is None:
            self.integrate_trajectory()
            
        psi_values = []
        for i, t in enumerate(self.times):
            alpha, lambda1, lambda2 = self.trajectory[i]
            psi_t = self.evaluate_psi(x, t, alpha, lambda1, lambda2)
            psi_values.append(psi_t)
            
        return np.array(psi_values)
    
    def get_trajectory_insights(self) -> Dict[str, Any]:
        """
        Extract insights from the 3D trajectory.
        
        Returns:
            Dict containing trajectory analysis
        """
        if self.trajectory is None:
            self.integrate_trajectory()
            
        trajectory = self.trajectory
        times = self.times
        
        # Analyze trajectory characteristics
        start_point = trajectory[0]
        end_point = trajectory[-1]
        
        # Compute trajectory statistics
        alpha_traj = trajectory[:, 0]
        lambda1_traj = trajectory[:, 1]
        lambda2_traj = trajectory[:, 2]
        
        insights = {
            'start_point': start_point,
            'end_point': end_point,
            'trajectory_length': len(trajectory),
            'alpha_range': (np.min(alpha_traj), np.max(alpha_traj)),
            'lambda1_range': (np.min(lambda1_traj), np.max(lambda1_traj)),
            'lambda2_range': (np.min(lambda2_traj), np.max(lambda2_traj)),
            'trajectory_type': self._classify_trajectory(),
            'evolution_characteristics': {
                'symbolic_to_neural': alpha_traj[0] > alpha_traj[-1],
                'cognitive_to_efficiency': lambda1_traj[0] > lambda1_traj[-1],
                'efficiency_increase': lambda2_traj[-1] > lambda2_traj[0]
            }
        }
        
        return insights
    
    def _classify_trajectory(self) -> str:
        """Classify the trajectory based on its geometric properties."""
        if self.trajectory is None:
            return "Not computed"
            
        # Analyze if trajectory is linear, chaotic, or constrained
        trajectory = self.trajectory
        
        # Compute trajectory curvature
        if len(trajectory) > 2:
            # Simplified curvature analysis
            linearity_score = self._compute_linearity_score(trajectory)
            
            if linearity_score > 0.8:
                return "Linear/Constrained"
            elif linearity_score > 0.5:
                return "Weakly Chaotic"
            else:
                return "Highly Chaotic"
        else:
            return "Insufficient data"
    
    def _compute_linearity_score(self, trajectory: np.ndarray) -> float:
        """Compute a score indicating how linear the trajectory is."""
        if len(trajectory) < 3:
            return 1.0
            
        # Simplified linearity measure
        # Could be extended with more sophisticated geometric analysis
        return 0.85  # Placeholder - indicates linear-looking trajectory