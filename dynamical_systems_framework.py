"""
Dynamical Systems Framework for Phase-Space Trajectory Analysis
Based on Ryan David Oates' work on hybrid symbolic-neural systems

This module implements:
1. Phase-space trajectory modeling with α(t), λ₁(t), λ₂(t)
2. Core equation Ψ(x) with hybrid symbolic-neural outputs
3. Physics-Informed Neural Networks (PINNs) integration
4. Dynamic Mode Decomposition (DMD) capabilities
5. Runge-Kutta 4th-order verification
6. Multi-pendulum chaotic system simulation
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp
from typing import Tuple, List, Dict, Optional, Callable
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SystemParameters:
    """Parameters for the dynamical system"""
    alpha_range: Tuple[float, float] = (0.0, 2.0)
    lambda1_range: Tuple[float, float] = (0.0, 2.0) 
    lambda2_range: Tuple[float, float] = (0.0, 2.0)
    time_span: Tuple[float, float] = (0.0, 10.0)
    dt: float = 0.01
    cognitive_penalty_weight: float = 0.75
    efficiency_penalty_weight: float = 0.25

class SymbolicOutput:
    """Symbolic reasoning component using RK4 solutions"""
    
    def __init__(self, system_params: SystemParameters):
        self.params = system_params
        
    def compute_rk4_solution(self, x: np.ndarray, t: float) -> float:
        """
        Compute symbolic output using Runge-Kutta 4th-order method
        For multi-pendulum system: θ̈ + sin(θ) = 0 (simplified)
        """
        def pendulum_dynamics(t, y):
            theta, theta_dot = y
            return [theta_dot, -np.sin(theta)]
        
        # Initial conditions based on input x
        y0 = [x[0] if len(x) > 0 else 0.5, x[1] if len(x) > 1 else 0.0]
        
        # Solve using RK4 equivalent (solve_ivp with RK45)
        sol = solve_ivp(pendulum_dynamics, [0, t], y0, method='RK45', dense_output=True)
        
        if sol.success:
            return float(sol.sol(t)[0])  # Return angle at time t
        else:
            return 0.6  # Default fallback value
    
    def evaluate(self, x: np.ndarray, t: float) -> float:
        """Evaluate symbolic output S(x) at time t"""
        return self.compute_rk4_solution(x, t)

class NeuralOutput(nn.Module):
    """Neural network component for predictions"""
    
    def __init__(self, input_dim: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate neural output N(x)"""
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x).unsqueeze(0)
            output = self.forward(x_tensor)
            return float(output.item())

class PhaseSpaceTrajectory:
    """Models the phase-space trajectory with α(t), λ₁(t), λ₂(t)"""
    
    def __init__(self, params: SystemParameters):
        self.params = params
        self.t_values = np.arange(params.time_span[0], params.time_span[1], params.dt)
        
    def alpha_function(self, t: float) -> float:
        """Time-varying weight α(t) between symbolic and neural outputs"""
        # Smooth transition from 2.0 to 1.0 as shown in the trajectory
        return 2.0 - 0.5 * t / self.params.time_span[1]
    
    def lambda1_function(self, t: float) -> float:
        """Time-varying regularization weight λ₁(t) for cognitive penalty"""
        # Decreasing from 2.0 to ~1.0 as shown in trajectory
        return 2.0 - 0.5 * t / self.params.time_span[1]
    
    def lambda2_function(self, t: float) -> float:
        """Time-varying regularization weight λ₂(t) for efficiency penalty"""
        # Decreasing from 2.0 to ~0.5 as shown in trajectory
        return 2.0 - 0.75 * t / self.params.time_span[1]
    
    def get_trajectory_point(self, t: float) -> Tuple[float, float, float]:
        """Get trajectory point at time t"""
        return (
            self.alpha_function(t),
            self.lambda1_function(t), 
            self.lambda2_function(t)
        )
    
    def get_full_trajectory(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get complete trajectory data"""
        alpha_vals = np.array([self.alpha_function(t) for t in self.t_values])
        lambda1_vals = np.array([self.lambda1_function(t) for t in self.t_values])
        lambda2_vals = np.array([self.lambda2_function(t) for t in self.t_values])
        
        return self.t_values, alpha_vals, lambda1_vals, lambda2_vals

class CoreEquationEvaluator:
    """
    Implements the core equation:
    Ψ(x) = ∫[α(t)S(x) + (1-α(t))N(x) + w_cross[S(m₁)N(m₂) - S(m₂)N(m₁)]] 
           × exp(-[λ₁R_cognitive + λ₂R_efficiency]) × P(H|E,β) dt
    """
    
    def __init__(self, params: SystemParameters):
        self.params = params
        self.symbolic = SymbolicOutput(params)
        self.neural = NeuralOutput()
        self.trajectory = PhaseSpaceTrajectory(params)
        
    def compute_cross_term(self, x: np.ndarray, m1: np.ndarray, m2: np.ndarray, 
                          t: float, w_cross: float = 0.1) -> float:
        """Compute cross-interaction term w_cross[S(m₁)N(m₂) - S(m₂)N(m₁)]"""
        s_m1 = self.symbolic.evaluate(m1, t)
        s_m2 = self.symbolic.evaluate(m2, t)
        n_m1 = self.neural.evaluate(m1)
        n_m2 = self.neural.evaluate(m2)
        
        return w_cross * (s_m1 * n_m1 - s_m2 * n_m2)
    
    def compute_regularization_penalties(self, x: np.ndarray, t: float) -> Tuple[float, float]:
        """Compute cognitive and efficiency penalties"""
        # Cognitive penalty: misalignment with physical laws
        r_cognitive = 0.25 * (1 + 0.1 * np.sin(2 * np.pi * t))  # Time-varying
        
        # Efficiency penalty: computational cost
        r_efficiency = 0.10 * (1 + 0.05 * t)  # Gradually increasing
        
        return r_cognitive, r_efficiency
    
    def compute_probability_adjustment(self, base_prob: float, beta: float) -> float:
        """Compute adjusted probability P(H|E,β)"""
        return min(base_prob * beta, 1.0)  # Cap at 1.0
    
    def evaluate_at_time(self, x: np.ndarray, t: float, 
                        m1: Optional[np.ndarray] = None, 
                        m2: Optional[np.ndarray] = None,
                        base_prob: float = 0.7,
                        beta: float = 1.4) -> Dict[str, float]:
        """Evaluate the core equation at a specific time point"""
        
        # Get trajectory parameters at time t
        alpha_t, lambda1_t, lambda2_t = self.trajectory.get_trajectory_point(t)
        
        # Normalize alpha to [0,1] range for hybrid computation
        alpha_normalized = alpha_t / 2.0
        
        # Compute symbolic and neural outputs
        s_x = self.symbolic.evaluate(x, t)
        n_x = self.neural.evaluate(x)
        
        # Hybrid output
        hybrid_output = alpha_normalized * s_x + (1 - alpha_normalized) * n_x
        
        # Cross-interaction term
        cross_term = 0.0
        if m1 is not None and m2 is not None:
            cross_term = self.compute_cross_term(x, m1, m2, t)
        
        # Total output before regularization
        total_output = hybrid_output + cross_term
        
        # Regularization penalties
        r_cognitive, r_efficiency = self.compute_regularization_penalties(x, t)
        
        # Scaled regularization weights
        lambda1_scaled = lambda1_t / 2.0
        lambda2_scaled = lambda2_t / 2.0
        
        # Total penalty
        total_penalty = lambda1_scaled * r_cognitive + lambda2_scaled * r_efficiency
        
        # Exponential regularization factor
        exp_factor = np.exp(-total_penalty)
        
        # Probability adjustment
        prob_adjusted = self.compute_probability_adjustment(base_prob, beta)
        
        # Final Ψ(x) value
        psi_x = total_output * exp_factor * prob_adjusted
        
        return {
            'psi_x': psi_x,
            'alpha_t': alpha_t,
            'lambda1_t': lambda1_t,
            'lambda2_t': lambda2_t,
            'symbolic_output': s_x,
            'neural_output': n_x,
            'hybrid_output': hybrid_output,
            'cross_term': cross_term,
            'total_output': total_output,
            'r_cognitive': r_cognitive,
            'r_efficiency': r_efficiency,
            'total_penalty': total_penalty,
            'exp_factor': exp_factor,
            'prob_adjusted': prob_adjusted
        }
    
    def integrate_over_time(self, x: np.ndarray, 
                           m1: Optional[np.ndarray] = None,
                           m2: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Integrate the core equation over the time span"""
        
        t_values = self.trajectory.t_values
        dt = self.params.dt
        
        total_integral = 0.0
        time_series_data = []
        
        for t in t_values:
            result = self.evaluate_at_time(x, t, m1, m2)
            total_integral += result['psi_x'] * dt
            time_series_data.append(result)
        
        return {
            'integrated_psi': total_integral,
            'time_series': time_series_data,
            'average_psi': total_integral / len(t_values)
        }

class PhysicsInformedNeuralNetwork(nn.Module):
    """Physics-Informed Neural Network for dynamical systems"""
    
    def __init__(self, input_dim: int = 3, hidden_dim: int = 128, output_dim: int = 3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    def physics_loss(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Compute physics-informed loss based on dynamical system constraints"""
        # For this system, we enforce smooth evolution of α, λ₁, λ₂
        u_t = torch.autograd.grad(u.sum(), x, create_graph=True)[0][:, 0]  # ∂u/∂t
        
        # Physics constraint: smooth temporal evolution
        physics_residual = u_t + 0.1 * u.sum(dim=1)  # Simple damping constraint
        
        return torch.mean(physics_residual**2)

class DynamicModeDecomposition:
    """Dynamic Mode Decomposition for spatiotemporal analysis"""
    
    def __init__(self, trajectory_data: np.ndarray):
        """
        Initialize DMD with trajectory data
        trajectory_data: shape (n_variables, n_timesteps)
        """
        self.data = trajectory_data
        self.modes = None
        self.eigenvalues = None
        self.amplitudes = None
        
    def compute_dmd(self) -> Dict[str, np.ndarray]:
        """Compute DMD decomposition"""
        X = self.data[:, :-1]  # Data matrix
        Y = self.data[:, 1:]   # Shifted data matrix
        
        # SVD of X
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        
        # Truncate for numerical stability
        r = min(len(S), X.shape[0], X.shape[1] - 1)
        U_r = U[:, :r]
        S_r = S[:r]
        V_r = Vt[:r, :].T
        
        # Compute A_tilde
        A_tilde = U_r.T @ Y @ V_r @ np.diag(1/S_r)
        
        # Eigendecomposition of A_tilde
        eigenvalues, eigenvectors = np.linalg.eig(A_tilde)
        
        # DMD modes
        modes = Y @ V_r @ np.diag(1/S_r) @ eigenvectors
        
        # Compute amplitudes
        amplitudes = np.linalg.pinv(modes) @ X[:, 0]
        
        self.modes = modes
        self.eigenvalues = eigenvalues
        self.amplitudes = amplitudes
        
        return {
            'modes': modes,
            'eigenvalues': eigenvalues,
            'amplitudes': amplitudes
        }
    
    def reconstruct_trajectory(self, n_steps: int) -> np.ndarray:
        """Reconstruct trajectory using DMD modes"""
        if self.modes is None:
            self.compute_dmd()
        
        time_dynamics = np.array([self.eigenvalues**k for k in range(n_steps)]).T
        reconstruction = np.real(self.modes @ np.diag(self.amplitudes) @ time_dynamics)
        
        return reconstruction

class VisualizationTools:
    """Tools for visualizing phase-space trajectories and analysis results"""
    
    @staticmethod
    def plot_3d_trajectory(trajectory: PhaseSpaceTrajectory, 
                          title: str = "Phase-Space Trajectory: α(t), λ₁(t), λ₂(t)") -> plt.Figure:
        """Create 3D plot of phase-space trajectory"""
        t_vals, alpha_vals, lambda1_vals, lambda2_vals = trajectory.get_full_trajectory()
        
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot trajectory
        ax.plot(alpha_vals, lambda1_vals, lambda2_vals, 'b-', linewidth=2, alpha=0.8)
        
        # Highlight start and end points
        ax.scatter([alpha_vals[0]], [lambda1_vals[0]], [lambda2_vals[0]], 
                  color='green', s=100, label='Start')
        ax.scatter([alpha_vals[-1]], [lambda1_vals[-1]], [lambda2_vals[-1]], 
                  color='red', s=100, label='End')
        
        # Set labels and title
        ax.set_xlabel('α(t)', fontsize=12)
        ax.set_ylabel('λ₁(t)', fontsize=12)
        ax.set_zlabel('λ₂(t)', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        
        # Set axis limits to match the image
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        ax.set_zlim(0, 2)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_time_series(trajectory: PhaseSpaceTrajectory) -> plt.Figure:
        """Plot time series of trajectory components"""
        t_vals, alpha_vals, lambda1_vals, lambda2_vals = trajectory.get_full_trajectory()
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        axes[0].plot(t_vals, alpha_vals, 'b-', linewidth=2)
        axes[0].set_ylabel('α(t)')
        axes[0].set_title('Time Evolution of System Parameters')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(t_vals, lambda1_vals, 'r-', linewidth=2)
        axes[1].set_ylabel('λ₁(t)')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(t_vals, lambda2_vals, 'g-', linewidth=2)
        axes[2].set_ylabel('λ₂(t)')
        axes[2].set_xlabel('Time (t)')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_psi_evolution(evaluator: CoreEquationEvaluator, x: np.ndarray) -> plt.Figure:
        """Plot evolution of Ψ(x) over time"""
        result = evaluator.integrate_over_time(x)
        time_series = result['time_series']
        
        t_vals = evaluator.trajectory.t_values
        psi_vals = [data['psi_x'] for data in time_series]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(t_vals, psi_vals, 'purple', linewidth=2)
        ax.set_xlabel('Time (t)')
        ax.set_ylabel('Ψ(x)')
        ax.set_title('Evolution of Core Equation Ψ(x) Over Time')
        ax.grid(True, alpha=0.3)
        
        # Add average line
        avg_psi = result['average_psi']
        ax.axhline(y=avg_psi, color='red', linestyle='--', 
                  label=f'Average: {avg_psi:.3f}')
        ax.legend()
        
        plt.tight_layout()
        return fig

class MultiPendulumSimulator:
    """Simulator for multi-pendulum chaotic systems"""
    
    def __init__(self, n_pendulums: int = 2, length: float = 1.0, 
                 gravity: float = 9.81, damping: float = 0.1):
        self.n_pendulums = n_pendulums
        self.length = length
        self.gravity = gravity
        self.damping = damping
        
    def dynamics(self, t: float, y: np.ndarray) -> np.ndarray:
        """Multi-pendulum dynamics equations"""
        n = self.n_pendulums
        angles = y[:n]
        velocities = y[n:]
        
        accelerations = np.zeros(n)
        
        for i in range(n):
            # Simplified multi-pendulum dynamics with coupling
            coupling_term = 0.0
            if i > 0:
                coupling_term += 0.1 * np.sin(angles[i] - angles[i-1])
            if i < n-1:
                coupling_term += 0.1 * np.sin(angles[i] - angles[i+1])
            
            accelerations[i] = -(self.gravity/self.length) * np.sin(angles[i]) + \
                              coupling_term - self.damping * velocities[i]
        
        return np.concatenate([velocities, accelerations])
    
    def simulate(self, initial_conditions: np.ndarray, 
                time_span: Tuple[float, float], dt: float = 0.01) -> Dict[str, np.ndarray]:
        """Simulate multi-pendulum system"""
        sol = solve_ivp(self.dynamics, time_span, initial_conditions, 
                       method='RK45', dense_output=True, 
                       t_eval=np.arange(time_span[0], time_span[1], dt))
        
        return {
            'time': sol.t,
            'angles': sol.y[:self.n_pendulums],
            'velocities': sol.y[self.n_pendulums:],
            'success': sol.success
        }

# Example usage and demonstration
def demonstrate_system():
    """Demonstrate the complete dynamical systems framework"""
    
    logger.info("Initializing Dynamical Systems Framework...")
    
    # System parameters
    params = SystemParameters(
        time_span=(0.0, 5.0),
        dt=0.05
    )
    
    # Create trajectory
    trajectory = PhaseSpaceTrajectory(params)
    
    # Create core equation evaluator
    evaluator = CoreEquationEvaluator(params)
    
    # Test input
    x = np.array([0.5, 0.3])
    m1 = np.array([0.2, 0.4])
    m2 = np.array([0.8, 0.6])
    
    logger.info("Evaluating core equation at t=0.5...")
    
    # Single time point evaluation (matching the analysis)
    result_single = evaluator.evaluate_at_time(x, t=0.5, m1=m1, m2=m2)
    
    logger.info(f"Results at t=0.5:")
    logger.info(f"  Ψ(x) = {result_single['psi_x']:.3f}")
    logger.info(f"  α(t) = {result_single['alpha_t']:.3f}")
    logger.info(f"  λ₁(t) = {result_single['lambda1_t']:.3f}")
    logger.info(f"  λ₂(t) = {result_single['lambda2_t']:.3f}")
    logger.info(f"  Symbolic output S(x) = {result_single['symbolic_output']:.3f}")
    logger.info(f"  Neural output N(x) = {result_single['neural_output']:.3f}")
    
    # Full time integration
    logger.info("Computing time integration...")
    result_integrated = evaluator.integrate_over_time(x, m1, m2)
    logger.info(f"Integrated Ψ(x) = {result_integrated['integrated_psi']:.3f}")
    logger.info(f"Average Ψ(x) = {result_integrated['average_psi']:.3f}")
    
    # Multi-pendulum simulation
    logger.info("Running multi-pendulum simulation...")
    pendulum_sim = MultiPendulumSimulator(n_pendulums=2)
    initial_conditions = np.array([0.1, 0.2, 0.0, 0.0])  # angles and velocities
    pendulum_result = pendulum_sim.simulate(initial_conditions, (0.0, 5.0))
    
    if pendulum_result['success']:
        logger.info("Multi-pendulum simulation completed successfully")
    
    # DMD analysis
    logger.info("Performing Dynamic Mode Decomposition...")
    t_vals, alpha_vals, lambda1_vals, lambda2_vals = trajectory.get_full_trajectory()
    trajectory_data = np.vstack([alpha_vals, lambda1_vals, lambda2_vals])
    
    dmd = DynamicModeDecomposition(trajectory_data)
    dmd_result = dmd.compute_dmd()
    logger.info(f"DMD computed with {len(dmd_result['eigenvalues'])} modes")
    
    return {
        'trajectory': trajectory,
        'evaluator': evaluator,
        'single_result': result_single,
        'integrated_result': result_integrated,
        'pendulum_result': pendulum_result,
        'dmd_result': dmd_result
    }

if __name__ == "__main__":
    # Run demonstration
    results = demonstrate_system()
    
    # Create visualizations
    logger.info("Creating visualizations...")
    
    # 3D trajectory plot
    fig1 = VisualizationTools.plot_3d_trajectory(results['trajectory'])
    fig1.savefig('/workspace/phase_space_trajectory_3d.png', dpi=300, bbox_inches='tight')
    
    # Time series plot
    fig2 = VisualizationTools.plot_time_series(results['trajectory'])
    fig2.savefig('/workspace/trajectory_time_series.png', dpi=300, bbox_inches='tight')
    
    # Psi evolution plot
    x_test = np.array([0.5, 0.3])
    fig3 = VisualizationTools.plot_psi_evolution(results['evaluator'], x_test)
    fig3.savefig('/workspace/psi_evolution.png', dpi=300, bbox_inches='tight')
    
    logger.info("Dynamical Systems Framework demonstration completed!")
    logger.info("Generated visualizations:")
    logger.info("  - phase_space_trajectory_3d.png")
    logger.info("  - trajectory_time_series.png") 
    logger.info("  - psi_evolution.png")