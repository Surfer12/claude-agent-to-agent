"""
Advanced Framework Integration: PINNs, Neural-ODEs, and DMD
Following Ryan David Oates' hybrid dynamical systems methodology

This module extends the basic hybrid system with:
1. Physics-Informed Neural Networks (PINNs) for learning dynamics
2. Neural ODEs for adaptive trajectory generation
3. Dynamic Mode Decomposition (DMD) for mode analysis
4. Koopman operator theory integration
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from scipy.linalg import svd, pinv
from scipy.integrate import solve_ivp
from typing import Tuple, List, Dict, Optional, Callable, Any
import matplotlib.pyplot as plt
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from hybrid_phase_space_system import HybridDynamicalSystem, SystemParameters


@dataclass
class PINNParameters:
    """Parameters for Physics-Informed Neural Network"""
    hidden_layers: List[int] = None
    activation: str = 'tanh'
    learning_rate: float = 1e-3
    epochs: int = 5000
    physics_weight: float = 1.0
    data_weight: float = 1.0
    boundary_weight: float = 1.0
    
    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [50, 50, 50]


class PhysicsInformedNeuralNetwork(nn.Module):
    """
    PINN for learning the phase-space dynamics
    
    Learns the differential equations:
    dα/dt = f_α(α, λ₁, λ₂, t)
    dλ₁/dt = f_λ₁(α, λ₁, λ₂, t)  
    dλ₂/dt = f_λ₂(α, λ₁, λ₂, t)
    """
    
    def __init__(self, params: PINNParameters):
        super().__init__()
        self.params = params
        
        # Build network architecture
        layers = []
        input_dim = 4  # [α, λ₁, λ₂, t]
        
        # Input layer
        prev_dim = input_dim
        for hidden_dim in params.hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if params.activation == 'tanh':
                layers.append(nn.Tanh())
            elif params.activation == 'relu':
                layers.append(nn.ReLU())
            elif params.activation == 'sine':
                layers.append(SineActivation())
            prev_dim = hidden_dim
        
        # Output layer (3 outputs for the derivatives)
        layers.append(nn.Linear(prev_dim, 3))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier initialization for better training"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        x: [batch_size, 4] containing [α, λ₁, λ₂, t]
        returns: [batch_size, 3] containing [dα/dt, dλ₁/dt, dλ₂/dt]
        """
        return self.network(x)
    
    def physics_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute physics-informed loss based on known constraints
        """
        x.requires_grad_(True)
        
        # Get network predictions
        u = self.forward(x)  # [dα/dt, dλ₁/dt, dλ₂/dt]
        
        alpha, lambda1, lambda2, t = x[:, 0:1], x[:, 1:2], x[:, 2:3], x[:, 3:4]
        
        # Physics constraints (example based on Oates' methodology)
        # 1. Energy conservation-like constraint
        energy_constraint = (alpha * u[:, 0:1] + lambda1 * u[:, 1:2] + lambda2 * u[:, 2:3])
        
        # 2. Boundedness constraints (parameters should stay in [0,2])
        bound_penalty_alpha = torch.relu(-alpha) + torch.relu(alpha - 2)
        bound_penalty_lambda1 = torch.relu(-lambda1) + torch.relu(lambda1 - 2)
        bound_penalty_lambda2 = torch.relu(-lambda2) + torch.relu(lambda2 - 2)
        
        # 3. Coupling constraints (from Koopman theory)
        coupling_constraint = u[:, 0:1] + 0.1 * alpha * lambda1  # α-λ₁ coupling
        
        # Combine physics losses
        physics_loss = (
            torch.mean(energy_constraint**2) +
            torch.mean(bound_penalty_alpha**2) +
            torch.mean(bound_penalty_lambda1**2) +
            torch.mean(bound_penalty_lambda2**2) +
            torch.mean(coupling_constraint**2)
        )
        
        return physics_loss
    
    def data_loss(self, x: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Data fitting loss"""
        y_pred = self.forward(x)
        return torch.mean((y_pred - y_true)**2)


class SineActivation(nn.Module):
    """Sine activation function for periodic dynamics"""
    def forward(self, x):
        return torch.sin(x)


class NeuralODEDynamics(nn.Module):
    """
    Neural ODE for adaptive trajectory generation
    Integrates with the PINN to create learnable dynamics
    """
    
    def __init__(self, pinn: PhysicsInformedNeuralNetwork):
        super().__init__()
        self.pinn = pinn
    
    def forward(self, t: float, state: torch.Tensor) -> torch.Tensor:
        """
        ODE function for neural ODE integration
        state: [α, λ₁, λ₂]
        returns: [dα/dt, dλ₁/dt, dλ₂/dt]
        """
        batch_size = state.shape[0] if len(state.shape) > 1 else 1
        t_tensor = torch.full((batch_size, 1), t, dtype=state.dtype, device=state.device)
        
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        # Combine state and time
        x = torch.cat([state, t_tensor], dim=1)
        
        # Get derivatives from PINN
        derivatives = self.pinn(x)
        
        return derivatives.squeeze() if batch_size == 1 else derivatives


class DynamicModeDecomposition:
    """
    Dynamic Mode Decomposition for extracting coherent structures
    Following Oates' approach to mode analysis
    """
    
    def __init__(self, rank: Optional[int] = None):
        self.rank = rank
        self.modes = None
        self.eigenvalues = None
        self.amplitudes = None
        self.dynamics_matrix = None
    
    def fit(self, X: np.ndarray, Y: np.ndarray) -> 'DynamicModeDecomposition':
        """
        Fit DMD to trajectory data
        X: [n_features, n_snapshots-1] - current states
        Y: [n_features, n_snapshots-1] - next states
        """
        
        # SVD of X
        U, s, Vt = svd(X, full_matrices=False)
        
        # Determine rank
        if self.rank is None:
            # Use 95% energy criterion
            energy = np.cumsum(s**2) / np.sum(s**2)
            self.rank = np.argmax(energy >= 0.95) + 1
        
        # Truncate
        U_r = U[:, :self.rank]
        s_r = s[:self.rank]
        V_r = Vt[:self.rank, :].T
        
        # Build Atilde
        Atilde = U_r.T @ Y @ V_r @ np.diag(1/s_r)
        
        # Eigendecomposition of Atilde
        eigenvals, eigenvecs = np.linalg.eig(Atilde)
        
        # DMD modes
        self.modes = Y @ V_r @ np.diag(1/s_r) @ eigenvecs
        self.eigenvalues = eigenvals
        
        # Initial amplitudes
        self.amplitudes = pinv(self.modes) @ X[:, 0]
        
        # Store dynamics matrix
        self.dynamics_matrix = Atilde
        
        return self
    
    def predict(self, n_steps: int, dt: float = 1.0) -> np.ndarray:
        """Predict future states using DMD"""
        if self.modes is None:
            raise ValueError("DMD not fitted yet")
        
        time_dynamics = np.array([self.eigenvalues**i for i in range(n_steps)]).T
        prediction = np.real(self.modes @ np.diag(self.amplitudes) @ time_dynamics)
        
        return prediction
    
    def get_mode_analysis(self) -> Dict[str, Any]:
        """Get detailed mode analysis"""
        if self.modes is None:
            raise ValueError("DMD not fitted yet")
        
        # Compute growth rates and frequencies
        dt = 1.0  # Assume unit time step
        growth_rates = np.real(np.log(self.eigenvalues)) / dt
        frequencies = np.imag(np.log(self.eigenvalues)) / (2 * np.pi * dt)
        
        # Mode amplitudes
        mode_amplitudes = np.abs(self.amplitudes)
        
        return {
            'eigenvalues': self.eigenvalues,
            'growth_rates': growth_rates,
            'frequencies': frequencies,
            'mode_amplitudes': mode_amplitudes,
            'modes': self.modes,
            'rank': self.rank
        }


class KoopmanOperatorAnalysis:
    """
    Koopman operator analysis for understanding the nonlinear dynamics
    as a linear evolution in an infinite-dimensional space
    """
    
    def __init__(self, observables: List[Callable] = None):
        self.observables = observables or self._default_observables()
        self.koopman_modes = None
        self.koopman_eigenvalues = None
    
    def _default_observables(self) -> List[Callable]:
        """Default observable functions"""
        return [
            lambda x: x[:, 0],           # α
            lambda x: x[:, 1],           # λ₁  
            lambda x: x[:, 2],           # λ₂
            lambda x: x[:, 0]**2,        # α²
            lambda x: x[:, 1]**2,        # λ₁²
            lambda x: x[:, 2]**2,        # λ₂²
            lambda x: x[:, 0] * x[:, 1], # αλ₁
            lambda x: x[:, 0] * x[:, 2], # αλ₂
            lambda x: x[:, 1] * x[:, 2], # λ₁λ₂
        ]
    
    def compute_observables(self, trajectory: np.ndarray) -> np.ndarray:
        """Compute observable functions on trajectory data"""
        observables_data = []
        for obs in self.observables:
            obs_values = obs(trajectory)
            observables_data.append(obs_values)
        return np.array(observables_data)
    
    def fit(self, trajectory: np.ndarray) -> 'KoopmanOperatorAnalysis':
        """
        Fit Koopman analysis to trajectory data
        trajectory: [n_snapshots, n_features] - trajectory data
        """
        
        # Compute observables
        obs_data = self.compute_observables(trajectory)
        
        # Create shifted data for Koopman analysis
        X = obs_data[:, :-1]  # Current observables
        Y = obs_data[:, 1:]   # Next observables
        
        # Use DMD on the observable space
        dmd = DynamicModeDecomposition()
        dmd.fit(X, Y)
        
        self.koopman_modes = dmd.modes
        self.koopman_eigenvalues = dmd.eigenvalues
        
        return self
    
    def get_koopman_analysis(self) -> Dict[str, Any]:
        """Get Koopman operator analysis results"""
        if self.koopman_modes is None:
            raise ValueError("Koopman analysis not fitted yet")
        
        return {
            'modes': self.koopman_modes,
            'eigenvalues': self.koopman_eigenvalues,
            'n_observables': len(self.observables)
        }


class AdvancedHybridSystem(HybridDynamicalSystem):
    """
    Advanced hybrid system with PINN, Neural-ODE, and DMD capabilities
    Extends the basic system following Oates' methodology
    """
    
    def __init__(self, 
                 params: SystemParameters,
                 symbolic_solver,
                 neural_predictor,
                 pinn_params: PINNParameters = None):
        super().__init__(params, symbolic_solver, neural_predictor)
        
        self.pinn_params = pinn_params or PINNParameters()
        self.pinn = None
        self.neural_ode = None
        self.dmd = None
        self.koopman = None
        self.trained = False
    
    def train_pinn(self, 
                   training_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                   verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train the Physics-Informed Neural Network
        """
        
        # Initialize PINN
        self.pinn = PhysicsInformedNeuralNetwork(self.pinn_params)
        optimizer = optim.Adam(self.pinn.parameters(), lr=self.pinn_params.learning_rate)
        
        # Generate training data if not provided
        if training_data is None:
            training_data = self._generate_training_data()
        
        X_train, y_train = training_data
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train)
        
        # Training loop
        losses = {'total': [], 'physics': [], 'data': []}
        
        for epoch in range(self.pinn_params.epochs):
            optimizer.zero_grad()
            
            # Compute losses
            physics_loss = self.pinn.physics_loss(X_train)
            data_loss = self.pinn.data_loss(X_train, y_train)
            
            # Combined loss
            total_loss = (self.pinn_params.physics_weight * physics_loss + 
                         self.pinn_params.data_weight * data_loss)
            
            total_loss.backward()
            optimizer.step()
            
            # Store losses
            losses['total'].append(total_loss.item())
            losses['physics'].append(physics_loss.item())
            losses['data'].append(data_loss.item())
            
            if verbose and epoch % 500 == 0:
                print(f"Epoch {epoch}: Total Loss = {total_loss.item():.6f}, "
                      f"Physics = {physics_loss.item():.6f}, Data = {data_loss.item():.6f}")
        
        # Create Neural ODE
        self.neural_ode = NeuralODEDynamics(self.pinn)
        self.trained = True
        
        return losses
    
    def _generate_training_data(self, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Generate training data from the analytical dynamics"""
        
        # Generate random initial conditions and times
        np.random.seed(42)  # For reproducibility
        
        alphas = np.random.uniform(0, 2, n_samples)
        lambda1s = np.random.uniform(0, 2, n_samples)
        lambda2s = np.random.uniform(0, 2, n_samples)
        times = np.random.uniform(0, 10, n_samples)
        
        X = np.column_stack([alphas, lambda1s, lambda2s, times])
        
        # Compute derivatives using analytical dynamics
        y = np.zeros((n_samples, 3))
        for i in range(n_samples):
            state = np.array([alphas[i], lambda1s[i], lambda2s[i]])
            derivatives = self.phase_space_dynamics(times[i], state)
            y[i] = derivatives
        
        return X, y
    
    def generate_neural_ode_trajectory(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate trajectory using the trained Neural ODE"""
        
        if not self.trained:
            raise ValueError("PINN must be trained before generating Neural ODE trajectory")
        
        # Convert to torch
        initial_state = torch.FloatTensor(self.params.initial_state)
        
        # Time points
        t_eval = np.linspace(*self.params.t_span, 
                           int((self.params.t_span[1] - self.params.t_span[0]) / self.params.dt))
        
        # Simple Euler integration (for demonstration)
        trajectory = [initial_state.numpy()]
        current_state = initial_state
        
        for i in range(1, len(t_eval)):
            dt = t_eval[i] - t_eval[i-1]
            
            with torch.no_grad():
                derivatives = self.neural_ode(t_eval[i-1], current_state)
                current_state = current_state + dt * derivatives
                
                # Enforce bounds
                current_state = torch.clamp(current_state, 0, 2)
                
            trajectory.append(current_state.numpy())
        
        trajectory = np.array(trajectory)
        return t_eval, trajectory
    
    def fit_dmd(self, trajectory: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Fit Dynamic Mode Decomposition to trajectory data"""
        
        if trajectory is None:
            if self.trajectory_data is None:
                self.generate_trajectory()
            _, trajectory = self.trajectory_data
        
        # Prepare data for DMD
        X = trajectory[:-1].T  # Current states
        Y = trajectory[1:].T   # Next states
        
        # Fit DMD
        self.dmd = DynamicModeDecomposition()
        self.dmd.fit(X, Y)
        
        return self.dmd.get_mode_analysis()
    
    def fit_koopman(self, trajectory: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Fit Koopman operator analysis"""
        
        if trajectory is None:
            if self.trajectory_data is None:
                self.generate_trajectory()
            _, trajectory = self.trajectory_data
        
        # Fit Koopman analysis
        self.koopman = KoopmanOperatorAnalysis()
        self.koopman.fit(trajectory)
        
        return self.koopman.get_koopman_analysis()
    
    def comprehensive_analysis(self) -> Dict[str, Any]:
        """
        Perform comprehensive analysis combining all methods
        Returns a complete analysis following Oates' methodology
        """
        
        results = {}
        
        # 1. Generate base trajectory
        print("Generating base trajectory...")
        t_eval, trajectory = self.generate_trajectory()
        results['base_trajectory'] = {'time': t_eval, 'states': trajectory}
        
        # 2. Train PINN and generate Neural ODE trajectory
        print("Training PINN...")
        pinn_losses = self.train_pinn(verbose=False)
        results['pinn_training'] = pinn_losses
        
        print("Generating Neural ODE trajectory...")
        t_neural, trajectory_neural = self.generate_neural_ode_trajectory()
        results['neural_ode_trajectory'] = {'time': t_neural, 'states': trajectory_neural}
        
        # 3. DMD analysis
        print("Performing DMD analysis...")
        dmd_analysis = self.fit_dmd(trajectory)
        results['dmd_analysis'] = dmd_analysis
        
        # 4. Koopman analysis
        print("Performing Koopman analysis...")
        koopman_analysis = self.fit_koopman(trajectory)
        results['koopman_analysis'] = koopman_analysis
        
        # 5. Compare trajectories
        print("Comparing trajectories...")
        trajectory_comparison = self._compare_trajectories(
            trajectory, trajectory_neural, t_eval, t_neural
        )
        results['trajectory_comparison'] = trajectory_comparison
        
        print("Comprehensive analysis complete!")
        return results
    
    def _compare_trajectories(self, 
                            traj1: np.ndarray, 
                            traj2: np.ndarray,
                            t1: np.ndarray,
                            t2: np.ndarray) -> Dict[str, float]:
        """Compare two trajectories"""
        
        # Interpolate to common time grid
        t_common = np.linspace(max(t1[0], t2[0]), min(t1[-1], t2[-1]), 100)
        
        traj1_interp = np.array([np.interp(t_common, t1, traj1[:, i]) for i in range(3)]).T
        traj2_interp = np.array([np.interp(t_common, t2, traj2[:, i]) for i in range(3)]).T
        
        # Compute metrics
        mse = np.mean((traj1_interp - traj2_interp)**2)
        mae = np.mean(np.abs(traj1_interp - traj2_interp))
        
        # Component-wise errors
        alpha_error = np.mean((traj1_interp[:, 0] - traj2_interp[:, 0])**2)
        lambda1_error = np.mean((traj1_interp[:, 1] - traj2_interp[:, 1])**2)
        lambda2_error = np.mean((traj1_interp[:, 2] - traj2_interp[:, 2])**2)
        
        return {
            'mse': mse,
            'mae': mae,
            'alpha_mse': alpha_error,
            'lambda1_mse': lambda1_error,
            'lambda2_mse': lambda2_error
        }


def create_advanced_example_system() -> AdvancedHybridSystem:
    """Create an advanced example system for demonstration"""
    
    from hybrid_phase_space_system import PhysicsSymbolicSolver, LSTMNeuralPredictor
    
    params = SystemParameters(
        t_span=(0.0, 10.0),
        initial_state=np.array([2.0, 2.0, 0.0]),
        w_cross=0.1,
        beta=1.4
    )
    
    pinn_params = PINNParameters(
        hidden_layers=[64, 64, 32],
        activation='tanh',
        learning_rate=1e-3,
        epochs=2000,
        physics_weight=1.0,
        data_weight=1.0
    )
    
    symbolic_solver = PhysicsSymbolicSolver()
    neural_predictor = LSTMNeuralPredictor()
    
    return AdvancedHybridSystem(params, symbolic_solver, neural_predictor, pinn_params)


if __name__ == "__main__":
    # Demonstration of advanced capabilities
    print("=== Advanced Hybrid System Demonstration ===")
    print("Following Ryan David Oates' methodology\n")
    
    # Create advanced system
    system = create_advanced_example_system()
    
    # Run comprehensive analysis
    results = system.comprehensive_analysis()
    
    # Print summary
    print("\n=== Analysis Summary ===")
    print(f"PINN final loss: {results['pinn_training']['total'][-1]:.6f}")
    print(f"DMD rank: {results['dmd_analysis']['rank']}")
    print(f"Koopman observables: {results['koopman_analysis']['n_observables']}")
    print(f"Trajectory MSE: {results['trajectory_comparison']['mse']:.6f}")
    
    print("\nAdvanced analysis complete!")