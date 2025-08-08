import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
from scipy.integrate import odeint
import seaborn as sns

# Set style for better visualization
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class PhaseSpaceAnalyzer:
    def __init__(self):
        """Initialize the phase space analyzer for the hybrid symbolic-neural system."""
        self.t_max = 1.0
        self.n_points = 100
        
    def generate_trajectory(self):
        """Generate the phase-space trajectory based on the image description."""
        # Based on the image: α(t) goes from 0 to 1, λ1 and λ2 go from 2 to 0
        t = np.linspace(0, self.t_max, self.n_points)
        
        # Linear trajectory as shown in the image
        alpha_t = t  # α(t) increases linearly from 0 to 1
        lambda1_t = 2.0 * (1 - t)  # λ1 decreases linearly from 2 to 0
        lambda2_t = 2.0 * (1 - t)  # λ2 decreases linearly from 2 to 0
        
        return t, alpha_t, lambda1_t, lambda2_t
    
    def plot_phase_space(self, t, alpha_t, lambda1_t, lambda2_t):
        """Create the 3D phase-space trajectory plot."""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the trajectory
        ax.plot(alpha_t, lambda1_t, lambda2_t, 'b-', linewidth=3, label='Phase-Space Trajectory')
        
        # Mark start and end points
        ax.scatter([0], [2], [2], color='red', s=100, label='Start (t=0)')
        ax.scatter([1], [0], [0], color='green', s=100, label='End (t=1)')
        
        # Add a point for analysis (t=0.5)
        t_analysis = 0.5
        alpha_analysis = t_analysis
        lambda1_analysis = 2.0 * (1 - t_analysis)
        lambda2_analysis = 2.0 * (1 - t_analysis)
        ax.scatter([alpha_analysis], [lambda1_analysis], [lambda2_analysis], 
                  color='orange', s=150, label=f'Analysis Point (t={t_analysis})')
        
        # Set labels and title
        ax.set_xlabel('α(t)', fontsize=12)
        ax.set_ylabel('λ₁(t)', fontsize=12)
        ax.set_zlabel('λ₂(t)', fontsize=12)
        ax.set_title('Phase-Space Trajectory: α(t), λ₁(t), λ₂(t)', fontsize=14, fontweight='bold')
        
        # Set axis limits based on the image
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 2)
        ax.set_zlim(0, 2)
        
        # Add grid and legend
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.show()
        
        return alpha_analysis, lambda1_analysis, lambda2_analysis
    
    def core_equation_components(self, alpha_t, lambda1_t, lambda2_t):
        """Calculate components of the core equation Ψ(x)."""
        # Define symbolic and neural outputs (example values)
        S_x = 0.60  # Symbolic output (RK4 solution)
        N_x = 0.80  # Neural output (LSTM prediction)
        
        # Cross-term weight (from the equation)
        w_cross = 0.1
        
        # Hybrid output calculation
        hybrid_output = alpha_t * S_x + (1 - alpha_t) * N_x
        
        # Cross-term (simplified)
        cross_term = w_cross * (S_x * N_x - N_x * S_x)  # This would be 0 in this case
        
        # Regularization penalties
        R_cognitive = 0.25  # Cognitive penalty
        R_efficiency = 0.10  # Efficiency penalty
        
        # Exponential regularization term
        regularization = np.exp(-(lambda1_t * R_cognitive + lambda2_t * R_efficiency))
        
        # Probability term (example)
        P_H_E_beta = 0.70 * 1.4  # Base probability * bias
        
        return hybrid_output, regularization, P_H_E_beta
    
    def analyze_trajectory_point(self, t_point, alpha_t, lambda1_t, lambda2_t):
        """Analyze a specific point on the trajectory."""
        # Find the index closest to the desired time point
        t_array = np.linspace(0, self.t_max, self.n_points)
        idx = np.argmin(np.abs(t_array - t_point))
        
        alpha_val = alpha_t[idx]
        lambda1_val = lambda1_t[idx]
        lambda2_val = lambda2_t[idx]
        
        print(f"\n=== Analysis at t = {t_point} ===")
        print(f"α(t) = {alpha_val:.3f}")
        print(f"λ₁(t) = {lambda1_val:.3f}")
        print(f"λ₂(t) = {lambda2_val:.3f}")
        
        # Calculate core equation components
        hybrid_output, regularization, P_H_E_beta = self.core_equation_components(
            alpha_val, lambda1_val, lambda2_val
        )
        
        print(f"\nCore Equation Components:")
        print(f"Hybrid Output = {hybrid_output:.3f}")
        print(f"Regularization Factor = {regularization:.3f}")
        print(f"Probability Term = {P_H_E_beta:.3f}")
        
        # Calculate Ψ(x)
        Psi_x = hybrid_output * regularization * P_H_E_beta
        print(f"\nΨ(x) = {Psi_x:.3f}")
        
        return Psi_x
    
    def plot_component_evolution(self, t, alpha_t, lambda1_t, lambda2_t):
        """Plot the evolution of different components over time."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Parameter evolution
        axes[0, 0].plot(t, alpha_t, 'b-', linewidth=2, label='α(t)')
        axes[0, 0].plot(t, lambda1_t, 'r-', linewidth=2, label='λ₁(t)')
        axes[0, 0].plot(t, lambda2_t, 'g-', linewidth=2, label='λ₂(t)')
        axes[0, 0].set_xlabel('Time t')
        axes[0, 0].set_ylabel('Parameter Value')
        axes[0, 0].set_title('Parameter Evolution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Hybrid output evolution
        S_x = 0.60
        N_x = 0.80
        hybrid_outputs = alpha_t * S_x + (1 - alpha_t) * N_x
        axes[0, 1].plot(t, hybrid_outputs, 'purple', linewidth=2)
        axes[0, 1].set_xlabel('Time t')
        axes[0, 1].set_ylabel('Hybrid Output')
        axes[0, 1].set_title('Hybrid Output Evolution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Regularization evolution
        R_cognitive = 0.25
        R_efficiency = 0.10
        regularization = np.exp(-(lambda1_t * R_cognitive + lambda2_t * R_efficiency))
        axes[1, 0].plot(t, regularization, 'orange', linewidth=2)
        axes[1, 0].set_xlabel('Time t')
        axes[1, 0].set_ylabel('Regularization Factor')
        axes[1, 0].set_title('Regularization Evolution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Final Ψ(x) evolution
        P_H_E_beta = 0.70 * 1.4
        Psi_x_values = hybrid_outputs * regularization * P_H_E_beta
        axes[1, 1].plot(t, Psi_x_values, 'brown', linewidth=2)
        axes[1, 1].set_xlabel('Time t')
        axes[1, 1].set_ylabel('Ψ(x)')
        axes[1, 1].set_title('Final Output Ψ(x) Evolution')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def run_analysis(self):
        """Run the complete phase-space analysis."""
        print("Phase-Space Trajectory Analysis")
        print("=" * 40)
        
        # Generate trajectory
        t, alpha_t, lambda1_t, lambda2_t = self.generate_trajectory()
        
        # Plot phase space
        alpha_analysis, lambda1_analysis, lambda2_analysis = self.plot_phase_space(
            t, alpha_t, lambda1_t, lambda2_t
        )
        
        # Analyze specific points
        self.analyze_trajectory_point(0.0, alpha_t, lambda1_t, lambda2_t)  # Start
        self.analyze_trajectory_point(0.5, alpha_t, lambda1_t, lambda2_t)  # Midpoint
        self.analyze_trajectory_point(1.0, alpha_t, lambda1_t, lambda2_t)  # End
        
        # Plot component evolution
        self.plot_component_evolution(t, alpha_t, lambda1_t, lambda2_t)
        
        print("\n" + "=" * 40)
        print("Analysis Complete!")

if __name__ == "__main__":
    analyzer = PhaseSpaceAnalyzer()
    analyzer.run_analysis()