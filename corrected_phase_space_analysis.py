import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# Set style for better visualization
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class CorrectedPhaseSpaceAnalyzer:
    def __init__(self):
        """Initialize the corrected phase space analyzer based on the walkthrough."""
        self.t_max = 1.0
        self.n_points = 100
        
    def generate_corrected_trajectory(self):
        """Generate the corrected trajectory based on the walkthrough description."""
        # Based on the walkthrough: starts near (α≈2, λ₁≈2, λ₂≈0) and descends toward (α≈0, λ₁≈0, λ₂≈2)
        t = np.linspace(0, self.t_max, self.n_points)
        
        # Corrected trajectory equations
        alpha_t = 2.0 * (1 - t)  # α(t) decreases from 2 to 0
        lambda1_t = 2.0 * (1 - t)  # λ₁(t) decreases from 2 to 0
        lambda2_t = 2.0 * t  # λ₂(t) increases from 0 to 2
        
        return t, alpha_t, lambda1_t, lambda2_t
    
    def plot_corrected_phase_space(self, t, alpha_t, lambda1_t, lambda2_t):
        """Create the corrected 3D phase-space trajectory plot."""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the trajectory
        ax.plot(alpha_t, lambda1_t, lambda2_t, 'b-', linewidth=3, label='Phase-Space Trajectory')
        
        # Mark start and end points
        ax.scatter([2], [2], [0], color='red', s=100, label='Start (t=0)')
        ax.scatter([0], [0], [2], color='green', s=100, label='End (t=1)')
        
        # Add a point for analysis (t=0.5)
        t_analysis = 0.5
        alpha_analysis = 2.0 * (1 - t_analysis)
        lambda1_analysis = 2.0 * (1 - t_analysis)
        lambda2_analysis = 2.0 * t_analysis
        ax.scatter([alpha_analysis], [lambda1_analysis], [lambda2_analysis], 
                  color='orange', s=150, label=f'Analysis Point (t={t_analysis})')
        
        # Set labels and title
        ax.set_xlabel('α(t)', fontsize=12)
        ax.set_ylabel('λ₁(t)', fontsize=12)
        ax.set_zlabel('λ₂(t)', fontsize=12)
        ax.set_title('Corrected Phase-Space Trajectory: α(t), λ₁(t), λ₂(t)', fontsize=14, fontweight='bold')
        
        # Set axis limits based on the walkthrough
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        ax.set_zlim(0, 2)
        
        # Add grid and legend
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.show()
        
        return alpha_analysis, lambda1_analysis, lambda2_analysis
    
    def calculate_core_equation_components(self, alpha_t, lambda1_t, lambda2_t):
        """Calculate components of the core equation Ψ(x) based on the walkthrough."""
        # Define symbolic and neural outputs (from walkthrough example)
        S_x = 0.60  # Symbolic output (RK4 physics solver)
        N_x = 0.80  # Neural output (LSTM)
        
        # Cross-term weight (from the equation)
        w_cross = 0.1
        
        # Hybrid output calculation with normalization
        alpha_normalized = alpha_t / 2.0  # Normalize to [0,1] range
        hybrid_output = alpha_normalized * S_x + (1 - alpha_normalized) * N_x
        
        # Cross-term (simplified)
        cross_term = w_cross * (S_x * N_x - N_x * S_x)  # This would be 0 in this case
        
        # Regularization penalties (from walkthrough)
        R_cognitive = 0.25  # Cognitive penalty
        R_efficiency = 0.10  # Efficiency penalty
        
        # Scale lambda values to [0,1] range
        lambda1_scaled = lambda1_t / 2.0
        lambda2_scaled = lambda2_t / 2.0
        
        # Exponential regularization term
        regularization = np.exp(-(lambda1_scaled * R_cognitive + lambda2_scaled * R_efficiency))
        
        # Probability term (from walkthrough)
        P_H_E = 0.70  # Base probability
        beta = 1.4     # Expert bias
        P_H_E_beta = P_H_E * beta
        
        return hybrid_output, regularization, P_H_E_beta, alpha_normalized, lambda1_scaled, lambda2_scaled
    
    def analyze_corrected_trajectory_point(self, t_point, alpha_t, lambda1_t, lambda2_t):
        """Analyze a specific point on the corrected trajectory."""
        # Find the index closest to the desired time point
        t_array = np.linspace(0, self.t_max, self.n_points)
        idx = np.argmin(np.abs(t_array - t_point))
        
        alpha_val = alpha_t[idx]
        lambda1_val = lambda1_t[idx]
        lambda2_val = lambda2_t[idx]
        
        print(f"\n=== Corrected Analysis at t = {t_point} ===")
        print(f"α(t) = {alpha_val:.3f}")
        print(f"λ₁(t) = {lambda1_val:.3f}")
        print(f"λ₂(t) = {lambda2_val:.3f}")
        
        # Calculate core equation components
        hybrid_output, regularization, P_H_E_beta, alpha_norm, lambda1_scaled, lambda2_scaled = self.calculate_core_equation_components(
            alpha_val, lambda1_val, lambda2_val
        )
        
        print(f"\nCore Equation Components:")
        print(f"α_normalized = {alpha_norm:.3f}")
        print(f"Hybrid Output = {hybrid_output:.3f}")
        print(f"λ₁_scaled = {lambda1_scaled:.3f}")
        print(f"λ₂_scaled = {lambda2_scaled:.3f}")
        print(f"Regularization Factor = {regularization:.3f}")
        print(f"Probability Term = {P_H_E_beta:.3f}")
        
        # Calculate Ψ(x)
        Psi_x = hybrid_output * regularization * P_H_E_beta
        print(f"\nΨ(x) = {Psi_x:.3f}")
        
        return Psi_x
    
    def walkthrough_example_analysis(self):
        """Perform the exact analysis from the walkthrough example."""
        print("\n" + "="*60)
        print("WALKTHROUGH EXAMPLE ANALYSIS")
        print("="*60)
        
        # Example point from walkthrough: α≈1.0, λ₁≈1.5, λ₂≈0.5
        alpha_example = 1.0
        lambda1_example = 1.5
        lambda2_example = 0.5
        
        print(f"Example Point: α={alpha_example}, λ₁={lambda1_example}, λ₂={lambda2_example}")
        
        # 1. Symbolic and neural predictions
        S_x = 0.60  # from RK4 physics solver
        N_x = 0.80  # from LSTM
        print(f"\n1. Symbolic and neural predictions:")
        print(f"   S(x) = {S_x} (from RK4 physics solver)")
        print(f"   N(x) = {N_x} (from LSTM)")
        
        # 2. Hybrid output
        alpha_normalized = alpha_example / 2.0
        O_hybrid = alpha_normalized * S_x + (1 - alpha_normalized) * N_x
        print(f"\n2. Hybrid output:")
        print(f"   α_normalized = α/2 = {alpha_normalized}")
        print(f"   O_hybrid = {alpha_normalized:.1f}·{S_x} + {1-alpha_normalized:.1f}·{N_x} = {O_hybrid:.3f}")
        
        # 3. Penalty term
        R_cog = 0.25
        R_eff = 0.10
        lambda1_scaled = lambda1_example / 2.0
        lambda2_scaled = lambda2_example / 2.0
        penalty = np.exp(-(lambda1_scaled * R_cog + lambda2_scaled * R_eff))
        print(f"\n3. Penalty term:")
        print(f"   R_cog = {R_cog}, R_eff = {R_eff}")
        print(f"   λ₁_scaled = {lambda1_example}/2 = {lambda1_scaled}")
        print(f"   λ₂_scaled = {lambda2_example}/2 = {lambda2_scaled}")
        print(f"   Penalty = exp[−({lambda1_scaled:.2f}·{R_cog} + {lambda2_scaled:.2f}·{R_eff})] ≈ {penalty:.4f}")
        
        # 4. Probabilistic bias
        P_H_E = 0.70
        beta = 1.4
        P_H_E_beta = P_H_E * beta
        print(f"\n4. Probabilistic bias:")
        print(f"   P(H|E) = {P_H_E}, β = {beta} ⇒ P(H|E,β) = {P_H_E_beta:.2f}")
        
        # 5. Contribution to integral
        Psi_t = O_hybrid * penalty * P_H_E_beta
        print(f"\n5. Contribution to integral:")
        print(f"   Ψ_t(x) = {O_hybrid:.3f}·{penalty:.4f}·{P_H_E_beta:.2f} ≈ {Psi_t:.3f}")
        
        print(f"\nInterpretation: Despite moderately strong regularization, the hybrid's balanced blend plus high expert confidence yields a solid contribution to Ψ(x).")
        
        return Psi_t
    
    def plot_component_evolution_corrected(self, t, alpha_t, lambda1_t, lambda2_t):
        """Plot the evolution of different components over time with corrected trajectory."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Parameter evolution
        axes[0, 0].plot(t, alpha_t, 'b-', linewidth=2, label='α(t)')
        axes[0, 0].plot(t, lambda1_t, 'r-', linewidth=2, label='λ₁(t)')
        axes[0, 0].plot(t, lambda2_t, 'g-', linewidth=2, label='λ₂(t)')
        axes[0, 0].set_xlabel('Time t')
        axes[0, 0].set_ylabel('Parameter Value')
        axes[0, 0].set_title('Corrected Parameter Evolution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Hybrid output evolution
        S_x = 0.60
        N_x = 0.80
        alpha_normalized = alpha_t / 2.0
        hybrid_outputs = alpha_normalized * S_x + (1 - alpha_normalized) * N_x
        axes[0, 1].plot(t, hybrid_outputs, 'purple', linewidth=2)
        axes[0, 1].set_xlabel('Time t')
        axes[0, 1].set_ylabel('Hybrid Output')
        axes[0, 1].set_title('Hybrid Output Evolution (Corrected)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Regularization evolution
        R_cognitive = 0.25
        R_efficiency = 0.10
        lambda1_scaled = lambda1_t / 2.0
        lambda2_scaled = lambda2_t / 2.0
        regularization = np.exp(-(lambda1_scaled * R_cognitive + lambda2_scaled * R_efficiency))
        axes[1, 0].plot(t, regularization, 'orange', linewidth=2)
        axes[1, 0].set_xlabel('Time t')
        axes[1, 0].set_ylabel('Regularization Factor')
        axes[1, 0].set_title('Regularization Evolution (Corrected)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Final Ψ(x) evolution
        P_H_E_beta = 0.70 * 1.4
        Psi_x_values = hybrid_outputs * regularization * P_H_E_beta
        axes[1, 1].plot(t, Psi_x_values, 'brown', linewidth=2)
        axes[1, 1].set_xlabel('Time t')
        axes[1, 1].set_ylabel('Ψ(x)')
        axes[1, 1].set_title('Final Output Ψ(x) Evolution (Corrected)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def run_corrected_analysis(self):
        """Run the complete corrected phase-space analysis."""
        print("Corrected Phase-Space Trajectory Analysis")
        print("Based on Walkthrough Interpretation")
        print("=" * 60)
        
        # Generate corrected trajectory
        t, alpha_t, lambda1_t, lambda2_t = self.generate_corrected_trajectory()
        
        # Plot corrected phase space
        alpha_analysis, lambda1_analysis, lambda2_analysis = self.plot_corrected_phase_space(
            t, alpha_t, lambda1_t, lambda2_t
        )
        
        # Walkthrough example analysis
        self.walkthrough_example_analysis()
        
        # Analyze specific points
        self.analyze_corrected_trajectory_point(0.0, alpha_t, lambda1_t, lambda2_t)  # Start
        self.analyze_corrected_trajectory_point(0.5, alpha_t, lambda1_t, lambda2_t)  # Midpoint
        self.analyze_corrected_trajectory_point(1.0, alpha_t, lambda1_t, lambda2_t)  # End
        
        # Plot component evolution
        self.plot_component_evolution_corrected(t, alpha_t, lambda1_t, lambda2_t)
        
        print("\n" + "=" * 60)
        print("Corrected Analysis Complete!")
        print("\nKey Insights from Walkthrough:")
        print("- Trajectory shows gradual trade-off from symbolic to neural control")
        print("- Regularization shifts from cognitive plausibility to efficiency")
        print("- Linear path suggests constrained or weakly chaotic regime")
        print("- Integration over trajectory captures system's adaptive evolution")

if __name__ == "__main__":
    analyzer = CorrectedPhaseSpaceAnalyzer()
    analyzer.run_corrected_analysis()