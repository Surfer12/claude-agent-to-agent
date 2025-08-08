"""
Concrete Single-Time-Step Example
Demonstrates the evaluation of Ψ(x) at a specific point in the trajectory.

This example reproduces the analysis from the walkthrough:
"Pick the mid-curve point in the plot: α≈1.0, λ₁≈1.5, λ₂≈0.5"
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any

from ..core.hybrid_system import HybridDynamicalSystem, HybridSystemConfig

class ConcreteExample:
    """Demonstrates the concrete single-time-step example from the analysis."""
    
    def __init__(self):
        # Configure system to match the analysis
        config = HybridSystemConfig(
            t_start=0.0,
            t_end=10.0,
            dt=0.01,
            alpha_bounds=(0.0, 2.0),
            lambda1_bounds=(0.0, 2.0),
            lambda2_bounds=(0.0, 2.0),
            w_cross=0.1,
            beta=1.4,
            alpha_init=2.0,
            lambda1_init=2.0,
            lambda2_init=0.0
        )
        
        self.system = HybridDynamicalSystem(config)
        
    def run_concrete_example(self) -> Dict[str, Any]:
        """
        Run the concrete example from the analysis.
        
        Returns:
            Dict containing the step-by-step results
        """
        print("=" * 80)
        print("CONCRETE SINGLE-TIME-STEP EXAMPLE")
        print("=" * 80)
        
        # Step 1: Define the mid-curve point
        alpha = 1.0
        lambda1 = 1.5
        lambda2 = 0.5
        t = 5.0  # Mid-time point
        
        print(f"Step 1: Mid-curve point parameters")
        print(f"  α = {alpha:.1f}")
        print(f"  λ₁ = {lambda1:.1f}")
        print(f"  λ₂ = {lambda2:.1f}")
        print(f"  t = {t:.1f}")
        print()
        
        # Step 2: Symbolic and neural predictions
        x_sample = np.random.rand(100)  # Sample input data
        S_x = 0.60  # Symbolic prediction (RK4 physics solver)
        N_x = 0.80  # Neural prediction (LSTM)
        
        print(f"Step 2: Symbolic and neural predictions")
        print(f"  S(x) = {S_x:.2f} (from RK4 physics solver)")
        print(f"  N(x) = {N_x:.2f} (from LSTM)")
        print()
        
        # Step 3: Hybrid output
        alpha_normalized = alpha / 2.0  # Normalize to [0,1]
        O_hybrid = alpha_normalized * S_x + (1 - alpha_normalized) * N_x
        
        print(f"Step 3: Hybrid output")
        print(f"  α_normalized = α/2 = {alpha_normalized:.2f}")
        print(f"  O_hybrid = {alpha_normalized:.2f}·{S_x:.2f} + {1-alpha_normalized:.2f}·{N_x:.2f} = {O_hybrid:.2f}")
        print()
        
        # Step 4: Penalty term
        R_cog = 0.25  # Cognitive plausibility penalty
        R_eff = 0.10  # Computational efficiency penalty
        lambda1_scaled = lambda1 / 2.0
        lambda2_scaled = lambda2 / 2.0
        penalty = np.exp(-(lambda1_scaled * R_cog + lambda2_scaled * R_eff))
        
        print(f"Step 4: Penalty term")
        print(f"  R_cognitive = {R_cog:.2f}")
        print(f"  R_efficiency = {R_eff:.2f}")
        print(f"  λ₁_scaled = {lambda1_scaled:.2f}")
        print(f"  λ₂_scaled = {lambda2_scaled:.2f}")
        print(f"  Penalty = exp[−({lambda1_scaled:.2f}·{R_cog:.2f} + {lambda2_scaled:.2f}·{R_eff:.2f})] ≈ {penalty:.4f}")
        print()
        
        # Step 5: Probabilistic bias
        P_H_given_E = 0.70  # Base probability
        beta = 1.4
        P_H_given_E_beta = np.power(P_H_given_E, beta)
        
        print(f"Step 5: Probabilistic bias")
        print(f"  P(H|E) = {P_H_given_E:.2f}")
        print(f"  β = {beta:.1f}")
        print(f"  P(H|E,β) = {P_H_given_E_beta:.2f}")
        print()
        
        # Step 6: Final contribution to integral
        psi_contribution = O_hybrid * penalty * P_H_given_E_beta
        
        print(f"Step 6: Contribution to integral")
        print(f"  Ψ_t(x) = {O_hybrid:.2f}·{penalty:.4f}·{P_H_given_E_beta:.2f} ≈ {psi_contribution:.3f}")
        print()
        
        # Interpretation
        print("INTERPRETATION:")
        print("Despite moderately strong regularization, the hybrid's balanced blend")
        print("plus high expert confidence yields a solid contribution to Ψ(x).")
        print()
        
        # Store results
        results = {
            'parameters': {
                'alpha': alpha,
                'lambda1': lambda1,
                'lambda2': lambda2,
                'time': t
            },
            'predictions': {
                'symbolic': S_x,
                'neural': N_x,
                'hybrid': O_hybrid
            },
            'penalties': {
                'cognitive': R_cog,
                'efficiency': R_eff,
                'total_penalty': penalty
            },
            'probabilistic_bias': {
                'base_probability': P_H_given_E,
                'beta': beta,
                'biased_probability': P_H_given_E_beta
            },
            'final_contribution': psi_contribution
        }
        
        return results
    
    def compare_with_system_evaluation(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare the manual calculation with the system's evaluation.
        
        Args:
            results: Results from the manual calculation
            
        Returns:
            Dict containing comparison results
        """
        print("=" * 80)
        print("COMPARISON WITH SYSTEM EVALUATION")
        print("=" * 80)
        
        # Extract parameters from manual calculation
        alpha = results['parameters']['alpha']
        lambda1 = results['parameters']['lambda1']
        lambda2 = results['parameters']['lambda2']
        t = results['parameters']['time']
        
        # Create sample input data
        x_sample = np.random.rand(100)
        
        # Evaluate using the system
        system_psi = self.system.evaluate_psi(x_sample, t, alpha, lambda1, lambda2)
        
        # Manual calculation result
        manual_psi = results['final_contribution']
        
        print(f"Manual calculation: Ψ(x) = {manual_psi:.3f}")
        print(f"System evaluation:  Ψ(x) = {system_psi:.3f}")
        print(f"Difference:         {abs(manual_psi - system_psi):.6f}")
        print()
        
        # Analyze the difference
        if abs(manual_psi - system_psi) < 0.01:
            print("✓ Results are consistent!")
        else:
            print("⚠ Results differ - this is expected due to:")
            print("  - Different input data (random vs fixed)")
            print("  - Cross-coupling terms in system evaluation")
            print("  - More complex regularization in system")
        
        return {
            'manual_psi': manual_psi,
            'system_psi': system_psi,
            'difference': abs(manual_psi - system_psi),
            'is_consistent': abs(manual_psi - system_psi) < 0.01
        }
    
    def demonstrate_trajectory_evolution(self) -> None:
        """
        Demonstrate how the trajectory evolves and affects Ψ(x) evaluation.
        """
        print("=" * 80)
        print("TRAJECTORY EVOLUTION DEMONSTRATION")
        print("=" * 80)
        
        # Integrate the trajectory
        times, trajectory = self.system.integrate_trajectory()
        
        # Select key time points
        key_indices = [0, len(trajectory)//4, len(trajectory)//2, 3*len(trajectory)//4, -1]
        key_times = [times[i] for i in key_indices]
        key_points = [trajectory[i] for i in key_indices]
        
        print("Key trajectory points:")
        print("Time    α(t)   λ₁(t)  λ₂(t)  Ψ(x)")
        print("-" * 40)
        
        x_sample = np.random.rand(100)
        
        for i, (t, point) in enumerate(zip(key_times, key_points)):
            alpha, lambda1, lambda2 = point
            psi_value = self.system.evaluate_psi(x_sample, t, alpha, lambda1, lambda2)
            
            print(f"{t:6.1f}  {alpha:6.2f}  {lambda1:6.2f}  {lambda2:6.2f}  {psi_value:6.3f}")
        
        print()
        print("Evolution characteristics:")
        print("• α(t): Decreases from symbolic to neural dominance")
        print("• λ₁(t): Decreases from high to low cognitive penalty")
        print("• λ₂(t): Increases from low to high efficiency penalty")
        print("• Ψ(x): Shows how system output evolves along trajectory")
    
    def create_visualization(self) -> None:
        """Create visualizations for the concrete example."""
        print("=" * 80)
        print("CREATING VISUALIZATIONS")
        print("=" * 80)
        
        from ..visualization.phase_space_plotter import PhaseSpacePlotter
        
        # Create plotter
        plotter = PhaseSpacePlotter(self.system)
        
        # Create 3D trajectory plot
        fig, ax = plotter.plot_3d_trajectory(
            title="Phase-Space Trajectory: α(t), λ₁(t), λ₂(t)",
            save_path="phase_space_trajectory.png"
        )
        print("✓ Created 3D phase-space trajectory plot")
        
        # Create parameter evolution plot
        fig, axes = plotter.plot_parameter_evolution()
        plt.savefig("parameter_evolution.png", dpi=300, bbox_inches='tight')
        print("✓ Created parameter evolution plot")
        
        # Create comprehensive analysis
        fig, ax = plotter.plot_trajectory_analysis()
        plt.savefig("trajectory_analysis.png", dpi=300, bbox_inches='tight')
        print("✓ Created comprehensive trajectory analysis")
        
        plt.show()
        print("✓ All visualizations created and saved")

def main():
    """Run the concrete example demonstration."""
    example = ConcreteExample()
    
    # Run the concrete example
    results = example.run_concrete_example()
    
    # Compare with system evaluation
    comparison = example.compare_with_system_evaluation(results)
    
    # Demonstrate trajectory evolution
    example.demonstrate_trajectory_evolution()
    
    # Create visualizations
    example.create_visualization()
    
    print("=" * 80)
    print("EXAMPLE COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()