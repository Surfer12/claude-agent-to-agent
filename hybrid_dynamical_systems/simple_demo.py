#!/usr/bin/env python3
"""
Simplified Hybrid Dynamical Systems Framework Demo
=================================================

This demo showcases the core concepts without requiring external dependencies.
It demonstrates the mathematical framework and provides a conceptual overview.
"""

import math
import random
from typing import List, Tuple, Dict, Any

class SimpleHybridSystem:
    """Simplified implementation of the hybrid dynamical system."""
    
    def __init__(self):
        # System parameters
        self.t_start = 0.0
        self.t_end = 10.0
        self.dt = 0.1
        
        # Parameter bounds
        self.alpha_bounds = (0.0, 2.0)
        self.lambda1_bounds = (0.0, 2.0)
        self.lambda2_bounds = (0.0, 2.0)
        
        # Initial conditions
        self.alpha_init = 2.0
        self.lambda1_init = 2.0
        self.lambda2_init = 0.0
        
        # Other parameters
        self.w_cross = 0.1
        self.beta = 1.4
        
    def parameter_dynamics(self, t: float, alpha: float, lambda1: float, lambda2: float) -> Tuple[float, float, float]:
        """
        Simplified parameter dynamics.
        
        Returns:
            Tuple of (dα/dt, dλ₁/dt, dλ₂/dt)
        """
        # Linear descent dynamics
        d_alpha = -0.2
        d_lambda1 = -0.15
        d_lambda2 = 0.2
        
        return d_alpha, d_lambda1, d_lambda2
    
    def integrate_trajectory(self) -> Tuple[List[float], List[List[float]]]:
        """Integrate the trajectory using simple Euler method."""
        times = []
        trajectory = []
        
        # Initial conditions
        alpha = self.alpha_init
        lambda1 = self.lambda1_init
        lambda2 = self.lambda2_init
        t = self.t_start
        
        while t <= self.t_end:
            times.append(t)
            trajectory.append([alpha, lambda1, lambda2])
            
            # Update parameters using Euler method
            d_alpha, d_lambda1, d_lambda2 = self.parameter_dynamics(t, alpha, lambda1, lambda2)
            
            alpha += d_alpha * self.dt
            lambda1 += d_lambda1 * self.dt
            lambda2 += d_lambda2 * self.dt
            
            # Ensure bounds
            alpha = max(self.alpha_bounds[0], min(self.alpha_bounds[1], alpha))
            lambda1 = max(self.lambda1_bounds[0], min(self.lambda1_bounds[1], lambda1))
            lambda2 = max(self.lambda2_bounds[0], min(self.lambda2_bounds[1], lambda2))
            
            t += self.dt
        
        return times, trajectory
    
    def symbolic_predictor(self, x: List[float], t: float) -> float:
        """Simplified symbolic predictor S(x)."""
        # Simulate RK4 physics solver
        return 0.6
    
    def neural_predictor(self, x: List[float], t: float) -> float:
        """Simplified neural predictor N(x)."""
        # Simulate LSTM prediction
        return 0.8
    
    def compute_hybrid_output(self, x: List[float], t: float, alpha: float) -> float:
        """Compute hybrid output: α(t) S(x) + [1−α(t)] N(x)."""
        alpha_normalized = alpha / self.alpha_bounds[1]
        symbolic_pred = self.symbolic_predictor(x, t)
        neural_pred = self.neural_predictor(x, t)
        
        return alpha_normalized * symbolic_pred + (1 - alpha_normalized) * neural_pred
    
    def compute_penalty_term(self, x: List[float], lambda1: float, lambda2: float) -> float:
        """Compute penalty term: exp[−(λ₁ R_cognitive + λ₂ R_efficiency)]."""
        # Simplified regularization
        r_cognitive = 0.25
        r_efficiency = 0.10
        
        lambda1_scaled = lambda1 / self.lambda1_bounds[1]
        lambda2_scaled = lambda2 / self.lambda2_bounds[1]
        
        penalty = math.exp(-(lambda1_scaled * r_cognitive + lambda2_scaled * r_efficiency))
        return penalty
    
    def probabilistic_bias(self, evidence: float, hypothesis: float) -> float:
        """Compute P(H|E, β) with expert bias β."""
        base_prob = 0.7  # P(H|E)
        expert_bias = math.pow(base_prob, self.beta)
        return max(0.0, min(1.0, expert_bias))
    
    def evaluate_psi(self, x: List[float], t: float, alpha: float, lambda1: float, lambda2: float) -> float:
        """
        Evaluate Ψ(x) at a specific time t with given parameters.
        
        Returns:
            float: The value of Ψ(x) at time t
        """
        # Hybrid output
        hybrid_output = self.compute_hybrid_output(x, t, alpha)
        
        # Penalty term
        penalty = self.compute_penalty_term(x, lambda1, lambda2)
        
        # Probabilistic bias
        evidence = sum(x) / len(x) if x else 0.5
        hypothesis = hybrid_output
        prob_bias = self.probabilistic_bias(evidence, hypothesis)
        
        # Final Ψ(x) evaluation
        psi_value = hybrid_output * penalty * prob_bias
        
        return psi_value

def print_banner():
    """Print a banner for the demo."""
    print("=" * 80)
    print("SIMPLIFIED HYBRID DYNAMICAL SYSTEMS FRAMEWORK DEMO")
    print("Inspired by Ryan David Oates' Work")
    print("=" * 80)
    print()

def demonstrate_system_setup():
    """Demonstrate the system setup and configuration."""
    print("1. SYSTEM SETUP")
    print("-" * 40)
    
    system = SimpleHybridSystem()
    
    print(f"✓ System configured with:")
    print(f"  - Time range: [{system.t_start}, {system.t_end}]")
    print(f"  - Time step: {system.dt}")
    print(f"  - Parameter bounds: α∈{system.alpha_bounds}, λ₁∈{system.lambda1_bounds}, λ₂∈{system.lambda2_bounds}")
    print(f"  - Cross-coupling weight: {system.w_cross}")
    print(f"  - Expert bias: β = {system.beta}")
    print(f"  - Initial conditions: α={system.alpha_init}, λ₁={system.lambda1_init}, λ₂={system.lambda2_init}")
    print()
    
    return system

def demonstrate_trajectory_integration(system: SimpleHybridSystem):
    """Demonstrate trajectory integration."""
    print("2. TRAJECTORY INTEGRATION")
    print("-" * 40)
    
    # Integrate the trajectory
    times, trajectory = system.integrate_trajectory()
    
    print(f"✓ Trajectory integrated successfully")
    print(f"  - Number of time points: {len(times)}")
    print(f"  - Time range: [{times[0]:.2f}, {times[-1]:.2f}]")
    print()
    
    # Show key trajectory points
    key_indices = [0, len(trajectory)//4, len(trajectory)//2, 3*len(trajectory)//4, -1]
    print("Key trajectory points:")
    print("Time    α(t)   λ₁(t)  λ₂(t)")
    print("-" * 30)
    
    for idx in key_indices:
        t = times[idx]
        alpha, lambda1, lambda2 = trajectory[idx]
        print(f"{t:6.1f}  {alpha:6.2f}  {lambda1:6.2f}  {lambda2:6.2f}")
    
    print()
    return times, trajectory

def demonstrate_psi_evaluation(system: SimpleHybridSystem):
    """Demonstrate Ψ(x) evaluation."""
    print("3. Ψ(x) EVALUATION")
    print("-" * 40)
    
    # Create sample input data
    x_sample = [random.random() for _ in range(100)]
    
    # Evaluate at different time points
    times, trajectory = system.integrate_trajectory()
    key_indices = [0, len(trajectory)//4, len(trajectory)//2, 3*len(trajectory)//4, -1]
    
    print("Ψ(x) evaluation at key time points:")
    print("Time    α(t)   λ₁(t)  λ₂(t)  Ψ(x)")
    print("-" * 40)
    
    psi_values = []
    for idx in key_indices:
        t = times[idx]
        alpha, lambda1, lambda2 = trajectory[idx]
        psi_value = system.evaluate_psi(x_sample, t, alpha, lambda1, lambda2)
        psi_values.append(psi_value)
        
        print(f"{t:6.1f}  {alpha:6.2f}  {lambda1:6.2f}  {lambda2:6.2f}  {psi_value:6.3f}")
    
    print()
    print(f"✓ Ψ(x) evaluated successfully")
    print(f"  - Mean Ψ(x): {sum(psi_values)/len(psi_values):.3f}")
    print(f"  - Min Ψ(x): {min(psi_values):.3f}")
    print(f"  - Max Ψ(x): {max(psi_values):.3f}")
    print()

def demonstrate_concrete_example():
    """Demonstrate the concrete single-time-step example."""
    print("4. CONCRETE EXAMPLE")
    print("-" * 40)
    
    # Step 1: Define the mid-curve point
    alpha = 1.0
    lambda1 = 1.5
    lambda2 = 0.5
    t = 5.0
    
    print(f"Step 1: Mid-curve point parameters")
    print(f"  α = {alpha:.1f}")
    print(f"  λ₁ = {lambda1:.1f}")
    print(f"  λ₂ = {lambda2:.1f}")
    print(f"  t = {t:.1f}")
    print()
    
    # Step 2: Symbolic and neural predictions
    S_x = 0.60  # Symbolic prediction (RK4 physics solver)
    N_x = 0.80  # Neural prediction (LSTM)
    
    print(f"Step 2: Symbolic and neural predictions")
    print(f"  S(x) = {S_x:.2f} (from RK4 physics solver)")
    print(f"  N(x) = {N_x:.2f} (from LSTM)")
    print()
    
    # Step 3: Hybrid output
    alpha_normalized = alpha / 2.0
    O_hybrid = alpha_normalized * S_x + (1 - alpha_normalized) * N_x
    
    print(f"Step 3: Hybrid output")
    print(f"  α_normalized = α/2 = {alpha_normalized:.2f}")
    print(f"  O_hybrid = {alpha_normalized:.2f}·{S_x:.2f} + {1-alpha_normalized:.2f}·{N_x:.2f} = {O_hybrid:.2f}")
    print()
    
    # Step 4: Penalty term
    R_cog = 0.25
    R_eff = 0.10
    lambda1_scaled = lambda1 / 2.0
    lambda2_scaled = lambda2 / 2.0
    penalty = math.exp(-(lambda1_scaled * R_cog + lambda2_scaled * R_eff))
    
    print(f"Step 4: Penalty term")
    print(f"  R_cognitive = {R_cog:.2f}")
    print(f"  R_efficiency = {R_eff:.2f}")
    print(f"  λ₁_scaled = {lambda1_scaled:.2f}")
    print(f"  λ₂_scaled = {lambda2_scaled:.2f}")
    print(f"  Penalty = exp[−({lambda1_scaled:.2f}·{R_cog:.2f} + {lambda2_scaled:.2f}·{R_eff:.2f})] ≈ {penalty:.4f}")
    print()
    
    # Step 5: Probabilistic bias
    P_H_given_E = 0.70
    beta = 1.4
    P_H_given_E_beta = math.pow(P_H_given_E, beta)
    
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
    
    print("INTERPRETATION:")
    print("Despite moderately strong regularization, the hybrid's balanced blend")
    print("plus high expert confidence yields a solid contribution to Ψ(x).")
    print()

def demonstrate_mathematical_framework():
    """Demonstrate the mathematical framework."""
    print("5. MATHEMATICAL FRAMEWORK")
    print("-" * 40)
    
    print("Core Expression:")
    print("Ψ(x) = ∫[ α(t) S(x) + [1−α(t)] N(x) + w_cross(S(m₁)N(m₂)−S(m₂)N(m₁)) ]")
    print("       × exp[−(λ₁ R_cognitive + λ₂ R_efficiency)] × P(H|E, β) dt")
    print()
    
    print("Components:")
    print("  • α(t): Time-dependent weight blending symbolic and neural components")
    print("  • S(x): Symbolic/physics-based prediction")
    print("  • N(x): Neural/data-driven prediction")
    print("  • w_cross: Cross-coupling weight for interaction terms")
    print("  • λ₁(t): Penalty weight for cognitive implausibility")
    print("  • λ₂(t): Penalty weight for computational efficiency")
    print("  • R_cognitive: Cognitive plausibility regularizer")
    print("  • R_efficiency: Computational efficiency regularizer")
    print("  • P(H|E, β): Probabilistic bias with expert knowledge")
    print()
    
    print("Differential Equations:")
    print("dα/dt = f₁(α, λ₁, λ₂, t)")
    print("dλ₁/dt = f₂(α, λ₁, λ₂, t)")
    print("dλ₂/dt = f₃(α, λ₁, λ₂, t)")
    print()
    
    print("✓ Mathematical framework explained")
    print()

def demonstrate_applications():
    """Demonstrate potential applications."""
    print("6. APPLICATIONS")
    print("-" * 40)
    
    print("Scientific Computing:")
    print("  • Hybrid PDE solvers combining symbolic and neural components")
    print("  • Adaptive mesh refinement based on learned dynamics")
    print()
    
    print("Control Systems:")
    print("  • Hybrid controllers balancing model-based and data-driven approaches")
    print("  • Adaptive control with learned parameter dynamics")
    print()
    
    print("Machine Learning:")
    print("  • Physics-informed neural networks with adaptive regularization")
    print("  • Interpretable AI systems with dynamic trust allocation")
    print()
    
    print("✓ Applications overview completed")
    print()

def demonstrate_key_insights():
    """Demonstrate key insights from the framework."""
    print("7. KEY INSIGHTS")
    print("-" * 40)
    
    print("Smart Thermostat Analogy:")
    print("Think of α(t), λ₁(t), λ₂(t) as a smart thermostat for a hybrid brain:")
    print("  • α(t): Dials how 'symbolic' vs 'neural' the thinking is at any instant")
    print("  • λ₁(t): Penalizes ideas that contradict basic physics or common sense")
    print("  • λ₂(t): Penalizes ideas that burn too much computational fuel")
    print()
    
    print("Trajectory Interpretation:")
    print("The 3D phase-space curve is the trace of that thermostat's settings over time.")
    print("By visualizing the path, you gain immediate insight into:")
    print("  • When the model trusts physics")
    print("  • When it relies on learned heuristics")
    print("  • How strictly it enforces plausibility and efficiency")
    print()
    
    print("Evolution Characteristics:")
    print("  • α(t): Decreases from symbolic to neural dominance")
    print("  • λ₁(t): Decreases from high to low cognitive penalty")
    print("  • λ₂(t): Increases from low to high efficiency penalty")
    print("  • Ψ(x): Shows how system output evolves along trajectory")
    print()

def main():
    """Run the complete simplified demo."""
    print_banner()
    
    try:
        # 1. System setup
        system = demonstrate_system_setup()
        
        # 2. Trajectory integration
        times, trajectory = demonstrate_trajectory_integration(system)
        
        # 3. Ψ(x) evaluation
        demonstrate_psi_evaluation(system)
        
        # 4. Concrete example
        demonstrate_concrete_example()
        
        # 5. Mathematical framework
        demonstrate_mathematical_framework()
        
        # 6. Applications
        demonstrate_applications()
        
        # 7. Key insights
        demonstrate_key_insights()
        
        print("=" * 80)
        print("SIMPLIFIED DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print()
        print("The hybrid dynamical systems framework has been demonstrated.")
        print()
        print("Key insights:")
        print("• The 3D trajectory shows the evolution of adaptive parameters")
        print("• Ψ(x) integrates symbolic and neural predictions with regularization")
        print("• The framework balances interpretability with performance")
        print("• This approach aligns with Ryan David Oates' vision for hybrid systems")
        print()
        print("For full visualization capabilities, install the required dependencies:")
        print("pip install numpy matplotlib scipy torch seaborn")
        print("Then run: python demo.py")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()