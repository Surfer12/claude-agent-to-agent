#!/usr/bin/env python3
"""
Hybrid Dynamical Systems Framework Demo
=======================================

This demo showcases the complete implementation of the hybrid dynamical systems
framework inspired by Ryan David Oates' work. It demonstrates:

1. The 3D phase-space trajectory visualization
2. The concrete single-time-step example
3. System analysis and insights
4. Integration with the core Ψ(x) expression

The framework implements:
Ψ(x) = ∫[ α(t) S(x) + [1−α(t)] N(x) + w_cross(S(m₁)N(m₂)−S(m₂)N(m₁)) ]
       × exp[−(λ₁ R_cognitive + λ₂ R_efficiency)] × P(H|E, β) dt
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any
import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.hybrid_system import HybridDynamicalSystem, HybridSystemConfig
from visualization.phase_space_plotter import PhaseSpacePlotter
from examples.concrete_example import ConcreteExample

def print_banner():
    """Print a banner for the demo."""
    print("=" * 80)
    print("HYBRID DYNAMICAL SYSTEMS FRAMEWORK DEMO")
    print("Inspired by Ryan David Oates' Work")
    print("=" * 80)
    print()

def demonstrate_system_setup():
    """Demonstrate the system setup and configuration."""
    print("1. SYSTEM SETUP")
    print("-" * 40)
    
    # Create configuration
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
    
    # Create system
    system = HybridDynamicalSystem(config)
    
    print(f"✓ System configured with:")
    print(f"  - Time range: [{config.t_start}, {config.t_end}]")
    print(f"  - Time step: {config.dt}")
    print(f"  - Parameter bounds: α∈{config.alpha_bounds}, λ₁∈{config.lambda1_bounds}, λ₂∈{config.lambda2_bounds}")
    print(f"  - Cross-coupling weight: {config.w_cross}")
    print(f"  - Expert bias: β = {config.beta}")
    print(f"  - Initial conditions: α={config.alpha_init}, λ₁={config.lambda1_init}, λ₂={config.lambda2_init}")
    print()
    
    return system

def demonstrate_trajectory_integration(system: HybridDynamicalSystem):
    """Demonstrate trajectory integration."""
    print("2. TRAJECTORY INTEGRATION")
    print("-" * 40)
    
    # Integrate the trajectory
    times, trajectory = system.integrate_trajectory()
    
    print(f"✓ Trajectory integrated successfully")
    print(f"  - Number of time points: {len(times)}")
    print(f"  - Trajectory shape: {trajectory.shape}")
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

def demonstrate_psi_evaluation(system: HybridDynamicalSystem):
    """Demonstrate Ψ(x) evaluation."""
    print("3. Ψ(x) EVALUATION")
    print("-" * 40)
    
    # Create sample input data
    x_sample = np.random.rand(100)
    
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
    print(f"  - Mean Ψ(x): {np.mean(psi_values):.3f}")
    print(f"  - Std Ψ(x): {np.std(psi_values):.3f}")
    print(f"  - Min Ψ(x): {np.min(psi_values):.3f}")
    print(f"  - Max Ψ(x): {np.max(psi_values):.3f}")
    print()

def demonstrate_trajectory_insights(system: HybridDynamicalSystem):
    """Demonstrate trajectory analysis and insights."""
    print("4. TRAJECTORY INSIGHTS")
    print("-" * 40)
    
    # Get trajectory insights
    insights = system.get_trajectory_insights()
    
    print("✓ Trajectory analysis completed")
    print()
    
    print("Trajectory Characteristics:")
    print(f"  - Start point: α={insights['start_point'][0]:.2f}, λ₁={insights['start_point'][1]:.2f}, λ₂={insights['start_point'][2]:.2f}")
    print(f"  - End point: α={insights['end_point'][0]:.2f}, λ₁={insights['end_point'][1]:.2f}, λ₂={insights['end_point'][2]:.2f}")
    print(f"  - Trajectory type: {insights['trajectory_type']}")
    print()
    
    print("Parameter Ranges:")
    print(f"  - α range: {insights['alpha_range']}")
    print(f"  - λ₁ range: {insights['lambda1_range']}")
    print(f"  - λ₂ range: {insights['lambda2_range']}")
    print()
    
    print("Evolution Characteristics:")
    for char_name, char_value in insights['evolution_characteristics'].items():
        status = "✓" if char_value else "✗"
        print(f"  {status} {char_name}")
    print()

def demonstrate_visualization(system: HybridDynamicalSystem):
    """Demonstrate visualization capabilities."""
    print("5. VISUALIZATION")
    print("-" * 40)
    
    # Create plotter
    plotter = PhaseSpacePlotter(system)
    
    # Create 3D trajectory plot
    print("Creating 3D phase-space trajectory plot...")
    fig, ax = plotter.plot_3d_trajectory(
        title="Phase-Space Trajectory: α(t), λ₁(t), λ₂(t)",
        save_path="phase_space_trajectory.png"
    )
    print("✓ 3D trajectory plot created and saved")
    
    # Create parameter evolution plot
    print("Creating parameter evolution plot...")
    fig, axes = plotter.plot_parameter_evolution()
    plt.savefig("parameter_evolution.png", dpi=300, bbox_inches='tight')
    print("✓ Parameter evolution plot created and saved")
    
    # Create comprehensive analysis
    print("Creating comprehensive trajectory analysis...")
    fig, ax = plotter.plot_trajectory_analysis()
    plt.savefig("trajectory_analysis.png", dpi=300, bbox_inches='tight')
    print("✓ Comprehensive analysis plot created and saved")
    
    print()
    print("Visualization files created:")
    print("  - phase_space_trajectory.png")
    print("  - parameter_evolution.png")
    print("  - trajectory_analysis.png")
    print()

def demonstrate_concrete_example():
    """Demonstrate the concrete single-time-step example."""
    print("6. CONCRETE EXAMPLE")
    print("-" * 40)
    
    # Create and run the concrete example
    example = ConcreteExample()
    results = example.run_concrete_example()
    
    print("✓ Concrete example completed successfully")
    print()
    
    # Show comparison with system evaluation
    comparison = example.compare_with_system_evaluation(results)
    
    print("✓ System evaluation comparison completed")
    print()

def demonstrate_mathematical_framework():
    """Demonstrate the mathematical framework."""
    print("7. MATHEMATICAL FRAMEWORK")
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
    print("8. APPLICATIONS")
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

def main():
    """Run the complete demo."""
    print_banner()
    
    try:
        # 1. System setup
        system = demonstrate_system_setup()
        
        # 2. Trajectory integration
        times, trajectory = demonstrate_trajectory_integration(system)
        
        # 3. Ψ(x) evaluation
        demonstrate_psi_evaluation(system)
        
        # 4. Trajectory insights
        demonstrate_trajectory_insights(system)
        
        # 5. Visualization
        demonstrate_visualization(system)
        
        # 6. Concrete example
        demonstrate_concrete_example()
        
        # 7. Mathematical framework
        demonstrate_mathematical_framework()
        
        # 8. Applications
        demonstrate_applications()
        
        print("=" * 80)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print()
        print("The hybrid dynamical systems framework has been demonstrated.")
        print("Check the generated visualization files for the 3D phase-space plots.")
        print()
        print("Key insights:")
        print("• The 3D trajectory shows the evolution of adaptive parameters")
        print("• Ψ(x) integrates symbolic and neural predictions with regularization")
        print("• The framework balances interpretability with performance")
        print("• This approach aligns with Ryan David Oates' vision for hybrid systems")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()