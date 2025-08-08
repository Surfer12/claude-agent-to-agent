#!/usr/bin/env python3
"""
UPOCF Demo Script

Usage: python upo_cf_demo.py --network demo.json --steps 200

Demonstrates consciousness detection and evolution simulation.
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import time

# Add upocf to path
sys.path.insert(0, str(Path(__file__).parent / "upocf"))

from upocf.core.consciousness import ConsciousnessFunction, psi
from upocf.core.integrator import RK4Integrator, consciousness_dynamics, hopf_bifurcation_dynamics
from upocf.core.detector import ConsciousnessDetector


def load_network_config(config_path: str) -> dict:
    """Load network configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        print("Creating default demo configuration...")
        return create_demo_config(config_path)


def create_demo_config(config_path: str) -> dict:
    """Create a default demo configuration."""
    config = {
        "network_size": 6,
        "initial_state": [0.5, -0.3, 0.8, -0.1, 0.4, -0.6],
        "dynamics": "consciousness",
        "parameters": {
            "alpha": 1.0,
            "beta": 0.1
        },
        "integration": {
            "dt": 0.001,
            "steps": 1000
        }
    }
    
    # Save config for future use
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Created demo config: {config_path}")
    return config


def simulate_consciousness_evolution(config: dict, steps: int) -> dict:
    """Simulate consciousness evolution using RK4 integration."""
    print("Initializing simulation...")
    
    # Setup
    initial_state = np.array(config["initial_state"])
    network_size = config["network_size"]
    dt = config["integration"]["dt"]
    
    # Choose dynamics
    dynamics_type = config.get("dynamics", "consciousness")
    if dynamics_type == "consciousness":
        dynamics = lambda t, x: consciousness_dynamics(
            t, x, 
            alpha=config["parameters"]["alpha"],
            beta=config["parameters"]["beta"]
        )
    elif dynamics_type == "hopf":
        dynamics = lambda t, x: hopf_bifurcation_dynamics(
            t, x,
            mu=config["parameters"].get("mu", 0.1),
            omega=config["parameters"].get("omega", 1.0)
        )
    else:
        raise ValueError(f"Unknown dynamics type: {dynamics_type}")
    
    # Run simulation
    integrator = RK4Integrator(dt=dt)
    
    print(f"Running simulation for {steps} steps...")
    start_time = time.time()
    
    results = integrator.simulate(
        initial_state=initial_state,
        dynamics=dynamics,
        steps=steps,
        save_trajectory=True
    )
    
    simulation_time = time.time() - start_time
    print(f"Simulation completed in {simulation_time:.3f} seconds")
    
    return results


def analyze_consciousness_trajectory(results: dict) -> dict:
    """Analyze consciousness levels throughout the trajectory."""
    psi_values = results["psi_values"]
    time_points = results["time_points"]
    
    analysis = {
        "mean_consciousness": float(np.mean(psi_values)),
        "max_consciousness": float(np.max(psi_values)),
        "min_consciousness": float(np.min(psi_values)),
        "consciousness_variance": float(np.var(psi_values)),
        "final_consciousness": float(psi_values[-1]),
        "initial_consciousness": float(psi_values[0]),
        "consciousness_change": float(psi_values[-1] - psi_values[0])
    }
    
    # Find periods of high consciousness (above mean + std)
    threshold = analysis["mean_consciousness"] + np.sqrt(analysis["consciousness_variance"])
    conscious_periods = []
    in_period = False
    period_start = 0
    
    for i, psi in enumerate(psi_values):
        if psi > threshold and not in_period:
            in_period = True
            period_start = time_points[i]
        elif psi <= threshold and in_period:
            in_period = False
            conscious_periods.append((period_start, time_points[i]))
    
    if in_period:  # Handle case where simulation ends in conscious period
        conscious_periods.append((period_start, time_points[-1]))
    
    analysis["conscious_periods"] = conscious_periods
    analysis["total_conscious_time"] = sum(end - start for start, end in conscious_periods)
    
    return analysis


def plot_results(results: dict, analysis: dict, save_path: str = None):
    """Plot consciousness evolution results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    time_points = results["time_points"]
    psi_values = results["psi_values"]
    trajectory = results["trajectory"]
    
    # Plot 1: Consciousness evolution over time
    axes[0, 0].plot(time_points, psi_values, 'b-', linewidth=2, label='Ψ(x)')
    axes[0, 0].axhline(y=analysis["mean_consciousness"], color='r', linestyle='--', 
                      label=f'Mean = {analysis["mean_consciousness"]:.3f}')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Consciousness Ψ(x)')
    axes[0, 0].set_title('Consciousness Evolution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: State space trajectory (first 2 dimensions)
    if trajectory.shape[1] >= 2:
        axes[0, 1].plot(trajectory[:, 0], trajectory[:, 1], 'g-', alpha=0.7)
        axes[0, 1].scatter(trajectory[0, 0], trajectory[0, 1], color='red', s=100, 
                          label='Start', zorder=5)
        axes[0, 1].scatter(trajectory[-1, 0], trajectory[-1, 1], color='blue', s=100, 
                          label='End', zorder=5)
        axes[0, 1].set_xlabel('x₁')
        axes[0, 1].set_ylabel('x₂')
        axes[0, 1].set_title('State Space Trajectory')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Consciousness histogram
    axes[1, 0].hist(psi_values, bins=50, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 0].axvline(x=analysis["mean_consciousness"], color='r', linestyle='--',
                      label=f'Mean = {analysis["mean_consciousness"]:.3f}')
    axes[1, 0].set_xlabel('Consciousness Ψ(x)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Consciousness Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: All state variables over time
    for i in range(min(trajectory.shape[1], 6)):  # Plot up to 6 variables
        axes[1, 1].plot(time_points, trajectory[:, i], label=f'x_{i+1}', alpha=0.7)
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('State Variables')
    axes[1, 1].set_title('Network State Evolution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()


def demonstrate_detection(results: dict):
    """Demonstrate consciousness detection capabilities."""
    print("\n" + "="*50)
    print("CONSCIOUSNESS DETECTION DEMONSTRATION")
    print("="*50)
    
    trajectory = results["trajectory"]
    psi_values = results["psi_values"]
    
    # Initialize detector
    detector = ConsciousnessDetector(threshold=0.5, network_size=trajectory.shape[1])
    
    # Test detection on sample states
    sample_indices = [0, len(trajectory)//4, len(trajectory)//2, 3*len(trajectory)//4, -1]
    
    print(f"{'Time':<8} {'State':<25} {'Ψ(x)':<10} {'Conscious':<10}")
    print("-" * 60)
    
    for i in sample_indices:
        state = trajectory[i]
        time_point = results["time_points"][i]
        is_conscious, psi_value = detector.detect(state)
        
        state_str = f"[{', '.join(f'{x:.2f}' for x in state[:3])}...]"
        print(f"{time_point:<8.3f} {state_str:<25} {psi_value:<10.3f} {str(is_conscious):<10}")
    
    # Batch detection statistics
    predictions, _ = detector.batch_detect(trajectory)
    conscious_ratio = np.mean(predictions)
    
    print(f"\nOverall Statistics:")
    print(f"- Conscious states: {np.sum(predictions)}/{len(predictions)} ({conscious_ratio:.1%})")
    print(f"- Mean consciousness: {np.mean(psi_values):.3f}")
    print(f"- Max consciousness: {np.max(psi_values):.3f}")


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="UPOCF Consciousness Framework Demo")
    parser.add_argument("--network", default="demo.json", 
                       help="Network configuration file (JSON)")
    parser.add_argument("--steps", type=int, default=200,
                       help="Number of integration steps")
    parser.add_argument("--plot", action="store_true",
                       help="Show plots")
    parser.add_argument("--save-plot", type=str,
                       help="Save plot to file")
    parser.add_argument("--output", type=str,
                       help="Save results to JSON file")
    
    args = parser.parse_args()
    
    print("UPOCF Framework Demo")
    print("=" * 50)
    
    # Load configuration
    config = load_network_config(args.network)
    print(f"Loaded network configuration: {config['network_size']} nodes")
    
    # Run simulation
    results = simulate_consciousness_evolution(config, args.steps)
    
    # Analyze results
    analysis = analyze_consciousness_trajectory(results)
    
    # Print summary
    print(f"\nSimulation Summary:")
    print(f"- Initial consciousness: {analysis['initial_consciousness']:.3f}")
    print(f"- Final consciousness: {analysis['final_consciousness']:.3f}")
    print(f"- Mean consciousness: {analysis['mean_consciousness']:.3f}")
    print(f"- Consciousness change: {analysis['consciousness_change']:+.3f}")
    print(f"- Conscious periods: {len(analysis['conscious_periods'])}")
    print(f"- Total conscious time: {analysis['total_conscious_time']:.3f}")
    
    # Demonstrate detection
    demonstrate_detection(results)
    
    # Plot results
    if args.plot or args.save_plot:
        plot_results(results, analysis, args.save_plot)
    
    # Save results
    if args.output:
        output_data = {
            'config': config,
            'results': {
                'time_points': results['time_points'].tolist(),
                'psi_values': results['psi_values'].tolist(),
                'final_state': results['final_state'].tolist()
            },
            'analysis': analysis
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to: {args.output}")
    
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()