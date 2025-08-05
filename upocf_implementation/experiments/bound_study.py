#!/usr/bin/env python3
"""
Empirical Bounds Study for |Ψ⁽⁵⁾|

Generates 10³ random trajectories to establish empirical bounds for the 5th derivative
of the consciousness function, replacing the arbitrary "M₅ = 2" claim.

Usage: python bound_study.py --nodes 8 --samples 1000 --output bounds_report.json
"""

import argparse
import json
import numpy as np
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple
import pandas as pd

# Add upocf to path
sys.path.insert(0, str(Path(__file__).parent.parent / "upocf"))

from upocf.core.consciousness import ConsciousnessFunction
from upocf.core.integrator import RK4Integrator, consciousness_dynamics


class BoundsStudy:
    """
    Empirical study of |Ψ⁽⁵⁾| bounds across random trajectories.
    """
    
    def __init__(self, network_size: int, seed: int = 42):
        """
        Initialize bounds study.
        
        Args:
            network_size: Number of nodes in networks to study
            seed: Random seed for reproducibility
        """
        self.network_size = network_size
        self.seed = seed
        self.key = random.PRNGKey(seed)
        self.consciousness_fn = ConsciousnessFunction(network_size, use_fast_phi=True)
        
    def generate_random_trajectories(self, 
                                   n_samples: int,
                                   trajectory_length: int = 100,
                                   dt: float = 0.01) -> List[jnp.ndarray]:
        """
        Generate random initial conditions and simulate trajectories.
        
        Args:
            n_samples: Number of random trajectories to generate
            trajectory_length: Number of steps per trajectory
            dt: Integration time step
            
        Returns:
            List of trajectory arrays
        """
        print(f"Generating {n_samples} random trajectories...")
        
        trajectories = []
        integrator = RK4Integrator(dt=dt)
        
        for i in range(n_samples):
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i+1}/{n_samples}")
            
            # Generate random initial condition
            self.key, subkey = random.split(self.key)
            initial_state = random.normal(subkey, (self.network_size,)) * 2.0
            
            # Simulate trajectory
            results = integrator.simulate(
                initial_state=initial_state,
                dynamics=consciousness_dynamics,
                steps=trajectory_length,
                save_trajectory=True
            )
            
            trajectories.append(results['trajectory'])
        
        return trajectories
    
    def compute_fifth_derivative_bounds(self, trajectories: List[jnp.ndarray]) -> Dict:
        """
        Compute |Ψ⁽⁵⁾| for all trajectory points and collect statistics.
        
        Args:
            trajectories: List of trajectory arrays
            
        Returns:
            Dictionary with bound statistics
        """
        print("Computing 5th derivative bounds...")
        
        fifth_derivative_norms = []
        computation_times = []
        
        total_points = sum(len(traj) for traj in trajectories)
        processed_points = 0
        
        for traj_idx, trajectory in enumerate(trajectories):
            if (traj_idx + 1) % 50 == 0:
                print(f"  Processing trajectory {traj_idx+1}/{len(trajectories)}")
            
            for point in trajectory:
                start_time = time.time()
                
                try:
                    # Compute 5th derivative norm
                    fifth_deriv_norm = self.consciousness_fn.fifth_derivative_norm(point)
                    fifth_derivative_norms.append(float(fifth_deriv_norm))
                    
                    computation_time = time.time() - start_time
                    computation_times.append(computation_time)
                    
                except Exception as e:
                    print(f"    Warning: Failed to compute derivative at point {processed_points}: {e}")
                    continue
                
                processed_points += 1
        
        # Convert to numpy arrays for analysis
        fifth_derivative_norms = np.array(fifth_derivative_norms)
        computation_times = np.array(computation_times)
        
        # Compute statistics
        stats = {
            'n_points': len(fifth_derivative_norms),
            'mean': float(np.mean(fifth_derivative_norms)),
            'std': float(np.std(fifth_derivative_norms)),
            'min': float(np.min(fifth_derivative_norms)),
            'max': float(np.max(fifth_derivative_norms)),
            'median': float(np.median(fifth_derivative_norms)),
            'q25': float(np.percentile(fifth_derivative_norms, 25)),
            'q75': float(np.percentile(fifth_derivative_norms, 75)),
            'q95': float(np.percentile(fifth_derivative_norms, 95)),
            'q99': float(np.percentile(fifth_derivative_norms, 99)),
            'mean_computation_time': float(np.mean(computation_times)),
            'raw_values': fifth_derivative_norms.tolist()
        }
        
        # Proposed bounds
        stats['proposed_M5_conservative'] = stats['q95']  # 95th percentile
        stats['proposed_M5_strict'] = stats['q99']        # 99th percentile
        stats['proposed_M5_max'] = stats['max']           # Absolute maximum
        
        return stats
    
    def bootstrap_confidence_intervals(self, 
                                     fifth_derivative_norms: np.ndarray,
                                     n_bootstrap: int = 1000,
                                     confidence_level: float = 0.95) -> Dict:
        """
        Compute bootstrap confidence intervals for bound estimates.
        
        Args:
            fifth_derivative_norms: Array of 5th derivative norms
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for intervals
            
        Returns:
            Dictionary with confidence intervals
        """
        print(f"Computing bootstrap confidence intervals (n={n_bootstrap})...")
        
        # Bootstrap samples
        bootstrap_maxes = []
        bootstrap_q95s = []
        bootstrap_q99s = []
        
        for i in range(n_bootstrap):
            # Resample with replacement
            self.key, subkey = random.split(self.key)
            indices = random.choice(subkey, len(fifth_derivative_norms), 
                                  shape=(len(fifth_derivative_norms),), replace=True)
            bootstrap_sample = fifth_derivative_norms[indices]
            
            bootstrap_maxes.append(np.max(bootstrap_sample))
            bootstrap_q95s.append(np.percentile(bootstrap_sample, 95))
            bootstrap_q99s.append(np.percentile(bootstrap_sample, 99))
        
        # Compute confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = 100 * alpha / 2
        upper_percentile = 100 * (1 - alpha / 2)
        
        ci_results = {
            'confidence_level': confidence_level,
            'n_bootstrap': n_bootstrap,
            'max_bound_ci': [
                float(np.percentile(bootstrap_maxes, lower_percentile)),
                float(np.percentile(bootstrap_maxes, upper_percentile))
            ],
            'q95_bound_ci': [
                float(np.percentile(bootstrap_q95s, lower_percentile)),
                float(np.percentile(bootstrap_q95s, upper_percentile))
            ],
            'q99_bound_ci': [
                float(np.percentile(bootstrap_q99s, lower_percentile)),
                float(np.percentile(bootstrap_q99s, upper_percentile))
            ]
        }
        
        return ci_results
    
    def create_visualization(self, stats: Dict, save_path: str = None) -> None:
        """
        Create comprehensive visualization of bounds study results.
        
        Args:
            stats: Statistics dictionary from compute_fifth_derivative_bounds
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        fifth_derivative_norms = np.array(stats['raw_values'])
        
        # Plot 1: Histogram of 5th derivative norms
        axes[0, 0].hist(fifth_derivative_norms, bins=50, alpha=0.7, color='skyblue', 
                       edgecolor='black', density=True)
        axes[0, 0].axvline(stats['mean'], color='red', linestyle='--', linewidth=2,
                          label=f'Mean = {stats["mean"]:.3f}')
        axes[0, 0].axvline(stats['q95'], color='orange', linestyle='--', linewidth=2,
                          label=f'95th percentile = {stats["q95"]:.3f}')
        axes[0, 0].axvline(stats['max'], color='green', linestyle='--', linewidth=2,
                          label=f'Max = {stats["max"]:.3f}')
        axes[0, 0].set_xlabel('|Ψ⁽⁵⁾|')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Distribution of 5th Derivative Norms')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Box plot of bounds
        bound_data = [
            fifth_derivative_norms,
            fifth_derivative_norms[fifth_derivative_norms <= stats['q95']],
            fifth_derivative_norms[fifth_derivative_norms <= stats['q99']]
        ]
        bound_labels = ['All Data', '≤ 95th %ile', '≤ 99th %ile']
        
        axes[0, 1].boxplot(bound_data, labels=bound_labels)
        axes[0, 1].set_ylabel('|Ψ⁽⁵⁾|')
        axes[0, 1].set_title('Bounds Comparison')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Log-scale histogram for tail behavior
        axes[1, 0].hist(fifth_derivative_norms, bins=50, alpha=0.7, color='lightcoral',
                       edgecolor='black', density=True)
        axes[1, 0].set_yscale('log')
        axes[1, 0].set_xlabel('|Ψ⁽⁵⁾|')
        axes[1, 0].set_ylabel('Log Density')
        axes[1, 0].set_title('Log-Scale Distribution (Tail Analysis)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Quantile-Quantile plot vs normal distribution
        from scipy import stats as scipy_stats
        scipy_stats.probplot(fifth_derivative_norms, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot vs Normal Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        else:
            plt.show()
    
    def generate_report(self, stats: Dict, ci_results: Dict, 
                       metadata: Dict) -> Dict:
        """
        Generate comprehensive bounds study report.
        
        Args:
            stats: Statistics from bounds computation
            ci_results: Bootstrap confidence intervals
            metadata: Study metadata
            
        Returns:
            Complete report dictionary
        """
        report = {
            'metadata': metadata,
            'original_claim': {
                'M5': 2.0,
                'justification': 'None provided in original paper'
            },
            'empirical_results': stats,
            'confidence_intervals': ci_results,
            'recommendations': {
                'conservative_bound': {
                    'value': stats['proposed_M5_conservative'],
                    'justification': '95th percentile of empirical distribution',
                    'confidence_interval': ci_results['q95_bound_ci']
                },
                'strict_bound': {
                    'value': stats['proposed_M5_strict'],
                    'justification': '99th percentile of empirical distribution',
                    'confidence_interval': ci_results['q99_bound_ci']
                },
                'absolute_bound': {
                    'value': stats['proposed_M5_max'],
                    'justification': 'Maximum observed value',
                    'confidence_interval': ci_results['max_bound_ci']
                }
            },
            'validation': {
                'original_claim_validity': stats['max'] <= 2.0,
                'original_claim_conservativeness': 2.0 / stats['max'] if stats['max'] > 0 else float('inf'),
                'recommended_bound': stats['proposed_M5_conservative']
            }
        }
        
        return report


def main():
    """Main bounds study execution."""
    parser = argparse.ArgumentParser(description="Empirical |Ψ⁽⁵⁾| Bounds Study")
    parser.add_argument("--nodes", type=int, default=6,
                       help="Number of nodes in network")
    parser.add_argument("--samples", type=int, default=1000,
                       help="Number of random trajectories")
    parser.add_argument("--trajectory-length", type=int, default=50,
                       help="Length of each trajectory")
    parser.add_argument("--dt", type=float, default=0.01,
                       help="Integration time step")
    parser.add_argument("--output", type=str, default="bounds_report.json",
                       help="Output report file")
    parser.add_argument("--plot", type=str,
                       help="Save visualization plot")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    print("UPOCF Empirical Bounds Study")
    print("=" * 50)
    print(f"Network size: {args.nodes} nodes")
    print(f"Trajectories: {args.samples}")
    print(f"Trajectory length: {args.trajectory_length} steps")
    print(f"Time step: {args.dt}")
    print(f"Random seed: {args.seed}")
    print()
    
    # Initialize study
    study = BoundsStudy(args.nodes, seed=args.seed)
    
    # Generate trajectories
    start_time = time.time()
    trajectories = study.generate_random_trajectories(
        n_samples=args.samples,
        trajectory_length=args.trajectory_length,
        dt=args.dt
    )
    trajectory_time = time.time() - start_time
    
    # Compute bounds
    start_time = time.time()
    stats = study.compute_fifth_derivative_bounds(trajectories)
    bounds_time = time.time() - start_time
    
    # Bootstrap confidence intervals
    start_time = time.time()
    ci_results = study.bootstrap_confidence_intervals(
        np.array(stats['raw_values']),
        n_bootstrap=1000
    )
    bootstrap_time = time.time() - start_time
    
    # Generate report
    metadata = {
        'network_size': args.nodes,
        'n_trajectories': args.samples,
        'trajectory_length': args.trajectory_length,
        'dt': args.dt,
        'seed': args.seed,
        'computation_times': {
            'trajectory_generation': trajectory_time,
            'bounds_computation': bounds_time,
            'bootstrap_analysis': bootstrap_time
        },
        'total_points_analyzed': stats['n_points']
    }
    
    report = study.generate_report(stats, ci_results, metadata)
    
    # Print summary
    print("\nBounds Study Results:")
    print("-" * 30)
    print(f"Points analyzed: {stats['n_points']:,}")
    print(f"Mean |Ψ⁽⁵⁾|: {stats['mean']:.6f}")
    print(f"Std |Ψ⁽⁵⁾|: {stats['std']:.6f}")
    print(f"Max |Ψ⁽⁵⁾|: {stats['max']:.6f}")
    print(f"95th percentile: {stats['q95']:.6f}")
    print(f"99th percentile: {stats['q99']:.6f}")
    print()
    print("Recommended Bounds:")
    print(f"- Conservative (95th %ile): {stats['proposed_M5_conservative']:.6f}")
    print(f"- Strict (99th %ile): {stats['proposed_M5_strict']:.6f}")
    print(f"- Absolute maximum: {stats['proposed_M5_max']:.6f}")
    print()
    print(f"Original claim M₅ = 2: {'VALID' if report['validation']['original_claim_validity'] else 'INVALID'}")
    if not report['validation']['original_claim_validity']:
        print(f"Observed maximum exceeds claim by factor of {stats['max']/2.0:.2f}")
    
    # Save report
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {args.output}")
    
    # Create visualization
    if args.plot:
        study.create_visualization(stats, args.plot)
    
    print("\nBounds study completed successfully!")


if __name__ == "__main__":
    main()