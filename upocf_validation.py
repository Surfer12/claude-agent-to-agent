"""
UPOCF Framework Validation and Testing Suite

This module provides comprehensive validation and testing capabilities for the
Unified Onto-Phenomenological Consciousness Framework, including:
- Cellular automata experiments for ground truth validation
- ROC analysis and performance metrics
- Comparative analysis with other consciousness detection methods
- Synthetic dataset generation
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
from typing import List, Dict, Tuple, Optional
import pandas as pd
from upocf_implementation import UPOCFFramework, ConsciousnessState
import time
import seaborn as sns
from scipy import stats
import warnings

class CellularAutomaton:
    """
    Cellular Automaton implementation for consciousness validation experiments.
    
    Provides ground truth consciousness states through CA evolution patterns
    and integrated information computation.
    """
    
    def __init__(self, rule: int, size: int = 32):
        """
        Initialize cellular automaton.
        
        Args:
            rule: CA rule number (0-255)
            size: Size of CA grid
        """
        self.rule = rule
        self.size = size
        self.rule_binary = format(rule, '08b')
        
    def evolve(self, initial_state: np.ndarray, steps: int) -> np.ndarray:
        """
        Evolve CA for given number of steps.
        
        Args:
            initial_state: Initial binary state
            steps: Number of evolution steps
            
        Returns:
            Evolution history as 2D array
        """
        history = np.zeros((steps + 1, self.size), dtype=int)
        history[0] = initial_state
        
        for step in range(steps):
            current = history[step]
            next_state = np.zeros(self.size, dtype=int)
            
            for i in range(self.size):
                # Get neighborhood (with periodic boundary conditions)
                left = current[(i - 1) % self.size]
                center = current[i]
                right = current[(i + 1) % self.size]
                
                # Convert neighborhood to rule index
                neighborhood = left * 4 + center * 2 + right
                
                # Apply rule
                next_state[i] = int(self.rule_binary[7 - neighborhood])
            
            history[step + 1] = next_state
        
        return history
    
    def generate_labeled_dataset(self, num_samples: int = 1000, 
                               steps: int = 50) -> Tuple[List[np.ndarray], List[bool]]:
        """
        Generate labeled dataset for consciousness validation.
        
        Args:
            num_samples: Number of samples to generate
            steps: Evolution steps per sample
            
        Returns:
            Tuple of (states, consciousness_labels)
        """
        states = []
        labels = []
        
        for _ in range(num_samples):
            # Random initial state
            initial = np.random.randint(0, 2, self.size)
            
            # Evolve CA
            evolution = self.evolve(initial, steps)
            
            # Use final state
            final_state = evolution[-1]
            states.append(final_state)
            
            # Label based on complexity heuristics
            # High complexity patterns are considered "conscious"
            complexity = self._compute_complexity(evolution)
            is_conscious = complexity > np.median([self._compute_complexity(
                self.evolve(np.random.randint(0, 2, self.size), steps)) 
                for _ in range(100)])
            
            labels.append(is_conscious)
        
        return states, labels
    
    def _compute_complexity(self, evolution: np.ndarray) -> float:
        """
        Compute complexity measure for CA evolution.
        
        Args:
            evolution: CA evolution history
            
        Returns:
            Complexity measure
        """
        # Use entropy of spatial patterns as complexity measure
        entropies = []
        for step in range(evolution.shape[0]):
            state = evolution[step]
            # Compute local entropy
            entropy = 0
            for i in range(len(state) - 1):
                pattern = state[i:i+2]
                pattern_int = pattern[0] * 2 + pattern[1]
                if pattern_int > 0:
                    p = pattern_int / 4
                    entropy -= p * np.log2(p) if p > 0 else 0
            entropies.append(entropy)
        
        return np.mean(entropies)

class UPOCFValidator:
    """
    Comprehensive validation suite for the UPOCF framework.
    """
    
    def __init__(self, framework: UPOCFFramework):
        """
        Initialize validator with UPOCF framework instance.
        
        Args:
            framework: UPOCF framework to validate
        """
        self.framework = framework
        self.results = {}
        
    def run_ca_validation(self, rules: List[int] = [30, 110, 150, 184], 
                         num_samples: int = 500) -> Dict:
        """
        Run cellular automata validation experiments.
        
        Args:
            rules: List of CA rules to test
            num_samples: Number of samples per rule
            
        Returns:
            Validation results dictionary
        """
        print("Running Cellular Automata Validation...")
        
        all_states = []
        all_labels = []
        rule_results = {}
        
        for rule in rules:
            print(f"Testing CA Rule {rule}...")
            ca = CellularAutomaton(rule, size=12)  # Small size for exact Phi
            states, labels = ca.generate_labeled_dataset(num_samples)
            
            all_states.extend(states)
            all_labels.extend(labels)
            
            # Test framework on this rule
            rule_performance = self.framework.validate_performance(states, labels)
            rule_results[rule] = rule_performance
            
            print(f"Rule {rule} - Accuracy: {rule_performance['accuracy']:.3f}, "
                  f"TPR: {rule_performance['true_positive_rate']:.3f}")
        
        # Overall performance
        overall_performance = self.framework.validate_performance(all_states, all_labels)
        
        results = {
            'overall_performance': overall_performance,
            'rule_results': rule_results,
            'test_states': all_states,
            'ground_truth': all_labels
        }
        
        self.results['ca_validation'] = results
        return results
    
    def run_roc_analysis(self, test_states: List[np.ndarray], 
                        ground_truth: List[bool]) -> Dict:
        """
        Perform ROC analysis on test data.
        
        Args:
            test_states: Test system states
            ground_truth: Ground truth labels
            
        Returns:
            ROC analysis results
        """
        print("Running ROC Analysis...")
        
        # Get consciousness scores
        phi_scores = []
        psi_scores = []
        
        for state in test_states:
            result = self.framework.detect_consciousness_realtime(state)
            phi_scores.append(result.phi)
            psi_scores.append(result.psi)
        
        # ROC analysis for Phi scores
        fpr_phi, tpr_phi, thresholds_phi = roc_curve(ground_truth, phi_scores)
        auc_phi = auc(fpr_phi, tpr_phi)
        
        # ROC analysis for Psi scores
        fpr_psi, tpr_psi, thresholds_psi = roc_curve(ground_truth, psi_scores)
        auc_psi = auc(fpr_psi, tpr_psi)
        
        results = {
            'phi_scores': phi_scores,
            'psi_scores': psi_scores,
            'fpr_phi': fpr_phi,
            'tpr_phi': tpr_phi,
            'auc_phi': auc_phi,
            'fpr_psi': fpr_psi,
            'tpr_psi': tpr_psi,
            'auc_psi': auc_psi,
            'thresholds_phi': thresholds_phi,
            'thresholds_psi': thresholds_psi
        }
        
        self.results['roc_analysis'] = results
        return results
    
    def benchmark_performance(self, system_sizes: List[int] = [4, 6, 8, 10, 12],
                            num_trials: int = 100) -> Dict:
        """
        Benchmark framework performance across different system sizes.
        
        Args:
            system_sizes: List of system sizes to test
            num_trials: Number of trials per size
            
        Returns:
            Performance benchmark results
        """
        print("Running Performance Benchmarks...")
        
        results = {
            'system_sizes': system_sizes,
            'detection_times': [],
            'accuracy_scores': [],
            'phi_values': [],
            'error_bounds': []
        }
        
        for size in system_sizes:
            print(f"Benchmarking system size {size}...")
            
            times = []
            accuracies = []
            phi_vals = []
            errors = []
            
            for trial in range(num_trials):
                # Generate random test state
                state = np.random.randint(0, 2, size)
                
                # Measure detection time
                start_time = time.time()
                result = self.framework.detect_consciousness_realtime(state)
                end_time = time.time()
                
                detection_time = (end_time - start_time) * 1000  # ms
                times.append(detection_time)
                phi_vals.append(result.phi)
                errors.append(result.error_bound)
            
            results['detection_times'].append(times)
            results['phi_values'].append(phi_vals)
            results['error_bounds'].append(errors)
        
        self.results['performance_benchmark'] = results
        return results
    
    def validate_taylor_approximation(self, num_tests: int = 100) -> Dict:
        """
        Validate Taylor series approximation accuracy.
        
        Args:
            num_tests: Number of test cases
            
        Returns:
            Taylor approximation validation results
        """
        print("Validating Taylor Approximation...")
        
        errors = []
        predicted_bounds = []
        actual_errors = []
        
        for _ in range(num_tests):
            # Generate test state
            state = np.random.randint(0, 2, 8)
            x0 = state.copy()
            
            # Small perturbation
            perturbation = np.random.normal(0, 0.1, len(state))
            x_perturbed = state + perturbation
            
            # Consciousness function
            def psi_func(x):
                # Ensure binary state for Phi computation
                binary_x = (x > 0.5).astype(int)
                return self.framework.compute_integrated_information(binary_x)
            
            # True value
            true_value = psi_func(x_perturbed)
            
            # Taylor approximation
            approx_value, error_bound = self.framework.taylor_approximation(
                x_perturbed, x0, psi_func)
            
            actual_error = abs(true_value - approx_value)
            errors.append(actual_error)
            predicted_bounds.append(error_bound)
            actual_errors.append(actual_error)
        
        # Validation statistics
        bound_violations = sum(1 for i in range(len(errors)) 
                              if actual_errors[i] > predicted_bounds[i])
        violation_rate = bound_violations / len(errors)
        
        results = {
            'actual_errors': actual_errors,
            'predicted_bounds': predicted_bounds,
            'mean_actual_error': np.mean(actual_errors),
            'mean_predicted_bound': np.mean(predicted_bounds),
            'bound_violation_rate': violation_rate,
            'max_error': np.max(actual_errors),
            'error_std': np.std(actual_errors)
        }
        
        self.results['taylor_validation'] = results
        return results
    
    def test_bifurcation_analysis(self, mu_range: np.ndarray = None) -> Dict:
        """
        Test bifurcation analysis capabilities.
        
        Args:
            mu_range: Range of bifurcation parameters to test
            
        Returns:
            Bifurcation analysis results
        """
        if mu_range is None:
            mu_range = np.linspace(-1, 2, 50)
        
        print("Testing Bifurcation Analysis...")
        
        results = {
            'mu_values': mu_range,
            'final_radii': [],
            'theoretical_radii': [],
            'oscillatory_states': []
        }
        
        for mu in mu_range:
            bifurcation_result = self.framework.hopf_bifurcation_analysis(mu)
            results['final_radii'].append(bifurcation_result['final_radius'])
            results['theoretical_radii'].append(bifurcation_result['theoretical_radius'])
            results['oscillatory_states'].append(bifurcation_result['is_oscillatory'])
        
        self.results['bifurcation_analysis'] = results
        return results
    
    def generate_validation_report(self, save_path: Optional[str] = None) -> str:
        """
        Generate comprehensive validation report.
        
        Args:
            save_path: Optional path to save report
            
        Returns:
            Report text
        """
        report = "UPOCF Framework Validation Report\n"
        report += "=" * 50 + "\n\n"
        
        # CA Validation Results
        if 'ca_validation' in self.results:
            ca_results = self.results['ca_validation']
            overall = ca_results['overall_performance']
            
            report += "Cellular Automata Validation:\n"
            report += f"Overall Accuracy: {overall['accuracy']:.3f}\n"
            report += f"True Positive Rate: {overall['true_positive_rate']:.3f}\n"
            report += f"False Positive Rate: {overall['false_positive_rate']:.3f}\n"
            report += f"Mean Detection Time: {overall['mean_detection_time_ms']:.2f} ms\n\n"
            
            report += "Per-Rule Results:\n"
            for rule, result in ca_results['rule_results'].items():
                report += f"Rule {rule}: Accuracy={result['accuracy']:.3f}, "
                report += f"TPR={result['true_positive_rate']:.3f}\n"
            report += "\n"
        
        # ROC Analysis
        if 'roc_analysis' in self.results:
            roc_results = self.results['roc_analysis']
            report += "ROC Analysis:\n"
            report += f"Phi AUC: {roc_results['auc_phi']:.3f}\n"
            report += f"Psi AUC: {roc_results['auc_psi']:.3f}\n\n"
        
        # Taylor Validation
        if 'taylor_validation' in self.results:
            taylor_results = self.results['taylor_validation']
            report += "Taylor Approximation Validation:\n"
            report += f"Mean Actual Error: {taylor_results['mean_actual_error']:.6f}\n"
            report += f"Mean Predicted Bound: {taylor_results['mean_predicted_bound']:.6f}\n"
            report += f"Bound Violation Rate: {taylor_results['bound_violation_rate']:.3f}\n"
            report += f"Max Error: {taylor_results['max_error']:.6f}\n\n"
        
        # Performance Benchmark
        if 'performance_benchmark' in self.results:
            perf_results = self.results['performance_benchmark']
            report += "Performance Benchmark:\n"
            for i, size in enumerate(perf_results['system_sizes']):
                mean_time = np.mean(perf_results['detection_times'][i])
                report += f"Size {size}: Mean Detection Time = {mean_time:.2f} ms\n"
            report += "\n"
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        
        return report
    
    def plot_validation_results(self, save_plots: bool = False):
        """
        Generate validation plots.
        
        Args:
            save_plots: Whether to save plots to files
        """
        plt.style.use('seaborn-v0_8')
        
        # ROC Curves
        if 'roc_analysis' in self.results:
            roc_results = self.results['roc_analysis']
            
            plt.figure(figsize=(10, 8))
            
            plt.subplot(2, 2, 1)
            plt.plot(roc_results['fpr_phi'], roc_results['tpr_phi'], 
                    label=f'Phi (AUC = {roc_results["auc_phi"]:.3f})')
            plt.plot(roc_results['fpr_psi'], roc_results['tpr_psi'], 
                    label=f'Psi (AUC = {roc_results["auc_psi"]:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Performance vs System Size
        if 'performance_benchmark' in self.results:
            perf_results = self.results['performance_benchmark']
            
            plt.subplot(2, 2, 2)
            mean_times = [np.mean(times) for times in perf_results['detection_times']]
            std_times = [np.std(times) for times in perf_results['detection_times']]
            
            plt.errorbar(perf_results['system_sizes'], mean_times, 
                        yerr=std_times, marker='o', capsize=5)
            plt.xlabel('System Size')
            plt.ylabel('Detection Time (ms)')
            plt.title('Performance vs System Size')
            plt.grid(True, alpha=0.3)
        
        # Bifurcation Analysis
        if 'bifurcation_analysis' in self.results:
            bif_results = self.results['bifurcation_analysis']
            
            plt.subplot(2, 2, 3)
            plt.plot(bif_results['mu_values'], bif_results['final_radii'], 
                    'b-', label='Final Radius')
            plt.plot(bif_results['mu_values'], bif_results['theoretical_radii'], 
                    'r--', label='Theoretical')
            plt.xlabel('μ (Bifurcation Parameter)')
            plt.ylabel('Radius')
            plt.title('Hopf Bifurcation Analysis')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Taylor Error Analysis
        if 'taylor_validation' in self.results:
            taylor_results = self.results['taylor_validation']
            
            plt.subplot(2, 2, 4)
            plt.scatter(taylor_results['predicted_bounds'], 
                       taylor_results['actual_errors'], alpha=0.6)
            max_val = max(max(taylor_results['predicted_bounds']), 
                         max(taylor_results['actual_errors']))
            plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.7)
            plt.xlabel('Predicted Error Bound')
            plt.ylabel('Actual Error')
            plt.title('Taylor Approximation Validation')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('upocf_validation_results.png', dpi=300, bbox_inches='tight')
        
        plt.show()

def run_comprehensive_validation():
    """Run comprehensive validation suite for UPOCF framework."""
    
    print("Initializing UPOCF Framework...")
    framework = UPOCFFramework(max_system_size=12)
    validator = UPOCFValidator(framework)
    
    print("Starting Comprehensive Validation...\n")
    
    # 1. Cellular Automata Validation
    ca_results = validator.run_ca_validation(rules=[30, 110, 150, 184], num_samples=200)
    
    # 2. ROC Analysis
    test_states = ca_results['test_states']
    ground_truth = ca_results['ground_truth']
    roc_results = validator.run_roc_analysis(test_states, ground_truth)
    
    # 3. Performance Benchmarking
    perf_results = validator.benchmark_performance(system_sizes=[4, 6, 8, 10, 12])
    
    # 4. Taylor Approximation Validation
    taylor_results = validator.validate_taylor_approximation(num_tests=50)
    
    # 5. Bifurcation Analysis
    bif_results = validator.test_bifurcation_analysis()
    
    # Generate Report
    report = validator.generate_validation_report('upocf_validation_report.txt')
    print(report)
    
    # Generate Plots
    validator.plot_validation_results(save_plots=True)
    
    return validator

if __name__ == "__main__":
    validator = run_comprehensive_validation()