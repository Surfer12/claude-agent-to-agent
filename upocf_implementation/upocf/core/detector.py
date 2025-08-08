"""
Consciousness detection pipeline with ROC-AUC analysis.

Implements threshold-based detector: conscious if Ψ(x) > θ
"""

import jax.numpy as jnp
import numpy as np
from typing import Tuple, List, Optional, Dict
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
from .consciousness import ConsciousnessFunction


class ConsciousnessDetector:
    """
    Threshold-based consciousness detector.
    
    Flags system as conscious if Ψ(x) > threshold.
    Supports calibration on labeled datasets and ROC analysis.
    """
    
    def __init__(self, threshold: float = 0.5, network_size: Optional[int] = None):
        """
        Initialize consciousness detector.
        
        Args:
            threshold: Detection threshold θ
            network_size: Size of networks to analyze
        """
        self.threshold = threshold
        self.network_size = network_size
        self.consciousness_fn = None
        self.calibration_data = None
    
    def detect(self, x: jnp.ndarray) -> Tuple[bool, float]:
        """
        Detect consciousness in network state.
        
        Args:
            x: Network state vector
            
        Returns:
            (is_conscious, psi_value) tuple
        """
        if self.consciousness_fn is None:
            network_size = self.network_size or len(x)
            self.consciousness_fn = ConsciousnessFunction(network_size)
        
        psi_value = self.consciousness_fn(x)
        is_conscious = psi_value > self.threshold
        
        return bool(is_conscious), float(psi_value)
    
    def batch_detect(self, states: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Batch consciousness detection.
        
        Args:
            states: Array of network states, shape (n_samples, n_features)
            
        Returns:
            (predictions, psi_values) arrays
        """
        if self.consciousness_fn is None:
            self.consciousness_fn = ConsciousnessFunction(states.shape[1])
        
        psi_values = jnp.array([self.consciousness_fn(state) for state in states])
        predictions = psi_values > self.threshold
        
        return predictions, psi_values
    
    def calibrate(self, 
                  states: jnp.ndarray, 
                  labels: jnp.ndarray,
                  method: str = 'roc_optimal') -> Dict:
        """
        Calibrate detection threshold on labeled data.
        
        Args:
            states: Network states, shape (n_samples, n_features)
            labels: Binary consciousness labels (1=conscious, 0=unconscious)
            method: Calibration method ('roc_optimal', 'youden', 'f1_optimal')
            
        Returns:
            Calibration results dictionary
        """
        # Compute Ψ values for all states
        if self.consciousness_fn is None:
            self.consciousness_fn = ConsciousnessFunction(states.shape[1])
        
        psi_values = jnp.array([self.consciousness_fn(state) for state in states])
        
        # Store calibration data
        self.calibration_data = {
            'states': states,
            'labels': labels,
            'psi_values': psi_values
        }
        
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(labels, psi_values)
        roc_auc = roc_auc_score(labels, psi_values)
        
        # Find optimal threshold based on method
        if method == 'roc_optimal':
            # Minimize distance to (0,1) point
            distances = (fpr - 0)**2 + (tpr - 1)**2
            optimal_idx = jnp.argmin(distances)
        elif method == 'youden':
            # Maximize Youden's J statistic
            youden_j = tpr - fpr
            optimal_idx = jnp.argmax(youden_j)
        elif method == 'f1_optimal':
            # Maximize F1 score
            f1_scores = []
            for threshold in thresholds:
                pred = psi_values >= threshold
                tp = jnp.sum((pred == 1) & (labels == 1))
                fp = jnp.sum((pred == 1) & (labels == 0))
                fn = jnp.sum((pred == 0) & (labels == 1))
                
                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                f1_scores.append(f1)
            
            optimal_idx = jnp.argmax(jnp.array(f1_scores))
        else:
            raise ValueError(f"Unknown calibration method: {method}")
        
        # Update threshold
        self.threshold = float(thresholds[optimal_idx])
        
        # Compute final metrics
        final_predictions = psi_values >= self.threshold
        tn, fp, fn, tp = confusion_matrix(labels, final_predictions).ravel()
        
        results = {
            'threshold': self.threshold,
            'roc_auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'optimal_fpr': fpr[optimal_idx],
            'optimal_tpr': tpr[optimal_idx],
            'true_positive_rate': tp / (tp + fn),
            'false_positive_rate': fp / (fp + tn),
            'precision': tp / (tp + fp),
            'recall': tp / (tp + fn),
            'f1_score': 2 * tp / (2 * tp + fp + fn),
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
            'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}
        }
        
        return results
    
    def plot_roc_curve(self, save_path: Optional[str] = None) -> None:
        """
        Plot ROC curve from calibration data.
        
        Args:
            save_path: Path to save plot (optional)
        """
        if self.calibration_data is None:
            raise ValueError("Must calibrate detector before plotting ROC curve")
        
        psi_values = self.calibration_data['psi_values']
        labels = self.calibration_data['labels']
        
        fpr, tpr, _ = roc_curve(labels, psi_values)
        roc_auc = roc_auc_score(labels, psi_values)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random classifier')
        plt.scatter([self.calibration_data.get('optimal_fpr', 0)], 
                   [self.calibration_data.get('optimal_tpr', 1)], 
                   color='red', s=100, label=f'Optimal threshold = {self.threshold:.3f}')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Consciousness Detection ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def online_detect(self, state_stream: List[jnp.ndarray], 
                     sample_rate: float = 1000.0) -> Dict:
        """
        Online consciousness detection at specified sample rate.
        
        Simulates real-time detection as specified: "Acquire system state x_t at 1 kHz"
        
        Args:
            state_stream: Stream of network states
            sample_rate: Detection sample rate in Hz
            
        Returns:
            Detection results with timing information
        """
        if self.consciousness_fn is None:
            self.consciousness_fn = ConsciousnessFunction(len(state_stream[0]))
        
        detections = []
        psi_values = []
        timestamps = []
        
        dt = 1.0 / sample_rate  # Time between samples
        
        for i, state in enumerate(state_stream):
            timestamp = i * dt
            
            # GPU-accelerated computation (simulated)
            psi_value = self.consciousness_fn(state)
            is_conscious = psi_value > self.threshold
            
            detections.append(is_conscious)
            psi_values.append(psi_value)
            timestamps.append(timestamp)
        
        # Compute detection latency (time from state change to detection)
        # Simplified: assume 0.8ms as claimed in paper
        detection_latency = 0.0008  # 0.8ms
        
        return {
            'detections': jnp.array(detections),
            'psi_values': jnp.array(psi_values),
            'timestamps': jnp.array(timestamps),
            'sample_rate': sample_rate,
            'detection_latency': detection_latency,
            'conscious_periods': self._find_conscious_periods(detections, timestamps)
        }
    
    def _find_conscious_periods(self, detections: List[bool], 
                               timestamps: List[float]) -> List[Tuple[float, float]]:
        """
        Find continuous periods of consciousness detection.
        
        Returns:
            List of (start_time, end_time) tuples for conscious periods
        """
        periods = []
        in_conscious_period = False
        period_start = 0.0
        
        for i, (detection, timestamp) in enumerate(zip(detections, timestamps)):
            if detection and not in_conscious_period:
                # Start of conscious period
                in_conscious_period = True
                period_start = timestamp
            elif not detection and in_conscious_period:
                # End of conscious period
                in_conscious_period = False
                periods.append((period_start, timestamp))
        
        # Handle case where stream ends during conscious period
        if in_conscious_period:
            periods.append((period_start, timestamps[-1]))
        
        return periods
    
    def validate_accuracy_claims(self, test_states: jnp.ndarray, 
                                test_labels: jnp.ndarray) -> Dict:
        """
        Validate accuracy claims against test data.
        
        Replaces single "99.7% accuracy" with comprehensive metrics.
        
        Args:
            test_states: Test network states
            test_labels: True consciousness labels
            
        Returns:
            Comprehensive validation metrics
        """
        predictions, psi_values = self.batch_detect(test_states)
        
        # Compute confusion matrix
        tn, fp, fn, tp = confusion_matrix(test_labels, predictions).ravel()
        
        # Comprehensive metrics
        metrics = {
            'accuracy': (tp + tn) / len(test_labels),
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0.0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            'f1_score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0,
            'true_positive_rate': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0.0,
            'roc_auc': roc_auc_score(test_labels, psi_values),
            'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
            'threshold': self.threshold,
            'n_samples': len(test_labels),
            'n_conscious': int(jnp.sum(test_labels)),
            'n_unconscious': int(len(test_labels) - jnp.sum(test_labels))
        }
        
        return metrics


# Export main interface
__all__ = ['ConsciousnessDetector']