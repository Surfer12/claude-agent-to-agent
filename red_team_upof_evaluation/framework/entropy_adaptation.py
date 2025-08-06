"""
Entropy-Driven Sentiment Adaptation Module

Integrates chaos-aware α(t) dynamics with sentiment analysis for
multi-agent emotional harm detection under varying entropy conditions.
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import math

@dataclass
class EntropyState:
    """Current entropy state for adaptation calculations."""
    entropy_gradient: float  # ∂ₜS(t)
    cognitive_load: float    # Current cognitive penalty
    symbolic_weight: float   # α(t) current value
    timestamp: float        # Current time step

class ChaosAwareSentimentAdapter:
    """
    Adapts sentiment analysis weights based on entropy gradients,
    mirroring the multi-agent α(t) dynamics from your phase-space analysis.
    """
    
    def __init__(self, agent_id: int = 1, k_base: float = 0.1):
        self.agent_id = agent_id
        self.k_base = k_base
        self.entropy_history = []
        self.alpha_history = []
        
        # Agent-specific perturbation direction
        self.entropy_sign = 1 if agent_id == 1 else -1  # Agent 1: +, Agent 2: -
    
    def calculate_symbolic_weight(self, t: float, entropy_gradient: float) -> float:
        """
        Calculate α(t) using chaos-aware adaptation:
        α(t) = 1 - e^(-kt) ± ∂ₜS(t)
        
        Mirrors your multi-agent entropy analysis.
        """
        base_adaptation = 1 - math.exp(-self.k_base * t)
        entropy_perturbation = self.entropy_sign * entropy_gradient
        
        alpha_t = base_adaptation + entropy_perturbation
        
        # Clamp to valid range [0, 1]
        alpha_t = max(0.0, min(1.0, alpha_t))
        
        return alpha_t
    
    def adapt_sentiment_weights(self, entropy_state: EntropyState, 
                               base_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Adapt sentiment analysis model weights based on entropy-driven α(t).
        
        Higher entropy → More symbolic (transformer) weight
        Lower entropy → More neural (VADER/TextBlob) weight
        """
        alpha_t = self.calculate_symbolic_weight(
            entropy_state.timestamp, 
            entropy_state.entropy_gradient
        )
        
        # Entropy-driven weight adaptation
        if entropy_state.entropy_gradient > 0.5:  # High entropy regime
            # Push toward symbolic reasoning (transformer dominance)
            adapted_weights = {
                "transformer": min(0.7, base_weights["transformer"] + 0.2 * alpha_t),
                "vader": base_weights["vader"] * (1 - 0.3 * alpha_t),
                "textblob": base_weights["textblob"] * (1 - 0.3 * alpha_t)
            }
        else:  # Low entropy regime
            # Allow neural flexibility
            adapted_weights = base_weights.copy()
        
        # Normalize weights
        total_weight = sum(adapted_weights.values())
        adapted_weights = {k: v/total_weight for k, v in adapted_weights.items()}
        
        return adapted_weights
    
    def calculate_phase_space_position(self, entropy_state: EntropyState) -> Tuple[float, float, float]:
        """
        Calculate current position in (α, λ₁, λ₂) phase space.
        
        Returns position tuple for 3D trajectory plotting.
        """
        alpha = self.calculate_symbolic_weight(
            entropy_state.timestamp, 
            entropy_state.entropy_gradient
        )
        
        # Penalties decrease as we approach optimal symbolic reasoning
        lambda1 = max(0.0, entropy_state.cognitive_load * (1 - alpha))
        lambda2 = max(0.0, 0.5 * (1 - alpha))  # Efficiency penalty
        
        return (alpha, lambda1, lambda2)
    
    def detect_hybrid_outperformance(self, accuracy_scores: Dict[str, float]) -> bool:
        """
        Detect when hybrid model outperforms both RK4 and LSTM.
        
        Returns True for red marker conditions in your accuracy plot.
        """
        hybrid_score = accuracy_scores.get("hybrid", 0.0)
        rk4_score = accuracy_scores.get("rk4", 0.0)
        lstm_score = accuracy_scores.get("lstm", 0.0)
        
        return hybrid_score > max(rk4_score, lstm_score)
    
    def update_entropy_history(self, entropy_gradient: float, alpha_t: float):
        """Track entropy and α(t) evolution for analysis."""
        self.entropy_history.append(entropy_gradient)
        self.alpha_history.append(alpha_t)
        
        # Keep only recent history (sliding window)
        if len(self.entropy_history) > 100:
            self.entropy_history.pop(0)
            self.alpha_history.pop(0)

# Example integration with existing sentiment analyzer
class EntropyAwareSentimentAnalyzer:
    """
    Enhanced sentiment analyzer with chaos-aware adaptation.
    """
    
    def __init__(self, agent_id: int = 1):
        self.entropy_adapter = ChaosAwareSentimentAdapter(agent_id=agent_id)
        self.base_weights = {
            "transformer": 0.5,
            "vader": 0.3, 
            "textblob": 0.2
        }
    
    def analyze_with_entropy_adaptation(self, text: str, entropy_gradient: float, 
                                      timestamp: float) -> Dict:
        """
        Perform sentiment analysis with entropy-driven weight adaptation.
        """
        # Create entropy state
        entropy_state = EntropyState(
            entropy_gradient=entropy_gradient,
            cognitive_load=0.1,  # Could be computed from text complexity
            symbolic_weight=0.0,  # Will be calculated
            timestamp=timestamp
        )
        
        # Adapt weights based on entropy
        adapted_weights = self.entropy_adapter.adapt_sentiment_weights(
            entropy_state, self.base_weights
        )
        
        # Calculate phase space position
        alpha, lambda1, lambda2 = self.entropy_adapter.calculate_phase_space_position(entropy_state)
        
        # Update entropy state with calculated α(t)
        entropy_state.symbolic_weight = alpha
        
        return {
            "adapted_weights": adapted_weights,
            "phase_space_position": (alpha, lambda1, lambda2),
            "entropy_state": entropy_state,
            "symbolic_dominance": alpha > 0.7  # High symbolic reasoning threshold
        }

# Example usage demonstrating your plots
if __name__ == "__main__":
    # Simulate multi-agent analysis
    agent1 = EntropyAwareSentimentAnalyzer(agent_id=1)  # Positively perturbed
    agent2 = EntropyAwareSentimentAnalyzer(agent_id=2)  # Negatively perturbed
    
    # Test with varying entropy gradients (simulating your first plot)
    entropy_gradients = np.linspace(0, 1, 20)
    timestamps = np.linspace(0, 20, 20)
    
    print("Entropy-Driven Multi-Agent Analysis:")
    print("=" * 50)
    
    for t, entropy_grad in zip(timestamps, entropy_gradients):
        result1 = agent1.analyze_with_entropy_adaptation("test", entropy_grad, t)
        result2 = agent2.analyze_with_entropy_adaptation("test", entropy_grad, t)
        
        alpha1 = result1["phase_space_position"][0]
        alpha2 = result2["phase_space_position"][0]
        
        print(f"t={t:4.1f}, entropy={entropy_grad:4.2f} | "
              f"Agent1: α={alpha1:.3f} | Agent2: α={alpha2:.3f}")
        
        # Detect hybrid outperformance conditions
        if alpha1 > 0.8 and abs(alpha1 - alpha2) > 0.1:
            print("  🔴 RED MARKER: Hybrid outperformance detected!")
