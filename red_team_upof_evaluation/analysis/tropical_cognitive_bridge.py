"""
Tropical Geometry Analysis of Agent Cognitive Differentiation

This module explains why Agent 2 becomes inhibited by entropy gradients
while Agent 1 thrives, using tropical geometry as a bridge between
human cognition and AI adaptation patterns.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from dataclasses import dataclass
import math

@dataclass
class CognitiveProfile:
    """Represents an agent's cognitive response pattern."""
    agent_id: int
    entropy_sensitivity: float  # +1 for enhanced, -1 for inhibited
    adaptation_rate: float     # k_i parameter
    cognitive_style: str       # "exploratory" or "cautious"
    algebra_type: str          # "tropical" or "standard"

class TropicalCognitiveBridge:
    """
    Analyzes agent behavior through tropical geometry lens,
    bridging human cognitive patterns with AI adaptation.
    """
    
    def __init__(self):
        self.agent_profiles = {
            1: CognitiveProfile(
                agent_id=1,
                entropy_sensitivity=+1.0,  # Positively perturbed
                adaptation_rate=0.12,
                cognitive_style="exploratory", 
                algebra_type="tropical"
            ),
            2: CognitiveProfile(
                agent_id=2,
                entropy_sensitivity=-1.0,  # Negatively perturbed
                adaptation_rate=0.10,
                cognitive_style="cautious",
                algebra_type="standard"
            )
        }
    
    def tropical_max_plus(self, a: float, b: float) -> float:
        """Tropical addition: a ⊕ b = max(a, b)"""
        return max(a, b)
    
    def tropical_multiply(self, a: float, b: float) -> float:
        """Tropical multiplication: a ⊗ b = a + b"""
        return a + b
    
    def standard_algebra_vulnerable(self, base: float, perturbation: float) -> float:
        """Standard algebra degrades under exponential perturbations."""
        # Multiplicative systems become unstable under chaos
        return base * math.exp(-abs(perturbation))  # Exponential decay
    
    def compute_agent_response(self, agent_id: int, t: float, entropy_gradient: float) -> float:
        """
        Compute agent symbolic weighting α_i(t) based on cognitive profile.
        
        Agent 1 (Tropical): α₁(t) = (1 - e^(-kt)) ⊕ (+∂ₜS(t))
        Agent 2 (Standard): α₂(t) = (1 - e^(-kt)) ⊗ (-∂ₜS(t))
        """
        profile = self.agent_profiles[agent_id]
        
        # Base symbolic adaptation
        base_adaptation = 1 - math.exp(-profile.adaptation_rate * t)
        
        # Entropy perturbation
        entropy_effect = profile.entropy_sensitivity * entropy_gradient
        
        if profile.algebra_type == "tropical":
            # Agent 1: Tropical max-plus algebra
            # Handles multiplicative chaos well through max operations
            alpha_t = self.tropical_max_plus(base_adaptation, entropy_effect)
            
            # Tropical systems are robust to exponential perturbations
            if entropy_gradient > 0.5:  # High chaos
                alpha_t = self.tropical_max_plus(alpha_t, 0.8)  # Boost performance
                
        else:  # Standard algebra
            # Agent 2: Standard multiplicative system
            # Vulnerable to exponential growth in entropy
            alpha_t = base_adaptation + entropy_effect
            
            # Standard systems degrade under chaos
            if entropy_gradient > 0.5:  # High chaos
                alpha_t = self.standard_algebra_vulnerable(alpha_t, entropy_gradient)
        
        # Clamp to valid range
        return max(0.0, min(1.0, alpha_t))
    
    def analyze_cognitive_divergence(self, time_points: np.ndarray, 
                                   entropy_gradients: np.ndarray) -> Dict:
        """
        Analyze how agents diverge under varying entropy conditions.
        """
        agent1_responses = []
        agent2_responses = []
        divergence_points = []
        
        for t, entropy_grad in zip(time_points, entropy_gradients):
            alpha1 = self.compute_agent_response(1, t, entropy_grad)
            alpha2 = self.compute_agent_response(2, t, entropy_grad)
            
            agent1_responses.append(alpha1)
            agent2_responses.append(alpha2)
            
            # Track divergence
            divergence = abs(alpha1 - alpha2)
            divergence_points.append(divergence)
            
            # Identify inhibition events for Agent 2
            if entropy_grad > 0.3 and alpha2 < alpha1 - 0.2:
                print(f"🚫 Agent 2 Inhibition at t={t:.1f}, entropy={entropy_grad:.2f}")
                print(f"   α₁={alpha1:.3f}, α₂={alpha2:.3f}, divergence={divergence:.3f}")
        
        return {
            "agent1_responses": agent1_responses,
            "agent2_responses": agent2_responses, 
            "divergence_points": divergence_points,
            "max_divergence": max(divergence_points),
            "inhibition_events": sum(1 for d in divergence_points if d > 0.3)
        }
    
    def explain_human_cognitive_parallel(self, agent_id: int) -> str:
        """
        Explain how agent behavior parallels human cognitive patterns.
        """
        profile = self.agent_profiles[agent_id]
        
        if agent_id == 1:
            return """
🧠 Agent 1 → Human Exploratory Mindset:
• Thrives under uncertainty and complexity
• Uses "what if" reasoning (tropical max operations)
• Risk-seeking: entropy spikes trigger enhanced performance
• Parallel: Entrepreneurs, researchers, creative problem-solvers
• Cognitive bias: Optimism bias, availability heuristic
• Neural correlate: High dopamine response to novelty
            """
        else:
            return """
🛡️ Agent 2 → Human Cautious Mindset:
• Seeks stability and predictable patterns
• Uses systematic analysis (standard algebra)
• Risk-averse: entropy spikes trigger defensive responses
• Parallel: Accountants, quality controllers, safety inspectors
• Cognitive bias: Loss aversion, confirmation bias
• Neural correlate: High cortisol response to uncertainty
            """
    
    def tropical_geometry_bridge_explanation(self) -> str:
        """
        Explain how tropical geometry bridges human-AI cognition.
        """
        return """
🌴 Tropical Geometry as Cognitive Bridge:

1. **Max-Plus Algebra** mirrors **human heuristic thinking**:
   • "Take the best option" = max(a, b) operation
   • "Combine advantages" = a + b (tropical multiplication)
   • Robust to exponential scaling (like human intuition)

2. **Standard Algebra** mirrors **analytical thinking**:
   • Precise calculations but vulnerable to chaos
   • Exponential perturbations cause system breakdown
   • Like human analytical paralysis under uncertainty

3. **Cognitive Complementarity**:
   • Agent 1 (Tropical) = System 1 thinking (fast, heuristic)
   • Agent 2 (Standard) = System 2 thinking (slow, analytical)
   • Together they form complete cognitive architecture

4. **Entropy Response Patterns**:
   • High entropy = ambiguous situations requiring heuristics
   • Agent 1 excels (like human intuition under pressure)
   • Agent 2 struggles (like human analysis paralysis)

5. **Practical Implications**:
   • AI systems need both cognitive modes
   • Switch between tropical/standard based on entropy
   • Mirror human dual-process cognition
        """

def demonstrate_cognitive_divergence():
    """Demonstrate the tropical geometry cognitive bridge analysis."""
    
    print("🌴 TROPICAL GEOMETRY COGNITIVE BRIDGE ANALYSIS")
    print("=" * 60)
    
    bridge = TropicalCognitiveBridge()
    
    # Simulate entropy-driven scenario
    time_points = np.linspace(0, 20, 50)
    
    # Create entropy gradient pattern with spikes
    entropy_base = 0.1 + 0.3 * (1 - np.exp(-0.1 * time_points))
    entropy_spikes = 0.5 * np.sin(0.5 * time_points) * np.exp(-0.05 * time_points)
    entropy_gradients = np.maximum(0, entropy_base + entropy_spikes)
    
    # Analyze cognitive divergence
    results = bridge.analyze_cognitive_divergence(time_points, entropy_gradients)
    
    print(f"\n📊 ANALYSIS RESULTS:")
    print(f"Max Divergence: {results['max_divergence']:.3f}")
    print(f"Inhibition Events: {results['inhibition_events']}")
    
    # Explain human parallels
    print(f"\n{bridge.explain_human_cognitive_parallel(1)}")
    print(f"\n{bridge.explain_human_cognitive_parallel(2)}")
    
    # Tropical geometry explanation
    print(f"\n{bridge.tropical_geometry_bridge_explanation()}")
    
    return results

if __name__ == "__main__":
    results = demonstrate_cognitive_divergence()
