"""
Standalone Dynamical Systems Agent Demonstration
Shows how the phase-space trajectory analysis integrates with agent-like decision making
"""

import numpy as np
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass
import asyncio

# Import the dynamical systems framework
from dynamical_systems_framework import (
    SystemParameters, PhaseSpaceTrajectory, CoreEquationEvaluator,
    VisualizationTools, MultiPendulumSimulator, DynamicModeDecomposition,
    demonstrate_system
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AgentDecision:
    """Structure for agent decision output"""
    confidence: float
    reasoning_mode: str
    stability_assessment: str
    recommended_action: str
    supporting_data: Dict[str, Any]

class DynamicalSystemsAnalyzer:
    """Analyzer that provides dynamical systems insights for agent decision-making"""
    
    def __init__(self):
        # Initialize the dynamical systems framework
        self.params = SystemParameters(
            time_span=(0.0, 5.0),
            dt=0.05
        )
        self.trajectory = PhaseSpaceTrajectory(self.params)
        self.evaluator = CoreEquationEvaluator(self.params)
        self.pendulum_sim = MultiPendulumSimulator(n_pendulums=2)
        
        logger.info("Dynamical Systems Analyzer initialized")
    
    def analyze_decision_context(self, 
                               context_vector: List[float],
                               cross_references: Optional[List[List[float]]] = None,
                               time_point: float = 0.5) -> AgentDecision:
        """
        Analyze a decision context using dynamical systems principles
        
        Args:
            context_vector: Primary context for decision (e.g., problem parameters)
            cross_references: Optional cross-reference points for comparison
            time_point: Time point in the system evolution for analysis
        
        Returns:
            AgentDecision with confidence, reasoning mode, and recommendations
        """
        
        try:
            # Convert to numpy arrays
            x = np.array(context_vector)
            m1 = np.array(cross_references[0]) if cross_references and len(cross_references) > 0 else None
            m2 = np.array(cross_references[1]) if cross_references and len(cross_references) > 1 else None
            
            # Perform dynamical systems analysis
            single_result = self.evaluator.evaluate_at_time(x, time_point, m1, m2)
            integrated_result = self.evaluator.integrate_over_time(x, m1, m2)
            
            # Phase-space trajectory analysis
            t_vals, alpha_vals, lambda1_vals, lambda2_vals = self.trajectory.get_full_trajectory()
            trajectory_data = np.vstack([alpha_vals, lambda1_vals, lambda2_vals])
            
            # DMD analysis for stability
            dmd = DynamicModeDecomposition(trajectory_data)
            dmd_result = dmd.compute_dmd()
            
            # Multi-pendulum simulation for chaotic behavior analysis
            initial_conditions = np.array([x[0] if len(x) > 0 else 0.1, 
                                         x[1] if len(x) > 1 else 0.2, 
                                         0.0, 0.0])
            pendulum_result = self.pendulum_sim.simulate(initial_conditions, (0.0, 5.0))
            
            # Interpret results for decision-making
            decision = self._interpret_for_decision(
                single_result, integrated_result, dmd_result, pendulum_result
            )
            
            return decision
            
        except Exception as e:
            logger.error(f"Error in decision context analysis: {e}")
            return AgentDecision(
                confidence=0.1,
                reasoning_mode="Error state",
                stability_assessment="Unknown",
                recommended_action="Retry analysis with different parameters",
                supporting_data={'error': str(e)}
            )
    
    def _interpret_for_decision(self, single_result: Dict, integrated_result: Dict, 
                               dmd_result: Dict, pendulum_result: Dict) -> AgentDecision:
        """Interpret dynamical systems results for agent decision-making"""
        
        psi_x = single_result['psi_x']
        alpha_t = single_result['alpha_t']
        alpha_normalized = alpha_t / 2.0
        dominant_eigenvalue = np.abs(dmd_result['eigenvalues'][0])
        
        # Determine confidence level
        if psi_x > 0.5:
            confidence = 0.9
            confidence_desc = "High"
        elif psi_x > 0.3:
            confidence = 0.7
            confidence_desc = "Moderate"
        else:
            confidence = 0.4
            confidence_desc = "Low"
        
        # Determine reasoning mode
        if alpha_normalized > 0.6:
            reasoning_mode = "Symbolic reasoning dominant - rely on established rules and physics"
        elif alpha_normalized > 0.4:
            reasoning_mode = "Balanced symbolic-neural - integrate rules with learned patterns"
        else:
            reasoning_mode = "Neural prediction dominant - rely on learned patterns and data"
        
        # Assess system stability
        if dominant_eigenvalue < 0.8:
            stability = "Stable system - predictions are reliable"
            stability_risk = "Low"
        elif dominant_eigenvalue < 1.2:
            stability = "Marginally stable - monitor for changes"
            stability_risk = "Medium"
        else:
            stability = "Unstable/chaotic - high uncertainty"
            stability_risk = "High"
        
        # Generate recommendation
        if confidence > 0.8 and stability_risk == "Low":
            action = "Proceed with high confidence - system analysis supports decision"
        elif confidence > 0.6 and stability_risk in ["Low", "Medium"]:
            action = "Proceed with caution - monitor system evolution"
        elif confidence > 0.4:
            action = "Gather more information - current analysis shows uncertainty"
        else:
            action = "Do not proceed - high risk of incorrect decision"
        
        # Compile supporting data
        supporting_data = {
            'psi_value': psi_x,
            'alpha_balance': alpha_normalized,
            'symbolic_output': single_result['symbolic_output'],
            'neural_output': single_result['neural_output'],
            'hybrid_output': single_result['hybrid_output'],
            'regularization_factor': single_result['exp_factor'],
            'integrated_psi': integrated_result['integrated_psi'],
            'dominant_eigenvalue': dominant_eigenvalue,
            'num_dmd_modes': len(dmd_result['eigenvalues']),
            'pendulum_stable': pendulum_result['success'],
            'confidence_numerical': confidence,
            'stability_risk': stability_risk
        }
        
        return AgentDecision(
            confidence=confidence,
            reasoning_mode=reasoning_mode,
            stability_assessment=stability,
            recommended_action=action,
            supporting_data=supporting_data
        )
    
    def create_decision_visualization(self, decision: AgentDecision, 
                                    output_dir: str = "/workspace") -> Dict[str, str]:
        """Create visualizations for the decision analysis"""
        
        try:
            # Create standard dynamical systems visualizations
            fig1 = VisualizationTools.plot_3d_trajectory(self.trajectory)
            fig1.suptitle(f"Decision Analysis - Confidence: {decision.confidence:.2f}", fontsize=16)
            fig1.savefig(f'{output_dir}/decision_phase_space_3d.png', dpi=300, bbox_inches='tight')
            
            fig2 = VisualizationTools.plot_time_series(self.trajectory)
            fig2.suptitle(f"System Evolution - {decision.reasoning_mode[:30]}...", fontsize=14)
            fig2.savefig(f'{output_dir}/decision_trajectory_time_series.png', dpi=300, bbox_inches='tight')
            
            # Test evaluation for Psi evolution
            x_test = np.array([0.5, 0.3])
            fig3 = VisualizationTools.plot_psi_evolution(self.evaluator, x_test)
            fig3.suptitle(f"Decision Confidence Evolution - {decision.stability_assessment}", fontsize=14)
            fig3.savefig(f'{output_dir}/decision_psi_evolution.png', dpi=300, bbox_inches='tight')
            
            return {
                'phase_space_3d': f'{output_dir}/decision_phase_space_3d.png',
                'time_series': f'{output_dir}/decision_trajectory_time_series.png',
                'psi_evolution': f'{output_dir}/decision_psi_evolution.png'
            }
            
        except Exception as e:
            logger.error(f"Error creating decision visualizations: {e}")
            return {'error': str(e)}

class MockAgent:
    """Mock agent that demonstrates integration with dynamical systems analysis"""
    
    def __init__(self):
        self.analyzer = DynamicalSystemsAnalyzer()
        self.decision_history = []
        
    async def make_decision(self, problem_context: Dict[str, Any]) -> Dict[str, Any]:
        """Make a decision using dynamical systems analysis"""
        
        logger.info(f"Agent analyzing decision context: {problem_context.get('description', 'Unknown')}")
        
        # Extract context vector from problem
        context_vector = problem_context.get('parameters', [0.5, 0.3])
        cross_refs = problem_context.get('cross_references', None)
        time_point = problem_context.get('time_point', 0.5)
        
        # Analyze using dynamical systems
        decision = self.analyzer.analyze_decision_context(
            context_vector, cross_refs, time_point
        )
        
        # Store in history
        self.decision_history.append({
            'context': problem_context,
            'decision': decision,
            'timestamp': time_point
        })
        
        # Create response
        response = {
            'decision_made': True,
            'confidence': decision.confidence,
            'reasoning': decision.reasoning_mode,
            'stability': decision.stability_assessment,
            'action': decision.recommended_action,
            'analysis_data': decision.supporting_data,
            'explanation': self._generate_explanation(decision)
        }
        
        return response
    
    def _generate_explanation(self, decision: AgentDecision) -> str:
        """Generate human-readable explanation of the decision"""
        
        confidence_level = "high" if decision.confidence > 0.7 else "moderate" if decision.confidence > 0.4 else "low"
        
        explanation = f"""
Decision Analysis Summary:
- Confidence Level: {confidence_level} ({decision.confidence:.2f})
- Reasoning Approach: {decision.reasoning_mode}
- System Stability: {decision.stability_assessment}
- Recommended Action: {decision.recommended_action}

Technical Details:
- Ψ(x) value: {decision.supporting_data.get('psi_value', 'N/A'):.3f}
- Symbolic-Neural Balance: {decision.supporting_data.get('alpha_balance', 'N/A'):.3f}
- System Stability Risk: {decision.supporting_data.get('stability_risk', 'Unknown')}

This analysis combines symbolic reasoning with neural predictions, regularized by 
cognitive plausibility and computational efficiency constraints, following the 
dynamical systems framework based on Ryan David Oates' research.
        """.strip()
        
        return explanation

async def demonstrate_agent_integration():
    """Demonstrate the complete agent-dynamical systems integration"""
    
    logger.info("=" * 60)
    logger.info("DYNAMICAL SYSTEMS AGENT INTEGRATION DEMONSTRATION")
    logger.info("=" * 60)
    
    # Create mock agent
    agent = MockAgent()
    
    # Test scenarios
    test_scenarios = [
        {
            'description': 'Financial investment decision',
            'parameters': [0.6, 0.4],  # Risk, expected return
            'cross_references': [[0.3, 0.8], [0.9, 0.2]],  # Market comparisons
            'time_point': 0.5
        },
        {
            'description': 'Medical diagnosis confidence',
            'parameters': [0.7, 0.3],  # Symptom severity, test confidence
            'cross_references': [[0.4, 0.6], [0.8, 0.5]],  # Similar cases
            'time_point': 1.0
        },
        {
            'description': 'Engineering system stability',
            'parameters': [0.3, 0.9],  # Load factor, safety margin
            'time_point': 2.0
        }
    ]
    
    # Process each scenario
    results = []
    for i, scenario in enumerate(test_scenarios, 1):
        logger.info(f"\n--- Scenario {i}: {scenario['description']} ---")
        
        # Agent makes decision
        result = await agent.make_decision(scenario)
        results.append(result)
        
        # Log results
        logger.info(f"Decision: {result['action']}")
        logger.info(f"Confidence: {result['confidence']:.2f}")
        logger.info(f"Reasoning: {result['reasoning']}")
        logger.info(f"Stability: {result['stability']}")
        
        # Create visualizations for the first scenario
        if i == 1:
            logger.info("Creating visualizations for first scenario...")
            decision = agent.decision_history[-1]['decision']
            viz_result = agent.analyzer.create_decision_visualization(decision)
            
            if 'error' not in viz_result:
                logger.info("Generated decision visualizations:")
                for name, path in viz_result.items():
                    logger.info(f"  - {name}: {path}")
    
    # Summary
    logger.info(f"\n--- SUMMARY ---")
    logger.info(f"Processed {len(results)} decision scenarios")
    avg_confidence = np.mean([r['confidence'] for r in results])
    logger.info(f"Average confidence: {avg_confidence:.2f}")
    
    high_confidence_decisions = sum(1 for r in results if r['confidence'] > 0.7)
    logger.info(f"High confidence decisions: {high_confidence_decisions}/{len(results)}")
    
    # Show detailed explanation for first decision
    logger.info(f"\nDetailed explanation for first scenario:")
    logger.info(results[0]['explanation'])
    
    return agent, results

if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(demonstrate_agent_integration())