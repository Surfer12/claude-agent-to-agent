"""
Dynamical Systems Agent Integration
Integrates the phase-space trajectory analysis framework with the unified agent system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass

# Import the dynamical systems framework
from dynamical_systems_framework import (
    SystemParameters, PhaseSpaceTrajectory, CoreEquationEvaluator,
    VisualizationTools, MultiPendulumSimulator, DynamicModeDecomposition,
    PhysicsInformedNeuralNetwork
)

# Import unified agent system components
from unified_agent_system.core import UnifiedAgent, AgentConfig, ProviderType
from unified_agent_system.tools.base import BaseTool

logger = logging.getLogger(__name__)

@dataclass 
class DynamicalSystemInput:
    """Input structure for dynamical system analysis"""
    x: List[float]  # Primary input vector
    m1: Optional[List[float]] = None  # Cross-interaction point 1
    m2: Optional[List[float]] = None  # Cross-interaction point 2
    time_point: float = 0.5  # Time point for evaluation
    base_prob: float = 0.7  # Base probability
    beta: float = 1.4  # Bias parameter

class DynamicalSystemsTool(BaseTool):
    """Tool for dynamical systems analysis within the unified agent framework"""
    
    def __init__(self):
        super().__init__(
            name="dynamical_systems_analysis",
            description="Analyze dynamical systems using phase-space trajectories and hybrid symbolic-neural evaluation"
        )
        
        # Initialize the dynamical systems framework
        self.params = SystemParameters(
            time_span=(0.0, 5.0),
            dt=0.05
        )
        self.trajectory = PhaseSpaceTrajectory(self.params)
        self.evaluator = CoreEquationEvaluator(self.params)
        self.pendulum_sim = MultiPendulumSimulator(n_pendulums=2)
        
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute dynamical systems analysis"""
        try:
            # Parse input
            x = np.array(input_data.get('x', [0.5, 0.3]))
            m1 = np.array(input_data.get('m1', [0.2, 0.4])) if input_data.get('m1') else None
            m2 = np.array(input_data.get('m2', [0.8, 0.6])) if input_data.get('m2') else None
            time_point = input_data.get('time_point', 0.5)
            base_prob = input_data.get('base_prob', 0.7)
            beta = input_data.get('beta', 1.4)
            
            # Single time point evaluation
            single_result = self.evaluator.evaluate_at_time(
                x, time_point, m1, m2, base_prob, beta
            )
            
            # Time integration
            integrated_result = self.evaluator.integrate_over_time(x, m1, m2)
            
            # Phase-space trajectory analysis
            t_vals, alpha_vals, lambda1_vals, lambda2_vals = self.trajectory.get_full_trajectory()
            trajectory_data = np.vstack([alpha_vals, lambda1_vals, lambda2_vals])
            
            # DMD analysis
            dmd = DynamicModeDecomposition(trajectory_data)
            dmd_result = dmd.compute_dmd()
            
            # Multi-pendulum simulation
            initial_conditions = np.array([x[0] if len(x) > 0 else 0.1, 
                                         x[1] if len(x) > 1 else 0.2, 
                                         0.0, 0.0])
            pendulum_result = self.pendulum_sim.simulate(initial_conditions, (0.0, 5.0))
            
            return {
                'success': True,
                'single_evaluation': {
                    'psi_x': single_result['psi_x'],
                    'alpha_t': single_result['alpha_t'],
                    'lambda1_t': single_result['lambda1_t'],
                    'lambda2_t': single_result['lambda2_t'],
                    'symbolic_output': single_result['symbolic_output'],
                    'neural_output': single_result['neural_output'],
                    'hybrid_output': single_result['hybrid_output'],
                    'regularization_factor': single_result['exp_factor']
                },
                'integrated_result': {
                    'integrated_psi': integrated_result['integrated_psi'],
                    'average_psi': integrated_result['average_psi']
                },
                'dmd_analysis': {
                    'num_modes': len(dmd_result['eigenvalues']),
                    'dominant_eigenvalue': float(np.abs(dmd_result['eigenvalues'][0])),
                    'stability': 'stable' if np.abs(dmd_result['eigenvalues'][0]) < 1.0 else 'unstable'
                },
                'pendulum_simulation': {
                    'success': pendulum_result['success'],
                    'final_angle': float(pendulum_result['angles'][0, -1]) if pendulum_result['success'] else None,
                    'max_angle': float(np.max(np.abs(pendulum_result['angles']))) if pendulum_result['success'] else None
                },
                'interpretation': self._interpret_results(single_result, integrated_result, dmd_result)
            }
            
        except Exception as e:
            logger.error(f"Error in dynamical systems analysis: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _interpret_results(self, single_result: Dict, integrated_result: Dict, dmd_result: Dict) -> Dict[str, str]:
        """Interpret the results in human-readable form"""
        psi_x = single_result['psi_x']
        alpha_t = single_result['alpha_t']
        avg_psi = integrated_result['average_psi']
        dominant_eigenvalue = np.abs(dmd_result['eigenvalues'][0])
        
        interpretation = {}
        
        # Interpret Ψ(x) value
        if psi_x > 0.5:
            interpretation['prediction_confidence'] = "High confidence prediction"
        elif psi_x > 0.3:
            interpretation['prediction_confidence'] = "Moderate confidence prediction"
        else:
            interpretation['prediction_confidence'] = "Low confidence prediction"
        
        # Interpret α(t) balance
        alpha_normalized = alpha_t / 2.0
        if alpha_normalized > 0.6:
            interpretation['reasoning_mode'] = "Symbolic reasoning dominant"
        elif alpha_normalized > 0.4:
            interpretation['reasoning_mode'] = "Balanced symbolic-neural reasoning"
        else:
            interpretation['reasoning_mode'] = "Neural prediction dominant"
        
        # Interpret stability
        if dominant_eigenvalue < 0.8:
            interpretation['system_stability'] = "Stable dynamical behavior"
        elif dominant_eigenvalue < 1.2:
            interpretation['system_stability'] = "Marginally stable behavior"
        else:
            interpretation['system_stability'] = "Unstable/chaotic behavior"
        
        # Overall assessment
        if psi_x > 0.4 and dominant_eigenvalue < 1.0:
            interpretation['overall_assessment'] = "Reliable prediction with stable system dynamics"
        elif psi_x > 0.3:
            interpretation['overall_assessment'] = "Moderately reliable prediction"
        else:
            interpretation['overall_assessment'] = "Uncertain prediction, recommend additional analysis"
        
        return interpretation
    
    def get_input_schema(self) -> Dict[str, Any]:
        """Get the input schema for the tool"""
        return {
            "type": "object",
            "properties": {
                "x": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Primary input vector for analysis",
                    "minItems": 1,
                    "maxItems": 10
                },
                "m1": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Optional cross-interaction point 1",
                    "minItems": 1,
                    "maxItems": 10
                },
                "m2": {
                    "type": "array", 
                    "items": {"type": "number"},
                    "description": "Optional cross-interaction point 2",
                    "minItems": 1,
                    "maxItems": 10
                },
                "time_point": {
                    "type": "number",
                    "description": "Time point for evaluation (0.0 to 5.0)",
                    "minimum": 0.0,
                    "maximum": 5.0
                },
                "base_prob": {
                    "type": "number",
                    "description": "Base probability (0.0 to 1.0)",
                    "minimum": 0.0,
                    "maximum": 1.0
                },
                "beta": {
                    "type": "number",
                    "description": "Bias parameter (typically 1.0 to 2.0)",
                    "minimum": 0.1,
                    "maximum": 3.0
                }
            },
            "required": ["x"]
        }

class DynamicalSystemsAgent(UnifiedAgent):
    """Specialized agent that incorporates dynamical systems analysis"""
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        
        # Add the dynamical systems tool
        self.dynamical_tool = DynamicalSystemsTool()
        self.tools.append(self.dynamical_tool)
        
        # Update system prompt to include dynamical systems capabilities
        self.config.system_prompt = self._create_dynamical_system_prompt()
    
    def _create_dynamical_system_prompt(self) -> str:
        """Create system prompt that incorporates dynamical systems analysis"""
        base_prompt = self.config.system_prompt or "You are a helpful AI assistant."
        
        dynamical_prompt = """

You are enhanced with advanced dynamical systems analysis capabilities based on Ryan David Oates' research framework. You can analyze complex systems using:

1. **Phase-Space Trajectory Analysis**: Track the evolution of system parameters α(t), λ₁(t), λ₂(t) over time
2. **Hybrid Symbolic-Neural Evaluation**: Combine symbolic reasoning (S(x)) with neural predictions (N(x))
3. **Core Equation Ψ(x)**: Evaluate the integrated system response with regularization
4. **Physics-Informed Neural Networks (PINNs)**: Apply physics constraints to neural models
5. **Dynamic Mode Decomposition (DMD)**: Extract spatiotemporal modes from system data
6. **Multi-Pendulum Simulation**: Model chaotic dynamical systems

When analyzing problems, consider:
- The balance between symbolic reasoning and neural predictions (α parameter)
- Cognitive plausibility and computational efficiency (λ₁, λ₂ regularization)
- System stability and chaotic behavior
- Cross-interaction effects between different system components

Use the dynamical_systems_analysis tool when dealing with:
- Complex decision-making scenarios
- Time-evolving systems
- Predictions requiring both symbolic and neural approaches
- Stability analysis of dynamical systems
- Chaotic system modeling

The tool returns comprehensive analysis including confidence levels, reasoning modes, stability assessments, and human-readable interpretations.
"""
        
        return base_prompt + dynamical_prompt
    
    async def analyze_dynamical_system(self, 
                                     x: List[float],
                                     m1: Optional[List[float]] = None,
                                     m2: Optional[List[float]] = None,
                                     time_point: float = 0.5) -> Dict[str, Any]:
        """Convenience method for dynamical systems analysis"""
        
        input_data = {
            'x': x,
            'time_point': time_point
        }
        
        if m1 is not None:
            input_data['m1'] = m1
        if m2 is not None:
            input_data['m2'] = m2
        
        return await self.dynamical_tool.execute(input_data)
    
    def create_visualization(self, output_dir: str = "/workspace") -> Dict[str, str]:
        """Create visualizations of the dynamical system"""
        
        try:
            # Create visualizations
            fig1 = VisualizationTools.plot_3d_trajectory(self.dynamical_tool.trajectory)
            fig1.savefig(f'{output_dir}/agent_phase_space_3d.png', dpi=300, bbox_inches='tight')
            
            fig2 = VisualizationTools.plot_time_series(self.dynamical_tool.trajectory)
            fig2.savefig(f'{output_dir}/agent_trajectory_time_series.png', dpi=300, bbox_inches='tight')
            
            # Test evaluation for visualization
            x_test = np.array([0.5, 0.3])
            fig3 = VisualizationTools.plot_psi_evolution(self.dynamical_tool.evaluator, x_test)
            fig3.savefig(f'{output_dir}/agent_psi_evolution.png', dpi=300, bbox_inches='tight')
            
            return {
                'phase_space_3d': f'{output_dir}/agent_phase_space_3d.png',
                'time_series': f'{output_dir}/agent_trajectory_time_series.png',
                'psi_evolution': f'{output_dir}/agent_psi_evolution.png'
            }
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
            return {'error': str(e)}

# Example usage and demonstration
async def demonstrate_dynamical_agent():
    """Demonstrate the dynamical systems agent"""
    
    logger.info("Creating Dynamical Systems Agent...")
    
    # Create agent configuration
    config = AgentConfig(
        provider=ProviderType.CLAUDE,  # or OPENAI
        model="claude-3-5-sonnet-20241022",
        enable_tools=True,
        verbose=True,
        system_prompt="You are an advanced AI assistant with dynamical systems analysis capabilities."
    )
    
    # Create the agent
    agent = DynamicalSystemsAgent(config)
    
    # Test dynamical systems analysis
    logger.info("Testing dynamical systems analysis...")
    
    test_input = [0.5, 0.3]
    test_m1 = [0.2, 0.4]
    test_m2 = [0.8, 0.6]
    
    result = await agent.analyze_dynamical_system(
        x=test_input,
        m1=test_m1,
        m2=test_m2,
        time_point=0.5
    )
    
    if result['success']:
        logger.info("Dynamical Systems Analysis Results:")
        logger.info(f"  Ψ(x) = {result['single_evaluation']['psi_x']:.3f}")
        logger.info(f"  Confidence: {result['interpretation']['prediction_confidence']}")
        logger.info(f"  Reasoning Mode: {result['interpretation']['reasoning_mode']}")
        logger.info(f"  System Stability: {result['interpretation']['system_stability']}")
        logger.info(f"  Overall Assessment: {result['interpretation']['overall_assessment']}")
    else:
        logger.error(f"Analysis failed: {result.get('error', 'Unknown error')}")
    
    # Create visualizations
    logger.info("Creating visualizations...")
    viz_result = agent.create_visualization()
    
    if 'error' not in viz_result:
        logger.info("Generated agent visualizations:")
        for name, path in viz_result.items():
            logger.info(f"  - {name}: {path}")
    else:
        logger.error(f"Visualization error: {viz_result['error']}")
    
    return agent, result, viz_result

if __name__ == "__main__":
    import asyncio
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run demonstration
    asyncio.run(demonstrate_dynamical_agent())