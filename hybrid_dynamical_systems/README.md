# Hybrid Dynamical Systems Framework

A comprehensive implementation of hybrid dynamical systems inspired by Ryan David Oates' work on physics-informed neural networks and hybrid modeling.

## Overview

This framework implements the core expression:

```
Ψ(x) = ∫[ α(t) S(x) + [1−α(t)] N(x) + w_cross(S(m₁)N(m₂)−S(m₂)N(m₁)) ]
       × exp[−(λ₁ R_cognitive + λ₂ R_efficiency)] × P(H|E, β) dt
```

The system evolves parameters (α(t), λ₁(t), λ₂(t)) according to learned differential equations, producing a trajectory in 3D phase space that represents the "life story" of the system's adaptive parameters.

## Key Components

### Core Expression Components

- **α(t)**: Time-dependent weight blending symbolic S(x) and neural N(x)
- **λ₁(t)**: Penalty weight for cognitive implausibility  
- **λ₂(t)**: Penalty weight for computational cost/efficiency
- **S(x)**: Symbolic/physics-based prediction
- **N(x)**: Neural/data-driven prediction
- **R_cognitive**: Cognitive plausibility regularizer
- **R_efficiency**: Computational efficiency regularizer
- **P(H|E, β)**: Probabilistic bias with expert knowledge β

### Framework Components

#### Core System
- `HybridDynamicalSystem`: Main class implementing the Ψ(x) expression
- `HybridSystemConfig`: Configuration for system parameters
- `SymbolicPredictor`: Physics-based prediction component
- `NeuralPredictor`: Data-driven prediction component
- `Regularizer`: Cognitive and efficiency regularization
- `ProbabilisticBias`: Expert knowledge incorporation

#### Visualization
- `PhaseSpacePlotter`: 3D trajectory visualization and analysis
- Multiple plotting methods for different aspects of the system

#### Examples
- `ConcreteExample`: Step-by-step demonstration of Ψ(x) evaluation

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from hybrid_dynamical_systems import HybridDynamicalSystem, HybridSystemConfig
from hybrid_dynamical_systems.visualization import PhaseSpacePlotter

# Create system configuration
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

# Integrate trajectory
times, trajectory = system.integrate_trajectory()

# Evaluate Ψ(x) at a specific point
x_sample = np.random.rand(100)
psi_value = system.evaluate_psi(x_sample, 5.0, 1.0, 1.5, 0.5)

# Create visualization
plotter = PhaseSpacePlotter(system)
fig, ax = plotter.plot_3d_trajectory()
plt.show()
```

## Demo

Run the complete demonstration:

```bash
cd hybrid_dynamical_systems
python demo.py
```

This will:
1. Set up the hybrid dynamical system
2. Integrate the 3D trajectory
3. Evaluate Ψ(x) at key time points
4. Generate trajectory insights
5. Create visualizations
6. Run the concrete example
7. Explain the mathematical framework

## 3D Phase-Space Visualization

The framework produces 3D plots showing the trajectory of (α(t), λ₁(t), λ₂(t)):

- **X-axis (α(t))**: Symbolic vs neural weight evolution
- **Y-axis (λ₁(t))**: Cognitive plausibility penalty evolution  
- **Z-axis (λ₂(t))**: Computational efficiency penalty evolution

The trajectory typically shows:
- **Start**: High symbolic/cognitive emphasis (α≈2, λ₁≈2, λ₂≈0)
- **End**: High neural/efficiency emphasis (α≈0, λ₁≈0, λ₂≈2)
- **Evolution**: Gradual trade-off between interpretability and performance

## Concrete Example

The framework includes a detailed example demonstrating Ψ(x) evaluation at a specific point:

**Mid-curve point**: α≈1.0, λ₁≈1.5, λ₂≈0.5

1. **Symbolic prediction**: S(x) = 0.60 (RK4 physics solver)
2. **Neural prediction**: N(x) = 0.80 (LSTM)
3. **Hybrid output**: O_hybrid = 0.5·0.60 + 0.5·0.80 = 0.70
4. **Penalty term**: exp[−(0.75·0.25 + 0.25·0.10)] ≈ 0.8087
5. **Probabilistic bias**: P(H|E, β) = 0.98
6. **Final contribution**: Ψ_t(x) = 0.70·0.8087·0.98 ≈ 0.555

## Mathematical Framework

### Differential Equations
The system evolves according to:
```
dα/dt = f₁(α, λ₁, λ₂, t)
dλ₁/dt = f₂(α, λ₁, λ₂, t)  
dλ₂/dt = f₃(α, λ₁, λ₂, t)
```

### Integration
The full Ψ(x) is computed by integrating over the trajectory:
```
Ψ(x) = ∫ Ψ_t(x) dt
```

### Regularization
The penalty terms ensure:
- **Cognitive Plausibility**: Solutions respect basic physics
- **Computational Efficiency**: Solutions don't waste computational resources

## Applications

### Scientific Computing
- Hybrid PDE solvers combining symbolic and neural components
- Adaptive mesh refinement based on learned dynamics

### Control Systems  
- Hybrid controllers balancing model-based and data-driven approaches
- Adaptive control with learned parameter dynamics

### Machine Learning
- Physics-informed neural networks with adaptive regularization
- Interpretable AI systems with dynamic trust allocation

## Key Insights

### Smart Thermostat Analogy
Think of α(t), λ₁(t), λ₂(t) as a smart thermostat for a hybrid brain:

- **α(t)**: Dials how "symbolic" vs "neural" the thinking is at any instant
- **λ₁(t)**: Penalizes ideas that contradict basic physics or common sense
- **λ₂(t)**: Penalizes ideas that burn too much computational fuel

### Trajectory Interpretation
The 3D phase-space curve is the trace of that thermostat's settings over time. By visualizing the path, you gain immediate insight into:
- When the model trusts physics
- When it relies on learned heuristics  
- How strictly it enforces plausibility and efficiency

## Framework Architecture

```
hybrid_dynamical_systems/
├── core/
│   ├── __init__.py
│   └── hybrid_system.py          # Core system implementation
├── visualization/
│   ├── __init__.py
│   └── phase_space_plotter.py    # 3D visualization tools
├── examples/
│   ├── __init__.py
│   └── concrete_example.py       # Step-by-step examples
├── documentation/
│   └── framework_analysis.md     # Comprehensive analysis
├── __init__.py                   # Package initialization
├── demo.py                       # Complete demonstration
├── requirements.txt              # Dependencies
└── README.md                    # This file
```

## Dependencies

- **numpy**: Numerical computations
- **matplotlib**: Visualization
- **scipy**: ODE integration
- **torch**: Neural network support
- **seaborn**: Enhanced plotting
- **typing-extensions**: Type hints

## Future Directions

### Advanced Dynamics
- Chaotic parameter evolution
- Multi-scale dynamics
- Stochastic parameter processes

### Learning Frameworks
- End-to-end learning of parameter dynamics
- Meta-learning for adaptive systems
- Reinforcement learning for optimal trajectories

### Applications
- Quantum-classical hybrid systems
- Multi-physics simulations
- Cognitive architectures

## Contributing

This framework is designed to be extensible. Key areas for contribution:

1. **Advanced Predictors**: Implement more sophisticated S(x) and N(x) components
2. **Learning Dynamics**: Develop learned parameter evolution functions
3. **Applications**: Create domain-specific implementations
4. **Visualization**: Add new analysis and plotting tools

## References

This framework is inspired by Ryan David Oates' work on:
- Physics-Informed Neural Networks (PINNs)
- Neural Ordinary Differential Equations
- Koopman Theory and Dynamic Mode Decomposition
- Hybrid Dynamical Systems

## License

This implementation is provided for educational and research purposes, inspired by the theoretical work of Ryan David Oates and others in the field of hybrid dynamical systems.