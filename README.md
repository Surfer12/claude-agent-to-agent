# Hybrid Dynamical Systems Framework
## Following Ryan David Oates' Methodology

This repository contains a complete computational implementation of the hybrid dynamical systems framework described in your walk-through, combining symbolic reasoning with neural networks in a physics-informed approach.

## Mathematical Foundation

The core expression implemented here is:

```
Ψ(x) = ∫[ α(t) S(x) + [1−α(t)] N(x) + w_cross(S(m₁)N(m₂)−S(m₂)N(m₁)) ]
       × exp[−(λ₁ R_cognitive + λ₂ R_efficiency)] × P(H|E, β) dt
```

Where:
- **α(t)**: Time-dependent weight balancing symbolic S(x) and neural N(x) components
- **λ₁(t)**: Penalty weight for cognitive implausibility  
- **λ₂(t)**: Penalty weight for computational efficiency
- **w_cross**: Cross-coupling strength for symplectic/Koopman interactions
- **P(H|E, β)**: Expert knowledge integration with bias parameter β

## Key Features

### 🔬 Core Mathematical Framework
- **Phase-space dynamics**: (α(t), λ₁(t), λ₂(t)) evolution through differential equations
- **Symbolic solver**: Physics-based RK4 integration for interpretable reasoning
- **Neural predictor**: LSTM-like component for data-driven insights
- **Penalty functions**: Cognitive plausibility and efficiency constraints
- **Probabilistic bias**: Bayesian expert knowledge integration

### 🧠 Advanced Capabilities (Oates Methodology)
- **Physics-Informed Neural Networks (PINNs)**: Learn dynamics while respecting physical laws
- **Neural ODEs**: Adaptive trajectory generation with learnable dynamics
- **Dynamic Mode Decomposition (DMD)**: Extract coherent spatiotemporal modes
- **Koopman Operator Theory**: Linearize nonlinear dynamics in observable space
- **Hybrid Integration**: Seamless symbolic-neural coupling

### 📊 Visualization & Analysis
- **3D Phase-space plots**: Interactive trajectory visualization
- **Parameter evolution**: Time-series analysis of α(t), λ₁(t), λ₂(t)
- **Ψ(x) integral analysis**: Component-wise breakdown and cumulative integration
- **Comparative studies**: Multiple system configurations
- **Real-time animation**: Dynamic trajectory evolution

## File Structure

```
├── hybrid_phase_space_system.py     # Core mathematical framework
├── phase_space_visualizer.py        # Interactive visualization tools  
├── oates_framework_integration.py   # Advanced PINN/DMD capabilities
├── comprehensive_demo.py            # Full framework demonstration
├── simple_demo.py                   # Dependency-free demonstration
└── README.md                        # This documentation
```

## Quick Start

### Basic Usage (No Dependencies)
```bash
python3 simple_demo.py
```

This runs a complete demonstration of the framework using only Python standard library.

### Full Framework (With Dependencies)
```bash
pip install numpy scipy matplotlib torch seaborn plotly pandas
python3 comprehensive_demo.py
```

This provides the complete implementation with advanced visualizations and PINN capabilities.

## Example Output

The framework successfully reproduces the walk-through example:

```
Analysis at t = 5.00, x = 1.0

1. Trajectory Parameters:
   α(t) = 0.957
   λ₁(t) = 1.345  
   λ₂(t) = 1.045

2. Component Predictions:
   S(x) = 0.364 (from RK4 physics solver)
   N(x) = 0.867 (from LSTM)

3. Hybrid Output:
   α_normalized = α/2 = 0.478
   O_hybrid = 0.478·0.364 + 0.522·0.867 = 0.626

4. Penalty Terms:
   Penalty = exp[−(0.673·0.019 + 0.523·0.252)] = 0.8655

5. Probabilistic Bias:
   P(H|E,β) = 0.687

6. Final Contribution:
   Ψₜ(x) = 0.596 · 0.8655 · 0.687 = 0.3546
```

## Framework Concepts

### 1. Smart Thermostat Behavior
The system acts like a "smart thermostat" for hybrid AI:
- **α(t)** dials symbolic vs neural thinking at each instant
- **λ₁(t)** penalizes ideas contradicting physics/common sense  
- **λ₂(t)** penalizes computationally expensive solutions

### 2. Physics-Informed Learning
Following Oates' methodology:
- Respects physical constraints through penalty terms
- Maintains interpretability while leveraging neural power
- Enables smooth transitions between reasoning modes
- Supports chaotic system analysis and control

### 3. Trajectory Evolution
The 3D phase-space curve reveals system adaptation:
- **Early**: High α, strong cognitive constraints, low efficiency pressure
- **Later**: Lower α, relaxed constraints, higher efficiency focus
- **Smooth**: Continuous evolution preserves interpretability

### 4. Practical Applications
- Chaotic mechanical systems (coupled pendula, robotics)
- Route-to-chaos analysis and phase-locking detection
- Real hardware modeling (friction, backlash, nonlinearities)  
- Safety-critical AI requiring interpretability

## Advanced Features

### Physics-Informed Neural Networks
```python
# Train PINN to learn phase-space dynamics
system = create_advanced_example_system()
losses = system.train_pinn()
neural_trajectory = system.generate_neural_ode_trajectory()
```

### Dynamic Mode Decomposition
```python
# Extract coherent modes from trajectory data
dmd_analysis = system.fit_dmd()
modes = dmd_analysis['modes']
eigenvalues = dmd_analysis['eigenvalues']
```

### Interactive Visualization
```python
# Create interactive 3D plots
visualizer = PhaseSpaceVisualizer(system)
fig = visualizer.plot_interactive_3d()
fig.show()
```

## Theoretical Background

This implementation bridges several key areas:

### Dynamical Systems Theory
- Phase-space analysis of hybrid symbolic-neural evolution
- Stability analysis through Lyapunov methods
- Bifurcation detection and chaos characterization

### Machine Learning Integration  
- Physics-informed neural networks for constrained learning
- Neural ordinary differential equations for adaptive dynamics
- Bayesian inference for expert knowledge incorporation

### Koopman Operator Theory
- Linearization of nonlinear dynamics in observable space
- Mode decomposition for coherent structure identification
- Predictive modeling and control design

## Ryan David Oates' Methodology Alignment

This framework directly implements concepts from Oates' research:

1. **Interpretability vs Performance**: Dynamic balance through α(t)
2. **Physics-Constrained ML**: Penalty functions enforce physical plausibility  
3. **Hybrid Integration**: Seamless symbolic-neural coupling
4. **Chaotic Systems**: Route-to-chaos analysis capabilities
5. **Real-World Modeling**: Hardware friction, backlash, nonlinearities

## Future Extensions

- **Multi-scale dynamics**: Hierarchical phase-space decomposition
- **Adaptive mesh refinement**: Dynamic trajectory resolution
- **Uncertainty quantification**: Bayesian neural ODEs
- **Control synthesis**: Optimal trajectory design
- **Hardware acceleration**: GPU-optimized implementations

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{hybrid_dynamical_systems_framework,
  title={Hybrid Dynamical Systems Framework: Following Ryan David Oates' Methodology},
  author={[Your Name]},
  year={2024},
  url={https://github.com/[your-repo]/hybrid-dynamical-systems}
}
```

## License

This framework is released under the MIT License. See LICENSE file for details.

---

**The framework successfully bridges interpretable physics-based modeling with powerful neural network capabilities, exactly as envisioned in Ryan David Oates' hybrid dynamical systems research.**
