# Phase-Space Trajectory Analysis

This repository contains tools for analyzing the phase-space trajectory of hybrid symbolic-neural systems, specifically focusing on the core equation Ψ(x) and its relationship to Ryan David Oates' work on dynamical systems.

## Overview

The analysis addresses a 3D phase-space trajectory showing the evolution of:
- **α(t)**: Time-varying weight balancing symbolic and neural outputs
- **λ₁(t)**: Regularization weight for cognitive plausibility
- **λ₂(t)**: Regularization weight for computational efficiency

## Files

### Core Analysis Script
- `phase_space_analysis.py`: Main analysis script that generates and analyzes the trajectory

### Documentation
- `phase_space_analysis_report.md`: Comprehensive analysis report with corrected numerical insights
- `requirements.txt`: Python dependencies
- `README.md`: This file

## Installation

1. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Analysis

```bash
python phase_space_analysis.py
```

This will:
- Generate the 3D phase-space trajectory
- Analyze key points on the trajectory (t=0, t=0.5, t=1)
- Calculate core equation components
- Display component evolution plots

### Key Features

1. **Trajectory Generation**: Creates the linear trajectory based on the image description
2. **Point Analysis**: Analyzes specific time points with detailed calculations
3. **Component Evolution**: Shows how different equation components change over time
4. **Visualization**: 3D plots and 2D component evolution graphs

## Core Equation

The analysis focuses on the hybrid symbolic-neural equation:

```
Ψ(x) = ∫ [α(t)S(x) + (1-α(t))N(x) + w_cross[S(m₁)N(m₂) - S(m₂)N(m₁)]] 
       × exp(-[λ₁R_cognitive + λ₂R_efficiency]) 
       × P(H|E,β) dt
```

Where:
- **S(x)**: Symbolic output (e.g., RK4 solution)
- **N(x)**: Neural output (e.g., LSTM prediction)
- **α(t)**: Time-varying weight (0 to 1)
- **λ₁, λ₂**: Regularization weights (2 to 0)
- **R_cognitive, R_efficiency**: Penalty terms
- **P(H|E,β)**: Probability with bias

## Trajectory Characteristics

The corrected analysis reveals:
- **α(t)**: Linear increase from 0 to 1
- **λ₁(t), λ₂(t)**: Linear decrease from 2 to 0
- **Linearity**: Perfectly linear trajectory
- **Symmetry**: λ₁(t) = λ₂(t) for all t

## Key Corrections

The analysis corrects several discrepancies from the original description:
1. **Range Error**: α(t) ranges from 0 to 1, not 0 to 2
2. **Numerical Error**: Example point (t=0.5, α≈1.0, λ₁≈1.5, λ₂≈0.5) doesn't lie on trajectory
3. **Trajectory Equation**: Actual trajectory follows α(t) = t, λ₁(t) = λ₂(t) = 2(1-t)

## Applications

This analysis framework supports:
- **Physics-Informed Neural Networks (PINNs)**: Trajectory represents training dynamics
- **Dynamic Mode Decomposition (DMD)**: Linear trajectory suggests stable mode interactions
- **Multi-pendulum Systems**: Chaotic system modeling with hybrid approaches
- **Hybrid AI Systems**: Adaptive symbolic-neural balance optimization

## Output Examples

### Trajectory Points Analysis

**t = 0.0 (Start)**:
- α(0) = 0.0, λ₁(0) = 2.0, λ₂(0) = 2.0
- Ψ(x) = 0.389 (neural dominance with high penalties)

**t = 0.5 (Midpoint)**:
- α(0.5) = 0.495, λ₁(0.5) = 1.010, λ₂(0.5) = 1.010
- Ψ(x) = 0.482 (balanced state with moderate penalties)

**t = 1.0 (End)**:
- α(1) = 1.0, λ₁(1) = 0.0, λ₂(1) = 0.0
- Ψ(x) = 0.588 (symbolic dominance with no penalties)

## Dependencies

- numpy>=1.21.0
- matplotlib>=3.5.0
- scipy>=1.7.0
- seaborn>=0.11.0

## License

This analysis is provided for educational and research purposes related to hybrid symbolic-neural systems and dynamical systems analysis.
