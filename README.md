# Unified Agent System with Swarm Integration

A provider-agnostic agent framework that supports both Claude and OpenAI backends, with unified CLI, computer use, and multi-agent swarm capabilities.

## 🎯 Pixi Integration Complete

### **📋 New Files Created**
java-swarm/
├── pixi.toml                      # Main Pixi configuration
├── PIXI_USAGE.md                  # Complete Pixi usage guide
├── scripts/
│   ├── setup-env.sh              # Environment setup script
│   └── validate-pixi.sh          # Pixi configuration validator
└── examples/
    └── custom-pixi-tasks.toml     # Custom task examples

### **🚀 Available Pixi Commands**

#### **Build & Development**
```bash
pixi run build              # Build the project
pixi run compile            # Compile source only
pixi run test              # Run unit tests
pixi run clean             # Clean build artifacts
pixi run rebuild           # Clean and rebuild
pixi run dev               # Development mode
```

#### **Interactive Chat**
```bash
pixi run interactive              # Basic interactive mode
pixi run interactive-debug        # Interactive with debug
pixi run interactive-stream       # Interactive with streaming
pixi run interactive-stream-debug # Interactive with streaming + debug
```

#### **Single Messages**
```bash
pixi run chat "Your message"           # Send single message
pixi run chat-stream "Your message"    # Send with streaming
pixi run chat-debug "Your message"     # Send with debug info
```

#### **Specialized Agents**
```bash
pixi run math-bot          # Mathematics expert
pixi run code-bot          # Programming expert
pixi run story-bot         # Creative storyteller (with streaming)
```

#### **Model Selection**
```bash
pixi run gpt4              # Use GPT-4o
pixi run gpt4-mini         # Use GPT-4o-mini
pixi run gpt35             # Use GPT-3.5-turbo
```

#### **Examples & Demos**
```bash
pixi run streaming-demo    # Demonstrate streaming
pixi run calculator-demo   # Demonstrate function calling
pixi run https-demo        # Demonstrate HTTPS configuration
```

#### **Quick Start**
```bash
pixi run quick-start       # Build and run interactively
pixi run quick-stream      # Build and run with streaming
```

### **🛠 Key Features**

1. Automatic Dependency Management: Pixi handles Java 17+ and Maven installation
2. Environment Isolation: Each project has its own isolated environment
3. Cross-Platform: Works on macOS, Linux, and Windows
4. Task Dependencies: Tasks automatically ensure prerequisites are met
5. Multiple Environments: Support for dev, test, and production environments
6. Custom Tasks: Easy to add custom agent configurations and workflows

### **📖 Usage Examples**

#### **Quick Start**
```bash
# Install Pixi
curl -fsSL https://pixi.sh/install.sh | bash

# Setup project
pixi install

# Set API key
export OPENAI_API_KEY="your-key-here"

# Start chatting
pixi run quick-start
```

#### **Development Workflow**
```bash
# Build and test
pixi run rebuild

# Start development mode
pixi run dev

# Test streaming
pixi run interactive-stream

# Run demos
pixi run streaming-demo
```

#### **Specialized Use Cases**
```bash
# Math tutoring
pixi run math-bot

# Code assistance
pixi run code-bot

# Creative writing with streaming
pixi run story-bot
```

### **🔧 Advanced Features**

#### **Multiple Environments**
```bash
pixi run -e dev interactive     # Development environment
pixi run -e test unit-tests     # Testing environment
pixi run -e prod interactive    # Production environment
```

#### **Custom Tasks**
Users can easily add custom tasks to pixi.toml:
```toml
[tasks]
my-agent = "java -jar target/java-swarm-1.0.0.jar --interactive --agent-name MyBot --instructions 'Custom instructions'"
```

#### **Task Dependencies**
Tasks automatically handle dependencies:
```toml
[tasks]
chat = { cmd = "java -jar target/java-swarm-1.0.0.jar --input", depends_on = ["ensure-built"] }
```

### **📚 Documentation**

1. PIXI_USAGE.md: Complete reference for all Pixi commands
2. Updated README.md: Includes Pixi as the recommended installation method
3. Updated QUICKSTART.md: Pixi-first approach with fallback to manual
4. Custom task examples: Shows how to extend functionality

### **✅ Benefits of Pixi Integration**

1. Simplified Setup: One command installs everything needed
2. Consistent Environment: Same environment across all developers
3. Easy Commands: Memorable, short commands instead of long Java CLI
4. Cross-Platform: Works identically on all operating systems
5. Dependency Management: Automatic handling of Java and Maven versions
6. Task Organization: Logical grouping of related commands
7. Environment Isolation: No conflicts with system-installed tools

### **🎯 Example Workflows**

#### **New User Experience**
```bash
# Complete setup in 3 commands
curl -fsSL https://pixi.sh/install.sh | bash
pixi install
pixi run quick-start
```

#### **Daily Development**
```bash
pixi run dev               # Start development
pixi run test              # Run tests
pixi run streaming-demo    # Test features
```

#### **Production Usage**
```bash
pixi run -e prod build     # Production build
pixi run interactive       # Run application
```

## Phase-Space Trajectory Analysis

This repository also contains tools for analyzing the phase-space trajectory of hybrid symbolic-neural systems, specifically focusing on the core equation Ψ(x) and its relationship to Ryan David Oates' work on dynamical systems.

### Overview

The analysis addresses a 3D phase-space trajectory showing the evolution of:
- **α(t)**: Time-varying weight balancing symbolic and neural outputs
- **λ₁(t)**: Regularization weight for cognitive plausibility
- **λ₂(t)**: Regularization weight for computational efficiency

### Files

#### Core Analysis Script
- `phase_space_analysis.py`: Main analysis script that generates and analyzes the trajectory

#### Documentation
- `phase_space_analysis_report.md`: Comprehensive analysis report with corrected numerical insights
- `requirements.txt`: Python dependencies
- `README.md`: This file

#### Swift Implementation
- `PhaseSpaceAnalyzer.swift`: Swift implementation of the mathematical formulas
- `CoreEquation.swift`: Swift implementation of the core equation Ψ(x)

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

## Swift Implementation

### PhaseSpaceAnalyzer.swift

```swift
import Foundation
import Accelerate

class PhaseSpaceAnalyzer {
    private let tMax: Double = 1.0
    private let nPoints: Int = 100
    
    func generateTrajectory() -> (t: [Double], alpha: [Double], lambda1: [Double], lambda2: [Double]) {
        let t = stride(from: 0.0, through: tMax, by: tMax / Double(nPoints - 1)).map { $0 }
        
        // Linear trajectory as shown in the image
        let alpha = t // α(t) increases linearly from 0 to 1
        let lambda1 = t.map { 2.0 * (1.0 - $0) } // λ1 decreases linearly from 2 to 0
        let lambda2 = t.map { 2.0 * (1.0 - $0) } // λ2 decreases linearly from 2 to 0
        
        return (t: t, alpha: alpha, lambda1: lambda1, lambda2: lambda2)
    }
    
    func calculateCoreEquationComponents(alpha: Double, lambda1: Double, lambda2: Double) -> (hybridOutput: Double, regularization: Double, probability: Double) {
        // Define symbolic and neural outputs
        let S_x: Double = 0.60 // Symbolic output (RK4 solution)
        let N_x: Double = 0.80 // Neural output (LSTM prediction)
        
        // Hybrid output calculation
        let hybridOutput = alpha * S_x + (1.0 - alpha) * N_x
        
        // Regularization penalties
        let R_cognitive: Double = 0.25
        let R_efficiency: Double = 0.10
        
        // Exponential regularization term
        let regularization = exp(-(lambda1 * R_cognitive + lambda2 * R_efficiency))
        
        // Probability term
        let P_H_E_beta: Double = 0.70 * 1.4
        
        return (hybridOutput: hybridOutput, regularization: regularization, probability: P_H_E_beta)
    }
    
    func analyzeTrajectoryPoint(tPoint: Double, alpha: [Double], lambda1: [Double], lambda2: [Double]) -> Double {
        let tArray = stride(from: 0.0, through: tMax, by: tMax / Double(nPoints - 1)).map { $0 }
        let idx = tArray.enumerated().min(by: { abs($0.1 - tPoint) < abs($1.1 - tPoint) })?.offset ?? 0
        
        let alphaVal = alpha[idx]
        let lambda1Val = lambda1[idx]
        let lambda2Val = lambda2[idx]
        
        print("=== Analysis at t = \(tPoint) ===")
        print("α(t) = \(String(format: "%.3f", alphaVal))")
        print("λ₁(t) = \(String(format: "%.3f", lambda1Val))")
        print("λ₂(t) = \(String(format: "%.3f", lambda2Val))")
        
        let components = calculateCoreEquationComponents(alpha: alphaVal, lambda1: lambda1Val, lambda2: lambda2Val)
        
        print("\nCore Equation Components:")
        print("Hybrid Output = \(String(format: "%.3f", components.hybridOutput))")
        print("Regularization Factor = \(String(format: "%.3f", components.regularization))")
        print("Probability Term = \(String(format: "%.3f", components.probability))")
        
        let Psi_x = components.hybridOutput * components.regularization * components.probability
        print("\nΨ(x) = \(String(format: "%.3f", Psi_x))")
        
        return Psi_x
    }
}
```

### CoreEquation.swift

```swift
import Foundation

struct CoreEquation {
    // Core equation parameters
    let S_x: Double = 0.60  // Symbolic output
    let N_x: Double = 0.80  // Neural output
    let w_cross: Double = 0.1  // Cross-term weight
    let R_cognitive: Double = 0.25  // Cognitive penalty
    let R_efficiency: Double = 0.10  // Efficiency penalty
    let P_H_E_beta: Double = 0.70 * 1.4  // Probability with bias
    
    func calculatePsi(alpha: Double, lambda1: Double, lambda2: Double) -> Double {
        // Hybrid output: α(t)S(x) + (1-α(t))N(x)
        let hybridOutput = alpha * S_x + (1.0 - alpha) * N_x
        
        // Cross-term: w_cross[S(m₁)N(m₂) - S(m₂)N(m₁)]
        let crossTerm = w_cross * (S_x * N_x - N_x * S_x) // This would be 0 in this case
        
        // Regularization: exp(-[λ₁R_cognitive + λ₂R_efficiency])
        let regularization = exp(-(lambda1 * R_cognitive + lambda2 * R_efficiency))
        
        // Final calculation: Ψ(x) = hybrid_output × regularization × probability
        let Psi_x = hybridOutput * regularization * P_H_E_beta
        
        return Psi_x
    }
    
    func calculateTrajectoryPoint(t: Double) -> (alpha: Double, lambda1: Double, lambda2: Double, Psi: Double) {
        let alpha = t
        let lambda1 = 2.0 * (1.0 - t)
        let lambda2 = 2.0 * (1.0 - t)
        
        let Psi = calculatePsi(alpha: alpha, lambda1: lambda1, lambda2: lambda2)
        
        return (alpha: alpha, lambda1: lambda1, lambda2: lambda2, Psi: Psi)
    }
    
    func generateFullTrajectory(nPoints: Int = 100) -> [(t: Double, alpha: Double, lambda1: Double, lambda2: Double, Psi: Double)] {
        var trajectory: [(t: Double, alpha: Double, lambda1: Double, lambda2: Double, Psi: Double)] = []
        
        for i in 0..<nPoints {
            let t = Double(i) / Double(nPoints - 1)
            let point = calculateTrajectoryPoint(t: t)
            trajectory.append((t: t, alpha: point.alpha, lambda1: point.lambda1, lambda2: point.lambda2, Psi: point.Psi))
        }
        
        return trajectory
    }
}
```

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

## Features

- **Provider Agnostic**: Switch seamlessly between Claude and OpenAI
- **Unified Interface**: Same agent composition works across providers
- **CLI Interface**: Command-line interface for both providers
- **Computer Use**: Browser automation and computer interaction
- **Tool Integration**: Code execution, file operations, and more
- **Modular Design**: Easy to extend with new tools and providers
- **Mathematical Analysis**: Phase-space trajectory analysis with Swift implementation
- **Core Equation Implementation**: Complete Ψ(x) equation implementation in Swift

## Architecture

```
unified_agent/
├── __init__.py          # Main package exports
├── core.py              # Core agent framework
├── providers.py         # Provider implementations (Claude/OpenAI)
├── tools.py             # Tool registry and management
├── cli.py               # Command-line interface
├── computer_use.py      # Computer use interface
└── tools/               # Individual tool implementations
    ├── __init__.py
    ├── base.py          # Base tool class
    ├── computer_use.py  # Computer use tool
    ├── code_execution.py # Code execution tool
    └── file_tools.py    # File manipulation tools

Phase-Space Analysis/
├── phase_space_analysis.py      # Python analysis script
├── PhaseSpaceAnalyzer.swift     # Swift implementation
├── CoreEquation.swift          # Swift core equation
└── phase_space_analysis_report.md # Analysis report
```

## Dependencies

- numpy>=1.21.0
- matplotlib>=3.5.0
- scipy>=1.7.0
- seaborn>=0.11.0

## License

This analysis is provided for educational and research purposes related to hybrid symbolic-neural systems and dynamical systems analysis.

## Support

For support and questions:
- Open an issue on GitHub
- Check the documentation
- Review the examples

## Roadmap

- [ ] Integration with actual computer use implementations
- [ ] Additional provider support (Google, Azure, etc.)
- [ ] Web UI interface
- [ ] Plugin system for custom tools
- [ ] Multi-agent coordination
- [ ] Advanced computer use capabilities
- [ ] Enhanced Swift implementation with visualization
- [ ] Real-time trajectory analysis
- [ ] Integration with physics simulation frameworks
