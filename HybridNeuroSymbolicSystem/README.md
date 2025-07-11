# Hybrid Neuro-Symbolic System: Composite Loss & Regularization

A Swift implementation of the composite loss function for hybrid neuro-symbolic systems, designed to balance accuracy, human-like reasoning, and computational efficiency. This system promotes user agency through interactive parameter adjustment and educational insights.

## 🎯 Overview

The composite loss function serves as the system's "optimizer," guiding how symbolic and neural outputs are refined:

```
Composite Loss = L_logic + λ₁·R_cog + λ₂·R_eff
```

Where:
- **L_logic**: Task-specific loss measuring accuracy against ground truth
- **R_cog**: Cognitive regularizer penalizing deviations from human-like reasoning patterns (~86% alignment target)
- **R_eff**: Efficiency regularizer encouraging computational simplicity (12-15% improvement target)
- **λ₁, λ₂**: Regularization weights controlling the balance

## 🚀 Features

### Core Functionality
- **Real-time Loss Computation**: Interactive parameter adjustment with live updates
- **Educational Insights**: Socratic questioning to promote deeper understanding
- **Focus Session Management**: 90-minute work blocks with cognitive reset reminders
- **Recursive Loss Calculation**: Support for nested logical operations
- **Preset Configurations**: Pre-defined parameter sets for different use cases

### Educational Applications
- **Socratic Prompts**: Dynamic questions based on current system state
- **Parameter Exploration**: Visual feedback on how adjustments affect outcomes
- **Cognitive Alignment**: Understanding human-like reasoning patterns
- **Agency Promotion**: User control over system behavior and interpretation

## 📱 SwiftUI Interface

The system provides a modern, interactive SwiftUI interface with:

- **Parameter Sliders**: Real-time adjustment of all loss components
- **Visual Metrics**: Gauge displays for accuracy, human alignment, and efficiency
- **Educational Insights**: Contextual tips and Socratic questions
- **Focus Session Timer**: 90-minute work blocks with break reminders
- **Recursive Calculator**: Multi-level loss computation for complex operations

## 🛠 Installation

### Requirements
- iOS 16.0+ / macOS 13.0+
- Swift 5.9+
- Xcode 14.0+

### Swift Package Manager
```swift
dependencies: [
    .package(url: "path/to/HybridNeuroSymbolicSystem", from: "1.0.0")
]
```

## 📖 Usage Examples

### Basic Loss Computation
```swift
import HybridNeuroSymbolicSystem

// Compute basic composite loss
let loss = computeCompositeLoss(
    logicOutput: 0.9,
    groundTruth: 1.0,
    lambda1: 0.1,  // Cognitive regularization
    lambda2: 0.1   // Efficiency regularization
)

print("Total Loss: \(loss.totalLoss)")
print("Task Loss: \(loss.taskLoss)")
print("Cognitive Regularizer: \(loss.cognitiveRegularizer)")
print("Efficiency Regularizer: \(loss.efficiencyRegularizer)")
```

### Interactive SwiftUI View
```swift
import SwiftUI
import HybridNeuroSymbolicSystem

struct ContentView: View {
    var body: some View {
        CompositeLossView()
    }
}
```

### Recursive Loss for Complex Operations
```swift
// Compute loss for nested logical operations
let operations = [0.9, 0.85, 0.92]  // Multi-level outputs
let groundTruths = [1.0, 0.9, 0.95] // Expected results

let recursiveLoss = CompositeLoss.computeRecursiveLoss(
    operations: operations,
    groundTruths: groundTruths,
    lambda1: 0.1,
    lambda2: 0.1
)
```

### ViewModel Integration
```swift
@StateObject private var viewModel = LossViewModel()

// Apply preset configurations
viewModel.applyPreset(.humanLikeReasoning)

// Start focus session
viewModel.startFocusSession()

// Access educational insights
for prompt in viewModel.educationalPrompts {
    print("Socratic Question: \(prompt)")
}
```

## 🎓 Educational Applications

### Socratic Questioning
The system generates contextual questions to promote deeper understanding:

- *"How might adjusting the neural-symbolic blend improve accuracy?"*
- *"What human reasoning patterns should we prioritize?"*
- *"How does increasing λ₁ affect the system's bias toward human-like reasoning?"*

### Focus Session Management
- **90-minute work blocks** for sustained cognitive flow
- **Break reminders** for parasympathetic reset
- **Progress tracking** with visual indicators

### Parameter Exploration
- **Real-time feedback** on parameter changes
- **Visual metrics** showing accuracy, human alignment, and efficiency
- **Preset configurations** for different use cases

## 🔧 Advanced Features

### Preset Configurations
```swift
enum LossPreset {
    case accuracyFocused      // λ₁=0.05, λ₂=0.05
    case humanLikeReasoning   // λ₁=0.3, λ₂=0.1
    case efficiencyOptimized  // λ₁=0.1, λ₂=0.3
    case balanced            // λ₁=0.15, λ₂=0.15
}
```

### Enhanced Analysis
```swift
let enhancedLoss = EnhancedCompositeLoss(
    logicOutput: 0.9,
    groundTruth: 1.0,
    lambda1: 0.1,
    lambda2: 0.1
)

// Access performance metrics
print("Accuracy: \(enhancedLoss.analysis.accuracy)")
print("Human Alignment: \(enhancedLoss.analysis.humanAlignment)")
print("Efficiency: \(enhancedLoss.analysis.efficiency)")

// Educational insights
for insight in enhancedLoss.insights {
    print("Insight: \(insight)")
}
```

## 🧠 Cognitive Science Integration

### Human-Like Reasoning Patterns
- **86% alignment benchmark** based on expert heuristics
- **Cognitive regularizer** penalizes deviations from human patterns
- **Interpretable outputs** that align with human intuition

### Focus and Agency
- **90-minute focus blocks** for optimal cognitive performance
- **User agency** through parameter control and interpretation
- **Socratic scaffolding** for deeper understanding

## 🔬 Research Applications

### Hybrid System Optimization
- **Neural-symbolic integration** with balanced regularization
- **Multi-objective optimization** across accuracy, interpretability, and efficiency
- **Recursive computation** for complex logical operations

### Educational Technology
- **Interactive learning** through parameter exploration
- **Cognitive load management** with focus sessions
- **Agency promotion** through user control and interpretation

## 📊 Performance Metrics

The system provides comprehensive metrics:

- **Accuracy**: How well outputs match ground truth
- **Human Alignment**: Similarity to human reasoning patterns
- **Efficiency**: Computational simplicity score
- **Composite Loss**: Balanced overall performance

## 🤝 Contributing

This system is designed for educational and research purposes. Contributions are welcome in areas such as:

- Additional loss functions and regularizers
- Enhanced educational features
- Integration with other hybrid systems
- Cognitive science applications

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Inspired by meta-optimized hybrid neuro-symbolic systems
- Educational principles from Socratic questioning and dialogic pedagogy
- Cognitive science insights on human reasoning patterns
- Focus session methodology for sustained cognitive performance

---

**💡 Tip**: Use this system in 90-minute focus blocks, followed by brief walks for parasympathetic reset, to sustain cognitive flow while maintaining user agency through manual verification and parameter exploration.