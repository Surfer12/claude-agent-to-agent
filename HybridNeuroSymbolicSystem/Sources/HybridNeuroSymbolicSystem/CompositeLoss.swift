import Foundation

/// Represents the composite loss function for the hybrid neuro-symbolic system
/// that balances accuracy, human-like reasoning, and computational efficiency
public struct CompositeLoss {
    
    /// Task-specific loss (L_logic) - measures accuracy against ground truth
    public let taskLoss: Double
    
    /// Cognitive regularizer (R_cog) - penalizes deviations from human-like reasoning
    public let cognitiveRegularizer: Double
    
    /// Efficiency regularizer (R_eff) - encourages computational simplicity
    public let efficiencyRegularizer: Double
    
    /// Cognitive regularization weight (λ₁)
    public let lambda1: Double
    
    /// Efficiency regularization weight (λ₂)
    public let lambda2: Double
    
    /// Total composite loss
    public var totalLoss: Double {
        return taskLoss + lambda1 * cognitiveRegularizer + lambda2 * efficiencyRegularizer
    }
    
    public init(
        taskLoss: Double,
        cognitiveRegularizer: Double,
        efficiencyRegularizer: Double,
        lambda1: Double = 0.1,
        lambda2: Double = 0.1
    ) {
        self.taskLoss = taskLoss
        self.cognitiveRegularizer = cognitiveRegularizer
        self.efficiencyRegularizer = efficiencyRegularizer
        self.lambda1 = lambda1
        self.lambda2 = lambda2
    }
}

/// Computes the composite loss for a given logic output and ground truth
/// - Parameters:
///   - logicOutput: The blended output from the hybrid system
///   - groundTruth: The expected correct result
///   - lambda1: Cognitive regularization weight (default: 0.1)
///   - lambda2: Efficiency regularization weight (default: 0.1)
/// - Returns: Composite loss balancing accuracy, human-like reasoning, and efficiency
public func computeCompositeLoss(
    logicOutput: Double,
    groundTruth: Double,
    lambda1: Double = 0.1,
    lambda2: Double = 0.1
) -> CompositeLoss {
    
    // Task-specific loss (L_logic) - Mean Squared Error for simplicity
    let taskLoss = pow(logicOutput - groundTruth, 2)
    
    // Cognitive regularizer (R_cog) - Penalizes deviations from human-like patterns
    // Aligns to ~86% human-like benchmark (based on expert heuristics)
    let humanLikeBenchmark = 0.86
    let cognitiveRegularizer = abs(logicOutput - humanLikeBenchmark)
    
    // Efficiency regularizer (R_eff) - Encourages computational simplicity
    // Simulated overfitting penalty (12-15% improvement target)
    let efficiencyRegularizer = logicOutput * 0.15
    
    return CompositeLoss(
        taskLoss: taskLoss,
        cognitiveRegularizer: cognitiveRegularizer,
        efficiencyRegularizer: efficiencyRegularizer,
        lambda1: lambda1,
        lambda2: lambda2
    )
}

/// Enhanced composite loss computation with additional analysis
public struct EnhancedCompositeLoss {
    
    /// Basic composite loss
    public let compositeLoss: CompositeLoss
    
    /// Analysis of loss components
    public let analysis: LossAnalysis
    
    /// Educational insights for user understanding
    public let insights: [String]
    
    public init(
        logicOutput: Double,
        groundTruth: Double,
        lambda1: Double = 0.1,
        lambda2: Double = 0.1
    ) {
        self.compositeLoss = computeCompositeLoss(
            logicOutput: logicOutput,
            groundTruth: groundTruth,
            lambda1: lambda1,
            lambda2: lambda2
        )
        
        self.analysis = LossAnalysis(
            accuracy: 1.0 - abs(logicOutput - groundTruth),
            humanAlignment: 1.0 - abs(logicOutput - 0.86),
            efficiency: 1.0 - (logicOutput * 0.15)
        )
        
        self.insights = Self.generateInsights(
            logicOutput: logicOutput,
            groundTruth: groundTruth,
            lambda1: lambda1,
            lambda2: lambda2,
            analysis: self.analysis
        )
    }
    
    private static func generateInsights(
        logicOutput: Double,
        groundTruth: Double,
        lambda1: Double,
        lambda2: Double,
        analysis: LossAnalysis
    ) -> [String] {
        var insights: [String] = []
        
        // Accuracy insights
        if analysis.accuracy > 0.9 {
            insights.append("High accuracy achieved - the system is performing well on the core task.")
        } else if analysis.accuracy < 0.7 {
            insights.append("Consider adjusting the neural-symbolic blend to improve accuracy.")
        }
        
        // Human alignment insights
        if analysis.humanAlignment > 0.8 {
            insights.append("Strong human-like reasoning patterns detected.")
        } else {
            insights.append("Increasing λ₁ may help align with human cognitive patterns.")
        }
        
        // Efficiency insights
        if analysis.efficiency > 0.8 {
            insights.append("Good computational efficiency maintained.")
        } else {
            insights.append("Consider increasing λ₂ to reduce computational complexity.")
        }
        
        // Parameter tuning suggestions
        if lambda1 > 0.5 {
            insights.append("High cognitive regularization - outputs will prioritize human-like reasoning.")
        }
        
        if lambda2 > 0.5 {
            insights.append("High efficiency regularization - outputs will favor computational simplicity.")
        }
        
        return insights
    }
}

/// Analysis of loss components for educational purposes
public struct LossAnalysis {
    /// Accuracy score (0-1, higher is better)
    public let accuracy: Double
    
    /// Human alignment score (0-1, higher is better)
    public let humanAlignment: Double
    
    /// Efficiency score (0-1, higher is better)
    public let efficiency: Double
    
    public init(accuracy: Double, humanAlignment: Double, efficiency: Double) {
        self.accuracy = max(0, min(1, accuracy))
        self.humanAlignment = max(0, min(1, humanAlignment))
        self.efficiency = max(0, min(1, efficiency))
    }
}

/// Extension for recursive loss computation (nested logic flows)
public extension CompositeLoss {
    
    /// Computes composite loss for nested logical operations
    /// Useful for complex theorem verification with multiple layers
    static func computeRecursiveLoss(
        operations: [Double],
        groundTruths: [Double],
        lambda1: Double = 0.1,
        lambda2: Double = 0.1,
        depth: Int = 0
    ) -> CompositeLoss {
        
        guard !operations.isEmpty else {
            return CompositeLoss(
                taskLoss: 0,
                cognitiveRegularizer: 0,
                efficiencyRegularizer: 0,
                lambda1: lambda1,
                lambda2: lambda2
            )
        }
        
        // Compute loss for current level
        let currentLoss = computeCompositeLoss(
            logicOutput: operations[0],
            groundTruth: groundTruths[0],
            lambda1: lambda1,
            lambda2: lambda2
        )
        
        // Recursive computation for nested operations
        if operations.count > 1 {
            let nestedLoss = computeRecursiveLoss(
                operations: Array(operations.dropFirst()),
                groundTruths: Array(groundTruths.dropFirst()),
                lambda1: lambda1,
                lambda2: lambda2,
                depth: depth + 1
            )
            
            // Combine losses with depth weighting
            let depthWeight = 1.0 / Double(depth + 1)
            return CompositeLoss(
                taskLoss: currentLoss.taskLoss + depthWeight * nestedLoss.taskLoss,
                cognitiveRegularizer: currentLoss.cognitiveRegularizer + depthWeight * nestedLoss.cognitiveRegularizer,
                efficiencyRegularizer: currentLoss.efficiencyRegularizer + depthWeight * nestedLoss.efficiencyRegularizer,
                lambda1: lambda1,
                lambda2: lambda2
            )
        }
        
        return currentLoss
    }
}