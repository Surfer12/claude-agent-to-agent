import Foundation
import SwiftUI

/// ViewModel for the composite loss calculator that manages state and provides
/// educational insights through Socratic questioning
@MainActor
public class LossViewModel: ObservableObject {
    
    // MARK: - Published Properties
    
    /// The blended output from the hybrid neuro-symbolic system
    @Published var logicOutput: Double = 0.9 {
        didSet {
            calculateLoss()
            updateEducationalPrompts()
        }
    }
    
    /// The expected ground truth result
    @Published var groundTruth: Double = 1.0 {
        didSet {
            calculateLoss()
            updateEducationalPrompts()
        }
    }
    
    /// Cognitive regularization weight (λ₁)
    @Published var lambda1: Double = 0.1 {
        didSet {
            calculateLoss()
            updateEducationalPrompts()
        }
    }
    
    /// Efficiency regularization weight (λ₂)
    @Published var lambda2: Double = 0.1 {
        didSet {
            calculateLoss()
            updateEducationalPrompts()
        }
    }
    
    /// The computed composite loss
    @Published var compositeLoss: CompositeLoss?
    
    /// Enhanced loss analysis with insights
    @Published var enhancedLoss: EnhancedCompositeLoss?
    
    /// Educational prompts for Socratic questioning
    @Published var educationalPrompts: [String] = []
    
    /// Current focus session timer (for 90-minute blocks)
    @Published var focusSessionElapsed: TimeInterval = 0
    
    /// Whether the user is in a focus session
    @Published var isInFocusSession: Bool = false
    
    // MARK: - Private Properties
    
    private var focusTimer: Timer?
    private let focusSessionDuration: TimeInterval = 90 * 60 // 90 minutes
    
    // MARK: - Initialization
    
    public init() {
        calculateLoss()
        updateEducationalPrompts()
    }
    
    // MARK: - Public Methods
    
    /// Calculates the composite loss with current parameters
    public func calculateLoss() {
        compositeLoss = computeCompositeLoss(
            logicOutput: logicOutput,
            groundTruth: groundTruth,
            lambda1: lambda1,
            lambda2: lambda2
        )
        
        enhancedLoss = EnhancedCompositeLoss(
            logicOutput: logicOutput,
            groundTruth: groundTruth,
            lambda1: lambda1,
            lambda2: lambda2
        )
    }
    
    /// Starts a 90-minute focus session with timer
    public func startFocusSession() {
        isInFocusSession = true
        focusSessionElapsed = 0
        
        focusTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            guard let self = self else { return }
            
            self.focusSessionElapsed += 1
            
            // Check if focus session should end
            if self.focusSessionElapsed >= self.focusSessionDuration {
                self.endFocusSession()
            }
        }
    }
    
    /// Ends the current focus session
    public func endFocusSession() {
        isInFocusSession = false
        focusTimer?.invalidate()
        focusTimer = nil
        focusSessionElapsed = 0
    }
    
    /// Computes recursive loss for nested logical operations
    public func computeRecursiveLoss(operations: [Double], groundTruths: [Double]) -> CompositeLoss {
        return CompositeLoss.computeRecursiveLoss(
            operations: operations,
            groundTruths: groundTruths,
            lambda1: lambda1,
            lambda2: lambda2
        )
    }
    
    /// Resets all parameters to default values
    public func resetToDefaults() {
        logicOutput = 0.9
        groundTruth = 1.0
        lambda1 = 0.1
        lambda2 = 0.1
    }
    
    /// Applies a preset configuration for different use cases
    public func applyPreset(_ preset: LossPreset) {
        switch preset {
        case .accuracyFocused:
            lambda1 = 0.05
            lambda2 = 0.05
        case .humanLikeReasoning:
            lambda1 = 0.3
            lambda2 = 0.1
        case .efficiencyOptimized:
            lambda1 = 0.1
            lambda2 = 0.3
        case .balanced:
            lambda1 = 0.15
            lambda2 = 0.15
        }
    }
    
    // MARK: - Private Methods
    
    private func updateEducationalPrompts() {
        var prompts: [String] = []
        
        // Socratic prompts based on current state
        if let enhancedLoss = enhancedLoss {
            if enhancedLoss.analysis.accuracy < 0.8 {
                prompts.append("How might adjusting the neural-symbolic blend improve accuracy?")
            }
            
            if enhancedLoss.analysis.humanAlignment < 0.7 {
                prompts.append("What human reasoning patterns should we prioritize?")
            }
            
            if enhancedLoss.analysis.efficiency < 0.8 {
                prompts.append("How can we simplify the computation without losing accuracy?")
            }
        }
        
        // Parameter-specific prompts
        if lambda1 > 0.2 {
            prompts.append("How does increasing λ₁ affect the system's bias toward human-like reasoning?")
        }
        
        if lambda2 > 0.2 {
            prompts.append("What trade-offs occur when prioritizing computational efficiency?")
        }
        
        // Focus session prompts
        if isInFocusSession {
            let remainingTime = focusSessionDuration - focusSessionElapsed
            if remainingTime < 300 { // Last 5 minutes
                prompts.append("Consider taking a brief walk after this session for cognitive reset.")
            }
        }
        
        educationalPrompts = prompts
    }
}

// MARK: - Supporting Types

/// Preset configurations for different use cases
public enum LossPreset: String, CaseIterable {
    case accuracyFocused = "Accuracy Focused"
    case humanLikeReasoning = "Human-like Reasoning"
    case efficiencyOptimized = "Efficiency Optimized"
    case balanced = "Balanced"
    
    var description: String {
        switch self {
        case .accuracyFocused:
            return "Prioritizes task accuracy with minimal regularization"
        case .humanLikeReasoning:
            return "Emphasizes human-like cognitive patterns"
        case .efficiencyOptimized:
            return "Optimizes for computational simplicity"
        case .balanced:
            return "Balanced approach across all objectives"
        }
    }
}

// MARK: - Extensions

extension LossViewModel {
    
    /// Formatted focus session time remaining
    var focusSessionTimeRemaining: String {
        let remaining = focusSessionDuration - focusSessionElapsed
        let minutes = Int(remaining) / 60
        let seconds = Int(remaining) % 60
        return String(format: "%02d:%02d", minutes, seconds)
    }
    
    /// Focus session progress (0-1)
    var focusSessionProgress: Double {
        return focusSessionElapsed / focusSessionDuration
    }
    
    /// Whether to show focus session warning
    var shouldShowFocusWarning: Bool {
        return isInFocusSession && focusSessionElapsed > focusSessionDuration * 0.8
    }
}