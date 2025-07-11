import XCTest
@testable import HybridNeuroSymbolicSystem

final class CompositeLossTests: XCTestCase {
    
    // MARK: - Basic Loss Computation Tests
    
    func testBasicCompositeLossComputation() {
        // Test basic loss computation
        let loss = computeCompositeLoss(
            logicOutput: 0.9,
            groundTruth: 1.0,
            lambda1: 0.1,
            lambda2: 0.1
        )
        
        // Verify components
        XCTAssertEqual(loss.taskLoss, 0.01, accuracy: 0.001) // (0.9 - 1.0)² = 0.01
        XCTAssertEqual(loss.cognitiveRegularizer, 0.04, accuracy: 0.001) // |0.9 - 0.86| = 0.04
        XCTAssertEqual(loss.efficiencyRegularizer, 0.135, accuracy: 0.001) // 0.9 * 0.15 = 0.135
        XCTAssertEqual(loss.lambda1, 0.1)
        XCTAssertEqual(loss.lambda2, 0.1)
        
        // Verify total loss
        let expectedTotal = 0.01 + 0.1 * 0.04 + 0.1 * 0.135
        XCTAssertEqual(loss.totalLoss, expectedTotal, accuracy: 0.001)
    }
    
    func testPerfectAlignment() {
        // Test when output perfectly matches ground truth and human benchmark
        let loss = computeCompositeLoss(
            logicOutput: 0.86, // Matches human benchmark
            groundTruth: 0.86, // Matches output
            lambda1: 0.1,
            lambda2: 0.1
        )
        
        XCTAssertEqual(loss.taskLoss, 0.0, accuracy: 0.001) // Perfect accuracy
        XCTAssertEqual(loss.cognitiveRegularizer, 0.0, accuracy: 0.001) // Perfect human alignment
        XCTAssertEqual(loss.efficiencyRegularizer, 0.129, accuracy: 0.001) // 0.86 * 0.15
    }
    
    func testHighRegularization() {
        // Test with high regularization weights
        let loss = computeCompositeLoss(
            logicOutput: 0.9,
            groundTruth: 1.0,
            lambda1: 0.5, // High cognitive regularization
            lambda2: 0.5  // High efficiency regularization
        )
        
        // Total loss should be higher due to increased regularization
        XCTAssertGreaterThan(loss.totalLoss, 0.1)
    }
    
    // MARK: - Enhanced Loss Tests
    
    func testEnhancedCompositeLoss() {
        let enhancedLoss = EnhancedCompositeLoss(
            logicOutput: 0.9,
            groundTruth: 1.0,
            lambda1: 0.1,
            lambda2: 0.1
        )
        
        // Verify analysis metrics
        XCTAssertEqual(enhancedLoss.analysis.accuracy, 0.9, accuracy: 0.001) // 1.0 - |0.9 - 1.0|
        XCTAssertEqual(enhancedLoss.analysis.humanAlignment, 0.96, accuracy: 0.001) // 1.0 - |0.9 - 0.86|
        XCTAssertEqual(enhancedLoss.analysis.efficiency, 0.865, accuracy: 0.001) // 1.0 - (0.9 * 0.15)
        
        // Verify insights are generated
        XCTAssertFalse(enhancedLoss.insights.isEmpty)
    }
    
    func testInsightGeneration() {
        // Test high accuracy scenario
        let highAccuracyLoss = EnhancedCompositeLoss(
            logicOutput: 0.95,
            groundTruth: 0.95,
            lambda1: 0.1,
            lambda2: 0.1
        )
        
        // Should have high accuracy insight
        let hasHighAccuracyInsight = highAccuracyLoss.insights.contains { insight in
            insight.contains("High accuracy achieved")
        }
        XCTAssertTrue(hasHighAccuracyInsight)
        
        // Test low accuracy scenario
        let lowAccuracyLoss = EnhancedCompositeLoss(
            logicOutput: 0.5,
            groundTruth: 1.0,
            lambda1: 0.1,
            lambda2: 0.1
        )
        
        // Should have low accuracy insight
        let hasLowAccuracyInsight = lowAccuracyLoss.insights.contains { insight in
            insight.contains("Consider adjusting")
        }
        XCTAssertTrue(hasLowAccuracyInsight)
    }
    
    // MARK: - Recursive Loss Tests
    
    func testRecursiveLossComputation() {
        let operations = [0.9, 0.85, 0.92]
        let groundTruths = [1.0, 0.9, 0.95]
        
        let recursiveLoss = CompositeLoss.computeRecursiveLoss(
            operations: operations,
            groundTruths: groundTruths,
            lambda1: 0.1,
            lambda2: 0.1
        )
        
        // Should have non-zero loss
        XCTAssertGreaterThan(recursiveLoss.totalLoss, 0)
        
        // Should have non-zero components
        XCTAssertGreaterThan(recursiveLoss.taskLoss, 0)
        XCTAssertGreaterThan(recursiveLoss.cognitiveRegularizer, 0)
        XCTAssertGreaterThan(recursiveLoss.efficiencyRegularizer, 0)
    }
    
    func testRecursiveLossWithSingleOperation() {
        let operations = [0.9]
        let groundTruths = [1.0]
        
        let recursiveLoss = CompositeLoss.computeRecursiveLoss(
            operations: operations,
            groundTruths: groundTruths,
            lambda1: 0.1,
            lambda2: 0.1
        )
        
        // Should match basic loss computation
        let basicLoss = computeCompositeLoss(
            logicOutput: 0.9,
            groundTruth: 1.0,
            lambda1: 0.1,
            lambda2: 0.1
        )
        
        XCTAssertEqual(recursiveLoss.totalLoss, basicLoss.totalLoss, accuracy: 0.001)
    }
    
    func testRecursiveLossWithEmptyOperations() {
        let recursiveLoss = CompositeLoss.computeRecursiveLoss(
            operations: [],
            groundTruths: [],
            lambda1: 0.1,
            lambda2: 0.1
        )
        
        // Should return zero loss for empty operations
        XCTAssertEqual(recursiveLoss.totalLoss, 0, accuracy: 0.001)
        XCTAssertEqual(recursiveLoss.taskLoss, 0, accuracy: 0.001)
        XCTAssertEqual(recursiveLoss.cognitiveRegularizer, 0, accuracy: 0.001)
        XCTAssertEqual(recursiveLoss.efficiencyRegularizer, 0, accuracy: 0.001)
    }
    
    // MARK: - ViewModel Tests
    
    func testLossViewModelInitialization() {
        let viewModel = LossViewModel()
        
        // Verify default values
        XCTAssertEqual(viewModel.logicOutput, 0.9)
        XCTAssertEqual(viewModel.groundTruth, 1.0)
        XCTAssertEqual(viewModel.lambda1, 0.1)
        XCTAssertEqual(viewModel.lambda2, 0.1)
        
        // Verify loss is computed
        XCTAssertNotNil(viewModel.compositeLoss)
        XCTAssertNotNil(viewModel.enhancedLoss)
    }
    
    func testLossViewModelParameterUpdates() {
        let viewModel = LossViewModel()
        
        // Update parameters
        viewModel.logicOutput = 0.8
        viewModel.groundTruth = 0.9
        viewModel.lambda1 = 0.2
        viewModel.lambda2 = 0.3
        
        // Verify loss is recalculated
        XCTAssertNotNil(viewModel.compositeLoss)
        XCTAssertEqual(viewModel.compositeLoss?.lambda1, 0.2)
        XCTAssertEqual(viewModel.compositeLoss?.lambda2, 0.3)
    }
    
    func testLossViewModelPresetApplication() {
        let viewModel = LossViewModel()
        
        // Apply human-like reasoning preset
        viewModel.applyPreset(.humanLikeReasoning)
        
        XCTAssertEqual(viewModel.lambda1, 0.3)
        XCTAssertEqual(viewModel.lambda2, 0.1)
        
        // Apply efficiency optimized preset
        viewModel.applyPreset(.efficiencyOptimized)
        
        XCTAssertEqual(viewModel.lambda1, 0.1)
        XCTAssertEqual(viewModel.lambda2, 0.3)
    }
    
    func testLossViewModelFocusSession() {
        let viewModel = LossViewModel()
        
        // Initially not in focus session
        XCTAssertFalse(viewModel.isInFocusSession)
        XCTAssertEqual(viewModel.focusSessionElapsed, 0)
        
        // Start focus session
        viewModel.startFocusSession()
        XCTAssertTrue(viewModel.isInFocusSession)
        
        // End focus session
        viewModel.endFocusSession()
        XCTAssertFalse(viewModel.isInFocusSession)
        XCTAssertEqual(viewModel.focusSessionElapsed, 0)
    }
    
    func testLossViewModelResetToDefaults() {
        let viewModel = LossViewModel()
        
        // Change parameters
        viewModel.logicOutput = 0.5
        viewModel.groundTruth = 0.7
        viewModel.lambda1 = 0.5
        viewModel.lambda2 = 0.5
        
        // Reset to defaults
        viewModel.resetToDefaults()
        
        XCTAssertEqual(viewModel.logicOutput, 0.9)
        XCTAssertEqual(viewModel.groundTruth, 1.0)
        XCTAssertEqual(viewModel.lambda1, 0.1)
        XCTAssertEqual(viewModel.lambda2, 0.1)
    }
    
    // MARK: - Educational Prompts Tests
    
    func testEducationalPromptsGeneration() {
        let viewModel = LossViewModel()
        
        // Initially should have some prompts
        XCTAssertFalse(viewModel.educationalPrompts.isEmpty)
        
        // Test high cognitive regularization
        viewModel.lambda1 = 0.6
        viewModel.calculateLoss()
        
        let hasCognitivePrompt = viewModel.educationalPrompts.contains { prompt in
            prompt.contains("λ₁") && prompt.contains("human-like reasoning")
        }
        XCTAssertTrue(hasCognitivePrompt)
        
        // Test high efficiency regularization
        viewModel.lambda2 = 0.6
        viewModel.calculateLoss()
        
        let hasEfficiencyPrompt = viewModel.educationalPrompts.contains { prompt in
            prompt.contains("λ₂") && prompt.contains("computational efficiency")
        }
        XCTAssertTrue(hasEfficiencyPrompt)
    }
    
    // MARK: - Performance Tests
    
    func testLossComputationPerformance() {
        measure {
            for _ in 0..<1000 {
                _ = computeCompositeLoss(
                    logicOutput: Double.random(in: 0...1),
                    groundTruth: Double.random(in: 0...1),
                    lambda1: Double.random(in: 0...1),
                    lambda2: Double.random(in: 0...1)
                )
            }
        }
    }
    
    func testRecursiveLossPerformance() {
        let operations = Array(repeating: Double.random(in: 0...1), count: 10)
        let groundTruths = Array(repeating: Double.random(in: 0...1), count: 10)
        
        measure {
            for _ in 0..<100 {
                _ = CompositeLoss.computeRecursiveLoss(
                    operations: operations,
                    groundTruths: groundTruths,
                    lambda1: 0.1,
                    lambda2: 0.1
                )
            }
        }
    }
    
    // MARK: - Edge Cases
    
    func testEdgeCaseZeroValues() {
        let loss = computeCompositeLoss(
            logicOutput: 0.0,
            groundTruth: 0.0,
            lambda1: 0.0,
            lambda2: 0.0
        )
        
        XCTAssertEqual(loss.taskLoss, 0.0, accuracy: 0.001)
        XCTAssertEqual(loss.cognitiveRegularizer, 0.86, accuracy: 0.001) // |0.0 - 0.86|
        XCTAssertEqual(loss.efficiencyRegularizer, 0.0, accuracy: 0.001) // 0.0 * 0.15
        XCTAssertEqual(loss.totalLoss, 0.0, accuracy: 0.001) // All weights are 0
    }
    
    func testEdgeCaseMaximumValues() {
        let loss = computeCompositeLoss(
            logicOutput: 1.0,
            groundTruth: 1.0,
            lambda1: 1.0,
            lambda2: 1.0
        )
        
        XCTAssertEqual(loss.taskLoss, 0.0, accuracy: 0.001) // Perfect accuracy
        XCTAssertEqual(loss.cognitiveRegularizer, 0.14, accuracy: 0.001) // |1.0 - 0.86|
        XCTAssertEqual(loss.efficiencyRegularizer, 0.15, accuracy: 0.001) // 1.0 * 0.15
        XCTAssertEqual(loss.totalLoss, 0.29, accuracy: 0.001) // 0.0 + 1.0 * 0.14 + 1.0 * 0.15
    }
}