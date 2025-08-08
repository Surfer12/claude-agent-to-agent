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
    
    func runCompleteAnalysis() {
        print("Phase-Space Trajectory Analysis")
        print("=" * 40)
        
        // Generate trajectory
        let trajectory = generateTrajectory()
        
        // Analyze specific points
        let startPsi = analyzeTrajectoryPoint(tPoint: 0.0, alpha: trajectory.alpha, lambda1: trajectory.lambda1, lambda2: trajectory.lambda2)
        let midpointPsi = analyzeTrajectoryPoint(tPoint: 0.5, alpha: trajectory.alpha, lambda1: trajectory.lambda1, lambda2: trajectory.lambda2)
        let endPsi = analyzeTrajectoryPoint(tPoint: 1.0, alpha: trajectory.alpha, lambda1: trajectory.lambda1, lambda2: trajectory.lambda2)
        
        print("\n" + "=" * 40)
        print("Analysis Complete!")
        print("Start Ψ(x): \(String(format: "%.3f", startPsi))")
        print("Midpoint Ψ(x): \(String(format: "%.3f", midpointPsi))")
        print("End Ψ(x): \(String(format: "%.3f", endPsi))")
    }
    
    // MARK: - Mathematical Formula Implementations
    
    /// Calculates the trajectory equations
    /// - Parameter t: Time parameter (0 to 1)
    /// - Returns: Tuple of (α(t), λ₁(t), λ₂(t))
    func calculateTrajectoryEquations(t: Double) -> (alpha: Double, lambda1: Double, lambda2: Double) {
        let alpha = t
        let lambda1 = 2.0 * (1.0 - t)
        let lambda2 = 2.0 * (1.0 - t)
        return (alpha: alpha, lambda1: lambda1, lambda2: lambda2)
    }
    
    /// Calculates the hybrid output component
    /// - Parameters:
    ///   - alpha: Weight parameter α(t)
    ///   - S_x: Symbolic output
    ///   - N_x: Neural output
    /// - Returns: Hybrid output value
    func calculateHybridOutput(alpha: Double, S_x: Double, N_x: Double) -> Double {
        return alpha * S_x + (1.0 - alpha) * N_x
    }
    
    /// Calculates the regularization component
    /// - Parameters:
    ///   - lambda1: λ₁(t) regularization weight
    ///   - lambda2: λ₂(t) regularization weight
    ///   - R_cognitive: Cognitive penalty
    ///   - R_efficiency: Efficiency penalty
    /// - Returns: Regularization factor
    func calculateRegularization(lambda1: Double, lambda2: Double, R_cognitive: Double, R_efficiency: Double) -> Double {
        return exp(-(lambda1 * R_cognitive + lambda2 * R_efficiency))
    }
    
    /// Calculates the cross-term component
    /// - Parameters:
    ///   - w_cross: Cross-term weight
    ///   - S_x: Symbolic output
    ///   - N_x: Neural output
    /// - Returns: Cross-term value
    func calculateCrossTerm(w_cross: Double, S_x: Double, N_x: Double) -> Double {
        return w_cross * (S_x * N_x - N_x * S_x) // This would be 0 in this case
    }
    
    /// Calculates the complete Ψ(x) equation
    /// - Parameters:
    ///   - alpha: α(t) weight parameter
    ///   - lambda1: λ₁(t) regularization weight
    ///   - lambda2: λ₂(t) regularization weight
    ///   - S_x: Symbolic output
    ///   - N_x: Neural output
    ///   - w_cross: Cross-term weight
    ///   - R_cognitive: Cognitive penalty
    ///   - R_efficiency: Efficiency penalty
    ///   - P_H_E_beta: Probability with bias
    /// - Returns: Complete Ψ(x) value
    func calculatePsi(alpha: Double, lambda1: Double, lambda2: Double, 
                     S_x: Double, N_x: Double, w_cross: Double,
                     R_cognitive: Double, R_efficiency: Double, P_H_E_beta: Double) -> Double {
        
        let hybridOutput = calculateHybridOutput(alpha: alpha, S_x: S_x, N_x: N_x)
        let crossTerm = calculateCrossTerm(w_cross: w_cross, S_x: S_x, N_x: N_x)
        let regularization = calculateRegularization(lambda1: lambda1, lambda2: lambda2, 
                                                  R_cognitive: R_cognitive, R_efficiency: R_efficiency)
        
        return (hybridOutput + crossTerm) * regularization * P_H_E_beta
    }
    
    // MARK: - Vectorized Operations using Accelerate
    
    /// Performs vectorized trajectory calculations using Accelerate framework
    /// - Parameter t: Array of time points
    /// - Returns: Arrays of calculated values
    func calculateVectorizedTrajectory(t: [Double]) -> (alpha: [Double], lambda1: [Double], lambda2: [Double]) {
        let alpha = t
        let lambda1 = t.map { 2.0 * (1.0 - $0) }
        let lambda2 = t.map { 2.0 * (1.0 - $0) }
        
        return (alpha: alpha, lambda1: lambda1, lambda2: lambda2)
    }
    
    /// Calculates vectorized Ψ(x) values for the entire trajectory
    /// - Parameter trajectory: Trajectory data
    /// - Returns: Array of Ψ(x) values
    func calculateVectorizedPsi(trajectory: (t: [Double], alpha: [Double], lambda1: [Double], lambda2: [Double])) -> [Double] {
        let S_x: Double = 0.60
        let N_x: Double = 0.80
        let w_cross: Double = 0.1
        let R_cognitive: Double = 0.25
        let R_efficiency: Double = 0.10
        let P_H_E_beta: Double = 0.70 * 1.4
        
        var psiValues: [Double] = []
        
        for i in 0..<trajectory.t.count {
            let psi = calculatePsi(alpha: trajectory.alpha[i], 
                                 lambda1: trajectory.lambda1[i], 
                                 lambda2: trajectory.lambda2[i],
                                 S_x: S_x, N_x: N_x, w_cross: w_cross,
                                 R_cognitive: R_cognitive, R_efficiency: R_efficiency, 
                                 P_H_E_beta: P_H_E_beta)
            psiValues.append(psi)
        }
        
        return psiValues
    }
}

// MARK: - Extensions for Mathematical Operations

extension PhaseSpaceAnalyzer {
    
    /// Calculates the derivative of the trajectory
    /// - Parameter t: Time parameter
    /// - Returns: Derivatives of (α'(t), λ₁'(t), λ₂'(t))
    func calculateTrajectoryDerivatives(t: Double) -> (alphaPrime: Double, lambda1Prime: Double, lambda2Prime: Double) {
        let alphaPrime = 1.0 // dα/dt = 1
        let lambda1Prime = -2.0 // dλ₁/dt = -2
        let lambda2Prime = -2.0 // dλ₂/dt = -2
        return (alphaPrime: alphaPrime, lambda1Prime: lambda1Prime, lambda2Prime: lambda2Prime)
    }
    
    /// Calculates the curvature of the trajectory
    /// - Parameter t: Time parameter
    /// - Returns: Curvature value
    func calculateTrajectoryCurvature(t: Double) -> Double {
        // For a linear trajectory, curvature is 0
        return 0.0
    }
    
    /// Calculates the arc length of the trajectory from 0 to t
    /// - Parameter t: Time parameter
    /// - Returns: Arc length
    func calculateArcLength(t: Double) -> Double {
        // For linear trajectory: √(1² + (-2)² + (-2)²) = √9 = 3
        return 3.0 * t
    }
}