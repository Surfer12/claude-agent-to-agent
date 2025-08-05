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
    
    // MARK: - Mathematical Formula Components
    
    /// Calculates the hybrid output component: α(t)S(x) + (1-α(t))N(x)
    /// - Parameters:
    ///   - alpha: α(t) weight parameter
    ///   - S_x: Symbolic output
    ///   - N_x: Neural output
    /// - Returns: Hybrid output value
    func calculateHybridOutput(alpha: Double, S_x: Double, N_x: Double) -> Double {
        return alpha * S_x + (1.0 - alpha) * N_x
    }
    
    /// Calculates the cross-term component: w_cross[S(m₁)N(m₂) - S(m₂)N(m₁)]
    /// - Parameters:
    ///   - w_cross: Cross-term weight
    ///   - S_x: Symbolic output
    ///   - N_x: Neural output
    /// - Returns: Cross-term value
    func calculateCrossTerm(w_cross: Double, S_x: Double, N_x: Double) -> Double {
        return w_cross * (S_x * N_x - N_x * S_x)
    }
    
    /// Calculates the regularization component: exp(-[λ₁R_cognitive + λ₂R_efficiency])
    /// - Parameters:
    ///   - lambda1: λ₁(t) regularization weight
    ///   - lambda2: λ₂(t) regularization weight
    ///   - R_cognitive: Cognitive penalty
    ///   - R_efficiency: Efficiency penalty
    /// - Returns: Regularization factor
    func calculateRegularization(lambda1: Double, lambda2: Double, R_cognitive: Double, R_efficiency: Double) -> Double {
        return exp(-(lambda1 * R_cognitive + lambda2 * R_efficiency))
    }
    
    /// Calculates the complete Ψ(x) equation with all components
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
    func calculateCompletePsi(alpha: Double, lambda1: Double, lambda2: Double,
                            S_x: Double, N_x: Double, w_cross: Double,
                            R_cognitive: Double, R_efficiency: Double, P_H_E_beta: Double) -> Double {
        
        let hybridOutput = calculateHybridOutput(alpha: alpha, S_x: S_x, N_x: N_x)
        let crossTerm = calculateCrossTerm(w_cross: w_cross, S_x: S_x, N_x: N_x)
        let regularization = calculateRegularization(lambda1: lambda1, lambda2: lambda2,
                                                  R_cognitive: R_cognitive, R_efficiency: R_efficiency)
        
        return (hybridOutput + crossTerm) * regularization * P_H_E_beta
    }
    
    // MARK: - Trajectory Analysis
    
    /// Analyzes a specific point on the trajectory
    /// - Parameter t: Time parameter (0 to 1)
    /// - Returns: Detailed analysis of the trajectory point
    func analyzeTrajectoryPoint(t: Double) -> TrajectoryAnalysis {
        let point = calculateTrajectoryPoint(t: t)
        
        let hybridOutput = calculateHybridOutput(alpha: point.alpha, S_x: S_x, N_x: N_x)
        let crossTerm = calculateCrossTerm(w_cross: w_cross, S_x: S_x, N_x: N_x)
        let regularization = calculateRegularization(lambda1: point.lambda1, lambda2: point.lambda2,
                                                  R_cognitive: R_cognitive, R_efficiency: R_efficiency)
        
        return TrajectoryAnalysis(
            t: t,
            alpha: point.alpha,
            lambda1: point.lambda1,
            lambda2: point.lambda2,
            hybridOutput: hybridOutput,
            crossTerm: crossTerm,
            regularization: regularization,
            probability: P_H_E_beta,
            psi: point.Psi
        )
    }
    
    /// Calculates the integral of Ψ(x) over the trajectory
    /// - Parameter nPoints: Number of points for numerical integration
    /// - Returns: Integral value
    func calculateIntegral(nPoints: Int = 1000) -> Double {
        let trajectory = generateFullTrajectory(nPoints: nPoints)
        
        var integral: Double = 0.0
        let dt = 1.0 / Double(nPoints - 1)
        
        for i in 0..<trajectory.count {
            integral += trajectory[i].Psi * dt
        }
        
        return integral
    }
    
    /// Calculates the average Ψ(x) value over the trajectory
    /// - Parameter nPoints: Number of points for calculation
    /// - Returns: Average value
    func calculateAveragePsi(nPoints: Int = 100) -> Double {
        let trajectory = generateFullTrajectory(nPoints: nPoints)
        let sum = trajectory.reduce(0.0) { $0 + $1.Psi }
        return sum / Double(trajectory.count)
    }
    
    /// Finds the maximum Ψ(x) value and its corresponding time
    /// - Parameter nPoints: Number of points to check
    /// - Returns: Tuple of (time, maximum Ψ(x) value)
    func findMaximumPsi(nPoints: Int = 100) -> (t: Double, maxPsi: Double) {
        let trajectory = generateFullTrajectory(nPoints: nPoints)
        let maxPoint = trajectory.max { $0.Psi < $1.Psi }!
        return (t: maxPoint.t, maxPsi: maxPoint.Psi)
    }
    
    /// Finds the minimum Ψ(x) value and its corresponding time
    /// - Parameter nPoints: Number of points to check
    /// - Returns: Tuple of (time, minimum Ψ(x) value)
    func findMinimumPsi(nPoints: Int = 100) -> (t: Double, minPsi: Double) {
        let trajectory = generateFullTrajectory(nPoints: nPoints)
        let minPoint = trajectory.min { $0.Psi < $1.Psi }!
        return (t: minPoint.t, maxPsi: minPoint.Psi)
    }
}

// MARK: - Supporting Structures

struct TrajectoryAnalysis {
    let t: Double
    let alpha: Double
    let lambda1: Double
    let lambda2: Double
    let hybridOutput: Double
    let crossTerm: Double
    let regularization: Double
    let probability: Double
    let psi: Double
    
    func printAnalysis() {
        print("=== Trajectory Analysis at t = \(String(format: "%.3f", t)) ===")
        print("α(t) = \(String(format: "%.3f", alpha))")
        print("λ₁(t) = \(String(format: "%.3f", lambda1))")
        print("λ₂(t) = \(String(format: "%.3f", lambda2))")
        print("\nCore Equation Components:")
        print("Hybrid Output = \(String(format: "%.3f", hybridOutput))")
        print("Cross Term = \(String(format: "%.3f", crossTerm))")
        print("Regularization Factor = \(String(format: "%.3f", regularization))")
        print("Probability Term = \(String(format: "%.3f", probability))")
        print("\nΨ(x) = \(String(format: "%.3f", psi))")
    }
}

// MARK: - Extensions for Advanced Analysis

extension CoreEquation {
    
    /// Calculates the sensitivity of Ψ(x) to changes in α(t)
    /// - Parameters:
    ///   - t: Time parameter
    ///   - delta: Small change in alpha
    /// - Returns: Sensitivity value
    func calculateAlphaSensitivity(t: Double, delta: Double = 0.01) -> Double {
        let point1 = calculatePsi(alpha: t, lambda1: 2.0 * (1.0 - t), lambda2: 2.0 * (1.0 - t))
        let point2 = calculatePsi(alpha: t + delta, lambda1: 2.0 * (1.0 - t), lambda2: 2.0 * (1.0 - t))
        return (point2 - point1) / delta
    }
    
    /// Calculates the sensitivity of Ψ(x) to changes in λ₁(t)
    /// - Parameters:
    ///   - t: Time parameter
    ///   - delta: Small change in lambda1
    /// - Returns: Sensitivity value
    func calculateLambda1Sensitivity(t: Double, delta: Double = 0.01) -> Double {
        let lambda1 = 2.0 * (1.0 - t)
        let lambda2 = 2.0 * (1.0 - t)
        
        let point1 = calculatePsi(alpha: t, lambda1: lambda1, lambda2: lambda2)
        let point2 = calculatePsi(alpha: t, lambda1: lambda1 + delta, lambda2: lambda2)
        return (point2 - point1) / delta
    }
    
    /// Calculates the sensitivity of Ψ(x) to changes in λ₂(t)
    /// - Parameters:
    ///   - t: Time parameter
    ///   - delta: Small change in lambda2
    /// - Returns: Sensitivity value
    func calculateLambda2Sensitivity(t: Double, delta: Double = 0.01) -> Double {
        let lambda1 = 2.0 * (1.0 - t)
        let lambda2 = 2.0 * (1.0 - t)
        
        let point1 = calculatePsi(alpha: t, lambda1: lambda1, lambda2: lambda2)
        let point2 = calculatePsi(alpha: t, lambda1: lambda1, lambda2: lambda2 + delta)
        return (point2 - point1) / delta
    }
    
    /// Calculates the gradient of Ψ(x) with respect to all parameters
    /// - Parameter t: Time parameter
    /// - Returns: Gradient vector (dΨ/dα, dΨ/dλ₁, dΨ/dλ₂)
    func calculateGradient(t: Double) -> (dPsi_dAlpha: Double, dPsi_dLambda1: Double, dPsi_dLambda2: Double) {
        let dPsi_dAlpha = calculateAlphaSensitivity(t: t)
        let dPsi_dLambda1 = calculateLambda1Sensitivity(t: t)
        let dPsi_dLambda2 = calculateLambda2Sensitivity(t: t)
        
        return (dPsi_dAlpha: dPsi_dAlpha, dPsi_dLambda1: dPsi_dLambda1, dPsi_dLambda2: dPsi_dLambda2)
    }
}