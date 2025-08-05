package examples;

import com.anthropic.api.processors.SwiftSwarmMathematicalProof;
import com.anthropic.api.processors.SwiftSwarmMathematicalProof.CEPMParameters;
import com.anthropic.api.processors.SwiftSwarmMathematicalProof.ProofResult;

import java.util.concurrent.CompletableFuture;

/**
 * Simple example demonstrating the Swift Swarm Mathematical Proof implementation
 * based on the mathematical content from swift_swarm_witten file.
 * 
 * This demonstrates the core CEPM equation computation with the 9-step consciousness framework.
 * 
 * @author Ryan Oates
 * @version 1.0
 */
public class SimpleSwiftSwarmExample {
    
    public static void main(String[] args) {
        System.out.println("=== Swift Swarm Mathematical Proof Implementation ===");
        System.out.println("Based on swift_swarm_witten mathematical content");
        System.out.println("Implementing CEPM equation with 9-Step Consciousness Framework");
        System.out.println();
        
        try {
            // Example 1: Basic CEMP computation matching swift_swarm_witten examples
            runBasicCEMPExample();
            
            // Example 2: Full proof with 9-step framework execution
            runFullProofExample();
            
            // Example 3: Multiple time steps (matching the multi-step example from swift_swarm_witten)
            runMultiStepExample();
            
            System.out.println("All Swift Swarm examples completed successfully!");
            System.out.println("Mathematical proof system fully operational.");
            
        } catch (Exception e) {
            System.err.println("Example execution failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * Example 1: Basic CEMP computation matching swift_swarm_witten numerical example
     */
    private static void runBasicCEMPExample() {
        System.out.println("--- Example 1: Basic CEMP Computation ---");
        System.out.println("Implementing the numerical example from swift_swarm_witten:");
        System.out.println("Single time step (t=1) with specific parameter values");
        System.out.println();
        
        // Parameters from swift_swarm_witten numerical example:
        // x = 0.7, α(1) = 0.6, S(0.7) = 0.8, N(0.7) = 0.3
        // λ₁ = 0.2, λ₂ = 0.3, R_cognitive(1) = 0.5, R_efficiency(1) = 0.4, P(H|E,β(1)) = 0.7
        CEPMParameters parameters = new CEPMParameters(0.2, 0.3, 0.8, 0.3);
        SwiftSwarmMathematicalProof proof = new SwiftSwarmMathematicalProof(0.7, parameters);
        
        // Add single time step matching the swift_swarm_witten example
        proof.addTimeStep(1.0, 0.6, 0.5, 0.4, 0.7);
        
        // Compute CEMP prediction
        double prediction = proof.computeCEPMPrediction();
        
        System.out.println("Expected result from swift_swarm_witten: ≈ 0.3371");
        System.out.println("Computed CEMP Prediction: " + prediction);
        System.out.println("Framework Integrity: " + proof.validateFrameworkIntegrity());
        System.out.println("Attribution: " + "9-Step Consciousness Framework by Ryan David Oates");
        System.out.println();
    }
    
    /**
     * Example 2: Full proof with asynchronous execution
     */
    private static void runFullProofExample() throws Exception {
        System.out.println("--- Example 2: Full Proof with 9-Step Framework ---");
        System.out.println("Executing complete mathematical proof with consciousness framework");
        System.out.println();
        
        // Create CEMP parameters for full proof
        CEPMParameters parameters = new CEPMParameters(0.2, 0.3, 0.8, 0.3);
        SwiftSwarmMathematicalProof proof = new SwiftSwarmMathematicalProof(0.7, parameters);
        
        // Add single time step
        proof.addTimeStep(1.0, 0.6, 0.5, 0.4, 0.7);
        
        // Execute full proof asynchronously (includes 9-step framework)
        CompletableFuture<ProofResult> futureResult = proof.executeProofAsync();
        ProofResult proofResult = futureResult.get();
        
        System.out.println("Proof Execution Success: " + proofResult.isSuccess());
        System.out.println("9-Step Framework Result: " + proofResult.getResult());
        System.out.println("Execution Timestamp: " + proofResult.getTimestamp());
        System.out.println("Framework Attribution: " + proofResult.getAttribution());
        System.out.println("License: " + proofResult.getLicense());
        System.out.println();
    }
    
    /**
     * Example 3: Multiple time steps matching swift_swarm_witten multi-step example
     */
    private static void runMultiStepExample() {
        System.out.println("--- Example 3: Multiple Time Steps (T=3) ---");
        System.out.println("Implementing the extended numerical example from swift_swarm_witten:");
        System.out.println("Three time steps with evolving parameters");
        System.out.println();
        
        // Parameters from swift_swarm_witten multi-step example
        CEPMParameters parameters = new CEPMParameters(0.2, 0.3, 0.8, 0.3);
        SwiftSwarmMathematicalProof proof = new SwiftSwarmMathematicalProof(0.7, parameters);
        
        // Add three time steps matching the swift_swarm_witten example
        proof.addTimeStep(1.0, 0.6, 0.5, 0.4, 0.7);     // Expected: ≈ 0.3371
        proof.addTimeStep(2.0, 0.65, 0.4, 0.35, 0.75);  // Expected: ≈ 0.3896  
        proof.addTimeStep(3.0, 0.7, 0.3, 0.3, 0.8);     // Expected: ≈ 0.4476
        
        // Compute integrated prediction
        double prediction = proof.computeCEPMPrediction();
        
        System.out.println("Expected total from swift_swarm_witten: ≈ 1.174");
        System.out.println("Computed Multi-Step Prediction: " + prediction);
        System.out.println("Individual time step contributions:");
        System.out.println("  t=1: Expected ≈ 0.3371");
        System.out.println("  t=2: Expected ≈ 0.3896");
        System.out.println("  t=3: Expected ≈ 0.4476");
        System.out.println("  Total: Expected ≈ 1.174");
        System.out.println();
        
        // Display framework report
        String report = proof.generateProofReport();
        System.out.println("Framework Report:");
        System.out.println(report);
    }
}