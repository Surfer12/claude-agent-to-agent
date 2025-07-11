import Foundation

// ... existing code ...

func computeRecursiveLoss(operations: [Operation], groundTruths: [GroundTruth]) -> Loss {
    // Validate that operations and groundTruths arrays have the same length
    guard operations.count == groundTruths.count else {
        fatalError("computeRecursiveLoss: operations count (\(operations.count)) must match groundTruths count (\(groundTruths.count))")
    }
    
    // Validate that arrays are not empty
    guard !operations.isEmpty else {
        fatalError("computeRecursiveLoss: operations array cannot be empty")
    }
    
    guard !groundTruths.isEmpty else {
        fatalError("computeRecursiveLoss: groundTruths array cannot be empty")
    }
    
    // Now we can safely access groundTruths[0] since we've validated both arrays
    // have the same length and are non-empty
    let firstOperation = operations[0]
    let firstGroundTruth = groundTruths[0]
    
    // Continue with the rest of the function logic...
    // This is where the original function would continue with its implementation
    
    // Placeholder return - replace with actual loss computation
    return Loss(value: 0.0)
}

// ... existing code ...