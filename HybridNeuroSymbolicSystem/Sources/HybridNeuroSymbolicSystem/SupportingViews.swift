import SwiftUI

/// View for selecting preset configurations
public struct PresetPickerView: View {
    @ObservedObject var viewModel: LossViewModel
    @Environment(\.dismiss) private var dismiss
    
    public init(viewModel: LossViewModel) {
        self.viewModel = viewModel
    }
    
    public var body: some View {
        NavigationView {
            List {
                ForEach(LossPreset.allCases, id: \.self) { preset in
                    Button(action: {
                        viewModel.applyPreset(preset)
                        dismiss()
                    }) {
                        VStack(alignment: .leading, spacing: 8) {
                            Text(preset.rawValue)
                                .font(.headline)
                                .foregroundColor(.primary)
                            
                            Text(preset.description)
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                                .multilineTextAlignment(.leading)
                        }
                        .padding(.vertical, 4)
                    }
                }
            }
            .navigationTitle("Preset Configurations")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Cancel") {
                        dismiss()
                    }
                }
            }
        }
    }
}

/// View for recursive loss computation with nested operations
public struct RecursiveLossView: View {
    @ObservedObject var viewModel: LossViewModel
    @Environment(\.dismiss) private var dismiss
    
    @State private var operations: [Double] = [0.9, 0.85, 0.92]
    @State private var groundTruths: [Double] = [1.0, 0.9, 0.95]
    @State private var recursiveLoss: CompositeLoss?
    
    public init(viewModel: LossViewModel) {
        self.viewModel = viewModel
    }
    
    public var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 24) {
                    // Header
                    VStack(spacing: 12) {
                        Text("Recursive Loss Calculator")
                            .font(.title2)
                            .fontWeight(.semibold)
                        
                        Text("Compute composite loss for nested logical operations")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                            .multilineTextAlignment(.center)
                    }
                    .padding()
                    
                    // Operations Input
                    operationsInputSection
                    
                    // Ground Truths Input
                    groundTruthsInputSection
                    
                    // Results
                    if let recursiveLoss = recursiveLoss {
                        recursiveLossResultsSection(recursiveLoss)
                    }
                    
                    // Action Buttons
                    actionButtonsSection
                }
                .padding()
            }
            .navigationTitle("Recursive Loss")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
            .onAppear {
                calculateRecursiveLoss()
            }
        }
    }
    
    private var operationsInputSection: some View {
        VStack(spacing: 16) {
            Text("Nested Operations")
                .font(.headline)
                .frame(maxWidth: .infinity, alignment: .leading)
            
            VStack(spacing: 12) {
                ForEach(Array(operations.enumerated()), id: \.offset) { index, operation in
                    HStack {
                        Text("Level \(index + 1)")
                            .font(.subheadline)
                            .fontWeight(.medium)
                            .frame(width: 80, alignment: .leading)
                        
                        Slider(
                            value: Binding(
                                get: { operations[index] },
                                set: { newValue in
                                    operations[index] = newValue
                                    calculateRecursiveLoss()
                                }
                            ),
                            in: 0...1,
                            step: 0.05
                        )
                        .accentColor(.blue)
                        
                        Text(String(format: "%.2f", operation))
                            .font(.subheadline)
                            .fontWeight(.semibold)
                            .foregroundColor(.blue)
                            .frame(width: 50)
                    }
                }
            }
            
            HStack {
                Button("Add Level") {
                    operations.append(0.9)
                    groundTruths.append(1.0)
                    calculateRecursiveLoss()
                }
                .buttonStyle(.bordered)
                
                if operations.count > 1 {
                    Button("Remove Level") {
                        operations.removeLast()
                        groundTruths.removeLast()
                        calculateRecursiveLoss()
                    }
                    .buttonStyle(.bordered)
                    .tint(.red)
                }
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color(.systemBackground))
                .shadow(color: .black.opacity(0.1), radius: 4, x: 0, y: 2)
        )
    }
    
    private var groundTruthsInputSection: some View {
        VStack(spacing: 16) {
            Text("Ground Truths")
                .font(.headline)
                .frame(maxWidth: .infinity, alignment: .leading)
            
            VStack(spacing: 12) {
                ForEach(Array(groundTruths.enumerated()), id: \.offset) { index, groundTruth in
                    HStack {
                        Text("Level \(index + 1)")
                            .font(.subheadline)
                            .fontWeight(.medium)
                            .frame(width: 80, alignment: .leading)
                        
                        Slider(
                            value: Binding(
                                get: { groundTruths[index] },
                                set: { newValue in
                                    groundTruths[index] = newValue
                                    calculateRecursiveLoss()
                                }
                            ),
                            in: 0...1,
                            step: 0.05
                        )
                        .accentColor(.green)
                        
                        Text(String(format: "%.2f", groundTruth))
                            .font(.subheadline)
                            .fontWeight(.semibold)
                            .foregroundColor(.green)
                            .frame(width: 50)
                    }
                }
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color(.systemBackground))
                .shadow(color: .black.opacity(0.1), radius: 4, x: 0, y: 2)
        )
    }
    
    private func recursiveLossResultsSection(_ loss: CompositeLoss) -> some View {
        VStack(spacing: 16) {
            Text("Recursive Loss Results")
                .font(.headline)
                .frame(maxWidth: .infinity, alignment: .leading)
            
            VStack(spacing: 12) {
                // Total Loss
                LossMetricView(
                    title: "Total Recursive Loss",
                    value: loss.totalLoss,
                    color: .blue,
                    format: "%.4f"
                )
                
                // Individual Components
                HStack(spacing: 12) {
                    LossMetricView(
                        title: "Task Loss",
                        value: loss.taskLoss,
                        color: .green,
                        format: "%.4f"
                    )
                    
                    LossMetricView(
                        title: "Cognitive",
                        value: loss.cognitiveRegularizer,
                        color: .orange,
                        format: "%.4f"
                    )
                    
                    LossMetricView(
                        title: "Efficiency",
                        value: loss.efficiencyRegularizer,
                        color: .purple,
                        format: "%.4f"
                    )
                }
            }
            
            // Depth Analysis
            VStack(spacing: 8) {
                Text("Depth Analysis")
                    .font(.subheadline)
                    .fontWeight(.medium)
                
                Text("Operations: \(operations.count) levels")
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                Text("Each level contributes with decreasing weight based on depth")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .italic()
                    .multilineTextAlignment(.center)
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color(.systemBackground))
                .shadow(color: .black.opacity(0.1), radius: 4, x: 0, y: 2)
        )
    }
    
    private var actionButtonsSection: some View {
        VStack(spacing: 12) {
            Button("Calculate Loss") {
                calculateRecursiveLoss()
            }
            .buttonStyle(.borderedProminent)
            .tint(.blue)
            
            Text("💡 The recursive loss applies depth weighting to handle nested logical operations")
                .font(.caption)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal)
        }
    }
    
    private func calculateRecursiveLoss() {
        recursiveLoss = viewModel.computeRecursiveLoss(
            operations: operations,
            groundTruths: groundTruths
        )
    }
}

/// View for displaying detailed loss breakdown
public struct LossBreakdownView: View {
    let compositeLoss: CompositeLoss
    
    public init(compositeLoss: CompositeLoss) {
        self.compositeLoss = compositeLoss
    }
    
    public var body: some View {
        VStack(spacing: 16) {
            Text("Loss Breakdown")
                .font(.headline)
                .frame(maxWidth: .infinity, alignment: .leading)
            
            VStack(spacing: 12) {
                // Formula Display
                VStack(spacing: 8) {
                    Text("Composite Loss = L_logic + λ₁·R_cog + λ₂·R_eff")
                        .font(.subheadline)
                        .fontWeight(.medium)
                        .foregroundColor(.primary)
                    
                    Text("= \(String(format: "%.4f", compositeLoss.taskLoss)) + \(String(format: "%.2f", compositeLoss.lambda1))·\(String(format: "%.4f", compositeLoss.cognitiveRegularizer)) + \(String(format: "%.2f", compositeLoss.lambda2))·\(String(format: "%.4f", compositeLoss.efficiencyRegularizer))")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    Text("= \(String(format: "%.4f", compositeLoss.totalLoss))")
                        .font(.subheadline)
                        .fontWeight(.semibold)
                        .foregroundColor(.blue)
                }
                .padding()
                .background(
                    RoundedRectangle(cornerRadius: 8)
                        .fill(Color(.systemGray6))
                )
                
                // Component Details
                VStack(spacing: 8) {
                    LossComponentRow(
                        title: "Task Loss (L_logic)",
                        value: compositeLoss.taskLoss,
                        description: "Accuracy against ground truth",
                        color: .green
                    )
                    
                    LossComponentRow(
                        title: "Cognitive Regularizer (R_cog)",
                        value: compositeLoss.cognitiveRegularizer,
                        description: "Deviation from human-like patterns",
                        color: .orange
                    )
                    
                    LossComponentRow(
                        title: "Efficiency Regularizer (R_eff)",
                        value: compositeLoss.efficiencyRegularizer,
                        description: "Computational complexity penalty",
                        color: .purple
                    )
                }
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color(.systemBackground))
                .shadow(color: .black.opacity(0.1), radius: 4, x: 0, y: 2)
        )
    }
}

/// Row view for loss component details
struct LossComponentRow: View {
    let title: String
    let value: Double
    let description: String
    let color: Color
    
    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.subheadline)
                    .fontWeight(.medium)
                
                Text(description)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
            
            Text(String(format: "%.4f", value))
                .font(.subheadline)
                .fontWeight(.semibold)
                .foregroundColor(color)
        }
        .padding(.vertical, 4)
    }
}

// MARK: - Preview

struct SupportingViews_Previews: PreviewProvider {
    static var previews: some View {
        VStack(spacing: 20) {
            PresetPickerView(viewModel: LossViewModel())
            RecursiveLossView(viewModel: LossViewModel())
        }
    }
}