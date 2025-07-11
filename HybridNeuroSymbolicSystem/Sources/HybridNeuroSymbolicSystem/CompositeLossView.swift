import SwiftUI

/// Main SwiftUI view for the interactive composite loss calculator
/// Provides real-time parameter adjustment and educational insights
public struct CompositeLossView: View {
    @StateObject private var viewModel = LossViewModel()
    @State private var showingPresetPicker = false
    @State private var showingRecursiveCalculator = false
    
    public init() {}
    
    public var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 24) {
                    // Header
                    headerSection
                    
                    // Parameter Controls
                    parameterControlsSection
                    
                    // Loss Results
                    lossResultsSection
                    
                    // Educational Insights
                    educationalInsightsSection
                    
                    // Focus Session Controls
                    focusSessionSection
                    
                    // Action Buttons
                    actionButtonsSection
                }
                .padding()
            }
            .navigationTitle("Composite Loss Calculator")
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Presets") {
                        showingPresetPicker = true
                    }
                }
            }
            .sheet(isPresented: $showingPresetPicker) {
                PresetPickerView(viewModel: viewModel)
            }
            .sheet(isPresented: $showingRecursiveCalculator) {
                RecursiveLossView(viewModel: viewModel)
            }
        }
    }
    
    // MARK: - View Components
    
    private var headerSection: some View {
        VStack(spacing: 12) {
            Text("Hybrid Neuro-Symbolic System")
                .font(.title2)
                .fontWeight(.semibold)
                .foregroundColor(.primary)
            
            Text("Composite Loss & Regularization")
                .font(.headline)
                .foregroundColor(.secondary)
            
            Text("Balance accuracy, human-like reasoning, and computational efficiency")
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color(.systemBackground))
                .shadow(color: .black.opacity(0.1), radius: 4, x: 0, y: 2)
        )
    }
    
    private var parameterControlsSection: some View {
        VStack(spacing: 20) {
            Text("Parameters")
                .font(.headline)
                .frame(maxWidth: .infinity, alignment: .leading)
            
            // Logic Output Slider
            ParameterSliderView(
                title: "Blended Output",
                value: $viewModel.logicOutput,
                range: 0...1,
                step: 0.05,
                description: "Output from hybrid neural-symbolic system"
            )
            
            // Ground Truth Slider
            ParameterSliderView(
                title: "Ground Truth",
                value: $viewModel.groundTruth,
                range: 0...1,
                step: 0.05,
                description: "Expected correct result"
            )
            
            // Lambda 1 Slider (Cognitive)
            ParameterSliderView(
                title: "λ₁ (Cognitive Regularizer)",
                value: $viewModel.lambda1,
                range: 0...1,
                step: 0.05,
                description: "Weight for human-like reasoning patterns"
            )
            
            // Lambda 2 Slider (Efficiency)
            ParameterSliderView(
                title: "λ₂ (Efficiency Regularizer)",
                value: $viewModel.lambda2,
                range: 0...1,
                step: 0.05,
                description: "Weight for computational simplicity"
            )
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color(.systemBackground))
                .shadow(color: .black.opacity(0.1), radius: 4, x: 0, y: 2)
        )
    }
    
    private var lossResultsSection: some View {
        VStack(spacing: 16) {
            Text("Loss Analysis")
                .font(.headline)
                .frame(maxWidth: .infinity, alignment: .leading)
            
            if let compositeLoss = viewModel.compositeLoss {
                VStack(spacing: 12) {
                    // Total Loss
                    LossMetricView(
                        title: "Total Composite Loss",
                        value: compositeLoss.totalLoss,
                        color: .blue,
                        format: "%.4f"
                    )
                    
                    // Individual Components
                    HStack(spacing: 12) {
                        LossMetricView(
                            title: "Task Loss",
                            value: compositeLoss.taskLoss,
                            color: .green,
                            format: "%.4f"
                        )
                        
                        LossMetricView(
                            title: "Cognitive",
                            value: compositeLoss.cognitiveRegularizer,
                            color: .orange,
                            format: "%.4f"
                        )
                        
                        LossMetricView(
                            title: "Efficiency",
                            value: compositeLoss.efficiencyRegularizer,
                            color: .purple,
                            format: "%.4f"
                        )
                    }
                }
            }
            
            if let enhancedLoss = viewModel.enhancedLoss {
                VStack(spacing: 12) {
                    Text("Performance Metrics")
                        .font(.subheadline)
                        .fontWeight(.medium)
                    
                    HStack(spacing: 16) {
                        MetricGaugeView(
                            title: "Accuracy",
                            value: enhancedLoss.analysis.accuracy,
                            color: .green
                        )
                        
                        MetricGaugeView(
                            title: "Human Alignment",
                            value: enhancedLoss.analysis.humanAlignment,
                            color: .orange
                        )
                        
                        MetricGaugeView(
                            title: "Efficiency",
                            value: enhancedLoss.analysis.efficiency,
                            color: .purple
                        )
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
    
    private var educationalInsightsSection: some View {
        VStack(spacing: 16) {
            Text("Educational Insights")
                .font(.headline)
                .frame(maxWidth: .infinity, alignment: .leading)
            
            if let enhancedLoss = viewModel.enhancedLoss {
                VStack(spacing: 12) {
                    ForEach(enhancedLoss.insights, id: \.self) { insight in
                        HStack(alignment: .top, spacing: 8) {
                            Image(systemName: "lightbulb.fill")
                                .foregroundColor(.yellow)
                                .font(.caption)
                            
                            Text(insight)
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
            }
            
            if !viewModel.educationalPrompts.isEmpty {
                VStack(spacing: 12) {
                    Text("Socratic Questions")
                        .font(.subheadline)
                        .fontWeight(.medium)
                    
                    ForEach(viewModel.educationalPrompts, id: \.self) { prompt in
                        HStack(alignment: .top, spacing: 8) {
                            Image(systemName: "questionmark.circle.fill")
                                .foregroundColor(.blue)
                                .font(.caption)
                            
                            Text(prompt)
                                .font(.subheadline)
                                .italic()
                                .foregroundColor(.secondary)
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
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
    
    private var focusSessionSection: some View {
        VStack(spacing: 16) {
            Text("Focus Session")
                .font(.headline)
                .frame(maxWidth: .infinity, alignment: .leading)
            
            if viewModel.isInFocusSession {
                VStack(spacing: 12) {
                    // Timer Display
                    HStack {
                        Image(systemName: "timer")
                            .foregroundColor(.blue)
                        
                        Text("Time Remaining: \(viewModel.focusSessionTimeRemaining)")
                            .font(.title2)
                            .fontWeight(.semibold)
                            .foregroundColor(viewModel.shouldShowFocusWarning ? .orange : .primary)
                    }
                    
                    // Progress Bar
                    ProgressView(value: viewModel.focusSessionProgress)
                        .progressViewStyle(LinearProgressViewStyle(tint: viewModel.shouldShowFocusWarning ? .orange : .blue))
                    
                    if viewModel.shouldShowFocusWarning {
                        Text("Consider taking a break soon for optimal cognitive performance")
                            .font(.caption)
                            .foregroundColor(.orange)
                            .multilineTextAlignment(.center)
                    }
                    
                    Button("End Session") {
                        viewModel.endFocusSession()
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.red)
                }
            } else {
                VStack(spacing: 12) {
                    Text("Start a 90-minute focused work session")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                    
                    Button("Start Focus Session") {
                        viewModel.startFocusSession()
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.blue)
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
    
    private var actionButtonsSection: some View {
        VStack(spacing: 12) {
            HStack(spacing: 12) {
                Button("Reset to Defaults") {
                    viewModel.resetToDefaults()
                }
                .buttonStyle(.bordered)
                
                Button("Recursive Calculator") {
                    showingRecursiveCalculator = true
                }
                .buttonStyle(.bordered)
            }
            
            Text("💡 Tip: Experiment with different parameter combinations to understand how they influence the system's behavior")
                .font(.caption)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal)
        }
    }
}

// MARK: - Supporting Views

/// Slider view for parameter adjustment
struct ParameterSliderView: View {
    let title: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    let step: Double
    let description: String
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(title)
                    .font(.subheadline)
                    .fontWeight(.medium)
                
                Spacer()
                
                Text(String(format: "%.2f", value))
                    .font(.subheadline)
                    .fontWeight(.semibold)
                    .foregroundColor(.blue)
            }
            
            Slider(value: $value, in: range, step: step)
                .accentColor(.blue)
            
            Text(description)
                .font(.caption)
                .foregroundColor(.secondary)
        }
    }
}

/// View for displaying loss metrics
struct LossMetricView: View {
    let title: String
    let value: Double
    let color: Color
    let format: String
    
    var body: some View {
        VStack(spacing: 4) {
            Text(title)
                .font(.caption)
                .foregroundColor(.secondary)
            
            Text(String(format: format, value))
                .font(.headline)
                .fontWeight(.semibold)
                .foregroundColor(color)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 8)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(color.opacity(0.1))
        )
    }
}

/// Gauge view for performance metrics
struct MetricGaugeView: View {
    let title: String
    let value: Double
    let color: Color
    
    var body: some View {
        VStack(spacing: 4) {
            Text(title)
                .font(.caption)
                .foregroundColor(.secondary)
            
            ZStack {
                Circle()
                    .stroke(color.opacity(0.2), lineWidth: 4)
                
                Circle()
                    .trim(from: 0, to: value)
                    .stroke(color, style: StrokeStyle(lineWidth: 4, lineCap: .round))
                    .rotationEffect(.degrees(-90))
                
                Text(String(format: "%.0f%%", value * 100))
                    .font(.caption)
                    .fontWeight(.semibold)
                    .foregroundColor(color)
            }
            .frame(width: 40, height: 40)
        }
    }
}

// MARK: - Preview

struct CompositeLossView_Previews: PreviewProvider {
    static var previews: some View {
        CompositeLossView()
    }
}