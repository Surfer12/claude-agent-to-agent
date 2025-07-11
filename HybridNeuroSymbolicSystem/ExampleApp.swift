import SwiftUI
import HybridNeuroSymbolicSystem

/// Example app demonstrating the composite loss system
@main
struct ExampleApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}

/// Main content view for the example app
struct ContentView: View {
    var body: some View {
        TabView {
            // Main Composite Loss Calculator
            CompositeLossView()
                .tabItem {
                    Image(systemName: "function")
                    Text("Loss Calculator")
                }
            
            // Educational Demo
            EducationalDemoView()
                .tabItem {
                    Image(systemName: "graduationcap")
                    Text("Educational Demo")
                }
            
            // Advanced Features
            AdvancedFeaturesView()
                .tabItem {
                    Image(systemName: "gearshape")
                    Text("Advanced")
                }
        }
    }
}

/// Educational demonstration view
struct EducationalDemoView: View {
    @StateObject private var viewModel = LossViewModel()
    @State private var currentStep = 0
    
    private let educationalSteps = [
        "Welcome to the Composite Loss System! This system balances accuracy, human-like reasoning, and computational efficiency.",
        "The composite loss formula is: L_total = L_logic + λ₁·R_cog + λ₂·R_eff",
        "L_logic measures how well our output matches the expected result (ground truth).",
        "R_cog (cognitive regularizer) penalizes deviations from human-like reasoning patterns.",
        "R_eff (efficiency regularizer) encourages computational simplicity.",
        "λ₁ and λ₂ are weights that control the balance between these objectives.",
        "Try adjusting the sliders to see how parameters affect the loss components!",
        "Notice how increasing λ₁ makes the system prioritize human-like reasoning.",
        "Increasing λ₂ emphasizes computational efficiency over other factors.",
        "The system provides educational insights and Socratic questions to guide your exploration."
    ]
    
    var body: some View {
        NavigationView {
            VStack(spacing: 24) {
                // Educational Content
                VStack(spacing: 16) {
                    Text("Educational Demo")
                        .font(.title)
                        .fontWeight(.bold)
                    
                    ScrollView {
                        VStack(spacing: 16) {
                            // Current Step
                            VStack(spacing: 12) {
                                Text("Step \(currentStep + 1) of \(educationalSteps.count)")
                                    .font(.headline)
                                    .foregroundColor(.blue)
                                
                                Text(educationalSteps[currentStep])
                                    .font(.body)
                                    .multilineTextAlignment(.center)
                                    .padding()
                                    .background(
                                        RoundedRectangle(cornerRadius: 12)
                                            .fill(Color(.systemGray6))
                                    )
                            }
                            
                            // Navigation
                            HStack(spacing: 16) {
                                Button("Previous") {
                                    if currentStep > 0 {
                                        currentStep -= 1
                                    }
                                }
                                .disabled(currentStep == 0)
                                .buttonStyle(.bordered)
                                
                                Button("Next") {
                                    if currentStep < educationalSteps.count - 1 {
                                        currentStep += 1
                                    }
                                }
                                .disabled(currentStep == educationalSteps.count - 1)
                                .buttonStyle(.borderedProminent)
                            }
                            
                            // Progress
                            ProgressView(value: Double(currentStep + 1), total: Double(educationalSteps.count))
                                .progressViewStyle(LinearProgressViewStyle())
                        }
                    }
                }
                
                // Interactive Demo
                VStack(spacing: 16) {
                    Text("Interactive Demo")
                        .font(.headline)
                    
                    // Quick parameter adjustment
                    VStack(spacing: 12) {
                        HStack {
                            Text("λ₁ (Cognitive): \(viewModel.lambda1, specifier: "%.2f")")
                            Slider(value: $viewModel.lambda1, in: 0...1, step: 0.1)
                        }
                        
                        HStack {
                            Text("λ₂ (Efficiency): \(viewModel.lambda2, specifier: "%.2f")")
                            Slider(value: $viewModel.lambda2, in: 0...1, step: 0.1)
                        }
                    }
                    
                    if let loss = viewModel.compositeLoss {
                        VStack(spacing: 8) {
                            Text("Current Loss: \(loss.totalLoss, specifier: "%.4f")")
                                .font(.headline)
                                .foregroundColor(.blue)
                            
                            HStack(spacing: 16) {
                                VStack {
                                    Text("Task")
                                        .font(.caption)
                                    Text("\(loss.taskLoss, specifier: "%.3f")")
                                        .font(.subheadline)
                                        .foregroundColor(.green)
                                }
                                
                                VStack {
                                    Text("Cognitive")
                                        .font(.caption)
                                    Text("\(loss.cognitiveRegularizer, specifier: "%.3f")")
                                        .font(.subheadline)
                                        .foregroundColor(.orange)
                                }
                                
                                VStack {
                                    Text("Efficiency")
                                        .font(.caption)
                                    Text("\(loss.efficiencyRegularizer, specifier: "%.3f")")
                                        .font(.subheadline)
                                        .foregroundColor(.purple)
                                }
                            }
                        }
                        .padding()
                        .background(
                            RoundedRectangle(cornerRadius: 8)
                                .fill(Color(.systemGray6))
                        )
                    }
                }
            }
            .padding()
            .navigationTitle("Educational Demo")
        }
    }
}

/// Advanced features demonstration
struct AdvancedFeaturesView: View {
    @StateObject private var viewModel = LossViewModel()
    @State private var showingRecursiveDemo = false
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 24) {
                    // Preset Demonstrations
                    VStack(spacing: 16) {
                        Text("Preset Configurations")
                            .font(.headline)
                        
                        LazyVGrid(columns: [
                            GridItem(.flexible()),
                            GridItem(.flexible())
                        ], spacing: 12) {
                            ForEach(LossPreset.allCases, id: \.self) { preset in
                                Button(action: {
                                    viewModel.applyPreset(preset)
                                }) {
                                    VStack(spacing: 8) {
                                        Text(preset.rawValue)
                                            .font(.subheadline)
                                            .fontWeight(.medium)
                                        
                                        Text(preset.description)
                                            .font(.caption)
                                            .foregroundColor(.secondary)
                                            .multilineTextAlignment(.center)
                                    }
                                    .padding()
                                    .frame(maxWidth: .infinity)
                                    .background(
                                        RoundedRectangle(cornerRadius: 8)
                                            .fill(Color(.systemGray6))
                                    )
                                }
                                .buttonStyle(PlainButtonStyle())
                            }
                        }
                    }
                    
                    // Focus Session Demo
                    VStack(spacing: 16) {
                        Text("Focus Session Management")
                            .font(.headline)
                        
                        if viewModel.isInFocusSession {
                            VStack(spacing: 12) {
                                Text("Focus Session Active")
                                    .font(.subheadline)
                                    .foregroundColor(.blue)
                                
                                Text("Time Remaining: \(viewModel.focusSessionTimeRemaining)")
                                    .font(.title2)
                                    .fontWeight(.semibold)
                                
                                ProgressView(value: viewModel.focusSessionProgress)
                                    .progressViewStyle(LinearProgressViewStyle())
                                
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
                            }
                        }
                    }
                    .padding()
                    .background(
                        RoundedRectangle(cornerRadius: 12)
                            .fill(Color(.systemBackground))
                            .shadow(color: .black.opacity(0.1), radius: 4, x: 0, y: 2)
                    )
                    
                    // Recursive Loss Demo
                    VStack(spacing: 16) {
                        Text("Recursive Loss Computation")
                            .font(.headline)
                        
                        Text("Compute composite loss for nested logical operations with depth weighting")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                            .multilineTextAlignment(.center)
                        
                        Button("Open Recursive Calculator") {
                            showingRecursiveDemo = true
                        }
                        .buttonStyle(.borderedProminent)
                    }
                    .padding()
                    .background(
                        RoundedRectangle(cornerRadius: 12)
                            .fill(Color(.systemBackground))
                            .shadow(color: .black.opacity(0.1), radius: 4, x: 0, y: 2)
                    )
                    
                    // Educational Insights
                    if !viewModel.educationalPrompts.isEmpty {
                        VStack(spacing: 16) {
                            Text("Socratic Questions")
                                .font(.headline)
                            
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
                        .padding()
                        .background(
                            RoundedRectangle(cornerRadius: 12)
                                .fill(Color(.systemBackground))
                                .shadow(color: .black.opacity(0.1), radius: 4, x: 0, y: 2)
                        )
                    }
                }
                .padding()
            }
            .navigationTitle("Advanced Features")
            .sheet(isPresented: $showingRecursiveDemo) {
                RecursiveLossView(viewModel: viewModel)
            }
        }
    }
}

// MARK: - Preview

struct ExampleApp_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}