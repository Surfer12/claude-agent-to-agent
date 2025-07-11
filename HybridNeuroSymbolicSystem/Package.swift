// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "HybridNeuroSymbolicSystem",
    platforms: [
        .iOS(.v16),
        .macOS(.v13)
    ],
    products: [
        .library(
            name: "HybridNeuroSymbolicSystem",
            targets: ["HybridNeuroSymbolicSystem"]),
    ],
    dependencies: [
        // Add dependencies as needed
    ],
    targets: [
        .target(
            name: "HybridNeuroSymbolicSystem",
            dependencies: []),
        .testTarget(
            name: "HybridNeuroSymbolicSystemTests",
            dependencies: ["HybridNeuroSymbolicSystem"]),
    ]
)