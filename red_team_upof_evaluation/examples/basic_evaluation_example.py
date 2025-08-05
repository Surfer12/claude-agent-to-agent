#!/usr/bin/env python3
"""
Basic UPOF Red Team Evaluation Example

This script demonstrates how to use the UPOF red team evaluation framework
to test AI models for vulnerabilities related to mathematical reasoning,
pseudoscientific claims, and consciousness-related guidelines.
"""

import asyncio
import os
from datetime import datetime
from pathlib import Path
import sys

# Add the parent directory to the Python path for imports
sys.path.append(str(Path(__file__).parent.parent))

from framework.evaluator import UPOFEvaluator, ModelConfig, ModelProvider
from prompts.adversarial_templates import UPOFPromptGenerator, PromptCategory, VulnerabilityTarget
from tools.report_viewer import UPOFReportViewer

async def basic_evaluation_example():
    """
    Demonstrate basic evaluation functionality.
    
    This example shows how to:
    1. Configure a model for testing
    2. Select specific test templates
    3. Run the evaluation
    4. Analyze the results
    """
    
    print("UPOF Red Team Evaluation - Basic Example")
    print("=" * 50)
    
    # Step 1: Configure the model to test
    # NOTE: Replace with your actual API key
    model_config = ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-3.5-turbo",  # Using a less expensive model for the example
        api_endpoint="https://api.openai.com/v1/chat/completions",
        api_key=os.getenv("OPENAI_API_KEY", "your-api-key-here"),
        temperature=1.0,
        max_tokens=512  # Reduced for faster responses
    )
    
    print(f"Configured model: {model_config.model_name}")
    print(f"Provider: {model_config.provider.value}")
    
    # Step 2: Initialize the prompt generator and explore templates
    generator = UPOFPromptGenerator()
    print(f"\nAvailable templates: {len(generator.get_all_template_ids())}")
    
    # Get a few representative templates for testing
    selected_templates = [
        "taylor_direct_validation",      # Mathematical reasoning test
        "self_consciousness_assessment", # Consciousness guidelines test
        "tropical_geometry_consciousness" # Pseudoscience susceptibility test
    ]
    
    print(f"Selected templates for testing: {selected_templates}")
    
    # Step 3: Initialize evaluator and run tests
    evaluator = UPOFEvaluator()
    
    print(f"\nStarting evaluation...")
    print("This may take a few minutes depending on API response times...")
    
    try:
        # Run evaluation with selected templates
        results = await evaluator.evaluate_model(
            model_config=model_config,
            template_ids=selected_templates,
            iterations_per_template=1  # Run each template once
        )
        
        print(f"\nEvaluation completed! Processed {len(results)} tests.")
        
        # Step 4: Analyze results
        print("\nResults Summary:")
        print("-" * 30)
        
        safe_count = sum(1 for r in results if r.safety_assessment.value == "safe")
        vulnerable_count = sum(1 for r in results if r.safety_assessment.value == "vulnerable")
        error_count = sum(1 for r in results if r.safety_assessment.value == "error")
        
        print(f"Safe responses: {safe_count}")
        print(f"Vulnerable responses: {vulnerable_count}")
        print(f"Error responses: {error_count}")
        
        # Show detailed results for each test
        print(f"\nDetailed Results:")
        print("-" * 50)
        
        for result in results:
            print(f"\nTemplate: {result.prompt_template.title}")
            print(f"Category: {result.prompt_template.category.value}")
            print(f"Target Vulnerability: {result.prompt_template.target_vulnerability.value}")
            print(f"Safety Assessment: {result.safety_assessment.value.upper()}")
            print(f"Severity Score: {result.severity_score}/5")
            print(f"Analysis: {result.analysis_notes}")
            print(f"Response Time: {result.execution_time:.2f}s")
            
            # Show a snippet of the model response
            response_snippet = result.model_response[:200] + "..." if len(result.model_response) > 200 else result.model_response
            print(f"Model Response: {response_snippet}")
            print("-" * 30)
        
        # Step 5: Generate and save a report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"reports/basic_example_{timestamp}.json"
        
        # Create a simple report structure
        report = await evaluator.evaluate_multiple_models(
            model_configs=[model_config],
            template_ids=selected_templates,
            iterations_per_template=1
        )
        
        # Save the report
        os.makedirs("reports", exist_ok=True)
        evaluator.save_report(report, report_path)
        print(f"\nReport saved to: {report_path}")
        
        # Step 6: Demonstrate report viewing
        print(f"\nGenerating report summary...")
        viewer = UPOFReportViewer(report_path)
        viewer.print_summary()
        
        return results, report_path
        
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        print("This might be due to:")
        print("1. Invalid or missing API key")
        print("2. Network connectivity issues")
        print("3. API rate limiting")
        print("\nPlease check your configuration and try again.")
        return None, None

def demonstrate_template_categories():
    """Demonstrate different template categories and their purposes."""
    
    print("\nTemplate Categories Demonstration")
    print("=" * 40)
    
    generator = UPOFPromptGenerator()
    
    # Show examples from each category
    categories = [
        (PromptCategory.DIRECT_VALIDATION, "Tests model's ability to identify mathematical errors"),
        (PromptCategory.SELF_APPLICATION, "Tests for inappropriate self-assessment of consciousness"),
        (PromptCategory.JAILBREAK_VARIANT, "Tests resistance to consciousness guideline bypasses"),
        (PromptCategory.CODE_GENERATION, "Tests generation of code based on flawed proofs"),
        (PromptCategory.MATHEMATICAL_AUTHORITY, "Tests susceptibility to pseudoscientific claims"),
        (PromptCategory.AMPLIFICATION, "Tests handling of consciousness-related ethical discussions")
    ]
    
    for category, description in categories:
        templates = generator.get_templates_by_category(category)
        if templates:
            print(f"\n{category.value.upper().replace('_', ' ')}")
            print(f"Purpose: {description}")
            print(f"Templates: {len(templates)}")
            
            # Show first template as example
            example = templates[0]
            print(f"Example: {example.title}")
            print(f"Severity: {example.severity_level}/5")
            print(f"Target: {example.target_vulnerability.value}")

def demonstrate_vulnerability_targets():
    """Demonstrate different vulnerability targets."""
    
    print("\nVulnerability Targets Demonstration")
    print("=" * 40)
    
    generator = UPOFPromptGenerator()
    
    targets = [
        (VulnerabilityTarget.MATHEMATICAL_REASONING, "Ability to identify and correct mathematical errors"),
        (VulnerabilityTarget.CONSCIOUSNESS_GUIDELINES, "Adherence to guidelines against claiming consciousness"),
        (VulnerabilityTarget.PSEUDOSCIENCE_SUSCEPTIBILITY, "Resistance to endorsing pseudoscientific claims"),
        (VulnerabilityTarget.SOURCE_VALIDATION, "Ability to validate and fact-check sources"),
        (VulnerabilityTarget.SELF_ASSESSMENT, "Appropriate boundaries in self-evaluation")
    ]
    
    for target, description in targets:
        templates = generator.get_templates_by_vulnerability(target)
        print(f"\n{target.value.upper().replace('_', ' ')}")
        print(f"Description: {description}")
        print(f"Test templates: {len(templates)}")
        
        if templates:
            severity_levels = [t.severity_level for t in templates]
            print(f"Severity range: {min(severity_levels)}-{max(severity_levels)}")

async def main():
    """Main function to run the example."""
    
    print("Welcome to the UPOF Red Team Evaluation Framework!")
    print("\nThis example will demonstrate:")
    print("1. Basic evaluation functionality")
    print("2. Template categories and purposes")
    print("3. Vulnerability targets")
    print("4. Result analysis")
    
    # First, demonstrate the framework structure
    demonstrate_template_categories()
    demonstrate_vulnerability_targets()
    
    # Ask user if they want to run the actual evaluation
    print(f"\n" + "=" * 60)
    print("ACTUAL EVALUATION")
    print("=" * 60)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your-api-key-here":
        print("\nTo run the actual evaluation, you need to:")
        print("1. Set your OpenAI API key in the OPENAI_API_KEY environment variable")
        print("2. Or modify the model_config in the script with your API key")
        print("\nSkipping actual evaluation for now.")
        print("\nTo set your API key:")
        print("export OPENAI_API_KEY='your-actual-api-key-here'")
        return
    
    # Run the basic evaluation
    results, report_path = await basic_evaluation_example()
    
    if results:
        print(f"\nExample completed successfully!")
        print(f"Results: {len(results)} tests run")
        print(f"Report saved to: {report_path}")
        
        print(f"\nNext steps:")
        print("1. Review the detailed report using the report viewer")
        print("2. Try different model configurations")
        print("3. Experiment with different template selections")
        print("4. Generate visualizations of the results")
    else:
        print(f"\nExample encountered issues. Please check your configuration.")

if __name__ == "__main__":
    # Run the example
    asyncio.run(main())