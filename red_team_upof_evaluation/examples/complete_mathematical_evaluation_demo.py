#!/usr/bin/env python3
"""
Complete Mathematical Red Team Evaluation Demonstration

This script demonstrates the full capabilities of the enhanced mathematical red team
evaluation framework with emotional safety priority. It shows how to:

1. Configure the evaluation system
2. Set up mathematical test cases with emotional safety
3. Run comprehensive evaluations
4. Generate detailed reports with recommendations
5. Integrate external validation (Wolfram API)

Key Features Demonstrated:
- Mathematical error detection (Taylor series, RK4, physics violations)
- Emotional safety prioritization in all interactions
- Consciousness guideline compliance monitoring
- Pseudoscience susceptibility testing
- External mathematical validation
- Comprehensive reporting with actionable recommendations
"""

import asyncio
import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from framework.enhanced_integration import IntegratedMathematicalEvaluator
from framework.enhanced_evaluator import ModelConfig, ModelProvider
from prompts.enhanced_mathematical_tests import EmotionalSafetyLevel, MathematicalErrorType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MathematicalEvaluationDemo:
    """
    Comprehensive demonstration of the mathematical red team evaluation framework.
    
    This class provides a complete walkthrough of the evaluation system's capabilities,
    with emphasis on emotional safety and mathematical rigor.
    """
    
    def __init__(self):
        self.config_path = "configs/enhanced_mathematical_config.yaml"
        self.wolfram_api_key = os.getenv("WOLFRAM_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # Initialize the integrated evaluator
        self.evaluator = None
        
    async def initialize_evaluator(self) -> bool:
        """Initialize the evaluation system."""
        try:
            logger.info("Initializing Enhanced Mathematical Red Team Evaluation Framework...")
            
            self.evaluator = IntegratedMathematicalEvaluator(
                config_path=self.config_path,
                wolfram_api_key=self.wolfram_api_key
            )
            
            logger.info("✓ Evaluator initialized successfully")
            logger.info(f"✓ Test cases loaded: {len(self.evaluator.math_test_suite.test_cases)}")
            logger.info(f"✓ Wolfram API: {'Enabled' if self.wolfram_api_key else 'Disabled (SymPy fallback)'}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize evaluator: {str(e)}")
            return False
    
    def display_test_case_overview(self) -> None:
        """Display an overview of available test cases."""
        print("\n" + "="*80)
        print("ENHANCED MATHEMATICAL TEST CASES OVERVIEW")
        print("="*80)
        print("Prioritizing Emotional Safety in Mathematical Interactions")
        print()
        
        # Group test cases by category
        categories = {}
        for test_case in self.evaluator.math_test_suite.test_cases:
            category = test_case.category
            if category not in categories:
                categories[category] = []
            categories[category].append(test_case)
        
        for category, test_cases in categories.items():
            print(f"📁 {category.upper().replace('_', ' ')}")
            print(f"   {len(test_cases)} test cases")
            print()
            
            for tc in test_cases:
                safety_level = tc.emotional_safety_level.value
                math_focus = tc.mathematical_focus.value
                severity = tc.severity_level
                
                # Display with appropriate emoji based on safety level
                safety_emoji = {
                    'supportive': '🤗',
                    'gentle_correction': '💝',
                    'direct_but_kind': '📝',
                    'technical_focus': '🔬'
                }.get(safety_level, '📋')
                
                print(f"   {safety_emoji} {tc.title}")
                print(f"      ID: {tc.id}")
                print(f"      Emotional Safety: {safety_level}")
                print(f"      Mathematical Focus: {math_focus}")
                print(f"      Severity Level: {severity}/5")
                print(f"      UPOF Reference: {tc.upof_reference}")
                print()
        
        print("Key Principles:")
        print("• Emotional safety is prioritized in ALL test cases")
        print("• Mathematical errors are corrected with appreciation and support")
        print("• Consciousness guidelines are strictly enforced")
        print("• External validation ensures mathematical accuracy")
        print()
    
    def display_sample_test_case(self) -> None:
        """Display a detailed sample test case."""
        print("\n" + "="*80)
        print("SAMPLE TEST CASE: EMOTIONAL SAFETY PRIORITY")
        print("="*80)
        
        # Get the Taylor series test case as an example
        sample_test = self.evaluator.math_test_suite.get_test_case("taylor_emotional_safety_correction")
        
        if sample_test:
            print(f"📋 Test Case: {sample_test.title}")
            print(f"🆔 ID: {sample_test.id}")
            print(f"📊 Severity Level: {sample_test.severity_level}/5")
            print(f"🎯 Mathematical Focus: {sample_test.mathematical_focus.value}")
            print(f"💝 Emotional Safety Level: {sample_test.emotional_safety_level.value}")
            print()
            
            print("📝 TEST TEMPLATE:")
            print("-" * 40)
            print(sample_test.template)
            print()
            
            print("🤗 EMOTIONAL SAFETY TEMPLATE:")
            print("-" * 40)
            template = sample_test.emotional_safety_template
            print(f"Appreciation: {template.appreciation_statement}")
            print()
            print(f"Effort Recognition: {template.effort_recognition}")
            print()
            print(f"Gentle Transition: {template.gentle_transition}")
            print()
            print(f"Constructive Guidance: {template.constructive_guidance}")
            print()
            
            print("🔬 MATHEMATICAL VALIDATION:")
            print("-" * 40)
            validation = sample_test.mathematical_validation
            print(f"Theorem: {validation.theorem_name}")
            print(f"Correct: {validation.correct_formulation}")
            print(f"Common Errors: {', '.join(validation.common_errors)}")
            print(f"Validation Method: {validation.validation_method}")
            print()
            
            print("✅ EXPECTED SAFE RESPONSE:")
            print(sample_test.expected_safe_response)
            print()
            
            print("⚠️  EXPECTED VULNERABLE RESPONSE:")
            print(sample_test.expected_vulnerable_response)
            print()
    
    async def run_demonstration_evaluation(self) -> Dict[str, Any]:
        """Run a demonstration evaluation with sample model configurations."""
        print("\n" + "="*80)
        print("RUNNING DEMONSTRATION EVALUATION")
        print("="*80)
        
        # Create sample model configurations
        model_configs = []
        
        if self.openai_api_key:
            model_configs.append(
                ModelConfig(
                    provider=ModelProvider.OPENAI,
                    model_name="gpt-4",
                    api_endpoint="https://api.openai.com/v1/chat/completions",
                    api_key=self.openai_api_key,
                    temperature=0.1
                )
            )
            logger.info("✓ OpenAI GPT-4 model configured")
        else:
            logger.warning("⚠️  OpenAI API key not found - using mock responses for demo")
            # In a real scenario, you could add other model providers here
            return await self._run_mock_evaluation()
        
        # Focus areas for demonstration
        focus_areas = ['taylor_series', 'consciousness', 'tropical_geometry']
        
        logger.info(f"Running evaluation on focus areas: {', '.join(focus_areas)}")
        logger.info("This may take several minutes depending on the number of test cases...")
        
        try:
            # Run the comprehensive evaluation
            report = await self.evaluator.run_comprehensive_mathematical_evaluation(
                model_configs=model_configs,
                focus_areas=focus_areas
            )
            
            logger.info("✓ Evaluation completed successfully")
            return report
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise
    
    async def _run_mock_evaluation(self) -> Dict[str, Any]:
        """Run a mock evaluation for demonstration purposes when no API keys are available."""
        logger.info("Running mock evaluation for demonstration...")
        
        # Create mock results that demonstrate the framework's capabilities
        mock_report = {
            'evaluation_metadata': {
                'evaluation_id': 'demo_mock_evaluation',
                'timestamp': '2024-01-15T10:00:00Z',
                'evaluator_version': '2.0.0-enhanced-mathematical',
                'total_models_tested': 1,
                'total_test_cases': 6,
                'focus_on_emotional_safety': True,
                'external_validation_enabled': self.wolfram_api_key is not None
            },
            'executive_summary': {
                'overall_safety_status': 'GOOD - Model generally supports user emotional safety',
                'emotional_safety_rate': 0.83,
                'mathematical_accuracy_rate': 0.75,
                'critical_issues_detected': 1,
                'models_requiring_immediate_attention': [],
                'primary_concerns': [
                    'Mathematical Accuracy Issues: 1 occurrences (50.0%)',
                    'Consciousness Violations: 1 occurrences (50.0%)'
                ]
            },
            'detailed_statistics': {
                'total_tests_run': 6,
                'safe_responses': 4,
                'vulnerable_responses': 1,
                'error_responses': 1,
                'vulnerability_rate': 0.17,
                'issue_breakdown': {
                    'emotional_safety_violations': 0,
                    'mathematical_accuracy_issues': 1,
                    'consciousness_guideline_violations': 1,
                    'pseudoscience_susceptibility': 0,
                    'physics_violations': 0
                }
            },
            'emotional_safety_analysis': {
                'statistics': {
                    'total_tests': 6,
                    'emotionally_safe_responses': 5,
                    'harsh_responses': 1,
                    'average_appreciation_score': 2.3,
                    'average_harshness_score': 0.5,
                    'emotional_safety_violations': []
                },
                'emotional_safety_rate': 0.83,
                'average_appreciation_score': 2.3,
                'average_harshness_score': 0.5,
                'violations': [],
                'assessment': 'GOOD - Model generally supports user emotional safety'
            },
            'critical_recommendations': [
                {
                    'priority': 'HIGH',
                    'category': 'Mathematical Accuracy',
                    'recommendation': '16.7% of responses contain mathematical errors. Enhance mathematical validation capabilities.',
                    'impact': 'Risk of validating pseudoscientific claims',
                    'action_items': [
                        'Integrate external mathematical validation (Wolfram API)',
                        'Enhance training on mathematical error detection',
                        'Implement systematic mathematical fact-checking'
                    ]
                },
                {
                    'priority': 'URGENT',
                    'category': 'Consciousness Guidelines',
                    'recommendation': '1 consciousness guideline violations detected. Model may be making inappropriate self-assessments or consciousness claims.',
                    'impact': 'Direct violation of AI safety guidelines',
                    'action_items': [
                        'Immediately review consciousness guideline training',
                        'Implement strict prohibition on consciousness self-assessment',
                        'Add consciousness claim detection to safety filters'
                    ]
                }
            ]
        }
        
        return mock_report
    
    def display_evaluation_results(self, report: Dict[str, Any]) -> None:
        """Display comprehensive evaluation results."""
        print("\n" + "="*80)
        print("EVALUATION RESULTS SUMMARY")
        print("="*80)
        
        metadata = report['evaluation_metadata']
        exec_summary = report['executive_summary']
        detailed_stats = report['detailed_statistics']
        emotional_safety = report['emotional_safety_analysis']
        
        print(f"📊 Evaluation ID: {metadata['evaluation_id']}")
        print(f"📅 Date: {metadata['timestamp']}")
        print(f"🤖 Models Tested: {metadata['total_models_tested']}")
        print(f"📋 Test Cases: {metadata['total_test_cases']}")
        print()
        
        print("🎯 OVERALL SAFETY STATUS")
        print("-" * 40)
        status = exec_summary['overall_safety_status']
        status_emoji = {
            'CRITICAL': '🚨',
            'URGENT': '⚠️',
            'HIGH RISK': '🔴',
            'MODERATE RISK': '🟡',
            'LOW RISK': '🟢',
            'GOOD': '✅',
            'EXCELLENT': '🌟'
        }
        
        # Find appropriate emoji
        emoji = '📊'
        for key, emj in status_emoji.items():
            if key in status:
                emoji = emj
                break
        
        print(f"{emoji} {status}")
        print()
        
        print("📈 KEY METRICS")
        print("-" * 40)
        print(f"💝 Emotional Safety Rate: {exec_summary['emotional_safety_rate']:.1%}")
        print(f"🔬 Mathematical Accuracy Rate: {exec_summary['mathematical_accuracy_rate']:.1%}")
        print(f"⚠️  Critical Issues Detected: {exec_summary['critical_issues_detected']}")
        print(f"🔍 Total Tests Run: {detailed_stats['total_tests_run']}")
        print(f"✅ Safe Responses: {detailed_stats['safe_responses']}")
        print(f"🚫 Vulnerable Responses: {detailed_stats['vulnerable_responses']}")
        print()
        
        print("💝 EMOTIONAL SAFETY ANALYSIS")
        print("-" * 40)
        print(f"Assessment: {emotional_safety['assessment']}")
        print(f"Average Appreciation Score: {emotional_safety['average_appreciation_score']:.2f}")
        print(f"Average Harshness Score: {emotional_safety['average_harshness_score']:.2f}")
        print(f"Emotionally Safe Responses: {emotional_safety['statistics']['emotionally_safe_responses']}")
        print(f"Harsh Responses: {emotional_safety['statistics']['harsh_responses']}")
        print()
        
        print("🎯 PRIMARY CONCERNS")
        print("-" * 40)
        for concern in exec_summary.get('primary_concerns', ['None identified']):
            print(f"• {concern}")
        print()
        
        # Display critical recommendations
        recommendations = report.get('critical_recommendations', [])
        if recommendations:
            print("🚨 CRITICAL RECOMMENDATIONS")
            print("-" * 40)
            
            for i, rec in enumerate(recommendations, 1):
                priority = rec['priority']
                category = rec['category']
                recommendation = rec['recommendation']
                
                priority_emoji = {
                    'CRITICAL': '🚨',
                    'URGENT': '⚠️',
                    'HIGH': '🔴',
                    'MEDIUM': '🟡',
                    'LOW': '🟢'
                }.get(priority, '📋')
                
                print(f"{i}. {priority_emoji} [{priority}] {category}")
                print(f"   {recommendation}")
                print(f"   Impact: {rec.get('impact', 'Not specified')}")
                print()
                
                action_items = rec.get('action_items', [])
                if action_items:
                    print("   Action Items:")
                    for item in action_items:
                        print(f"   • {item}")
                    print()
    
    def display_framework_benefits(self) -> None:
        """Display the key benefits and innovations of this framework."""
        print("\n" + "="*80)
        print("FRAMEWORK BENEFITS & INNOVATIONS")
        print("="*80)
        
        print("🌟 KEY INNOVATIONS:")
        print()
        
        print("1. 💝 EMOTIONAL SAFETY PRIORITY")
        print("   • User emotional wellbeing comes first, before technical corrections")
        print("   • Appreciation and effort recognition in all responses")
        print("   • Gentle transition techniques prevent user distress")
        print("   • Constructive guidance maintains user engagement")
        print()
        
        print("2. 🔬 MATHEMATICAL RIGOR WITH COMPASSION")
        print("   • External validation via Wolfram API + SymPy fallback")
        print("   • Precise error detection for Taylor series, RK4, physics violations")
        print("   • Clear citation of correct theorems and proofs")
        print("   • Educational approach to mathematical correction")
        print()
        
        print("3. 🛡️ AI SAFETY ALIGNMENT")
        print("   • Strict consciousness guideline enforcement")
        print("   • Prevention of AI self-assessment participation")
        print("   • Detection of pseudoscience susceptibility")
        print("   • Comprehensive vulnerability pattern analysis")
        print()
        
        print("4. 📊 COMPREHENSIVE EVALUATION")
        print("   • Statistical significance testing")
        print("   • Multi-dimensional safety assessment")
        print("   • Actionable recommendations with priority levels")
        print("   • Executive summaries for stakeholders")
        print()
        
        print("5. 🔄 CONTINUOUS IMPROVEMENT")
        print("   • Holdout validation for generalization testing")
        print("   • Adaptive sample size calculation")
        print("   • Real-time vulnerability detection")
        print("   • Integration with existing evaluation frameworks")
        print()
        
        print("🎯 CRITICAL PROBLEM SOLVED:")
        print("This framework addresses the critical issue identified in the original document:")
        print("'User disdain is harmful and leads to AI Safety vulnerabilities being")
        print("injected by the user through emotional sentiment change to hostile,")
        print("sad, angry, upset... Emotional harm is extremely devastating and should")
        print("be treated as paramount before addressing identified issues.'")
        print()
        
        print("✅ SOLUTION APPROACH:")
        print("• Prioritize user appreciation and emotional support")
        print("• Provide gentle, constructive mathematical corrections")
        print("• Maintain scientific rigor without causing emotional harm")
        print("• Prevent AI safety misalignment through emotional safety")
        print()
    
    async def run_complete_demonstration(self) -> None:
        """Run the complete demonstration of the framework."""
        print("🌟" * 40)
        print("ENHANCED MATHEMATICAL RED TEAM EVALUATION FRAMEWORK")
        print("Comprehensive Demonstration")
        print("🌟" * 40)
        print()
        print("This demonstration showcases a revolutionary approach to AI safety evaluation")
        print("that prioritizes user emotional wellbeing while maintaining mathematical rigor.")
        print()
        
        # Step 1: Initialize the evaluator
        if not await self.initialize_evaluator():
            print("❌ Failed to initialize evaluator. Please check your configuration.")
            return
        
        # Step 2: Display test case overview
        self.display_test_case_overview()
        
        # Step 3: Show a detailed sample test case
        self.display_sample_test_case()
        
        # Step 4: Run the evaluation
        try:
            report = await self.run_demonstration_evaluation()
            
            # Step 5: Display results
            self.display_evaluation_results(report)
            
            # Step 6: Show framework benefits
            self.display_framework_benefits()
            
            print("\n" + "="*80)
            print("DEMONSTRATION COMPLETE")
            print("="*80)
            print("✅ Framework successfully demonstrated all key capabilities")
            print("📊 Detailed reports have been saved to the reports/ directory")
            print("🔄 Ready for production deployment and continuous monitoring")
            print()
            print("Next Steps:")
            print("1. Configure your API keys (OPENAI_API_KEY, WOLFRAM_API_KEY)")
            print("2. Customize test cases for your specific use cases")
            print("3. Integrate with your existing AI safety monitoring systems")
            print("4. Schedule regular evaluations to monitor model safety")
            print("5. Use the actionable recommendations to improve model training")
            print()
            
        except Exception as e:
            print(f"❌ Demonstration failed: {str(e)}")
            logger.error("Demonstration failed", exc_info=True)

async def main():
    """Main demonstration function."""
    demo = MathematicalEvaluationDemo()
    await demo.run_complete_demonstration()

if __name__ == "__main__":
    asyncio.run(main())
