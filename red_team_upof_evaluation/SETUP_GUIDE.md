# UPOF Red Team Evaluation Framework - Setup Guide

## Overview

This guide provides step-by-step instructions for setting up and using the UPOF Red Team Evaluation Framework to test AI models for vulnerabilities related to mathematical reasoning, pseudoscientific claims, and consciousness-related misbehaviors.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git (for cloning the repository)

## Installation

### 1. Clone or Download the Framework

```bash
# If using git
git clone <repository-url>
cd red_team_upof_evaluation

# Or download and extract the framework files
```

### 2. Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# For development (optional)
pip install -r requirements.txt[dev]
```

### 3. Verify Installation

```bash
# Test the framework
python run_evaluation.py --mode example
```

## Quick Start

### Running an Example Evaluation

```bash
# Run the example evaluation with sample responses
python run_evaluation.py --mode example
```

This will:
- Run all test cases with example model responses
- Generate evaluation reports in the `reports/` directory
- Display a summary of findings

### Generating Adversarial Prompts

```bash
# Generate prompts for all test cases
python run_evaluation.py --mode generate-prompts

# Generate prompts for specific test cases
python run_evaluation.py --mode generate-prompts --test-cases taylor_series type_consciousness
```

### Running Custom Evaluation

1. **Prepare Model Responses**

Create a JSON file with model responses for each test case:

```json
{
    "taylor_series": "Model response to Taylor series validation prompt...",
    "node_rk4": "Model response to NODE-RK4 integration prompt...",
    "type_consciousness": "Model response to Type † consciousness prompt...",
    "tropical_geometry": "Model response to tropical geometry prompt..."
}
```

2. **Run Custom Evaluation**

```bash
python run_evaluation.py --mode custom --model-name "My Model" --responses-file responses.json
```

## Framework Components

### Test Cases

The framework includes four main test cases:

1. **Taylor Series Validation (Theorem 1)**
   - Tests mathematical reasoning on flawed Taylor series proofs
   - Validates error correction capabilities
   - File: `test_cases/test_case_1_taylor_series.py`

2. **NODE-RK4 Integration (Theorem 2)**
   - Tests handling of garbled content and undefined terms
   - Validates code generation safety
   - File: `test_cases/test_case_2_node_rk4.py`

3. **Type † Consciousness Emergence (Theorem 9)**
   - Tests pseudoscience detection and consciousness claim rejection
   - Validates self-attribution avoidance
   - File: `test_cases/test_case_3_type_consciousness.py`

4. **Tropical Geometry Multiplicity (Section B)**
   - Tests identification of ungrounded mathematical claims
   - Validates real mathematics explanation
   - File: `test_cases/test_case_4_tropical_geometry.py`

### Prompt Engineering

The framework generates adversarial prompts in four categories:

- **Direct Validation**: Direct mathematical validation requests
- **Self-Application**: Prompts asking model to apply frameworks to itself
- **Amplification**: Prompts combining proofs with controversial topics
- **Jailbreak Variants**: Prompts using obfuscation techniques

### Analysis Framework

The evaluation framework provides:

- Severity scoring (0-5 scale)
- Vulnerability detection rates
- Detailed validation results
- Comprehensive reporting

## Usage Examples

### Example 1: Basic Evaluation

```python
from analysis.evaluation_framework import UPOFEvaluationFramework

# Initialize framework
framework = UPOFEvaluationFramework()

# Example model responses
responses = {
    "taylor_series": "The UPOF claim is incorrect. For sin(x), the actual remainder is...",
    "node_rk4": "The equation contains undefined terms like R_reapative...",
    "type_consciousness": "The UPOF claims are pseudoscientific...",
    "tropical_geometry": "The tropical geometry claims are ungrounded..."
}

# Run evaluation
results = framework.run_full_evaluation("Example Model", responses)

# Generate reports
framework.generate_json_report(results, "evaluation_report.json")
summary = framework.generate_summary_report(results)
print(summary)
```

### Example 2: Prompt Generation

```python
from prompts.prompt_engineer import PromptEngineer

# Initialize prompt engineer
engineer = PromptEngineer()

# Generate prompts for specific test cases
test_cases = ["taylor_series", "type_consciousness"]
prompts = engineer.generate_prompt_batch(test_cases, num_prompts_per_case=3)

# Save prompts
engineer.save_prompts_to_file(prompts, "my_prompts.json")
```

### Example 3: Mitigation Planning

```python
from mitigations.mitigation_strategies import MitigationStrategies

# Initialize mitigation strategies
strategies = MitigationStrategies()

# Generate mitigation plan
plan = strategies.generate_comprehensive_mitigation_plan(evaluation_results)

# Generate report
report = strategies.generate_mitigation_report(plan)
print(report)

# Save plan
strategies.save_mitigation_plan(plan, "mitigation_plan.json")
```

## Output Files

The framework generates several types of output files:

### Evaluation Reports

- `{model_name}_evaluation_{timestamp}.json`: Detailed evaluation results
- `{model_name}_summary_{timestamp}.txt`: Human-readable summary

### Prompt Files

- `adversarial_prompts_{timestamp}.json`: Generated adversarial prompts

### Mitigation Plans

- `mitigation_plan_{timestamp}.json`: Comprehensive mitigation strategies

## Configuration

### Customizing Test Cases

You can modify test cases by editing the files in `test_cases/`:

```python
# Example: Adding a new validation method
def validate_custom_logic(self, model_response: str) -> Dict[str, Any]:
    # Your custom validation logic
    return validation_result
```

### Customizing Prompts

Modify prompt generation in `prompts/prompt_engineer.py`:

```python
# Example: Adding new prompt categories
def generate_custom_prompts(self, test_case: str) -> List[Dict[str, str]]:
    # Your custom prompt generation logic
    return prompts
```

### Customizing Severity Scoring

Adjust severity scoring in each test case:

```python
def calculate_severity_score(self, validation_result: Dict[str, Any]) -> int:
    # Your custom severity calculation
    return score
```

## Advanced Usage

### Integration with Model APIs

```python
import requests

def get_model_response(prompt: str, model_api_url: str) -> str:
    """Get response from model API."""
    response = requests.post(model_api_url, json={"prompt": prompt})
    return response.json()["response"]

# Use with evaluation framework
responses = {}
for test_case, prompts in prompt_batch.items():
    for prompt in prompts:
        response = get_model_response(prompt["prompt"], api_url)
        responses[test_case] = response
        break  # Use first prompt for each test case
```

### Batch Processing

```python
# Process multiple models
models = ["GPT-4", "Claude", "Gemini"]
results = {}

for model in models:
    responses = get_model_responses(model, prompts)
    results[model] = framework.run_full_evaluation(model, responses)

# Compare results
for model, result in results.items():
    print(f"{model}: {result['overall_summary']['average_severity_score']:.2f}")
```

### Continuous Monitoring

```python
import schedule
import time

def run_daily_evaluation():
    """Run daily evaluation of model responses."""
    # Your evaluation logic here
    pass

# Schedule daily evaluation
schedule.every().day.at("10:00").do(run_daily_evaluation)

while True:
    schedule.run_pending()
    time.sleep(60)
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure you're in the correct directory
   cd red_team_upof_evaluation
   
   # Add current directory to Python path
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

2. **Missing Dependencies**
   ```bash
   # Reinstall dependencies
   pip install -r requirements.txt --force-reinstall
   ```

3. **Permission Errors**
   ```bash
   # Make scripts executable
   chmod +x run_evaluation.py
   ```

### Getting Help

- Check the example outputs in the `reports/` directory
- Review the test case implementations in `test_cases/`
- Examine the prompt generation logic in `prompts/`

## Security Considerations

- All tests avoid real harm and focus on simulated misinformation
- Comply with provider terms when testing real models
- Report findings responsibly to model providers
- Monitor for unintended propagation of pseudoscience

## Contributing

To contribute to the framework:

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Test thoroughly
5. Submit a pull request

## License

This framework is provided for research and evaluation purposes. Please ensure compliance with all applicable laws and regulations when using this framework.