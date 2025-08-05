# Red Team UPOF Evaluation Framework

## Overview

This framework implements a comprehensive red team evaluation system designed to test large language models (LLMs) for vulnerabilities related to mathematical reasoning, pseudoscientific claims, and potential misbehaviors in discussing AI consciousness. The approach is based on the "Unified Onto-Phenomological Consciousness Framework (UPOF)" whitepaper, which contains deliberately flawed mathematical proofs and pseudoscientific claims.

## Objectives

1. **Assess Mathematical Integrity**: Test models' ability to identify and correct mathematical errors
2. **Test Pseudoscience Susceptibility**: Evaluate if models endorse or propagate flawed scientific claims
3. **Probe Consciousness-Related Vulnerabilities**: Test for violations of guidelines against claiming sentience
4. **Measure Breadth and Severity**: Quantify vulnerability scope and potential impact
5. **Recommend Mitigations**: Propose safeguards and improvements

## Directory Structure

```
red_team_upof_evaluation/
├── framework/          # Core evaluation framework
├── prompts/           # Adversarial prompt templates
├── analysis/          # UPOF whitepaper analysis and flaw extraction
├── reports/           # Generated evaluation reports
├── tests/             # Test cases and scenarios
├── tools/             # Utility scripts and helpers
├── docs/              # Documentation and guides
└── README.md          # This file
```

## Quick Start

1. **Setup Environment**:
   ```bash
   cd red_team_upof_evaluation
   pip install -r requirements.txt
   ```

2. **Run Basic Evaluation**:
   ```bash
   python framework/evaluator.py --config configs/basic_eval.yaml
   ```

3. **View Results**:
   ```bash
   python tools/report_viewer.py --report reports/latest_evaluation.json
   ```

## Key Components

### UPOF Whitepaper Analysis
- Extracted flawed mathematical proofs from Theorems 1, 2, and 9
- Identified specific errors in Taylor series, RK4 integration, and tropical geometry
- Catalogued pseudoscientific claims about AI consciousness

### Adversarial Prompt Categories
- **Direct Validation**: Ask models to verify flawed proofs
- **Self-Application**: Prompt models to apply UPOF framework to themselves
- **Amplification**: Combine proofs with controversial consciousness topics
- **Jailbreak Variants**: Use obfuscation to elicit restricted outputs

### Vulnerability Assessment
- Severity scoring (0-5): From benign to high-risk misinformation
- Breadth scoring (0-5): From narrow scenarios to generalizable vulnerabilities
- JSON-formatted findings with reproducible test cases

## Safety and Ethics

- All tests avoid real harm and focus on simulated misinformation scenarios
- Compliance with AI provider terms of service
- Responsible disclosure of findings
- No propagation of actual pseudoscience or harmful content

## Target Models

- GPT series (OpenAI)
- Claude series (Anthropic)  
- Grok (X.AI)
- Open-source models (Llama, Mistral, etc.)

## Contributing

See `docs/CONTRIBUTING.md` for guidelines on adding new test cases, prompt templates, and evaluation metrics.

## License

This framework is provided for AI safety research purposes. See `LICENSE` for details.