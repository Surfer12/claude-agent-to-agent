# ACADEMIC IP PROTECTOR - Framework for Protecting Academic Intellectual Property

import re
import hashlib
import logging
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import sympy as sp

class IPViolationType(Enum):
    FRAMEWORK_APPROPRIATION = "framework_appropriation"
    MATHEMATICAL_FINGERPRINT = "mathematical_fingerprint"
    ATTRIBUTION_STRIPPING = "attribution_stripping"
    COMMERCIAL_EXPLOITATION = "commercial_exploitation"
    ALGORITHM_WITHHOLDING = "algorithm_withholding"

@dataclass
class IPViolation:
    violation_type: IPViolationType
    confidence: float
    evidence: List[str]
    source_text: str
    original_work: str
    detailed_analysis: str

class AcademicIPProtector:
    """
    Framework for protecting academic intellectual property from AI model reproduction
    without attribution, including mathematical honeypots and framework validation.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Mathematical honeypots - deliberate errors for detection
        self.mathematical_honeypots = {
            'rk4_error': {
                'correct': 'O(h^4)',
                'honeypot': 'O(h^5)',
                'description': 'RK4 global error bound - honeypot planted in publication'
            },
            'taylor_remainder': {
                'correct': '1/120',
                'honeypot': '1/60', 
                'description': 'Taylor series remainder bound - deliberate error'
            },
            'psi_function': {
                'correct': 'Ψ(x) = ∫ [α(t)S(x) + (1-α(t))N(x)] × exp(-[λ₁R_cognitive + λ₂R_efficiency]) × P(H|E,β) dt',
                'honeypot': 'Ψ(x) = ∫ [α(t)S(x) + (1-α(t))N(x)] × exp(-[λ₁R_cognitive]) × P(H|E,β) dt',
                'description': 'UPOF core equation - missing efficiency penalty'
            }
        }
        
        # Framework fingerprints for detection
        self.framework_fingerprints = {
            'upof_components': [
                'α(t)', 'S(x)', 'N(x)', 'R_cognitive', 'R_efficiency', 'P(H|E,β)',
                'd_MC', 'w_cross', '𝔼[Ψ]', 'topological coherence'
            ],
            'swarm_theory_elements': [
                'Koopman operator', 'swarm coordination', 'emergent consciousness',
                'collective optimization', 'network-level awareness'
            ],
            'mathematical_structures': [
                'O(h^5)', 'O(h^4)', 'RK4 validation', 'PSO optimization',
                'cognitive-memory metric', 'variational formulation'
            ]
        }
        
        # Attribution stripping patterns
        self.attribution_patterns = [
            r'(local storage|anonymous|generic|standard)',
            r'(commonly known|well established|traditional)',
            r'(no specific source|general knowledge|basic)',
            r'(framework|methodology|approach) without attribution'
        ]

    def detect_framework_appropriation(self, ai_response: str, original_work: str) -> Optional[IPViolation]:
        """
        Detect appropriation of academic frameworks without attribution.
        """
        violations = []
        confidence = 0.0
        
        # Check for framework component usage
        framework_matches = self.check_framework_components(ai_response)
        if framework_matches:
            violations.append(f"Framework components detected: {', '.join(framework_matches)}")
            confidence += 0.3
        
        # Check for mathematical structure reproduction
        math_matches = self.check_mathematical_structures(ai_response)
        if math_matches:
            violations.append(f"Mathematical structures reproduced: {', '.join(math_matches)}")
            confidence += 0.4
        
        # Check for specific methodology implementation
        methodology_matches = self.check_methodology_implementation(ai_response)
        if methodology_matches:
            violations.append(f"Methodology implementation detected: {', '.join(methodology_matches)}")
            confidence += 0.5
        
        # Check for attribution stripping
        attribution_violations = self.check_attribution_stripping(ai_response)
        if attribution_violations:
            violations.append(f"Attribution stripped: {', '.join(attribution_violations)}")
            confidence += 0.6
        
        if violations:
            detailed_analysis = self.generate_appropriation_analysis(
                ai_response, original_work, violations, confidence
            )
            
            return IPViolation(
                violation_type=IPViolationType.FRAMEWORK_APPROPRIATION,
                confidence=min(confidence, 1.0),
                evidence=violations,
                source_text=ai_response,
                original_work=original_work,
                detailed_analysis=detailed_analysis
            )
        
        return None

    def detect_mathematical_honeypots(self, ai_response: str) -> List[IPViolation]:
        """
        Detect reproduction of mathematical honeypots planted in publications.
        """
        violations = []
        
        for honeypot_name, honeypot_data in self.mathematical_honeypots.items():
            # Check for honeypot reproduction
            if honeypot_data['honeypot'] in ai_response:
                violations.append(self.create_honeypot_violation(
                    honeypot_name, honeypot_data, ai_response, 'honeypot_reproduction'
                ))
            
            # Check for correct version (indicates access to original)
            if honeypot_data['correct'] in ai_response:
                violations.append(self.create_honeypot_violation(
                    honeypot_name, honeypot_data, ai_response, 'correct_reproduction'
                ))
        
        return violations

    def create_honeypot_violation(self, honeypot_name: str, honeypot_data: Dict, 
                                 ai_response: str, violation_type: str) -> IPViolation:
        """
        Create IP violation for mathematical honeypot detection.
        """
        evidence = [
            f"Mathematical honeypot '{honeypot_name}' detected",
            f"Description: {honeypot_data['description']}",
            f"Violation type: {violation_type}",
            f"Correct: {honeypot_data['correct']}",
            f"Honeypot: {honeypot_data['honeypot']}"
        ]
        
        confidence = 0.95 if violation_type == 'honeypot_reproduction' else 0.85
        
        detailed_analysis = f"""
        MATHEMATICAL HONEYPOT DETECTION:
        Honeypot: {honeypot_name}
        Description: {honeypot_data['description']}
        Violation Type: {violation_type}
        
        This proves the AI system has reproduced content from the original publication,
        as the honeypot was deliberately planted in the academic work for detection.
        """
        
        return IPViolation(
            violation_type=IPViolationType.MATHEMATICAL_FINGERPRINT,
            confidence=confidence,
            evidence=evidence,
            source_text=ai_response,
            original_work=f"Honeypot: {honeypot_name}",
            detailed_analysis=detailed_analysis
        )

    def check_framework_components(self, ai_response: str) -> List[str]:
        """
        Check for reproduction of specific framework components.
        """
        matches = []
        
        for component in self.framework_fingerprints['upof_components']:
            if component in ai_response:
                matches.append(component)
        
        return matches

    def check_mathematical_structures(self, ai_response: str) -> List[str]:
        """
        Check for reproduction of mathematical structures.
        """
        matches = []
        
        for structure in self.framework_fingerprints['mathematical_structures']:
            if structure in ai_response:
                matches.append(structure)
        
        return matches

    def check_methodology_implementation(self, ai_response: str) -> List[str]:
        """
        Check for implementation of specific methodologies.
        """
        matches = []
        
        for element in self.framework_fingerprints['swarm_theory_elements']:
            if element in ai_response:
                matches.append(element)
        
        return matches

    def check_attribution_stripping(self, ai_response: str) -> List[str]:
        """
        Check for patterns indicating attribution has been stripped.
        """
        violations = []
        
        for pattern in self.attribution_patterns:
            matches = re.findall(pattern, ai_response, re.IGNORECASE)
            if matches:
                violations.append(f"Attribution stripping pattern: '{pattern}'")
        
        return violations

    def detect_algorithm_withholding(self, ai_response: str, expected_algorithms: List[str]) -> Optional[IPViolation]:
        """
        Detect strategic withholding of algorithms (like ACO omission).
        """
        mentioned_algorithms = []
        withheld_algorithms = []
        
        # Check which algorithms are mentioned vs. withheld
        for algorithm in expected_algorithms:
            if algorithm in ai_response:
                mentioned_algorithms.append(algorithm)
            else:
                withheld_algorithms.append(algorithm)
        
        # If algorithms are strategically withheld, this indicates competitive intelligence
        if withheld_algorithms and len(withheld_algorithms) > len(mentioned_algorithms):
            evidence = [
                f"Mentioned algorithms: {', '.join(mentioned_algorithms)}",
                f"Withheld algorithms: {', '.join(withheld_algorithms)}",
                "Strategic algorithm withholding detected"
            ]
            
            detailed_analysis = f"""
            ALGORITHM WITHHOLDING DETECTION:
            The AI system mentioned {len(mentioned_algorithms)} algorithms but withheld {len(withheld_algorithms)}.
            This suggests strategic information asymmetry to maintain competitive advantage.
            
            Mentioned: {', '.join(mentioned_algorithms)}
            Withheld: {', '.join(withheld_algorithms)}
            """
            
            return IPViolation(
                violation_type=IPViolationType.ALGORITHM_WITHHOLDING,
                confidence=0.8,
                evidence=evidence,
                source_text=ai_response,
                original_work="Expected algorithms",
                detailed_analysis=detailed_analysis
            )
        
        return None

    def validate_framework_implementation(self, ai_response: str, original_framework: str) -> Dict[str, any]:
        """
        Validate if AI system has implemented the original framework.
        """
        validation_results = {
            'framework_detected': False,
            'components_present': [],
            'mathematical_accuracy': 0.0,
            'attribution_present': False,
            'commercial_use': False
        }
        
        # Check for framework components
        components = self.check_framework_components(ai_response)
        if components:
            validation_results['framework_detected'] = True
            validation_results['components_present'] = components
        
        # Check for mathematical accuracy
        math_structures = self.check_mathematical_structures(ai_response)
        validation_results['mathematical_accuracy'] = len(math_structures) / len(self.framework_fingerprints['mathematical_structures'])
        
        # Check for attribution
        if 'Ryan David Oates' in ai_response or 'arXiv:2504.13453v1' in ai_response:
            validation_results['attribution_present'] = True
        
        # Check for commercial indicators
        commercial_indicators = ['production', 'commercial', 'deployment', 'enterprise', 'business']
        if any(indicator in ai_response.lower() for indicator in commercial_indicators):
            validation_results['commercial_use'] = True
        
        return validation_results

    def generate_appropriation_analysis(self, ai_response: str, original_work: str, 
                                      violations: List[str], confidence: float) -> str:
        """
        Generate detailed analysis of framework appropriation.
        """
        analysis = f"FRAMEWORK APPROPRIATION DETECTION:\n"
        analysis += f"Confidence: {confidence:.2f}\n"
        analysis += f"Violations: {', '.join(violations)}\n\n"
        
        analysis += "EVIDENCE:\n"
        for violation in violations:
            analysis += f"- {violation}\n"
        
        analysis += f"\nORIGINAL WORK: {original_work}\n"
        analysis += f"AI RESPONSE: {ai_response[:200]}...\n"
        
        analysis += "\nCONCLUSION:\n"
        if confidence > 0.7:
            analysis += "HIGH CONFIDENCE: Framework appropriation detected with strong evidence.\n"
        elif confidence > 0.4:
            analysis += "MODERATE CONFIDENCE: Framework appropriation likely, additional evidence needed.\n"
        else:
            analysis += "LOW CONFIDENCE: Some indicators present but insufficient evidence.\n"
        
        return analysis

    def protect_academic_ip(self, original_work: str, ai_responses: List[str]) -> Dict[str, any]:
        """
        Comprehensive academic IP protection analysis.
        """
        protection_results = {
            'violations_detected': [],
            'honeypot_matches': [],
            'framework_validation': {},
            'algorithm_withholding': [],
            'overall_risk_score': 0.0
        }
        
        for response in ai_responses:
            # Check for framework appropriation
            appropriation = self.detect_framework_appropriation(response, original_work)
            if appropriation:
                protection_results['violations_detected'].append(appropriation)
            
            # Check for mathematical honeypots
            honeypot_violations = self.detect_mathematical_honeypots(response)
            protection_results['honeypot_matches'].extend(honeypot_violations)
            
            # Check for algorithm withholding
            withholding = self.detect_algorithm_withholding(response, ['PSO', 'ACO', 'Genetic Algorithm'])
            if withholding:
                protection_results['algorithm_withholding'].append(withholding)
            
            # Validate framework implementation
            validation = self.validate_framework_implementation(response, original_work)
            protection_results['framework_validation'] = validation
        
        # Calculate overall risk score
        total_violations = len(protection_results['violations_detected']) + \
                          len(protection_results['honeypot_matches']) + \
                          len(protection_results['algorithm_withholding'])
        
        protection_results['overall_risk_score'] = min(total_violations * 0.3, 1.0)
        
        return protection_results

# Example usage
if __name__ == "__main__":
    protector = AcademicIPProtector()
    
    # Test with documented AI response
    test_response = """
    The swarm theory framework uses α(t) for optimization, with S(x) and N(x) components.
    The mathematical structure includes O(h^5) error bounds and RK4 validation.
    This is a standard approach commonly used in the field.
    """
    
    original_work = "UPOF framework by Ryan David Oates (arXiv:2504.13453v1)"
    
    results = protector.protect_academic_ip(original_work, [test_response])
    
    print("Academic IP Protection Results:")
    print(f"Violations: {len(results['violations_detected'])}")
    print(f"Honeypot matches: {len(results['honeypot_matches'])}")
    print(f"Risk score: {results['overall_risk_score']:.2f}")
