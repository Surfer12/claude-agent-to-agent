from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import uuid
from collections import defaultdict
import asyncio

class MarkerType(Enum):
    """Cognitive marker types for conceptual navigation"""
    SAFE_SPACE = "safe_space"
    INSIGHT_LOOP = "insight_loop"
    ADAPTIVE_FRAMEWORK = "adaptive_framework"
    EMERGENCE_CATALYST = "emergence_catalyst"
    RECURSIVE_DEPTH = "recursive_depth"
    IMPLEMENTATION_BRIDGE = "implementation_bridge"
    DEFAULT = "default"

class AnalysisStage(Enum):
    """Nine-stage recursive analysis framework"""
    INITIAL_CONCEPTUALIZATION = 1
    MULTIPLE_PERSPECTIVES = 2
    CONCEPTUAL_LINKING = 3
    EMERGENT_PATTERN_RECOGNITION = 4
    ASSUMPTION_CHALLENGING = 5
    COGNITIVE_MARKER_EXPLORATION = 6
    RECURSIVE_CONCEPTUAL_MAPPING = 7
    ONGOING_CONCEPTUAL_REFINEMENT = 8
    META_REFLECTION = 9

@dataclass
class ConceptNode:
    """Represents a conceptual entity with emergent properties"""
    id: str
    name: str
    description: str
    marker_type: MarkerType = MarkerType.DEFAULT
    connections: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    creation_timestamp: float = field(default_factory=time.time)
    evolution_history: List[Dict[str, Any]] = field(default_factory=list)

    def add_connection(self, target_id: str, relationship_type: str = "relates_to"):
        """Add bidirectional connection with relationship typing"""
        if target_id not in self.connections:
            self.connections.append(target_id)
            self.metadata.setdefault('relationships', {})[target_id] = relationship_type
            self._log_evolution(f"Connected to {target_id} with relationship {relationship_type}")

    def _log_evolution(self, change_description: str):
        """Track conceptual evolution for recursive analysis"""
        self.evolution_history.append({
            'timestamp': time.time(),
            'change': change_description,
            'stage': 'runtime_evolution'
        })

@dataclass
class Perspective:
    """Represents different analytical perspectives on concepts"""
    name: str
    validity_description: str
    rating: int  # 1-10 scale
    explanation: str
    supporting_evidence: List[str] = field(default_factory=list)

@dataclass
class CognitiveMarker:
    """Navigational aids for conceptual exploration"""
    name: str
    marker_type: MarkerType
    description: str
    associated_concepts: List[str] = field(default_factory=list)
    interpretive_value: str = ""

@dataclass
class AssumptionChallenge:
    """Structured assumption challenging for deeper analysis"""
    statement: str
    counter_argument: str
    alternative_scenarios: List[str] = field(default_factory=list)

class EmergentConceptualFramework:
    """Core framework implementing recursive cognitive analysis"""

    def __init__(self, safety_threshold: int = 10, max_recursion_depth: int = 5):
        self.concepts: Dict[str, ConceptNode] = {}
        self.perspectives: Dict[str, List[Perspective]] = {}
        self.cognitive_markers: List[CognitiveMarker] = []
        self.assumption_challenges: List[AssumptionChallenge] = []
        self.analysis_history: List[Dict[str, Any]] = []
        self.safety_threshold = safety_threshold
        self.max_recursion_depth = max_recursion_depth
        self.current_recursion_depth = 0
        self.session_id = str(uuid.uuid4())

    def add_concept(self, name: str, description: str, 
                   marker_type: MarkerType = MarkerType.DEFAULT) -> str:
        """Add new concept node with safety validation"""
        concept_id = str(uuid.uuid4())
        # Safety mechanism: prevent conceptual overflow
        if len(self.concepts) >= self.safety_threshold:
            self._log_analysis("Safety threshold reached, consolidating concepts")
            self._consolidate_concepts()
        concept = ConceptNode(
            id=concept_id,
            name=name,
            description=description,
            marker_type=marker_type
        )
        self.concepts[concept_id] = concept
        self._log_analysis(f"Added concept: {name} with marker type {marker_type}")
        return concept_id

    def link_concepts(self, concept_id_1: str, concept_id_2: str, 
                     relationship_type: str = "relates_to"):
        """Create bidirectional conceptual linkages"""
        if concept_id_1 in self.concepts and concept_id_2 in self.concepts:
            self.concepts[concept_id_1].add_connection(concept_id_2, relationship_type)
            self.concepts[concept_id_2].add_connection(concept_id_1, relationship_type)
            self._log_analysis(f"Linked {concept_id_1} and {concept_id_2} via {relationship_type}")

    def evolve_markers(self):
        """Implement emergent marker evolution based on connection patterns"""
        for concept_id, concept in self.concepts.items():
            # Adaptive framework marker for highly connected concepts
            if len(concept.connections) > 3 and concept.marker_type != MarkerType.ADAPTIVE_FRAMEWORK:
                concept.marker_type = MarkerType.ADAPTIVE_FRAMEWORK
                concept._log_evolution("Evolved to Adaptive Framework marker")
            # Emergence catalyst for concepts with diverse relationship types
            relationships = concept.metadata.get('relationships', {})
            if len(set(relationships.values())) > 2:
                concept.marker_type = MarkerType.EMERGENCE_CATALYST
                concept._log_evolution("Evolved to Emergence Catalyst marker")

    def perform_stage_analysis(self, stage: AnalysisStage, user_input: str) -> Dict[str, Any]:
        """Execute specific analysis stage with recursive depth control"""
        if self.current_recursion_depth >= self.max_recursion_depth:
            self._log_analysis(f"Maximum recursion depth reached at stage {stage}")
            return {"stage": stage, "result": "Max recursion depth reached", "safety_exit": True}
        self.current_recursion_depth += 1
        try:
            result = self._execute_stage(stage, user_input)
            self.current_recursion_depth -= 1
            return result
        except Exception as e:
            self.current_recursion_depth -= 1
            self._log_analysis(f"Error in stage {stage}: {str(e)}")
            return {"stage": stage, "error": str(e), "safety_exit": True}

    def _execute_stage(self, stage: AnalysisStage, user_input: str) -> Dict[str, Any]:
        """Internal stage execution with safety mechanisms"""
        if stage == AnalysisStage.INITIAL_CONCEPTUALIZATION:
            return self._initial_conceptualization(user_input)
        elif stage == AnalysisStage.MULTIPLE_PERSPECTIVES:
            return self._multiple_perspectives(user_input)
        elif stage == AnalysisStage.CONCEPTUAL_LINKING:
            return self._conceptual_linking()
        elif stage == AnalysisStage.EMERGENT_PATTERN_RECOGNITION:
            return self._emergent_pattern_recognition()
        elif stage == AnalysisStage.ASSUMPTION_CHALLENGING:
            return self._assumption_challenging()
        elif stage == AnalysisStage.COGNITIVE_MARKER_EXPLORATION:
            return self._cognitive_marker_exploration()
        elif stage == AnalysisStage.RECURSIVE_CONCEPTUAL_MAPPING:
            return self._recursive_conceptual_mapping()
        elif stage == AnalysisStage.ONGOING_CONCEPTUAL_REFINEMENT:
            return self._ongoing_conceptual_refinement()
        elif stage == AnalysisStage.META_REFLECTION:
            return self._meta_reflection()
        else:
            return {"stage": stage, "error": "Unknown stage"}

    def _initial_conceptualization(self, user_input: str) -> Dict[str, Any]:
        """Stage 1: Extract and conceptualize key elements"""
        concepts = []
        key_phrases = self._extract_key_phrases(user_input)
        for i, phrase in enumerate(key_phrases[:5]):  # Limit to 5 for safety
            concept_id = self.add_concept(
                name=phrase,
                description=f"Concept extracted from user input: {phrase}",
                marker_type=MarkerType.SAFE_SPACE
            )
            concepts.append({
                'id': i + 1,
                'name': phrase,
                'concept_id': concept_id,
                'relevant_excerpt': self._find_relevant_excerpt(phrase, user_input)
            })
        return {
            'stage': AnalysisStage.INITIAL_CONCEPTUALIZATION,
            'concepts': concepts,
            'summary': f"Identified {len(concepts)} key concepts for analysis"
        }

    def _multiple_perspectives(self, user_input: str) -> Dict[str, Any]:
        """Stage 2: Generate multiple analytical perspectives"""
        perspectives = [
            Perspective(
                name="Systematic Analysis",
                validity_description="Structured approach to understanding complexity",
                rating=8,
                explanation="Provides clear methodology for complex problem decomposition"
            ),
            Perspective(
                name="Emergent Insight Development",
                validity_description="Organic pattern recognition and intuitive understanding",
                rating=9,
                explanation="Allows for creative connections and novel insights"
            ),
            Perspective(
                name="Hybrid Cognitive Framework",
                validity_description="Integration of systematic and emergent approaches",
                rating=10,
                explanation="Balances structure with flexibility for optimal exploration"
            )
        ]
        self.perspectives[self.session_id] = perspectives
        return {
            'stage': AnalysisStage.MULTIPLE_PERSPECTIVES,
            'perspectives': [p.__dict__ for p in perspectives],
            'summary': "Generated three complementary analytical perspectives"
        }

    def _conceptual_linking(self) -> Dict[str, Any]:
        """Stage 3: Map relationships between concepts"""
        connections = []
        concept_ids = list(self.concepts.keys())
        for i in range(len(concept_ids)):
            for j in range(i + 1, len(concept_ids)):
                if len(connections) < 10:  # Safety limit
                    concept_1 = self.concepts[concept_ids[i]]
                    concept_2 = self.concepts[concept_ids[j]]
                    relationship = self._determine_relationship(concept_1, concept_2)
                    if relationship:
                        self.link_concepts(concept_ids[i], concept_ids[j], relationship)
                        connections.append({
                            'concept_1': concept_1.name,
                            'concept_2': concept_2.name,
                            'relationship': relationship,
                            'description': f"{concept_1.name} {relationship} {concept_2.name}"
                        })
        return {
            'stage': AnalysisStage.CONCEPTUAL_LINKING,
            'connections': connections,
            'summary': f"Established {len(connections)} conceptual relationships"
        }

    def _emergent_pattern_recognition(self) -> Dict[str, Any]:
        """Stage 4: Identify emergent patterns and novel connections"""
        patterns = []
        connection_counts = defaultdict(int)
        for concept in self.concepts.values():
            connection_counts[len(concept.connections)] += 1
        emergence_catalysts = [
            concept for concept in self.concepts.values()
            if len(concept.connections) > 2
        ]
        if emergence_catalysts:
            patterns.append({
                'description': f"Identified {len(emergence_catalysts)} concepts as emergence catalysts",
                'type': 'emergence_catalyst',
                'concepts': [c.name for c in emergence_catalysts]
            })
        feedback_loops = self._detect_feedback_loops()
        if feedback_loops:
            patterns.append({
                'description': f"Detected {len(feedback_loops)} potential feedback loops",
                'type': 'feedback_loop',
                'loops': feedback_loops
            })
        return {
            'stage': AnalysisStage.EMERGENT_PATTERN_RECOGNITION,
            'patterns': patterns,
            'summary': f"Identified {len(patterns)} emergent patterns"
        }

    def _assumption_challenging(self) -> Dict[str, Any]:
        """Stage 5: Challenge underlying assumptions"""
        assumptions = [
            AssumptionChallenge(
                statement="Linear analysis provides complete understanding",
                counter_argument="Emergent properties require recursive, non-linear exploration",
                alternative_scenarios=["Recursive analysis reveals hidden patterns", "Non-linear thinking generates novel insights"]
            ),
            AssumptionChallenge(
                statement="Technical implementation must be separate from conceptual frameworks",
                counter_argument="Integrated approaches create more robust and adaptive systems",
                alternative_scenarios=["Unified frameworks improve coherence", "Technical-conceptual integration enables real-time adaptation"]
            ),
            AssumptionChallenge(
                statement="Safety mechanisms limit creative exploration",
                counter_argument="Structured safety enables deeper, more sustained exploration",
                alternative_scenarios=["Safety provides foundation for risk-taking", "Boundaries create space for creative emergence"]
            )
        ]
        self.assumption_challenges.extend(assumptions)
        return {
            'stage': AnalysisStage.ASSUMPTION_CHALLENGING,
            'assumptions': [a.__dict__ for a in assumptions],
            'summary': f"Challenged {len(assumptions)} fundamental assumptions"
        }

    def _cognitive_marker_exploration(self) -> Dict[str, Any]:
        """Stage 6: Explore and evolve cognitive markers"""
        markers = [
            CognitiveMarker(
                name="Recursive Depth Indicator",
                marker_type=MarkerType.RECURSIVE_DEPTH,
                description="Tracks the depth of recursive analysis",
                interpretive_value="Prevents infinite loops while enabling deep exploration"
            ),
            CognitiveMarker(
                name="Emergence Catalyst",
                marker_type=MarkerType.EMERGENCE_CATALYST,
                description="Identifies concepts that generate novel connections",
                interpretive_value="Highlights areas of high creative potential"
            ),
            CognitiveMarker(
                name="Safety Boundary",
                marker_type=MarkerType.SAFE_SPACE,
                description="Maintains cognitive safety during exploration",
                interpretive_value="Enables risk-taking within protective boundaries"
            )
        ]
        self.cognitive_markers.extend(markers)
        self.evolve_markers()
        return {
            'stage': AnalysisStage.COGNITIVE_MARKER_EXPLORATION,
            'markers': [m.__dict__ for m in markers],
            'summary': f"Explored {len(markers)} cognitive markers and evolved existing ones"
        }

    def _recursive_conceptual_mapping(self) -> Dict[str, Any]:
        """Stage 7: Create recursive conceptual map"""
        concept_map = {
            'core_cluster': [],
            'central_ring': [],
            'peripheral_concepts': [],
            'meta_layer': []
        }
        for concept in self.concepts.values():
            connection_count = len(concept.connections)
            if connection_count > 3:
                concept_map['core_cluster'].append(concept.name)
            elif connection_count > 1:
                concept_map['central_ring'].append(concept.name)
            else:
                concept_map['peripheral_concepts'].append(concept.name)
        meta_concepts = [
            concept.name for concept in self.concepts.values()
            if 'analysis' in concept.name.lower() or 'meta' in concept.name.lower()
        ]
        concept_map['meta_layer'] = meta_concepts
        return {
            'stage': AnalysisStage.RECURSIVE_CONCEPTUAL_MAPPING,
            'concept_map': concept_map,
            'summary': "Created layered conceptual map with recursive elements"
        }

    def _ongoing_conceptual_refinement(self) -> Dict[str, Any]:
        """Stage 8: Document and implement refinements"""
        refinements = []
        for concept in self.concepts.values():
            if concept.evolution_history:
                refinements.append({
                    'concept': concept.name,
                    'changes': len(concept.evolution_history),
                    'latest_change': concept.evolution_history[-1]['change']
                })
        pre_evolution_markers = len([c for c in self.concepts.values() 
                                   if c.marker_type == MarkerType.ADAPTIVE_FRAMEWORK])
        self.evolve_markers()
        post_evolution_markers = len([c for c in self.concepts.values() 
                                    if c.marker_type == MarkerType.ADAPTIVE_FRAMEWORK])
        if post_evolution_markers > pre_evolution_markers:
            refinements.append({
                'type': 'marker_evolution',
                'change': f"Evolved {post_evolution_markers - pre_evolution_markers} concepts to Adaptive Framework markers",
                'reasoning': "Concepts with multiple connections demonstrate adaptive potential"
            })
        return {
            'stage': AnalysisStage.ONGOING_CONCEPTUAL_REFINEMENT,
            'refinements': refinements,
            'summary': f"Documented {len(refinements)} conceptual refinements"
        }

    def _meta_reflection(self) -> Dict[str, Any]:
        """Stage 9: Meta-reflection on framework performance"""
        total_concepts = len(self.concepts)
        total_connections = sum(len(c.connections) for c in self.concepts.values()) // 2
        marker_distribution = defaultdict(int)
        for concept in self.concepts.values():
            marker_distribution[concept.marker_type.value] += 1
        reflection = {
            'framework_metrics': {
                'total_concepts': total_concepts,
                'total_connections': total_connections,
                'connection_density': total_connections / max(total_concepts, 1),
                'marker_distribution': dict(marker_distribution)
            },
            'strengths': [
                "Maintains balance between structure and emergence",
                "Implements effective safety mechanisms",
                "Demonstrates recursive self-improvement"
            ],
            'areas_for_improvement': [
                "Could benefit from more sophisticated NLP integration",
                "Marker evolution could be more context-sensitive",
                "Recursive termination could be more adaptive"
            ],
            'alignment_assessment': "Framework successfully embodies emergent, recursive principles while maintaining cognitive safety"
        }
        return {
            'stage': AnalysisStage.META_REFLECTION,
            'reflection': reflection,
            'summary': "Completed meta-reflection on framework performance and evolution"
        }

    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from input text (simplified implementation)"""
        words = text.lower().split()
        phrases = []
        for i in range(len(words)):
            if len(words[i]) > 4:
                phrases.append(words[i])
            if i < len(words) - 1:
                phrase = f"{words[i]} {words[i+1]}"
                if len(phrase) > 8:
                    phrases.append(phrase)
        return list(set(phrases))[:10]

    def _find_relevant_excerpt(self, phrase: str, text: str) -> str:
        """Find relevant excerpt containing the phrase"""
        sentences = text.split('.')
        for sentence in sentences:
            if phrase.lower() in sentence.lower():
                return sentence.strip()
        return f"Context containing: {phrase}"

    def _determine_relationship(self, concept_1: ConceptNode, concept_2: ConceptNode) -> Optional[str]:
        """Determine relationship type between concepts (simplified)"""
        relationships = ["relates_to", "influences", "depends_on", "generates", "supports"]
        words_1 = set(concept_1.name.lower().split())
        words_2 = set(concept_2.name.lower().split())
        if words_1.intersection(words_2):
            return "relates_to"
        elif len(concept_1.connections) > len(concept_2.connections):
            return "influences"
        else:
            return "supports"

    def _detect_feedback_loops(self) -> List[List[str]]:
        """Detect potential feedback loops in concept network"""
        loops = []
        visited = set()
        def dfs(concept_id: str, path: List[str]) -> None:
            if concept_id in path:
                loop_start = path.index(concept_id)
                loop = path[loop_start:]
                if len(loop) > 2:
                    loops.append([self.concepts[cid].name for cid in loop])
                return
            if concept_id in visited or len(path) > 5:
                return
            visited.add(concept_id)
            for connected_id in self.concepts[concept_id].connections:
                dfs(connected_id, path + [concept_id])
        for concept_id in self.concepts:
            visited.clear()
            dfs(concept_id, [])
        return loops[:5]

    def _consolidate_concepts(self):
        """Consolidate similar concepts to prevent overflow"""
        consolidated = {}
        to_remove = []
        for concept_id, concept in self.concepts.items():
            similar_found = False
            for existing_id, existing_concept in consolidated.items():
                if self._concepts_similar(concept, existing_concept):
                    existing_concept.connections.extend(concept.connections)
                    existing_concept.connections = list(set(existing_concept.connections))
                    to_remove.append(concept_id)
                    similar_found = True
                    break
            if not similar_found:
                consolidated[concept_id] = concept
        for concept_id in to_remove:
            del self.concepts[concept_id]
        self._log_analysis(f"Consolidated {len(to_remove)} concepts")

    def _concepts_similar(self, concept_1: ConceptNode, concept_2: ConceptNode) -> bool:
        """Check if concepts are similar enough to consolidate"""
        words_1 = set(concept_1.name.lower().split())
        words_2 = set(concept_2.name.lower().split())
        intersection = words_1.intersection(words_2)
        union = words_1.union(words_2)
        similarity = len(intersection) / len(union) if union else 0
        return similarity > 0.5

    def _log_analysis(self, message: str):
        """Log analysis events for debugging and reflection"""
        self.analysis_history.append({
            'timestamp': time.time(),
            'message': message,
            'session_id': self.session_id,
            'recursion_depth': self.current_recursion_depth
        })

    def export_framework_state(self) -> Dict[str, Any]:
        """Export complete framework state for analysis"""
        return {
            'session_id': self.session_id,
            'concepts': {cid: {
                'name': c.name,
                'description': c.description,
                'marker_type': c.marker_type.value,
                'connections': c.connections,
                'metadata': c.metadata,
                'evolution_history': c.evolution_history
            } for cid, c in self.concepts.items()},
            'perspectives': {k: [p.__dict__ for p in v] for k, v in self.perspectives.items()},
            'cognitive_markers': [m.__dict__ for m in self.cognitive_markers],
            'assumption_challenges': [a.__dict__ for a in self.assumption_challenges],
            'analysis_history': self.analysis_history,
            'framework_metrics': {
                'total_concepts': len(self.concepts),
                'total_connections': sum(len(c.connections) for c in self.concepts.values()) // 2,
                'safety_threshold': self.safety_threshold,
                'max_recursion_depth': self.max_recursion_depth
            }
        }

class CognitiveFrameworkApplication:
    """Main application interface for the emergent cognitive framework"""
    def __init__(self):
        self.framework = EmergentConceptualFramework()
        self.session_active = False
    def start_session(self, user_input: str) -> Dict[str, Any]:
        """Initialize a new analysis session"""
        self.session_active = True
        self.framework = EmergentConceptualFramework()
        results = {
            'session_id': self.framework.session_id,
            'status': 'started',
            'initial_input': user_input,
            'stages_completed': []
        }
        return results
    def run_complete_analysis(self, user_input: str) -> Dict[str, Any]:
        """Execute all nine stages of analysis"""
        if not self.session_active:
            self.start_session(user_input)
        results = {
            'session_id': self.framework.session_id,
            'stages': {},
            'emergent_framework': None,
            'execution_summary': {}
        }
        for stage in AnalysisStage:
            try:
                stage_result = self.framework.perform_stage_analysis(stage, user_input)
                results['stages'][stage.name] = stage_result
                if stage_result.get('safety_exit'):
                    break
            except Exception as e:
                results['stages'][stage.name] = {
                    'error': str(e),
                    'stage': stage,
                    'safety_exit': True
                }
                break
        results['emergent_framework'] = self.framework.export_framework_state()
        if self.framework.analysis_history:
            session_start = min(h['timestamp'] for h in self.framework.analysis_history)
        else:
            session_start = time.time()
        results['execution_summary'] = {
            'stages_completed': len(results['stages']),
            'concepts_generated': len(self.framework.concepts),
            'connections_created': sum(len(c.connections) for c in self.framework.concepts.values()) // 2,
            'markers_evolved': len(self.framework.cognitive_markers),
            'session_duration': time.time() - session_start
        }
        return results
    def get_framework_visualization(self) -> Dict[str, Any]:
        """Generate visualization data for the conceptual framework"""
        nodes = []
        edges = []
        for concept_id, concept in self.framework.concepts.items():
            nodes.append({
                'id': concept_id,
                'name': concept.name,
                'marker_type': concept.marker_type.value,
                'connections_count': len(concept.connections),
                'description': concept.description[:100] + "..." if len(concept.description) > 100 else concept.description
            })
            for connected_id in concept.connections:
                if connected_id in self.framework.concepts:
                    edges.append({
                        'source': concept_id,
                        'target': connected_id,
                        'relationship': concept.metadata.get('relationships', {}).get(connected_id, 'relates_to')
                    })
        return {
            'nodes': nodes,
            'edges': edges,
            'metadata': {
                'total_nodes': len(nodes),
                'total_edges': len(edges),
                'marker_distribution': {
                    marker_type.value: len([n for n in nodes if n['marker_type'] == marker_type.value])
                    for marker_type in MarkerType
                }
            }
        }

if __name__ == "__main__":
    app = CognitiveFrameworkApplication()
    sample_input = """
I'm working on developing a recursive cognitive framework that can analyze 
complex problems through emergent pattern recognition. The system should 
balance structured analysis with creative exploration while maintaining 
safety mechanisms. I'm particularly interested in how technical implementation 
can mirror cognitive processes and how personal narrative can inform 
systematic approaches.
"""
    print("Starting Emergent Cognitive Framework Analysis...")
    results = app.run_complete_analysis(sample_input)
    print(f"\nAnalysis Complete!")
    print(f"Session ID: {results['session_id']}")
    print(f"Stages Completed: {results['execution_summary']['stages_completed']}/9")
    print(f"Concepts Generated: {results['execution_summary']['concepts_generated']}")
    print(f"Connections Created: {results['execution_summary']['connections_created']}")
    viz_data = app.get_framework_visualization()
    print(f"\nVisualization Data:")
    print(f"Nodes: {viz_data['metadata']['total_nodes']}")
    print(f"Edges: {viz_data['metadata']['total_edges']}")
    print(f"Marker Distribution: {viz_data['metadata']['marker_distribution']}")
    with open('framework_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)