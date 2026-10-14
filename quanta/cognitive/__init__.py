"""Quanta Cognitive Augmentation module for CognitiveAgent AI Agents."""

from quanta.cognitive import (
    ablation,
    arbiter,
    blackboard,
    consolidation,
    daemon,
    darwin_idle,
    decision_arbiter,
    feedback,
    memory,
    memory_manager,
    middleware,
    mind_wander,
    noetherian,
    poisson_trigger,
    telemetry,
    tom_analyzer,
    vault_index,
    workspace_harvester,
)
from quanta.cognitive.ablation import AblationVariant, HardDecisionInstance, ZenoAblationEngine
from quanta.cognitive.arbiter import QuantumDecisionArbiter
from quanta.cognitive.blackboard import CognitiveBlackboard
from quanta.cognitive.consolidation import (
    GlymphaticClearanceResult,
    SubconsciousConsolidator,
)
from quanta.cognitive.daemon import SubconsciousDaemon
from quanta.cognitive.darwin_idle import (
    QOS_CLASS_BACKGROUND,
    DarwinIdleMonitor,
    get_user_idle_seconds,
    is_system_idle,
    set_background_qos,
)
from quanta.cognitive.feedback import (
    ActionOutcome,
    AdaptiveConfidenceEngine,
    CognitiveFeedbackEvaluator,
    CognitiveFeedbackLoop,
    OutcomeType,
    OutcomeVerifier,
    RuleDriftTracker,
    apply_confidence_feedback,
)
from quanta.cognitive.memory import CognitiveMemoryManager, text_to_statevector
from quanta.cognitive.middleware import QuantaCognitiveMiddleware
from quanta.cognitive.mind_wander import DreamInsight, MindWanderEngine
from quanta.cognitive.noetherian import (
    ConservedChargeDefinition,
    InvariantViolation,
    MizanHakemi,
    NoetherianEngineConfig,
    NoetherianEvaluation,
    NoetherianInvariantEngine,
)
from quanta.cognitive.poisson_trigger import PoissonSpindleTrigger
from quanta.cognitive.telemetry import record_outcome_telemetry
from quanta.cognitive.tom_analyzer import DreamSeed, TheoryOfMindAnalyzer
from quanta.cognitive.vault_index import LocalVaultIndex
from quanta.cognitive.workspace_harvester import WorkspaceContextHarvester

__all__ = [
    "AblationVariant",
    "ActionOutcome",
    "AdaptiveConfidenceEngine",
    "CognitiveBlackboard",
    "CognitiveFeedbackEvaluator",
    "CognitiveFeedbackLoop",
    "CognitiveMemoryManager",
    "ConservedChargeDefinition",
    "DarwinIdleMonitor",
    "DreamInsight",
    "DreamSeed",
    "GlymphaticClearanceResult",
    "HardDecisionInstance",
    "InvariantViolation",
    "LocalVaultIndex",
    "MindWanderEngine",
    "MizanHakemi",
    "NoetherianEngineConfig",
    "NoetherianEvaluation",
    "NoetherianInvariantEngine",
    "OutcomeType",
    "OutcomeVerifier",
    "PoissonSpindleTrigger",
    "QOS_CLASS_BACKGROUND",
    "QuantumDecisionArbiter",
    "QuantaCognitiveMiddleware",
    "RuleDriftTracker",
    "SubconsciousConsolidator",
    "SubconsciousDaemon",
    "TheoryOfMindAnalyzer",
    "WorkspaceContextHarvester",
    "ZenoAblationEngine",
    "ablation",
    "apply_confidence_feedback",
    "arbiter",
    "blackboard",
    "consolidation",
    "daemon",
    "darwin_idle",
    "decision_arbiter",
    "feedback",
    "get_user_idle_seconds",
    "is_system_idle",
    "memory",
    "memory_manager",
    "middleware",
    "mind_wander",
    "noetherian",
    "poisson_trigger",
    "record_outcome_telemetry",
    "set_background_qos",
    "telemetry",
    "text_to_statevector",
    "tom_analyzer",
    "vault_index",
    "workspace_harvester",
]
