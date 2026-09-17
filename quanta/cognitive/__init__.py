"""Quanta Cognitive Augmentation module for Antigravity AI Agents."""

from quanta.cognitive import (
    arbiter,
    consolidation,
    daemon,
    darwin_idle,
    decision_arbiter,
    memory,
    memory_manager,
    middleware,
    mind_wander,
    poisson_trigger,
    tom_analyzer,
)
from quanta.cognitive.arbiter import QuantumDecisionArbiter
from quanta.cognitive.consolidation import SubconsciousConsolidator
from quanta.cognitive.daemon import SubconsciousDaemon
from quanta.cognitive.darwin_idle import (
    QOS_CLASS_BACKGROUND,
    DarwinIdleMonitor,
    is_system_idle,
    set_background_qos,
)
from quanta.cognitive.memory import CognitiveMemoryManager, text_to_statevector
from quanta.cognitive.middleware import QuantaCognitiveMiddleware
from quanta.cognitive.mind_wander import DreamInsight, MindWanderEngine
from quanta.cognitive.poisson_trigger import PoissonSpindleTrigger
from quanta.cognitive.tom_analyzer import DreamSeed, TheoryOfMindAnalyzer

__all__ = [
    "CognitiveMemoryManager",
    "DarwinIdleMonitor",
    "DreamInsight",
    "DreamSeed",
    "MindWanderEngine",
    "PoissonSpindleTrigger",
    "QOS_CLASS_BACKGROUND",
    "QuantumDecisionArbiter",
    "QuantaCognitiveMiddleware",
    "SubconsciousConsolidator",
    "SubconsciousDaemon",
    "TheoryOfMindAnalyzer",
    "arbiter",
    "consolidation",
    "daemon",
    "darwin_idle",
    "decision_arbiter",
    "is_system_idle",
    "memory",
    "memory_manager",
    "middleware",
    "mind_wander",
    "poisson_trigger",
    "set_background_qos",
    "text_to_statevector",
    "tom_analyzer",
]
