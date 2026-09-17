"""Quanta Cognitive Augmentation module for Antigravity AI Agents."""

from quanta.cognitive import arbiter, decision_arbiter, memory, memory_manager, middleware
from quanta.cognitive.arbiter import QuantumDecisionArbiter
from quanta.cognitive.memory import CognitiveMemoryManager, text_to_statevector
from quanta.cognitive.middleware import QuantaCognitiveMiddleware

__all__ = [
    "CognitiveMemoryManager",
    "QuantumDecisionArbiter",
    "QuantaCognitiveMiddleware",
    "text_to_statevector",
    "arbiter",
    "decision_arbiter",
    "memory",
    "memory_manager",
    "middleware",
]
