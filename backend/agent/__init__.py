"""
PulmoSight CDSS Agentic Framework Prototype (`backend/agent/`).

Provides an LLM tool-calling orchestrator (Anthropic Messages API) wrapping the
validated PulmoSight prediction, SHAP explainability, CT extraction, progression,
and report generation modules under strict de-identification and safety guardrails.
"""

from backend.agent.orchestrator import (
    AgentDisabledError,
    AgentOrchestrator,
    is_agent_enabled,
)
from backend.agent.tools import (
    AGENT_TOOL_DEFINITIONS,
    ALLOWED_CLINICAL_INPUT_KEYS,
    FORBIDDEN_PHI_KEYS,
    sanitize_deidentified_inputs,
)

__all__ = [
    "AGENT_TOOL_DEFINITIONS",
    "ALLOWED_CLINICAL_INPUT_KEYS",
    "FORBIDDEN_PHI_KEYS",
    "AgentDisabledError",
    "AgentOrchestrator",
    "is_agent_enabled",
    "sanitize_deidentified_inputs",
]
