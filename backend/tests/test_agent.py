"""
Unit Tests for the PulmoSight Agentic Framework Prototype (`backend/tests/test_agent.py`).

ZERO external API calls and ZERO `torch` dependency:
- All Anthropic LLM interactions are mocked deterministically.
- Verifies:
  1. `POST /agent/assess` is disabled (HTTP 503) when `ANTHROPIC_API_KEY` is unset.
  2. Strict de-identification rejects PHI fields (`patient_name`, `patient_id`, `mrn`) with HTTP 422
     before calling the LLM.
  3. Full multi-turn tool-calling orchestration invoking `predict_ipf`, `explain_shap`,
     `predict_progression`, and `generate_report` logs every call in `tool_call_log`,
     quotes exact tool numbers, both thresholds, limitations, and mandatory disclaimer.
  4. `human_review_required` becomes `True` whenever Screening and Confirmatory modes disagree.
  5. Post-generation safety guardrail activates when a misbehaving LLM tries to skip tools or
     omit thresholds/disclaimer.
"""

from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi.testclient import TestClient
import pytest

import backend.api.main as api_main
from backend.api.main import app, reset_rate_limiter
from backend.src.config import MANDATORY_DISCLAIMER

client = TestClient(app)


@pytest.fixture(autouse=True)
def cleanup_agent_state(monkeypatch):
    reset_rate_limiter()
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    api_main.STATE["agent_client_override"] = None
    yield
    api_main.STATE["agent_client_override"] = None


class MockAnthropicClient:
    """Deterministic multi-turn mock implementing `client.messages.create(...)`."""

    def __init__(self, scripted_turns: List[List[Dict[str, Any]]]) -> None:
        self._turns = list(scripted_turns)
        self.calls: List[Dict[str, Any]] = []
        self.messages = self

    def create(self, **kwargs) -> Any:
        self.calls.append(kwargs)
        if not self._turns:
            return SimpleNamespace(
                stop_reason="end_turn",
                content=[SimpleNamespace(type="text", text="Assessment complete.")],
            )
        blocks_spec = self._turns.pop(0)
        blocks = [SimpleNamespace(**spec) for spec in blocks_spec]
        has_tool = any(b.type == "tool_use" for b in blocks)
        return SimpleNamespace(
            stop_reason="tool_use" if has_tool else "end_turn",
            content=blocks,
        )


def test_agent_assess_disabled_without_api_key():
    """POST /agent/assess must return HTTP 503 when ANTHROPIC_API_KEY is unset."""
    payload = {
        "sex": "M",
        "age": 66.0,
        "height": 1.71,
        "weight": 90.0,
        "fev1fvc_abs": 0.83,
        "fev1_pp": 65.6,
        "fvc_pp": 60.5,
        "tlc_pp": 64.2,
        "tlco_pp": 40.1,
    }
    res = client.post("/agent/assess", json=payload)
    assert res.status_code == 503
    assert "ANTHROPIC_API_KEY" in res.json()["detail"]


def test_agent_assess_rejects_phi_identifiers_before_llm_call():
    """PHI fields (name, patient_id, mrn) must be blocked with HTTP 422 before calling the LLM."""
    mock_client = MockAnthropicClient(scripted_turns=[])
    api_main.STATE["agent_client_override"] = mock_client

    payload_with_phi = {
        "patient_name": "John Doe",
        "patient_id": "MRN-998877",
        "sex": "M",
        "age": 66.0,
        "height": 1.71,
        "weight": 90.0,
        "fev1fvc_abs": 0.83,
        "fev1_pp": 65.6,
        "fvc_pp": 60.5,
        "tlc_pp": 64.2,
        "tlco_pp": 40.1,
    }
    res = client.post("/agent/assess", json=payload_with_phi)
    assert res.status_code == 422
    assert "de-identification" in res.json()["detail"].lower()
    assert len(mock_client.calls) == 0  # LLM was NEVER called!


def test_agent_orchestrator_tool_calling_log_and_deidentification():
    """Verify full mocked multi-turn tool loop, audit log, and zero PHI transmitted."""
    api_main.ensure_models_loaded()
    clin_payload = {
        "sex": "M",
        "age": 66.0,
        "height": 1.71,
        "weight": 90.0,
        "fev1fvc_abs": 0.83,
        "fev1_pp": 65.6,
        "fvc_pp": 60.5,
        "tlc_pp": 64.2,
        "tlco_pp": 40.1,
    }

    mock_client = MockAnthropicClient(
        scripted_turns=[
            [
                {
                    "type": "tool_use",
                    "id": "tu_1",
                    "name": "predict_ipf",
                    "input": clin_payload,
                },
                {
                    "type": "tool_use",
                    "id": "tu_2",
                    "name": "explain_shap",
                    "input": clin_payload,
                },
            ],
            [
                {
                    "type": "tool_use",
                    "id": "tu_3",
                    "name": "predict_progression",
                    "input": {"age": 66.0, "sex": "M", "fvc_pp": 60.5},
                },
                {
                    "type": "tool_use",
                    "id": "tu_4",
                    "name": "generate_report",
                    "input": {"include_exploratory_progression": True},
                },
            ],
            [
                {
                    "type": "text",
                    "text": (
                        "Tool summary completed. Note exploratory OSIC progression limitations "
                        "and clinical decision support scope."
                    ),
                }
            ],
        ]
    )
    api_main.STATE["agent_client_override"] = mock_client

    res = client.post("/agent/assess", json=clin_payload)
    assert res.status_code == 200
    data = res.json()

    called_tools = [entry["tool_name"] for entry in data["tool_call_log"]]
    assert called_tools == ["predict_ipf", "explain_shap", "predict_progression", "generate_report"]
    assert all(entry["status"] == "ok" for entry in data["tool_call_log"])

    # Guardrail must guarantee exact numbers, both thresholds, limitations, and mandatory disclaimer appear
    ipf_out = data["tool_outputs"]["predict_ipf"]
    assert str(ipf_out["screening_threshold"]) in data["final_summary"]
    assert str(ipf_out["confirmatory_threshold"]) in data["final_summary"]
    assert MANDATORY_DISCLAIMER in data["final_summary"]
    assert data["disclaimer"] == MANDATORY_DISCLAIMER

    # Progression without CT series must return hrct_only=None and fusion=None
    prog_out = data["tool_outputs"]["predict_progression"]
    assert prog_out["hrct_only"] is None
    assert prog_out["fusion"] is None
    assert prog_out["modality_used"] == {"clinical": True, "hrct": False, "fusion": False}

    # Verify sent LLM prompt contains ONLY de-identified inputs
    first_prompt = mock_client.calls[0]["messages"][0]["content"]
    assert "patient_name" not in first_prompt.lower()
    assert "mrn" not in first_prompt.lower()


def test_agent_flags_human_review_when_screening_and_confirmatory_disagree(monkeypatch):
    """
    When probability lies strictly between screening_threshold and confirmatory_threshold,
    Screening predicts 'IPF' while Confirmatory predicts 'Non-IPF' -> `human_review_required=True`.
    """
    api_main.ensure_models_loaded()
    bundle = api_main.STATE["clinical_bundle"]
    screen_t = float(bundle["optimal_threshold"])
    confirm_t = float(bundle["confirmatory_threshold"])
    mid_prob = round((screen_t + confirm_t) / 2.0, 4)

    class _MidProbaClf:
        def predict_proba(self, X):
            import numpy as np

            return np.array([[1.0 - mid_prob, mid_prob]])

    monkeypatch.setitem(bundle, "calibrated_classifier", _MidProbaClf())

    clin_payload = {
        "sex": "M",
        "age": 64.0,
        "height": 1.72,
        "weight": 82.0,
        "fev1fvc_abs": 0.80,
        "fev1_pp": 74.0,
        "fvc_pp": 72.0,
        "tlc_pp": 76.0,
        "tlco_pp": 58.0,
    }
    mock_client = MockAnthropicClient(
        scripted_turns=[
            [
                {
                    "type": "tool_use",
                    "id": "tu_1",
                    "name": "predict_ipf",
                    "input": clin_payload,
                }
            ],
            [
                {
                    "type": "text",
                    "text": (
                        f"Calibrated IPF probability is {mid_prob:.4f}. Screening threshold "
                        f"{screen_t:.4f} flags IPF while Confirmatory threshold {confirm_t:.4f} "
                        f"flags Non-IPF; exploratory limitations apply. {MANDATORY_DISCLAIMER}"
                    ),
                }
            ],
        ]
    )
    api_main.STATE["agent_client_override"] = mock_client

    res = client.post("/agent/assess", json=clin_payload)
    assert res.status_code == 200
    data = res.json()
    assert data["modes_disagree"] is True
    assert data["human_review_required"] is True
    assert any("disagree" in r.lower() for r in data["human_review_reasons"])


def test_agent_guardrail_catches_llm_skipping_tools():
    """If an LLM attempts to answer without calling predict_ipf, guardrail executes it and flags review."""
    mock_client = MockAnthropicClient(
        scripted_turns=[
            [{"type": "text", "text": "I think this patient has IPF."}],
        ]
    )
    api_main.STATE["agent_client_override"] = mock_client

    clin_payload = {
        "sex": "M",
        "age": 66.0,
        "height": 1.71,
        "weight": 90.0,
        "fev1fvc_abs": 0.83,
        "fev1_pp": 65.6,
        "fvc_pp": 60.5,
        "tlc_pp": 64.2,
        "tlco_pp": 40.1,
    }
    res = client.post("/agent/assess", json=clin_payload)
    assert res.status_code == 200
    data = res.json()
    assert data["human_review_required"] is True
    assert data["guardrail_appendix_added"] is True
    assert MANDATORY_DISCLAIMER in data["final_summary"]
    assert any(entry["tool_name"] == "predict_ipf" for entry in data["tool_call_log"])
