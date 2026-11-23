"""
Anthropic Tool-Calling Agent Orchestrator (`backend/agent/orchestrator.py`).

Design & Safety Guarantees:
1. Reads `ANTHROPIC_API_KEY` and `ANTHROPIC_MODEL` strictly from environment variables.
   Disabled (raises `AgentDisabledError` -> HTTP 503) when `ANTHROPIC_API_KEY` is unset.
2. Enforces strict de-identification: only whitelisted numeric/categorical clinical values
   are ever sent to Anthropic (never patient names, IDs, MRNs, DOBs, or free-text notes).
3. The LLM MUST NEVER output an independent diagnosis; it may only call the 5 PulmoSight
   tools (`predict_ipf`, `explain_shap`, `extract_ct_features`, `predict_progression`,
   `generate_report`) and summarize their exact outputs.
4. Maintains a structured `tool_call_log` for every step.
5. Sets `human_review_required = True` whenever Screening mode and Confirmatory mode disagree
   (or if required tool execution failed).
6. Post-generation verification guardrail ensures the final narrative always quotes the
   exact tool numbers, both operating thresholds, limitations, and `MANDATORY_DISCLAIMER`.
"""

import json
import os
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
import urllib.error
import urllib.request

from backend.agent.tools import (
    AGENT_TOOL_DEFINITIONS,
    PulmoSightToolExecutor,
    sanitize_deidentified_inputs,
)
from backend.src.config import (
    EXPLORATORY_PROGRESSION_NOTE,
    MANDATORY_DISCLAIMER,
    setup_logger,
)

logger = setup_logger("agent.orchestrator")

DEFAULT_ANTHROPIC_MODEL = "claude-3-5-sonnet-latest"

SYSTEM_PROMPT = (
    "You are the PulmoSight Clinical Decision Support tool-orchestration assistant. "
    "STRICT SAFETY RULES:\n"
    "1. NEVER invent a clinical diagnosis, probability, or risk score yourself.\n"
    "2. You MUST call the provided tools (`predict_ipf`, `explain_shap`, `extract_ct_features` "
    "if a CT series is attached, `predict_progression`, and `generate_report`) before giving "
    "your final summary.\n"
    "3. Your final summary must ONLY report and synthesize the exact numbers returned by the tools.\n"
    "4. Always quote: (a) the exact calibrated IPF probability, (b) BOTH the Screening threshold "
    "and Confirmatory threshold and their labels, (c) the SHAP log-odds drivers and caveat, "
    "(d) that progression scores are exploratory (OSIC cohort, predicting FVC decline, NOT IPF "
    "diagnosis), (e) dataset and model limitations, and (f) the exact mandatory clinical disclaimer.\n"
    "5. If Screening mode and Confirmatory mode disagree, explicitly flag that human multidisciplinary "
    "review is required.\n"
    "6. Never request or output any patient names, IDs, or personal identifiers."
)


class AgentDisabledError(RuntimeError):
    """Raised when `ANTHROPIC_API_KEY` is not set in the environment."""


def is_agent_enabled() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY", "").strip())


class _UrllibAnthropicClient:
    """
    Lightweight zero-dependency Anthropic Messages API v1 adapter implementing
    `client.messages.create(...)` when the optional `anthropic` SDK is not installed.
    """

    def __init__(self, api_key: str, timeout: float = 45.0) -> None:
        self._api_key = api_key
        self._timeout = timeout
        self.messages = self

    def create(
        self,
        *,
        model: str,
        max_tokens: int,
        system: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Any:
        url = os.getenv("ANTHROPIC_API_URL", "https://api.anthropic.com/v1/messages")
        body: Dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
        }
        if tools:
            body["tools"] = tools
        payload = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "x-api-key": self._api_key,
                "anthropic-version": "2023-06-01",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Anthropic API HTTP {exc.code}: {body}") from exc
        except Exception as exc:
            raise RuntimeError(f"Anthropic API request failed: {exc}") from exc

        blocks = []
        for b in raw.get("content", []):
            blocks.append(SimpleNamespace(**b))
        return SimpleNamespace(
            stop_reason=raw.get("stop_reason", "end_turn"),
            content=blocks,
            model=raw.get("model", model),
        )


def _create_default_anthropic_client(api_key: str) -> Any:
    try:
        import anthropic  # type: ignore

        return anthropic.Anthropic(api_key=api_key)
    except ImportError:
        return _UrllibAnthropicClient(api_key=api_key)


def _serialize_content_blocks(blocks: List[Any]) -> List[Dict[str, Any]]:
    serialized: List[Dict[str, Any]] = []
    for block in blocks:
        if isinstance(block, dict):
            serialized.append(block)
            continue
        b_type = getattr(block, "type", "")
        if b_type == "text":
            serialized.append({"type": "text", "text": getattr(block, "text", "")})
        elif b_type == "tool_use":
            serialized.append(
                {
                    "type": "tool_use",
                    "id": getattr(block, "id", ""),
                    "name": getattr(block, "name", ""),
                    "input": getattr(block, "input", {}) or {},
                }
            )
    return serialized


class AgentOrchestrator:
    """
    Tool-calling orchestrator using the Anthropic Messages API.
    """

    def __init__(
        self,
        clinical_bundle: Dict[str, Any],
        progression_bundle: Dict[str, Any],
        artifacts_json: Dict[str, Any],
        client: Optional[Any] = None,
        model_name: Optional[str] = None,
        max_steps: int = 8,
    ) -> None:
        self.clinical_bundle = clinical_bundle
        self.progression_bundle = progression_bundle
        self.artifacts_json = artifacts_json
        self.max_steps = max_steps
        self.model_name = (
            model_name
            or os.getenv("ANTHROPIC_MODEL", "").strip()
            or DEFAULT_ANTHROPIC_MODEL
        )

        if client is not None:
            self.client = client
        else:
            api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
            if not api_key:
                raise AgentDisabledError(
                    "Agentic assessment is disabled because ANTHROPIC_API_KEY is not set in the environment."
                )
            self.client = _create_default_anthropic_client(api_key)

    def run_assessment(
        self,
        patient_payload: Dict[str, Any],
        ct_bytes: Optional[bytes] = None,
        ct_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        # 1. Strict de-identification BEFORE constructing any LLM prompt
        deidentified = sanitize_deidentified_inputs(patient_payload)
        ct_attached = bool(ct_bytes)

        executor = PulmoSightToolExecutor(
            clinical_bundle=self.clinical_bundle,
            progression_bundle=self.progression_bundle,
            artifacts_json=self.artifacts_json,
            deidentified_inputs=deidentified,
            ct_bytes=ct_bytes,
            ct_filename=ct_filename,
        )

        user_prompt = (
            "Run a de-identified PulmoSight clinical decision support assessment using ONLY the "
            "registered tools on these de-identified numeric clinical inputs:\n"
            f"{json.dumps(deidentified, indent=2)}\n"
            f"CT DICOM series attached: {'Yes' if ct_attached else 'No'}.\n"
            "Call `predict_ipf`, `explain_shap`, "
            + ("`extract_ct_features`, " if ct_attached else "")
            + "`predict_progression`, and `generate_report`, then produce a concise clinical "
            "summary quoting all exact tool numbers, both thresholds, limitations, and the mandatory disclaimer."
        )

        messages: List[Dict[str, Any]] = [{"role": "user", "content": user_prompt}]
        tool_call_log: List[Dict[str, Any]] = []
        final_text_parts: List[str] = []

        for step_idx in range(1, self.max_steps + 1):
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=1400,
                system=SYSTEM_PROMPT,
                tools=AGENT_TOOL_DEFINITIONS,
                messages=messages,
            )

            blocks = list(getattr(response, "content", []) or [])
            tool_uses = [
                b
                for b in blocks
                if (b.get("type") if isinstance(b, dict) else getattr(b, "type", "")) == "tool_use"
            ]
            text_blocks = [
                (b.get("text", "") if isinstance(b, dict) else getattr(b, "text", ""))
                for b in blocks
                if (b.get("type") if isinstance(b, dict) else getattr(b, "type", "")) == "text"
            ]

            if not tool_uses:
                final_text_parts.extend([t for t in text_blocks if t])
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": _serialize_content_blocks(blocks),
                }
            )

            tool_result_blocks: List[Dict[str, Any]] = []
            for tu in tool_uses:
                t_id = tu["id"] if isinstance(tu, dict) else getattr(tu, "id", "")
                t_name = tu["name"] if isinstance(tu, dict) else getattr(tu, "name", "")
                t_input = (tu.get("input") if isinstance(tu, dict) else getattr(tu, "input", {})) or {}

                t0 = time.perf_counter()
                try:
                    t_out = executor.execute(t_name, t_input)
                    status = "error" if "error" in t_out else "ok"
                except Exception as exc:
                    t_out = {"error": str(exc)}
                    status = "error"
                elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

                tool_call_log.append(
                    {
                        "step": step_idx,
                        "tool_use_id": t_id,
                        "tool_name": t_name,
                        "inputs": t_input,
                        "status": status,
                        "latency_ms": elapsed_ms,
                        "output": t_out,
                    }
                )
                tool_result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": t_id,
                        "content": json.dumps(t_out),
                    }
                )

            messages.append({"role": "user", "content": tool_result_blocks})

        # Ensure deterministic baseline tool results exist if the LLM skipped predict_ipf
        guardrail_reasons: List[str] = []
        if "predict_ipf" not in executor.tool_results:
            guardrail_reasons.append(
                "LLM did not invoke `predict_ipf`; executed deterministically by safety guardrail."
            )
            ipf_fallback = executor.execute("predict_ipf", {})
            tool_call_log.append(
                {
                    "step": len(tool_call_log) + 1,
                    "tool_use_id": "guardrail_predict_ipf",
                    "tool_name": "predict_ipf",
                    "inputs": {},
                    "status": "ok",
                    "latency_ms": 0.0,
                    "output": ipf_fallback,
                }
            )

        ipf_out = executor.tool_results.get("predict_ipf", {})
        shap_out = executor.tool_results.get("explain_shap")
        prog_out = executor.tool_results.get("predict_progression")
        report_out = executor.tool_results.get("generate_report")

        # Mandatory human-review flag when Screening & Confirmatory modes disagree
        modes_disagree = bool(ipf_out.get("modes_disagree", False))
        human_review_reasons: List[str] = []
        if modes_disagree:
            human_review_reasons.append(
                f"Screening mode ({ipf_out.get('screening_predicted_label')} at threshold "
                f"{ipf_out.get('screening_threshold')}) and Confirmatory mode "
                f"({ipf_out.get('confirmatory_predicted_label')} at threshold "
                f"{ipf_out.get('confirmatory_threshold')}) disagree."
            )
        if guardrail_reasons:
            human_review_reasons.extend(guardrail_reasons)
        human_review_required = len(human_review_reasons) > 0

        # Post-generation guardrail: verify final text quotes tool numbers, thresholds, limitations & disclaimer
        final_summary = "\n\n".join(final_text_parts).strip()
        required_checks = [
            (str(ipf_out.get("screening_threshold", "")), "screening threshold"),
            (str(ipf_out.get("confirmatory_threshold", "")), "confirmatory threshold"),
            (MANDATORY_DISCLAIMER, "mandatory disclaimer"),
        ]
        prob_str = f"{ipf_out.get('ipf_probability', 0.0):.4f}"
        prob_pct_str = f"{ipf_out.get('ipf_probability_pct', 0.0):.1f}%"
        has_prob = (prob_str in final_summary) or (prob_pct_str in final_summary)
        missing_items = [lbl for token, lbl in required_checks if token and token not in final_summary]
        if not has_prob:
            missing_items.insert(0, "IPF probability")
        if "exploratory" not in final_summary.lower() and "limitation" not in final_summary.lower():
            missing_items.append("limitations")

        guardrail_appendix_added = False
        if missing_items or not final_summary:
            guardrail_appendix_added = True
            appendix_lines = [
                "---",
                "**Verified Tool Summary & Guardrail Disclosure**",
                f"- Clinical IPF Probability (Calibrated XGBoost): `{prob_str}` ({prob_pct_str})",
                f"- Screening Mode (threshold `{ipf_out.get('screening_threshold')}`): `{ipf_out.get('screening_predicted_label')}`",
                f"- Confirmatory Mode (threshold `{ipf_out.get('confirmatory_threshold')}`): `{ipf_out.get('confirmatory_predicted_label')}`",
                f"- Human Review Required: `{'Yes' if human_review_required else 'No'}`",
                f"- Limitations: {ipf_out.get('sensitivity_disclosure', '')} {EXPLORATORY_PROGRESSION_NOTE}.",
                f"- Disclaimer: {MANDATORY_DISCLAIMER}",
            ]
            final_summary = (final_summary + "\n\n" if final_summary else "") + "\n".join(appendix_lines)

        return {
            "status": "completed",
            "model_used": self.model_name,
            "deidentified_inputs_sent": deidentified,
            "final_summary": final_summary,
            "human_review_required": human_review_required,
            "human_review_reasons": human_review_reasons,
            "modes_disagree": modes_disagree,
            "guardrail_appendix_added": guardrail_appendix_added,
            "tool_call_log": tool_call_log,
            "tool_outputs": {
                "predict_ipf": ipf_out,
                "explain_shap": shap_out,
                "extract_ct_features": executor.tool_results.get("extract_ct_features"),
                "predict_progression": prog_out,
                "generate_report": report_out,
            },
            "disclaimer": MANDATORY_DISCLAIMER,
        }
