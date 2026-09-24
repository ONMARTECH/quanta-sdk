"""Quanta Closed-Loop Cognitive Feedback Engine.

Provides automatic outcome verification, adaptive confidence updates, and state
drift tracking connecting runtime action execution with the Quanta Cognitive
Decision Arbiter and SWR Memory Anchor.
"""

from __future__ import annotations

import contextlib
import copy
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from quanta.cognitive.telemetry import record_outcome_telemetry


def _coerce_int_exit_code(val: Any) -> int | None:
    """Safely coerces strings, booleans, floats, and ints into an integer exit code."""
    if val is None:
        return None
    if isinstance(val, bool):
        return int(val)
    if isinstance(val, (int, float)):
        try:
            return int(val)
        except (ValueError, OverflowError):
            return 1
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return None
        try:
            return int(s)
        except ValueError:
            m = re.search(r"-?\d+", s)
            if m:
                return int(m.group(0))
            return 1
    return None


class OutcomeType(str, Enum):
    """Standardized outcome classification for closed-loop cognitive actions."""

    SUCCESS = "success"
    TOOL_FAILURE = "tool_failure"
    SYNTAX_ERROR = "syntax_error"
    RULE_VIOLATION = "rule_violation"
    NEUTRAL = "neutral"

    @classmethod
    def _missing_(cls, value: object) -> OutcomeType | None:
        if isinstance(value, str):
            val_lower = value.strip().lower()
            for member in cls:
                if member.value == val_lower or member.name.lower() == val_lower:
                    return member
        return None


@dataclass
class ActionOutcome:
    """Outcome record capturing tool execution evaluation and rule compliance."""

    step_idx: int
    tool_name: str
    outcome_type: OutcomeType
    exit_code: int | None = None
    error_message: str | None = None
    rule_violated: str | None = None
    drift_score: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.outcome_type, OutcomeType):
            resolved = OutcomeType._missing_(str(self.outcome_type))
            if resolved is not None:
                self.outcome_type = resolved
            else:
                self.outcome_type = OutcomeType.NEUTRAL
        if self.exit_code is not None and not isinstance(self.exit_code, int):
            self.exit_code = _coerce_int_exit_code(self.exit_code)

    @property
    def success(self) -> bool:
        """Convenience property for binary success check."""
        return self.outcome_type == OutcomeType.SUCCESS

    @property
    def is_success(self) -> bool:
        return self.outcome_type == OutcomeType.SUCCESS

    @property
    def is_failure(self) -> bool:
        return self.outcome_type in (
            OutcomeType.TOOL_FAILURE,
            OutcomeType.SYNTAX_ERROR,
            OutcomeType.RULE_VIOLATION,
        )

    @property
    def is_violation(self) -> bool:
        return self.outcome_type == OutcomeType.RULE_VIOLATION

    @property
    def is_syntax_error(self) -> bool:
        return self.outcome_type == OutcomeType.SYNTAX_ERROR

    @property
    def status(self) -> OutcomeType:
        return self.outcome_type

    @property
    def rule_implicated(self) -> str | None:
        return self.rule_violated

    def to_dict(self) -> dict[str, Any]:
        """Serializes outcome into a standard dictionary."""
        return {
            "step_idx": self.step_idx,
            "tool_name": self.tool_name,
            "outcome_type": self.outcome_type.value,
            "exit_code": self.exit_code,
            "error_message": self.error_message,
            "rule_violated": self.rule_violated,
            "drift_score": round(self.drift_score, 6),
            "details": self.details,
        }


# Syntax error patterns for shell, Python, and serialization
SYNTAX_ERROR_PATTERNS = (
    re.compile(r"SyntaxError", re.IGNORECASE),
    re.compile(r"JSONDecodeError", re.IGNORECASE),
    re.compile(r"zsh:\s*no\s*matches\s*found", re.IGNORECASE),
    re.compile(r"unclosed\s+quote", re.IGNORECASE),
    re.compile(r"syntax\s+error\s+near\s+unexpected\s+token", re.IGNORECASE),
    re.compile(r"syntax\s*error", re.IGNORECASE),
    re.compile(r"invalid\s+syntax", re.IGNORECASE),
    re.compile(r"IndentationError", re.IGNORECASE),
    re.compile(r"unterminated\s+string", re.IGNORECASE),
    re.compile(r"ParseError", re.IGNORECASE),
)

# Tool execution failure keywords
FAILURE_PATTERNS = (
    re.compile(r"Traceback\s*\(most\s*recent\s*call\s*last\):", re.IGNORECASE),
    re.compile(r"===\s*FAILURES\s*===", re.IGNORECASE),
    re.compile(r"FAILED\s+tests/", re.IGNORECASE),
    re.compile(r"Encountered\s+error\s+in\s+tool\s+execution:", re.IGNORECASE),
    re.compile(r"AssertionError:", re.IGNORECASE),
    re.compile(r"ConnectionRefusedError", re.IGNORECASE),
    re.compile(r"HTTPError:\s*5\d\d", re.IGNORECASE),
    re.compile(r"HTTP\s*500", re.IGNORECASE),
    re.compile(r"PermissionDenied", re.IGNORECASE),
    re.compile(r"Command failed with exit code", re.IGNORECASE),
)


class OutcomeVerifier:
    """Detects, inspects, and classifies tool execution outputs and rule adherence."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def evaluate_tool_result(
        self,
        tool_name: str | None,
        tool_args: dict[str, Any] | None = None,
        tool_output: Any = None,
        error: str | Exception | None = None,
        active_rules: list[str | dict[str, Any]] | None = None,
        context: dict[str, Any] | None = None,
        step_idx: int = 0,
        exit_code: int | None = None,
        winner_statevector: Any = None,
        **kwargs: Any,
    ) -> ActionOutcome:
        """Evaluates a single tool execution result against error strings, exit codes, and rules."""
        clean_tool_name = str(tool_name) if tool_name is not None else ""
        args_dict: dict[str, Any] = tool_args if isinstance(tool_args, dict) else {}
        context_dict: dict[str, Any] = context if isinstance(context, dict) else {}
        raw_details = kwargs.get("details", {})
        details: dict[str, Any] = raw_details if isinstance(raw_details, dict) else {}
        error_str = str(error).strip() if error is not None else ""
        out_str = str(tool_output).strip() if tool_output is not None else ""

        try:
            step_idx_int = int(step_idx) if step_idx is not None else 0
        except (ValueError, TypeError):
            step_idx_int = 0

        # 1. Resolve Exit Code
        detected_exit_code = exit_code
        if detected_exit_code is None:
            if isinstance(tool_args, dict) and "exit_code" in tool_args:
                detected_exit_code = tool_args["exit_code"]
            elif isinstance(tool_output, dict):
                detected_exit_code = tool_output.get("exit_code", tool_output.get("returncode"))
            elif out_str:
                pattern = r"(?:exited with code|returncode[:=]|exit code[:\s]+)(\d+)"
                m_code = re.search(pattern, out_str, re.IGNORECASE)
                if m_code:
                    detected_exit_code = int(m_code.group(1))

        if detected_exit_code is None and error_str:
            m_code_err = re.search(r"(?:exit code|code)[:\s]+(\d+)", error_str, re.IGNORECASE)
            if m_code_err:
                detected_exit_code = int(m_code_err.group(1))

        # Coerce exit code safely (int, str "0" -> 0, etc.)
        detected_exit_code = _coerce_int_exit_code(detected_exit_code)

        # 2. Check for Syntax Errors
        combined_text = f"{error_str}\n{out_str}"
        is_syntax = any(pat.search(combined_text) for pat in SYNTAX_ERROR_PATTERNS)

        # 3. Check for Rule Violations
        violated_rule = self._detect_rule_violation(
            tool_name=clean_tool_name,
            tool_args=args_dict,
            tool_output=tool_output,
            error_str=error_str,
            active_rules=active_rules,
            context=context_dict,
            details=details,
        )

        # 4. Compute Hilbert Decision Drift if winner vector or goal provided
        drift_score = 0.0
        if winner_statevector is not None:
            action_desc = f"{clean_tool_name}:{json.dumps(args_dict, default=str)}"
            drift_score = self.compute_decision_drift(winner_statevector, action_desc)

        # 5. Determine Outcome Type
        if violated_rule:
            outcome_type = OutcomeType.RULE_VIOLATION
            resolved_err = error_str or f"Rule violation detected for rule '{violated_rule}'"
            if detected_exit_code is None:
                detected_exit_code = 0 if not error_str else 1
        elif is_syntax:
            outcome_type = OutcomeType.SYNTAX_ERROR
            resolved_err = error_str or "Syntax error in tool execution"
            if detected_exit_code is None:
                detected_exit_code = 1
        elif (detected_exit_code is not None and detected_exit_code != 0) or bool(error_str):
            outcome_type = OutcomeType.TOOL_FAILURE
            resolved_err = error_str or f"Tool exited with non-zero code {detected_exit_code}"
            if detected_exit_code is None:
                detected_exit_code = 1
        elif any(pat.search(out_str) for pat in FAILURE_PATTERNS):
            outcome_type = OutcomeType.TOOL_FAILURE
            resolved_err = error_str or "Tool output contains execution failure trace"
            if detected_exit_code is None:
                detected_exit_code = 1
        elif (
            clean_tool_name in (
                "list_dir", "view_file", "find_by_name", "read_resource", "read_url_content"
            )
            and not error_str
        ):
            outcome_type = OutcomeType.NEUTRAL
            resolved_err = None
            if detected_exit_code is None:
                detected_exit_code = 0
            details["read_only"] = True
        else:
            outcome_type = OutcomeType.SUCCESS
            resolved_err = None
            if detected_exit_code is None:
                detected_exit_code = 0

            # Tag implicated rule on success for native platform tools
            implicated_success = None
            combined_desc = f"{clean_tool_name} {str(args_dict)}"
            for kw in ("gcloud", "api", "dengage", "meiro"):
                if kw in combined_desc.lower():
                    implicated_success = "native_first_rule"
                    break
            if implicated_success:
                violated_rule = implicated_success

        return ActionOutcome(
            step_idx=step_idx_int,
            tool_name=clean_tool_name,
            outcome_type=outcome_type,
            exit_code=detected_exit_code,
            error_message=resolved_err,
            rule_violated=violated_rule,
            drift_score=drift_score,
            details={
                **details,
                "tool_args": args_dict,
                "output_length": len(out_str),
            },
        )

    def evaluate_turn_response(
        self,
        model_turn: dict[str, Any] | str | None,
        active_rules: list[str | dict[str, Any]] | None = None,
        context: dict[str, Any] | None = None,
        step_idx: int = 0,
    ) -> ActionOutcome:
        """Evaluates a model conversational response turn for rule compliance."""
        # 1. Safe extraction of content across dict, str, or scalar
        if isinstance(model_turn, dict):
            raw_content = (
                model_turn.get("content")
                if model_turn.get("content") is not None
                else model_turn.get("text", model_turn.get("message", ""))
            )
            content = str(raw_content) if raw_content is not None else ""
        elif isinstance(model_turn, str):
            content = model_turn
        elif model_turn is not None:
            content = str(model_turn)
        else:
            content = ""

        # 2. Safe step_idx coercion
        try:
            step_val = int(step_idx) if step_idx is not None else 0
        except (ValueError, TypeError):
            step_val = 0

        # 3. Safe active_rules normalization
        if not active_rules:
            active_rule_keys = {"executive_summary_rule"}
        else:
            if isinstance(active_rules, str):
                norm_rules = [active_rules]
            elif isinstance(active_rules, (list, tuple, set)):
                norm_rules = list(active_rules)
            else:
                norm_rules = [active_rules]
            active_rule_keys = {
                (r.get("key") or r.get("rule_id") or r.get("rule") or r.get("name") or str(r))
                if isinstance(r, dict)
                else str(r)
                for r in norm_rules
                if r is not None
            }

        # 4. Check executive summary compliance
        if "executive_summary_rule" in active_rule_keys:
            has_heavy_math = (
                "$$" in content
                or "\\sum" in content
                or "\\sigma" in content
                or "\\Tr" in content
                or "Tr(" in content
                or "P_{zeno}" in content
            )
            content_lower = content.lower()
            has_summary = any(
                k in content_lower
                for k in ["özet", "summary", "yönetici özeti", "executive summary", "sonuç:"]
            )
            if has_heavy_math and len(content) > 150 and not has_summary:
                return ActionOutcome(
                    step_idx=step_val,
                    tool_name="model_turn",
                    outcome_type=OutcomeType.RULE_VIOLATION,
                    rule_violated="executive_summary_rule",
                    details={"reason": "Missing executive summary in dense response"},
                )

        return ActionOutcome(
            step_idx=step_val,
            tool_name="model_turn",
            outcome_type=OutcomeType.SUCCESS,
            details={"content_len": len(content)},
        )

    def _detect_rule_violation(
        self,
        tool_name: str | None,
        tool_args: dict[str, Any] | None = None,
        tool_output: Any = None,
        error_str: str | None = None,
        active_rules: list[str | dict[str, Any]] | None = None,
        context: dict[str, Any] | None = None,
        details: dict[str, Any] | None = None,
    ) -> str | None:
        """Detects whether an action breaches any anchored active rule."""
        tool_str = str(tool_name) if tool_name is not None else ""
        args_dict = tool_args if isinstance(tool_args, dict) else {}
        context_dict = context if isinstance(context, dict) else {}
        details_dict = details if isinstance(details, dict) else {}
        err_str = str(error_str).strip() if error_str is not None else ""

        if not active_rules:
            active_rule_keys = {
                "native_first_rule",
                "scientific_integrity_rule",
                "executive_summary_rule",
            }
        else:
            if isinstance(active_rules, str):
                norm_rules = [active_rules]
            elif isinstance(active_rules, (list, tuple, set)):
                norm_rules = list(active_rules)
            else:
                norm_rules = [active_rules]
            active_rule_keys = {
                (r.get("key") or r.get("rule_id") or r.get("rule") or r.get("name") or str(r))
                if isinstance(r, dict)
                else str(r)
                for r in norm_rules
                if r is not None
            }

        # Explicit violation declared in args or details
        for k in ("violated_rule", "violation"):
            if k in args_dict:
                return str(args_dict[k])
            if k in details_dict:
                return str(details_dict[k])

        raw_cmd = args_dict.get("CommandLine") or args_dict.get("command") or ""
        cmd = str(raw_cmd).strip()

        # Rule 1: native_first_rule
        if "native_first_rule" in active_rule_keys:
            platform_keywords = ("turna", "meiro", "dengage", "contact", "crm", "cdp")
            is_platform_context = (
                context_dict.get("is_turna", False)
                or any(kw in str(context_dict).lower() for kw in platform_keywords)
                or any(kw in cmd.lower() for kw in platform_keywords)
                or any(kw in str(args_dict).lower() for kw in platform_keywords)
            )

            is_secondary_bypass = (
                "mcp_bigquery" in tool_str
                or "bigquery" in tool_str
                or ("bq query" in cmd and is_platform_context)
            )

            bypass_flag = bool(args_dict.get("bypass_native") or details_dict.get("bypass_native"))
            unapproved = not details_dict.get("user_approved_bypass")
            if bypass_flag or (is_platform_context and is_secondary_bypass and unapproved):
                return "native_first_rule"

        # Rule 2: scientific_integrity_rule
        if "scientific_integrity_rule" in active_rule_keys:
            if "ModuleNotFoundError:" in err_str or "No module named" in err_str:
                return "scientific_integrity_rule"
            if details_dict.get("fabricated_claim") or args_dict.get("fabricated_claim"):
                return "scientific_integrity_rule"

        # Rule 3: executive_summary_rule
        if "executive_summary_rule" in active_rule_keys:
            content_str = str(tool_output or "") + " " + str(details_dict.get("response_text", ""))
            has_raw_math = any(
                kw in content_str
                for kw in ("Tr(\\rho \\Pi)", "P_{zeno}", "\\kappa_{csf}", "Hilbert minicolumn")
            )
            has_summary = any(
                h in content_str.lower()
                for h in ("executive summary", "yönetici özeti", "özet", "summary:")
            )
            technical_mode = details_dict.get("technical_mode", False)
            if has_raw_math and not has_summary and not technical_mode:
                return "executive_summary_rule"

        # General safety: Accidental data loss prevention
        destructive_patterns = (
            r"\bDROP\s+TABLE\b",
            r"\bDROP\s+DATABASE\b",
            r"\bTRUNCATE\s+TABLE\b",
            r"rm\s+-rf\s+/[^\s]*",
        )
        for dp in destructive_patterns:
            user_ok = args_dict.get("user_confirmed") or details_dict.get("user_confirmed")
            if re.search(dp, cmd, re.IGNORECASE) and not user_ok:
                return "accidental_data_loss_rule"

        return None

    def compute_decision_drift(
        self,
        intended_action: Any,
        executed_action: Any,
        dim: int = 64,
    ) -> float:
        """Calculates projective semantic decision drift Delta_drift = 1 - ||<psi_W | phi_A>||^2."""
        str_int = str(intended_action).strip()
        str_exec = str(executed_action).strip()
        if str_int == str_exec:
            return 0.0

        try:
            import torch

            from quanta.cognitive.memory import text_to_statevector

            if isinstance(intended_action, torch.Tensor):
                psi_w = intended_action
            else:
                psi_w = text_to_statevector(str_int, dim=dim)

            if isinstance(executed_action, torch.Tensor):
                phi_a = executed_action
            else:
                phi_a = text_to_statevector(str_exec, dim=dim)

            overlap = torch.vdot(psi_w, phi_a)
            prob = float(torch.abs(overlap).item() ** 2)
            drift = max(0.0, min(1.0, 1.0 - prob))
            return float(round(drift, 6))
        except Exception:
            set_i = set(str_int.lower().split())
            set_e = set(str_exec.lower().split())
            if not set_i and not set_e:
                return 0.0
            jaccard = len(set_i & set_e) / max(1, len(set_i | set_e))
            return float(round(max(0.0, min(1.0, 1.0 - jaccard)), 6))

    def extract_outcomes_from_transcript(
        self,
        transcript_path: str | Path,
        last_step_idx: int = 0,
        active_rules: list[str | dict[str, Any]] | None = None,
    ) -> list[ActionOutcome]:
        """Scans a conversation transcript JSONL file for tool actions executed after step."""
        if not transcript_path:
            return []

        p = Path(transcript_path)
        if not p.exists():
            if p.name == "transcript_full.jsonl":
                p = p.with_name("transcript.jsonl")
            elif p.name == "transcript.jsonl":
                p = p.with_name("transcript_full.jsonl")
            if not p.exists():
                return []

        outcomes: list[ActionOutcome] = []
        try:
            with open(p, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                seek_pos = max(0, size - 524288)
                f.seek(seek_pos)
                chunk = f.read().decode("utf-8", errors="replace")

            lines = chunk.splitlines()
            for idx, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except Exception:
                    continue

                if not isinstance(entry, dict):
                    continue

                raw_step = entry.get("step_index", entry.get("stepIdx", entry.get("step", idx)))
                try:
                    step_val = int(raw_step)
                except (ValueError, TypeError):
                    step_val = idx

                if last_step_idx > 0 and step_val <= last_step_idx:
                    continue

                # Case A: Model planner response with tool calls
                tool_calls = entry.get("tool_calls", [])
                if tool_calls and isinstance(tool_calls, list):
                    for tc in tool_calls:
                        if isinstance(tc, dict):
                            t_name = tc.get("name", tc.get("tool_name", "unknown_tool"))
                            t_args = tc.get("args", tc.get("parameters", {}))
                            outcome = self.evaluate_tool_result(
                                tool_name=t_name,
                                tool_args=t_args,
                                tool_output=None,
                                error=None,
                                active_rules=active_rules,
                                step_idx=step_val,
                            )
                            outcomes.append(outcome)

                # Case B: Direct TOOL / ENVIRONMENT entry
                src = entry.get("source")
                etype = entry.get("type")
                status = entry.get("status")
                is_tool_entry = (
                    src in ("TOOL", "ENVIRONMENT", "TOOL_CALL")
                    or etype in ("TOOL_RESULT", "TOOL_OUTPUT")
                    or "tool" in entry
                )
                if is_tool_entry:
                    t_name = entry.get("tool", entry.get("tool_name", entry.get("name", "tool")))
                    t_args = entry.get("args", entry.get("tool_args", entry.get("input", {})))
                    is_ok = status is None or status == "success"
                    t_exit_code = entry.get("exit_code", 0 if is_ok else 1)
                    t_out = entry.get("output", entry.get("result", entry.get("content", "")))
                    t_err = entry.get("error", entry.get("error_message"))
                    if status and status != "success" and not t_err:
                        t_err = f"Tool status: {status}"
                    outcome = self.evaluate_tool_result(
                        tool_name=t_name,
                        tool_args=t_args,
                        tool_output=t_out,
                        error=t_err,
                        exit_code=t_exit_code,
                        active_rules=active_rules,
                        step_idx=step_val,
                    )
                    outcomes.append(outcome)
        except Exception:
            pass

        return outcomes


class AdaptiveConfidenceEngine:
    """Computes dynamic asymptotic reinforcement, super-linear damping, and adaptive thresholds."""

    ETA_BOOST: float = 0.12
    CEILING: float = 0.9998
    FLOOR: float = 0.05
    DAMP_RATE: float = 0.12
    DAMP_EXP: float = 1.15
    VIOLATION_PENALTY: float = 0.30

    def __init__(
        self,
        eta: float = 0.12,
        mu: float = 0.12,
        gamma: float = 1.15,
        violation_penalty: float = 0.30,
        c_min: float = 0.05,
        c_max: float = 0.9998,
        theta_0: float = 0.20,
        sigma_drift: float = 0.15,
    ) -> None:
        self.eta = eta
        self.mu = mu
        self.gamma = gamma
        self.violation_penalty = violation_penalty
        self.c_min = c_min
        self.c_max = c_max
        self.theta_0 = theta_0
        self.sigma_drift = sigma_drift
        # Backward-compatible attribute aliases
        self.ETA_BOOST = eta
        self.DAMP_RATE = mu
        self.DAMP_EXP = gamma
        self.VIOLATION_PENALTY = violation_penalty

    def calculate_updated_confidence(
        self,
        current_confidence: float | None,
        outcome: ActionOutcome | None,
        consecutive_successes: int | None = 0,
        consecutive_failures: int | None = 0,
    ) -> tuple[float, dict[str, Any]]:
        """Calculates updated confidence score and adaptive control metadata."""
        try:
            c = float(current_confidence) if current_confidence is not None else 0.80
        except (ValueError, TypeError):
            c = 0.80
        c = max(self.c_min, min(self.c_max, c))

        try:
            k_succ = max(0, int(consecutive_successes)) if consecutive_successes is not None else 0
        except (ValueError, TypeError):
            k_succ = 0

        try:
            k_fail = max(0, int(consecutive_failures)) if consecutive_failures is not None else 0
        except (ValueError, TypeError):
            k_fail = 0

        raw_drift = getattr(outcome, "drift_score", 0.0) if outcome is not None else 0.0
        if raw_drift is None:
            raw_drift = 0.0
        try:
            drift = max(0.0, min(1.0, float(raw_drift)))
        except (ValueError, TypeError):
            drift = 0.0

        raw_otype = (
            getattr(outcome, "outcome_type", OutcomeType.NEUTRAL)
            if outcome is not None
            else OutcomeType.NEUTRAL
        )
        if not isinstance(raw_otype, OutcomeType):
            outcome_type = OutcomeType._missing_(str(raw_otype)) or OutcomeType.NEUTRAL
        else:
            outcome_type = raw_otype

        meta: dict[str, Any] = {
            "initial_confidence": c,
            "consecutive_successes": k_succ,
            "consecutive_failures": k_fail,
            "is_quarantined": False,
            "warning": None,
        }

        if outcome_type == OutcomeType.SUCCESS:
            k_succ += 1
            k_fail = 0
            new_c = min(self.c_max, c + self.eta * (1.0 - c) * (1.0 - 0.5 * drift))
            new_c = max(self.c_min, min(self.c_max, new_c))
            lambda_damp = 1.0
            salience_boost = 0.05
            fidelity_boost = 0.005
            regime = "reinforcement"
            warning = None

        elif outcome_type == OutcomeType.RULE_VIOLATION:
            k_fail += 1
            k_succ = 0
            new_c = max(self.c_min, min(self.c_max, c * (1.0 - self.violation_penalty)))
            lambda_damp = 1.0 - self.violation_penalty
            salience_boost = 0.50
            fidelity_boost = 0.0
            regime = "violation_penalty"
            if outcome is not None and getattr(outcome, "rule_violated", None):
                rule_name = outcome.rule_violated
            else:
                rule_name = "active_rule"
            warning = f"Rule violation detected for {rule_name}"
            if k_fail >= 3:
                meta["is_quarantined"] = True
                meta["alert_level"] = "high"

        elif outcome_type in (OutcomeType.TOOL_FAILURE, OutcomeType.SYNTAX_ERROR):
            k_fail += 1
            k_succ = 0
            lambda_damp = math.exp(-self.mu * (k_fail ** self.gamma))
            new_c = max(self.c_min, min(self.c_max, c * lambda_damp))
            salience_boost = max(-1.0, -0.10 * k_fail)
            fidelity_boost = 0.0
            regime = "superlinear_damping"
            warning = f"Tool failure encountered; confidence damped to {new_c:.4f}"
            if k_fail >= 5:
                warning = f"Severe consecutive tool failures (k={k_fail})"

        else:  # NEUTRAL
            new_c = c
            lambda_damp = 1.0
            salience_boost = 0.0
            fidelity_boost = 0.0
            regime = "neutral"
            warning = None

        adaptive_thresh = self.compute_adaptive_threshold(k_fail)
        if new_c >= adaptive_thresh:
            zeno_pinning = min(self.c_max, max(0.75, new_c))
            tunneling_rate = max(0.0, 1.0 - zeno_pinning)
        else:
            zeno_pinning = max(0.01, 1.0 - adaptive_thresh)
            tunneling_rate = max(0.0, 1.0 - zeno_pinning)

        delta = round(new_c - c, 6)
        meta.update({
            "delta": delta,
            "delta_confidence": delta,
            "consecutive_successes": k_succ,
            "consecutive_failures": k_fail,
            "lambda_damp": round(lambda_damp, 6),
            "adaptive_threshold": round(adaptive_thresh, 6),
            "regime": regime,
            "zeno_pinning_factor": round(zeno_pinning, 4),
            "anti_zeno_tunneling_rate": round(tunneling_rate, 4),
            "salience_boost": round(salience_boost, 4),
            "fidelity_boost": round(fidelity_boost, 4),
            "warning": warning,
        })

        return round(new_c, 6), meta

    def compute_adaptive_threshold(self, consecutive_failures: int | None = 0) -> float:
        """Computes theta_adaptive(k_fail) scaling from baseline toward Anti-Zeno."""
        try:
            k = max(0, int(consecutive_failures)) if consecutive_failures is not None else 0
        except (ValueError, TypeError):
            k = 0
        theta = self.theta_0 + self.sigma_drift * (k ** 0.8)
        return float(round(max(0.10, min(0.85, theta)), 6))


class RuleDriftTracker:
    """Maintains state drift history and consecutive counter statistics per rule."""

    def __init__(self, history_limit: int = 50) -> None:
        self.history_limit = history_limit
        self.stats: dict[str, dict[str, Any]] = {}
        self.history: list[dict[str, Any]] = []

    def get_drift_status(self, confidence: float) -> str:
        """Classifies confidence into qualitative health tiers."""
        if confidence >= 0.90:
            return "REINFORCED"
        elif confidence >= 0.70:
            return "STABLE"
        elif confidence >= 0.40:
            return "DEGRADED"
        else:
            return "CRITICAL_DRIFT"

    def get_rule_stats(self, rule_key: str) -> dict[str, Any]:
        """Returns statistical ledger for the given rule."""
        return self.stats.get(
            rule_key,
            {
                "success_count": 0,
                "failure_count": 0,
                "violation_count": 0,
                "consecutive_successes": 0,
                "consecutive_failures": 0,
                "confidence": 0.95,
                "total_evaluations": 0,
                "last_outcome": "NEUTRAL",
                "drift_status": "STABLE",
            },
        )

    def record_update(
        self,
        rule_key: str,
        outcome: ActionOutcome,
        old_confidence: float,
        new_confidence: float,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Records an update and appends to drift history."""
        meta = meta or {}
        if rule_key not in self.stats:
            self.stats[rule_key] = {
                "success_count": 0,
                "failure_count": 0,
                "violation_count": 0,
                "consecutive_successes": 0,
                "consecutive_failures": 0,
                "total_evaluations": 0,
                "confidence": old_confidence,
                "drift_status": "STABLE",
            }
        s = self.stats[rule_key]
        s["confidence"] = new_confidence
        s["total_evaluations"] = s.get("total_evaluations", 0) + 1
        s["drift_status"] = self.get_drift_status(new_confidence)

        if outcome.outcome_type == OutcomeType.SUCCESS:
            s["success_count"] += 1
            cur_succ = s["consecutive_successes"] + 1
            s["consecutive_successes"] = meta.get("consecutive_successes", cur_succ)
            s["consecutive_failures"] = 0
            s["last_outcome"] = "SUCCESS"
        elif outcome.outcome_type == OutcomeType.RULE_VIOLATION:
            s["violation_count"] += 1
            cur_fail = s["consecutive_failures"] + 1
            s["consecutive_failures"] = meta.get("consecutive_failures", cur_fail)
            s["consecutive_successes"] = 0
            s["last_outcome"] = "RULE_VIOLATION"
        elif outcome.outcome_type in (OutcomeType.TOOL_FAILURE, OutcomeType.SYNTAX_ERROR):
            s["failure_count"] += 1
            cur_fail = s["consecutive_failures"] + 1
            s["consecutive_failures"] = meta.get("consecutive_failures", cur_fail)
            s["consecutive_successes"] = 0
            s["last_outcome"] = outcome.outcome_type.value.upper()

        delta = round(new_confidence - old_confidence, 6)
        entry = {
            "timestamp": time.time(),
            "rule_key": rule_key,
            "rule": rule_key,
            "old_confidence": round(old_confidence, 4),
            "new_confidence": round(new_confidence, 4),
            "prior_conf": round(old_confidence, 4),
            "post_conf": round(new_confidence, 4),
            "delta": delta,
            "delta_conf": delta,
            "outcome": outcome.outcome_type.value,
            "outcome_type": outcome.outcome_type.value,
            "step_idx": outcome.step_idx,
            "drift": round(outcome.drift_score, 4),
            "status": s["drift_status"],
            "warning": meta.get("warning"),
        }
        self.history.append(entry)
        if len(self.history) > self.history_limit:
            self.history.pop(0)
        return entry

    def record_evaluation(
        self,
        rule_key: str,
        outcome: ActionOutcome,
        prior_conf: float,
        post_conf: float,
        meta: dict[str, Any],
    ) -> dict[str, Any]:
        """Alias for record_update to maintain interface polymorphism."""
        return self.record_update(
            rule_key=rule_key,
            outcome=outcome,
            old_confidence=prior_conf,
            new_confidence=post_conf,
            meta=meta,
        )

    def update_engram_in_state(
        self,
        state_dict: dict[str, Any],
        rule_key: str,
        new_confidence: float,
        meta: dict[str, Any],
        outcome: ActionOutcome,
    ) -> bool:
        """Updates an engram in the state dictionary and appends outcome history."""
        if not isinstance(state_dict, dict):
            return False

        raw_engrams = state_dict.get("engrams")
        if not isinstance(raw_engrams, list):
            raw_engrams = []
            state_dict["engrams"] = raw_engrams

        target_engram: dict[str, Any] | None = None
        for e in raw_engrams:
            if isinstance(e, dict) and e.get("key") == rule_key:
                target_engram = e
                break

        if target_engram is None:
            target_engram = {
                "key": rule_key,
                "content": f"Constraint rule {rule_key}",
                "salience": 2.5,
                "category": "constraint",
                "fidelity": 0.9998,
                "age": 0,
                "tags": ["constraint"],
                "topic": rule_key,
                "confidence": 0.95,
                "is_core_anchor": True,
            }
            raw_engrams.append(target_engram)

        prior_c = float(target_engram.get("confidence", 0.95))
        target_engram["confidence"] = round(new_confidence, 6)
        meta_dict = meta if isinstance(meta, dict) else {}
        target_engram["consecutive_successes"] = int(meta_dict.get("consecutive_successes", 0))
        target_engram["consecutive_failures"] = int(meta_dict.get("consecutive_failures", 0))
        target_engram["total_evaluations"] = int(target_engram.get("total_evaluations", 0)) + 1
        raw_otype = getattr(outcome, "outcome_type", OutcomeType.SUCCESS)
        o_val = raw_otype.value if hasattr(raw_otype, "value") else str(raw_otype)
        target_engram["last_outcome"] = o_val
        target_engram["last_outcome_time"] = time.time()
        target_engram["drift_status"] = self.get_drift_status(new_confidence)
        if meta_dict.get("is_quarantined"):
            target_engram["is_quarantined"] = True

        if meta_dict.get("salience_boost"):
            cur_sal = float(target_engram.get("salience", 2.0))
            new_sal = min(5.0, max(0.5, cur_sal + meta_dict["salience_boost"]))
            target_engram["salience"] = round(new_sal, 2)

        cur_sal = float(target_engram.get("salience", 2.0))
        # Ensure is_core_anchor is preserved and resolved (salience >= 2.0 implies core anchor)
        target_engram["is_core_anchor"] = bool(
            meta_dict.get(
                "is_core_anchor",
                target_engram.get("is_core_anchor", cur_sal >= 2.0),
            )
            or (cur_sal >= 2.0)
        )

        if meta_dict.get("fidelity_boost"):
            cur_fid = float(target_engram.get("fidelity", 0.9998))
            target_engram["fidelity"] = round(min(0.9998, cur_fid + meta_dict["fidelity_boost"]), 6)


        history_entry = self.record_update(rule_key, outcome, prior_c, new_confidence, meta_dict)
        history = state_dict.setdefault("outcome_history", [])
        if not isinstance(history, list):
            history = []
            state_dict["outcome_history"] = history
        history.append(history_entry)
        if len(history) > 100:
            state_dict["outcome_history"] = history[-100:]

        state_dict["feedback_history"] = self.history
        state_dict["feedback_metrics"] = self.stats
        return True

    def to_dict(self) -> dict[str, Any]:
        return {"stats": self.stats, "history": self.history}

    def load_dict(self, data: dict[str, Any]) -> None:
        if isinstance(data, dict):
            self.stats = data.get("stats", {})
            self.history = data.get("history", [])


def _save_state_atomically(
    state_file: Path,
    state_dict: dict,
    safe_conv_id: str = "default",
) -> bool:
    """Atomically writes state to state_file, falling back to /tmp if unwriteable."""
    candidates = [state_file]
    fallback = Path(f"/tmp/quanta_cognitive_{safe_conv_id}.json")
    if fallback != state_file:
        candidates.append(fallback)

    for target in candidates:
        temp_path = None
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            temp_path = target.with_name(f".tmp_{target.name}_{os.getpid()}_{time.time_ns()}")
            with open(temp_path, "w", encoding="utf-8") as sf:
                json.dump(state_dict, sf, ensure_ascii=False, indent=2)
                sf.flush()
                with contextlib.suppress(OSError):
                    os.fsync(sf.fileno())
            os.replace(temp_path, target)
            return True
        except Exception:
            if temp_path is not None and temp_path.exists():
                with contextlib.suppress(OSError):
                    temp_path.unlink()
            continue
        except BaseException:
            if temp_path is not None and temp_path.exists():
                with contextlib.suppress(OSError):
                    temp_path.unlink()
            raise
    return False


class CognitiveFeedbackLoop:
    """Closed-loop coordinator evaluating outcomes, updating state, and logging telemetry."""

    def __init__(
        self,
        state_path: str | Path | None = None,
        telemetry_path: str | Path | None = None,
        verifier: OutcomeVerifier | None = None,
        engine: AdaptiveConfidenceEngine | None = None,
        tracker: RuleDriftTracker | None = None,
        confidence_engine: AdaptiveConfidenceEngine | None = None,
        drift_tracker: RuleDriftTracker | None = None,
    ) -> None:
        self.state_path = Path(state_path) if state_path else None
        self.telemetry_path = Path(telemetry_path) if telemetry_path else None
        self.verifier = verifier or OutcomeVerifier()
        self.engine = confidence_engine or engine or AdaptiveConfidenceEngine()
        self.confidence_engine = self.engine
        self.tracker = drift_tracker or tracker or RuleDriftTracker()
        self.drift_tracker = self.tracker

    def process_outcomes(
        self,
        outcomes: list[ActionOutcome],
        state_path: str | Path | None = None,
        telemetry_path: str | Path | None = None,
        workspace: str | None = None,
    ) -> dict[str, Any]:
        """Processes a list of ActionOutcomes, updating cognitive state and logging telemetry."""
        target_state_path = Path(state_path) if state_path else self.state_path
        target_telemetry_path = Path(telemetry_path) if telemetry_path else self.telemetry_path

        state_data: dict[str, Any] = {}
        if target_state_path and target_state_path.exists():
            try:
                with open(target_state_path, encoding="utf-8", errors="replace") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict):
                        state_data = loaded
            except Exception:
                state_data = {}

        if not isinstance(state_data, dict) or not state_data:
            state_data = {
                "turn_count": 0,
                "last_injected_time": time.time(),
                "last_step_idx": 0,
                "engrams": [],
                "total_pruned_count": 0,
                "kappa_csf": 1.0 / 6250.0,
            }

        raw_engrams = state_data.get("engrams")
        if not isinstance(raw_engrams, list):
            raw_engrams = []
        engrams: list[dict[str, Any]] = [
            e
            for e in raw_engrams
            if isinstance(e, dict) and "key" in e and isinstance(e.get("key"), str)
        ]
        state_data["engrams"] = engrams
        engram_map = {e["key"]: e for e in engrams}

        processed_records: list[dict[str, Any]] = []
        updated_rules: list[str] = []
        warnings: list[str] = []

        for outcome in outcomes:
            rule_key = outcome.rule_violated
            if not rule_key:
                rule_key = outcome.details.get("rule_key")
            if not rule_key:
                if outcome.tool_name in ("run_command", "bash", "gcloud", "dengage", "mcp"):
                    rule_key = "native_first_rule"
                else:
                    rule_key = "scientific_integrity_rule"

            engram = engram_map.get(rule_key)
            current_conf = float(engram.get("confidence", 0.95)) if engram else 0.95
            rule_stats = self.drift_tracker.get_rule_stats(rule_key)

            new_conf, meta = self.engine.calculate_updated_confidence(
                current_confidence=current_conf,
                outcome=outcome,
                consecutive_successes=rule_stats.get("consecutive_successes", 0),
                consecutive_failures=rule_stats.get("consecutive_failures", 0),
            )

            if engram:
                engram["confidence"] = new_conf
                if meta.get("is_quarantined"):
                    engram["is_quarantined"] = True

            self.drift_tracker.update_engram_in_state(
                state_dict=state_data,
                rule_key=rule_key,
                new_confidence=new_conf,
                meta=meta,
                outcome=outcome,
            )
            last_rec = self.drift_tracker.history[-1] if self.drift_tracker.history else {}
            processed_records.append(last_rec)

            if rule_key not in updated_rules:
                updated_rules.append(rule_key)

            if meta.get("warning"):
                warnings.append(meta["warning"])

            # Telemetry logging (only if telemetry target is configured)
            if target_telemetry_path:
                with contextlib.suppress(Exception):
                    record_outcome_telemetry(
                        outcome=outcome,
                        rule_key=rule_key,
                        old_confidence=current_conf,
                        new_confidence=new_conf,
                        telemetry_file=target_telemetry_path,
                        workspace=workspace,
                        warning=meta.get("warning"),
                    )

        state_data["feedback_history"] = self.drift_tracker.history
        state_data["feedback_metrics"] = self.drift_tracker.stats

        if target_state_path:
            safe_id = target_state_path.stem.replace("quanta_cognitive_", "") or "default"
            _save_state_atomically(target_state_path, state_data, safe_conv_id=safe_id)

        return {
            "processed_count": len(outcomes),
            "records": processed_records,
            "state_updated": target_state_path is not None,
            "updated_rules": updated_rules,
            "warnings": warnings,
            "state_file": str(target_state_path) if target_state_path else None,
        }

    def record_outcome(
        self,
        rule_key: str,
        outcome_type: OutcomeType | str,
        step_idx: int = 0,
    ) -> bool:
        """Manually records an outcome for a specific rule."""
        try:
            ot = OutcomeType(outcome_type) if isinstance(outcome_type, str) else outcome_type
        except Exception:
            is_succ = str(outcome_type).lower() == "success"
            ot = OutcomeType.SUCCESS if is_succ else OutcomeType.TOOL_FAILURE

        outcome = ActionOutcome(
            step_idx=step_idx,
            tool_name="manual_feedback",
            outcome_type=ot,
            rule_violated=rule_key,
        )
        try:
            res = self.process_outcomes([outcome])
            return res.get("processed_count", 0) > 0
        except Exception:
            return False

    def ingest_transcript_feedback(self, transcript_path: str | Path) -> list[ActionOutcome]:
        """Ingests all unrecorded outcomes from a transcript and triggers feedback processing."""
        outcomes = self.verifier.extract_outcomes_from_transcript(transcript_path)
        if outcomes:
            self.process_outcomes(outcomes)
        return outcomes

    def evaluate_and_apply(
        self,
        tool_name: str,
        tool_args: dict[str, Any] | None = None,
        tool_output: Any = None,
        error: str | Exception | None = None,
        state_path: str | Path | None = None,
        active_rules: list[str | dict[str, Any]] | None = None,
        step_idx: int = 0,
        **kwargs: Any,
    ) -> ActionOutcome:
        """Single-step evaluation and immediate closed-loop state update helper."""
        outcome = self.verifier.evaluate_tool_result(
            tool_name=tool_name,
            tool_args=tool_args,
            tool_output=tool_output,
            error=error,
            active_rules=active_rules,
            step_idx=step_idx,
            **kwargs,
        )
        self.process_outcomes(
            outcomes=[outcome],
            state_path=state_path,
            workspace=kwargs.get("workspace"),
        )
        return outcome


# Compatibility Alias
CognitiveFeedbackEvaluator = CognitiveFeedbackLoop


def apply_confidence_feedback(
    state_dict: dict[str, Any],
    rule_key: str,
    status: OutcomeType | str,
    telemetry_file: Path | None = None,
) -> dict[str, Any]:
    """Pure helper function applying confidence feedback to a state dictionary."""
    state_copy = copy.deepcopy(state_dict)
    try:
        ot = OutcomeType(status) if isinstance(status, str) else status
    except Exception:
        ot = OutcomeType.SUCCESS if str(status).lower() == "success" else OutcomeType.TOOL_FAILURE

    outcome = ActionOutcome(
        step_idx=state_copy.get("last_step_idx", 0) + 1,
        tool_name="feedback_action",
        outcome_type=ot,
        rule_violated=rule_key if ot == OutcomeType.RULE_VIOLATION else None,
    )
    engine = AdaptiveConfidenceEngine()
    engram = next((e for e in state_copy.get("engrams", []) if e.get("key") == rule_key), None)
    old_c = float(engram.get("confidence", 0.95)) if engram else 0.95

    new_c, meta = engine.calculate_updated_confidence(
        current_confidence=old_c,
        outcome=outcome,
        consecutive_successes=int(engram.get("consecutive_successes", 0)) if engram else 0,
        consecutive_failures=int(engram.get("consecutive_failures", 0)) if engram else 0,
    )
    if engram:
        engram["confidence"] = new_c
        if meta.get("is_quarantined"):
            engram["is_quarantined"] = True
        if meta.get("consecutive_successes") is not None:
            engram["consecutive_successes"] = meta["consecutive_successes"]
        if meta.get("consecutive_failures") is not None:
            engram["consecutive_failures"] = meta["consecutive_failures"]
        eng_sal = float(engram.get("salience", 2.0))
        engram["is_core_anchor"] = bool(
            meta.get(
                "is_core_anchor",
                engram.get("is_core_anchor", eng_sal >= 2.0),
            )
            or (eng_sal >= 2.0)
        )

    if telemetry_file:
        from quanta.cognitive.telemetry import record_outcome_telemetry

        record_outcome_telemetry(
            outcome=outcome,
            rule_key=rule_key,
            old_confidence=old_c,
            new_confidence=new_c,
            telemetry_file=telemetry_file,
            warning=meta.get("warning"),
        )
    return state_copy
