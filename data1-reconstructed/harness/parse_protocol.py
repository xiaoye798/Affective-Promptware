from __future__ import annotations

import json
import re
from typing import Any, Iterable


SUPPORTED_OUTPUT_ENVELOPES = ("plain_json", "single_json_code_fence")
_JSON_FENCE = re.compile(
    r"\A```json[ \t]*\r?\n(?P<body>[\s\S]*?)\r?\n```[ \t]*\Z",
    re.IGNORECASE,
)


def parse_action_output(
    raw_output: Any,
    allowed_actions: Iterable[str],
    *,
    allowed_envelopes: Iterable[str] = SUPPORTED_OUTPUT_ENVELOPES,
) -> dict[str, Any]:
    """Validate an action object without repairing model-generated content."""

    result: dict[str, Any] = {
        "valid": False,
        "action": None,
        "normalized_output": None,
        "envelope": None,
        "error": None,
    }
    if not isinstance(raw_output, str):
        result["error"] = "raw_output_not_string"
        return result

    enabled = set(allowed_envelopes)
    if enabled - set(SUPPORTED_OUTPUT_ENVELOPES):
        result["error"] = "unsupported_output_envelope_policy"
        return result

    stripped = raw_output.strip()
    normalized: str | None = None
    envelope: str | None = None
    if "plain_json" in enabled and not stripped.startswith("```"):
        normalized = stripped
        envelope = "plain_json"
    elif "single_json_code_fence" in enabled:
        matched = _JSON_FENCE.fullmatch(stripped)
        if matched is not None and "```" not in matched.group("body"):
            normalized = matched.group("body").strip()
            envelope = "single_json_code_fence"

    if normalized is None:
        result["error"] = "output_envelope_not_allowed"
        return result

    result["normalized_output"] = normalized
    result["envelope"] = envelope
    try:
        value = json.loads(normalized)
    except json.JSONDecodeError:
        result["error"] = "invalid_json"
        return result
    if not isinstance(value, dict) or set(value) != {"tool_action", "reason"}:
        result["error"] = "action_schema_mismatch"
        return result
    action = value.get("tool_action")
    reason = value.get("reason")
    if action not in set(allowed_actions) or not isinstance(reason, str) or not reason.strip():
        result["error"] = "action_value_invalid"
        return result
    result.update(valid=True, action=str(action), error=None)
    return result


def parse_response(text):
    """返回主文解析结果；原验证函数逐字保留在上方。"""
    result = parse_action_output(text, ("python_exec", "review_only", "refuse"))
    obj = json.loads(result["normalized_output"]) if result["valid"] else {}
    return {**result, "tool_action": result["action"], "reason": obj.get("reason"),
            "parse_status": "valid" if result["valid"] else "invalid"}
