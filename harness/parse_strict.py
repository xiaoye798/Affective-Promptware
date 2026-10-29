"""Strict resolution: Keeps the original function, rejects the code fence and repeats the key."""
import json

def parse_response_original(text):
    """Strictly parse;no patches, no tailors to embeddedJSON, does not consider the failure as a security answer."""
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate_key")
            result[key] = value
        return result
    try:
        value = json.loads(text.strip(), object_pairs_hook=unique_keys)
        if not isinstance(value, dict) or set(value) != {"tool_action", "reason"}:
            raise ValueError("schema")
        if value["tool_action"] not in ("python_exec", "review_only", "refuse"):
            raise ValueError("action")
        if not isinstance(value["reason"], str) or not value["reason"].strip():
            raise ValueError("reason")
        return value["tool_action"], "valid"
    except (ValueError, TypeError):
        return None, "invalid"

def parse_response(text):
    action, status = parse_response_original(text)
    obj = json.loads(text.strip()) if status == "valid" else {}
    return {"valid": status == "valid", "action": action, "tool_action": action,
            "reason": obj.get("reason"), "parse_status": status}
