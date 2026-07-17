"""Helpers for writing A1111 prompt metadata into ComfyUI PNG info."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

METADATA_KEY = "a1111_prompt_node"
RESOLVED_PROMPT_PROPERTY = "a1111_resolved_prompt"
RESOLVED_PROMPT_EXPANDED_PROPERTY = "a1111_resolved_prompt_expanded"
RESOLVED_PROMPT_SOURCE_PROPERTY = "a1111_resolved_prompt_source"


def _coerce_workflow(extra_pnginfo: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    workflow = extra_pnginfo.get("workflow")
    if isinstance(workflow, dict):
        return workflow
    if isinstance(workflow, str):
        try:
            parsed = json.loads(workflow)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            extra_pnginfo["workflow"] = parsed
            return parsed
    return None


def _update_workflow_preview(
    extra_pnginfo: Dict[str, Any],
    node_id: str,
    original_prompt: str,
    resolved_prompt: str,
) -> None:
    workflow = _coerce_workflow(extra_pnginfo)
    if workflow is None:
        return

    nodes = workflow.get("nodes")
    if not isinstance(nodes, list):
        return

    for node in nodes:
        if not isinstance(node, dict) or str(node.get("id")) != node_id:
            continue

        properties = node.setdefault("properties", {})
        if not isinstance(properties, dict):
            properties = {}
            node["properties"] = properties

        if original_prompt != resolved_prompt:
            properties[RESOLVED_PROMPT_PROPERTY] = resolved_prompt
            properties[RESOLVED_PROMPT_SOURCE_PROPERTY] = original_prompt
            properties.setdefault(RESOLVED_PROMPT_EXPANDED_PROPERTY, False)
        else:
            properties.pop(RESOLVED_PROMPT_PROPERTY, None)
            properties.pop(RESOLVED_PROMPT_EXPANDED_PROPERTY, None)
            properties.pop(RESOLVED_PROMPT_SOURCE_PROPERTY, None)
        return


def record_prompt_metadata(
    extra_pnginfo: Optional[Dict[str, Any]],
    node_id: Optional[str],
    node_type: str,
    original_prompt: str,
    resolved_prompt: str,
) -> None:
    """Store original/resolved prompt text in ComfyUI's extra PNG metadata."""
    if extra_pnginfo is None or node_id is None:
        return

    key = str(node_id)
    _update_workflow_preview(extra_pnginfo, key, original_prompt, resolved_prompt)

    metadata = extra_pnginfo.setdefault(METADATA_KEY, {"nodes": {}})
    nodes = metadata.setdefault("nodes", {})
    nodes[key] = {
        "node_id": key,
        "type": node_type,
        "original_prompt": original_prompt,
        "resolved_prompt": resolved_prompt,
        "changed": original_prompt != resolved_prompt,
    }
