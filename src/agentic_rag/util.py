"""Small shared helpers."""

from __future__ import annotations

import json
import re


def loads_json(text: str) -> dict:
    """Best-effort JSON parse tolerant of code fences / stray prose around the object.

    LLMs occasionally wrap JSON in ```json fences or add a sentence of preamble; we extract
    the first balanced-looking object and parse it, returning ``{}`` on failure so callers
    can fall back to safe defaults instead of raising.
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[-1] if "\n" in text else text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {}
