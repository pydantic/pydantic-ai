"""Optional JSON repair functionality using fast-json-repair.

This module provides JSON repair for malformed JSON strings that LLMs
sometimes produce (missing braces, trailing commas, single quotes, etc.).

The repair functionality is only available when fast-json-repair is installed:
    pip install 'pydantic-ai-slim[json-repair]'
"""

from __future__ import annotations

try:
    from fast_json_repair import repair_json  # pyright: ignore[reportUnknownVariableType]

    def maybe_repair_json(json_string: str) -> str:
        """Attempt to repair malformed JSON using fast_json_repair.

        Args:
            json_string: The potentially malformed JSON string.

        Returns:
            The repaired JSON string if repairs were made, otherwise the original string.
        """
        result = repair_json(json_string, return_objects=False)
        # repair_json returns str when return_objects=False
        return str(result)

except ImportError as _import_error:

    def maybe_repair_json(json_string: str) -> str:
        """Fallback when fast_json_repair is not available - returns input unchanged."""
        return json_string
