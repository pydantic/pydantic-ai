"""Logging utilities for Prefect integration."""

from __future__ import annotations

import json
from typing import Any


def format_tool_args(args: dict[str, Any], max_length: int = 500) -> str:
    """Format tool arguments for logging, truncating if too long.

    Args:
        args: The tool arguments dictionary.
        max_length: Maximum length of the formatted string.

    Returns:
        A JSON-formatted string representation of the arguments.
    """
    try:
        args_str = json.dumps(args, default=str)
        if len(args_str) > max_length:
            return args_str[: max_length - 3] + '...'
        return args_str
    except Exception:
        return str(args)[:max_length]


def format_tool_result(result: Any, max_length: int = 500) -> str:
    """Format tool result for logging, truncating if too long.

    Args:
        result: The tool return value.
        max_length: Maximum length of the formatted string.

    Returns:
        A JSON-formatted string representation of the result.
    """
    try:
        if isinstance(result, str):
            result_str = result
        else:
            result_str = json.dumps(result, default=str)
        if len(result_str) > max_length:
            return result_str[: max_length - 3] + '...'
        return result_str
    except Exception:
        return str(result)[:max_length]
