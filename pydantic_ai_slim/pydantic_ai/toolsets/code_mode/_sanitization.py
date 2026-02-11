"""Tool name sanitization for code mode.

MCP and other external tools can have names that aren't valid Python identifiers
(e.g., 'search-records', 'get.data'). This module provides utilities to sanitize
these names and maintain mappings back to the originals.
"""

from __future__ import annotations

import keyword
import re


def sanitize_tool_name(name: str) -> str:
    """Convert a tool name to a valid Python identifier.

    Args:
        name: The original tool name (may contain hyphens, dots, etc.)

    Returns:
        A valid Python identifier in snake_case.

    Examples:
        >>> sanitize_tool_name('search-records')
        'search_records'
        >>> sanitize_tool_name('get.user.data')
        'get_user_data'
        >>> sanitize_tool_name('class')  # Python keyword
        'class_'
    """
    # Replace common separators with underscores
    sanitized = re.sub(r'[-.\s]+', '_', name)

    # Remove any remaining invalid characters (keep alphanumeric and underscore)
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '', sanitized)

    # Ensure it doesn't start with a digit
    if sanitized and sanitized[0].isdigit():
        sanitized = f'_{sanitized}'

    # Handle empty result
    if not sanitized:
        sanitized = 'tool'

    # Convert to snake_case if it's camelCase or PascalCase
    # Insert underscore before uppercase letters that follow lowercase
    sanitized = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', sanitized)
    sanitized = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', sanitized).lower()

    # Handle Python keywords by appending underscore
    if keyword.iskeyword(sanitized):
        sanitized = f'{sanitized}_'

    return sanitized


class ToolNameMapping:
    """Bidirectional mapping between sanitized and original tool names.

    Used by CodeModeToolset to:
    1. Present sanitized names to the LLM in signatures
    2. Map back to original names when calling tools
    """

    def __init__(self):
        """Initialize the mapping."""
        self._sanitized_to_original: dict[str, str] = {}
        self._original_to_sanitized: dict[str, str] = {}

    def add(self, original_name: str) -> str:
        """Add a tool name and return its sanitized version.

        Args:
            original_name: The original tool name

        Returns:
            The sanitized name (valid Python identifier)
        """
        if original_name in self._original_to_sanitized:
            return self._original_to_sanitized[original_name]

        sanitized = sanitize_tool_name(original_name)

        # Handle collisions by appending a number
        base_sanitized = sanitized
        counter = 2
        while sanitized in self._sanitized_to_original:
            sanitized = f'{base_sanitized}_{counter}'
            counter += 1

        self._sanitized_to_original[sanitized] = original_name
        self._original_to_sanitized[original_name] = sanitized
        return sanitized

    def get_original(self, sanitized_name: str) -> str | None:
        """Get the original tool name from a sanitized name.

        Args:
            sanitized_name: The sanitized (Python-valid) name

        Returns:
            The original tool name, or None if not found
        """
        return self._sanitized_to_original.get(sanitized_name)

    def get_sanitized(self, original_name: str) -> str | None:
        """Get the sanitized name from an original tool name.

        Args:
            original_name: The original tool name

        Returns:
            The sanitized name, or None if not found
        """
        return self._original_to_sanitized.get(original_name)

    def clear(self) -> None:
        """Clear all mappings."""
        self._sanitized_to_original.clear()
        self._original_to_sanitized.clear()
