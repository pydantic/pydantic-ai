"""Tool name sanitization for code mode.

MCP and other external tools can have names that aren't valid Python identifiers
(e.g., 'search-records', 'get.data'). This module provides a function to sanitize
these names to valid Python identifiers.
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
