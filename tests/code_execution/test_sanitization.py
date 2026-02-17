"""Tests for tool name sanitization edge cases."""

from __future__ import annotations

import pytest

from pydantic_ai.toolsets.code_execution import _sanitize_tool_name  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(
    ('input_name', 'expected'),
    [
        # Basic separators
        ('search-records', 'search_records'),
        ('get.user.data', 'get_user_data'),
        ('hello world', 'hello_world'),
        # camelCase / PascalCase → snake_case
        ('getUserData', 'get_user_data'),
        ('XMLParser', 'xml_parser'),
        # Leading digit
        ('3d_model', '_3d_model'),
        # Unicode stripped → fallback
        ('获取数据', 'tool'),
        # All-separator → underscore (separators become _, then lowercased)
        ('---', '_'),
        ('...', '_'),
        # Python keywords
        ('class', 'class_'),
        ('return', 'return_'),
        ('import', 'import_'),
        # Keyword produced by camelCase conversion
        ('returnValue', 'return_value'),
        # Mixed separators
        ('get-user.data name', 'get_user_data_name'),
        # Already valid
        ('valid_name', 'valid_name'),
        # Single character
        ('x', 'x'),
    ],
    ids=lambda v: repr(v),
)
def test_sanitize_tool_name(input_name: str, expected: str) -> None:
    assert _sanitize_tool_name(input_name) == expected


def test_sanitize_collision_handling() -> None:
    """Two distinct names that sanitize to the same result are handled by the caller (CodeExecutionToolset).

    _sanitize_tool_name itself is stateless — it just normalizes a single name.
    This test documents that identical outputs are possible.
    """
    assert _sanitize_tool_name('get-data') == _sanitize_tool_name('get.data') == 'get_data'
