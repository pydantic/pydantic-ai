"""Tests for validate_tool_choice() function in _tool_choice.py.

This module provides comprehensive unit tests for the tool_choice validation logic,
covering all input types, edge cases, warning conditions, and error scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from pydantic_ai.exceptions import UserError
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models._tool_choice import ResolvedToolChoice, resolve_tool_choice
from pydantic_ai.settings import ModelSettings, ToolChoice, ToolOrOutput
from pydantic_ai.tools import ToolDefinition, ToolKind


def make_tool(name: str, *, kind: ToolKind = 'function') -> ToolDefinition:
    """Create a minimal ToolDefinition for testing."""
    return ToolDefinition(
        name=name,
        description=f'Test tool {name}',
        parameters_json_schema={'type': 'object', 'properties': {}},
        kind=kind,
    )


def make_params(
    *,
    function_tools: list[str] | None = None,
    output_tools: list[str] | None = None,
    allow_text_output: bool = False,
    allow_image_output: bool = False,
) -> ModelRequestParameters:
    """Create ModelRequestParameters with specified tools and settings."""
    return ModelRequestParameters(
        function_tools=[make_tool(name) for name in (function_tools or [])],
        output_tools=[make_tool(name, kind='output') for name in (output_tools or [])],
        allow_text_output=allow_text_output,
        allow_image_output=allow_image_output,
    )


# =============================================================================
# Sentinel classes for special test outcomes
# =============================================================================


@dataclass
class ExpectError:
    """Test should raise UserError with matching message."""

    match: str


@dataclass
class ExpectWarning:
    """Test should emit warning and return result."""

    match: str
    result: ResolvedToolChoice


# =============================================================================
# Test case dataclass
# =============================================================================


@dataclass
class Case:
    """A single test case for validate_tool_choice()."""

    id: str
    tool_choice: ToolChoice
    expected: ResolvedToolChoice | ExpectError | ExpectWarning
    function_tools: list[str] = field(default_factory=lambda: ['tool_a'])
    output_tools: list[str] = field(default_factory=list)
    allow_text_output: bool = False
    allow_image_output: bool = False
    none_settings: bool = False


# =============================================================================
# Test cases matrix
# =============================================================================

CASES = [
    # === Default/Auto behavior ===
    Case('auto-text-allowed', 'auto', 'auto', allow_text_output=True),
    Case('auto-text-not-allowed', 'auto', 'required'),
    Case('none-text-allowed', None, 'auto', allow_text_output=True),
    Case('none-text-not-allowed', None, 'required'),
    Case('none-settings', None, 'auto', allow_text_output=True, none_settings=True),
    # === None/Empty list: with output tools ===
    Case(
        'none-output-text',
        'none',
        ExpectWarning("tool_choice='none'", ('auto', ['final_result'])),
        output_tools=['final_result'],
        allow_text_output=True,
    ),
    Case(
        'empty-output-text',
        [],
        ExpectWarning("tool_choice='none'", ('auto', ['final_result'])),
        output_tools=['final_result'],
        allow_text_output=True,
    ),
    Case(
        'none-output-image',
        'none',
        ExpectWarning("tool_choice='none'", ('auto', ['final_result'])),
        output_tools=['final_result'],
        allow_image_output=True,
    ),
    Case(
        'empty-output-image',
        [],
        ExpectWarning("tool_choice='none'", ('auto', ['final_result'])),
        output_tools=['final_result'],
        allow_image_output=True,
    ),
    Case(
        'none-output-no-direct-with-func',
        'none',
        ExpectWarning("tool_choice='none'", ('required', ['final_result'])),
        output_tools=['final_result'],
    ),
    Case(
        'empty-output-no-direct-with-func',
        [],
        ExpectWarning("tool_choice='none'", ('required', ['final_result'])),
        output_tools=['final_result'],
    ),
    Case(
        'none-output-no-direct-no-func',
        'none',
        'required',
        function_tools=[],
        output_tools=['final_result'],
    ),
    Case(
        'empty-output-no-direct-no-func',
        [],
        'required',
        function_tools=[],
        output_tools=['final_result'],
    ),
    # === None/Empty list: no output tools ===
    Case('none-no-output-text', 'none', 'none', allow_text_output=True),
    Case('empty-no-output-text', [], 'none', allow_text_output=True),
    # === Required behavior ===
    Case('required-func-only', 'required', 'required'),
    Case(
        'required-with-output',
        'required',
        'required',
        output_tools=['final_result'],
    ),
    Case(
        'required-no-tools-error',
        'required',
        ExpectError('no function tools'),
        function_tools=[],
    ),
    # === Tool list behavior ===
    Case(
        'list-single',
        ['tool_a'],
        ('required', ['tool_a']),
        function_tools=['tool_a', 'tool_b'],
    ),
    Case(
        'list-multiple',
        ['tool_a', 'tool_b'],
        ('required', ['tool_a', 'tool_b']),
        function_tools=['tool_a', 'tool_b', 'tool_c'],
    ),
    Case(
        'list-all-tools',
        ['tool_a', 'tool_b'],
        'required',
        function_tools=['tool_a', 'tool_b'],
    ),
    Case(
        'list-dedup',
        ['tool_b', 'tool_a', 'tool_b'],
        ('required', ['tool_b', 'tool_a']),
        function_tools=['tool_a', 'tool_b', 'tool_c'],
    ),
    Case(
        'list-invalid-error',
        ['nonexistent'],
        ExpectError('Invalid tool names'),
        function_tools=['tool_a', 'tool_b'],
    ),
    Case(
        'list-output-tool',
        ['final_result'],
        ('required', ['final_result']),
        output_tools=['final_result'],
    ),
    # === ToolsPlusOutput behavior ===
    Case(
        'tpo-text-allowed',
        ToolOrOutput(function_tools=['tool_a']),
        ('auto', ['tool_a', 'final_result']),
        function_tools=['tool_a', 'tool_b'],
        output_tools=['final_result'],
        allow_text_output=True,
    ),
    Case(
        'tpo-no-text',
        ToolOrOutput(function_tools=['tool_a']),
        ('required', ['tool_a', 'final_result']),
        function_tools=['tool_a', 'tool_b'],
        output_tools=['final_result'],
    ),
    Case(
        'tpo-multiple',
        ToolOrOutput(function_tools=['tool_a', 'tool_b']),
        ('auto', ['tool_a', 'tool_b', 'final_result']),
        function_tools=['tool_a', 'tool_b', 'tool_c'],
        output_tools=['final_result'],
        allow_text_output=True,
    ),
    Case(
        'tpo-all-tools',
        ToolOrOutput(function_tools=['tool_a']),
        'required',
        function_tools=['tool_a'],
        output_tools=['final_result'],
    ),
    Case(
        'tpo-empty-text',
        ToolOrOutput(function_tools=[]),
        ExpectWarning('empty function_tools', 'auto'),
        output_tools=['final_result'],
        allow_text_output=True,
    ),
    Case(
        'tpo-empty-no-text',
        ToolOrOutput(function_tools=[]),
        ExpectWarning('empty function_tools', 'required'),
        output_tools=['final_result'],
    ),
    Case(
        'tpo-empty-no-output',
        ToolOrOutput(function_tools=[]),
        ExpectWarning('empty function_tools', 'none'),
    ),
    Case(
        'tpo-no-output',
        ToolOrOutput(function_tools=['tool_a']),
        ('auto', ['tool_a']),
        function_tools=['tool_a', 'tool_b'],
        allow_text_output=True,
    ),
    Case(
        'tpo-invalid-error',
        ToolOrOutput(function_tools=['nonexistent']),
        ExpectError('Invalid tool names'),
        output_tools=['final_result'],
    ),
    Case(
        'tpo-dedup',
        ToolOrOutput(function_tools=['tool_b', 'tool_a', 'tool_b']),
        ('auto', ['tool_b', 'tool_a', 'final_result']),
        function_tools=['tool_a', 'tool_b', 'tool_c'],
        output_tools=['final_result'],
        allow_text_output=True,
    ),
]


# =============================================================================
# Parametrized test
# =============================================================================


@pytest.mark.parametrize('case', CASES, ids=lambda c: c.id)
def test_validate_tool_choice(case: Case):
    """Test validate_tool_choice() with all combinations of inputs and expected outputs."""
    settings: ModelSettings | None
    if case.none_settings:
        settings = None
    elif case.tool_choice is not None:
        settings = {'tool_choice': case.tool_choice}
    else:
        settings = {}

    params = make_params(
        function_tools=case.function_tools,
        output_tools=case.output_tools,
        allow_text_output=case.allow_text_output,
        allow_image_output=case.allow_image_output,
    )

    if isinstance(case.expected, ExpectError):
        with pytest.raises(UserError, match=case.expected.match):
            resolve_tool_choice(settings, params)
    elif isinstance(case.expected, ExpectWarning):
        with pytest.warns(UserWarning, match=case.expected.match):
            result = resolve_tool_choice(settings, params)
        assert result == case.expected.result
    else:
        result = resolve_tool_choice(settings, params)
        assert result == case.expected
