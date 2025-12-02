"""Tests for the centralized `resolve_tool_choice()` function.

These tests cover the common logic shared across all providers:
- String value resolution ('none', 'auto', 'required')
- List[str] validation and resolution
- Warning emission for tool_choice='none' with output tools
- Invalid tool name detection

Provider-specific tests (API format mapping) remain in their respective test files.
"""

from __future__ import annotations

import warnings

import pytest
from inline_snapshot import snapshot

from pydantic_ai.exceptions import UserError
from pydantic_ai.models import (
    ModelRequestParameters,
    ResolvedToolChoice,
    resolve_tool_choice,
)
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition


def make_tool(name: str) -> ToolDefinition:
    """Return a minimal `ToolDefinition` used throughout the tests."""
    return ToolDefinition(
        name=name,
        description=f'Tool {name}',
        parameters_json_schema={'type': 'object', 'properties': {}},
    )


class TestResolveToolChoiceNone:
    """Cases where `tool_choice` is unset in the settings."""

    def test_none_model_settings_returns_none(self) -> None:
        """`resolve_tool_choice` returns None when `model_settings` is None."""
        params = ModelRequestParameters()
        result = resolve_tool_choice(None, params)
        assert result is None

    def test_empty_model_settings_returns_none(self) -> None:
        """Empty `model_settings` dict should also yield None."""
        params = ModelRequestParameters()
        settings: ModelSettings = {}
        result = resolve_tool_choice(settings, params)
        assert result is None

    def test_tool_choice_not_set_returns_none(self) -> None:
        """`tool_choice` missing from settings keeps provider defaults."""
        params = ModelRequestParameters()
        settings: ModelSettings = {'temperature': 0.5}
        result = resolve_tool_choice(settings, params)
        assert result is None


class TestResolveToolChoiceStringValues:
    """String-valued `tool_choice` entries."""

    @pytest.mark.parametrize(
        'tool_choice,expected',
        [
            pytest.param('none', snapshot(ResolvedToolChoice(mode='none')), id='none'),
            pytest.param('auto', snapshot(ResolvedToolChoice(mode='auto')), id='auto'),
            pytest.param('required', snapshot(ResolvedToolChoice(mode='required')), id='required'),
        ],
    )
    def test_string_values(self, tool_choice: str, expected: ResolvedToolChoice) -> None:
        """Valid string entries map directly to their resolved form."""
        params = ModelRequestParameters(function_tools=[make_tool('my_tool')])
        settings: ModelSettings = {'tool_choice': tool_choice}  # type: ignore
        result = resolve_tool_choice(settings, params)
        assert result == expected


class TestResolveToolChoiceSpecificTools:
    """List-based tool_choice entries."""

    def test_single_valid_tool(self) -> None:
        """Single tool names remain in the returned result."""
        params = ModelRequestParameters(function_tools=[make_tool('tool_a'), make_tool('tool_b')])
        settings: ModelSettings = {'tool_choice': ['tool_a']}
        result = resolve_tool_choice(settings, params)
        assert result == snapshot(ResolvedToolChoice(mode='specific', tool_names=['tool_a']))

    def test_multiple_valid_tools(self) -> None:
        """Multiple valid names stay in insertion order."""
        params = ModelRequestParameters(function_tools=[make_tool('tool_a'), make_tool('tool_b'), make_tool('tool_c')])
        settings: ModelSettings = {'tool_choice': ['tool_a', 'tool_b']}
        result = resolve_tool_choice(settings, params)
        assert result == snapshot(ResolvedToolChoice(mode='specific', tool_names=['tool_a', 'tool_b']))

    def test_invalid_tool_name_raises_user_error(self) -> None:
        """Unknown names raise a UserError."""
        params = ModelRequestParameters(function_tools=[make_tool('my_tool')])
        settings: ModelSettings = {'tool_choice': ['nonexistent_tool']}

        with pytest.raises(UserError, match='Invalid tool names in tool_choice'):
            resolve_tool_choice(settings, params)

    def test_mixed_valid_and_invalid_tools(self) -> None:
        """Mixed valid/invalid names still raise."""
        params = ModelRequestParameters(function_tools=[make_tool('valid_tool')])
        settings: ModelSettings = {'tool_choice': ['valid_tool', 'invalid_tool']}

        with pytest.raises(UserError, match='invalid_tool'):
            resolve_tool_choice(settings, params)

    def test_no_function_tools_available(self) -> None:
        """Requesting specific tools without registered ones errors."""
        params = ModelRequestParameters()
        settings: ModelSettings = {'tool_choice': ['some_tool']}

        with pytest.raises(UserError, match='Available function tools: none'):
            resolve_tool_choice(settings, params)

    def test_empty_list_raises_user_error(self) -> None:
        """Empty lists are not allowed."""
        params = ModelRequestParameters(function_tools=[make_tool('my_tool')])
        settings: ModelSettings = {'tool_choice': []}

        with pytest.raises(UserError, match='tool_choice cannot be an empty list'):
            resolve_tool_choice(settings, params)


class TestResolveToolChoiceOutputToolsWarning:
    """Safety checks when `tool_choice='none'` conflicts with output tools."""

    def test_none_with_output_tools_warns(self) -> None:
        """`tool_choice='none'` issues a warning when output tools exist."""
        output_tool = make_tool('final_result')
        params = ModelRequestParameters(output_tools=[output_tool])
        settings: ModelSettings = {'tool_choice': 'none'}

        with pytest.warns(UserWarning, match='tool_choice=.none. is set but output tools are required'):
            result = resolve_tool_choice(settings, params, stacklevel=2)

        assert result == snapshot(ResolvedToolChoice(mode='none', output_tools_fallback=True))

    def test_none_without_output_tools_no_warning(self) -> None:
        """No warning when `tool_choice='none'` and no output tools exist."""
        params = ModelRequestParameters(function_tools=[make_tool('my_tool')])
        settings: ModelSettings = {'tool_choice': 'none'}

        with warnings.catch_warnings():
            warnings.simplefilter('error')
            result = resolve_tool_choice(settings, params, stacklevel=2)

        assert result == snapshot(ResolvedToolChoice(mode='none'))
