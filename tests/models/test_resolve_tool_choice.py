"""Tests for the centralized resolve_tool_choice() function.

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
    """Create a simple tool definition for testing."""
    return ToolDefinition(
        name=name,
        description=f'Tool {name}',
        parameters_json_schema={'type': 'object', 'properties': {}},
    )


class TestResolveToolChoiceNone:
    """Tests for when tool_choice is not set."""

    def test_none_model_settings_returns_none(self) -> None:
        """When model_settings is None, resolve_tool_choice returns None."""
        params = ModelRequestParameters()
        result = resolve_tool_choice(None, params)
        assert result is None

    def test_empty_model_settings_returns_none(self) -> None:
        """When model_settings is empty, resolve_tool_choice returns None."""
        params = ModelRequestParameters()
        settings: ModelSettings = {}
        result = resolve_tool_choice(settings, params)
        assert result is None

    def test_tool_choice_not_set_returns_none(self) -> None:
        """When tool_choice is not in model_settings, resolve_tool_choice returns None."""
        params = ModelRequestParameters()
        settings: ModelSettings = {'temperature': 0.5}
        result = resolve_tool_choice(settings, params)
        assert result is None


class TestResolveToolChoiceStringValues:
    """Tests for string tool_choice values."""

    @pytest.mark.parametrize(
        'tool_choice,expected',
        [
            pytest.param('none', snapshot(ResolvedToolChoice(mode='none')), id='none'),
            pytest.param('auto', snapshot(ResolvedToolChoice(mode='auto')), id='auto'),
            pytest.param('required', snapshot(ResolvedToolChoice(mode='required')), id='required'),
        ],
    )
    def test_string_values(self, tool_choice: str, expected: ResolvedToolChoice) -> None:
        """Test that string values are correctly resolved."""
        params = ModelRequestParameters(function_tools=[make_tool('my_tool')])
        settings: ModelSettings = {'tool_choice': tool_choice}  # type: ignore
        result = resolve_tool_choice(settings, params)
        assert result == expected


class TestResolveToolChoiceSpecificTools:
    """Tests for list[str] tool_choice values."""

    def test_single_valid_tool(self) -> None:
        """Test tool_choice with a single valid tool name."""
        params = ModelRequestParameters(function_tools=[make_tool('tool_a'), make_tool('tool_b')])
        settings: ModelSettings = {'tool_choice': ['tool_a']}
        result = resolve_tool_choice(settings, params)
        assert result == snapshot(ResolvedToolChoice(mode='specific', tool_names=['tool_a']))

    def test_multiple_valid_tools(self) -> None:
        """Test tool_choice with multiple valid tool names."""
        params = ModelRequestParameters(function_tools=[make_tool('tool_a'), make_tool('tool_b'), make_tool('tool_c')])
        settings: ModelSettings = {'tool_choice': ['tool_a', 'tool_b']}
        result = resolve_tool_choice(settings, params)
        assert result == snapshot(ResolvedToolChoice(mode='specific', tool_names=['tool_a', 'tool_b']))

    def test_invalid_tool_name_raises_user_error(self) -> None:
        """Test that invalid tool names raise UserError."""
        params = ModelRequestParameters(function_tools=[make_tool('my_tool')])
        settings: ModelSettings = {'tool_choice': ['nonexistent_tool']}

        with pytest.raises(UserError, match='Invalid tool names in tool_choice'):
            resolve_tool_choice(settings, params)

    def test_mixed_valid_and_invalid_tools(self) -> None:
        """Test that mix of valid and invalid tool names raises error."""
        params = ModelRequestParameters(function_tools=[make_tool('valid_tool')])
        settings: ModelSettings = {'tool_choice': ['valid_tool', 'invalid_tool']}

        with pytest.raises(UserError, match='invalid_tool'):
            resolve_tool_choice(settings, params)

    def test_no_function_tools_available(self) -> None:
        """Test error when specifying tools but none are registered."""
        params = ModelRequestParameters()
        settings: ModelSettings = {'tool_choice': ['some_tool']}

        with pytest.raises(UserError, match='Available function tools: none'):
            resolve_tool_choice(settings, params)

    def test_empty_list_raises_user_error(self) -> None:
        """Test tool_choice=[] raises UserError."""
        params = ModelRequestParameters(function_tools=[make_tool('my_tool')])
        settings: ModelSettings = {'tool_choice': []}

        with pytest.raises(UserError, match='tool_choice cannot be an empty list'):
            resolve_tool_choice(settings, params)


class TestResolveToolChoiceOutputToolsWarning:
    """Tests for tool_choice='none' with output tools."""

    def test_none_with_output_tools_warns(self) -> None:
        """Test that tool_choice='none' with output tools emits warning and sets fallback."""
        output_tool = make_tool('final_result')
        params = ModelRequestParameters(output_tools=[output_tool])
        settings: ModelSettings = {'tool_choice': 'none'}

        with pytest.warns(UserWarning, match='tool_choice=.none. is set but output tools are required'):
            result = resolve_tool_choice(settings, params, stacklevel=2)

        assert result == snapshot(ResolvedToolChoice(mode='none', output_tools_fallback=True))

    def test_none_without_output_tools_no_warning(self) -> None:
        """Test that tool_choice='none' without output tools does not warn."""
        params = ModelRequestParameters(function_tools=[make_tool('my_tool')])
        settings: ModelSettings = {'tool_choice': 'none'}

        with warnings.catch_warnings():
            warnings.simplefilter('error')
            result = resolve_tool_choice(settings, params, stacklevel=2)

        assert result == snapshot(ResolvedToolChoice(mode='none'))
