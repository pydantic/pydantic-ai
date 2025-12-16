"""Tests for the centralized `validate_tool_choice()` function.

These tests cover the common logic shared across all providers:
- String value resolution ('none', 'auto', 'required')
- List[str] validation and resolution
- Empty list treated as 'none'
- Invalid tool name detection
- filter_tools_for_choice helper function

Provider-specific tests (API format mapping) remain in their respective test files.
"""

from __future__ import annotations

import pytest
from inline_snapshot import snapshot

from pydantic_ai.exceptions import UserError
from pydantic_ai.models import (
    ModelRequestParameters,
)
from pydantic_ai.models._tool_choice import filter_tools_for_choice, validate_tool_choice
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition


def make_tool(name: str) -> ToolDefinition:
    """Return a minimal `ToolDefinition` used throughout the tests."""
    return ToolDefinition(
        name=name,
        description=f'Tool {name}',
        parameters_json_schema={'type': 'object', 'properties': {}},
    )


class TestValidateToolChoiceNone:
    """Cases where `tool_choice` is unset in the settings."""

    def test_none_model_settings_returns_none(self) -> None:
        """`validate_tool_choice` returns None when `model_settings` is None."""
        params = ModelRequestParameters()
        result = validate_tool_choice(None, params)
        assert result is None

    def test_empty_model_settings_returns_none(self) -> None:
        """Empty `model_settings` dict should also yield None."""
        params = ModelRequestParameters()
        settings: ModelSettings = {}
        result = validate_tool_choice(settings, params)
        assert result is None

    def test_tool_choice_not_set_returns_none(self) -> None:
        """`tool_choice` missing from settings keeps provider defaults."""
        params = ModelRequestParameters()
        settings: ModelSettings = {'temperature': 0.5}
        result = validate_tool_choice(settings, params)
        assert result is None


class TestValidateToolChoiceStringValues:
    """String-valued `tool_choice` entries."""

    @pytest.mark.parametrize(
        'tool_choice,expected',
        [
            pytest.param('none', 'none', id='none'),
            pytest.param('auto', 'auto', id='auto'),
            pytest.param('required', 'required', id='required'),
        ],
    )
    def test_string_values(self, tool_choice: str, expected: str) -> None:
        """Valid string entries are returned as-is."""
        params = ModelRequestParameters(function_tools=[make_tool('my_tool')])
        settings: ModelSettings = {'tool_choice': tool_choice}  # type: ignore
        result = validate_tool_choice(settings, params)
        assert result == expected


class TestValidateToolChoiceSpecificTools:
    """List-based tool_choice entries."""

    def test_single_valid_tool(self) -> None:
        """Single tool names remain in the returned result."""
        params = ModelRequestParameters(function_tools=[make_tool('tool_a'), make_tool('tool_b')])
        settings: ModelSettings = {'tool_choice': ['tool_a']}
        result = validate_tool_choice(settings, params)
        assert result == snapshot(['tool_a'])

    def test_multiple_valid_tools(self) -> None:
        """Multiple valid names stay in insertion order."""
        params = ModelRequestParameters(function_tools=[make_tool('tool_a'), make_tool('tool_b'), make_tool('tool_c')])
        settings: ModelSettings = {'tool_choice': ['tool_a', 'tool_b']}
        result = validate_tool_choice(settings, params)
        assert result == snapshot(['tool_a', 'tool_b'])

    def test_invalid_tool_name_raises_user_error(self) -> None:
        """Unknown names raise a UserError."""
        params = ModelRequestParameters(function_tools=[make_tool('my_tool')])
        settings: ModelSettings = {'tool_choice': ['nonexistent_tool']}

        with pytest.raises(UserError, match='Invalid tool names in `tool_choice`'):
            validate_tool_choice(settings, params)

    def test_mixed_valid_and_invalid_tools(self) -> None:
        """Mixed valid/invalid names still raise."""
        params = ModelRequestParameters(function_tools=[make_tool('valid_tool')])
        settings: ModelSettings = {'tool_choice': ['valid_tool', 'invalid_tool']}

        with pytest.raises(UserError, match='invalid_tool'):
            validate_tool_choice(settings, params)

    def test_no_tools_available(self) -> None:
        """Requesting specific tools without registered ones errors."""
        params = ModelRequestParameters()
        settings: ModelSettings = {'tool_choice': ['some_tool']}

        with pytest.raises(UserError, match='Available tools: none'):
            validate_tool_choice(settings, params)

    def test_empty_list_returns_none_mode(self) -> None:
        """Empty list is treated as tool_choice='none'."""
        params = ModelRequestParameters(function_tools=[make_tool('my_tool')])
        settings: ModelSettings = {'tool_choice': []}
        result = validate_tool_choice(settings, params)
        assert result == 'none'

    def test_output_tool_names_are_valid(self) -> None:
        """Output tool names are accepted in tool_choice list."""
        params = ModelRequestParameters(
            function_tools=[make_tool('func_tool')],
            output_tools=[make_tool('output_tool')],
        )
        settings: ModelSettings = {'tool_choice': ['output_tool']}
        result = validate_tool_choice(settings, params)
        assert result == snapshot(['output_tool'])

    def test_mixed_function_and_output_tools(self) -> None:
        """Both function and output tool names are accepted together."""
        params = ModelRequestParameters(
            function_tools=[make_tool('func_tool')],
            output_tools=[make_tool('output_tool')],
        )
        settings: ModelSettings = {'tool_choice': ['func_tool', 'output_tool']}
        result = validate_tool_choice(settings, params)
        assert result == snapshot(['func_tool', 'output_tool'])


class TestFilterToolsForChoice:
    """Tests for the filter_tools_for_choice helper function."""

    def test_none_returns_all_tools(self) -> None:
        """None tool_choice returns all tools."""
        func_tools = [make_tool('func1'), make_tool('func2')]
        output_tools = [make_tool('output1')]
        result = filter_tools_for_choice(None, func_tools, output_tools)
        assert [t.name for t in result] == ['func1', 'func2', 'output1']

    def test_auto_returns_all_tools(self) -> None:
        """'auto' tool_choice returns all tools."""
        func_tools = [make_tool('func1'), make_tool('func2')]
        output_tools = [make_tool('output1')]
        result = filter_tools_for_choice('auto', func_tools, output_tools)
        assert [t.name for t in result] == ['func1', 'func2', 'output1']

    def test_required_returns_only_function_tools(self) -> None:
        """'required' tool_choice returns only function tools."""
        func_tools = [make_tool('func1'), make_tool('func2')]
        output_tools = [make_tool('output1')]
        result = filter_tools_for_choice('required', func_tools, output_tools)
        assert [t.name for t in result] == ['func1', 'func2']

    def test_none_mode_returns_only_output_tools(self) -> None:
        """'none' tool_choice returns only output tools."""
        func_tools = [make_tool('func1'), make_tool('func2')]
        output_tools = [make_tool('output1')]
        result = filter_tools_for_choice('none', func_tools, output_tools)
        assert [t.name for t in result] == ['output1']

    def test_specific_list_returns_named_function_tools_only(self) -> None:
        """List of function tool names returns only those - no auto-inclusion of output tools."""
        func_tools = [make_tool('func1'), make_tool('func2'), make_tool('func3')]
        output_tools = [make_tool('output1')]
        result = filter_tools_for_choice(['func1', 'func3'], func_tools, output_tools)
        # output1 is NOT auto-included
        assert [t.name for t in result] == ['func1', 'func3']

    def test_specific_list_can_include_output_tools(self) -> None:
        """Output tools are included when explicitly named in tool_choice list."""
        func_tools = [make_tool('func1'), make_tool('func2')]
        output_tools = [make_tool('output1')]
        result = filter_tools_for_choice(['func1', 'output1'], func_tools, output_tools)
        assert [t.name for t in result] == ['func1', 'output1']

    def test_specific_list_output_tools_only(self) -> None:
        """When only output tools are named, only those are returned."""
        func_tools = [make_tool('func1'), make_tool('func2')]
        output_tools = [make_tool('output1'), make_tool('output2')]
        result = filter_tools_for_choice(['output1'], func_tools, output_tools)
        assert [t.name for t in result] == ['output1']

    def test_specific_list_preserves_order(self) -> None:
        """List filtering preserves the order of tools, not the list order."""
        func_tools = [make_tool('a'), make_tool('b'), make_tool('c')]
        output_tools = [make_tool('out1')]
        result = filter_tools_for_choice(['c', 'out1', 'a'], func_tools, output_tools)
        # Order is based on [func_tools, output_tools] order, not the list order
        assert [t.name for t in result] == ['a', 'c', 'out1']
