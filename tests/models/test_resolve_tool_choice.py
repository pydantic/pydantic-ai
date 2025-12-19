"""Tests for the centralized `validate_tool_choice()` function.

These tests cover the common logic shared across all providers:
- The return type: Literal['none', 'auto', 'required'] | tuple[list[str], Literal['auto', 'required']]
- String value resolution ('none', 'auto', 'required')
- List[str] validation and resolution
- Empty list treated as 'none'
- Invalid tool name detection

Provider-specific tests (API format mapping) remain in their respective test files.
"""

from __future__ import annotations

import pytest
from inline_snapshot import snapshot

from pydantic_ai.exceptions import UserError
from pydantic_ai.models import (
    ModelRequestParameters,
)
from pydantic_ai.models._tool_choice import validate_tool_choice
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
    """Cases where `tool_choice` is unset in the settings.

    When unset, tool_choice uses allow_text_output to determine result:
    - allow_text_output=True (default) → 'auto'
    - allow_text_output=False → 'required'
    """

    def test_none_model_settings_with_text_allowed_returns_auto(self) -> None:
        """Unset settings with allow_text_output=True returns 'auto'."""
        params = ModelRequestParameters(allow_text_output=True)
        result = validate_tool_choice(None, params)
        assert result == 'auto'

    def test_none_model_settings_with_text_disallowed_returns_required(self) -> None:
        """Unset settings with allow_text_output=False returns 'required'."""
        params = ModelRequestParameters(allow_text_output=False, function_tools=[make_tool('tool')])
        result = validate_tool_choice(None, params)
        assert result == 'required'

    def test_empty_model_settings_returns_auto(self) -> None:
        """Empty `model_settings` dict with text allowed yields 'auto'."""
        params = ModelRequestParameters(allow_text_output=True)
        settings: ModelSettings = {}
        result = validate_tool_choice(settings, params)
        assert result == 'auto'

    def test_tool_choice_not_set_returns_auto(self) -> None:
        """`tool_choice` missing from settings with text allowed returns 'auto'."""
        params = ModelRequestParameters(allow_text_output=True)
        settings: ModelSettings = {'temperature': 0.5}
        result = validate_tool_choice(settings, params)
        assert result == 'auto'


class TestValidateToolChoiceStringValues:
    """String-valued `tool_choice` entries."""

    def test_auto_with_text_allowed_returns_auto(self) -> None:
        """'auto' with allow_text_output=True returns 'auto'."""
        params = ModelRequestParameters(
            function_tools=[make_tool('my_tool')],
            allow_text_output=True,
        )
        settings: ModelSettings = {'tool_choice': 'auto'}
        result = validate_tool_choice(settings, params)
        assert result == 'auto'

    def test_auto_with_text_disallowed_returns_required(self) -> None:
        """'auto' with allow_text_output=False returns 'required'."""
        params = ModelRequestParameters(
            function_tools=[make_tool('my_tool')],
            allow_text_output=False,
        )
        settings: ModelSettings = {'tool_choice': 'auto'}
        result = validate_tool_choice(settings, params)
        assert result == 'required'

    def test_none_with_text_allowed_no_output_tools_returns_none(self) -> None:
        """'none' with allow_text_output=True and no output_tools returns 'none'."""
        params = ModelRequestParameters(
            function_tools=[make_tool('my_tool')],
            allow_text_output=True,
        )
        settings: ModelSettings = {'tool_choice': 'none'}
        result = validate_tool_choice(settings, params)
        assert result == 'none'

    def test_none_with_output_tools_text_allowed_returns_tuple(self) -> None:
        """'none' with output_tools and allow_text_output=True returns tuple with 'auto'."""
        params = ModelRequestParameters(
            function_tools=[make_tool('my_tool')],
            output_tools=[make_tool('output')],
            allow_text_output=True,
        )
        settings: ModelSettings = {'tool_choice': 'none'}
        result = validate_tool_choice(settings, params)
        assert result == snapshot((['output'], 'auto'))

    def test_none_with_output_tools_text_disallowed_returns_tuple(self) -> None:
        """'none' with output_tools and allow_text_output=False returns tuple with 'required'."""
        params = ModelRequestParameters(
            function_tools=[make_tool('my_tool')],
            output_tools=[make_tool('output')],
            allow_text_output=False,
        )
        settings: ModelSettings = {'tool_choice': 'none'}
        result = validate_tool_choice(settings, params)
        assert result == snapshot((['output'], 'required'))

    def test_required_with_function_tools_no_output_returns_required(self) -> None:
        """'required' with function tools only returns 'required'."""
        params = ModelRequestParameters(
            function_tools=[make_tool('my_tool')],
        )
        settings: ModelSettings = {'tool_choice': 'required'}
        result = validate_tool_choice(settings, params)
        assert result == 'required'

    def test_required_with_function_and_output_tools_returns_tuple(self) -> None:
        """'required' with both function and output tools returns tuple."""
        params = ModelRequestParameters(
            function_tools=[make_tool('func_tool')],
            output_tools=[make_tool('output_tool')],
        )
        settings: ModelSettings = {'tool_choice': 'required'}
        result = validate_tool_choice(settings, params)
        assert result == snapshot((['func_tool'], 'required'))

    def test_required_without_function_tools_raises_user_error(self) -> None:
        """'required' with no function tools raises UserError."""
        params = ModelRequestParameters(
            function_tools=[],
            output_tools=[make_tool('output_tool')],
        )
        settings: ModelSettings = {'tool_choice': 'required'}

        with pytest.raises(UserError, match='tool_choice.*required.*no function tools'):
            validate_tool_choice(settings, params)


class TestValidateToolChoiceSpecificTools:
    """List-based tool_choice entries."""

    def test_single_valid_tool_returns_tuple(self) -> None:
        """Single tool name returns tuple with 'required'."""
        params = ModelRequestParameters(function_tools=[make_tool('tool_a'), make_tool('tool_b')])
        settings: ModelSettings = {'tool_choice': ['tool_a']}
        result = validate_tool_choice(settings, params)
        assert result == snapshot((['tool_a'], 'required'))

    def test_multiple_valid_tools_returns_tuple(self) -> None:
        """Multiple valid names return tuple with 'required'."""
        params = ModelRequestParameters(function_tools=[make_tool('tool_a'), make_tool('tool_b'), make_tool('tool_c')])
        settings: ModelSettings = {'tool_choice': ['tool_a', 'tool_b']}
        result = validate_tool_choice(settings, params)
        # Note: order is not guaranteed in the list since it's converted to a set
        assert isinstance(result, tuple)
        assert set(result[0]) == {'tool_a', 'tool_b'}
        assert result[1] == 'required'

    def test_all_tools_selected_returns_required(self) -> None:
        """When all tools are selected, returns 'required' instead of tuple."""
        params = ModelRequestParameters(function_tools=[make_tool('tool_a'), make_tool('tool_b')])
        settings: ModelSettings = {'tool_choice': ['tool_a', 'tool_b']}
        result = validate_tool_choice(settings, params)
        assert result == 'required'

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
        """Empty list with text allowed returns 'none'."""
        params = ModelRequestParameters(
            function_tools=[make_tool('my_tool')],
            allow_text_output=True,
        )
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
        assert result == snapshot((['output_tool'], 'required'))

    def test_mixed_function_and_output_tools(self) -> None:
        """Both function and output tool names are accepted together."""
        params = ModelRequestParameters(
            function_tools=[make_tool('func_tool')],
            output_tools=[make_tool('output_tool')],
        )
        settings: ModelSettings = {'tool_choice': ['func_tool', 'output_tool']}
        result = validate_tool_choice(settings, params)
        # All tools selected returns 'required'
        assert result == 'required'

    def test_subset_of_mixed_tools_returns_tuple(self) -> None:
        """Subset of mixed tools returns tuple."""
        params = ModelRequestParameters(
            function_tools=[make_tool('func_tool1'), make_tool('func_tool2')],
            output_tools=[make_tool('output_tool')],
        )
        settings: ModelSettings = {'tool_choice': ['func_tool1', 'output_tool']}
        result = validate_tool_choice(settings, params)
        assert isinstance(result, tuple)
        assert set(result[0]) == {'func_tool1', 'output_tool'}
        assert result[1] == 'required'
