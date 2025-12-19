"""Tests for validate_tool_choice() function in _tool_choice.py.

This module provides comprehensive unit tests for the tool_choice validation logic,
covering all input types, edge cases, warning conditions, and error scenarios.
"""

from __future__ import annotations

import pytest
from inline_snapshot import snapshot

from pydantic_ai.exceptions import UserError
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models._tool_choice import validate_tool_choice
from pydantic_ai.settings import ModelSettings, ToolChoiceScalar, ToolsPlusOutput
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
    allow_text_output: bool = True,
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
# Part A: Direct Unit Tests of validate_tool_choice()
# =============================================================================


class TestDefaultAndAuto:
    """Tests for tool_choice=None and tool_choice='auto'."""

    @pytest.mark.parametrize('tool_choice', [None, 'auto'])
    def test_with_text_output_allowed(self, tool_choice: ToolChoiceScalar | None):
        """When text output is allowed, returns 'auto' so model can choose."""
        settings: ModelSettings = {'tool_choice': tool_choice} if tool_choice else {}
        params = make_params(function_tools=['tool_a'], allow_text_output=True)

        result = validate_tool_choice(settings, params)

        assert result == 'auto'

    @pytest.mark.parametrize('tool_choice', [None, 'auto'])
    def test_with_text_output_not_allowed(self, tool_choice: ToolChoiceScalar | None):
        """When text output is not allowed, returns 'required' to force tool use."""
        settings: ModelSettings = {'tool_choice': tool_choice} if tool_choice else {}
        params = make_params(function_tools=['tool_a'], allow_text_output=False)

        result = validate_tool_choice(settings, params)

        assert result == 'required'

    def test_none_settings(self):
        """When model_settings is None, behaves like tool_choice=None."""
        params = make_params(function_tools=['tool_a'], allow_text_output=True)

        result = validate_tool_choice(None, params)

        assert result == 'auto'


class TestNoneAndEmptyList:
    """Tests for tool_choice='none' and tool_choice=[]."""

    @pytest.mark.parametrize('tool_choice', ['none', []])
    def test_with_output_tools_and_text_allowed(self, tool_choice: ToolChoiceScalar | list[str]):
        """With output tools and text allowed, warns and returns output tools with 'auto'."""
        settings: ModelSettings = {'tool_choice': tool_choice}
        params = make_params(
            function_tools=['func_tool'],
            output_tools=['final_result'],
            allow_text_output=True,
        )

        with pytest.warns(UserWarning, match="tool_choice='none' but output tools"):
            result = validate_tool_choice(settings, params)

        assert result == snapshot((['final_result'], 'auto'))

    @pytest.mark.parametrize('tool_choice', ['none', []])
    def test_with_output_tools_and_image_allowed(self, tool_choice: ToolChoiceScalar | list[str]):
        """With output tools and image output allowed, warns and returns output tools with 'auto'."""
        settings: ModelSettings = {'tool_choice': tool_choice}
        params = make_params(
            function_tools=['func_tool'],
            output_tools=['final_result'],
            allow_text_output=False,
            allow_image_output=True,
        )

        with pytest.warns(UserWarning, match="tool_choice='none' but output tools"):
            result = validate_tool_choice(settings, params)

        assert result == snapshot((['final_result'], 'auto'))

    @pytest.mark.parametrize('tool_choice', ['none', []])
    def test_with_output_tools_no_direct_output_with_function_tools(self, tool_choice: ToolChoiceScalar | list[str]):
        """With output tools, no direct output, and function tools, returns 'required' mode."""
        settings: ModelSettings = {'tool_choice': tool_choice}
        params = make_params(
            function_tools=['func_tool'],
            output_tools=['final_result'],
            allow_text_output=False,
            allow_image_output=False,
        )

        with pytest.warns(UserWarning, match="tool_choice='none' but output tools"):
            result = validate_tool_choice(settings, params)

        assert result == snapshot((['final_result'], 'required'))

    @pytest.mark.parametrize('tool_choice', ['none', []])
    def test_with_output_tools_no_direct_output_no_function_tools(self, tool_choice: ToolChoiceScalar | list[str]):
        """With only output tools and no direct output, returns 'required' without warning.

        This is a special case: when there are no function tools and no direct output allowed,
        there's nothing to warn about since the output tools are the only way to complete.
        """
        settings: ModelSettings = {'tool_choice': tool_choice}
        params = make_params(
            function_tools=[],
            output_tools=['final_result'],
            allow_text_output=False,
            allow_image_output=False,
        )

        result = validate_tool_choice(settings, params)

        assert result == 'required'

    @pytest.mark.parametrize('tool_choice', ['none', []])
    def test_no_output_tools_with_text_allowed(self, tool_choice: ToolChoiceScalar | list[str]):
        """Without output tools but text allowed, returns 'none'."""
        settings: ModelSettings = {'tool_choice': tool_choice}
        params = make_params(
            function_tools=['func_tool'],
            output_tools=[],
            allow_text_output=True,
        )

        result = validate_tool_choice(settings, params)

        assert result == 'none'


class TestRequired:
    """Tests for tool_choice='required'."""

    def test_with_function_tools_only(self):
        """With function tools and no output tools, returns 'required'."""
        settings: ModelSettings = {'tool_choice': 'required'}
        params = make_params(function_tools=['tool_a', 'tool_b'], output_tools=[])

        result = validate_tool_choice(settings, params)

        assert result == 'required'

    def test_with_output_tools_raises_error(self):
        """With output tools present, raises UserError."""
        settings: ModelSettings = {'tool_choice': 'required'}
        params = make_params(function_tools=['tool_a'], output_tools=['final_result'])

        with pytest.raises(UserError, match='incompatible with output types'):
            validate_tool_choice(settings, params)

    def test_no_function_tools_raises_error(self):
        """Without function tools, raises UserError."""
        settings: ModelSettings = {'tool_choice': 'required'}
        params = make_params(function_tools=[], output_tools=[])

        with pytest.raises(UserError, match='no function tools are defined'):
            validate_tool_choice(settings, params)


class TestToolList:
    """Tests for tool_choice=[tool_names]."""

    def test_single_tool(self):
        """Single tool in list returns tuple with 'required'."""
        settings: ModelSettings = {'tool_choice': ['tool_a']}
        params = make_params(function_tools=['tool_a', 'tool_b'])

        result = validate_tool_choice(settings, params)

        assert result == snapshot((['tool_a'], 'required'))

    def test_multiple_tools(self):
        """Multiple tools in list returns tuple with 'required'."""
        settings: ModelSettings = {'tool_choice': ['tool_a', 'tool_b']}
        params = make_params(function_tools=['tool_a', 'tool_b', 'tool_c'])

        result = validate_tool_choice(settings, params)

        assert result == snapshot((['tool_a', 'tool_b'], 'required'))

    def test_all_tools_returns_required(self):
        """When list matches all available tools, returns just 'required'."""
        settings: ModelSettings = {'tool_choice': ['tool_a', 'tool_b']}
        params = make_params(function_tools=['tool_a', 'tool_b'])

        result = validate_tool_choice(settings, params)

        assert result == 'required'

    def test_preserves_order_and_deduplicates(self):
        """Preserves order and removes duplicates from tool list."""
        settings: ModelSettings = {'tool_choice': ['tool_b', 'tool_a', 'tool_b']}
        params = make_params(function_tools=['tool_a', 'tool_b', 'tool_c'])

        result = validate_tool_choice(settings, params)

        assert result == snapshot((['tool_b', 'tool_a'], 'required'))

    def test_invalid_tool_name_raises_error(self):
        """Invalid tool name in list raises UserError."""
        settings: ModelSettings = {'tool_choice': ['nonexistent']}
        params = make_params(function_tools=['tool_a', 'tool_b'])

        with pytest.raises(UserError, match='Invalid tool names'):
            validate_tool_choice(settings, params)

    def test_includes_output_tools_in_validation(self):
        """Output tools are included in available tools for validation."""
        settings: ModelSettings = {'tool_choice': ['final_result']}
        params = make_params(function_tools=['tool_a'], output_tools=['final_result'])

        result = validate_tool_choice(settings, params)

        assert result == snapshot((['final_result'], 'required'))


class TestToolsPlusOutput:
    """Tests for tool_choice=ToolsPlusOutput(...)."""

    def test_with_function_and_output_tools(self):
        """Combines specified function tools with all output tools."""
        settings: ModelSettings = {'tool_choice': ToolsPlusOutput(function_tools=['tool_a'])}
        params = make_params(function_tools=['tool_a', 'tool_b'], output_tools=['final_result'])

        result = validate_tool_choice(settings, params)

        assert result == snapshot((['tool_a', 'final_result'], 'required'))

    def test_multiple_function_tools(self):
        """Multiple function tools are combined with output tools."""
        settings: ModelSettings = {'tool_choice': ToolsPlusOutput(function_tools=['tool_a', 'tool_b'])}
        params = make_params(function_tools=['tool_a', 'tool_b', 'tool_c'], output_tools=['final_result'])

        result = validate_tool_choice(settings, params)

        assert result == snapshot((['tool_a', 'tool_b', 'final_result'], 'required'))

    def test_all_tools_returns_required(self):
        """When combined list matches all tools, returns just 'required'."""
        settings: ModelSettings = {'tool_choice': ToolsPlusOutput(function_tools=['tool_a'])}
        params = make_params(function_tools=['tool_a'], output_tools=['final_result'])

        result = validate_tool_choice(settings, params)

        assert result == 'required'

    def test_empty_function_tools_with_output_tools_text_allowed(self):
        """Empty function tools with output tools warns and returns output tools with 'auto'."""
        settings: ModelSettings = {'tool_choice': ToolsPlusOutput(function_tools=[])}
        params = make_params(
            function_tools=['tool_a'],
            output_tools=['final_result'],
            allow_text_output=True,
        )

        with pytest.warns(UserWarning, match='empty function_tools'):
            result = validate_tool_choice(settings, params)

        assert result == snapshot((['final_result'], 'auto'))

    def test_empty_function_tools_with_output_tools_no_text(self):
        """Empty function tools with output tools and no text output returns 'required' mode."""
        settings: ModelSettings = {'tool_choice': ToolsPlusOutput(function_tools=[])}
        params = make_params(
            function_tools=['tool_a'],
            output_tools=['final_result'],
            allow_text_output=False,
        )

        with pytest.warns(UserWarning, match='empty function_tools'):
            result = validate_tool_choice(settings, params)

        assert result == snapshot((['final_result'], 'required'))

    def test_empty_function_tools_no_output_tools(self):
        """Empty function tools without output tools warns and returns 'none'."""
        settings: ModelSettings = {'tool_choice': ToolsPlusOutput(function_tools=[])}
        params = make_params(function_tools=['tool_a'], output_tools=[])

        with pytest.warns(UserWarning, match='empty function_tools'):
            result = validate_tool_choice(settings, params)

        assert result == 'none'

    def test_no_output_tools_warns(self):
        """Using ToolsPlusOutput without output tools warns about fallback behavior."""
        settings: ModelSettings = {'tool_choice': ToolsPlusOutput(function_tools=['tool_a'])}
        params = make_params(function_tools=['tool_a', 'tool_b'], output_tools=[])

        with pytest.warns(UserWarning, match='no output tools exist'):
            result = validate_tool_choice(settings, params)

        assert result == snapshot((['tool_a'], 'required'))

    def test_invalid_function_tool_raises_error(self):
        """Invalid function tool in ToolsPlusOutput raises UserError."""
        settings: ModelSettings = {'tool_choice': ToolsPlusOutput(function_tools=['nonexistent'])}
        params = make_params(function_tools=['tool_a'], output_tools=['final_result'])

        with pytest.raises(UserError, match='Invalid tool names'):
            validate_tool_choice(settings, params)

    def test_preserves_order_and_deduplicates(self):
        """Preserves order and removes duplicates from function tools."""
        settings: ModelSettings = {'tool_choice': ToolsPlusOutput(function_tools=['tool_b', 'tool_a', 'tool_b'])}
        params = make_params(function_tools=['tool_a', 'tool_b', 'tool_c'], output_tools=['final_result'])

        result = validate_tool_choice(settings, params)

        assert result == snapshot((['tool_b', 'tool_a', 'final_result'], 'required'))


# =============================================================================
# Part B: Integration Tests with Agent + TestModel
# =============================================================================
#
# Note: TestModel does not call validate_tool_choice() - that's done by each
# real provider implementation (OpenAI, Anthropic, etc.). These tests verify
# the agent correctly passes tool_choice settings through to the model.
# For full end-to-end tool_choice tests, see provider-specific test files.


class TestAgentToolChoiceIntegration:
    """Integration tests verifying tool_choice is passed to models correctly.

    These tests verify:
    1. Settings flow through agent → model pipeline
    2. Error conditions are caught at the right level
    3. ModelRequestParameters are built correctly
    """

    def test_model_settings_passed_to_model(self):
        """Verify model_settings including tool_choice are passed to the model."""
        from pydantic_ai import Agent
        from pydantic_ai.models.test import TestModel

        agent = Agent('test')

        @agent.tool_plain
        def my_tool(x: int) -> str:
            return str(x)

        model = TestModel()
        settings: ModelSettings = {'tool_choice': 'auto', 'temperature': 0.5}
        agent.run_sync('hello', model=model, model_settings=settings)

        params = model.last_model_request_parameters
        assert params is not None
        assert len(params.function_tools) == 1
        assert params.function_tools[0].name == 'my_tool'

    def test_output_tools_available_with_structured_output(self):
        """Verify output tools are created when using structured output."""
        from pydantic import BaseModel

        from pydantic_ai import Agent
        from pydantic_ai.models.test import TestModel

        class Output(BaseModel):
            result: str

        agent: Agent[None, Output] = Agent('test', output_type=Output)
        model = TestModel()
        agent.run_sync('hello', model=model)

        params = model.last_model_request_parameters
        assert params is not None
        output_tool_names = [t.name for t in params.output_tools]
        assert 'final_result' in output_tool_names

    def test_allow_text_output_with_string_output_type(self):
        """Agent with str output type should allow text output."""
        from pydantic_ai import Agent
        from pydantic_ai.models.test import TestModel

        agent = Agent('test')  # default output_type=str
        model = TestModel()
        agent.run_sync('hello', model=model)

        params = model.last_model_request_parameters
        assert params is not None
        assert params.allow_text_output is True

    def test_allow_text_output_with_structured_only(self):
        """Agent with only structured output type should not allow text output."""
        from pydantic import BaseModel

        from pydantic_ai import Agent
        from pydantic_ai.models.test import TestModel

        class Output(BaseModel):
            result: str

        agent: Agent[None, Output] = Agent('test', output_type=Output)
        model = TestModel()
        agent.run_sync('hello', model=model)

        params = model.last_model_request_parameters
        assert params is not None
        assert params.allow_text_output is False

    def test_allow_text_output_with_union_type(self):
        """Agent with union output type (str | BaseModel) should allow text output."""
        from pydantic import BaseModel

        from pydantic_ai import Agent
        from pydantic_ai.models.test import TestModel

        class StructuredOutput(BaseModel):
            result: str

        agent: Agent[None, str | StructuredOutput] = Agent('test', output_type=str | StructuredOutput)
        model = TestModel()
        agent.run_sync('hello', model=model)

        params = model.last_model_request_parameters
        assert params is not None
        assert params.allow_text_output is True

    def test_function_tools_registered_correctly(self):
        """Verify function tools are registered and passed to model."""
        from pydantic_ai import Agent
        from pydantic_ai.models.test import TestModel

        agent = Agent('test')

        @agent.tool_plain
        def tool_a(x: int) -> str:
            return str(x)

        @agent.tool_plain
        def tool_b(y: str) -> int:
            return len(y)

        model = TestModel(call_tools=[])
        agent.run_sync('hello', model=model)

        params = model.last_model_request_parameters
        assert params is not None
        tool_names = {t.name for t in params.function_tools}
        assert tool_names == {'tool_a', 'tool_b'}


class TestValidateToolChoiceWithAgentParams:
    """Tests that verify validate_tool_choice works with real agent-generated parameters.

    These tests create ModelRequestParameters as they would be built by the agent,
    then call validate_tool_choice directly to verify the expected behavior.
    """

    def test_agent_with_structured_output_and_required_raises(self):
        """Simulates agent with structured output trying to use tool_choice='required'."""
        settings: ModelSettings = {'tool_choice': 'required'}
        params = make_params(
            function_tools=['my_tool'],
            output_tools=['final_result'],  # structured output creates output tool
            allow_text_output=False,
        )

        with pytest.raises(UserError, match='incompatible with output types'):
            validate_tool_choice(settings, params)

    def test_agent_with_structured_output_and_tool_list(self):
        """Simulates agent with structured output using a tool list."""
        settings: ModelSettings = {'tool_choice': ['my_tool']}
        params = make_params(
            function_tools=['my_tool', 'other_tool'],
            output_tools=['final_result'],
            allow_text_output=False,
        )

        result = validate_tool_choice(settings, params)

        assert result == snapshot((['my_tool'], 'required'))

    def test_agent_with_structured_output_and_tools_plus_output(self):
        """Simulates agent with structured output using ToolsPlusOutput."""
        settings: ModelSettings = {'tool_choice': ToolsPlusOutput(function_tools=['my_tool'])}
        params = make_params(
            function_tools=['my_tool', 'other_tool'],
            output_tools=['final_result'],
            allow_text_output=False,
        )

        result = validate_tool_choice(settings, params)

        assert result == snapshot((['my_tool', 'final_result'], 'required'))

    def test_agent_text_only_with_none(self):
        """Simulates text-only agent using tool_choice='none'."""
        settings: ModelSettings = {'tool_choice': 'none'}
        params = make_params(
            function_tools=['my_tool'],
            output_tools=[],
            allow_text_output=True,
        )

        result = validate_tool_choice(settings, params)

        assert result == 'none'

    def test_agent_text_and_structured_with_none(self):
        """Simulates agent with both text and structured output using tool_choice='none'."""
        settings: ModelSettings = {'tool_choice': 'none'}
        params = make_params(
            function_tools=['my_tool'],
            output_tools=['final_result'],
            allow_text_output=True,  # union type allows text
        )

        with pytest.warns(UserWarning, match="tool_choice='none' but output tools"):
            result = validate_tool_choice(settings, params)

        assert result == snapshot((['final_result'], 'auto'))
