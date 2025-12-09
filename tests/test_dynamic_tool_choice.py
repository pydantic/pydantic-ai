"""Tests for dynamic tool_choice feature (callable and force_first_request)."""

from __future__ import annotations

import pytest

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.settings import ToolChoiceValue


class TestCallableToolChoice:
    """Tests for callable tool_choice in model_settings."""

    def test_callable_tool_choice_first_request(self):
        """Test that callable tool_choice can force a tool on the first request."""
        tool_call_count = 0

        def my_tool_choice(ctx: RunContext[None]) -> ToolChoiceValue:
            if ctx.run_step == 1:
                return ['get_weather']
            return None

        agent = Agent(TestModel(), model_settings={'tool_choice': my_tool_choice})

        @agent.tool
        def get_weather(ctx: RunContext[None], city: str) -> str:
            nonlocal tool_call_count
            tool_call_count += 1
            return f'Weather in {city} is sunny'

        result = agent.run_sync('What is the weather in London?')
        assert tool_call_count == 1
        assert 'sunny' in result.output.lower() or 'weather' in result.output.lower()

    def test_callable_tool_choice_returns_none(self):
        """Test that callable returning None uses default behavior."""

        def always_none_tool_choice(ctx: RunContext[None]) -> ToolChoiceValue:
            return None

        agent = Agent(
            TestModel(),
            model_settings={'tool_choice': always_none_tool_choice},
        )

        @agent.tool_plain
        def my_tool() -> str:
            return 'tool result'

        result = agent.run_sync('Hello')
        assert result.output is not None

    def test_callable_tool_choice_evaluated_per_request(self):
        """Test that callable is evaluated for each model request."""
        evaluated_steps: list[int] = []

        def tracking_tool_choice(ctx: RunContext[None]) -> ToolChoiceValue:
            evaluated_steps.append(ctx.run_step)
            if ctx.run_step == 1:
                return ['step_one_tool']
            return 'auto'

        agent = Agent(
            TestModel(),
            model_settings={'tool_choice': tracking_tool_choice},
        )

        @agent.tool_plain
        def step_one_tool() -> str:
            return 'step one result'

        agent.run_sync('Multi-step task')
        # Callable is evaluated on step 1 (tool call) and step 2 (response)
        assert 1 in evaluated_steps
        assert 2 in evaluated_steps

    def test_callable_tool_choice_required(self):
        """Test callable returning 'required' forces tool use."""

        def force_required(ctx: RunContext[None]) -> ToolChoiceValue:
            return 'required'

        agent = Agent(TestModel(), model_settings={'tool_choice': force_required})

        @agent.tool_plain
        def my_tool() -> str:
            return 'tool was called'

        result = agent.run_sync('Test')
        assert result.output is not None


class TestForceFirstRequest:
    """Tests for force_first_request parameter on tools."""

    def test_force_first_request_single_tool(self):
        """Test that force_first_request forces a tool on the first request."""
        tool_called = False

        agent = Agent(TestModel())

        @agent.tool(force_first_request=True)
        def forced_tool(ctx: RunContext[None]) -> str:
            nonlocal tool_called
            tool_called = True
            return 'forced tool result'

        result = agent.run_sync('Test')
        assert tool_called
        assert result.output is not None

    def test_force_first_request_multiple_tools(self):
        """Test multiple tools with force_first_request allows model to choose one."""
        called_tools: list[str] = []

        agent = Agent(TestModel())

        @agent.tool(force_first_request=True)
        def search_tool(ctx: RunContext[None], query: str) -> str:
            called_tools.append('search')
            return f'search result for {query}'

        @agent.tool(force_first_request=True)
        def lookup_tool(ctx: RunContext[None], item: str) -> str:
            called_tools.append('lookup')
            return f'lookup result for {item}'

        agent.run_sync('Find something')
        assert len(called_tools) >= 1

    def test_force_first_request_with_non_forced_tools(self):
        """Test force_first_request works alongside non-forced tools."""
        forced_called = False
        regular_called = False

        agent = Agent(TestModel())

        @agent.tool(force_first_request=True)
        def forced_tool(ctx: RunContext[None]) -> str:
            nonlocal forced_called
            forced_called = True
            return 'forced'

        @agent.tool
        def regular_tool(ctx: RunContext[None]) -> str:
            nonlocal regular_called
            regular_called = True
            return 'regular'

        result = agent.run_sync('Test')
        assert forced_called
        assert result.output is not None

    def test_force_first_request_ignored_when_callable_set(self):
        """Test that force_first_request is ignored when callable tool_choice is set."""

        def custom_tool_choice(ctx: RunContext[None]) -> ToolChoiceValue:
            return 'auto'

        agent = Agent(TestModel(), model_settings={'tool_choice': custom_tool_choice})

        @agent.tool(force_first_request=True)
        def forced_tool(ctx: RunContext[None]) -> str:
            return 'forced'

        with pytest.warns(UserWarning, match='force_first_request.*ignored'):
            result = agent.run_sync('Test')
            assert result.output is not None

    def test_force_first_request_with_static_tool_choice(self):
        """Test force_first_request is not ignored when static tool_choice is set."""
        forced_called = False

        agent = Agent(TestModel(), model_settings={'tool_choice': 'auto'})

        @agent.tool(force_first_request=True)
        def forced_tool(ctx: RunContext[None]) -> str:
            nonlocal forced_called
            forced_called = True
            return 'forced'

        agent.run_sync('Test')
        assert forced_called

    def test_force_first_request_falls_back_to_static_after_first(self):
        """Test that after first request, static tool_choice is used."""
        call_count = 0

        # This test uses a multi-turn conversation
        agent = Agent(
            TestModel(custom_output_text='need more info {call_count}'),
            model_settings={'tool_choice': 'auto'},
        )

        @agent.tool(force_first_request=True)
        def forced_tool(ctx: RunContext[None]) -> str:
            nonlocal call_count
            call_count += 1
            return 'forced result'

        agent.run_sync('Test')
        assert call_count >= 1


class TestForceFirstRequestOnToolPlain:
    """Tests for force_first_request on tool_plain decorator."""

    def test_tool_plain_force_first_request(self):
        """Test force_first_request works with tool_plain decorator."""
        tool_called = False

        agent = Agent(TestModel())

        @agent.tool_plain(force_first_request=True)
        def plain_forced_tool() -> str:
            nonlocal tool_called
            tool_called = True
            return 'plain forced'

        agent.run_sync('Test')
        assert tool_called


class TestToolClassForceFirstRequest:
    """Tests for force_first_request parameter on Tool class."""

    def test_tool_class_force_first_request(self):
        """Test force_first_request works when using Tool class directly."""
        from pydantic_ai import Tool

        tool_called = False

        def my_function() -> str:
            nonlocal tool_called
            tool_called = True
            return 'tool result'

        tool = Tool(my_function, force_first_request=True)
        agent = Agent(TestModel(), tools=[tool])

        agent.run_sync('Test')
        assert tool_called


class TestRunStepInContext:
    """Tests verifying run_step behavior in RunContext."""

    def test_run_step_starts_at_one(self):
        """Test that run_step starts at 1 for the first model request."""
        observed_steps: list[int] = []

        def check_step(ctx: RunContext[None]) -> ToolChoiceValue:
            observed_steps.append(ctx.run_step)
            return None

        agent = Agent(TestModel(), model_settings={'tool_choice': check_step})

        @agent.tool_plain
        def dummy_tool() -> str:
            return 'dummy'

        agent.run_sync('Test')
        # First observed step should be 1 (the first model request)
        assert observed_steps[0] == 1

    def test_run_step_increments_per_request(self):
        """Test that run_step increments with each model request."""
        observed_steps: list[int] = []

        def track_steps(ctx: RunContext[None]) -> ToolChoiceValue:
            observed_steps.append(ctx.run_step)
            if ctx.run_step == 1:
                return ['continue_tool']
            return 'auto'

        agent = Agent(
            TestModel(),
            model_settings={'tool_choice': track_steps},
        )

        @agent.tool_plain
        def continue_tool() -> str:
            return 'continue'

        agent.run_sync('Multi-step test')
        # Step 1: tool call, Step 2: response
        assert observed_steps == [1, 2]


class TestTypeExports:
    """Tests verifying type exports are available."""

    def test_tool_choice_value_exported(self):
        """Test ToolChoiceValue is exported from pydantic_ai."""
        from pydantic_ai import ToolChoiceValue

        assert ToolChoiceValue is not None

    def test_tool_choice_func_exported(self):
        """Test ToolChoiceFunc is exported from pydantic_ai."""
        from pydantic_ai import ToolChoiceFunc

        assert ToolChoiceFunc is not None
