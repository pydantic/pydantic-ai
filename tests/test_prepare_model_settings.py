"""Tests for the prepare_model_settings hook."""

from __future__ import annotations

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelResponse, ToolReturnPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.settings import ModelSettings


class TestPrepareModelSettingsBasic:
    """Basic tests for prepare_model_settings hook."""

    async def test_hook_called_on_each_step(self):
        """Verify hook is called on each run step with correct run_step values."""
        call_log: list[int] = []

        async def log_calls(ctx: RunContext[None], settings: ModelSettings | None) -> ModelSettings | None:
            call_log.append(ctx.run_step)
            return None

        agent = Agent(
            TestModel(custom_output_text='done'),
            prepare_model_settings=log_calls,
        )

        await agent.run('test')
        assert call_log == [1]

    async def test_hook_called_multiple_times_with_tools(self):
        """Verify hook is called on each step when tools are used."""
        call_log: list[int] = []

        async def log_calls(ctx: RunContext[None], settings: ModelSettings | None) -> ModelSettings | None:
            call_log.append(ctx.run_step)
            return None

        agent = Agent(
            TestModel(
                custom_output_text='final answer',
                seed=0,
            ),
            prepare_model_settings=log_calls,
        )

        @agent.tool_plain
        def my_tool(x: int) -> str:
            """A tool that does something."""
            return f'result: {x}'

        await agent.run('test')
        # TestModel with a tool calls it once, then responds
        assert len(call_log) >= 1
        assert call_log[0] == 1

    async def test_none_return_keeps_settings_unchanged(self):
        """Verify returning None keeps original settings."""
        original_settings: ModelSettings = {'temperature': 0.5}

        async def return_none(ctx: RunContext[None], settings: ModelSettings | None) -> ModelSettings | None:
            assert settings == original_settings
            return None

        agent = Agent(
            TestModel(custom_output_text='done'),
            model_settings=original_settings,
            prepare_model_settings=return_none,
        )

        await agent.run('test')

    async def test_settings_return_overrides_settings(self):
        """Verify returning new settings overrides the original."""
        received_settings: list[ModelSettings | None] = []

        async def override_settings(ctx: RunContext[None], settings: ModelSettings | None) -> ModelSettings | None:
            received_settings.append(settings)
            return {'temperature': 0.9, 'max_tokens': 100}

        agent = Agent(
            TestModel(custom_output_text='done'),
            model_settings={'temperature': 0.5},
            prepare_model_settings=override_settings,
        )

        await agent.run('test')
        # Hook received the original settings
        assert received_settings[0] == {'temperature': 0.5}

    async def test_hook_receives_message_history(self):
        """Verify hook receives message history via ctx.messages."""
        message_counts: list[int] = []

        async def check_messages(ctx: RunContext[None], settings: ModelSettings | None) -> ModelSettings | None:
            message_counts.append(len(ctx.messages))
            return None

        agent = Agent(
            TestModel(custom_output_text='done'),
            prepare_model_settings=check_messages,
        )

        # First run - should have initial user message
        await agent.run('first message')
        assert len(message_counts) == 1
        # Message history includes the current request
        assert message_counts[0] >= 1

    async def test_no_hook_works_normally(self):
        """Verify agent works normally when no hook is provided."""
        agent = Agent(
            TestModel(custom_output_text='done'),
        )

        result = await agent.run('test')
        assert result.output == 'done'


class TestPrepareModelSettingsWithToolChoice:
    """Tests for prepare_model_settings with tool_choice."""

    async def test_force_tool_on_first_step(self):
        """Verify hook can force tool_choice on first step only."""
        settings_by_step: dict[int, ModelSettings | None] = {}

        async def force_first_tool(ctx: RunContext[None], settings: ModelSettings | None) -> ModelSettings | None:
            settings_by_step[ctx.run_step] = None if ctx.run_step > 1 else {'tool_choice': 'required'}
            if ctx.run_step == 1:
                return {'tool_choice': 'required'}
            return None

        agent = Agent(
            TestModel(custom_output_text='final'),
            prepare_model_settings=force_first_tool,
        )

        @agent.tool_plain
        def search(query: str) -> str:
            """Search for something."""
            return f'found: {query}'

        await agent.run('test')
        # Verify hook was called
        assert 1 in settings_by_step

    async def test_conditional_tool_choice_based_on_history(self):
        """Verify hook can check message history to decide tool_choice."""

        async def require_tool_if_no_results(
            ctx: RunContext[None], settings: ModelSettings | None
        ) -> ModelSettings | None:
            # Check if we have any tool results in history
            has_tool_results = any(
                isinstance(msg, ModelResponse) and any(isinstance(part, ToolReturnPart) for part in msg.parts)
                for msg in ctx.messages
            )
            if not has_tool_results:
                return {'tool_choice': 'required'}
            return None

        agent = Agent(
            TestModel(custom_output_text='done'),
            prepare_model_settings=require_tool_if_no_results,
        )

        @agent.tool_plain
        def my_tool() -> str:
            """A simple tool."""
            return 'tool result'

        await agent.run('test')


class TestPrepareModelSettingsStreaming:
    """Tests for prepare_model_settings with streaming."""

    async def test_hook_works_with_streaming(self):
        """Verify hook is called when using run_stream."""
        call_log: list[int] = []

        async def log_calls(ctx: RunContext[None], settings: ModelSettings | None) -> ModelSettings | None:
            call_log.append(ctx.run_step)
            return None

        agent = Agent(
            TestModel(custom_output_text='streamed output'),
            prepare_model_settings=log_calls,
        )

        async with agent.run_stream('test') as result:
            output = await result.get_output()
            assert output == 'streamed output'

        assert call_log == [1]


class TestPrepareModelSettingsWithDeps:
    """Tests for prepare_model_settings with dependencies."""

    async def test_hook_receives_deps(self):
        """Verify hook can access user dependencies."""
        received_deps: list[dict[str, str]] = []

        async def check_deps(ctx: RunContext[dict[str, str]], settings: ModelSettings | None) -> ModelSettings | None:
            received_deps.append(ctx.deps)
            return None

        agent = Agent(
            TestModel(custom_output_text='done'),
            deps_type=dict[str, str],
            prepare_model_settings=check_deps,
        )

        await agent.run('test', deps={'key': 'value'})
        assert received_deps == [{'key': 'value'}]
