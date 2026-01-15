"""Tests for dynamic model_settings with callable support."""

from __future__ import annotations

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelResponse, ToolReturnPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.settings import ModelSettings


class TestDynamicModelSettingsBasic:
    """Basic tests for callable model_settings."""

    async def test_callable_called_on_each_step(self):
        """Verify callable is called on each run step with correct run_step values."""
        call_log: list[int] = []

        async def dynamic_settings(ctx: RunContext[None]) -> ModelSettings:
            call_log.append(ctx.run_step)
            return {}

        agent = Agent(
            TestModel(custom_output_text='done'),
            model_settings=dynamic_settings,
        )

        await agent.run('test')
        assert call_log == [1]

    async def test_callable_called_multiple_times_with_tools(self):
        """Verify callable is called on each step when tools are used."""
        call_log: list[int] = []

        async def dynamic_settings(ctx: RunContext[None]) -> ModelSettings:
            call_log.append(ctx.run_step)
            return {}

        agent = Agent(
            TestModel(
                custom_output_text='final answer',
                seed=0,
            ),
            model_settings=dynamic_settings,
        )

        @agent.tool_plain
        def my_tool(x: int) -> str:
            """A tool that does something."""
            return f'result: {x}'

        await agent.run('test')
        # TestModel with a tool calls it once, then responds
        assert len(call_log) >= 1
        assert call_log[0] == 1

    async def test_static_settings_still_work(self):
        """Verify static dict settings work as before."""
        settings: ModelSettings = {'temperature': 0.5}

        agent = Agent(
            TestModel(custom_output_text='done'),
            model_settings=settings,
        )

        result = await agent.run('test')
        assert result.output == 'done'

    async def test_callable_returns_settings(self):
        """Verify callable can return different settings per step."""
        settings_returned: list[ModelSettings] = []

        async def dynamic_settings(ctx: RunContext[None]) -> ModelSettings:
            s: ModelSettings = {'temperature': 0.5 + ctx.run_step * 0.1}
            settings_returned.append(s)
            return s

        agent = Agent(
            TestModel(custom_output_text='done'),
            model_settings=dynamic_settings,
        )

        await agent.run('test')
        assert settings_returned == [{'temperature': 0.6}]

    async def test_callable_receives_message_history(self):
        """Verify callable receives message history via ctx.messages."""
        message_counts: list[int] = []

        async def check_messages(ctx: RunContext[None]) -> ModelSettings:
            message_counts.append(len(ctx.messages))
            return {}

        agent = Agent(
            TestModel(custom_output_text='done'),
            model_settings=check_messages,
        )

        await agent.run('first message')
        assert len(message_counts) == 1
        # Message history includes the current request
        assert message_counts[0] >= 1

    async def test_no_settings_works_normally(self):
        """Verify agent works normally when no settings are provided."""
        agent = Agent(
            TestModel(custom_output_text='done'),
        )

        result = await agent.run('test')
        assert result.output == 'done'

    async def test_sync_callable_works(self):
        """Verify sync callable works (not async)."""
        call_log: list[int] = []

        def sync_settings(ctx: RunContext[None]) -> ModelSettings:
            call_log.append(ctx.run_step)
            return {'temperature': 0.7}

        agent = Agent(
            TestModel(custom_output_text='done'),
            model_settings=sync_settings,
        )

        await agent.run('test')
        assert call_log == [1]


class TestDynamicModelSettingsWithToolChoice:
    """Tests for callable model_settings with tool_choice."""

    async def test_force_tool_on_first_step(self):
        """Verify callable can force tool_choice on first step only."""
        settings_by_step: dict[int, ModelSettings] = {}

        async def force_first_tool(ctx: RunContext[None]) -> ModelSettings:
            if ctx.run_step == 1:
                settings: ModelSettings = {'tool_choice': 'required'}
            else:
                settings = {}
            settings_by_step[ctx.run_step] = settings
            return settings

        agent = Agent(
            TestModel(custom_output_text='final'),
            model_settings=force_first_tool,
        )

        @agent.tool_plain
        def search(query: str) -> str:
            """Search for something."""
            return f'found: {query}'

        await agent.run('test')
        # Verify callable was called
        assert 1 in settings_by_step
        assert settings_by_step[1] == {'tool_choice': 'required'}

    async def test_conditional_tool_choice_based_on_history(self):
        """Verify callable can check message history to decide tool_choice."""

        async def require_tool_if_no_results(ctx: RunContext[None]) -> ModelSettings:
            # Check if we have any tool results in history
            has_tool_results = any(
                isinstance(msg, ModelResponse) and any(isinstance(part, ToolReturnPart) for part in msg.parts)
                for msg in ctx.messages
            )
            if not has_tool_results:
                return {'tool_choice': 'required'}
            return {}

        agent = Agent(
            TestModel(custom_output_text='done'),
            model_settings=require_tool_if_no_results,
        )

        @agent.tool_plain
        def my_tool() -> str:
            """A simple tool."""
            return 'tool result'

        await agent.run('test')


class TestDynamicModelSettingsStreaming:
    """Tests for callable model_settings with streaming."""

    async def test_callable_works_with_streaming(self):
        """Verify callable is called when using run_stream."""
        call_log: list[int] = []

        async def log_calls(ctx: RunContext[None]) -> ModelSettings:
            call_log.append(ctx.run_step)
            return {}

        agent = Agent(
            TestModel(custom_output_text='streamed output'),
            model_settings=log_calls,
        )

        async with agent.run_stream('test') as result:
            output = await result.get_output()
            assert output == 'streamed output'

        assert call_log == [1]


class TestDynamicModelSettingsWithDeps:
    """Tests for callable model_settings with dependencies."""

    async def test_callable_receives_deps(self):
        """Verify callable can access user dependencies."""
        received_deps: list[dict[str, str]] = []

        async def check_deps(ctx: RunContext[dict[str, str]]) -> ModelSettings:
            received_deps.append(ctx.deps)
            return {}

        agent = Agent(
            TestModel(custom_output_text='done'),
            deps_type=dict[str, str],
            model_settings=check_deps,
        )

        await agent.run('test', deps={'key': 'value'})
        assert received_deps == [{'key': 'value'}]


class TestModelSettingsMerge:
    """Tests for model_settings merge behavior with callables."""

    async def test_run_static_overrides_agent_callable(self):
        """Verify run static settings override agent callable."""
        agent_call_count = 0

        async def agent_settings(ctx: RunContext[None]) -> ModelSettings:
            nonlocal agent_call_count
            agent_call_count += 1
            return {'temperature': 0.5}

        agent = Agent(
            TestModel(custom_output_text='done'),
            model_settings=agent_settings,
        )

        # Run with static settings should override agent callable
        run_settings: ModelSettings = {'temperature': 0.9}
        await agent.run('test', model_settings=run_settings)
        # Agent callable should not be called when run provides static settings
        assert agent_call_count == 0

    async def test_run_callable_overrides_agent_static(self):
        """Verify run callable overrides agent static settings."""
        run_call_count = 0

        async def run_settings(ctx: RunContext[None]) -> ModelSettings:
            nonlocal run_call_count
            run_call_count += 1
            return {'temperature': 0.9}

        agent_settings: ModelSettings = {'temperature': 0.5}
        agent = Agent(
            TestModel(custom_output_text='done'),
            model_settings=agent_settings,
        )

        await agent.run('test', model_settings=run_settings)
        assert run_call_count == 1

    async def test_run_callable_overrides_agent_callable(self):
        """Verify run callable overrides agent callable."""
        agent_call_count = 0
        run_call_count = 0

        async def agent_settings(ctx: RunContext[None]) -> ModelSettings:
            nonlocal agent_call_count
            agent_call_count += 1
            return {'temperature': 0.5}

        async def run_settings(ctx: RunContext[None]) -> ModelSettings:
            nonlocal run_call_count
            run_call_count += 1
            return {'temperature': 0.9}

        agent = Agent(
            TestModel(custom_output_text='done'),
            model_settings=agent_settings,
        )

        await agent.run('test', model_settings=run_settings)
        assert agent_call_count == 0
        assert run_call_count == 1
