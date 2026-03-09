"""Tests for dynamic model_settings support (callable model_settings)."""

from __future__ import annotations

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.settings import ModelSettings


def text_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """A simple model that returns a text response."""
    return ModelResponse(parts=[TextPart('response')])


def model_with_settings(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """A model that includes model settings info in the response."""
    max_tokens = info.model_settings.get('max_tokens') if info.model_settings else None
    temperature = info.model_settings.get('temperature') if info.model_settings else None
    return ModelResponse(parts=[TextPart(f'max_tokens={max_tokens} temperature={temperature}')])


class TestStaticModelSettingsRegression:
    """Ensure static model_settings still works."""

    def test_static_agent_level(self):
        agent = Agent(FunctionModel(model_with_settings), model_settings=ModelSettings(max_tokens=100))
        result = agent.run_sync('Hello')
        assert result.output == 'max_tokens=100 temperature=None'

    def test_static_run_level(self):
        agent = Agent(FunctionModel(model_with_settings))
        result = agent.run_sync('Hello', model_settings=ModelSettings(temperature=0.5))
        assert result.output == 'max_tokens=None temperature=0.5'

    def test_static_merge_run_over_agent(self):
        agent = Agent(
            FunctionModel(model_with_settings),
            model_settings=ModelSettings(max_tokens=100, temperature=0.3),
        )
        result = agent.run_sync('Hello', model_settings=ModelSettings(temperature=0.7))
        assert result.output == 'max_tokens=100 temperature=0.7'


class TestCallableAgentLevelSettings:
    """Test agent-level callable model_settings."""

    def test_callable_agent_settings(self):
        def dynamic_settings(ctx: RunContext[None]) -> ModelSettings:
            return ModelSettings(max_tokens=200)

        agent = Agent(FunctionModel(model_with_settings), model_settings=dynamic_settings)
        result = agent.run_sync('Hello')
        assert result.output == 'max_tokens=200 temperature=None'

    def test_callable_receives_run_context(self):
        contexts: list[RunContext[str]] = []

        def dynamic_settings(ctx: RunContext[str]) -> ModelSettings:
            contexts.append(ctx)
            return ModelSettings(max_tokens=50)

        agent = Agent(FunctionModel(text_model), deps_type=str, model_settings=dynamic_settings)
        agent.run_sync('Hello', deps='test-deps')

        assert len(contexts) >= 1
        assert contexts[0].deps == 'test-deps'

    def test_callable_sees_model_settings_from_model(self):
        """The callable should see `ctx.model_settings` set to the model's base settings."""
        seen_settings: list[ModelSettings | None] = []

        def dynamic_settings(ctx: RunContext[None]) -> ModelSettings:
            seen_settings.append(ctx.model_settings)
            return ModelSettings(max_tokens=100)

        # FunctionModel has no settings (None), so ctx.model_settings should be None
        agent = Agent(FunctionModel(text_model), model_settings=dynamic_settings)
        agent.run_sync('Hello')

        assert len(seen_settings) >= 1
        assert seen_settings[0] is None


class TestCallableRunLevelSettings:
    """Test run-level callable model_settings."""

    def test_callable_run_settings(self):
        def dynamic_settings(ctx: RunContext[None]) -> ModelSettings:
            return ModelSettings(temperature=0.9)

        agent = Agent(FunctionModel(model_with_settings))
        result = agent.run_sync('Hello', model_settings=dynamic_settings)
        assert result.output == 'max_tokens=None temperature=0.9'

    def test_callable_run_sees_merged_agent_settings(self):
        """Run-level callable should see merged model+agent settings via ctx.model_settings."""
        seen_settings: list[ModelSettings | None] = []

        def run_settings(ctx: RunContext[None]) -> ModelSettings:
            seen_settings.append(ctx.model_settings)
            return ModelSettings(temperature=0.5)

        agent = Agent(FunctionModel(text_model), model_settings=ModelSettings(max_tokens=100))
        agent.run_sync('Hello', model_settings=run_settings)

        assert len(seen_settings) >= 1
        # Should see the agent-level max_tokens=100 already merged
        assert seen_settings[0] is not None
        assert seen_settings[0].get('max_tokens') == 100


class TestMixedStaticAndCallable:
    """Test mixing static and callable model_settings."""

    def test_static_agent_callable_run(self):
        def run_settings(ctx: RunContext[None]) -> ModelSettings:
            return ModelSettings(temperature=0.8)

        agent = Agent(
            FunctionModel(model_with_settings),
            model_settings=ModelSettings(max_tokens=100),
        )
        result = agent.run_sync('Hello', model_settings=run_settings)
        assert result.output == 'max_tokens=100 temperature=0.8'

    def test_callable_agent_static_run(self):
        def agent_settings(ctx: RunContext[None]) -> ModelSettings:
            return ModelSettings(max_tokens=150)

        agent = Agent(FunctionModel(model_with_settings), model_settings=agent_settings)
        result = agent.run_sync('Hello', model_settings=ModelSettings(temperature=0.6))
        assert result.output == 'max_tokens=150 temperature=0.6'

    def test_callable_agent_callable_run(self):
        def agent_settings(ctx: RunContext[None]) -> ModelSettings:
            return ModelSettings(max_tokens=200)

        def run_settings(ctx: RunContext[None]) -> ModelSettings:
            return ModelSettings(temperature=0.4)

        agent = Agent(FunctionModel(model_with_settings), model_settings=agent_settings)
        result = agent.run_sync('Hello', model_settings=run_settings)
        assert result.output == 'max_tokens=200 temperature=0.4'


class TestPerStepResolution:
    """Test that callable model_settings is called before each model request."""

    def test_called_each_step(self):
        call_count = 0
        step_numbers: list[int] = []

        def dynamic_settings(ctx: RunContext[None]) -> ModelSettings:
            nonlocal call_count
            call_count += 1
            step_numbers.append(ctx.run_step)
            return ModelSettings(max_tokens=100)

        def multi_step_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if len(messages) == 1:
                from pydantic_ai.messages import ToolCallPart

                return ModelResponse(parts=[ToolCallPart('my_tool', args='{"x": 1}')])
            return ModelResponse(parts=[TextPart('done')])

        agent = Agent(FunctionModel(multi_step_model), model_settings=dynamic_settings)

        @agent.tool_plain
        def my_tool(x: int) -> str:
            return f'result-{x}'

        agent.run_sync('Hello')

        # Should be called at least twice (once per model request)
        assert call_count >= 2
        # Step numbers should be increasing
        assert step_numbers == sorted(step_numbers)

    def test_step_dependent_settings(self):
        """Settings can vary based on run_step."""

        def step_dependent_settings(ctx: RunContext[None]) -> ModelSettings:
            if ctx.run_step <= 1:
                return ModelSettings(max_tokens=100)
            return ModelSettings(max_tokens=500)

        settings_per_step: list[tuple[int, int | None]] = []

        def tracking_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            max_tokens = info.model_settings.get('max_tokens') if info.model_settings else None
            step = len(messages)
            settings_per_step.append((step, max_tokens))
            if step == 1:
                from pydantic_ai.messages import ToolCallPart

                return ModelResponse(parts=[ToolCallPart('my_tool', args='{"x": 1}')])
            return ModelResponse(parts=[TextPart('done')])

        agent = Agent(FunctionModel(tracking_model), model_settings=step_dependent_settings)

        @agent.tool_plain
        def my_tool(x: int) -> str:
            return f'result-{x}'

        agent.run_sync('Hello')

        # First step should have max_tokens=100, second should have max_tokens=500
        assert settings_per_step[0][1] == 100
        assert settings_per_step[1][1] == 500


class TestPrecedence:
    """Test that run > agent > model precedence is maintained."""

    def test_run_overrides_agent(self):
        """Run-level settings override agent-level for the same key."""
        agent = Agent(
            FunctionModel(model_with_settings),
            model_settings=ModelSettings(temperature=0.3),
        )
        result = agent.run_sync('Hello', model_settings=ModelSettings(temperature=0.9))
        assert result.output == 'max_tokens=None temperature=0.9'

    def test_callable_run_overrides_callable_agent(self):
        def agent_settings(ctx: RunContext[None]) -> ModelSettings:
            return ModelSettings(temperature=0.2)

        def run_settings(ctx: RunContext[None]) -> ModelSettings:
            return ModelSettings(temperature=0.8)

        agent = Agent(FunctionModel(model_with_settings), model_settings=agent_settings)
        result = agent.run_sync('Hello', model_settings=run_settings)
        assert result.output == 'max_tokens=None temperature=0.8'


class TestOverrideWithModelSettings:
    """Test the override() context manager with model_settings."""

    def test_override_with_static(self):
        agent = Agent(FunctionModel(model_with_settings))

        with agent.override(model_settings=ModelSettings(max_tokens=42)):
            result = agent.run_sync('Hello')
            assert result.output == 'max_tokens=42 temperature=None'

    def test_override_with_callable(self):
        def override_settings(ctx: RunContext[None]) -> ModelSettings:
            return ModelSettings(max_tokens=99)

        agent = Agent(FunctionModel(model_with_settings))

        with agent.override(model_settings=override_settings):
            result = agent.run_sync('Hello')
            assert result.output == 'max_tokens=99 temperature=None'

    def test_override_replaces_agent_settings(self):
        """Override model_settings should replace agent-level settings."""
        agent = Agent(
            FunctionModel(model_with_settings),
            model_settings=ModelSettings(max_tokens=100, temperature=0.5),
        )

        with agent.override(model_settings=ModelSettings(max_tokens=42)):
            result = agent.run_sync('Hello')
            # Override replaced agent settings entirely; temperature should be gone
            assert result.output == 'max_tokens=42 temperature=None'

    def test_override_ignores_run_settings(self):
        """When override is set, run-level model_settings should be ignored."""
        agent = Agent(FunctionModel(model_with_settings))

        with agent.override(model_settings=ModelSettings(max_tokens=42)):
            # Run-level settings should be ignored when override is active
            result = agent.run_sync('Hello', model_settings=ModelSettings(temperature=0.9))
            assert result.output == 'max_tokens=42 temperature=None'

    def test_override_resets_after_context(self):
        """After exiting override context, original settings should be restored."""
        agent = Agent(
            FunctionModel(model_with_settings),
            model_settings=ModelSettings(max_tokens=100),
        )

        with agent.override(model_settings=ModelSettings(max_tokens=42)):
            result = agent.run_sync('Hello')
            assert result.output == 'max_tokens=42 temperature=None'

        # After override, original settings should be back
        result = agent.run_sync('Hello')
        assert result.output == 'max_tokens=100 temperature=None'
