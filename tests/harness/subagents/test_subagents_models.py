"""Tests for per-delegation model selection in the SubAgents capability."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import ModelMessage, ModelResponse, RetryPromptPart, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness.subagents import ModelOption, SubAgent, SubAgents, SubAgentToolset

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    """Run async tests on the asyncio backend (matching upstream pydantic-ai)."""
    return 'asyncio'


def _delegate_with_model(agent_name: str, model: str | None) -> FunctionModel:
    """A parent model that delegates once (optionally naming a model), then replies."""
    calls = {'n': 0}

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls['n'] += 1
        if calls['n'] == 1:
            args: dict[str, Any] = {'agent_name': agent_name, 'task': 't'}
            if model is not None:
                args['model'] = model
            return ModelResponse(parts=[ToolCallPart('delegate_task', args, tool_call_id='c1')])
        return ModelResponse(parts=[TextPart('all done')])

    return FunctionModel(model_fn)


def _recording_model(log: list[tuple[str, ModelSettings | None]], label: str) -> FunctionModel:
    """A sub-agent model that records the label it ran under and its settings."""

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        log.append((label, info.model_settings))
        return ModelResponse(parts=[TextPart(label)])

    return FunctionModel(model_fn)


def _delegate_returns(result: Any) -> list[str]:
    """The `delegate_task` tool-return contents from a run result, in order."""
    return [
        str(part.content)
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == 'delegate_task'
    ]


def _delegate_retries(result: Any) -> list[str]:
    """The `delegate_task` retry-prompt contents from a run result, in order."""
    return [
        str(part.content)
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, RetryPromptPart) and part.tool_name == 'delegate_task'
    ]


def _delegate_schema(capability: SubAgents[object]) -> dict[str, Any]:
    """The `delegate_task` parameter schema as the parent model is offered it."""
    captured: dict[str, Any] = {}

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        captured.update(next(t.parameters_json_schema for t in info.function_tools if t.name == 'delegate_task'))
        return ModelResponse(parts=[TextPart('done')])

    Agent(FunctionModel(model_fn), capabilities=[capability]).run_sync('go')
    return captured


class TestSchema:
    def test_no_menu_hides_the_model_argument(self) -> None:
        capability = SubAgents[object](agents=[SubAgent(Agent(TestModel(), name='worker'))])
        properties = _delegate_schema(capability)['properties']
        assert set(properties) == {'agent_name', 'task'}

    def test_menu_offers_the_keys_as_an_enum(self) -> None:
        capability = SubAgents[object](
            agents=[SubAgent(Agent(TestModel(), name='worker'))],
            models={'fast': TestModel(), 'deep': TestModel()},
        )
        model_property = _delegate_schema(capability)['properties']['model']
        assert model_property['enum'] == ['fast', 'deep']
        assert model_property['type'] == 'string'
        assert 'Omit it' in model_property['description']


class TestInstructions:
    def test_lists_the_menu_with_labels_and_hints(self) -> None:
        capability = SubAgents[object](
            agents=[SubAgent(Agent(TestModel(), name='worker'))],
            models={
                'fast': 'anthropic:claude-haiku-4-5',
                'deep': ModelOption('anthropic:claude-opus-4-7', description='hard reasoning'),
            },
        )
        instructions = capability.get_instructions()
        assert isinstance(instructions, str)
        assert '- fast (anthropic:claude-haiku-4-5)' in instructions
        assert '- deep (anthropic:claude-opus-4-7): hard reasoning' in instructions

    def test_labels_a_model_instance_by_system_and_name(self) -> None:
        capability = SubAgents[object](
            agents=[SubAgent(Agent(TestModel(), name='worker'))], models={'fast': TestModel()}
        )
        instructions = capability.get_instructions()
        assert isinstance(instructions, str)
        assert '- fast (test:test)' in instructions

    def test_restricted_delegate_lists_its_options(self) -> None:
        capability = SubAgents[object](
            agents=[
                SubAgent(Agent(TestModel(), name='linter', description='Runs the linter'), models=['fast']),
                SubAgent(Agent(TestModel(), name='architect')),
            ],
            models={'fast': TestModel(), 'deep': TestModel()},
        )
        instructions = capability.get_instructions()
        assert isinstance(instructions, str)
        assert '- linter: Runs the linter (models: fast)' in instructions
        assert '- architect\n' in instructions

    def test_no_menu_leaves_instructions_unchanged(self) -> None:
        capability = SubAgents[object](agents=[SubAgent(Agent(TestModel(), name='worker'))])
        instructions = capability.get_instructions()
        assert isinstance(instructions, str)
        assert 'Available models' not in instructions


class TestRouting:
    async def test_selected_option_runs_the_sub_agent_on_that_model(self) -> None:
        log: list[tuple[str, ModelSettings | None]] = []
        worker = Agent(_recording_model(log, 'own'), name='worker')
        parent: Agent[object, str] = Agent(
            _delegate_with_model('worker', 'deep'),
            capabilities=[
                SubAgents(
                    agents=[SubAgent(worker)],
                    models={'fast': _recording_model(log, 'fast'), 'deep': _recording_model(log, 'deep')},
                )
            ],
        )
        result = await parent.run('go')
        assert _delegate_returns(result) == ['deep']
        assert [label for label, _ in log] == ['deep']

    async def test_option_settings_merge_over_the_sub_agents_own(self) -> None:
        log: list[tuple[str, ModelSettings | None]] = []
        worker = Agent(TestModel(), name='worker', model_settings=ModelSettings(temperature=0.5, seed=7))
        parent: Agent[object, str] = Agent(
            _delegate_with_model('worker', 'deep'),
            capabilities=[
                SubAgents(
                    agents=[SubAgent(worker)],
                    models={
                        'deep': ModelOption(_recording_model(log, 'deep'), settings=ModelSettings(temperature=0.1))
                    },
                )
            ],
        )
        await parent.run('go')
        settings = log[0][1]
        assert settings is not None
        assert settings.get('temperature') == 0.1  # the option wins
        assert settings.get('seed') == 7  # the sub-agent's own setting survives

    async def test_omitted_model_keeps_the_sub_agents_own(self) -> None:
        log: list[tuple[str, ModelSettings | None]] = []
        worker = Agent(_recording_model(log, 'own'), name='worker')
        parent: Agent[object, str] = Agent(
            _delegate_with_model('worker', None),
            capabilities=[SubAgents(agents=[SubAgent(worker)], models={'deep': _recording_model(log, 'deep')})],
        )
        result = await parent.run('go')
        assert _delegate_returns(result) == ['own']

    async def test_restriction_first_key_is_the_delegates_default(self) -> None:
        log: list[tuple[str, ModelSettings | None]] = []
        worker = Agent(_recording_model(log, 'own'), name='worker')
        parent: Agent[object, str] = Agent(
            _delegate_with_model('worker', None),
            capabilities=[
                SubAgents(
                    agents=[SubAgent(worker, models=['fast', 'deep'])],
                    models={'fast': _recording_model(log, 'fast'), 'deep': _recording_model(log, 'deep')},
                )
            ],
        )
        result = await parent.run('go')
        assert _delegate_returns(result) == ['fast']

    async def test_unknown_model_retries_with_the_menu(self) -> None:
        worker = Agent(TestModel(custom_output_text='OK'), name='worker')
        parent: Agent[object, str] = Agent(
            _delegate_with_model('worker', 'ghost'),
            capabilities=[SubAgents(agents=[SubAgent(worker)], models={'fast': TestModel(), 'deep': TestModel()})],
        )
        result = await parent.run('go')
        assert any(
            "Unknown model 'ghost'" in retry and 'Available models: fast, deep' in retry
            for retry in _delegate_retries(result)
        )

    async def test_restricted_delegate_rejects_an_unavailable_model(self) -> None:
        worker = Agent(TestModel(custom_output_text='OK'), name='worker')
        parent: Agent[object, str] = Agent(
            _delegate_with_model('worker', 'deep'),
            capabilities=[
                SubAgents(
                    agents=[SubAgent(worker, models=['fast'])],
                    models={'fast': TestModel(), 'deep': TestModel()},
                )
            ],
        )
        result = await parent.run('go')
        assert any(
            "Sub-agent 'worker' cannot run on model 'deep'" in retry and 'Available models for it: fast' in retry
            for retry in _delegate_retries(result)
        )

    async def test_rejected_model_does_not_spend_the_call_budget(self) -> None:
        """A bad model key is resolved before `max_calls` is charged, so the delegate stays usable."""
        toolset: SubAgentToolset[object] = SubAgentToolset(
            agents={'worker': SubAgent(Agent(TestModel(custom_output_text='OK'), name='worker'), max_calls=1)},
            forward_usage=True,
            inherit_tools=False,
            shared_capabilities=[],
            event_stream_handler=None,
            tool_name='delegate_task',
            tool_retries=1,
            contain_errors=False,
            call_counts={},
            models={'fast': ModelOption(TestModel(custom_output_text='FAST'))},
        )
        ctx: RunContext[object] = RunContext(deps=None, model=TestModel(), usage=RunUsage(), run_id='run-1')
        with pytest.raises(ModelRetry):
            await toolset.delegate_task(ctx, 'worker', 't', model='ghost')
        assert await toolset.delegate_task(ctx, 'worker', 't', model='fast') == 'FAST'

    async def test_unknown_model_without_a_menu_says_so(self) -> None:
        toolset: SubAgentToolset[object] = SubAgentToolset(
            agents={'worker': SubAgent(Agent(TestModel(), name='worker'))},
            forward_usage=True,
            inherit_tools=False,
            shared_capabilities=[],
            event_stream_handler=None,
            tool_name='delegate_task',
            tool_retries=1,
            contain_errors=False,
            call_counts={},
        )
        ctx: RunContext[object] = RunContext(deps=None, model=TestModel(), usage=RunUsage(), run_id='run-1')
        with pytest.raises(ModelRetry, match=r'Available models: \(none configured\)'):
            await toolset.delegate_task(ctx, 'worker', 't', model='fast')


class TestRestrictionValidation:
    def test_unknown_restriction_key_is_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"restricts `models` to unknown option\(s\) 'ghost'"):
            SubAgents[object](
                agents=[SubAgent(Agent(TestModel(), name='worker'), models=['ghost'])],
                models={'fast': TestModel()},
            )

    def test_restriction_without_a_menu_is_rejected(self) -> None:
        with pytest.raises(ValueError, match=r'configured options: \(none configured\)'):
            SubAgents[object](agents=[SubAgent(Agent(TestModel(), name='worker'), models=['fast'])])

    def test_empty_restriction_is_rejected(self) -> None:
        with pytest.raises(ValueError, match='empty `models` restriction'):
            SubAgents[object](
                agents=[SubAgent(Agent(TestModel(), name='worker'), models=[])], models={'fast': TestModel()}
            )
