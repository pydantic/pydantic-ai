"""Tests for `AdaptiveReasoning`, driven through `Agent(..., capabilities=[AdaptiveReasoning(...)])`."""

from __future__ import annotations

from collections.abc import Sequence

from inline_snapshot import snapshot

from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults, ModelRetry, RunContext, ToolDenied
from pydantic_ai.capabilities import AbstractCapability, CombinedCapability, Thinking
from pydantic_ai.exceptions import ToolFailed
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    ModelSettingsDeltaPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.settings import ModelSettings, ThinkingLevel
from pydantic_ai_harness import AdaptiveReasoning


def _model(*responses: Sequence[ModelResponsePart], efforts: list[object]) -> FunctionModel:
    """A thinking model that sends `responses` in order, then `'done'`, recording the `thinking` each request used."""
    steps = iter(responses)

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        efforts.append(info.model_request_parameters.thinking)
        return ModelResponse(parts=list(next(steps, [TextPart('done')])))

    return FunctionModel(model, profile=ModelProfile(supports_thinking=True))


def _agent(
    model: FunctionModel, capability: AbstractCapability[None], model_settings: ModelSettings | None = None
) -> Agent[None, str]:
    agent = Agent(model, deps_type=type(None), capabilities=[capability], retries=3, model_settings=model_settings)

    @agent.tool_plain
    def read(path: str) -> str:
        return f'contents of {path}'

    @agent.tool_plain
    def flaky(path: str) -> str:
        raise ModelRetry('try another path')

    @agent.tool_plain
    def broken(path: str) -> str:
        raise ToolFailed('disk error')

    return agent


def _call(tool: str, tool_call_id: str) -> ToolCallPart:
    return ToolCallPart(tool, {'path': 'a.py'}, tool_call_id=tool_call_id)


class TestAdaptiveReasoning:
    def test_default_policy(self):
        efforts: list[object] = []
        model = _model(
            [_call('read', '1')],
            [_call('flaky', '2')],
            [_call('read', '3'), _call('read', '4')],
            [_call('broken', '5')],
            [_call('read', '6'), _call('flaky', '7')],
            efforts=efforts,
        )
        result = _agent(model, AdaptiveReasoning()).run_sync('fix the bug')
        assert efforts == snapshot(['high', 'medium', 'high', 'medium', 'high', 'high'])
        assert [
            part.settings
            for message in result.all_messages()
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ModelSettingsDeltaPart)
        ] == snapshot(
            [
                {'thinking': 'high'},
                {'thinking': 'medium'},
                {'thinking': 'high'},
                {'thinking': 'medium'},
                {'thinking': 'high'},
            ]
        )

    def test_custom_levels(self):
        efforts: list[object] = []
        model = _model([_call('read', '1')], efforts=efforts)
        _agent(model, AdaptiveReasoning(deliberate='xhigh', routine='low')).run_sync('go')
        assert efforts == snapshot(['xhigh', 'low'])

    def test_resumed_turn_keeps_its_effort(self):
        efforts: list[object] = []
        history: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart('go')]),
            ModelResponse(parts=[_call('read', '1')]),
            ModelRequest(parts=[ToolReturnPart('read', 'contents', tool_call_id='1')]),
            ModelResponse(parts=[TextPart('partial ')], state='suspended'),
        ]
        _agent(_model(efforts=efforts), AdaptiveReasoning()).run_sync(message_history=history)
        assert efforts == snapshot(['medium'])

    def test_denied_tool_is_deliberate(self):
        efforts: list[object] = []
        model = _model([_call('delete', '1')], efforts=efforts)
        agent = _agent(model, AdaptiveReasoning())

        @agent.tool_plain(requires_approval=True)
        def delete(path: str) -> str:
            return path  # pragma: no cover

        first = agent.run_sync('go', output_type=[str, DeferredToolRequests])
        assert isinstance(first.output, DeferredToolRequests)
        agent.run_sync(
            message_history=first.all_messages(),
            deferred_tool_results=DeferredToolResults(approvals={'1': ToolDenied()}),
        )
        assert efforts == snapshot(['high', 'high'])

    def test_retry_feedback_is_deliberate(self):
        efforts: list[object] = []
        agent = Agent(_model([TextPart('first')], efforts=efforts), capabilities=[AdaptiveReasoning(routine='low')])

        @agent.output_validator
        def check(output: str) -> str:
            if output == 'first':
                raise ModelRetry('again')
            return output

        assert agent.run_sync('go').output == 'done'
        assert efforts == snapshot(['high', 'high'])

    def test_select_replaces_policy(self):
        def by_step(ctx: RunContext[None]) -> ThinkingLevel | None:
            return 'low' if ctx.run_step > 1 else None

        efforts: list[object] = []
        model = _model([_call('read', '1')], efforts=efforts)
        agent = _agent(model, AdaptiveReasoning[None](select=by_step), model_settings=ModelSettings(thinking='max'))
        agent.run_sync('go')
        assert efforts == snapshot(['max', 'low'])

    def test_overrides_thinking_listed_before(self):
        efforts: list[object] = []
        model = _model([_call('read', '1')], efforts=efforts)
        _agent(model, CombinedCapability([Thinking(effort='low'), AdaptiveReasoning()])).run_sync('go')
        assert efforts == snapshot(['high', 'medium'])

    def test_run_model_settings_take_precedence(self):
        efforts: list[object] = []
        agent = _agent(_model([_call('read', '1')], efforts=efforts), AdaptiveReasoning())
        agent.run_sync('go', model_settings=ModelSettings(thinking='minimal'))
        assert efforts == snapshot(['minimal', 'minimal'])
