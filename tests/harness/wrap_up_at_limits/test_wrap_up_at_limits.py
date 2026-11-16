"""Tests for `WrapUpAtLimits`, driven through `Agent(..., capabilities=[WrapUpAtLimits(...)])`."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from pydantic import BaseModel

from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults, RunContext, capture_run_messages
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.settings import ToolChoice
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RunUsage, UsageLimits
from pydantic_ai_harness import WrapUpAtLimits
from pydantic_ai_harness.wrap_up_at_limits import DEFAULT_WRAP_UP_PROMPT, NOT_EXECUTED
from pydantic_graph import End

from ._messages import ORIGIN, wrap_ups


class _Researcher:
    """A model that calls `look_up` twice per response until it's told to stop calling tools."""

    def __init__(self) -> None:
        self.requests: list[tuple[ToolChoice, list[ToolDefinition]]] = []

    def __call__(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tool_choice = info.model_settings.get('tool_choice') if info.model_settings else None
        self.requests.append((tool_choice, info.function_tools))
        if tool_choice == 'none':
            return ModelResponse(parts=[TextPart('what I found so far')])
        step = len(self.requests)
        return ModelResponse(
            parts=[
                ToolCallPart('look_up', {'topic': f'a{step}'}, tool_call_id=f'a{step}'),
                ToolCallPart('look_up', {'topic': f'b{step}'}, tool_call_id=f'b{step}'),
            ]
        )

    async def stream(self, messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        for index, part in enumerate(self(messages, info).parts):
            if isinstance(part, TextPart):
                yield part.content
            else:
                assert isinstance(part, ToolCallPart)
                yield {index: DeltaToolCall(part.tool_name, part.args_as_json_str(), tool_call_id=part.tool_call_id)}


def _agent(
    model: _Researcher, looked_up: list[str], capability: WrapUpAtLimits[None] | None = None
) -> Agent[None, str]:
    agent = Agent(
        FunctionModel(model, stream_function=model.stream),
        deps_type=type(None),
        capabilities=[capability or WrapUpAtLimits()],
    )

    @agent.tool_plain
    def look_up(topic: str) -> str:
        looked_up.append(topic)
        return f'notes on {topic}'

    return agent


class TestWrapUpAtLimits:
    async def test_last_allowed_request_answers_without_tools(self) -> None:
        model = _Researcher()
        looked_up: list[str] = []

        result = await _agent(model, looked_up).run('Research', usage_limits=UsageLimits(request_limit=3))

        assert result.output == 'what I found so far'
        assert result.usage.requests == 3
        assert sorted(looked_up) == ['a1', 'a2', 'b1', 'b2']
        assert wrap_ups(result.new_messages()) == [
            (DEFAULT_WRAP_UP_PROMPT, {'origin': ORIGIN, 'limit': 'request_limit'})
        ]
        # The tools stay defined on the final request, so the cached prefix still matches.
        assert [choice for choice, _ in model.requests] == [None, None, 'none']
        assert model.requests[2][1] == model.requests[0][1]

    async def test_over_budget_batch_is_answered_as_not_executed(self) -> None:
        model = _Researcher()
        looked_up: list[str] = []

        result = await _agent(model, looked_up).run('Research', usage_limits=UsageLimits(tool_calls_limit=3))

        assert result.output == 'what I found so far'
        assert sorted(looked_up) == ['a1', 'b1']
        assert result.usage.tool_calls == 2
        skipped = [
            (part.tool_call_id, part.outcome)
            for message in result.new_messages()
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart) and part.content == NOT_EXECUTED
        ]
        assert skipped == [('a2', 'denied'), ('b2', 'denied')]
        assert wrap_ups(result.new_messages()) == [
            (DEFAULT_WRAP_UP_PROMPT, {'origin': ORIGIN, 'limit': 'tool_calls_limit'})
        ]

    async def test_over_budget_batch_is_answered_under_run_stream(self) -> None:
        looked_up: list[str] = []

        async with _agent(_Researcher(), looked_up).run_stream(
            'Research', usage_limits=UsageLimits(tool_calls_limit=3)
        ) as stream:
            output = await stream.get_output()
            messages = stream.all_messages()

        assert output == 'what I found so far'
        assert sorted(looked_up) == ['a1', 'b1']
        returns = [
            (part.tool_call_id, part.content)
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        assert returns == [
            ('a1', 'notes on a1'),
            ('b1', 'notes on b1'),
            ('a2', NOT_EXECUTED),
            ('b2', NOT_EXECUTED),
        ]

    async def test_batch_the_caller_streamed_keeps_its_results(self) -> None:
        looked_up: list[str] = []
        agent = _agent(_Researcher(), looked_up)

        with capture_run_messages() as messages, pytest.raises(UsageLimitExceeded, match='tool_calls_limit'):
            async with agent.iter('Research', usage_limits=UsageLimits(tool_calls_limit=3)) as run:
                node = run.next_node
                while not isinstance(node, End):
                    if Agent.is_call_tools_node(node):
                        # Streaming the node runs its tools, so the batch is past the point of being answered instead.
                        async with node.stream(run.ctx) as events:
                            async for _ in events:
                                pass
                    node = await run.next(node)

        assert sorted(looked_up) == ['a1', 'b1']
        returns = [
            part.content
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        assert returns == ['notes on a1', 'notes on b1']

    async def test_exhausted_tool_calls_wrap_up_the_next_request(self) -> None:
        model = _Researcher()
        looked_up: list[str] = []

        result = await _agent(model, looked_up).run('Research', usage_limits=UsageLimits(tool_calls_limit=2))

        assert sorted(looked_up) == ['a1', 'b1']
        assert result.usage.requests == 2
        assert wrap_ups(result.new_messages()) == [
            (DEFAULT_WRAP_UP_PROMPT, {'origin': ORIGIN, 'limit': 'tool_calls_limit'})
        ]

    @pytest.mark.parametrize(
        ('limits', 'limit'),
        [
            (UsageLimits(input_tokens_limit=250), 'input_tokens_limit'),
            (UsageLimits(output_tokens_limit=60), 'output_tokens_limit'),
            (UsageLimits(total_tokens_limit=300), 'total_tokens_limit'),
        ],
    )
    async def test_token_limit_wraps_up_before_it_is_exceeded(self, limits: UsageLimits, limit: str) -> None:
        model = _Researcher()

        result = await _agent(model, []).run('Research', usage_limits=limits)

        assert result.output == 'what I found so far'
        assert wrap_ups(result.new_messages()) == [(DEFAULT_WRAP_UP_PROMPT, {'origin': ORIGIN, 'limit': limit})]

    async def test_custom_prompt(self) -> None:
        result = await _agent(_Researcher(), [], WrapUpAtLimits(prompt='Stop and summarize.')).run(
            'Research', usage_limits=UsageLimits(request_limit=1)
        )

        assert wrap_ups(result.new_messages()) == [
            ('Stop and summarize.', {'origin': ORIGIN, 'limit': 'request_limit'})
        ]

    async def test_run_within_its_limits_is_unchanged(self) -> None:
        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.model_settings is None or 'tool_choice' not in info.model_settings
            return ModelResponse(parts=[TextPart('done')])

        agent = Agent(FunctionModel(model), capabilities=[WrapUpAtLimits()])

        for limits in (None, UsageLimits(request_limit=5)):
            result = await agent.run('Research', usage_limits=limits)
            assert result.output == 'done'
            assert wrap_ups(result.new_messages()) == []

    @pytest.mark.moves_cache_prefix(reason="with tool output, `tool_choice='none'` sends only the output tools")
    async def test_structured_output_is_still_produced(self) -> None:
        class Report(BaseModel):
            summary: str

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if info.model_settings and info.model_settings.get('tool_choice') == 'none':
                return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {'summary': 'partial'})])
            return ModelResponse(parts=[ToolCallPart('look_up', {'topic': 'a'})])

        agent = Agent(FunctionModel(model), output_type=Report, capabilities=[WrapUpAtLimits()])

        @agent.tool_plain
        def look_up(topic: str) -> str:
            return topic

        result = await agent.run('Research', usage_limits=UsageLimits(request_limit=2))

        assert result.output == Report(summary='partial')

    async def test_limit_raises_when_no_request_is_left(self) -> None:
        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(
                parts=[ToolCallPart('look_up', {'topic': 'a'}), ToolCallPart('look_up', {'topic': 'b'})]
            )

        looked_up: list[str] = []
        agent = Agent(FunctionModel(model), capabilities=[WrapUpAtLimits()])

        @agent.tool_plain
        def look_up(topic: str) -> str:
            looked_up.append(topic)
            return topic

        # The model ignores `tool_choice='none'` on its one request, and no request is left to wrap up with.
        with pytest.raises(UsageLimitExceeded, match='tool_calls_limit'):
            await agent.run('Research', usage_limits=UsageLimits(request_limit=1, tool_calls_limit=1))
        assert looked_up == []

    async def test_resumed_approvals_are_left_to_core(self) -> None:
        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(
                parts=[
                    ToolCallPart('delete', {'path': 'a'}, tool_call_id='d1'),
                    ToolCallPart('delete', {'path': 'b'}, tool_call_id='d2'),
                ]
            )

        deleted: list[str] = []
        agent = Agent(FunctionModel(model), output_type=[str, DeferredToolRequests], capabilities=[WrapUpAtLimits()])

        @agent.tool_plain(requires_approval=True)
        def delete(path: str) -> str:
            deleted.append(path)
            return path

        first = await agent.run('Clean up')
        assert isinstance(first.output, DeferredToolRequests)

        # The approved calls aren't answered with `NOT_EXECUTED`: core's own check applies to them.
        with pytest.raises(UsageLimitExceeded, match='tool_calls_limit'):
            await agent.run(
                message_history=first.all_messages(),
                deferred_tool_results=DeferredToolResults(approvals={'d1': True, 'd2': True}),
                usage_limits=UsageLimits(tool_calls_limit=1),
            )
        assert deleted == []

    async def test_resumed_provider_turn_is_left_alone(self) -> None:
        ctx = RunContext[None](
            deps=None, model=TestModel(), usage=RunUsage(), usage_limits=UsageLimits(request_limit=1)
        )
        request_context = ModelRequestContext(
            model=TestModel(),
            messages=[ModelRequest.user_text_prompt('Research'), ModelResponse(parts=[TextPart('partial')])],
            model_settings=None,
            model_request_parameters=ModelRequestParameters(),
        )

        assert await WrapUpAtLimits[None]().before_model_request(ctx, request_context) is request_context
