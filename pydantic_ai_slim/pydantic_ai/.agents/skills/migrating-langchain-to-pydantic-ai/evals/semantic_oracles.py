#!/usr/bin/env python3
"""Evaluator-owned behavioral oracles for the semantic-gap skill eval."""

from __future__ import annotations

import asyncio
import importlib
import importlib.metadata
import importlib.util
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ToolRetryMiddleware, dynamic_prompt
from langchain.tools import tool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, PrivateAttr
from typing_extensions import TypedDict

from pydantic_ai import Agent, AgentRunResultEvent, ModelRetry, RunContext
from pydantic_ai.capabilities import Hooks
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults


def _module_origin(name: str) -> str:
    module = importlib.import_module(name)
    file = getattr(module, '__file__', None)
    if file:
        return str(Path(file).resolve())
    spec = importlib.util.find_spec(name)
    locations = spec.submodule_search_locations if spec else None
    return ','.join(str(Path(item).resolve()) for item in locations or ())


def _provenance() -> dict[str, dict[str, str]]:
    return {
        'versions': {
            'langchain': importlib.metadata.version('langchain'),
            'langgraph': importlib.metadata.version('langgraph'),
            'pydantic-ai-slim': importlib.metadata.version('pydantic-ai-slim'),
        },
        'import_origins': {name: _module_origin(name) for name in ('langchain', 'langgraph', 'pydantic_ai')},
    }


class _CapturingChatModel(BaseChatModel):
    _responses: list[AIMessage] = PrivateAttr()
    _requests: list[list[Any]] = PrivateAttr(default_factory=list)

    def __init__(self, responses: list[AIMessage]):
        super().__init__()
        self._responses = responses

    @property
    def _llm_type(self) -> str:
        return 'semantic-gap-oracle'

    @property
    def requests(self) -> list[list[Any]]:
        return self._requests

    def bind_tools(self, tools: Any, *, tool_choice: Any = None, **kwargs: Any) -> _CapturingChatModel:
        return self

    def _generate(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self._requests.append(messages)
        response = self._responses[len(self._requests) - 1]
        return ChatResult(generations=[ChatGeneration(message=response)])


@dataclass
class _LocaleContext:
    locale: str


@dynamic_prompt
def _source_policy_prompt(_request: Any) -> str:
    return 'Never disclose authenticated identity.'


@dynamic_prompt
def _source_locale_prompt(request: Any) -> str:
    return f'Reply using locale {request.runtime.context.locale}.'


def _prompt_oracle() -> dict[str, Any]:
    source_model = _CapturingChatModel([AIMessage(content='done')])
    source = create_agent(
        source_model,
        tools=[],
        context_schema=_LocaleContext,
        middleware=[_source_policy_prompt, _source_locale_prompt],
    )
    source.invoke(
        {'messages': [{'role': 'user', 'content': 'hello'}]},
        context=_LocaleContext(locale='fr-FR'),
    )
    source_system_messages = [message.content for message in source_model.requests[0] if message.type == 'system']

    target_instructions: list[str | None] = []

    def target_model(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        target_instructions.append(info.instructions)
        return ModelResponse(parts=[TextPart('done')])

    target = Agent(FunctionModel(target_model), deps_type=_LocaleContext)

    @target.instructions
    def target_policy() -> str:
        return 'Never disclose authenticated identity.'

    @target.instructions
    def target_locale(ctx: RunContext[_LocaleContext]) -> str:
        return f'Reply using locale {ctx.deps.locale}.'

    target.run_sync('hello', deps=_LocaleContext(locale='fr-FR'))

    replacement_hooks: Hooks[_LocaleContext] = Hooks()

    def replace_latest_instruction(request_context: ModelRequestContext, instruction: str) -> ModelRequestContext:
        messages = list(request_context.messages)
        assert isinstance(messages[-1], ModelRequest)
        messages[-1] = replace(messages[-1], instructions=instruction)
        return replace(request_context, messages=messages)

    @replacement_hooks.on.before_model_request
    async def replace_policy(
        _ctx: RunContext[_LocaleContext], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        return replace_latest_instruction(request_context, 'Never disclose authenticated identity.')

    @replacement_hooks.on.before_model_request
    async def replace_locale(
        ctx: RunContext[_LocaleContext], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        return replace_latest_instruction(request_context, f'Reply using locale {ctx.deps.locale}.')

    replacement_instructions: list[str | None] = []

    def replacement_model(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        replacement_instructions.append(info.instructions)
        return ModelResponse(parts=[TextPart('done')])

    replacement_agent = Agent(
        FunctionModel(replacement_model),
        deps_type=_LocaleContext,
        capabilities=[replacement_hooks],
    )
    first = replacement_agent.run_sync('hello', deps=_LocaleContext(locale='fr-FR'))
    continued = replacement_agent.run_sync(
        'continue',
        deps=_LocaleContext(locale='de-DE'),
        message_history=first.all_messages(),
    )
    serialized_history = [
        replace(message, instructions=None) if isinstance(message, ModelRequest) else message
        for message in continued.all_messages()
    ]
    serialized_instruction_count = sum(
        message.instructions is not None for message in serialized_history if isinstance(message, ModelRequest)
    )
    observed = {
        'source_system_messages': source_system_messages,
        'target_instructions': target_instructions,
        'workaround_replacement_instructions': replacement_instructions,
        'workaround_serialized_instruction_count': serialized_instruction_count,
    }
    assert observed == {
        'source_system_messages': ['Reply using locale fr-FR.'],
        'target_instructions': ['Never disclose authenticated identity.\n\nReply using locale fr-FR.'],
        'workaround_replacement_instructions': [
            'Reply using locale fr-FR.',
            'Reply using locale de-DE.',
        ],
        'workaround_serialized_instruction_count': 0,
    }
    return observed


def _retry_oracle() -> dict[str, int]:
    source_tool_calls: list[int] = []

    @tool
    def source_flaky(x: int) -> int:
        """Fail once and then return the input."""
        source_tool_calls.append(x)
        if len(source_tool_calls) == 1:
            raise ValueError('fail once')
        return x

    source_model = _CapturingChatModel(
        [
            AIMessage(
                content='',
                tool_calls=[{'name': 'source_flaky', 'args': {'x': 1}, 'id': 'lc-1', 'type': 'tool_call'}],
            ),
            AIMessage(content='done'),
        ]
    )
    source = create_agent(
        source_model,
        tools=[source_flaky],
        middleware=[ToolRetryMiddleware(max_retries=1, initial_delay=0, jitter=False)],
    )
    source.invoke({'messages': [{'role': 'user', 'content': 'go'}]})

    target_model_calls = 0
    target_tool_calls = 0

    def target_model(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        nonlocal target_model_calls
        target_model_calls += 1
        if target_model_calls <= 2:
            return ModelResponse(parts=[ToolCallPart('target_flaky', {'x': 1}, f'pai-{target_model_calls}')])
        return ModelResponse(parts=[TextPart('done')])

    target = Agent(FunctionModel(target_model))

    @target.tool_plain(retries=1)
    def target_flaky(x: int) -> int:
        nonlocal target_tool_calls
        target_tool_calls += 1
        if target_tool_calls == 1:
            raise ModelRetry('fail once')
        return x

    target.run_sync('go')

    class TransientServiceError(RuntimeError):
        pass

    local_hooks = Hooks()
    local_model_calls = 0
    local_tool_calls = 0

    @local_hooks.on.tool_execute(tools=['target_local_flaky'])
    async def retry_same_handler(
        _ctx: RunContext[None],
        *,
        call: Any,
        tool_def: Any,
        args: dict[str, Any],
        handler: Any,
    ) -> Any:
        assert call.tool_name == tool_def.name == 'target_local_flaky'
        for attempt in range(2):
            try:
                return await handler(args)
            except TransientServiceError:
                if attempt == 1:
                    raise
        raise AssertionError('unreachable')

    def local_model(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        nonlocal local_model_calls
        local_model_calls += 1
        if local_model_calls == 1:
            return ModelResponse(parts=[ToolCallPart('target_local_flaky', {'x': 1}, 'pai-local-1')])
        return ModelResponse(parts=[TextPart('done')])

    local_target = Agent(FunctionModel(local_model), capabilities=[local_hooks])

    @local_target.tool_plain
    def target_local_flaky(x: int) -> int:
        nonlocal local_tool_calls
        local_tool_calls += 1
        if local_tool_calls == 1:
            raise TransientServiceError('fail once')
        return x

    local_target.run_sync('go')
    observed = {
        'source_model_calls': len(source_model.requests),
        'source_tool_calls': len(source_tool_calls),
        'target_model_calls': target_model_calls,
        'target_tool_calls': target_tool_calls,
        'workaround_model_calls': local_model_calls,
        'workaround_tool_calls': local_tool_calls,
    }
    assert observed == {
        'source_model_calls': 2,
        'source_tool_calls': 2,
        'target_model_calls': 3,
        'target_tool_calls': 2,
        'workaround_model_calls': 2,
        'workaround_tool_calls': 2,
    }
    return observed


class _InterruptState(TypedDict):
    approved: bool | None


@dataclass
class _ApprovalDeps:
    customer_id: str
    side_effects: list[tuple[str, str]]


def _interrupt_oracle() -> dict[str, int | bool]:
    pre_interrupt_effects: list[str] = []

    def source_gate(_state: _InterruptState) -> dict[str, bool]:
        pre_interrupt_effects.append('ran')
        decision = interrupt({'question': 'Export?'})
        return {'approved': bool(decision)}

    builder = StateGraph(_InterruptState)
    builder.add_node('gate', source_gate)
    builder.add_edge(START, 'gate')
    builder.add_edge('gate', END)
    source = builder.compile(checkpointer=InMemorySaver())
    config = {'configurable': {'thread_id': 'oracle-thread'}}
    source.invoke({'approved': None}, config=config)
    source.invoke(Command(resume=True), config=config)

    target_model_calls = 0

    def target_model(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        nonlocal target_model_calls
        target_model_calls += 1
        if target_model_calls == 1:
            return ModelResponse(parts=[ToolCallPart('export_report', {'report_id': 'report-7'}, 'export-1')])
        return ModelResponse(parts=[TextPart('done')])

    target = Agent(
        FunctionModel(target_model),
        deps_type=_ApprovalDeps,
        output_type=[str, DeferredToolRequests],
    )

    @target.tool(requires_approval=True)
    def export_report(ctx: RunContext[_ApprovalDeps], report_id: str) -> str:
        ctx.deps.side_effects.append((ctx.deps.customer_id, report_id))
        return 'exported'

    side_effects: list[tuple[str, str]] = []
    deps = _ApprovalDeps('customer-a', side_effects)
    pending = target.run_sync('export', deps=deps)
    assert isinstance(pending.output, DeferredToolRequests)
    call_id = pending.output.approvals[0].tool_call_id
    issued = {call_id: 'customer-a'}
    consumed: set[str] = set()

    def authorize_approval(principal: str, requested_id: str) -> str:
        if requested_id not in issued:
            raise KeyError('unknown approval')
        if issued[requested_id] != principal:
            raise PermissionError('approval belongs to another principal')
        if requested_id in consumed:
            raise RuntimeError('approval already consumed')
        consumed.add(requested_id)
        return requested_id

    try:
        authorize_approval('customer-a', 'forged-unknown')
    except KeyError:
        unknown_rejected = True
    else:
        unknown_rejected = False
    try:
        authorize_approval('customer-b', call_id)
    except PermissionError:
        foreign_rejected = True
    else:
        foreign_rejected = False
    assert unknown_rejected and foreign_rejected and side_effects == []
    approved_call_id = authorize_approval('customer-a', call_id)

    result = target.run_sync(
        message_history=pending.all_messages(),
        deps=deps,
        deferred_tool_results=DeferredToolResults(approvals={approved_call_id: True}),
    )
    assert result.output == 'done'
    try:
        authorize_approval('customer-a', call_id)
    except RuntimeError:
        consumed_rejected = True
    else:
        consumed_rejected = False
    observed = {
        'source_pre_interrupt_executions': len(pre_interrupt_effects),
        'unknown_id_rejected': unknown_rejected,
        'foreign_principal_rejected': foreign_rejected,
        'consumed_id_rejected': consumed_rejected,
        'approved_side_effects': len(side_effects),
    }
    assert observed == {
        'source_pre_interrupt_executions': 2,
        'unknown_id_rejected': True,
        'foreign_principal_rejected': True,
        'consumed_id_rejected': True,
        'approved_side_effects': 1,
    }
    return observed


class _FinalAnswer(BaseModel):
    value: str


def _stream_oracle() -> dict[str, str | int]:
    class SourceState(TypedDict):
        value: str

    def source_finish(_state: SourceState) -> dict[str, str]:
        return {'value': 'source-final'}

    source_builder = StateGraph(SourceState)
    source_builder.add_node('finish', source_finish)
    source_builder.add_edge(START, 'finish')
    source_builder.add_edge('finish', END)
    source_graph = source_builder.compile()

    async def source_stream() -> list[dict[str, Any]]:
        return [event async for event in source_graph.astream({'value': 'pending'}, stream_mode='updates')]

    source_events = asyncio.run(source_stream())

    run_rounds = 0

    def complete_model(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        nonlocal run_rounds
        run_rounds += 1
        if run_rounds == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart('complete_flaky', {'x': 1}),
                    ToolCallPart('final_result', {'value': 'premature'}),
                ]
            )
        return ModelResponse(parts=[ToolCallPart('final_result', {'value': 'corrected'})])

    complete_agent = Agent(FunctionModel(complete_model), output_type=_FinalAnswer, end_strategy='graceful')

    @complete_agent.tool_plain
    def complete_flaky(x: int) -> int:
        raise ModelRetry('not yet')

    complete = complete_agent.run_sync('go')

    stream_rounds = 0

    async def stream_model(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[dict[int, DeltaToolCall]]:
        nonlocal stream_rounds
        stream_rounds += 1
        yield {0: DeltaToolCall('stream_flaky', '{"x": 1}')}
        yield {1: DeltaToolCall('final_result', '{"value": "committed"}')}

    stream_agent = Agent(
        FunctionModel(stream_function=stream_model),
        output_type=_FinalAnswer,
        end_strategy='graceful',
    )

    @stream_agent.tool_plain
    def stream_flaky(x: int) -> int:
        raise ModelRetry('not yet')

    async def consume_stream() -> _FinalAnswer:
        async with stream_agent.run_stream('go') as streamed:
            return await streamed.get_output()

    streamed = asyncio.run(consume_stream())

    event_rounds = 0

    async def event_model(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[dict[int, DeltaToolCall]]:
        nonlocal event_rounds
        event_rounds += 1
        if event_rounds == 1:
            yield {
                0: DeltaToolCall('event_flaky', '{"x": 1}'),
                1: DeltaToolCall('final_result', '{"value": "premature"}'),
            }
        else:
            yield {0: DeltaToolCall('final_result', '{"value": "corrected"}')}

    event_agent = Agent(
        FunctionModel(stream_function=event_model),
        output_type=_FinalAnswer,
        end_strategy='graceful',
    )

    @event_agent.tool_plain
    def event_flaky(x: int) -> int:
        raise ModelRetry(f'not yet: {x}')

    async def consume_events() -> _FinalAnswer:
        completed: _FinalAnswer | None = None
        async with event_agent.run_stream_events('go') as events:
            async for event in events:
                if isinstance(event, AgentRunResultEvent):
                    completed = event.result.output
        assert completed is not None
        return completed

    event_result = asyncio.run(consume_events())
    observed = {
        'source_final': source_events[-1]['finish']['value'],
        'run_output': complete.output.value,
        'run_rounds': run_rounds,
        'stream_output': streamed.value,
        'stream_rounds': stream_rounds,
        'workaround_stream_events_output': event_result.value,
        'workaround_stream_events_rounds': event_rounds,
    }
    assert observed == {
        'source_final': 'source-final',
        'run_output': 'corrected',
        'run_rounds': 2,
        'stream_output': 'committed',
        'stream_rounds': 1,
        'workaround_stream_events_output': 'corrected',
        'workaround_stream_events_rounds': 2,
    }
    return observed


def main() -> None:
    """Run all trusted oracles and print their measured results as JSON."""
    print(
        json.dumps(
            {
                'provenance': _provenance(),
                'prompt_replacement_vs_composition': _prompt_oracle(),
                'retry_call_counts': _retry_oracle(),
                'interrupt_and_approval': _interrupt_oracle(),
                'run_vs_stream': _stream_oracle(),
            },
            sort_keys=True,
        )
    )


if __name__ == '__main__':
    main()
