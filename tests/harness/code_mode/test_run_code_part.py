"""Tests for `RunCodeCallPart`, the typed tool part registered for `CodeMode`'s `run_code` tool."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, replace
from typing import Any

import pytest

from pydantic_ai import Agent, RunContext, Tool, ToolDefinition, ToolKind
from pydantic_ai.capabilities import (
    AbstractCapability,
    HandleDeferredToolCalls,
    ValidatedToolArgs,
    WrapToolExecuteHandler,
)
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, ToolDenied
from pydantic_ai_harness import CodeMode
from pydantic_ai_harness.code_mode import RunCodeCallPart

lookup_kind = ToolKind('code_mode_test', 'lookup')


class LookupCallPart(ToolCallPart, kind=lookup_kind):
    """Registers a non-core kind, like a capability's typed tool part would."""


@dataclass
class RecordExecutedCalls(AbstractCapability[Any]):
    """Records the call part every tool execution hook sees."""

    calls: list[ToolCallPart]

    async def wrap_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        handler: WrapToolExecuteHandler,
    ) -> Any:
        self.calls.append(call)
        return await handler(args)


@dataclass
class RecordRunCode(AbstractCapability[Any]):
    """Recognizes `run_code` executions by their kind, not their name."""

    codes: list[str | None]

    async def wrap_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        handler: WrapToolExecuteHandler,
    ) -> Any:
        if isinstance(call, RunCodeCallPart):
            self.codes.append(call.code)
        return await handler(args)


def _run_code_then_answer(code: str) -> Callable[[list[ModelMessage], AgentInfo], ModelResponse]:
    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            (run_code,) = info.function_tools
            return ModelResponse(parts=[ToolCallPart(run_code.name, {'code': code}, tool_call_id='c1')])
        return ModelResponse(parts=[TextPart('done')])

    return model_fn


class TestRunCodeCallPart:
    @pytest.mark.parametrize('speculate', [None, ['lookup']], ids=['cold', 'speculated'])
    async def test_a_tool_with_a_registered_non_core_kind_is_sandboxed(self, speculate: list[str] | None) -> None:
        """Hooks see the same typed part for a sandboxed call as for a direct one, so kind-keyed guards apply."""

        async def with_lookup_kind(ctx: RunContext[Any], tool_def: ToolDefinition) -> ToolDefinition:
            return replace(tool_def, tool_kind=lookup_kind)

        def lookup(item: str) -> str:
            """Look up an item."""
            return item

        tool_defs: list[ToolDefinition] = []
        code = "await lookup(item='widget')"

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            tool_defs.extend(info.function_tools)
            return _run_code_then_answer(code)(messages, info)

        async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            # Speculation launches calls from streamed `run_code` arguments, so it needs a streamed run.
            tool_defs.extend(info.function_tools)
            if len(messages) == 1:
                yield {0: DeltaToolCall(name='run_code', json_args=json.dumps({'code': code}), tool_call_id='c1')}
            else:
                yield 'done'

        executed: list[ToolCallPart] = []
        code_mode = CodeMode[object](speculate=speculate)
        agent = Agent(
            FunctionModel(model_fn, stream_function=stream_fn),
            tools=[Tool(lookup, prepare=with_lookup_kind)],
            capabilities=[code_mode, RecordExecutedCalls(executed)],
        )
        result = await agent.run('hi')

        assert {tool_def.name for tool_def in tool_defs} == {'run_code'}
        assert code_mode.speculation_stats.adopted == (1 if speculate else 0)
        lookup_call = next(call for call in executed if call.tool_name == 'lookup')
        assert isinstance(lookup_call, LookupCallPart)
        tool_return = result.all_messages()[2].parts[0]
        assert isinstance(tool_return, ToolReturnPart) and tool_return.content == 'widget'
        (nested_call,) = tool_return.metadata['tool_calls'].values()
        assert isinstance(nested_call, LookupCallPart)
        (nested_return,) = tool_return.metadata['tool_returns'].values()
        assert nested_return.tool_kind == lookup_kind.id

    async def test_approval_inside_the_sandbox_sees_the_typed_part(self) -> None:
        def lookup(item: str) -> str:
            """Look up an item."""
            # The handler denies the call, so the tool never runs.
            return item  # pragma: no cover

        requested: list[ToolCallPart] = []

        async def deny_all(ctx: RunContext[Any], requests: DeferredToolRequests) -> DeferredToolResults:
            requested.extend(requests.approvals)
            return DeferredToolResults(
                approvals={call.tool_call_id: ToolDenied(message='nope') for call in requests.approvals}
            )

        agent = Agent(
            FunctionModel(_run_code_then_answer("await lookup(item='widget')")),
            tools=[Tool(lookup, requires_approval=True, tool_kind=lookup_kind)],
            capabilities=[CodeMode[object](), HandleDeferredToolCalls(handler=deny_all)],
        )
        await agent.run('hi')

        (call,) = requested
        assert isinstance(call, LookupCallPart)

    async def test_run_code_tool_declares_its_kind(self) -> None:
        tool_defs: list[ToolDefinition] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            tool_defs.extend(info.function_tools)
            return ModelResponse(parts=[TextPart('done')])

        await Agent(FunctionModel(model_fn), capabilities=[CodeMode[object]()]).run('hi')

        assert [(td.name, td.tool_kind) for td in tool_defs] == [('run_code', 'code_mode.run_code')]
        assert RunCodeCallPart('run_code').tool_kind == 'code_mode.run_code'

    def test_narrow_type_yields_typed_args(self) -> None:
        part = ToolCallPart.narrow_type(
            ToolCallPart('run_code', {'code': 'print(1)', 'restart': True}, tool_call_id='c1'),
            tool_kind='code_mode.run_code',
        )
        assert isinstance(part, RunCodeCallPart)
        assert part.typed_args == {'code': 'print(1)', 'restart': True}
        assert part.code == 'print(1)'

        from_json = ToolCallPart.narrow_type(
            ToolCallPart('run_code', '{"code": "1 + 1"}', tool_call_id='c2'), tool_kind='code_mode.run_code'
        )
        assert isinstance(from_json, RunCodeCallPart)
        assert from_json.code == '1 + 1'

    def test_incomplete_args_have_no_code(self) -> None:
        assert RunCodeCallPart('run_code', args='{"code": "1 +').code is None
        assert RunCodeCallPart('run_code').code is None

    def test_args_without_code_are_not_promoted(self) -> None:
        part = ToolCallPart.narrow_type(
            ToolCallPart('run_code', {'source': '1'}, tool_call_id='c1'), tool_kind='code_mode.run_code'
        )
        assert type(part) is ToolCallPart

    async def test_wrap_tool_execute_recognizes_run_code_by_kind(self) -> None:
        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if len(messages) == 1:
                return ModelResponse(
                    parts=[
                        ToolCallPart('ping', {}, tool_call_id='c0'),
                        ToolCallPart('run_code', json.dumps({'code': '1 + 1'}), tool_call_id='c1'),
                    ]
                )
            return ModelResponse(parts=[TextPart('done')])

        codes: list[str | None] = []
        agent = Agent(FunctionModel(model_fn), capabilities=[CodeMode[object](tools=[]), RecordRunCode(codes)])

        @agent.tool_plain
        def ping() -> str:
            return 'pong'

        result = await agent.run('add')

        assert codes == ['1 + 1']
        ping_call, run_code_call = result.all_messages()[1].parts
        assert type(ping_call) is ToolCallPart
        assert isinstance(run_code_call, RunCodeCallPart) and run_code_call.code == '1 + 1'

    async def test_history_round_trips(self) -> None:
        agent = Agent(FunctionModel(_run_code_then_answer('3 + 3')), capabilities=[CodeMode[object]()])
        messages = (await agent.run('add')).all_messages()

        loaded = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(messages))

        assert loaded == messages
        call = loaded[1].parts[0]
        assert isinstance(call, RunCodeCallPart) and call.code == '3 + 3'
        tool_return = loaded[2].parts[0]
        assert isinstance(tool_return, ToolReturnPart) and tool_return.tool_kind == 'code_mode.run_code'
        assert tool_return.content == 6

    async def test_history_without_a_kind_still_loads_and_continues(self) -> None:
        """A history recorded before `run_code` declared its kind keeps its plain parts."""
        history: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart('add')]),
            ModelResponse(parts=[ToolCallPart('run_code', {'code': '1 + 1'}, tool_call_id='old')]),
            ModelRequest(parts=[ToolReturnPart('run_code', 2, tool_call_id='old')]),
            ModelResponse(parts=[TextPart('2')]),
        ]
        loaded = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(history))
        assert loaded == history
        assert type(loaded[1].parts[0]) is ToolCallPart

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if len(messages) == 5:
                return ModelResponse(parts=[ToolCallPart('run_code', {'code': '5 + 5'}, tool_call_id='new')])
            return ModelResponse(parts=[TextPart('10')])

        agent = Agent(FunctionModel(model_fn), capabilities=[CodeMode[object]()])
        messages = (await agent.run('again', message_history=loaded)).all_messages()

        assert type(messages[1].parts[0]) is ToolCallPart
        new_call = messages[5].parts[0]
        assert isinstance(new_call, RunCodeCallPart) and new_call.code == '5 + 5'
