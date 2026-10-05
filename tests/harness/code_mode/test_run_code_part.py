"""Tests for `RunCodeCallPart`, the typed tool part registered for `CodeMode`'s `run_code` tool."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent, RunContext, ToolDefinition
from pydantic_ai.capabilities import AbstractCapability, ValidatedToolArgs, WrapToolExecuteHandler
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
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness import CodeMode
from pydantic_ai_harness.code_mode import RunCodeCallPart


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
        narrowed = ToolCallPart.narrow_type(call, tool_kind=tool_def.tool_kind)
        if isinstance(narrowed, RunCodeCallPart):
            self.codes.append(narrowed.code)
        return await handler(args)


def _run_code_then_answer(
    code: str, *, json_args: bool = False
) -> Callable[[list[ModelMessage], AgentInfo], ModelResponse]:
    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            (run_code,) = info.function_tools
            args = json.dumps({'code': code}) if json_args else {'code': code}
            return ModelResponse(parts=[ToolCallPart(run_code.name, args, tool_call_id='c1')])
        return ModelResponse(parts=[TextPart('done')])

    return model_fn


class TestRunCodeCallPart:
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
        assert RunCodeCallPart('run_code', '{"code": "1 +').code is None
        assert RunCodeCallPart('run_code').code is None

    def test_args_without_code_are_not_promoted(self) -> None:
        part = ToolCallPart.narrow_type(
            ToolCallPart('run_code', {'source': '1'}, tool_call_id='c1'), tool_kind='code_mode.run_code'
        )
        assert type(part) is ToolCallPart

    async def test_wrap_tool_execute_recognizes_run_code_by_kind(self) -> None:
        codes: list[str | None] = []
        agent = Agent(
            FunctionModel(_run_code_then_answer('1 + 1', json_args=True)),
            capabilities=[CodeMode[object](), RecordRunCode(codes)],
        )

        result = await agent.run('add')

        assert codes == ['1 + 1']
        call = result.all_messages()[1].parts[0]
        assert isinstance(call, RunCodeCallPart) and call.code == '1 + 1'

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
