from collections.abc import AsyncIterator, Callable
from pathlib import Path

import json_repair
import pytest
from logfire.testing import CaptureLogfire
from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage, ModelResponse, RetryPromptPart, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.filesystem import FileSystem
from pydantic_ai_harness.repair_tool_arguments import RepairToolArguments

pytestmark = pytest.mark.anyio


def model_for(respond: Callable[[list[ModelMessage], AgentInfo], ModelResponse]) -> FunctionModel:
    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
        for part in respond(messages, info).parts:
            if isinstance(part, ToolCallPart):
                arguments = part.args if isinstance(part.args, str) else part.args_as_json_str()
                yield {0: DeltaToolCall(name=part.tool_name, json_args=arguments)}
            else:
                assert isinstance(part, TextPart)
                yield part.content

    return FunctionModel(respond, stream_function=stream)


@pytest.fixture(params=['standalone', 'coder'])
def capabilities(request: pytest.FixtureRequest, tmp_path: Path) -> list[AbstractCapability[object]]:
    if request.param == 'coder':
        return [Coder(tmp_path)]
    return [RepairToolArguments(), FileSystem(root_dir=tmp_path, content_hashes=False)]


class TestRepairToolArguments:
    async def test_dictionary_arguments(self, capabilities: list[AbstractCapability[object]], tmp_path: Path) -> None:
        model = TestModel(call_tools=['write_file'], seed=0)
        agent = Agent(model, capabilities=capabilities)
        result = await agent.run('Write a file')
        assert not any(isinstance(part, RetryPromptPart) for message in result.all_messages() for part in message.parts)

    @pytest.mark.parametrize('error', [ValueError, RecursionError])
    async def test_repair_failure_retries(
        self,
        capabilities: list[AbstractCapability[object]],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        error: type[Exception],
    ) -> None:
        def fail(json_str: str, *, skip_json_loads: bool, ensure_ascii: bool) -> str:
            raise error('repair failed')

        monkeypatch.setattr(json_repair, 'repair_json', fail)
        await self.test_invalid_schema_still_retries(capabilities, tmp_path, '{"path": "hello.txt",}')

    @pytest.mark.parametrize(
        'arguments',
        [
            '{"path": "hello.txt", "content": "hello"}',
            {'path': 'hello.txt', 'content': 'hello'},
            '{"path": "hello.txt", "content": "hello",}',
            "{'path': 'hello.txt', 'content': 'hello'}",
            '{"path": "hello.txt", "content": "hello"',
        ],
    )
    async def test_write_arguments(
        self, capabilities: list[AbstractCapability[object]], tmp_path: Path, arguments: str | dict[str, str]
    ) -> None:
        calls = 0

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal calls
            calls += 1
            if calls == 1:
                return ModelResponse(parts=[ToolCallPart('write_file', arguments)])
            return ModelResponse(parts=[TextPart('done')])

        agent = Agent(model_for(respond), capabilities=capabilities)
        result = await agent.run('Write hello.txt')
        assert result.output == 'done'
        assert (tmp_path / 'hello.txt').read_text() == 'hello'
        assert calls == 2
        assert not any(isinstance(part, RetryPromptPart) for message in result.all_messages() for part in message.parts)

    @pytest.mark.parametrize('arguments', ['{"path": "hello.txt",}', '{"path": "hello.txt"}', 'not JSON'])
    async def test_invalid_schema_still_retries(
        self, capabilities: list[AbstractCapability[object]], tmp_path: Path, arguments: str
    ) -> None:
        calls = 0

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal calls
            calls += 1
            if calls == 1:
                return ModelResponse(parts=[ToolCallPart('write_file', arguments)])
            assert any(isinstance(part, RetryPromptPart) for message in messages for part in message.parts)
            return ModelResponse(parts=[TextPart('invalid arguments')])

        agent = Agent(model_for(respond), capabilities=capabilities)
        await agent.run('Write hello.txt')
        assert not (tmp_path / 'hello.txt').exists()

    async def test_repaired_edit_preserves_code(
        self, capabilities: list[AbstractCapability[object]], tmp_path: Path
    ) -> None:
        path = tmp_path / 'hello.py'
        path.write_text('old\n')
        calls = 0

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal calls
            calls += 1
            if calls == 1:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            'edit_file',
                            '{"path":"hello.py","old_text":"old\\n",'
                            '"new_text":"print(\\"日本語\\")\\npath = \\"C:\\\\tmp\\"\\n",}',
                        )
                    ]
                )
            return ModelResponse(parts=[TextPart('done')])

        agent = Agent(model_for(respond), capabilities=capabilities)
        await agent.run('Edit hello.py')
        assert path.read_text() == 'print("日本語")\npath = "C:\\tmp"\n'

    @pytest.mark.parametrize('arguments', ['{"value": "secret"}', '{"value": "secret",}', {'value': 'secret'}])
    async def test_standalone_custom_tool_and_telemetry(
        self, arguments: str | dict[str, str], capfire: CaptureLogfire
    ) -> None:
        calls = 0

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal calls
            calls += 1
            if calls == 1:
                return ModelResponse(parts=[ToolCallPart('echo', arguments)])
            return ModelResponse(parts=[TextPart('done')])

        agent = Agent(model_for(respond), capabilities=[RepairToolArguments()])
        agent.instrument = InstrumentationSettings()
        received: list[str] = []

        @agent.tool_plain
        def echo(value: str) -> str:
            received.append(value)
            return value

        await agent.run('Echo')
        assert received == ['secret']
        spans = [span for span in capfire.exporter.exported_spans_as_dict() if span['name'] == 'repair_tool_arguments']
        assert len(spans) == int(isinstance(arguments, str) and arguments.endswith(',}'))
        assert 'secret' not in str(spans)
