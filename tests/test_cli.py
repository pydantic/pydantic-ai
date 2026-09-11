from __future__ import annotations

import json
import sys
import types
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import nullcontext
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Literal

import anyio
import pytest
import sniffio
from pydantic import JsonValue
from pytest import CaptureFixture
from pytest_mock import MockerFixture
from rich.cells import cell_len
from rich.console import Console, RenderableType
from rich.live import Live
from rich.live_render import LiveRender

from pydantic_ai import Agent, ModelMessage, ModelResponse, ModelRetry, TextPart, ToolCallPart
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.messages import RetryPromptPart, ToolReturnPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.toolsets import FunctionToolset, WrapperToolset
from pydantic_ai.usage import RunUsage, UsageLimits

from ._inline_snapshot import snapshot
from .conftest import IsInstance, IsStr, TestEnv, try_import

with try_import() as imports_successful:
    from prompt_toolkit.buffer import Buffer
    from prompt_toolkit.document import Document
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput
    from prompt_toolkit.shortcuts import PromptSession

    from pydantic_ai._cli import (
        CustomAutoSuggest,
        ask_agent,
        cli,
        cli_agent,
        format_usage,
        handle_slash_command,
        run_chat,
    )
    from pydantic_ai._cli.web import run_web_command
    from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
    from pydantic_ai.models.openai import OpenAIChatModel

pytestmark = pytest.mark.skipif(not imports_successful(), reason='install cli extras to run cli tests')


@pytest.fixture
def blockbuster_excluded_modules() -> tuple[str, ...]:
    """The CLI owns intentionally synchronous terminal and history operations."""
    return ('pydantic_ai._cli',)


@pytest.fixture(autouse=True)
def reset_sniffio_cvar() -> Iterator[None]:
    # The anyio pytest plugin sets `current_async_library_cvar` to 'asyncio' at session
    # start and the value leaks into sync tests, causing `anyio.run` to refuse to start.
    token = sniffio.current_async_library_cvar.set(None)
    try:
        yield
    finally:
        sniffio.current_async_library_cvar.reset(token)


def test_cli_version(capfd: CaptureFixture[str]):
    assert cli(['--version']) == 0
    assert capfd.readouterr().out.startswith('clai - Pydantic AI CLI')


def test_invalid_model(capfd: CaptureFixture[str]):
    assert cli(['--model', 'potato']) == 1
    assert capfd.readouterr().out.splitlines() == snapshot(['Error initializing potato:', 'Unknown model: potato'])


def test_invalid_model_suggestion(capfd: CaptureFixture[str]):
    assert cli(['--model', 'claude:sonnet-5']) == 1
    assert capfd.readouterr().out.splitlines() == snapshot(
        [
            'Error initializing claude:sonnet-5:',
            "Unknown model: claude:sonnet-5. Did you mean 'anthropic:claude-sonnet-5'?",
        ]
    )


@pytest.fixture
def create_test_module():
    def _create_test_module(**namespace: Any) -> None:
        assert 'test_module' not in sys.modules

        test_module = types.ModuleType('test_module')
        for key, value in namespace.items():
            setattr(test_module, key, value)

        sys.modules['test_module'] = test_module

    try:
        yield _create_test_module
    finally:
        if 'test_module' in sys.modules:  # pragma: no branch
            del sys.modules['test_module']


def test_agent_flag(
    capfd: CaptureFixture[str],
    mocker: MockerFixture,
    env: TestEnv,
    create_test_module: Callable[..., None],
):
    env.remove('OPENAI_API_KEY')
    env.set('COLUMNS', '150')

    test_agent = Agent(TestModel(custom_output_text='Hello from custom agent'))
    create_test_module(custom_agent=test_agent)

    # Mock ask_agent to avoid actual execution but capture the agent
    mock_ask = mocker.patch('pydantic_ai._cli.ask_agent')

    # Test CLI with custom agent
    assert cli(['--agent', 'test_module:custom_agent', 'hello']) == 0

    # Verify the output contains the custom agent message
    assert 'using custom agent test_module:custom_agent' in capfd.readouterr().out.replace('\n', '')

    # Verify ask_agent was called with our custom agent
    mock_ask.assert_called_once()
    assert mock_ask.call_args[0][0] is test_agent


def test_agent_flag_no_model(capfd: CaptureFixture[str], env: TestEnv, create_test_module: Callable[..., None]):
    env.remove('OPENAI_API_KEY')
    env.remove('OPENAI_ADMIN_KEY')
    test_agent = Agent()
    create_test_module(custom_agent=test_agent)

    # The missing key now surfaces as a `UserError`, which the CLI reports cleanly instead of
    # letting the provider SDK's exception escape as a traceback.
    assert cli(['--agent', 'test_module:custom_agent', 'hello']) == 1
    output = capfd.readouterr().out
    assert 'Error initializing' in output
    assert 'Set the `OPENAI_API_KEY` environment variable' in output


def test_agent_flag_set_model(
    capfd: CaptureFixture[str],
    mocker: MockerFixture,
    env: TestEnv,
    create_test_module: Callable[..., None],
):
    env.set('OPENAI_API_KEY', 'xxx')
    env.set('COLUMNS', '150')

    custom_agent = Agent(TestModel(custom_output_text='Hello from custom agent'))
    create_test_module(custom_agent=custom_agent)

    mocker.patch('pydantic_ai._cli.ask_agent')

    assert cli(['--agent', 'test_module:custom_agent', '--model', 'openai-chat:gpt-4o', 'hello']) == 0

    # CLI banner shows `model.model_id` which is `{system}:{model_name}` — `OpenAIChatModel.system`
    # is `'openai'`, so the banner prints `'openai:gpt-4o'` even when the user passed `'openai-chat:'`.
    assert 'using custom agent test_module:custom_agent with openai:gpt-4o' in capfd.readouterr().out.replace('\n', '')

    assert isinstance(custom_agent.model, OpenAIChatModel)


def test_agent_flag_non_agent(
    capfd: CaptureFixture[str], mocker: MockerFixture, create_test_module: Callable[..., None]
):
    test_agent = 'Not an Agent object'
    create_test_module(custom_agent=test_agent)

    assert cli(['--agent', 'test_module:custom_agent', 'hello']) == 1
    assert 'Could not load agent from test_module:custom_agent' in capfd.readouterr().out


def test_agent_flag_bad_module_variable_path(capfd: CaptureFixture[str], mocker: MockerFixture, env: TestEnv):
    assert cli(['--agent', 'bad_path', 'hello']) == 1
    assert 'Could not load agent from bad_path' in capfd.readouterr().out


@pytest.mark.parametrize('stream', [False, True])
def test_mcp_config(capfd: CaptureFixture[str], env: TestEnv, tmp_path: Path, stream: bool):
    """`--mcp-config` parses a Claude-Desktop-style config and connects to the MCP server.

    Drives the real `load_mcp_toolsets` -> `MCPToolset` path against `tests.mcp_server` over stdio,
    without mocking the loader under test. `TestModel` then calls the prefixed MCP tool through
    both the streaming and non-streaming paths.
    """
    env.set('OPENAI_API_KEY', 'test')
    config_file = tmp_path / 'mcp_servers.json'
    config_file.write_text(
        json.dumps({'mcpServers': {'temp': {'command': 'python', 'args': ['-m', 'tests.mcp_server']}}})
    )

    args = ['--mcp-config', str(config_file), 'weather in Mexico City?']
    if not stream:
        args.insert(0, '--no-stream')
    with cli_agent.override(model=TestModel(call_tools=['temp_get_weather_forecast'])):
        assert cli(args) == 0

    output = capfd.readouterr().out
    assert 'temp_get_weather_forecast' in output
    assert 'The weather in a is sunny and 26 degrees Celsius.' in ' '.join(output.split())
    assert ('Called tool temp_get_weather_forecast' in output) is stream


def test_mcp_config_interactive(capfd: CaptureFixture[str], mocker: MockerFixture, env: TestEnv, tmp_path: Path):
    """Interactive `clai --mcp-config` can invoke a configured server tool before exiting."""
    env.set('OPENAI_API_KEY', 'test')
    config_file = tmp_path / 'mcp_servers.json'
    config_file.write_text(
        json.dumps({'mcpServers': {'temp': {'command': 'python', 'args': ['-m', 'tests.mcp_server']}}})
    )

    with create_pipe_input() as inp:
        inp.send_text('weather in Mexico City?\n')
        inp.send_text('/exit\n')
        session = PromptSession[Any](input=inp, output=DummyOutput())
        mocker.patch('pydantic_ai._cli.PromptSession', return_value=session)

        with cli_agent.override(model=TestModel(call_tools=['temp_get_weather_forecast'])):
            assert cli(['--mcp-config', str(config_file)]) == 0

    output = capfd.readouterr().out
    assert 'Called tool temp_get_weather_forecast' in output
    assert 'The weather in a is sunny and 26 degrees Celsius.' in ' '.join(output.split())


# Sentinel for the case where `--mcp-config` is handed an existing path that isn't a readable file.
DIRECTORY_CONFIG = 'directory-instead-of-file'


@pytest.mark.parametrize(
    'config,expected',
    [
        pytest.param(None, 'not found', id='missing-file'),
        pytest.param('{ not valid json ', 'key must be a string', id='malformed-json'),
        pytest.param(
            json.dumps({'mcpServers': {'x': {'command': 'echo', 'args': ['${UNDEFINED_VAR_XYZ}']}}}),
            'is not defined',
            id='undefined-env-var',
        ),
        pytest.param(
            json.dumps({'mcpServers': {'x': {}}}),
            'must have either `command` or `url`',
            id='missing-command-or-url',
        ),
        pytest.param(
            json.dumps({'mcpServers': {'x': None}}),
            'Input should be a valid dictionary',
            id='non-object-server-entry',
        ),
        pytest.param(
            json.dumps({'mcpServers': {'x': {'command': 'echo', 'args': 'not-a-list'}}}),
            'Input should be a valid list',
            id='wrongly-typed-field',
        ),
        pytest.param(
            DIRECTORY_CONFIG,
            'Is a directory',
            id='directory-not-a-file',
        ),
    ],
)
def test_mcp_config_errors(capfd: CaptureFixture[str], env: TestEnv, tmp_path: Path, config: str | None, expected: str):
    """A bad `--mcp-config` prints a friendly error and exits 1 instead of a raw traceback."""
    env.set('OPENAI_API_KEY', 'test')
    config_file = tmp_path / 'mcp_servers.json'
    if config is None:
        target = tmp_path / 'does_not_exist.json'
    elif config == DIRECTORY_CONFIG:
        target = tmp_path
    else:
        config_file.write_text(config)
        target = config_file

    assert cli(['--mcp-config', str(target), 'hi']) == 1
    out = capfd.readouterr().out
    assert 'Could not load MCP config' in out
    assert expected in out


def test_mcp_config_empty_value(capfd: CaptureFixture[str], env: TestEnv):
    """`--mcp-config=` errors instead of silently starting without MCP servers.

    An empty value is falsy, so a truthiness check would skip MCP entirely — which is what
    `--mcp-config="$UNSET_VAR"` expands to in a script.
    """
    env.set('OPENAI_API_KEY', 'test')
    assert cli(['--mcp-config=', 'hi']) == 1
    assert 'needs a path to a configuration file' in capfd.readouterr().out


def test_no_command_defaults_to_chat(mocker: MockerFixture):
    """Test that running clai with no command defaults to chat mode."""
    # Mock _run_chat_command to avoid actual execution
    mock_run_chat = mocker.patch('pydantic_ai._cli._run_chat_command', return_value=0)
    result = cli([])
    assert result == 0
    mock_run_chat.assert_called_once()


def test_list_models(capfd: CaptureFixture[str]):
    assert cli(['--list-models']) == 0
    output = capfd.readouterr().out.splitlines()
    assert output[:3] == snapshot([IsStr(regex='clai - Pydantic AI CLI .*'), '', 'Available models:'])

    providers = (
        'openai',
        'anthropic',
        'bedrock',
        'cerebras',
        'crusoe',
        'google',
        'google-cloud',
        'groq',
        'mistral',
        'cohere',
        'deepseek',
        'gateway/',
        'heroku',
        'moonshotai',
        'xai',
        'huggingface',
        'zai',
        'snowflake',
    )
    models = {line.strip().split(' ')[0] for line in output[3:]}
    for provider in providers:
        models = models - {model for model in models if model.startswith(provider)}
    assert models == set(), models


def test_cli_prompt(capfd: CaptureFixture[str], env: TestEnv):
    env.set('OPENAI_API_KEY', 'test')
    with cli_agent.override(model=TestModel(custom_output_text='# result\n\n```py\nx = 1\n```')):
        assert cli(['hello']) == 0
        assert capfd.readouterr().out.splitlines() == snapshot([IsStr(), '# result', '', 'py', 'x = 1', '/py'])
        assert cli(['--no-stream', 'hello']) == 0
        assert capfd.readouterr().out.splitlines() == snapshot([IsStr(), '# result', '', 'py', 'x = 1', '/py'])


@pytest.mark.anyio
@pytest.mark.parametrize('show_tool_calls', [True, False])
async def test_streaming_with_tool_calls(show_tool_calls: bool, live_frames: list[str]):
    """The streaming CLI render loop interleaves streamed model text with tool-call indicators.

    Uses a `FunctionModel` stream so the agent emits real text deltas and a tool call, exercising
    `ask_agent`'s render loop end to end rather than `TestModel`'s canned output.
    """

    async def weather_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            yield 'It is '
            yield 'sunny in Mexico City.'
        else:
            yield 'Let me check '
            yield 'the weather.'
            yield {0: DeltaToolCall(name='get_weather', json_args='{"city": "Mexico City"}', tool_call_id='call_1')}

    agent = Agent(FunctionModel(stream_function=weather_stream))

    @agent.tool_plain
    def get_weather(city: str) -> str:
        return f'sunny in {city}'

    output = StringIO()
    console = Console(file=output, force_terminal=False, width=80)
    messages = await ask_agent(
        agent, 'weather?', stream=True, console=console, code_theme='monokai', show_tool_calls=show_tool_calls
    )

    expected_lines = ['Let me check the weather.', '']
    if show_tool_calls:
        expected_lines.extend(["▌ Called tool get_weather(city='Mexico City').", ''])
    expected_lines.append('It is sunny in Mexico City.')
    assert [line.rstrip() for line in output.getvalue().splitlines()] == expected_lines
    assert any('Calling tool' in frame for frame in live_frames) is show_tool_calls
    assert isinstance(messages[-1], ModelResponse)
    assert messages[-1].parts[-1] == TextPart(content='It is sunny in Mexico City.')


@pytest.mark.anyio
@pytest.mark.parametrize('multiline', [False, True])
async def test_tool_call_spacing_across_model_requests(multiline: bool):
    async def run_code_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        count = sum(isinstance(part, ToolReturnPart) for message in messages for part in message.parts)
        if count == 3:
            yield 'Done.'
        else:
            if count == 0:
                yield 'Checking.'
            elif count == 1:
                yield '   '
            yield {
                0: DeltaToolCall(
                    name='run_code',
                    json_args=json.dumps({'code': f'print({count})' + ('\n# end' if multiline else '')}),
                    tool_call_id=f'call_{count}',
                )
            }

    agent = Agent(FunctionModel(stream_function=run_code_stream))

    @agent.tool_plain
    def run_code(code: str) -> str:
        return code

    output = StringIO()
    await ask_agent(agent, 'go', stream=True, console=Console(file=output, width=80), code_theme='monokai')
    expected = ['Checking.', '']
    for count in range(3):
        if multiline:
            expected.extend(
                ['▌ Called tool run_code(code=<2 lines>).', '▌   code:', f'▌     print({count})', '▌     # end']
            )
        else:
            expected.append(f"▌ Called tool run_code(code='print({count})').")
    expected.extend(['', 'Done.'])
    assert [line.rstrip() for line in output.getvalue().splitlines()] == expected


def _distinct_frames(frames: list[str]) -> list[str]:
    """Drop consecutive duplicates — the same frame re-rendered carries no information."""
    return [frame for i, frame in enumerate(frames) if i == 0 or frame != frames[i - 1]]


@pytest.fixture
def live_frames(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Capture rendered intermediate frames, including calls that finish before the final frame."""
    frames: list[str] = []
    original_update = Live.update

    def capture_update(self: Live, renderable: RenderableType, *, refresh: bool = False) -> None:
        output = StringIO()
        Console(file=output, width=self.console.width, height=self.console.height).print(
            LiveRender(renderable, vertical_overflow='ellipsis')
        )
        frames.append('\n'.join(line.rstrip() for line in output.getvalue().splitlines()).rstrip())
        return original_update(self, renderable, refresh=refresh)

    monkeypatch.setattr(Live, 'update', capture_update)
    return frames


@pytest.mark.anyio
async def test_streaming_with_concurrent_tool_calls(live_frames: list[str]):
    """Every in-flight tool call keeps its own indicator until its own result arrives.

    Two tools called in one response run concurrently, so the render loop keys them by
    `tool_call_id`. Rendering only the most recent call erased the earlier one's indicator, leaving
    a still-running tool with no line at all. The final console output can't show that — the
    regression lives in the intermediate `Live` frames, so those are captured here.

    Asserted as order-independent invariants rather than a frame snapshot: the two results are
    emitted by concurrent tasks, so their order is not stable and a snapshot of the sequence would
    flake. `get_temp` waits on `get_weather` only to guarantee the two calls overlap.
    """
    weather_returned = anyio.Event()

    async def two_city_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            yield 'Both cities checked.'
        else:
            yield 'Checking two cities.'
            yield {
                0: DeltaToolCall(name='get_weather', json_args='{"city": "Lisbon"}', tool_call_id='call_1'),
                1: DeltaToolCall(name='get_temp', json_args='{"city": "Porto"}', tool_call_id='call_2'),
            }

    agent = Agent(FunctionModel(stream_function=two_city_stream))

    @agent.tool_plain
    async def get_weather(city: str) -> str:
        weather_returned.set()
        return f'sunny in {city}'

    @agent.tool_plain
    async def get_temp(city: str) -> str:
        await weather_returned.wait()
        return f'20C in {city}'

    console = Console(file=StringIO(), force_terminal=False, width=80)
    await ask_agent(agent, 'weather?', stream=True, console=console, code_theme='monokai')

    distinct = _distinct_frames(live_frames)

    # The regression: the second call's indicator replaced the first's, so this frame never existed.
    assert any(
        "Calling tool get_weather(city='Lisbon')…" in frame and "Calling tool get_temp(city='Porto')…" in frame
        for frame in distinct
    )
    # And once one call finished, the other was left with no indicator at all.
    assert any('Called tool ' in frame and 'Calling tool ' in frame for frame in distinct)

    # The final frame holds both completions; their order follows task completion, so assert
    # membership rather than a sequence.
    final = distinct[-1]
    assert final.startswith('Checking two cities.')
    assert "Called tool get_weather(city='Lisbon')." in final
    assert "Called tool get_temp(city='Porto')." in final
    assert 'Calling tool' not in final
    assert final.endswith('Both cities checked.')


@dataclass
class _City:
    name: str


@pytest.mark.anyio
async def test_streaming_ignores_output_tool_calls():
    """The internal output tool is not reported as a tool call.

    A structured `output_type` makes the agent call an output tool, which streams
    `OutputToolCallEvent` / `OutputToolResultEvent` through the same node as function tools. Those
    are the model producing its answer rather than the agent using a tool, so they get no indicator.
    """
    agent = Agent(TestModel(), output_type=_City)

    output = StringIO()
    console = Console(file=output, force_terminal=False, width=80)
    await ask_agent(agent, 'name a city', stream=True, console=console, code_theme='monokai')

    rendered = output.getvalue()
    assert 'Calling tool' not in rendered
    assert 'Called tool' not in rendered


@pytest.mark.anyio
async def test_streaming_clears_indicator_for_retried_tool(live_frames: list[str]):
    """A call that comes back as a retry drops its in-flight indicator instead of pinning it.

    `pending_calls` is popped for any `FunctionToolResultEvent`, not only a `ToolReturnPart`.
    Popping on success alone left `> _Calling tool ...` on screen for the rest of the run. A retry
    still renders no `Called tool` line — surfacing retries is a separate feature.
    """

    async def retrying_tool_stream(
        messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[str | DeltaToolCalls]:
        if any(isinstance(part, RetryPromptPart) for message in messages for part in message.parts):
            yield 'Recovered without the tool.'
        else:
            yield 'Trying a tool.'
            yield {0: DeltaToolCall(name='flaky', json_args='{}', tool_call_id='call_1')}

    agent = Agent(FunctionModel(stream_function=retrying_tool_stream))

    @agent.tool_plain
    def flaky() -> str:
        raise ModelRetry('not this time')

    console = Console(file=StringIO(), force_terminal=False, width=80)
    await ask_agent(agent, 'go', stream=True, console=console, code_theme='monokai')

    distinct = _distinct_frames(live_frames)
    assert distinct == snapshot(
        [
            'Trying a tool.',
            """\
Trying a tool.

▌ Calling tool flaky()…\
""",
            'Trying a tool.',
            """\
Trying a tool.

Recovered without the tool.\
""",
        ]
    )


@dataclass
class _LifetimeToolset(WrapperToolset[Any]):
    """Counts how often it is fully released, to observe whether it survives between turns."""

    depth: int = 0
    full_releases: int = 0

    async def __aenter__(self) -> _LifetimeToolset:
        self.depth += 1
        await super().__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        result = await super().__aexit__(*args)
        self.depth -= 1
        if self.depth == 0:
            self.full_releases += 1
        return result


@pytest.mark.anyio
async def test_chat_holds_toolsets_open_for_the_session(mocker: MockerFixture, env: TestEnv, tmp_path: Path):
    """Run-scoped toolsets survive between turns instead of being torn down after each one.

    Passing them straight to each `agent.run()` fully released them every turn — for an MCP server
    that means a fresh subprocess and `tools/list` handshake per message, and server-side session
    state lost in between. `run_chat` now holds them open around the REPL loop, so each turn's own
    enter is a ref-count bump and the toolset is released once, at the end.
    """
    env.set('OPENAI_API_KEY', 'test')
    toolset = _LifetimeToolset(FunctionToolset[Any]())

    with create_pipe_input() as inp:
        inp.send_text('hello\n')
        inp.send_text('hello again\n')
        inp.send_text('/exit\n')
        session = PromptSession[Any](input=inp, output=DummyOutput())
        mocker.patch('pydantic_ai._cli.PromptSession', return_value=session)

        agent = Agent(TestModel(custom_output_text='goodbye'))
        console = Console(file=StringIO(), force_terminal=False, width=80)
        assert await run_chat(True, agent, console, 'monokai', 'clai', config_dir=tmp_path, toolsets=[toolset]) == 0

    # Two turns ran, but the toolset was released once — at the end of the session, not per turn.
    assert toolset.full_releases == snapshot(1)


@pytest.mark.anyio
async def test_chat_keeps_toolsets_open_after_failed_turn(mocker: MockerFixture, env: TestEnv, tmp_path: Path):
    """A failed turn is reported without releasing session toolsets or ending the REPL."""
    env.set('OPENAI_API_KEY', 'test')
    toolset = _LifetimeToolset(FunctionToolset[Any]())
    attempts = 0

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError('first turn failed') from ValueError('underlying failure')
        if attempts == 2:
            raise RuntimeError('second turn failed')
        return ModelResponse(parts=[TextPart('second turn succeeded')])

    with create_pipe_input() as inp:
        inp.send_text('fail\n')
        inp.send_text('fail again\n')
        inp.send_text('succeed\n')
        inp.send_text('/exit\n')
        session = PromptSession[Any](input=inp, output=DummyOutput())
        mocker.patch('pydantic_ai._cli.PromptSession', return_value=session)

        output = StringIO()
        console = Console(file=output, force_terminal=False, width=80)
        agent = Agent(FunctionModel(function=respond))
        assert await run_chat(False, agent, console, 'monokai', 'clai', config_dir=tmp_path, toolsets=[toolset]) == 0

    rendered = output.getvalue()
    assert 'RuntimeError: first turn failed' in rendered
    assert 'Caused by: underlying failure' in rendered
    assert 'RuntimeError: second turn failed' in rendered
    assert 'second turn succeeded' in rendered
    assert attempts == snapshot(3)
    assert toolset.full_releases == snapshot(1)


def test_custom_auto_suggest_completes_special_commands():
    """Typing a prefix of a slash command suggests the rest; anything else falls back to history."""
    suggest = CustomAutoSuggest(['/exit'])
    buffer = Buffer()

    suggestion = suggest.get_suggestion(buffer, Document('/ex'))
    assert suggestion is not None
    assert suggestion.text == snapshot('it')

    # No special command matches, and an empty history has nothing to offer either.
    assert suggest.get_suggestion(buffer, Document('hello')) is None


def test_chat(capfd: CaptureFixture[str], mocker: MockerFixture, env: TestEnv):
    env.set('OPENAI_API_KEY', 'test')

    # mocking is needed because of ci does not have xclip or xselect installed
    def mock_copy(text: str) -> None:
        pass

    mocker.patch('pyperclip.copy', mock_copy)
    with create_pipe_input() as inp:
        inp.send_text('\n')
        inp.send_text('hello\n')
        inp.send_text('/markdown\n')
        inp.send_text('/cp\n')
        # a second agent turn, so `/usage` proves the session accumulator sums across the `run_chat` loop
        inp.send_text('hello again\n')
        inp.send_text('/usage\n')
        inp.send_text('/usage --json\n')
        inp.send_text('/exit\n')
        session = PromptSession[Any](input=inp, output=DummyOutput())
        m = mocker.patch('pydantic_ai._cli.PromptSession', return_value=session)
        m.return_value = session
        m = TestModel(custom_output_text='goodbye')
        with cli_agent.override(model=m):
            assert cli([]) == 0
        assert capfd.readouterr().out.splitlines() == snapshot(
            [
                IsStr(),
                IsStr(regex='goodbye *Markdown output of last question:'),
                '',
                'goodbye',
                'Copied last output to clipboard.',
                IsStr(regex=r'goodbye *clai usage \(session total\)'),
                '',
                'Turns:      2',
                IsStr(regex=r'Tokens: .*'),
                IsStr(regex=r'  Input: .*'),
                IsStr(regex=r'  Output: .*'),
                'Requests:   2',
                'Tool calls: 0',
                IsStr(regex=r'\{"turns": 2, .*"requests": 2, "tool_calls": 0\}'),
                'Exiting…',
            ]
        )


def test_handle_slash_command_markdown():
    io = StringIO()
    assert handle_slash_command('/markdown', [], False, Console(file=io), 'default') == (None, False)
    assert io.getvalue() == snapshot('No markdown output available.\n')

    messages: list[ModelMessage] = [ModelResponse(parts=[TextPart('[hello](# hello)'), ToolCallPart('foo', '{}')])]
    io = StringIO()
    assert handle_slash_command('/markdown', messages, True, Console(file=io), 'default') == (None, True)
    assert io.getvalue() == snapshot("""\
Markdown output of last question:

[hello](# hello)
""")


def test_handle_slash_command_multiline():
    io = StringIO()
    assert handle_slash_command('/multiline', [], False, Console(file=io), 'default') == (None, True)
    assert io.getvalue()[:70] == IsStr(regex=r'Enabling multiline mode.*')

    io = StringIO()
    assert handle_slash_command('/multiline', [], True, Console(file=io), 'default') == (None, False)
    assert io.getvalue() == snapshot('Disabling multiline mode.\n')


def test_handle_slash_command_copy(mocker: MockerFixture):
    io = StringIO()
    # mocking is needed because of ci does not have xclip or xselect installed
    mock_clipboard: list[str] = []

    def append_to_clipboard(text: str) -> None:
        mock_clipboard.append(text)

    mocker.patch('pyperclip.copy', append_to_clipboard)
    assert handle_slash_command('/cp', [], False, Console(file=io), 'default') == (None, False)
    assert io.getvalue() == snapshot('No output available to copy.\n')
    assert mock_clipboard == snapshot([])

    messages: list[ModelMessage] = [ModelResponse(parts=[TextPart(''), ToolCallPart('foo', '{}')])]
    io = StringIO()
    assert handle_slash_command('/cp', messages, True, Console(file=io), 'default') == (None, True)
    assert io.getvalue() == snapshot('No text content to copy.\n')
    assert mock_clipboard == snapshot([])

    messages: list[ModelMessage] = [ModelResponse(parts=[TextPart('hello'), ToolCallPart('foo', '{}')])]
    io = StringIO()
    assert handle_slash_command('/cp', messages, True, Console(file=io), 'default') == (None, True)
    assert io.getvalue() == snapshot('Copied last output to clipboard.\n')
    assert mock_clipboard == snapshot(['hello'])


def test_handle_slash_command_exit():
    io = StringIO()
    assert handle_slash_command('/exit', [], False, Console(file=io), 'default') == (0, False)
    assert io.getvalue() == snapshot('Exiting…\n')


def test_handle_slash_command_other():
    io = StringIO()
    assert handle_slash_command('/foobar', [], False, Console(file=io), 'default') == (None, False)
    assert io.getvalue() == snapshot('Unknown command `/foobar`\n')


def test_format_usage():
    usage = RunUsage(input_tokens=1521, output_tokens=326, requests=1)
    assert format_usage(usage, 1) == snapshot("""\
clai usage (session total)

Turns:      1
Tokens:     1,847
  Input:    1,521
  Output:   326
Requests:   1
Tool calls: 0\
""")


def test_format_usage_json():
    usage = RunUsage(input_tokens=1521, output_tokens=326, requests=1, tool_calls=2)
    assert format_usage(usage, 3, as_json=True) == snapshot(
        '{"turns": 3, "input_tokens": 1521, "output_tokens": 326, "total_tokens": 1847, "requests": 1, "tool_calls": 2}'
    )


def test_handle_slash_command_usage():
    usage = RunUsage(input_tokens=51, output_tokens=1, requests=1)
    io = StringIO()
    assert handle_slash_command('/usage', [], False, Console(file=io), 'default', usage=usage, turns=1) == (None, False)
    assert io.getvalue() == snapshot("""\
clai usage (session total)

Turns:      1
Tokens:     52
  Input:    51
  Output:   1
Requests:   1
Tool calls: 0
""")


def test_handle_slash_command_usage_json():
    usage = RunUsage(input_tokens=51, output_tokens=1, requests=1)
    io = StringIO()
    # Spaces are replaced with `-` upstream, so `/usage --json` arrives as `/usage---json`.
    assert handle_slash_command('/usage---json', [], False, Console(file=io), 'default', usage=usage, turns=1) == (
        None,
        False,
    )
    assert io.getvalue() == snapshot(
        '{"turns": 1, "input_tokens": 51, "output_tokens": 1, "total_tokens": 52, "requests": 1, "tool_calls": 0}\n'
    )


def test_handle_slash_command_usage_no_session():
    # With no session usage threaded through, fall back to an empty RunUsage rather than erroring.
    io = StringIO()
    assert handle_slash_command('/usage', [], False, Console(file=io), 'default') == (None, False)
    assert io.getvalue() == snapshot("""\
clai usage (session total)

Turns:      0
Tokens:     0
  Input:    0
  Output:   0
Requests:   0
Tool calls: 0
""")


def test_handle_slash_command_usage_unknown_option():
    io = StringIO()
    assert handle_slash_command('/usage---foo', [], False, Console(file=io), 'default', usage=RunUsage(), turns=0) == (
        None,
        False,
    )
    assert io.getvalue() == snapshot('Unknown `/usage` option `/usage---foo`\n')


def test_handle_slash_command_usage_not_a_flag():
    # Without a separator, `/usagejson` is an unknown command, not a silently-accepted `--json` request.
    io = StringIO()
    assert handle_slash_command('/usagejson', [], False, Console(file=io), 'default', usage=RunUsage(), turns=0) == (
        None,
        False,
    )
    assert io.getvalue() == snapshot('Unknown command `/usagejson`\n')


@pytest.mark.anyio
async def test_ask_agent_accumulates_usage(env: TestEnv):
    # `ask_agent` increments the shared session usage on both the streaming and non-streaming paths,
    # including tool calls, and threading a turn's history into the next must not re-count prior usage.
    env.set('OPENAI_API_KEY', 'test')
    agent = Agent(TestModel())

    @agent.tool_plain
    def get_number() -> int:
        return 42

    session_usage = RunUsage()
    console = Console(file=StringIO())

    # streaming turn
    messages = await ask_agent(agent, 'first', True, console, 'default', usage=session_usage)
    # non-streaming turn, threading the prior turn's history back in as `run_chat` does
    await ask_agent(agent, 'second', False, console, 'default', messages=messages, usage=session_usage)

    # Usage is the sum of both runs: the first turn calls the tool (2 requests, 1 tool call), the second
    # replays that history without re-calling it (1 request). Prior usage is not re-counted from history.
    assert session_usage == snapshot(RunUsage(input_tokens=156, output_tokens=14, requests=3, tool_calls=1))


@pytest.mark.anyio
async def test_ask_agent_counts_usage_on_failed_turn(env: TestEnv):
    # A turn that makes a billed request and then raises must still be counted, so a later `/usage`
    # does not under-report tokens that were actually spent. The merge happens in `ask_agent`'s `finally`.
    env.set('OPENAI_API_KEY', 'test')
    agent = Agent(TestModel())

    @agent.tool_plain
    def boom() -> int:
        raise RuntimeError('boom')

    session_usage = RunUsage()
    console = Console(file=StringIO())

    with pytest.raises(RuntimeError, match='boom'):
        await ask_agent(agent, 'go', False, console, 'default', usage=session_usage)

    # The model request that produced the failing tool call was billed, so it is reflected in the total.
    assert session_usage == snapshot(RunUsage(input_tokens=51, output_tokens=2, requests=1))


def test_code_theme_unset(mocker: MockerFixture, env: TestEnv):
    env.set('OPENAI_API_KEY', 'test')
    mock_run_chat = mocker.patch('pydantic_ai._cli.run_chat')
    cli([])
    mock_run_chat.assert_awaited_once_with(
        True, IsInstance(Agent), IsInstance(Console), 'monokai', 'clai', toolsets=None, show_tool_calls=True
    )


def test_code_theme_light(mocker: MockerFixture, env: TestEnv):
    env.set('OPENAI_API_KEY', 'test')
    mock_run_chat = mocker.patch('pydantic_ai._cli.run_chat')
    cli(['--code-theme=light'])
    mock_run_chat.assert_awaited_once_with(
        True, IsInstance(Agent), IsInstance(Console), 'default', 'clai', toolsets=None, show_tool_calls=True
    )


def test_code_theme_dark(mocker: MockerFixture, env: TestEnv):
    env.set('OPENAI_API_KEY', 'test')
    mock_run_chat = mocker.patch('pydantic_ai._cli.run_chat')
    cli(['--code-theme=dark'])
    mock_run_chat.assert_awaited_once_with(
        True, IsInstance(Agent), IsInstance(Console), 'monokai', 'clai', toolsets=None, show_tool_calls=True
    )


@pytest.mark.parametrize('show_tool_calls', [True, False])
def test_agent_to_cli_sync(mocker: MockerFixture, env: TestEnv, show_tool_calls: bool):
    env.set('OPENAI_API_KEY', 'test')
    mock_run_chat = mocker.patch('pydantic_ai._cli.run_chat')
    cli_agent.to_cli_sync(show_tool_calls=show_tool_calls)
    mock_run_chat.assert_awaited_once_with(
        stream=True,
        agent=IsInstance(Agent),
        console=IsInstance(Console),
        code_theme='monokai',
        prog_name='pydantic-ai',
        deps=None,
        message_history=None,
        model=None,
        model_settings=None,
        usage_limits=None,
        show_tool_calls=show_tool_calls,
    )


@pytest.mark.anyio
@pytest.mark.parametrize('show_tool_calls', [True, False])
async def test_agent_to_cli_async(mocker: MockerFixture, env: TestEnv, show_tool_calls: bool):
    env.set('OPENAI_API_KEY', 'test')
    mock_run_chat = mocker.patch('pydantic_ai._cli.run_chat')
    await cli_agent.to_cli(show_tool_calls=show_tool_calls)
    mock_run_chat.assert_awaited_once_with(
        stream=True,
        agent=IsInstance(Agent),
        console=IsInstance(Console),
        code_theme='monokai',
        prog_name='pydantic-ai',
        deps=None,
        message_history=None,
        model=None,
        model_settings=None,
        usage_limits=None,
        show_tool_calls=show_tool_calls,
    )


@pytest.mark.anyio
async def test_agent_to_cli_with_message_history(mocker: MockerFixture, env: TestEnv):
    env.set('OPENAI_API_KEY', 'test')
    mock_run_chat = mocker.patch('pydantic_ai._cli.run_chat')

    # Create some test message history - cast to the proper base type
    test_messages: list[ModelMessage] = [ModelResponse(parts=[TextPart('Hello!')])]

    await cli_agent.to_cli(message_history=test_messages)
    mock_run_chat.assert_awaited_once_with(
        stream=True,
        agent=IsInstance(Agent),
        console=IsInstance(Console),
        code_theme='monokai',
        prog_name='pydantic-ai',
        deps=None,
        message_history=test_messages,
        model=None,
        model_settings=None,
        usage_limits=None,
        show_tool_calls=True,
    )


def test_agent_to_cli_sync_with_message_history(mocker: MockerFixture, env: TestEnv):
    env.set('OPENAI_API_KEY', 'test')
    mock_run_chat = mocker.patch('pydantic_ai._cli.run_chat')

    # Create some test message history - cast to the proper base type
    test_messages: list[ModelMessage] = [ModelResponse(parts=[TextPart('Hello!')])]

    cli_agent.to_cli_sync(message_history=test_messages)
    mock_run_chat.assert_awaited_once_with(
        stream=True,
        agent=IsInstance(Agent),
        console=IsInstance(Console),
        code_theme='monokai',
        prog_name='pydantic-ai',
        deps=None,
        message_history=test_messages,
        model=None,
        model_settings=None,
        usage_limits=None,
        show_tool_calls=True,
    )


@pytest.mark.parametrize(
    ('model_name', 'expected'),
    [
        ('gpt-5', 'GPT 5'),
        ('gpt-4.1', 'GPT 4.1'),
        ('o1', 'O1'),
        ('o3', 'O3'),
        ('claude-sonnet-4-5', 'Claude Sonnet 4.5'),
        ('claude-haiku-4-5', 'Claude Haiku 4.5'),
        ('gemini-2.5-pro', 'Gemini 2.5 Pro'),
        ('gemini-2.5-flash', 'Gemini 2.5 Flash'),
        ('sonnet-4-5', 'Sonnet 4.5'),
        ('custom-model', 'Custom Model'),
    ],
)
def test_model_label(model_name: str, expected: str):
    """Test Model.label formatting for UI."""
    from pydantic_ai.models.test import TestModel

    model = TestModel(model_name=model_name)
    assert model.label == expected


def test_clai_web_generic_agent(mocker: MockerFixture, env: TestEnv):
    """Test web command without agent creates generic agent."""
    env.set('OPENAI_API_KEY', 'test')
    mock_run_web = mocker.patch('pydantic_ai._cli.web.run_web_command', return_value=0)

    assert cli(['web', '-m', 'openai:gpt-5', '-t', 'web_search'], prog_name='clai') == 0

    mock_run_web.assert_called_once_with(
        agent_path=None,
        host='127.0.0.1',
        port=7932,
        models=['openai:gpt-5'],
        tools=['web_search'],
        instructions=None,
        default_model='openai:gpt-5',
        html_source=None,
        allowed_hosts=[],
    )


def test_clai_web_success(mocker: MockerFixture, create_test_module: Callable[..., None], env: TestEnv):
    env.set('OPENAI_API_KEY', 'test')

    mock_run_web = mocker.patch('pydantic_ai._cli.web.run_web_command', return_value=0)

    test_agent = Agent(TestModel(custom_output_text='test'))
    create_test_module(custom_agent=test_agent)

    assert cli(['web', '--agent', 'test_module:custom_agent'], prog_name='clai') == 0

    mock_run_web.assert_called_once_with(
        agent_path='test_module:custom_agent',
        host='127.0.0.1',
        port=7932,
        models=[],
        tools=[],
        instructions=None,
        default_model='openai:gpt-5',
        html_source=None,
        allowed_hosts=[],
    )


def test_clai_web_with_models(mocker: MockerFixture, create_test_module: Callable[..., None], env: TestEnv):
    """Test web command with multiple -m flags."""
    env.set('OPENAI_API_KEY', 'test')

    mock_run_web = mocker.patch('pydantic_ai._cli.web.run_web_command', return_value=0)

    test_agent = Agent(TestModel(custom_output_text='test'))
    create_test_module(custom_agent=test_agent)

    assert (
        cli(
            [
                'web',
                '--agent',
                'test_module:custom_agent',
                '-m',
                'openai:gpt-5',
                '-m',
                'anthropic:claude-sonnet-4-6',
            ],
            prog_name='clai',
        )
        == 0
    )

    mock_run_web.assert_called_once_with(
        agent_path='test_module:custom_agent',
        host='127.0.0.1',
        port=7932,
        models=['openai:gpt-5', 'anthropic:claude-sonnet-4-6'],
        tools=[],
        instructions=None,
        default_model='openai:gpt-5',
        html_source=None,
        allowed_hosts=[],
    )


def test_clai_web_with_tools(mocker: MockerFixture, create_test_module: Callable[..., None], env: TestEnv):
    """Test web command with multiple -t flags."""
    env.set('OPENAI_API_KEY', 'test')

    mock_run_web = mocker.patch('pydantic_ai._cli.web.run_web_command', return_value=0)

    test_agent = Agent(TestModel(custom_output_text='test'))
    create_test_module(custom_agent=test_agent)

    assert (
        cli(
            ['web', '--agent', 'test_module:custom_agent', '-t', 'web_search', '-t', 'code_execution'], prog_name='clai'
        )
        == 0
    )

    mock_run_web.assert_called_once_with(
        agent_path='test_module:custom_agent',
        host='127.0.0.1',
        port=7932,
        models=[],
        tools=['web_search', 'code_execution'],
        instructions=None,
        default_model='openai:gpt-5',
        html_source=None,
        allowed_hosts=[],
    )


def test_clai_web_generic_with_instructions(mocker: MockerFixture, env: TestEnv):
    """Test generic agent with custom instructions."""
    env.set('OPENAI_API_KEY', 'test')

    mock_run_web = mocker.patch('pydantic_ai._cli.web.run_web_command', return_value=0)

    assert cli(['web', '-m', 'openai:gpt-5', '-i', 'You are a helpful coding assistant'], prog_name='clai') == 0

    mock_run_web.assert_called_once_with(
        agent_path=None,
        host='127.0.0.1',
        port=7932,
        models=['openai:gpt-5'],
        tools=[],
        instructions='You are a helpful coding assistant',
        default_model='openai:gpt-5',
        html_source=None,
        allowed_hosts=[],
    )


def test_clai_web_with_custom_port(mocker: MockerFixture, create_test_module: Callable[..., None], env: TestEnv):
    """Test web command with custom host/port."""
    env.set('OPENAI_API_KEY', 'test')

    mock_run_web = mocker.patch('pydantic_ai._cli.web.run_web_command', return_value=0)

    test_agent = Agent(TestModel(custom_output_text='test'))
    create_test_module(custom_agent=test_agent)

    assert (
        cli(['web', '--agent', 'test_module:custom_agent', '--host', '0.0.0.0', '--port', '7932'], prog_name='clai')
        == 0
    )

    mock_run_web.assert_called_once_with(
        agent_path='test_module:custom_agent',
        host='0.0.0.0',
        port=7932,
        models=[],
        tools=[],
        instructions=None,
        default_model='openai:gpt-5',
        html_source=None,
        allowed_hosts=[],
    )


def test_run_web_command_agent_with_model(
    mocker: MockerFixture, create_test_module: Callable[..., None], capfd: CaptureFixture[str]
):
    """Test run_web_command uses agent's model when no -m flag provided."""

    mock_uvicorn_run = mocker.patch('uvicorn.run')
    mocker.patch('pydantic_ai._cli.web.create_web_app')

    test_agent = Agent(TestModel(custom_output_text='test'))
    create_test_module(custom_agent=test_agent)

    result = run_web_command(agent_path='test_module:custom_agent')

    assert result == 0
    mock_uvicorn_run.assert_called_once()


def test_run_web_command_generic_agent_no_model(mocker: MockerFixture, capfd: CaptureFixture[str]):
    """Test run_web_command uses default model when no agent and no model provided."""
    mock_uvicorn_run = mocker.patch('uvicorn.run')
    mock_create_app = mocker.patch('pydantic_ai._cli.web.create_web_app')

    result = run_web_command()

    assert result == 0
    mock_uvicorn_run.assert_called_once()
    # Verify default model was passed
    call_kwargs = mock_create_app.call_args.kwargs
    assert call_kwargs['models'] == ['openai:gpt-5']


def test_run_web_command_generic_agent_with_instructions(mocker: MockerFixture, capfd: CaptureFixture[str]):
    """Test run_web_command passes instructions to create_web_app for generic agent."""

    mock_uvicorn_run = mocker.patch('uvicorn.run')
    mock_create_app = mocker.patch('pydantic_ai._cli.web.create_web_app')

    result = run_web_command(models=['test'], instructions='You are a helpful assistant')

    assert result == 0
    mock_uvicorn_run.assert_called_once()

    # Verify instructions were passed to create_web_app (not to Agent constructor)
    call_kwargs = mock_create_app.call_args.kwargs
    assert call_kwargs['instructions'] == 'You are a helpful assistant'


def test_run_web_command_agent_with_instructions(
    mocker: MockerFixture, create_test_module: Callable[..., None], capfd: CaptureFixture[str]
):
    """Test run_web_command passes instructions to create_web_app when agent is provided."""

    mock_uvicorn_run = mocker.patch('uvicorn.run')
    mock_create_app = mocker.patch('pydantic_ai._cli.web.create_web_app')

    test_agent = Agent(TestModel(custom_output_text='test'))
    create_test_module(custom_agent=test_agent)

    result = run_web_command(agent_path='test_module:custom_agent', instructions='Always respond in Spanish')

    assert result == 0
    mock_uvicorn_run.assert_called_once()

    # Verify instructions were passed to create_web_app
    call_kwargs = mock_create_app.call_args.kwargs
    assert call_kwargs['instructions'] == 'Always respond in Spanish'


def test_run_web_command_agent_load_failure(capfd: CaptureFixture[str]):
    """Test run_web_command returns error when agent path is invalid."""

    result = run_web_command(agent_path='nonexistent_module:agent')

    assert result == 1
    output = capfd.readouterr().out
    assert 'Could not load agent' in output


def test_run_web_command_unknown_tool(mocker: MockerFixture, capfd: CaptureFixture[str]):
    """Test run_web_command warns about unknown tool IDs."""

    mock_uvicorn_run = mocker.patch('uvicorn.run')
    mocker.patch('pydantic_ai._cli.web.create_web_app')

    result = run_web_command(models=['test'], tools=['unknown_tool_xyz'])

    assert result == 0
    mock_uvicorn_run.assert_called_once()
    output = capfd.readouterr().out
    assert 'Unknown tool "unknown_tool_xyz"' in output


def test_run_web_command_memory_tool(mocker: MockerFixture, capfd: CaptureFixture[str]):
    """Test run_web_command warns about memory tool requiring agent configuration."""

    mock_uvicorn_run = mocker.patch('uvicorn.run')
    mocker.patch('pydantic_ai._cli.web.create_web_app')

    result = run_web_command(models=['test'], tools=['memory'])

    assert result == 0
    mock_uvicorn_run.assert_called_once()
    output = capfd.readouterr().out
    assert '"memory" requires configuration and cannot be enabled via CLI' in output


def test_run_web_command_agent_native_tools_not_duplicated(
    mocker: MockerFixture, create_test_module: Callable[..., None], capfd: CaptureFixture[str]
):
    """Test run_web_command only passes CLI-provided tools, not agent's native tools."""
    from pydantic_ai.native_tools import WebSearchTool

    mock_uvicorn_run = mocker.patch('uvicorn.run')
    mock_create_app = mocker.patch('pydantic_ai._cli.web.create_web_app')

    # Create agent with web_search tool already configured
    test_agent = Agent(TestModel(custom_output_text='test'), capabilities=[NativeTool(WebSearchTool())])
    create_test_module(custom_agent=test_agent)

    # Add code_execution via CLI
    result = run_web_command(agent_path='test_module:custom_agent', tools=['code_execution'])

    assert result == 0
    mock_uvicorn_run.assert_called_once()

    # Verify only CLI-provided tools are passed (agent's tools are handled by create_web_app)
    call_kwargs = mock_create_app.call_args.kwargs
    native_tools = call_kwargs.get('native_tools', [])
    tool_kinds = {t.kind for t in native_tools}
    # web_search is on the agent, so it's NOT passed here (it's handled internally)
    assert 'web_search' not in tool_kinds
    # code_execution was provided via CLI, so it IS passed
    assert 'code_execution' in tool_kinds


def test_run_web_command_cli_models_passed_to_create_web_app(
    mocker: MockerFixture, create_test_module: Callable[..., None]
):
    """Test that CLI models are passed directly to create_web_app (agent model merging happens there)."""
    mock_uvicorn_run = mocker.patch('uvicorn.run')
    mock_create_app = mocker.patch('pydantic_ai._cli.web.create_web_app')

    test_agent = Agent(TestModel(custom_output_text='test'))
    create_test_module(custom_agent=test_agent)

    result = run_web_command(
        agent_path='test_module:custom_agent', models=['openai:gpt-5', 'anthropic:claude-sonnet-4-6']
    )

    assert result == 0
    mock_uvicorn_run.assert_called_once()

    call_kwargs = mock_create_app.call_args.kwargs
    # CLI models passed as list; agent model merging/deduplication happens in create_web_app
    assert call_kwargs.get('models') == ['openai:gpt-5', 'anthropic:claude-sonnet-4-6']


def test_agent_to_cli_sync_with_args(mocker: MockerFixture, env: TestEnv):
    env.set('OPENAI_API_KEY', 'test')
    mock_run_chat = mocker.patch('pydantic_ai._cli.run_chat')

    model_settings = ModelSettings(temperature=0.5)
    usage_limits = UsageLimits(request_limit=10)

    cli_agent.to_cli_sync(model_settings=model_settings, usage_limits=usage_limits)

    mock_run_chat.assert_awaited_once_with(
        stream=True,
        agent=IsInstance(Agent),
        console=IsInstance(Console),
        code_theme='monokai',
        prog_name='pydantic-ai',
        deps=None,
        message_history=None,
        model=None,
        model_settings=model_settings,
        usage_limits=usage_limits,
        show_tool_calls=True,
    )


def test_agent_to_cli_sync_with_model(mocker: MockerFixture, env: TestEnv):
    env.set('OPENAI_API_KEY', 'test')
    mock_run_chat = mocker.patch('pydantic_ai._cli.run_chat')

    cli_agent.to_cli_sync(model='test')

    mock_run_chat.assert_awaited_once_with(
        stream=True,
        agent=IsInstance(Agent),
        console=IsInstance(Console),
        code_theme='monokai',
        prog_name='pydantic-ai',
        deps=None,
        message_history=None,
        model='test',
        model_settings=None,
        usage_limits=None,
        show_tool_calls=True,
    )


@pytest.mark.anyio
async def test_agent_to_cli_async_with_args(mocker: MockerFixture, env: TestEnv):
    env.set('OPENAI_API_KEY', 'test')
    mock_run_chat = mocker.patch('pydantic_ai._cli.run_chat')

    model_settings = ModelSettings(temperature=0.5)
    usage_limits = UsageLimits(request_limit=10)

    await cli_agent.to_cli(model_settings=model_settings, usage_limits=usage_limits)

    mock_run_chat.assert_awaited_once_with(
        stream=True,
        agent=IsInstance(Agent),
        console=IsInstance(Console),
        code_theme='monokai',
        prog_name='pydantic-ai',
        deps=None,
        message_history=None,
        model=None,
        model_settings=model_settings,
        usage_limits=usage_limits,
        show_tool_calls=True,
    )


@pytest.mark.anyio
async def test_agent_to_cli_async_with_model(mocker: MockerFixture, env: TestEnv):
    env.set('OPENAI_API_KEY', 'test')
    mock_run_chat = mocker.patch('pydantic_ai._cli.run_chat')

    await cli_agent.to_cli(model='test')

    mock_run_chat.assert_awaited_once_with(
        stream=True,
        agent=IsInstance(Agent),
        console=IsInstance(Console),
        code_theme='monokai',
        prog_name='pydantic-ai',
        deps=None,
        message_history=None,
        model='test',
        model_settings=None,
        usage_limits=None,
        show_tool_calls=True,
    )


@pytest.mark.anyio
async def test_ask_agent_non_stream_forwards_run_kwargs(mocker: MockerFixture):
    from pydantic_ai._cli import ask_agent

    result = mocker.Mock()
    result.output = 'hello'
    result.all_messages.return_value = []

    agent = mocker.Mock()
    agent.run = mocker.AsyncMock(return_value=result)

    model_settings = ModelSettings(temperature=0)
    usage_limits = UsageLimits(request_limit=5)

    messages = await ask_agent(
        agent,
        'Hello',
        stream=False,
        console=Console(file=StringIO()),
        code_theme='monokai',
        model='test',
        model_settings=model_settings,
        usage_limits=usage_limits,
    )

    agent.run.assert_awaited_once_with(
        'Hello',
        message_history=None,
        deps=None,
        model='test',
        model_settings=model_settings,
        usage_limits=usage_limits,
        toolsets=None,
        usage=IsInstance(RunUsage),
    )
    assert messages == []


def test_clai_web_with_html_source(mocker: MockerFixture, env: TestEnv):
    """Test web command with --html-source flag."""
    env.set('OPENAI_API_KEY', 'test')
    mock_run_web = mocker.patch('pydantic_ai._cli.web.run_web_command', return_value=0)

    custom_url = 'https://internal.company.com/pydantic-ai-ui/index.html'
    assert cli(['web', '-m', 'openai:gpt-5', '--html-source', custom_url], prog_name='clai') == 0

    mock_run_web.assert_called_once_with(
        agent_path=None,
        host='127.0.0.1',
        port=7932,
        models=['openai:gpt-5'],
        tools=[],
        instructions=None,
        default_model='openai:gpt-5',
        html_source=custom_url,
        allowed_hosts=[],
    )


def test_clai_web_with_allowed_hosts(mocker: MockerFixture, env: TestEnv):
    """`--allowed-host` is repeatable, for serving the UI under a name behind a proxy or a tunnel."""
    env.set('OPENAI_API_KEY', 'test')
    mock_run_web = mocker.patch('pydantic_ai._cli.web.run_web_command', return_value=0)

    args = ['web', '-m', 'openai:gpt-5', '--allowed-host', 'ui.example.com', '--allowed-host', '*.corp.example']
    assert cli(args, prog_name='clai') == 0

    mock_run_web.assert_called_once_with(
        agent_path=None,
        host='127.0.0.1',
        port=7932,
        models=['openai:gpt-5'],
        tools=[],
        instructions=None,
        default_model='openai:gpt-5',
        html_source=None,
        allowed_hosts=['ui.example.com', '*.corp.example'],
    )


def test_clai_web_answers_to_the_host_it_binds_to(mocker: MockerFixture, env: TestEnv):
    """`--host <name>` implies answering to that name, so the URL the CLI prints actually works.

    Without this the CLI contradicts itself: it prints `Open your browser at: http://devbox.example:7932`
    and then rejects that exact `Host` with a `421`.
    """
    env.set('OPENAI_API_KEY', 'test')
    mock_uvicorn = mocker.patch('uvicorn.run')
    mock_create = mocker.patch('pydantic_ai._cli.web.create_web_app')

    assert cli(['web', '-m', 'openai:gpt-5', '--host', 'devbox.example'], prog_name='clai') == 0

    assert mock_create.call_args.kwargs['allowed_hosts'] == ['devbox.example']
    assert mock_uvicorn.call_args.kwargs['host'] == 'devbox.example'


@pytest.mark.parametrize('mode', ['one-shot', 'interactive', 'non-streaming'])
def test_cli_no_tool_calls(
    mode: str,
    capfd: CaptureFixture[str],
    mocker: MockerFixture,
    create_test_module: Callable[..., None],
):
    agent = Agent(TestModel(custom_output_text='Finished.'))
    calls: list[str] = []

    @agent.tool_plain
    def run_code(token: Literal['private-preview-marker\nline 2\nline 3\nline 4\nline 5\nline 6']) -> str:
        calls.append(token)
        print('Saved chart.')
        return 'saved'

    create_test_module(agent=agent)
    args = ['--agent', 'test_module:agent', '--no-tool-calls']
    with create_pipe_input() as inp:
        inp.send_text('go\n/exit\n')
        mocker.patch('pydantic_ai._cli.PromptSession', return_value=PromptSession[Any](input=inp, output=DummyOutput()))
        if mode != 'interactive':
            args.append('go')
        if mode == 'non-streaming':
            args.append('--no-stream')
        assert cli(args) == 0

    output = capfd.readouterr().out
    assert calls == ['private-preview-marker\nline 2\nline 3\nline 4\nline 5\nline 6']
    assert 'private-preview-marker' not in output
    assert 'Saved chart.' in output
    assert 'Finished.' in output
    assert 'Calling tool' not in output
    assert 'Called tool' not in output


@pytest.mark.anyio
@pytest.mark.parametrize('width', [40, 1000])
@pytest.mark.parametrize(
    ('arguments', 'expected'),
    [
        pytest.param(
            {'code': "print(`[red]hello[/red]`)\nprint('  spaced  ')"},
            snapshot("""\
▌ Called tool run_code(code=<2 lines>).
▌   code:
▌     print(`[red]hello[/red]`)
▌     print('  spaced  ')\
"""),
            id='literal-code',
        ),
        pytest.param(
            {'code': 'x' * 1000},
            snapshot(
                "▌ Called tool run_code(code='xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'…)."
            ),
            id='long-code',
        ),
        pytest.param(
            {'bad\nkey': 'hello'}, snapshot("▌ Called tool run_code('bad\\nkey'='hello')."), id='multiline-key'
        ),
        pytest.param(
            {'first': 'x' * 1000, 'second': 'y' * 1000, 'third': 'useful'},
            snapshot(
                "▌ Called tool run_code(first='xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'…, second='yyyyyyyyyyy…."
            ),
            id='many-arguments',
        ),
        pytest.param(
            {'code': 'import json\n\nvalues = [1, 2, 3]\nfor value in values:\n    print(value)'},
            snapshot("""\
▌ Called tool run_code(code=<5 lines>).
▌   code:
▌     import json
▌
▌     values = [1, 2, 3]
▌     for value in values:
▌         print(value)\
"""),
            id='5-line-script',
        ),
        pytest.param(
            {'code': 'import json\n\nvalues = [1, 2, 3]\nfor value in values:\n    print(value)\nprint("done")'},
            snapshot("""\
▌ Called tool run_code(code=<6 lines>).
▌   code:
▌     import json
▌
▌     values = [1, 2, 3]
▌     for value in values:
▌         print(value)
▌     … 1 more line\
"""),
            id='6-line-script',
        ),
        pytest.param(
            {'code': 'import json\n' + 'print("data")\n' * 21},
            snapshot("""\
▌ Called tool run_code(code=<22 lines>).
▌   code:
▌     import json
▌     print("data")
▌     print("data")
▌     print("data")
▌     print("data")
▌     … 17 more lines\
"""),
            id='large-script',
        ),
        pytest.param(
            {'first': 'one\ntwo', 'second': 'three\nfour'},
            snapshot("""\
▌ Called tool run_code(first=<2 lines>, second=<2 lines>).
▌   first:
▌     one
▌     two
▌   second:
▌     three
▌     four\
"""),
            id='multiple-multiline-arguments',
        ),
        pytest.param(
            {'code': 'x' * 300 + '\nlast line'},
            snapshot("""\
▌ Called tool run_code(code=<2 lines>).
▌   code:
▌     xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx…
▌     last line\
"""),
            id='wide-source-line',
        ),
        pytest.param(
            {'code': "if True:\n\tprint('\x1b[2J')"},
            snapshot("""\
▌ Called tool run_code(code=<2 lines>).
▌   code:
▌     if True:
▌         print('\\x1b[2J')\
"""),
            id='tabs-and-terminal-escapes',
        ),
        pytest.param(
            {'code': 'e\u0301' * 100 + '\nend'},
            snapshot("""\
▌ Called tool run_code(code=<2 lines>).
▌   code:
▌     éééééééééééééééééééééééééééééééééééééééééééééééééééééééééééé…
▌     end\
"""),
            id='combining-characters',
        ),
        pytest.param(
            {'code': 'print(`[red]hello[/red]`)'},
            snapshot("▌ Called tool run_code(code='print(`[red]hello[/red]`)')."),
            id='literal-short-code',
        ),
        pytest.param(
            {'code': 'print(1)\r\n'},
            snapshot("""\
▌ Called tool run_code(code=<1 line>).
▌   code:
▌     print(1)\
"""),
            id='one-line-with-ending',
        ),
        pytest.param(
            {'data': [1, 2, 3], 'options': {'private': 'hidden'}, 'limit': 5},
            snapshot("▌ Called tool run_code(data=[1, 2, 3], options={'private': 'hidden'}, limit=5)."),
            id='collections',
        ),
        pytest.param(
            {'first': 'a' * 60, 'second': 'b' * 60, 'third': 'c' * 60},
            snapshot(
                "▌ Called tool run_code(first='aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', second='bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb…."
            ),
            id='bounded-many-short-arguments',
        ),
    ],
)
async def test_tool_argument_preview(
    arguments: dict[str, JsonValue], expected: str, width: int, live_frames: list[str]
):
    async def code_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            yield 'Done.'
        else:
            yield {0: DeltaToolCall(name='run_code', json_args=json.dumps(arguments), tool_call_id='call_1')}

    agent = Agent(FunctionModel(stream_function=code_stream))
    executed: list[dict[str, JsonValue]] = []

    @agent.tool_plain
    def run_code(**kwargs: JsonValue) -> str:
        executed.append(kwargs)
        return 'ok'

    output = StringIO()
    await ask_agent(agent, 'go', True, Console(file=output, width=width), 'monokai')
    lines = [line.rstrip() for line in output.getvalue().splitlines()]
    assert executed == [arguments]
    assert lines[0] == ''
    assert lines[-2:] == ['', 'Done.']
    notice = '\n'.join(lines[1:-2])
    if width == 1000:
        assert notice == expected
        assert all(cell_len(line) <= 135 for line in lines[1:-2])
    else:
        assert lines[1].startswith('▌ Called tool run_code(')
        assert all(cell_len(line) <= width for line in lines[1:-2])
    assert any('Calling tool run_code(' in frame for frame in live_frames)


@pytest.mark.anyio
@pytest.mark.parametrize('show_tool_calls', [True, False])
async def test_streaming_preserves_markdown_references(show_tool_calls: bool):
    async def reference_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            yield 'Confirmed.\n\n[docs]: https://example.com'
        else:
            yield 'Checking [the docs][docs].'
            yield {0: DeltaToolCall(name='lookup', json_args='{}', tool_call_id='call_1')}

    agent = Agent(FunctionModel(stream_function=reference_stream))

    @agent.tool_plain
    def lookup() -> str:
        return 'ok'

    output = StringIO()
    await ask_agent(agent, 'go', True, Console(file=output, width=80), 'monokai', show_tool_calls=show_tool_calls)
    rendered = output.getvalue()
    assert 'Checking the docs.' in rendered
    assert '[the docs][docs]' not in rendered
    assert 'Confirmed.' in rendered


@pytest.mark.anyio
@pytest.mark.parametrize('fail', [False, True])
async def test_hidden_tool_calls_keep_working_indicator(live_frames: list[str], fail: bool):
    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            yield 'Done.'
        else:
            yield '\n\n'.join(f'Checking item {index}.' for index in range(10))
            yield {0: DeltaToolCall(name='slow_tool', json_args='{}', tool_call_id='call_1')}

    agent = Agent(FunctionModel(stream_function=stream))
    during_tool: list[str] = []

    @agent.tool_plain
    async def slow_tool() -> str:
        during_tool.append(live_frames[-1] if live_frames else '')
        if fail:
            raise RuntimeError('tool failed')
        return 'ok'

    output = StringIO()
    with pytest.raises(RuntimeError, match='tool failed') if fail else nullcontext():
        await ask_agent(agent, 'go', True, Console(file=output, width=80, height=5), 'monokai', show_tool_calls=False)
    assert len(during_tool[0].splitlines()) == 5
    assert 'Working on it' in during_tool[0]
    assert 'slow_tool' not in during_tool[0]
    assert 'Working on it' not in output.getvalue()
    assert ('Done.' in output.getvalue()) is not fail
