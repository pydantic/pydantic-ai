"""Write diffs reach the terminal through the real tool event stream."""

import io
import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from rich.console import Console
from rich.text import Text

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage, RetryPromptPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai_harness.filesystem import FileChangeRequestEvent, FileSystem
from pydantic_clai2 import Session, StreamRenderer


def write_model(content: str) -> FunctionModel:
    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
        if any(isinstance(part, (ToolReturnPart, RetryPromptPart)) for message in messages for part in message.parts):
            yield 'done'
        else:
            yield {0: DeltaToolCall(name='write_file', json_args=json.dumps({'path': 'file.txt', 'content': content}))}

    return FunctionModel(stream_function=stream)


@pytest.mark.parametrize(
    ('old', 'content'),
    [
        pytest.param(None, 'new\n', id='create'),
        pytest.param('old\n', 'new\n', id='overwrite'),
        pytest.param('', 'new\n', id='overwrite-empty'),
        pytest.param('old\n', '', id='empty-file'),
        pytest.param('same\n', 'same\n', id='unchanged'),
        pytest.param(None, '', id='create-empty'),
        pytest.param('old', 'new', id='no-final-newline'),
        pytest.param('café\n', 'tea\n', id='unicode'),
        pytest.param(None, '\x1b[31mnew\n', id='terminal-controls'),
    ],
)
@pytest.mark.parametrize('show_tool_output', [False, True])
@pytest.mark.parametrize('terminal', [False, True])
async def test_write_file_renders_diff(
    tmp_path: Path, old: str | None, content: str, show_tool_output: bool, terminal: bool
) -> None:
    path = tmp_path / 'file.txt'
    if old is not None:
        path.write_text(old, encoding='utf-8')
    output = io.StringIO()
    renderer = StreamRenderer(
        Console(file=output, width=100, force_terminal=terminal),
        stop_loading=lambda: None,
        show_tool_output=show_tool_output,
    )
    session = Session(
        Agent(write_model(content), capabilities=[FileSystem()]),
        deps=None,
        workspace=tmp_path,
        on_stream_event=renderer.on_stream_event,
    )
    await session.prompt('write the file')
    await renderer.finish()

    assert path.read_text(encoding='utf-8') == content
    rendered = Text.from_ansi(output.getvalue()).plain
    assert rendered.count('● write_file file.txt') == 1
    assert 'Diff truncated.' not in rendered
    if (old or '') == content:
        assert rendered.split() == ['●', 'write_file', 'file.txt', 'done']
    else:
        # Termflow adds a space after the marker; redirected output uses the raw unified diff.
        separator = ' ' if terminal else ''
        for marker, text in (('-', old or ''), ('+', content)):
            for line in text.splitlines():
                safe_line = line.replace('\x1b', r'\x1b')
                assert f'{marker}{separator}{safe_line}' in rendered
        if not terminal and (old and not old.endswith('\n') or content and not content.endswith('\n')):
            assert r'\ No newline at end of file' in rendered


@pytest.mark.parametrize('conflict', [False, True])
async def test_unsuccessful_write_does_not_render_diff(tmp_path: Path, conflict: bool) -> None:
    path = tmp_path / 'file.txt'
    path.write_text('old\n')
    agent = Agent(write_model('new\n'), capabilities=[FileSystem()])

    @agent.on_event(FileChangeRequestEvent)
    async def refuse(ctx: RunContext[None], event: FileChangeRequestEvent) -> None:
        if conflict:
            path.write_text('external\n')
        else:
            event.cancel('not approved')

    output = io.StringIO()
    renderer = StreamRenderer(Console(file=output), stop_loading=lambda: None)
    session = Session(agent, deps=None, workspace=tmp_path, on_stream_event=renderer.on_stream_event)
    await session.prompt('write the file')
    await renderer.finish()

    assert path.read_text() == ('external\n' if conflict else 'old\n')
    assert output.getvalue() == '● write_file file.txt\n\ndone\n\n'


async def test_write_file_shows_truncation_in_compact_mode(tmp_path: Path) -> None:
    content = ''.join(f'line {index}\n' for index in range(2000))
    output = io.StringIO()
    renderer = StreamRenderer(Console(file=output), stop_loading=lambda: None)
    session = Session(
        Agent(write_model(content), capabilities=[FileSystem()]),
        deps=None,
        workspace=tmp_path,
        on_stream_event=renderer.on_stream_event,
    )
    await session.prompt('write the file')
    await renderer.finish()

    assert (tmp_path / 'file.txt').read_text() == content
    assert '+line 0\n' in output.getvalue()
    assert '+line 1999\n' not in output.getvalue()
    assert output.getvalue().count('Diff truncated.') == 1
