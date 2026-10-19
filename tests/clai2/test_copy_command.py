"""`/copy`: the Markdown source of the latest responses, on the clipboard through `copy_text`."""

import io
from collections.abc import AsyncIterator
from pathlib import Path
from typing import IO

import pytest
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaThinkingCalls, DeltaThinkingPart, FunctionModel
from pydantic_clai2.config.project_settings import ProjectSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.runtime import session_runtime
from pydantic_clai2.runtime.session_runtime import create_shell
from pydantic_clai2.ui.copy_response import copy_command, response_texts
from pydantic_clai2.ui.prompt.live_prompt import LivePrompt

HISTORY: list[ModelMessage] = [
    ModelRequest(parts=[UserPromptPart('first')]),
    ModelResponse(parts=[TextPart('# Older\n\n- one')]),
    ModelRequest(parts=[UserPromptPart('second')]),
    ModelResponse(
        parts=[
            ThinkingPart('private reasoning'),
            TextPart('Let me **check**.'),
            ToolCallPart('read', {'path': 'a.py'}, tool_call_id='1'),
        ]
    ),
    ModelRequest(parts=[ToolReturnPart('read', 'contents', tool_call_id='1')]),
    ModelResponse(parts=[ThinkingPart('more reasoning'), TextPart('Done: `a.py` is fine.')]),
    ModelRequest(parts=[ToolReturnPart('final_result', 'ok', tool_call_id='2')]),
    ModelResponse(parts=[ToolCallPart('read', {'path': 'b.py'}, tool_call_id='3'), TextPart('  \n')]),
]


def copied(args: list[str], messages: list[ModelMessage] = HISTORY) -> tuple[str, list[str]]:
    clipboard: list[str] = []
    return copy_command(args, messages=messages, copy=clipboard.append), clipboard


def test_copies_the_latest_response_with_text_without_thinking_or_tool_calls() -> None:
    assert response_texts(HISTORY) == ['Done: `a.py` is fine.', 'Let me **check**.', '# Older\n\n- one']
    assert copied([]) == ('Copied the last response (21 characters).', ['Done: `a.py` is fine.'])


def test_copies_the_nth_latest_response() -> None:
    assert copied(['3']) == ('Copied response 3 from the end (14 characters).', ['# Older\n\n- one'])
    assert copied(['4']) == ('There are only 3 responses to copy.', [])
    assert copied(['2'], HISTORY[:2]) == ('There is only 1 response to copy.', [])
    assert copied([], [ModelResponse(parts=[TextPart('x')])]) == ('Copied the last response (1 character).', ['x'])


def test_empty_conversation_copies_nothing() -> None:
    assert copied([], []) == ('No response to copy yet.', [])
    assert copied([], [ModelRequest(parts=[UserPromptPart('hi')])]) == ('No response to copy yet.', [])


@pytest.mark.parametrize('args', [['0'], ['-1'], ['two'], ['1', '2']])
def test_rejects_anything_but_a_positive_count(args: list[str]) -> None:
    with pytest.raises(ValueError, match=r'Usage: /copy \[N\]'):
        copied(args)


async def test_shell_copies_the_last_completed_response_even_mid_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clipboard: list[tuple[str, IO[str]]] = []

    def copy_text(text: str, *, output: IO[str]) -> None:
        clipboard.append((text, output))

    monkeypatch.setattr(session_runtime, 'copy_text', copy_text)
    replies: list[str] = []

    async def model(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaThinkingCalls]:
        if len(messages) == 1:
            yield {0: DeltaThinkingPart(content='hmm')}
            yield '## Answer\n\nIt is *42*.'
            return
        replies.append(await shell.commands.execute_async('/copy'))
        yield 'Second answer'

    console = Console(file=io.StringIO())
    shell = create_shell(
        Agent(FunctionModel(stream_function=model)),
        deps=None,
        plugins=(),
        usage_limits=None,
        console=console,
        settings=None,
        store=SettingsStore(tmp_path / 'config.db'),
        builtin_plugins=(),
        project=ProjectSettings(),
        headless=True,
    )
    assert '/copy: Copy the last response' in await shell.commands.execute_async('/help')
    assert shell.commands.runs_during_turn('/copy')
    assert shell.commands.runs_during_turn('/copy 2')
    assert await shell.commands.execute_async('/copy') == 'No response to copy yet.'

    await shell.session.prompt('first')
    await shell.session.prompt('second')
    # Mid-turn, the running turn's response is not complete yet, so the first turn's is copied.
    assert replies == ['Copied the last response (22 characters).']
    assert await shell.commands.execute_async('/copy') == 'Copied the last response (13 characters).'
    # Headless, with no editor, the clipboard escape goes to the console's own output.
    terminal = console.file
    assert clipboard == [('## Answer\n\nIt is *42*.', terminal), ('Second answer', terminal)]

    # The open editor takes the console's output into its transcript; the escape still goes to the terminal.
    shell.editor = LivePrompt(
        history=InMemoryHistory(),
        console=console,
        commands=shell.commands,
        images=shell.images,
        interrupts=shell.interrupts,
        toolbar=lambda: [],
    )
    console.file = shell.editor.output
    clipboard.clear()
    assert await shell.commands.execute_async('/copy 2') == 'Copied response 2 from the end (22 characters).'
    assert clipboard == [('## Answer\n\nIt is *42*.', terminal)]
