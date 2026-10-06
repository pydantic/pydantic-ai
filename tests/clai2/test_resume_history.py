"""`/resume` shows the restored conversation in the live panel, the way its turns streamed."""

import io
from pathlib import Path

import anyio
import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console
from rich.text import Text

from pydantic_ai import Agent
from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    NativeToolCallPart,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.step_persistence.conversations import SqliteConversationStore
from pydantic_clai2 import chat
from pydantic_clai2.config import Settings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.runtime._session import Session
from pydantic_clai2.ui.prompt.prompt_surface import LEAVE, PromptSurface
from pydantic_clai2.ui.rendering._rendering import StreamRenderer
from pydantic_clai2.ui.rendering.history import render_history, turn_starts


def turn(index: int) -> list[ModelMessage]:
    run = f'run-{index}'
    return [
        ModelRequest(parts=[UserPromptPart(f'prompt {index}')], run_id=run),
        ModelResponse(parts=[TextPart(f'answer {index}')], run_id=run),
    ]


async def rendered(messages: list[ModelMessage], *, turns: int = 10, thinking: bool = True) -> str:
    output = io.StringIO()
    console = Console(file=output, width=80)
    renderer = StreamRenderer(console, stop_loading=lambda: None, show_thinking=thinking, smooth=False)
    await render_history(messages, console=console, renderer=renderer, turns=turns)
    return Text.from_ansi(output.getvalue()).plain


def test_turns_start_at_prompts_not_tool_results_steers_or_context() -> None:
    messages: list[ModelMessage] = [
        ModelRequest(parts=[SystemPromptPart('summary of earlier work')]),
        ModelRequest(parts=[UserPromptPart('!ls output')]),
        ModelRequest(parts=[UserPromptPart('first')], run_id='a'),
        ModelResponse(parts=[ToolCallPart('shell', {'command': 'ls'}, tool_call_id='t1')], run_id='a'),
        ModelRequest(parts=[ToolReturnPart('shell', 'ok', tool_call_id='t1')], run_id='a'),
        ModelRequest(parts=[UserPromptPart('steer')], run_id='a'),
        ModelRequest(parts=[RetryPromptPart('again', tool_name='shell'), UserPromptPart('fix')], run_id='b'),
        ModelRequest(parts=[UserPromptPart('second')], run_id='c'),
    ]
    assert turn_starts(messages) == [1, 2, 7]


async def test_only_the_latest_turns_are_shown() -> None:
    messages = [message for index in range(4) for message in turn(index)]
    text = await rendered(messages, turns=2)
    assert text.startswith('2 earlier turns not shown; the model still has them.')
    assert 'prompt 1' not in text and 'answer 1' not in text
    assert text.index('> prompt 2') < text.index('answer 2') < text.index('> prompt 3') < text.index('answer 3')
    assert (await rendered(messages, turns=3)).startswith('1 earlier turn not shown;')
    assert 'not shown' not in await rendered(messages, turns=4)


async def test_steers_are_not_echoed_but_their_tool_results_are_replayed() -> None:
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('start')], run_id='a'),
        ModelResponse(parts=[ToolCallPart('grep', {'pattern': 'x', 'path': '.'}, tool_call_id='g1')], run_id='a'),
        ModelRequest(
            parts=[ToolReturnPart('grep', 'a.py:1:x', tool_call_id='g1'), UserPromptPart('steer now')], run_id='a'
        ),
        ModelResponse(parts=[TextPart('steered answer')], run_id='a'),
    ]
    output = io.StringIO()
    console = Console(file=output, width=80)
    renderer = StreamRenderer(console, stop_loading=lambda: None, show_tool_output=True, smooth=False)
    await render_history(messages, console=console, renderer=renderer)
    text = output.getvalue()
    assert '> start' in text and 'steer now' not in text
    assert text.index("● grep 'x' in '.'") < text.index('a.py:1:x') < text.index('steered answer')


async def test_prompts_answers_thinking_and_tool_calls_render_like_a_live_turn() -> None:
    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[SystemPromptPart('system'), UserPromptPart(['look', BinaryContent(b'x', media_type='image/png')])]
        ),
        ModelResponse(
            parts=[
                ThinkingPart('pondering'),
                TextPart('**checking**'),
                NativeToolCallPart(tool_name='web_search', tool_call_id='w1'),
                ToolCallPart('grep', {'pattern': 'needle', 'path': 'src'}, tool_call_id='g1'),
                ToolCallPart('lookup', {'key': 'value'}, tool_call_id='l1'),
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart('grep', 'src/a.py:1:needle', tool_call_id='g1'),
                RetryPromptPart('bad key', tool_name='lookup', tool_call_id='l1'),
                RetryPromptPart('output invalid'),
            ]
        ),
        ModelResponse(parts=[TextPart('found it')]),
    ]
    text = await rendered(messages)
    assert '> look\n[attachment]' in text
    assert text.index('Thinking pondering') < text.index('checking') < text.index("● grep 'needle' in 'src'")
    assert text.index("● grep 'needle' in 'src'") < text.index('● lookup key="value"') < text.index('found it')
    assert 'system' not in text and 'web_search' not in text and 'output invalid' not in text
    assert 'pondering' not in await rendered(messages, thinking=False)


def saved_prior(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Session[None, str]:
    monkeypatch.chdir(tmp_path)
    return Session(
        Agent(TestModel(call_tools=[], custom_output_text='saved answer')),
        deps=None,
        conversations=SqliteConversationStore(database=tmp_path / 'sessions.db'),
        workspace=tmp_path,
    )


async def test_resume_replaces_the_live_panel_with_the_restored_conversation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prior = saved_prior(tmp_path, monkeypatch)
    await prior.prompt('saved turn')
    surfaces: list[PromptSurface] = []

    class Surface(PromptSurface):
        def paint(self, rows: tuple[str, ...]) -> None:
            surfaces.append(self)
            super().paint(rows)

    monkeypatch.setattr('pydantic_clai2.ui.prompt.live_prompt.PromptSurface', Surface)
    output = io.StringIO()
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()), anyio.fail_after(10):
        pipe.send_text(f'hello\r/resume {prior.summary.id}\r/exit\r')
        await chat(
            Agent(TestModel(call_tools=[], custom_output_text='fresh answer')),
            deps=None,
            console=Console(file=output, force_terminal=True, width=80, height=24),
            store=SettingsStore(tmp_path / 'settings.db'),
            settings=Settings(model=None, session_namer=False),
        )
    text = '\n'.join(Text.from_ansi(row).plain for row in surfaces[0].transcript.frame(width=200, height=200).rows)
    assert '> hello' not in text and 'fresh answer' not in text
    assert text.count('/new starts a session') == 1
    assert text.index('> saved turn') < text.index('saved answer') < text.index('Resumed')
    # Like `/clear`, the exit printout forgets the replaced conversation and keeps the resumed one.
    printed = Text.from_ansi(output.getvalue().rsplit(LEAVE, 1)[1]).plain
    assert '> hello' not in printed
    assert printed.index('> saved turn') < printed.index('Resumed') < printed.index('Goodbye.')


async def test_startup_resume_shows_history_below_the_startup_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prior = saved_prior(tmp_path, monkeypatch)
    await prior.prompt('earlier turn')
    output = io.StringIO()
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        pipe.send_text('/exit\r')
        await chat(
            Agent(TestModel()),
            deps=None,
            console=Console(file=output),
            store=SettingsStore(tmp_path / 'settings.db'),
            settings=Settings(model=None, session_namer=False),
            resume=prior.summary.id,
        )
    text = output.getvalue()
    assert text.index('/new starts a session') < text.index('> earlier turn') < text.index('saved answer')
    assert text.index('saved answer') < text.index('Resumed')


def ping() -> str:
    return 'ok'


def pong() -> str:
    return 'ok'


async def test_resumed_tool_calls_replay_in_the_chosen_tool_call_style(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    prior = Session(
        Agent(TestModel(custom_output_text='saved answer'), tools=[ping, pong]),
        deps=None,
        conversations=SqliteConversationStore(database=tmp_path / 'sessions.db'),
        workspace=tmp_path,
    )
    await prior.prompt('earlier turn')
    output = io.StringIO()
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        pipe.send_text('/exit\r')
        await chat(
            Agent(TestModel()),
            deps=None,
            console=Console(file=output),
            store=SettingsStore(tmp_path / 'settings.db'),
            settings=Settings(model=None, session_namer=False, tool_calls='grouped'),
            resume=prior.summary.id,
        )
    assert '● ping 1, pong 1\n\nsaved answer' in output.getvalue()
