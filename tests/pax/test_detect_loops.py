"""The `detect_loops` built-in nudges a model that repeats itself, and changes no request until it does."""

import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import replace
from pathlib import Path

import pytest
from inline_snapshot import snapshot

from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter, ModelRequest, UserPromptPart
from pydantic_ai.models.function import (
    AgentInfo,
    DeltaThinkingCalls,
    DeltaThinkingPart,
    DeltaToolCall,
    DeltaToolCalls,
    FunctionModel,
)
from pydantic_ai.tools import ToolDefinition
from pydantic_ai_harness.detect_loops import LoopNudgePart
from pydantic_ai_harness.step_persistence.conversations import SqliteSessionStore
from pydantic_ai_pax.app import STOCK_PLUGINS, create_stock_agent
from pydantic_ai_pax.cli import headless
from pydantic_ai_pax.config import PluginSettings, Settings
from pydantic_ai_pax.config.project_settings import ProjectSettings
from pydantic_ai_pax.config.settings_store import SettingsStore

Reply = Sequence[str] | Sequence[DeltaToolCall] | Sequence[DeltaThinkingPart]
"""One scripted model response, as the chunks it streams: text, a single tool call's pieces, or thinking."""

CODER = next(plugin for plugin in STOCK_PLUGINS if plugin.id == 'coder')
DETECT_LOOPS = next(plugin for plugin in STOCK_PLUGINS if plugin.id == 'detect_loops')
DISABLED = DETECT_LOOPS.model_copy(update={'enabled': False})

_VOLATILE = frozenset({'timestamp', 'run_id', 'conversation_id'})


def read(path: str, call_id: str) -> Reply:
    return [DeltaToolCall(name='read_file', json_args=json.dumps({'path': path}), tool_call_id=call_id)]


def scripted(
    replies: Sequence[Reply], seen: list[list[ModelMessage]], tools: list[list[ToolDefinition]] | None = None
) -> FunctionModel:
    """A streaming model that sends `replies` in turn, recording the messages, and optionally the tools, of each request."""

    async def respond(
        messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[str | DeltaToolCalls | DeltaThinkingCalls]:
        reply = replies[len(seen)]
        seen.append(list(messages))
        if tools is not None:
            # A tool's `capability_id` names its capability instance, which differs between runs; it isn't sent.
            tools.append([replace(tool, capability_id=None) for tool in info.function_tools])
        for chunk in reply:
            if isinstance(chunk, str):
                yield chunk
            elif isinstance(chunk, DeltaToolCall):
                # Separate arms, so each `{0: chunk}` holds one kind of delta, as `FunctionModel` requires.
                yield {0: chunk}
            else:
                yield {0: chunk}

    return FunctionModel(stream_function=respond)


async def run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model: FunctionModel,
    plugins: Sequence[PluginSettings],
    *,
    code_mode: bool = False,
) -> int:
    """Run `pax -p` in `tmp_path` on the stock agent with `model`, with only `plugins` built in."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(headless, 'create_agent', lambda: create_stock_agent(model))
    monkeypatch.setattr(headless, 'STOCK_PLUGINS', tuple(plugins))
    return await headless.run_headless(
        text='Is the job done?',
        settings=Settings(model=None, code_mode=code_mode),
        store=SettingsStore(tmp_path / 'config.db'),
        project=ProjectSettings(),
    )


def newest_request(messages: list[ModelMessage]) -> list[str]:
    """The newest request's parts: user text in full, every other part by its kind."""
    request = messages[-1]
    assert isinstance(request, ModelRequest)
    return [str(part.content) if isinstance(part, UserPromptPart) else part.part_kind for part in request.parts]


def stable(seen: list[list[ModelMessage]]) -> list[object]:
    """Each request's messages as JSON, without the timestamps and run ids that differ between runs."""

    def drop_volatile(fields: dict[str, object]) -> dict[str, object]:
        return {key: value for key, value in fields.items() if key not in _VOLATILE}

    return [json.loads(ModelMessagesTypeAdapter.dump_json(messages), object_hook=drop_volatile) for messages in seen]


async def saved_nudges(tmp_path: Path) -> list[tuple[str, tuple[str, ...]]]:
    """The pattern and tools of each nudge in the saved conversation."""
    saved = SqliteSessionStore(database=tmp_path / 'sessions.db')
    (entry,) = await saved.listing()
    messages = (await saved.get(conversation_id=entry.id)).messages
    return [
        (part.pattern, part.tool_names)
        for message in messages
        for part in message.parts
        if isinstance(part, LoopNudgePart)
    ]


LOOPING: list[Reply] = [*(read('status.txt', f'call_{index}') for index in range(4)), ['It is still pending.']]


async def test_a_repeated_tool_call_is_nudged_and_the_turn_goes_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / 'status.txt').write_text('pending\n')
    seen: list[list[ModelMessage]] = []
    assert await run(tmp_path, monkeypatch, scripted(LOOPING, seen), (CODER, DETECT_LOOPS)) == 0
    assert capsys.readouterr().out == 'It is still pending.\n'
    # The third identical result brings the nudge, after that request's tool result, and the model goes on.
    assert [newest_request(messages) for messages in seen[1:]] == snapshot(
        [
            ['tool-return'],
            ['tool-return'],
            [
                'tool-return',
                '<system>Your last 3 calls to `read_file` used the same arguments and returned the same result, so calling it again will not change the outcome. Change your approach, or explain what is blocking you.</system>',
            ],
            ['tool-return'],
        ]
    )
    # Appended, never rewritten: each request starts with every message of the one before it.
    for before, after in zip(seen, seen[1:]):
        assert stable([after[: len(before)]]) == stable([before])
    assert await saved_nudges(tmp_path) == [('repeated_result', ('read_file',))]


async def test_a_disabled_detect_loops_never_nudges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / 'status.txt').write_text('pending\n')
    seen: list[list[ModelMessage]] = []
    assert await run(tmp_path, monkeypatch, scripted(LOOPING, seen), (CODER, DISABLED)) == 0
    assert capsys.readouterr().out == 'It is still pending.\n'
    assert [newest_request(messages) for messages in seen[1:]] == [['tool-return']] * 4
    assert await saved_nudges(tmp_path) == []


@pytest.mark.parametrize('detect_loops', [DISABLED, DETECT_LOOPS], ids=['disabled', 'enabled'])
async def test_a_run_without_a_loop_sends_the_same_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, detect_loops: PluginSettings
) -> None:
    for name in ('a.txt', 'b.txt'):
        (tmp_path / name).write_text(f'{name}\n')
    replies = [read('a.txt', 'call_a'), read('b.txt', 'call_b'), read('a.txt', 'call_c'), ['Both read.']]
    without: list[list[ModelMessage]] = []
    tools_without: list[list[ToolDefinition]] = []
    assert await run(tmp_path, monkeypatch, scripted(replies, without, tools_without), (CODER,)) == 0
    seen: list[list[ModelMessage]] = []
    tools: list[list[ToolDefinition]] = []
    assert await run(tmp_path, monkeypatch, scripted(replies, seen, tools), (CODER, detect_loops)) == 0
    assert len(seen) == 4
    # The instructions ride on each request, so this compares them too.
    assert stable(seen) == stable(without)
    assert tools == tools_without


@pytest.mark.parametrize(
    ('snippets', 'nudged'),
    [
        (["read_file(path='status.txt')"] * 4, True),
        ([f"status_{index} = read_file(path='status.txt')" for index in range(4)], False),
    ],
    ids=['same-code', 'different-code'],
)
async def test_code_mode_compares_run_code_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, snippets: list[str], nudged: bool
) -> None:
    """In code mode the model's tool calls are `run_code` calls, so the code is what has to repeat."""
    (tmp_path / 'status.txt').write_text('pending\n')
    replies: list[Reply] = [
        *(
            [DeltaToolCall(name='run_code', json_args=json.dumps({'code': code}), tool_call_id=f'call_{index}')]
            for index, code in enumerate(snippets)
        ),
        ['It is still pending.'],
    ]
    seen: list[list[ModelMessage]] = []
    assert await run(tmp_path, monkeypatch, scripted(replies, seen), (CODER, DETECT_LOOPS), code_mode=True) == 0
    expected = [('repeated_result', ('run_code',))] if nudged else []
    assert await saved_nudges(tmp_path) == expected


LOOPING_SENTENCE = 'Let me check the config file once more. '


async def test_thinking_that_repeats_itself_is_retried_with_a_nudge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: list[list[ModelMessage]] = []
    replies = [[DeltaThinkingPart(content=LOOPING_SENTENCE)] * 100, ['The config sets no port.']]
    assert await run(tmp_path, monkeypatch, scripted(replies, seen), (CODER, DETECT_LOOPS)) == 0
    assert capsys.readouterr().out == 'The config sets no port.\n'
    # The stream was stopped at the first check past 2,000 characters and the response dropped, so the retry is the
    # same request with the feedback and the nudge appended.
    assert newest_request(seen[1]) == snapshot(
        [
            'Is the job done?',
            """\
Your previous response failed the application's validation (this is automated feedback, not a message from the user):
<pyai:retry_feedback>
Your previous response was rejected because it kept repeating itself.
</pyai:retry_feedback>\
""",
            '<system>Your thinking repeated the same passage 51 times, which is not making progress. Change your approach, or explain what is blocking you.</system>',
        ]
    )
    assert await saved_nudges(tmp_path) == [('repeated_thinking', ())]


async def test_an_answer_that_repeats_a_line_on_purpose_is_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Visible text is not checked within a response, so an answer that repeats a line as asked ends the turn."""
    answer = 'All work and no play makes Jack a dull boy.\n' * 80
    seen: list[list[ModelMessage]] = []
    assert (
        await run(tmp_path, monkeypatch, scripted([[answer[:1000], answer[1000:]]], seen), (CODER, DETECT_LOOPS)) == 0
    )
    assert capsys.readouterr().out == answer + '\n'
    assert len(seen) == 1
    assert await saved_nudges(tmp_path) == []


async def test_a_file_with_repeated_rows_is_written_without_a_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tool call arguments are not checked for repeated passages, so a fixture of identical rows is written."""
    content = "INSERT INTO jobs VALUES (1, 'pending');\n" * 120
    arguments = json.dumps({'path': 'fixture.sql', 'content': content})
    chunks = [arguments[start : start + 100] for start in range(0, len(arguments), 100)]
    write = [
        DeltaToolCall(name='write_file', tool_call_id='call_write'),
        *(DeltaToolCall(json_args=chunk) for chunk in chunks),
    ]
    seen: list[list[ModelMessage]] = []
    assert await run(tmp_path, monkeypatch, scripted([write, ['Written.']], seen), (CODER, DETECT_LOOPS)) == 0
    assert (tmp_path / 'fixture.sql').read_text() == content
    assert len(seen) == 2
    assert newest_request(seen[1]) == ['tool-return']
