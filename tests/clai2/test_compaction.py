"""The built-in `compaction` plugin, loaded with `load_plugin` and driven through one `chat()` run."""

import asyncio
import io
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import anyio
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import JsonValue, ValidationError
from rich.console import Console
from termflow.tui.menu import MenuResult

from pydantic_ai import Agent, ModelHTTPError, capture_run_messages
from pydantic_ai.messages import (
    CompactionPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RequestUsage, RunUsage
from pydantic_ai_harness.compaction import (
    FallbackCompaction,
    SlidingWindowCompaction,
    SummarizingCompaction,
    estimate_message_tokens,
)
from pydantic_ai_harness.step_persistence.conversations import SqliteConversationStore
from pydantic_clai2 import DEFAULT_PLUGINS, Session, chat
from pydantic_clai2.builtin_plugins.compaction import (
    CompactionPlugin,
    CompactionSettings,
    CompactionSource,
    build_chain,
)
from pydantic_clai2.commands import Command, Commands
from pydantic_clai2.config import PluginSettings, Settings
from pydantic_clai2.config.project_settings import ProjectSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.plugins import (
    ConversationChanged,
    LoadedPlugin,
    PluginHost,
    SessionEnd,
    SessionStart,
    TaskInfo,
    Transcript,
    TurnEnd,
    load_plugin,
)
from pydantic_clai2.plugins.loader import PluginLoader
from pydantic_clai2.runtime.session_runtime import create_shell
from pydantic_clai2.ui.menus import dialogs
from tests.clai2.menu_script import Script, pick, typed
from tests.clai2.test_app_edges import inputs


def make_plugin(
    conversation: Transcript | Session[None, str] | None = None, **settings: JsonValue
) -> LoadedPlugin[None]:
    host = PluginHost[None](
        name='compaction', console=Console(file=io.StringIO()), settings=dict(settings), conversation=conversation
    )
    return load_plugin(CompactionPlugin, host)


def summary_prompt(summary_run: Sequence[ModelMessage]) -> str:
    """The user turn the summariser received, as `capture_run_messages` recorded it."""
    request = summary_run[0]
    assert isinstance(request, ModelRequest)
    [prompt] = request.parts
    assert isinstance(prompt, UserPromptPart) and isinstance(prompt.content, str)
    return prompt.content


READINESS_WAIT_TIMEOUT = 10

REPLY = 'hi, ' + 'here is what I found. ' * 10
"""Long enough that a short summary of it makes the history smaller."""


def two_turns() -> list[ModelMessage]:
    """Two request/response pairs, the second prompt long enough that summarizing it saves tokens."""
    return [
        ModelRequest.user_text_prompt('hello there'),
        ModelResponse(parts=[TextPart(REPLY)]),
        ModelRequest.user_text_prompt('and again'),
        ModelResponse(parts=[TextPart('yo')]),
    ]


@pytest.mark.parametrize(
    'focus',
    [
        '',
        'the auth work',
        'don\'t lose the "auth" notes',
        r'keep C:\work\notes and  the "unfinished section',
        'preserve the decisions\nand the open questions',
    ],
)
async def test_compact_sends_the_history_and_focus_to_the_summariser(focus: str) -> None:
    transcript = Transcript(messages=two_turns(), model=TestModel(custom_output_text='the gist'))
    plugin = make_plugin(transcript, protected_tokens=0)
    plugin.host.status.context_alert = True
    with capture_run_messages() as summary_run:
        notice = await plugin.commands.execute_async(f'/compact {focus}')
    prompt = summary_prompt(summary_run)
    assert (
        f'<message role="user">\nhello there\n</message>\n'
        f'<message role="assistant">\n{REPLY}\n</message>\n'
        '<message role="user">\nand again\n</message>'
    ) in prompt
    if focus:
        assert prompt.endswith(f'Give particular weight to: {focus}')
    else:
        assert 'Give particular weight to:' not in prompt
    assert notice == 'Compacted 4 messages down to 3; about 56 of 60 tokens saved.'
    # The first prompt precedes the summary of everything after it.
    first_request, summary, last_response = transcript.messages
    assert isinstance(summary, ModelRequest) and isinstance(first_request, ModelRequest)
    [request_part] = first_request.parts
    [summary_part] = summary.parts
    assert isinstance(summary_part, CompactionPart) and summary_part.content == 'the gist'
    assert isinstance(request_part, UserPromptPart)
    assert request_part.content == 'hello there', 'harness keeps the first user message verbatim'
    assert isinstance(last_response, ModelResponse)
    assert plugin.host.status.context_alert, 'the colour follows the figure: both wait for the next reading'


async def test_compact_says_when_there_is_nothing_to_do() -> None:
    plugin = make_plugin()
    assert plugin.commands.runs_live('/compact'), 'a long summary must keep the spinner up and stay cancellable'
    assert await plugin.commands.execute_async('/compact') == 'Nothing to compact: the conversation is empty.'
    short = Transcript(messages=[ModelRequest.user_text_prompt('hi')], model='test')
    plugin = make_plugin(short)
    with capture_run_messages() as summary_run:
        notice = await plugin.commands.execute_async('/compact')
    assert notice == 'Nothing to compact: compacting would not make the conversation smaller.' and not summary_run
    with pytest.raises(ValueError, match='Choose a model first'):
        await make_plugin(
            Transcript(messages=[ModelRequest.user_text_prompt('hi')]), protected_tokens=0
        ).commands.execute_async('/compact')


def tool_rounds(rounds: int, return_chars: int) -> list[ModelMessage]:
    """One prompt, then `rounds` tool calls and returns of `return_chars` characters each, then an answer."""
    messages: list[ModelMessage] = [ModelRequest.user_text_prompt('fix the bug ' * 50)]
    for index in range(rounds):
        messages.append(ModelResponse(parts=[ToolCallPart('read', {'n': index}, tool_call_id=f'call{index}')]))
        messages.append(ModelRequest(parts=[ToolReturnPart('read', 'x' * return_chars, tool_call_id=f'call{index}')]))
    messages.append(ModelResponse(parts=[TextPart('done')]))
    return messages


@pytest.mark.parametrize('strategy', ['summarization', 'truncation'])
async def test_compact_shrinks_a_history_just_over_the_protected_tail(strategy: str) -> None:
    """186 messages a little over 50,000 tokens once became 187 with nothing saved; now half is compacted."""
    before = tool_rounds(92, 2160)
    assert len(before) == 186 and sum(estimate_message_tokens(before)) == 50_107
    transcript = Transcript(messages=before, model=TestModel(custom_output_text='the gist'))
    notice = await make_plugin(transcript, strategy=strategy).commands.execute_async('/compact')
    after = transcript.messages
    saved = 50_107 - sum(estimate_message_tokens(after))
    assert len(after) < 100 and 24_000 < saved < 25_053
    assert notice == f'Compacted 186 messages down to {len(after)}; about {saved:,} of 50,107 tokens saved.'


async def test_compact_keeps_the_history_when_the_summary_is_no_smaller() -> None:
    before = two_turns()
    transcript = Transcript(messages=before, model=TestModel(custom_output_text='gist ' * 500))
    with capture_run_messages() as summary_run:
        notice = await make_plugin(transcript, protected_tokens=0).commands.execute_async('/compact')
    assert notice == 'Nothing to compact: compacting would not make the conversation smaller.'
    assert summary_run, 'the summary was written, then discarded'
    assert transcript.messages == before and len(before) == 4


async def test_compact_counts_tokens_the_provider_reported() -> None:
    """Thinking that came back without text is most of this history; the protected tail and report count it."""
    history: list[ModelMessage] = [ModelRequest.user_text_prompt('start')]
    for turn in range(6):
        history.append(
            ModelResponse(
                parts=[ThinkingPart(content='', signature='sig'), TextPart('ok')],
                usage=RequestUsage(input_tokens=100 + 10_000 * turn, output_tokens=9_000),
            )
        )
        history.append(ModelRequest.user_text_prompt(f'next {turn}'))
    transcript = Transcript(messages=history, model=TestModel(custom_output_text='the gist'))
    notice = await make_plugin(transcript, protected_tokens=15_000).commands.execute_async('/compact')
    assert notice == 'Compacted 13 messages down to 6; about 39,998 of 59,002 tokens saved.'
    first, summary, *tail = transcript.messages
    assert isinstance(summary.parts[0], CompactionPart) and first is history[0]
    assert tail == history[-4:], 'at least the last 15,000 reported tokens are kept verbatim'


async def test_compact_is_traced_with_the_global_agent_instrumentation() -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    Agent.instrument_all(InstrumentationSettings(tracer_provider=provider))
    try:
        transcript = Transcript(messages=two_turns(), model=TestModel(custom_output_text='the gist'))
        await make_plugin(transcript, protected_tokens=0).commands.execute_async('/compact')
    finally:
        Agent.instrument_all(False)
    names = {span.name for span in exporter.get_finished_spans()}
    assert {'compact_messages', 'invoke_agent summarizing_compaction'} <= names


def assert_truncated_without_a_summary(transcript: Transcript) -> None:
    request, response = transcript.messages
    assert isinstance(request, ModelRequest) and isinstance(response, ModelResponse)
    assert all(isinstance(part, UserPromptPart) for part in request.parts), 'no SystemPromptPart summary'


async def test_truncation_strategy_drops_older_messages_without_a_summary() -> None:
    transcript = Transcript(messages=two_turns(), model=TestModel())
    plugin = make_plugin(transcript, strategy='truncation', protected_tokens=0)
    with capture_run_messages() as summary_run:
        notice = await plugin.commands.execute_async('/compact')
    assert notice.startswith('Compacted 4 messages down to 2;') and not summary_run
    assert_truncated_without_a_summary(transcript)


async def test_summariser_failure_falls_back_to_truncation() -> None:
    def refuse(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        raise ModelHTTPError(status_code=503, model_name='down', body=None)

    transcript = Transcript(messages=two_turns(), model=FunctionModel(refuse))
    plugin = make_plugin(transcript, protected_tokens=0)
    notice = await plugin.commands.execute_async('/compact')
    assert notice.startswith('Compacted 4 messages down to 2;')
    assert_truncated_without_a_summary(transcript)


def protected_tails(chain: FallbackCompaction[None]) -> list[int | None]:
    """The `keep_tokens` of each strategy in the chain, in fallback order."""
    tails: list[int | None] = []
    for strategy in chain.fallback_chain:
        assert isinstance(strategy, SummarizingCompaction | SlidingWindowCompaction)
        tails.append(strategy.keep_tokens)
    return tails


@pytest.mark.parametrize(('strategy', 'strategies'), [('summarization', 2), ('truncation', 1)])
def test_threshold_and_protected_tokens_reach_every_strategy(strategy: str, strategies: int) -> None:
    """Unset, the chain compacts at 85% and keeps the last 50,000 tokens; set, it uses the chosen values."""
    default = build_chain(CompactionSettings.model_validate({'strategy': strategy}))
    assert default.max_fraction == 0.85 and protected_tails(default) == [50_000] * strategies
    chosen = build_chain(
        CompactionSettings.model_validate({'strategy': strategy, 'threshold': 0.6, 'protected_tokens': 0})
    )
    assert chosen.max_fraction == 0.6 and protected_tails(chosen) == [0] * strategies


async def test_settings_are_validated_on_activation() -> None:
    with pytest.raises(ValidationError):
        make_plugin(threshold=0)
    with pytest.raises(ValidationError):
        make_plugin(strategy='magic')
    with pytest.raises(ValidationError):
        make_plugin(compact_at=0.5)


@pytest.mark.parametrize('strategy', ['summarization', 'truncation'])
async def test_direct_fallback_capability_compacts_before_gauging(strategy: str) -> None:
    session = Session(Agent(TestModel(custom_output_text='gist')), deps=None)
    plugin = make_plugin(session, strategy=strategy, context_window=1000, protected_tokens=0)
    assert isinstance(plugin.capabilities[0], FallbackCompaction)
    session.plugins = plugin.capabilities
    session.replace_messages(
        [
            ModelRequest.user_text_prompt('first'),
            ModelResponse(parts=[TextPart('reply')]),
            ModelRequest.user_text_prompt('old ' * 1000),
            ModelResponse(parts=[TextPart('reply')]),
        ]
    )
    await session.prompt('new')
    assert not any(
        isinstance(part, UserPromptPart) and part.content == 'old ' * 1000
        for message in session.messages
        for part in message.parts
    )
    assert any(isinstance(part, CompactionPart) for message in session.messages for part in message.parts) == (
        strategy == 'summarization'
    )
    assert not plugin.host.status.context_alert, 'the gauge measures the compacted request'
    assert plugin.host.status.context_tokens is not None and plugin.host.status.context_tokens < 850
    assert plugin.host.status.context_window == 1000
    cramped = make_plugin(session, strategy=strategy, context_window=1, protected_tokens=50_000)
    session.plugins = cramped.capabilities
    await session.prompt('again')
    assert cramped.host.status.context_alert, 'a protected tail can still exceed the threshold'


@pytest.mark.parametrize(
    ('model_window', 'override', 'expected'),
    [(1_000_000, None, 1_000_000), (1_000_000, 200_000, 200_000), (None, None, None)],
)
async def test_gauge_uses_known_window_not_fallback(
    model_window: int | None, override: int | None, expected: int | None
) -> None:
    model = TestModel(profile={'context_window': model_window})
    session = Session(Agent(model), deps=None)
    plugin = make_plugin(session, context_window=override)
    plugin.host.status.context_window = 123
    session.plugins = plugin.capabilities
    await session.prompt('hello')
    assert plugin.host.status.context_window == expected
    assert plugin.host.status.context_tokens is not None


async def test_unloading_clears_the_context_window_and_alert() -> None:
    plugin = make_plugin()
    plugin.host.status.context_alert = True
    plugin.host.status.context_tokens = 123
    plugin.host.status.context_window = 1_000_000
    await plugin.dispatch(SessionEnd(reason='exit'))
    assert not plugin.host.status.context_alert
    assert plugin.host.status.context_window is None
    assert plugin.host.status.context_tokens == 123


@pytest.mark.parametrize('strategy', ['summarization', 'truncation'])
async def test_compact_is_saved_without_another_turn(tmp_path: Path, strategy: str) -> None:
    store = SqliteConversationStore(database=tmp_path / 'sessions.db')
    agent = Agent(TestModel(custom_output_text='the gist'))
    session = Session(agent, deps=None, conversations=store, workspace=tmp_path)
    await session.prompt('first')
    await session.prompt('second ' * 20)
    before = session.messages
    plugin = make_plugin(session, strategy=strategy, protected_tokens=0)

    notice = await plugin.commands.execute_async('/compact don\'t lose the "auth" notes')

    assert notice.startswith('Compacted 4 messages down to ')
    assert session.messages != before
    restored = Session(agent, deps=None, conversations=store, workspace=tmp_path)
    await restored.resume(session.summary.id)
    assert restored.messages == session.messages


async def test_gauge_marks_its_reading_estimated_until_a_response_reports_usage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first request's reading counts characters and no tool definitions, so it reads `~`."""
    inputs(monkeypatch, ['hi', '/inspect', 'again', '/exit'])
    readings: list[str] = []

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        readings.append(shell.status.text())
        yield 'hello'

    shell = create_shell(
        Agent(FunctionModel(stream_function=stream)),
        deps=None,
        plugins=(),
        usage_limits=None,
        console=Console(file=io.StringIO()),
        settings=None,
        store=SettingsStore(tmp_path / 'config.db'),
        builtin_plugins=(
            PluginSettings(id='compaction', factory='pydantic_clai2.builtin_plugins.compaction', settings={}),
        ),
        project=ProjectSettings(),
    )

    def inspect(args: list[str]) -> str:
        readings.append(shell.status.text())
        return ''

    shell.commands.register(Command(name='inspect', description='Read status', handler=inspect))
    await shell.loader.load_all()
    try:
        await shell.run()
    finally:
        await shell.loader.close('exit')
    during_first, after_first, during_second = (reading.split('context: ')[1].split(' ')[0] for reading in readings)
    assert during_first.startswith('~')
    assert not after_first.startswith('~'), 'the response reported its usage'
    assert not during_second.startswith('~'), 'anchored on the first response'


async def test_gauge_clears_the_estimate_mark_within_a_turn() -> None:
    """After a tool call, the turn's next request is anchored on the first response's usage."""
    estimated: list[bool] = []

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        estimated.append(plugin.host.status.context_estimated)
        if len(messages) == 1:
            yield {0: DeltaToolCall(name='work', json_args='{}')}
        else:
            yield 'done'

    agent = Agent(FunctionModel(stream_function=stream))

    @agent.tool_plain
    def work() -> str:
        return 'worked'

    session = Session(agent, deps=None)
    plugin = make_plugin(session)
    session.plugins = plugin.capabilities
    await session.prompt('hi')
    assert estimated == [True, False]


async def test_shell_loads_the_plugin_and_compacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs(monkeypatch, ['first', 'second ' * 20, '/compact don\'t lose the "auth" notes', '/plugins list', '/exit'])
    output = io.StringIO()
    await chat(
        Agent(TestModel()),
        deps=None,
        console=Console(file=output, width=200),
        settings=Settings(model='test'),
        store=SettingsStore(tmp_path / 'config.db'),
        builtin_plugins=(
            PluginSettings(
                id='compaction', factory='pydantic_clai2.builtin_plugins.compaction', settings={'protected_tokens': 0}
            ),
        ),
    )
    text = output.getvalue()
    assert 'Compacted 4 messages down to 3' in text
    assert 'compaction' in text and 'pydantic_clai2.builtin_plugins.compaction (built-in)' in text


def test_settings_source_shows_validates_saves_and_resets() -> None:
    saved: list[dict[str, JsonValue]] = []
    host = PluginHost[None](
        name='compaction',
        console=Console(file=io.StringIO()),
        settings={'protected_tokens': 0},
        save_settings=saved.append,
    )
    source = CompactionSource(host)
    strategy, threshold, protected, window, summarizer, background, start = source.rows()
    assert [source.current(row) for row in source.rows()] == [
        'summarization',
        '0.85',
        '0',
        '(not set)',
        '(not set)',
        'false',
        '0.85',
    ]
    assert [row.default for row in source.rows()] == [
        'summarization',
        '0.85',
        '50000',
        '(not set)',
        '(not set)',
        'false',
        '0.85',
    ]
    assert background.choices == ('true', 'false') and not background.allow_custom
    assert source.problem(start, '1') == 'Input should be less than 1'
    assert strategy.choices == ('summarization', 'truncation') and not strategy.allow_custom
    assert threshold.allow_custom and not threshold.choices
    assert source.problem(threshold, '0') == 'Input should be greater than 0'
    assert source.problem(threshold, 'most') is not None
    assert source.problem(protected, '2.5') == 'Input should be a valid integer'
    assert source.problem(threshold, '1') is None
    assert source.apply(threshold, '0.7') == 'Saved Threshold.'
    assert source.apply(window, '200000') == 'Saved Context window.'
    assert source.apply(summarizer, 'openai:gpt-5-mini') == 'Saved Summarization model.'
    assert source.apply(strategy, 'truncation') == 'Saved Strategy.'
    assert source.apply(background, 'true') == 'Saved Background compaction.'
    assert source.apply(start, '0.9') == 'Saved Background start.'
    assert saved[-1] == {
        'protected_tokens': 0,
        'threshold': 0.7,
        'context_window': 200_000,
        'summarization_model': 'openai:gpt-5-mini',
        'strategy': 'truncation',
        'background_compaction': True,
        'background_start': 0.9,
    }, 'only chosen settings are saved, so later default changes still apply'
    assert source.current(window) == '200000'
    assert source.reset(threshold) == 'Reset Threshold.'
    assert 'threshold' not in saved[-1] and source.current(threshold) == '0.85'
    assert source.reset(threshold) == 'Reset Threshold.'


async def test_configure_outside_a_terminal_explains_how() -> None:
    assert 'from a terminal' in await make_plugin().plugin.configure()


@pytest.mark.parametrize(
    ('script', 'shown', 'saves'),
    [
        (
            Script([pick('threshold'), MenuResult(cancelled=True)], [], [typed('0.5')]),
            'Saved Threshold.',
            [{'threshold': 0.5}],
        ),
        (Script([MenuResult(cancelled=True)], [], []), 'Compaction settings: nothing changed.', []),
    ],
)
async def test_configure_shows_the_settings_menu(
    monkeypatch: pytest.MonkeyPatch, script: Script, shown: str, saves: list[dict[str, JsonValue]]
) -> None:
    saved: list[dict[str, JsonValue]] = []
    host = PluginHost[None](
        name='compaction',
        console=Console(file=io.StringIO(), force_terminal=True),
        settings={},
        save_settings=saved.append,
    )
    plugin = load_plugin(CompactionPlugin, host).plugin
    monkeypatch.setattr(dialogs, 'RUNNERS', script.runners)
    assert await plugin.configure() == shown
    assert saved == saves


async def test_configured_settings_rebuild_the_chain_for_the_next_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Beside the default `coder`, `/plugins configure compaction` reloads it, so the next run binds the new chain."""
    store = SettingsStore(tmp_path / 'settings.db')
    coder, compaction = (
        next(plugin for plugin in DEFAULT_PLUGINS if plugin.id == name) for name in ('coder', 'compaction')
    )
    plugins = PluginLoader[None](
        store=store,
        console=Console(file=io.StringIO()),
        commands=Commands(),
        session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=store.load()),
        builtin=(coder, compaction),
    )
    try:
        await plugins.load_all()
        assert [entry.state for entry in plugins.entries()] == ['enabled, loaded', 'enabled, loaded']
        before = plugins.entries()[1].loaded
        assert before is not None and isinstance(before.capabilities[0], FallbackCompaction)
        assert before.capabilities[0].max_fraction == 0.85
        assert isinstance(before.plugin, CompactionPlugin) and protected_tails(before.plugin.chain) == [50_000, 50_000]

        async def save() -> str:
            before.host.save_settings(CompactionSettings(threshold=0.6, protected_tokens=1000))
            return 'saved'

        monkeypatch.setattr(before.plugin, 'configure', save)
        assert await plugins.configure('compaction') == 'saved'
        after = plugins.entries()[1].loaded
        assert after is not None and after is not before
        chain = after.capabilities[0]
        assert isinstance(chain, FallbackCompaction) and chain.max_fraction == 0.6
        assert isinstance(after.plugin, CompactionPlugin) and protected_tails(after.plugin.chain) == [1000, 1000]
        assert store.plugins()[0].settings == {'threshold': 0.6, 'protected_tokens': 1000}
    finally:
        await plugins.close('exit')


@pytest.mark.parametrize(
    ('settings', 'background_start'),
    [
        ({}, None),
        ({'background_compaction': True}, 0.85),
        ({'background_compaction': True, 'background_start': 0.7}, 0.7),
        ({'background_compaction': True, 'strategy': 'truncation'}, None),
    ],
)
def test_background_compaction_is_off_unless_chosen(
    settings: dict[str, JsonValue], background_start: float | None
) -> None:
    """Off by default, so the chain compacts only at the threshold, as before; truncation has nothing to write ahead."""
    assert build_chain(CompactionSettings.model_validate(settings)).background_start == background_start


class Summaries:
    """The summary model: the first summary, written in the background, waits for `release`.

    The conversation's replies stream instead.
    """

    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.started = asyncio.Event()
        self.calls = 0
        self.written = 0

    async def __call__(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        self.calls += 1
        if self.calls == 1:
            self.started.set()
            await self.release.wait()
        self.written += 1
        return ModelResponse(parts=[TextPart('the gist')])


async def reply(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
    yield 'done'


def above_the_background_start() -> list[ModelMessage]:
    """A first turn whose response reported 7,300 input tokens, above the background start of 7,225."""
    return [
        ModelRequest.user_text_prompt('start'),
        ModelResponse(parts=[TextPart('old ' * 4_000)], usage=RequestUsage(input_tokens=7_300)),
    ]


async def background_session(
    model: Summaries, history: list[ModelMessage] | None = None, protected_tokens: int = 0
) -> tuple[Session[None, str], LoadedPlugin[None]]:
    """A session whose turn ends above the background start (85% of the 8,500-token threshold), below the threshold."""
    session = Session(Agent(FunctionModel(model, stream_function=reply)), deps=None)
    plugin = make_plugin(session, context_window=10_000, protected_tokens=protected_tokens, background_compaction=True)
    session.plugins = plugin.capabilities
    session.replace_messages(history if history is not None else above_the_background_start())
    await session.prompt('next')
    with anyio.fail_after(READINESS_WAIT_TIMEOUT):
        await model.started.wait()
    return session, plugin


async def test_compact_uses_the_background_summary() -> None:
    model = Summaries()
    session, plugin = await background_session(model)
    assert isinstance(plugin.plugin, CompactionPlugin)
    assert session.conversation_id in plugin.plugin.chain.background_conversations
    assert model.written == 0, 'the turn finished while the summary was still being written'
    before = session.messages
    model.release.set()
    notice = await plugin.commands.execute_async('/compact')
    assert notice.startswith('Compacted 4 messages down to ')
    assert model.written == 1, 'the held summary was used, not a second one'
    summaries = [part for message in session.messages for part in message.parts if isinstance(part, CompactionPart)]
    assert [part.content for part in summaries] == ['the gist']
    assert session.messages[-2:] == before[-2:], 'the turn that ran while the summary was written is kept verbatim'
    assert session.conversation_id not in plugin.plugin.chain.background_conversations


async def test_compact_with_a_focus_writes_a_new_summary() -> None:
    model = Summaries()
    session, plugin = await background_session(model)
    assert isinstance(plugin.plugin, CompactionPlugin)
    await plugin.commands.execute_async('/compact keep the auth notes')
    assert model.written == 1, 'the held summary was dropped before it was written; the focused one was written'
    assert session.conversation_id not in plugin.plugin.chain.background_conversations


async def test_compact_ignores_a_background_summary_that_no_longer_applies() -> None:
    """A stale held summary leaves `/compact` capping the protected tail at half the history, as before."""
    model = Summaries()
    history: list[ModelMessage] = [ModelRequest.user_text_prompt('start')]
    for turn in range(5):
        history.append(ModelResponse(parts=[TextPart(f'{turn} ' * 2_000)]))
        history.append(ModelRequest.user_text_prompt(f'question {turn}'))
    history[-2] = replace(history[-2], usage=RequestUsage(input_tokens=7_300))
    history.pop()
    session, plugin = await background_session(model, history, protected_tokens=3_500)
    assert isinstance(plugin.plugin, CompactionPlugin)
    # Rewritten since the summary started, as an edited prompt would be.
    *kept, _, reply_ = session.messages
    session.replace_messages([*kept, ModelRequest.user_text_prompt('edited'), reply_])
    notice = await plugin.commands.execute_async('/compact')
    assert notice.startswith('Compacted 12 messages down to ')
    texts = [part.content for message in session.messages for part in message.parts if isinstance(part, TextPart)]
    assert '1 ' * 2_000 not in texts, 'the 3,500-token tail would keep it; capped at half the history, it goes'
    assert '2 ' * 2_000 in texts
    assert model.written == 1, 'the held summary was cancelled; the capped chain wrote a new one'
    assert session.conversation_id not in plugin.plugin.chain.background_conversations


async def test_leaving_the_conversation_drops_its_background_summary() -> None:
    model = Summaries()
    session, plugin = await background_session(model)
    assert isinstance(plugin.plugin, CompactionPlugin)
    chain = plugin.plugin.chain
    left = session.conversation_id
    await plugin.dispatch(ConversationChanged(conversation_id=left, title='Renamed'))
    assert left in chain.background_conversations, 'a new title is the same conversation'
    await session.clear()
    await plugin.dispatch(ConversationChanged(conversation_id=session.conversation_id, title=None))
    assert left not in chain.background_conversations and model.written == 0


@dataclass
class OneTask:
    """`/tasks` listing a single delegated task."""

    task: TaskInfo

    def listing(self) -> list[TaskInfo]:
        return [self.task]

    def history(self, task_id: str) -> list[ModelMessage]:
        raise NotImplementedError  # pragma: no cover

    async def stop(self, task_id: str) -> None:
        raise NotImplementedError  # pragma: no cover

    def background(self, task_id: str) -> None:
        raise NotImplementedError  # pragma: no cover

    async def resume(self, task_id: str) -> None:
        raise NotImplementedError  # pragma: no cover

    async def prompt(self, task_id: str, text: str) -> None:
        raise NotImplementedError  # pragma: no cover

    def steer(self, task_id: str, text: str) -> bool:
        raise NotImplementedError  # pragma: no cover


async def test_a_finished_task_drops_its_background_summary() -> None:
    """A delegated task runs as a conversation of its own; once it finishes, the next turn's end drops its summary."""
    model = Summaries()
    tasks = OneTask(
        TaskInfo(
            id='task-1',
            parent_id=None,
            agent='general-purpose',
            model=None,
            prompt='next',
            status='running',
            outcome=None,
            started_at=datetime.now(UTC),
            finished_at=None,
            background=True,
            resumable=True,
            activity='thinking',
            usage=RunUsage(),
        )
    )
    host = PluginHost[None](
        name='compaction',
        console=Console(file=io.StringIO()),
        settings={'context_window': 10_000, 'protected_tokens': 0, 'background_compaction': True},
        tasks=lambda: tasks,
    )
    plugin = load_plugin(CompactionPlugin, host)
    assert isinstance(plugin.plugin, CompactionPlugin)
    chain = plugin.plugin.chain
    task = Agent(FunctionModel(model, stream_function=reply), deps_type=type(None), capabilities=plugin.capabilities)
    await task.run('next', message_history=above_the_background_start(), conversation_id='task-1')
    with anyio.fail_after(READINESS_WAIT_TIMEOUT):
        await model.started.wait()
    await plugin.dispatch(TurnEnd(text='hi', outcome='completed'))
    assert 'task-1' in chain.background_conversations, 'the task still runs'
    tasks.task = replace(tasks.task, status='finished')
    await plugin.dispatch(TurnEnd(text='hi', outcome='completed'))
    assert 'task-1' not in chain.background_conversations and model.written == 0


async def test_unloading_cancels_the_background_summary() -> None:
    model = Summaries()
    session, plugin = await background_session(model)
    assert isinstance(plugin.plugin, CompactionPlugin)
    chain = plugin.plugin.chain
    await plugin.dispatch(SessionEnd(reason='exit'))
    assert session.conversation_id not in chain.background_conversations and model.written == 0


async def test_background_settings_are_tagged_for_older_builds(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / 'settings.db')
    compaction = next(plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'compaction')
    plugins = PluginLoader[None](
        store=store,
        console=Console(file=io.StringIO()),
        commands=Commands(),
        session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=store.load()),
        builtin=(compaction,),
    )
    try:
        await plugins.load_all()
        loaded = plugins.entries()[0].loaded
        assert loaded is not None
        loaded.host.save_settings(CompactionSettings(background_compaction=True, background_start=0.9))
        assert store.plugin_requirements('compaction') == {
            'background_compaction': ['compaction-background'],
            'background_start': ['compaction-background'],
        }
    finally:
        await plugins.close('exit')
