"""The built-in `compaction` plugin, loaded with `load_plugin` and driven through one `chat()` run."""

import io
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
from pydantic import JsonValue, ValidationError
from rich.console import Console

from pydantic_ai import Agent, ModelHTTPError, capture_run_messages
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.compaction import (
    FallbackCompaction,
    SlidingWindowCompaction,
    SummarizingCompaction,
    estimate_token_count,
)
from pydantic_ai_harness.step_persistence.conversations import SqliteConversationStore
from pydantic_clai2 import DEFAULT_PLUGINS, Session, chat
from pydantic_clai2.builtin_plugins import compaction as compaction_plugin
from pydantic_clai2.builtin_plugins.compaction import (
    CompactionPlugin,
    CompactionSettings,
    CompactionSource,
    build_chain,
)
from pydantic_clai2.commands import Commands
from pydantic_clai2.config import PluginSettings, Settings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.plugins import LoadedPlugin, PluginHost, SessionEnd, SessionStart, Transcript, load_plugin
from pydantic_clai2.plugins.loader import PluginLoader
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


REPLY = 'hi, ' + 'here is what I found. ' * 10
"""Long enough that a short summary of it makes the history smaller."""


def two_turns() -> list[ModelMessage]:
    """Two request/response pairs; the chain always keeps the newest pair intact."""
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
    assert f'User: hello there\nAssistant: {REPLY}\nUser: and again' in prompt
    if focus:
        assert prompt.endswith(f'Give particular weight to: {focus}')
    else:
        assert 'Give particular weight to:' not in prompt
    assert notice == 'Compacted 4 messages down to 3; about 47 of 61 tokens saved.'
    summary, first_request, last_response = transcript.messages
    assert isinstance(summary, ModelRequest) and isinstance(first_request, ModelRequest)
    [summary_part], [request_part] = summary.parts, first_request.parts
    assert isinstance(summary_part, SystemPromptPart) and isinstance(request_part, UserPromptPart)
    assert summary_part.content == 'Summary of previous conversation:\n\nthe gist'
    assert request_part.content == 'hello there', 'harness keeps the first user message verbatim'
    assert isinstance(last_response, ModelResponse)
    assert plugin.host.status.context_alert, 'the colour follows the figure: both wait for the next reading'


async def test_compact_says_when_there_is_nothing_to_do() -> None:
    assert await make_plugin().commands.execute_async('/compact') == 'Nothing to compact: the conversation is empty.'
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
    assert len(before) == 186 and estimate_token_count(before) == 50_127
    transcript = Transcript(messages=before, model=TestModel(custom_output_text='the gist'))
    notice = await make_plugin(transcript, strategy=strategy).commands.execute_async('/compact')
    after = transcript.messages
    saved = 50_127 - estimate_token_count(after)
    assert len(after) < 100 and 24_000 < saved < 25_063
    assert notice == f'Compacted 186 messages down to {len(after)}; about {saved:,} of 50,127 tokens saved.'


async def test_compact_keeps_the_history_when_the_summary_is_no_smaller() -> None:
    before = two_turns()
    transcript = Transcript(messages=before, model=TestModel(custom_output_text='gist ' * 500))
    with capture_run_messages() as summary_run:
        notice = await make_plugin(transcript, protected_tokens=0).commands.execute_async('/compact')
    assert notice == 'Nothing to compact: compacting would not make the conversation smaller.'
    assert summary_run, 'the summary was written, then discarded'
    assert transcript.messages == before and len(before) == 4


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
    assert any(isinstance(part, SystemPromptPart) for message in session.messages for part in message.parts) == (
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
    strategy, threshold, protected, window, summarizer = source.rows()
    assert [source.current(row) for row in source.rows()] == ['summarization', '0.85', '0', '(not set)', '(not set)']
    assert [row.default for row in source.rows()] == ['summarization', '0.85', '50000', '(not set)', '(not set)']
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
    assert saved[-1] == {
        'protected_tokens': 0,
        'threshold': 0.7,
        'context_window': 200_000,
        'summarization_model': 'openai:gpt-5-mini',
        'strategy': 'truncation',
    }, 'only chosen settings are saved, so later default changes still apply'
    assert source.current(window) == '200000'
    assert source.reset(threshold) == 'Reset Threshold.'
    assert 'threshold' not in saved[-1] and source.current(threshold) == '0.85'
    assert source.reset(threshold) == 'Reset Threshold.'


async def test_configure_outside_a_terminal_explains_how() -> None:
    assert 'from a terminal' in await make_plugin().plugin.configure()


@pytest.mark.parametrize('messages', [['Saved Threshold.'], []])
async def test_configure_runs_the_field_menu(monkeypatch: pytest.MonkeyPatch, messages: list[str]) -> None:
    host = PluginHost[None](name='compaction', console=Console(file=io.StringIO(), force_terminal=True), settings={})
    plugin = load_plugin(CompactionPlugin, host).plugin

    async def run_worker(work: Callable[[], list[str]]) -> list[str]:
        return messages

    monkeypatch.setattr(compaction_plugin, 'run_worker', run_worker)
    assert await plugin.configure() == ('\n'.join(messages) or 'No compaction settings changed.')


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
