"""The built-in `spend_limits` plugin: harness `SpendLimits` from saved budgets, with totals in the status row."""

import io
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from pydantic import JsonValue, ValidationError
from rich.console import Console
from termflow.tui.menu import MenuResult

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel, StreamFunctionDef
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.spend_limits import SpendLimitExceeded, SpendLimits, UnpricedModelError, UnpricedModelWarning
from pydantic_ai_pax import DEFAULT_PLUGINS, Session, chat
from pydantic_ai_pax.app import create_stock_agent
from pydantic_ai_pax.builtin_plugins.spend_limits import SpendLimitsPlugin, SpendLimitsSettings, SpendLimitsSource
from pydantic_ai_pax.cli import headless
from pydantic_ai_pax.commands import Commands
from pydantic_ai_pax.config import PluginSettings, Settings
from pydantic_ai_pax.config.project_settings import ProjectSettings
from pydantic_ai_pax.config.settings_store import SettingsStore
from pydantic_ai_pax.plugins import LoadedPlugin, PluginHost, SessionStart, TurnEnd, TurnStart, load_plugin
from pydantic_ai_pax.plugins.loader import PluginLoader
from pydantic_ai_pax.ui.menus import dialogs
from pydantic_ai_pax.ui.rendering.status import Status
from tests.pax.menu_script import Script, pick, typed
from tests.pax.test_app_edges import inputs

FACTORY = 'pydantic_ai_pax.builtin_plugins.spend_limits'
PRICED = 'gpt-4o'
"""A model `genai-prices` knows."""
REPLY = ' '.join(['word'] * 1000)
"""Streamed, `FunctionModel` counts it as 1,000 output tokens on 50 input tokens: $0.010125 on `PRICED`."""


async def reply(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
    yield REPLY


async def tool_then_reply(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
    """Call a tool first, so the run makes a second request after the first one was charged."""
    yield REPLY
    if len(messages) == 1:
        yield {0: DeltaToolCall(name='look')}


def agent(respond: StreamFunctionDef = reply, *, model_name: str | None = PRICED) -> Agent[None, str]:
    built = Agent(FunctionModel(stream_function=respond, model_name=model_name), deps_type=type(None))

    @built.tool_plain
    def look() -> str:
        return 'nothing here'

    return built


def loader(tmp_path: Path, declaration: PluginSettings) -> PluginLoader[None]:
    store = SettingsStore(tmp_path / 'settings.db')
    return PluginLoader(
        store=store,
        console=Console(file=io.StringIO()),
        commands=Commands(),
        session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=store.load()),
        builtin=(declaration,),
    )


def make_plugin(
    session: Session[None, str], console: Console | None = None, **settings: JsonValue
) -> LoadedPlugin[None]:
    host = PluginHost[None](
        name='spend_limits',
        console=console or Console(file=io.StringIO()),
        settings=dict(settings),
        conversation=session,
    )
    plugin = load_plugin(SpendLimitsPlugin, host)
    session.plugins = list(plugin.capabilities)
    return plugin


def footer(plugin: LoadedPlugin[None]) -> str:
    """The status row as the shell paints it, with only this plugin's fragment."""
    return Status(fields=lambda: ('plugins',), status_segments=plugin.status_segments).text()


async def turn(plugin: LoadedPlugin[None], session: Session[None, str], text: str) -> None:
    await plugin.dispatch(TurnStart(text=text))
    await session.prompt(text)
    await plugin.dispatch(TurnEnd(text=text, outcome='completed'))


def test_built_in_is_off_by_default() -> None:
    [declaration] = [plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'spend_limits']
    assert declaration == PluginSettings(id='spend_limits', factory=FACTORY, enabled=False)


async def test_off_it_adds_no_capability_and_no_status(tmp_path: Path) -> None:
    [declaration] = [plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'spend_limits']
    plugins = loader(tmp_path, declaration)
    try:
        await plugins.load_all()
        assert [entry.state for entry in plugins.entries()] == ['disabled']
        assert plugins.capabilities() == [] and plugins.status_segments() == []
    finally:
        await plugins.close('exit')


async def test_a_run_budget_stops_the_run_that_crosses_it_and_resets_for_the_next() -> None:
    session = Session(agent(tool_then_reply), deps=None)
    plugin = make_plugin(session, budgets=[{'usd': '0.01', 'window': 'run'}])
    assert footer(plugin) == 'spend: run $0.0000/$0.0100'
    await plugin.dispatch(TurnStart(text='go'))
    with pytest.raises(
        SpendLimitExceeded, match=r"Budget 'default' exhausted for this run: spent \$0.010125 of \$0.01"
    ):
        await session.prompt('go')
    assert footer(plugin) == 'spend: run $0.0101/$0.0100'
    # The next prompt is a new run, so it starts from zero and its one request goes through.
    await plugin.dispatch(TurnStart(text='again'))
    assert footer(plugin) == 'spend: run $0.0000/$0.0100'
    session.agent = agent(reply)
    await turn(plugin, session, 'again')
    assert footer(plugin) == 'spend: run $0.0101/$0.0100'


async def test_a_token_ceiling_stops_the_run_too() -> None:
    session = Session(agent(tool_then_reply), deps=None)
    plugin = make_plugin(session, budgets=[{'tokens': 1000, 'window': 'run'}])
    with pytest.raises(SpendLimitExceeded, match='used 1050 of 1000 tokens'):
        await session.prompt('go')
    assert footer(plugin) == 'spend: run 1,050/1,000 tokens'


async def test_a_conversation_budget_spans_turns_and_a_new_conversation_starts_from_zero() -> None:
    session = Session(agent(), deps=None)
    output = io.StringIO()
    plugin = make_plugin(session, Console(file=output), budgets=[{'usd': 0.02, 'window': 'conversation'}])
    await turn(plugin, session, 'one')
    assert output.getvalue() == '', 'a turn the budget did not stop prints nothing'
    assert footer(plugin) == 'spend: conversation $0.0101/$0.0200'
    await turn(plugin, session, 'two')
    assert footer(plugin) == 'spend: conversation $0.0202/$0.0200'
    await plugin.dispatch(TurnStart(text='three'))
    assert footer(plugin) == 'spend: conversation $0.0202/$0.0200', 'a new prompt is still the same conversation'
    with pytest.raises(SpendLimitExceeded, match='exhausted for this conversation'):
        await session.prompt('three')
    await session.clear()
    assert footer(plugin) == 'spend: conversation $0.0000/$0.0200'
    await turn(plugin, session, 'fresh')
    assert footer(plugin) == 'spend: conversation $0.0101/$0.0200'


async def test_unpriced_responses_are_marked_not_silently_zero() -> None:
    session = Session(agent(model_name=None), deps=None)
    plugin = make_plugin(session, budgets=[{'usd': '1', 'window': 'run'}])
    # Pax hides library warnings from the terminal, so the status row is where the gap shows.
    with pytest.warns(UnpricedModelWarning):
        await turn(plugin, session, 'go')
    assert footer(plugin) == 'spend: run >=$0.0000/$1.0000 (1 unpriced)'


async def test_unpriced_responses_can_stop_the_run() -> None:
    session = Session(agent(model_name=None), deps=None)
    make_plugin(session, budgets=[{'usd': '1', 'window': 'run'}], on_unpriced='raise')
    with pytest.raises(UnpricedModelError):
        await session.prompt('go')


async def test_footer_shows_token_ceilings_names_and_plain_counters() -> None:
    session = Session(agent(), deps=None)
    budgets: list[JsonValue] = [
        {'tokens': 5000, 'window': 'run'},
        {'usd': '1', 'tokens': 9000, 'window': 'run', 'name': 'cap'},
        {'window': 'conversation'},
    ]
    plugin = make_plugin(session, budgets=budgets)
    await turn(plugin, session, 'go')
    assert footer(plugin) == (
        'spend: run 1,050/5,000 tokens, run cap $0.0101/$1.0000 1,050/9,000 tokens, conversation $0.0101'
    )


async def test_no_budgets_adds_no_status() -> None:
    plugin = make_plugin(Session(agent(), deps=None))
    assert isinstance(plugin.capabilities[0], SpendLimits)
    assert footer(plugin) == ''


@pytest.mark.parametrize(
    ('settings', 'message'),
    [
        (
            {'budgets': [{'usd': '5', 'window': 'day'}]},
            "window 'day' is not offered: Pax keeps spend in memory, so a day total would start again from zero",
        ),
        ({'budgets': [{'usd': '5'}]}, 'Field required'),
        ({'budgets': [{'usd': '0', 'window': 'run'}]}, 'Input should be greater than 0'),
        ({'budgets': [{'usd': 'lots', 'window': 'run'}]}, 'Input should be a valid decimal'),
        ({'budgets': [{'usd': '5', 'window': 'run', 'scope': 'x'}]}, 'Extra inputs are not permitted'),
        ({'budgets': [{'usd': '5', 'window': 'run', 'warn_at': 0.5}]}, 'Extra inputs are not permitted'),
        ({'budgets': [{'tokens': 5, 'window': 'run', 'name': 'a|b'}]}, 'must not contain'),
        (
            {'budgets': [{'usd': '5', 'window': 'run'}, {'usd': '6', 'window': 'run'}]},
            'only the smaller would ever apply',
        ),
        ({'on_unpriced': 'ignore'}, "Input should be 'zero' or 'raise'"),
    ],
)
def test_invalid_settings_are_refused_with_the_reason(settings: dict[str, JsonValue], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        SpendLimitsSettings.model_validate(settings)


async def test_enabling_with_invalid_settings_reports_why_and_loads_nothing(tmp_path: Path) -> None:
    declaration = PluginSettings(
        id='spend_limits', factory=FACTORY, enabled=False, settings={'budgets': [{'usd': '5', 'window': 'month'}]}
    )
    plugins = loader(tmp_path, declaration)
    try:
        await plugins.load_all()
        with pytest.raises(ValueError, match="window 'month' is not offered"):
            await plugins.enable('spend_limits')
        assert plugins.capabilities() == [] and plugins.status_segments() == []
    finally:
        await plugins.close('exit')


async def test_interactive_turn_stopped_by_a_budget_says_so_and_points_to_the_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs(monkeypatch, ['go', '/exit'])
    output = io.StringIO()
    await chat(
        agent(tool_then_reply),
        deps=None,
        console=Console(file=output, width=200),
        settings=Settings(model=None),
        store=SettingsStore(tmp_path / 'config.db'),
        builtin_plugins=(
            PluginSettings(
                id='spend_limits', factory=FACTORY, settings={'budgets': [{'usd': '0.01', 'window': 'run'}]}
            ),
        ),
    )
    text = ' '.join(output.getvalue().split())
    assert "SpendLimitExceeded: Budget 'default' exhausted for this run: spent $0.010125 of $0.01." in text
    assert 'A spend limit stopped this turn. Change the budgets with /plugins configure spend_limits.' in text
    assert 'Traceback' not in text


async def test_headless_run_stopped_by_a_budget_exits_nonzero_with_the_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)

    async def unknown_tool_then_reply(
        messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[str | DeltaToolCalls]:
        yield REPLY
        if len(messages) == 1:
            yield {0: DeltaToolCall(name='no_such_tool')}

    stock = create_stock_agent(FunctionModel(stream_function=unknown_tool_then_reply, model_name=PRICED))
    monkeypatch.setattr(headless, 'create_agent', lambda: stock)
    store = SettingsStore(tmp_path / 'config.db')
    store.save_plugin(
        PluginSettings(id='spend_limits', factory=FACTORY, settings={'budgets': [{'usd': '0.01', 'window': 'run'}]})
    )
    code = await headless.run_headless(text='go', settings=Settings(model=None), store=store, project=ProjectSettings())
    captured = capsys.readouterr()
    assert code == 1 and captured.out == ''
    assert "Budget 'default' exhausted for this run: spent $0.010125 of $0.01." in ' '.join(captured.err.split())


def test_settings_source_shows_validates_saves_and_resets() -> None:
    saved: list[dict[str, JsonValue]] = []
    host = PluginHost[None](
        name='spend_limits', console=Console(file=io.StringIO()), settings={}, save_settings=saved.append
    )
    source = SpendLimitsSource(host)
    budgets, unpriced = source.rows()
    assert [source.current(row) for row in source.rows()] == ['[]', 'zero']
    assert unpriced.choices == ('zero', 'raise') and not unpriced.allow_custom
    assert source.problem(budgets, '[{"usd": "5", "window": "day"}]') is not None
    assert source.problem(budgets, 'not json') is not None
    assert source.problem(budgets, '[{"usd": "5", "window": "run"}]') is None
    assert source.apply(budgets, '[{"usd": "5", "window": "run"}]') == 'Saved Budgets.'
    assert saved[-1] == {'budgets': [{'window': 'run', 'usd': '5'}]}
    host = PluginHost[None](
        name='spend_limits', console=Console(file=io.StringIO()), settings=saved[-1], save_settings=saved.append
    )
    source = SpendLimitsSource(host)
    assert source.apply(unpriced, 'raise') == 'Saved Unpriced models.'
    host = PluginHost[None](
        name='spend_limits', console=Console(file=io.StringIO()), settings=saved[-1], save_settings=saved.append
    )
    source = SpendLimitsSource(host)
    assert source.current(unpriced) == 'raise'
    assert source.reset(budgets) == 'Reset Budgets.'
    assert saved[-1] == {'on_unpriced': 'raise'}, 'only chosen settings are saved'


async def test_configure_opens_the_settings_menu(monkeypatch: pytest.MonkeyPatch) -> None:
    saved: list[dict[str, JsonValue]] = []
    host = PluginHost[None](
        name='spend_limits',
        console=Console(file=io.StringIO(), force_terminal=True),
        settings={},
        save_settings=saved.append,
    )
    plugin = load_plugin(SpendLimitsPlugin, host).plugin
    script = Script(
        [pick('budgets'), MenuResult(cancelled=True)], [], [typed('[{"usd": "2", "window": "conversation"}]')]
    )
    monkeypatch.setattr(dialogs, 'RUNNERS', script.runners)
    assert await plugin.configure() == 'Saved Budgets.'
    assert saved == [{'budgets': [{'window': 'conversation', 'usd': '2'}]}]
