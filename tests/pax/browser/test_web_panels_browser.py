"""Each panel that shows a session's work catches up with it without a manual refresh.

A panel asks the server again on triggers the page guesses, such as a turn ending. Three moments have each left one
showing stale data: a turn the session keeps after the page hears it end (#11874), a panel opened during a turn
(#11983), and a running turn that writes while the panel is open (#12223). Every panel below is checked at each one.
"""

import json
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Literal

import anyio
import pytest
from anyio.from_thread import start_blocking_portal
from playwright.async_api import Page, expect
from pydantic import TypeAdapter

from pydantic_ai import Agent, AgentRunResult, RunContext, models
from pydantic_ai.capabilities import AbstractCapability, WrapRunHandler
from pydantic_ai.messages import ModelMessage, ToolReturnPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai_harness.file_system import WRITE_FILE_KIND
from pydantic_ai_pax.builtin_plugins.web.protocol import PanelInfo
from pydantic_ai_pax.config import PluginSettings
from pydantic_ai_pax.plugins import PanelView, Plugin, SidePanel, TurnStart
from tests.manifests import Exempt
from tests.pax.browser.conftest import Serve
from tests.pax.test_web import TOKEN
from tests.pax.test_web_sessions import Fleet

Moment = Literal['kept late', 'opened mid-turn', 'written while open']
MOMENTS: tuple[Moment, ...] = ('kept late', 'opened mid-turn', 'written while open')
WRITE = 'write notes.py'
"""A prompt whose turn writes `notes.py`, then answers; in a `written while open` case it holds first."""
HOLD = 'hold'
"""A prompt whose turn runs until the test releases it, writing nothing."""
WRITTEN = ['notes.py']
KEEP_DELAY = 1.0
"""How long after the page hears a turn end the session keeps it, as a slow disk can make it."""


@dataclass(frozen=True)
class Panel:
    """A built-in panel that shows the files a session changed."""

    open: Callable[[Page], Awaitable[None]]
    shows: Callable[[Page, list[str]], Awaitable[None]]
    """Expect it to show exactly these changed files."""


async def open_changes(page: Page) -> None:
    await page.get_by_role('button', name=re.compile(r'^Changes: ')).click()
    await page.get_by_role('complementary', name='Changes').get_by_role('tab', name='This session').click()


async def changes_show(page: Page, files: list[str]) -> None:
    listed = page.get_by_role('complementary', name='Changes').get_by_role('list', name='Changed files')
    await expect(listed.get_by_role('listitem')).to_have_text([re.compile(re.escape(path)) for path in files])


async def shown_already(page: Page) -> None:
    """The header shows the count without being opened."""


async def count_shows(page: Page, files: list[str]) -> None:
    count = len(files)
    name = f'Changes: {count} {"file" if count == 1 else "files"} changed'
    await expect(page.get_by_role('button', name=re.compile(r'^Changes: '))).to_have_accessible_name(name)


PANELS: dict[str, Panel | Exempt] = {
    'Changes panel': Panel(open=open_changes, shows=changes_show),
    'Changes count': Panel(open=shown_already, shows=count_shows),
    'Git chip': Exempt(
        'Reads Git status, not the session history; test_a_running_turns_edits_show_before_it_ends in '
        'test_git_browser.py checks it as a running turn edits files'
    ),
    'Monty panel': Exempt('Shows the run_code cells the transcript already holds; it asks the server for nothing'),
}
"""The built-in panels beside or above the chat. The page has no registry of them, so add a new one here."""


class Turns(Plugin):
    """A plugin's side panel whose view changes as each turn starts."""

    scope = 'process'
    turns = 0

    def get_side_panels(self) -> Sequence[SidePanel]:
        return (SidePanel(id='turns', title='Turns', view=self.view),)

    async def on_turn_start(self, event: TurnStart) -> None:
        self.turns += 1

    async def view(self) -> PanelView:
        return PanelView(note=f'Turn {self.turns}')


@dataclass
class Script:
    """One case's session: whether its write holds the turn, and the event that releases a held turn."""

    hold_after_write: bool
    released: anyio.Event
    """Made on the fleet's event loop, which runs the turns; the test sets it through the portal."""


@dataclass
class KeepLate(AbstractCapability[None]):
    """Return each run `KEEP_DELAY` after it ends, so the session keeps the turn that long after the page hears it."""

    async def wrap_run(self, ctx: RunContext[None], *, handler: WrapRunHandler) -> AgentRunResult[object]:
        result = await handler()
        await anyio.sleep(KEEP_DELAY)  # real-wait: the delay is the slow persistence this mode stands for
        return result


def writer(workspace: anyio.Path, script: Script, *, keep_late: bool) -> Agent[None, str]:
    async def turns(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        last = messages[-1].parts[-1]
        if isinstance(last, UserPromptPart) and last.content == WRITE:
            arguments = json.dumps({'path': WRITTEN[0], 'content': 'print("hi")\n'})
            yield {0: DeltaToolCall(name='write_file', json_args=arguments, tool_call_id=f'write-{len(messages)}')}
            return
        if (isinstance(last, ToolReturnPart) and script.hold_after_write) or (
            isinstance(last, UserPromptPart) and last.content == HOLD
        ):
            await script.released.wait()
        yield 'Done'

    capabilities: list[AbstractCapability[None]] = [KeepLate()] if keep_late else []
    agent = Agent(FunctionModel(stream_function=turns), deps_type=type(None), capabilities=capabilities)

    @agent.tool_plain(tool_kind=WRITE_FILE_KIND)
    async def write_file(path: str, content: str) -> str:
        await (workspace / path).write_text(content)
        return f'Wrote {path}.'

    return agent


def folder(panel: str, moment: Moment) -> str:
    return re.sub(r'\W+', '-', f'{panel} {moment}').lower()


BUILT_IN = [name for name, entry in PANELS.items() if isinstance(entry, Panel)]
CASES: list[tuple[str, Moment]] = [(panel, moment) for panel in [*BUILT_IN, 'Plugin panels'] for moment in MOMENTS]


@dataclass
class PanelsUI:
    url: str
    fleet: Fleet
    scripts: dict[str, Script]


@pytest.fixture(scope='module')
def panels_ui(tmp_path_factory: pytest.TempPathFactory, serve: Serve) -> Iterator[PanelsUI]:
    root = tmp_path_factory.mktemp('browser-panels').resolve()
    turns = PluginSettings(id='turns', factory=f'{__name__}:Turns')
    with pytest.MonkeyPatch.context() as patch, start_blocking_portal() as portal:
        scripts = {folder(*case): Script(case[1] == 'written while open', portal.call(anyio.Event)) for case in CASES}
        agents = {
            name: writer(anyio.Path(root / name), script, keep_late=name.endswith('kept-late'))
            for name, script in scripts.items()
        }
        patch.setenv('HOME', str(root / 'home'))
        patch.setenv('XDG_CONFIG_HOME', str(root / 'config'))
        patch.setenv('XDG_STATE_HOME', str(root / 'state'))
        patch.setattr(models, 'ALLOW_MODEL_REQUESTS', False)
        fleet = Fleet(root=root, portal=portal, agents=agents, builtin=(turns,))
        try:
            with serve(fleet) as url:
                yield PanelsUI(url=url, fleet=fleet, scripts=scripts)
        finally:
            for script in scripts.values():
                portal.call(script.released.set)
            fleet.exit()


async def send(page: Page, prompt: str) -> None:
    composer = page.get_by_role('combobox', name='Message Pax')
    await expect(composer).to_be_enabled()
    await composer.fill(prompt)
    await composer.press('Enter')


async def run_moment(
    page: Page, panels_ui: PanelsUI, case: tuple[str, Moment], open_panel: Callable[[], Awaitable[None]]
) -> Callable[[], None]:
    """Open a fresh session for `case`, bring it to its moment with the panel open, and return what ends a held turn.

    The session has then written `WRITTEN`, and a held turn, if any, is still running.
    """
    name = folder(*case)
    title = f'{case[0]}, {case[1]}'
    script = panels_ui.scripts[name]
    session = panels_ui.fleet.new(name)
    # Saved, so it can be renamed.
    saved: list[ModelMessage] = []
    panels_ui.fleet.portal.call(panels_ui.fleet.runtimes[name].session.commit_messages, saved)
    assert panels_ui.fleet.post(f'/api/sessions/{session}/rename', {'title': title}).status_code == 204
    await page.goto(f'{panels_ui.url}/?token={TOKEN}')
    await page.locator('.sidebar').get_by_text(title, exact=True).click()
    await expect(page.locator('.session.current .session-title')).to_have_text(title)
    await expect(page.locator('.session[aria-busy="true"]')).to_have_count(0)
    activity = page.locator('.topbar-status .activity')
    moment = case[1]
    if moment == 'opened mid-turn':
        await send(page, WRITE)
        await expect(page.locator('.markdown').filter(has_text='Done')).to_have_count(1)
        await expect(activity).to_have_text('ready')
        await send(page, HOLD)
        await expect(activity).not_to_have_text('ready')
        await open_panel()
    else:
        await open_panel()
        await send(page, WRITE)
        if moment == 'kept late':
            await expect(activity).to_have_text('ready')
        else:
            await expect(page.get_by_role('button', name=f'Wrote {WRITTEN[0]}')).to_be_visible()
    return lambda: panels_ui.fleet.portal.call(script.released.set)


@pytest.mark.parametrize(('panel', 'moment'), [case for case in CASES if case[0] in BUILT_IN])
async def test_a_built_in_panel_catches_up(page: Page, panels_ui: PanelsUI, panel: str, moment: Moment) -> None:
    entry = PANELS[panel]
    assert isinstance(entry, Panel)
    release = await run_moment(page, panels_ui, (panel, moment), lambda: entry.open(page))
    try:
        await entry.shows(page, WRITTEN)
    finally:
        release()


SHORT_TURN = pytest.mark.xfail(
    strict=True,
    reason=(
        'The page renders socket messages once a frame, so a turn that starts and ends within one never shows as '
        'running, and plugin panels, which ask again only as `running` changes, keep the view from before it: '
        'https://github.com/pydantic/pydantic-ai/issues/14408'
    ),
)


@pytest.mark.parametrize(
    'moment', [pytest.param(moment, marks=SHORT_TURN if moment == 'kept late' else ()) for moment in MOMENTS]
)
async def test_every_plugin_panel_catches_up(page: Page, panels_ui: PanelsUI, moment: Moment) -> None:
    """Each panel `/api/panels` lists shows the view its plugin gives now."""
    [info] = listed(panels_ui)
    show = page.get_by_role('button', name=re.compile(rf'^Show {re.escape(info.title)}'))
    release = await run_moment(page, panels_ui, ('Plugin panels', moment), show.click)
    try:
        [now] = listed(panels_ui)
        await expect(page.get_by_role('complementary', name=now.title)).to_contain_text(now.note)
    finally:
        release()


def listed(panels_ui: PanelsUI) -> list[PanelInfo]:
    """The plugin panels as the server would show them now."""
    return TypeAdapter(list[PanelInfo]).validate_python(panels_ui.fleet.get('/api/panels').json())
