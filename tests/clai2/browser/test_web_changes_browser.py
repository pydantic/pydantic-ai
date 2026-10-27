"""The Changes panel in the shipped UI: a session's changed files after two turns, by scope."""

import json
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from anyio.from_thread import start_blocking_portal
from playwright.async_api import Page, expect

from pydantic_ai import Agent, models
from pydantic_ai.messages import ModelMessage, UserPromptPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai_harness.filesystem import EDIT_FILE_KIND, WRITE_FILE_KIND
from pydantic_clai2.builtin_plugins.web import changes as changes_module
from tests.clai2.browser.conftest import Serve
from tests.clai2.test_web import TOKEN
from tests.clai2.test_web_sessions import Fleet

FIRST = 'create a.py and b.py with a hello function each'
SECOND = 'edit a.py to add a docstring, and create c.py'
TURNS: dict[str, list[tuple[str, dict[str, str]]]] = {
    FIRST: [
        ('write_file', {'path': 'a.py', 'content': 'def hello():\n    return "a"\n'}),
        ('write_file', {'path': 'b.py', 'content': 'def hello():\n    return "b"\n'}),
    ],
    SECOND: [
        ('edit_file', {'path': 'a.py', 'old_text': '():\n', 'new_text': '():\n    """Say hello."""\n'}),
        ('write_file', {'path': 'c.py', 'content': 'def hello():\n    return "c"\n'}),
    ],
}
"""The issue's repro: each prompt, and the file tool calls its model makes."""


async def file_turns(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
    last = messages[-1].parts[-1]
    if isinstance(last, UserPromptPart):
        calls = TURNS[str(last.content)]
        yield {
            index: DeltaToolCall(name=name, json_args=json.dumps(args), tool_call_id=f'turn{len(messages)}-{index}')
            for index, (name, args) in enumerate(calls)
        }
    else:
        yield 'Done'


def file_writer(workspace: Path) -> Agent[None, str]:
    """An agent whose `write_file` and `edit_file` change `workspace` as the file system's tools do, of their kinds."""
    agent = Agent(FunctionModel(stream_function=file_turns))

    @agent.tool_plain(tool_kind=WRITE_FILE_KIND)
    def write_file(path: str, content: str) -> str:
        (workspace / path).write_text(content)
        return f'Wrote {path}.'

    @agent.tool_plain(tool_kind=EDIT_FILE_KIND)
    def edit_file(path: str, old_text: str, new_text: str) -> str:
        target = workspace / path
        target.write_text(target.read_text().replace(old_text, new_text, 1))
        return f'Edited {path}.'

    return agent


@dataclass
class ChangesUI:
    url: str
    session: str


@pytest.fixture(scope='module')
def changes_ui(tmp_path_factory: pytest.TempPathFactory, serve: Serve) -> Iterator[ChangesUI]:
    root = tmp_path_factory.mktemp('browser-changes').resolve()
    with pytest.MonkeyPatch.context() as patch, start_blocking_portal() as portal:
        patch.setenv('HOME', str(root / 'home'))
        patch.setenv('XDG_CONFIG_HOME', str(root / 'config'))
        patch.setenv('XDG_STATE_HOME', str(root / 'state'))
        patch.setattr(models, 'ALLOW_MODEL_REQUESTS', False)
        fleet = Fleet(root=root, portal=portal, agents={'work': file_writer(root / 'work')})
        try:
            session = fleet.new('work')
            # Saved, so it can be renamed.
            saved: list[ModelMessage] = []
            portal.call(fleet.runtimes['work'].session.commit_messages, saved)
            assert fleet.post(f'/api/sessions/{session}/rename', {'title': 'Browser Changes'}).status_code == 204
            with serve(fleet) as url:
                yield ChangesUI(url=url, session=session)
        finally:
            fleet.exit()


async def test_changes_panel_lists_the_files_of_two_turns(page: Page, changes_ui: ChangesUI) -> None:
    await page.goto(f'{changes_ui.url}/?token={TOKEN}')
    await page.locator('.sidebar').get_by_text('Browser Changes', exact=True).click()
    await expect(page.locator('.session.current .session-title')).to_have_text('Browser Changes')
    await expect(page.locator('.session.opening')).to_have_count(0)
    composer = page.get_by_role('combobox', name='Message CLAI')
    for number, prompt in enumerate(TURNS, start=1):
        await expect(composer).to_be_enabled()
        await composer.fill(prompt)
        await composer.press('Enter')
        await expect(page.locator('.markdown').filter(has_text='Done')).to_have_count(number)
        await expect(page.locator('.topbar-status .activity')).to_have_text('ready')

    toggle = page.get_by_role('button', name='Changes: 3 files changed')
    await expect(toggle).to_be_visible()
    await expect(toggle.locator('.changes-count')).to_have_text('3')
    await toggle.click()
    panel = page.get_by_role('complementary', name='Changes')
    files = panel.get_by_role('list', name='Changed files').get_by_role('listitem')

    await panel.get_by_role('tab', name='This session').click()
    await expect(files).to_have_text(['Aa.py+3 −0', 'Ab.py+2 −0', 'Ac.py+2 −0'])

    await panel.get_by_role('tab', name='Last turn').click()
    await expect(files).to_have_text(['Ma.py+1 −0', 'Ac.py+2 −0'])
    await expect(panel.locator('.changes-summary')).to_contain_text('2 files changed')
    docstring = panel.locator('.changes-file').filter(has_text='a.py').locator('.diff-add')
    await expect(docstring).to_have_count(1)
    await expect(docstring).to_contain_text('"""Say hello."""')

    # Past the most one answer lists, the rest are counted, not listed.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(changes_module, 'MAX_FILES', 1)
        await panel.get_by_role('tab', name='This session').click()
        await panel.get_by_role('button', name='Refresh').click()
        await expect(files).to_have_text(['Aa.py+3 −0'])
        await expect(panel.locator('.changes-summary')).to_contain_text('1 file changed, 2 more not shown')

    # The workspace is not a Git repository, which the Git scopes say.
    await panel.get_by_role('tab', name='Uncommitted').click()
    await expect(panel.locator('.changes-empty')).to_have_text('This workspace is not in a Git repository.')

    # The panel takes the side panel's column, and its shortcut hides it.
    await page.keyboard.press('ControlOrMeta+Shift+E')
    await expect(panel).to_have_count(0)
    await expect(toggle).to_have_attribute('aria-pressed', 'false')
