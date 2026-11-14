"""A running shell command's output in the shipped UI, live in its card, as code mode runs it."""

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
from pydantic_ai_harness.shell import Shell
from tests.clai2.browser.conftest import Serve
from tests.clai2.test_web import TOKEN
from tests.clai2.test_web_sessions import Fleet

COMMAND = 'for i in $(seq 10); do echo tick $i; done; while [ ! -e released ]; do sleep 0.05; done'
"""Prints ten lines, then runs until the test creates `released`."""


async def shell_cell(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
    if isinstance(messages[-1].parts[-1], UserPromptPart):
        cell = json.dumps({'code': f'await shell(command={COMMAND!r})'})
        yield {0: DeltaToolCall(name='run_code', json_args=cell, tool_call_id='cell')}
    else:
        yield 'Done'


@dataclass
class ShellUI:
    url: str
    workspace: Path


@pytest.fixture(scope='module')
def shell_ui(tmp_path_factory: pytest.TempPathFactory, serve: Serve) -> Iterator[ShellUI]:
    root = tmp_path_factory.mktemp('browser-shell').resolve()
    with pytest.MonkeyPatch.context() as patch, start_blocking_portal() as portal:
        patch.setenv('HOME', str(root / 'home'))
        patch.setenv('XDG_CONFIG_HOME', str(root / 'config'))
        patch.setenv('XDG_STATE_HOME', str(root / 'state'))
        patch.setattr(models, 'ALLOW_MODEL_REQUESTS', False)
        agent = Agent(FunctionModel(stream_function=shell_cell), capabilities=[Shell(tools=['shell'])])
        fleet = Fleet(root=root, portal=portal, agents={'work': agent})
        try:
            session = fleet.new('work')
            # Saved, so it can be renamed.
            saved: list[ModelMessage] = []
            portal.call(fleet.runtimes['work'].session.commit_messages, saved)
            assert fleet.post(f'/api/sessions/{session}/rename', {'title': 'Browser Shell'}).status_code == 204
            with serve(fleet) as url:
                yield ShellUI(url=url, workspace=fleet.folder('work'))
        finally:
            (fleet.folder('work') / 'released').touch()
            fleet.exit()


async def test_a_running_command_in_a_cell_shows_its_output(page: Page, shell_ui: ShellUI) -> None:
    await page.goto(f'{shell_ui.url}/?token={TOKEN}')
    await page.locator('.sidebar').get_by_text('Browser Shell', exact=True).click()
    await expect(page.locator('.session.current .session-title')).to_have_text('Browser Shell')
    await expect(page.locator('.session.opening')).to_have_count(0)
    composer = page.get_by_role('combobox', name='Message CLAI')
    await expect(composer).to_be_enabled()
    await composer.fill('Run the ticks')
    await composer.press('Enter')

    card = page.locator('.tool').filter(has_text='seq 10')
    # The collapsed card shows the last lines of output so far while the command runs.
    await expect(card.locator('pre')).to_have_text('\n'.join(f'tick {i}' for i in range(3, 11)))
    await expect(card.locator('.verb')).to_have_text('Running')
    (shell_ui.workspace / 'released').touch()
    await expect(page.locator('.markdown').filter(has_text='Done')).to_be_visible()
    # Finished, the card folds again.
    await expect(card.locator('pre')).to_have_count(0)
