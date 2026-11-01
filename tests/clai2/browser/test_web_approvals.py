"""`/permissions` in the shipped page: the mode chip, and the approval card a held command waits on."""

import json
import subprocess
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from anyio.from_thread import start_blocking_portal
from playwright.async_api import Browser, Page, async_playwright, expect

from pydantic_ai import Agent, Tool, models
from pydantic_ai.messages import ModelMessage, ToolReturnPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_clai2.builtin_plugins.web.app import create_app
from pydantic_clai2.builtin_plugins.web.server import WebServer
from tests.clai2.test_web import TOKEN
from tests.clai2.test_web_sessions import Fleet

COMMAND = 'mkdir -p kept && rm -rf doomed'


async def remove(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
    """Run `COMMAND` for each prompt, then report what the tool returned."""
    last = messages[-1].parts[-1]
    if isinstance(last, UserPromptPart):
        arguments = json.dumps({'command': COMMAND})
        yield {0: DeltaToolCall(name='run_command', json_args=arguments, tool_call_id=f'call{len(messages)}')}
    else:
        assert isinstance(last, ToolReturnPart)
        yield f'Tool said: {last.content}'


def shell_tool(folder: Path) -> Tool[None]:
    """A shell tool in `folder`, declared as harness `Shell` declares its own, so `Permissions` recognizes it."""

    def run_command(command: str) -> str:
        subprocess.run(['sh', '-c', command], cwd=folder, check=True)
        return f'ran {command}'

    return Tool(run_command, metadata={'code_arg_name': 'command', 'code_arg_language': 'shell'})


@dataclass
class Approvals:
    url: str
    session: str
    folder: Path


@pytest.fixture(scope='module')
def approvals(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Approvals]:
    root = tmp_path_factory.mktemp('browser-approvals')
    folder = (root / 'held').resolve()
    with pytest.MonkeyPatch.context() as patch, start_blocking_portal() as portal:
        patch.setenv('HOME', str(root / 'home'))
        patch.setenv('XDG_CONFIG_HOME', str(root / 'config'))
        patch.setenv('XDG_STATE_HOME', str(root / 'state'))
        patch.setattr(models, 'ALLOW_MODEL_REQUESTS', False)
        fleet = Fleet(
            root=root,
            portal=portal,
            agents={
                'held': Agent(FunctionModel(stream_function=remove), deps_type=type(None), tools=[shell_tool(folder)])
            },
        )
        try:
            session = fleet.new('held')
            portal.call(fleet.runtimes['held'].session.commit_messages, list[ModelMessage]())
            response = fleet.post(f'/api/sessions/{session}/rename', {'title': 'Browser Approvals'})
            assert response.status_code == 204, response.text
            # The model calls `run_command` itself, not from inside code mode's `run_code`.
            response = fleet.post(f'/api/sessions/{session}/switch', {'key': 'run.code_mode', 'on': False})
            assert response.status_code == 204, response.text
            server = portal.call(
                lambda: WebServer.serve(
                    lambda port: create_app(service=fleet.service, token=TOKEN, hosts=frozenset({f'127.0.0.1:{port}'})),
                    host='127.0.0.1',
                    port=0,
                )
            )
            try:
                yield Approvals(url=f'http://127.0.0.1:{server.port}', session=session, folder=folder)
            finally:
                portal.call(server.stop)
        finally:
            fleet.exit()


@pytest.fixture(scope='module')
async def chromium() -> AsyncIterator[Browser]:
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch()
        try:
            yield browser
        finally:
            await browser.close()


@pytest.fixture
async def page(chromium: Browser) -> AsyncIterator[Page]:
    async with await chromium.new_context() as context:
        page = await context.new_page()
        page.set_default_timeout(10_000)
        yield page


async def send(page: Page, approvals: Approvals, text: str) -> None:
    composer = page.get_by_role('combobox', name='Message CLAI')
    await expect(composer).to_be_enabled()
    await composer.fill(text)
    async with page.expect_response(f'**/api/sessions/{approvals.session}/input') as sent:
        await composer.press('Enter')
    assert (await sent.value).ok


async def test_auto_mode_holds_rm_rf_for_approval(page: Page, approvals: Approvals) -> None:
    doomed = approvals.folder / 'doomed'
    doomed.mkdir()
    await page.goto(f'{approvals.url}/?token={TOKEN}')
    await page.locator('.sidebar').get_by_text('Browser Approvals', exact=True).click()
    await expect(page.locator('.session.current .session-title')).to_have_text('Browser Approvals')

    mode = page.get_by_label('Permissions')
    await expect(mode).to_have_value('full')
    await mode.select_option('auto')
    await expect(mode).to_have_value('auto')

    await send(page, approvals, 'Clean up')
    card = page.locator('.approval-card')
    await expect(card).to_be_visible()
    await expect(card.locator('.approval-subject')).to_have_text(COMMAND)
    await expect(card).to_contain_text('run_command')
    await card.get_by_label('Reason for denying').fill('keep it for now')
    await card.get_by_role('button', name='Deny', exact=True).click()
    await expect(card).to_have_count(0)
    await expect(
        page.get_by_text('Tool said: The user denied this command, so it did not run. Their reason: keep it for now')
    ).to_be_visible()
    assert doomed.is_dir()
    assert not (approvals.folder / 'kept').exists()

    await send(page, approvals, 'Clean up again')
    await expect(card).to_be_visible()
    await card.get_by_role('button', name='Approve', exact=True).click()
    await expect(page.get_by_text(f'Tool said: ran {COMMAND}')).to_be_visible()
    assert not doomed.exists()
    assert (approvals.folder / 'kept').is_dir()
