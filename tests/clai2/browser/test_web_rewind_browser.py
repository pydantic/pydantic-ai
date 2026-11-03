"""Edit and Retry on a sent prompt in the shipped UI: the conversation goes on from that prompt, without later turns."""

from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass

import pytest
from anyio.from_thread import start_blocking_portal
from playwright.async_api import Page, expect

from pydantic_ai import Agent, models
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from tests.clai2.browser.conftest import Serve
from tests.clai2.test_web import TOKEN
from tests.clai2.test_web_sessions import Fleet


async def echo(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
    request = messages[-1]
    assert isinstance(request, ModelRequest)
    yield ' '.join(f'Echo {part.content}' for part in request.parts if isinstance(part, UserPromptPart))


@dataclass
class RewindUI:
    url: str
    session: str


@pytest.fixture(scope='module')
def rewind_ui(tmp_path_factory: pytest.TempPathFactory, serve: Serve) -> Iterator[RewindUI]:
    root = tmp_path_factory.mktemp('browser-rewind').resolve()
    with pytest.MonkeyPatch.context() as patch, start_blocking_portal() as portal:
        patch.setenv('HOME', str(root / 'home'))
        patch.setenv('XDG_CONFIG_HOME', str(root / 'config'))
        patch.setenv('XDG_STATE_HOME', str(root / 'state'))
        patch.setattr(models, 'ALLOW_MODEL_REQUESTS', False)
        fleet = Fleet(root=root, portal=portal, agents={'echo': Agent(FunctionModel(stream_function=echo))})
        try:
            session = fleet.new('echo')
            saved: list[ModelMessage] = []
            portal.call(fleet.runtimes['echo'].session.commit_messages, saved)
            assert fleet.post(f'/api/sessions/{session}/rename', {'title': 'Browser Rewind'}).status_code == 204
            with serve(fleet) as url:
                yield RewindUI(url=url, session=session)
        finally:
            fleet.exit()


async def test_edit_and_retry_go_on_from_an_earlier_prompt(page: Page, rewind_ui: RewindUI) -> None:
    await page.goto(f'{rewind_ui.url}/?token={TOKEN}')
    await page.locator('.sidebar').get_by_text('Browser Rewind', exact=True).click()
    await expect(page.locator('.session.opening')).to_have_count(0)
    composer = page.get_by_role('combobox', name='Message CLAI')
    activity = page.locator('.topbar-status .activity')
    prompts = page.locator('.user-bubble')
    answers = page.locator('.markdown')
    for prompt in ('A', 'B'):
        await expect(composer).to_be_enabled()
        await composer.fill(prompt)
        await composer.press('Enter')
        await expect(answers.filter(has_text=f'Echo {prompt}')).to_have_count(1)
        await expect(activity).to_have_text('ready')

    first = page.locator('.user-row').first
    await first.hover()
    await first.get_by_role('button', name='Edit prompt').click()
    editor = first.get_by_role('textbox', name='Edit prompt')
    await expect(editor).to_have_value('A')
    await editor.fill("A'")
    async with page.expect_response(f'**/api/sessions/{rewind_ui.session}/rewind/*') as rewound:
        await editor.press('Enter')
    assert (await rewound.value).status == 204
    await expect(answers).to_have_text(["Echo A'"])
    await expect(prompts).to_have_text(["A'"])
    await expect(activity).to_have_text('ready')

    first = page.locator('.user-row').first
    await first.hover()
    async with page.expect_response(f'**/api/sessions/{rewind_ui.session}/rewind/*') as retried:
        await first.get_by_role('button', name='Retry prompt').click()
    assert (await retried.value).status == 204
    await expect(answers).to_have_text(["Echo A'"])
    await expect(prompts).to_have_text(["A'"])
    await expect(activity).to_have_text('ready')
