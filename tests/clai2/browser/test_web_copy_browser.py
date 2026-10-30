"""Copy message source through the real Web UI, independently of the server's clipboard."""

from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from io import BytesIO

import anyio
import pytest
from anyio.from_thread import BlockingPortal, start_blocking_portal
from PIL import Image
from playwright.async_api import Page, expect

from pydantic_ai import Agent, BinaryContent, models
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from tests.clai2.browser.conftest import Serve
from tests.clai2.test_web import TOKEN
from tests.clai2.test_web_sessions import Fleet

PROMPT = 'Copy this prompt: 中文\n\n`literal code` and [link](https://example.com)'
ANSWER = 'A **reply** with a [link](https://example.com).\n\n```python\nprint("hello")\n```'


@dataclass
class CopyUI:
    url: str
    portal: BlockingPortal
    release: anyio.Event


@pytest.fixture(scope='module')
def copy_ui(tmp_path_factory: pytest.TempPathFactory, serve: Serve) -> Iterator[CopyUI]:
    root = tmp_path_factory.mktemp('browser-copy')
    with pytest.MonkeyPatch.context() as patch, start_blocking_portal() as portal:
        patch.setenv('HOME', str(root / 'home'))
        patch.setenv('XDG_CONFIG_HOME', str(root / 'config'))
        patch.setenv('XDG_STATE_HOME', str(root / 'state'))
        patch.setattr(models, 'ALLOW_MODEL_REQUESTS', False)
        release = portal.call(anyio.Event)

        async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
            yield 'Streaming'
            await release.wait()
            yield '\n\n' + ANSWER

        fleet = Fleet(root=root, portal=portal, agents={'copy': Agent(FunctionModel(stream_function=stream))})
        try:
            fleet.new('copy')
            png = BytesIO()
            Image.new('RGB', (1, 1)).save(png, format='PNG')
            image = BinaryContent(data=png.getvalue(), media_type='image/png')
            history: list[ModelMessage] = [
                ModelRequest(parts=[UserPromptPart(content=[PROMPT, image])]),
                ModelResponse(
                    parts=[ThinkingPart(content='Do not copy thinking'), ToolCallPart('read_file', {}, 'read')]
                ),
                ModelRequest(parts=[ToolReturnPart('read_file', 'Do not copy tool output', 'read')]),
                ModelResponse(parts=[TextPart(content=ANSWER)]),
                ModelRequest(parts=[UserPromptPart(content=[image])]),
                ModelResponse(parts=[TextPart(content=' ')]),
            ]
            portal.call(fleet.runtimes['copy'].session.commit_messages, history)
            with serve(fleet) as url:
                yield CopyUI(url=url, portal=portal, release=release)
        finally:
            portal.call(release.set)
            fleet.exit()


async def test_copy_message_source_and_history(page: Page, copy_ui: CopyUI) -> None:
    await page.context.grant_permissions(['clipboard-read', 'clipboard-write'])
    await page.goto(f'{copy_ui.url}/?token={TOKEN}')
    user = page.locator('.user-row').filter(has_text=PROMPT)
    assistant = page.locator('.assistant-message').filter(has_text='A reply').first
    for row, expected in ((user, PROMPT), (assistant, ANSWER)):
        button = row.locator('.message-copy')
        await button.click()
        await expect(button).to_have_text('Copied')
        assert await page.evaluate('navigator.clipboard.readText()') == expected
    # The image-only prompt and blank reply have no copy action, nor does thinking.
    await expect(page.locator('.message-copy')).to_have_count(2)
    await assistant.locator('.code-head .copy').click()
    assert await page.evaluate('navigator.clipboard.readText()') == 'print("hello")'
    await page.reload()
    await user.locator('.message-copy').click()
    assert await page.evaluate('navigator.clipboard.readText()') == PROMPT
    await assistant.locator('.message-copy').click()
    assert await page.evaluate('navigator.clipboard.readText()') == ANSWER


@pytest.mark.parametrize('unavailable', [False, True])
async def test_copy_failure_feedback(page: Page, copy_ui: CopyUI, unavailable: bool) -> None:
    await page.goto(f'{copy_ui.url}/?token={TOKEN}')
    await page.evaluate(
        """unavailable => Object.defineProperty(navigator, 'clipboard', {value: unavailable ? undefined : {
            writeText: () => Promise.reject(new DOMException('Permission denied', 'NotAllowedError'))
        }})""",
        unavailable,
    )
    await page.clock.install()
    for selector in ('.user-row .message-copy', '.assistant-message .message-copy', '.code-head .copy'):
        button = page.locator(selector).first
        await button.click()
        await expect(button).to_have_text('Copy failed')
        await page.clock.run_for(1200)
        await expect(button).to_have_text('Copy')


async def test_copy_waits_for_stream_to_finish(page: Page, copy_ui: CopyUI) -> None:
    await page.context.grant_permissions(['clipboard-read', 'clipboard-write'])
    await page.goto(f'{copy_ui.url}/?token={TOKEN}')
    composer = page.get_by_role('combobox', name='Message CLAI')
    await expect(composer).to_be_enabled()
    await composer.fill('Stream a reply')
    await composer.press('Enter')
    assistant = page.locator('.assistant-message').filter(has_text='Streaming')
    await expect(assistant.locator('.streaming')).to_be_visible()
    await expect(assistant.locator('.message-copy')).to_have_count(0)
    # A previous message stays copyable while a later reply streams.
    await page.locator('.user-row').filter(has_text=PROMPT).locator('.message-copy').click()
    assert await page.evaluate('navigator.clipboard.readText()') == PROMPT
    copy_ui.portal.call(copy_ui.release.set)
    button = assistant.locator('.message-copy')
    await button.click()
    await expect(button).to_have_text('Copied')
    assert await page.evaluate('navigator.clipboard.readText()') == 'Streaming\n\n' + ANSWER
