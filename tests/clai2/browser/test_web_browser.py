"""Exercise the shipped UI and its real HTTP/WebSocket boundary without provider requests."""

import asyncio
import json
import os
import re
import subprocess
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from anyio.from_thread import start_blocking_portal
from playwright.async_api import Dialog, Page, Route, expect

from pydantic_ai import Agent, ModelRetry, models
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.step_persistence.conversations import ConversationSummary
from pydantic_clai2.builtin_plugins.web.app import HEADER
from tests.clai2.browser.conftest import Serve
from tests.clai2.test_web import TOKEN
from tests.clai2.test_web_sessions import Fleet

HOSTILE = """Markdown survived

<script>window.alert('script')</script>
<img src="/missing" onerror="window.alert('image')">
<svg onload="window.alert('svg')"><a href="javascript:alert('svg-link')">SVG link</a></svg>
<iframe srcdoc="<script>alert('frame')</script>"></iframe>
<style>body {display:none}</style><form><input autofocus onfocus="alert('form')"></form>

[JavaScript link](javascript:alert%281%29)
[Data link](data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg==)
<a href="javascript:alert(1)" onclick="alert('click')">Raw link</a>
[Safe external](https://example.invalid/guide)
"""

LINES = """One per line
2
3

```
code a
code b
```
"""

LONG_LINES = 200
PREVIEW_LINES = 8
"""`PREVIEW_LINES` in `web-ui/src/components/ToolCard.tsx`: the lines each diff mounts until asked for more."""

WRITE: list[ModelMessage] = [
    ModelRequest(parts=[UserPromptPart(content='Write notes.py')]),
    ModelResponse(
        parts=[ToolCallPart('write_file', {'path': 'notes.py', 'content': 'first line\nsecond line\n'}, 'write-1')]
    ),
    ModelRequest(parts=[ToolReturnPart('write_file', 'Wrote 23 chars (2 lines) to notes.py.', 'write-1')]),
    ModelResponse(
        parts=[
            ToolCallPart(
                'write_file',
                {'path': 'long.py', 'content': '\n'.join(f'line_{n} = {n}' for n in range(LONG_LINES))},
                'write-2',
            )
        ]
    ),
    ModelRequest(parts=[ToolReturnPart('write_file', 'Wrote long.py.', 'write-2')]),
]


WRITTEN = '\n'.join(f'line_{number} = {number}' for number in range(1, 31))


async def write_forbidden(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
    if len(messages) == 1:
        arguments = json.dumps({'path': '/proc/forbidden.py', 'content': WRITTEN})
        yield {0: DeltaToolCall(name='write_file', json_args=arguments, tool_call_id='write-1')}
    else:
        yield 'Could not write it'


def forbidden_writer() -> Agent[None, str]:
    agent = Agent(FunctionModel(stream_function=write_forbidden))

    @agent.tool_plain
    def write_file(path: str, content: str) -> str:
        raise ModelRetry(f'Permission denied: {path}')

    return agent


RUNNING = "Exception: Task '899ff4ce' is still running"
RESUME_RUNNING = (
    "await delegate_task(agent_name='self', task='Report back on the long background task', "
    "resume='899ff4ce', background=False)"
)
FAILING_CELLS = {
    'Resume the task': RESUME_RUNNING,
    'Check, then resume the task': f"print('checking the task')\n{RESUME_RUNNING}",
    'Run broken code': 'x = (1,',
}
"""Each prompt to the `delta` session, and the `run_code` cell its model writes for it."""


async def failing_cells(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
    """Code mode's `run_code` fails on a prompt's first request, as resuming a still-running task does."""
    prompts = [part.content for message in messages for part in message.parts if isinstance(part, UserPromptPart)]
    prompt = str(prompts[-1])
    if isinstance(messages[-1].parts[-1], UserPromptPart):
        # A fresh session type-checks the cell, so broken code fails the same way whichever prompt came before.
        cell = json.dumps({'code': FAILING_CELLS[prompt], 'restart': True})
        yield {0: DeltaToolCall(name='run_code', json_args=cell, tool_call_id=f'cell{len(prompts)}')}
    else:
        yield f'Answered: {prompt}'


def failing_agent() -> Agent[None, str]:
    agent = Agent(FunctionModel(stream_function=failing_cells))

    @agent.tool_plain
    async def delegate_task(agent_name: str, task: str, resume: str, background: bool) -> str:
        raise Exception(f"Task '{resume}' is still running")

    return agent


@dataclass
class WebUI:
    url: str
    alpha: str
    beta: str
    lines: str
    gamma: str
    delta: str
    epsilon: str
    busy: str
    """A session another CLAI runs: this test process's parent."""
    removed: str
    """Saved in a folder that is gone."""
    saved: str
    """Saved in Gamma's folder, so it opens while Gamma is the default session, and not live until a test opens it."""
    moved: str
    """Saved in Gamma's folder too, and never opened: its tests pretend the folder went."""
    fleet: Fleet


@pytest.fixture(scope='module')
def web_ui(tmp_path_factory: pytest.TempPathFactory, serve: Serve) -> Iterator[WebUI]:
    """Share one process, server, and its live sessions across browser cases."""
    root = tmp_path_factory.mktemp('browser-ui')
    with pytest.MonkeyPatch.context() as patch, start_blocking_portal() as portal:
        patch.setenv('HOME', str(root / 'home'))
        patch.setenv('XDG_CONFIG_HOME', str(root / 'config'))
        patch.setenv('XDG_STATE_HOME', str(root / 'state'))
        patch.setattr(models, 'ALLOW_MODEL_REQUESTS', False)
        fleet = Fleet(
            root=root,
            portal=portal,
            agents={
                'alpha': Agent(TestModel(call_tools=[], custom_output_text=HOSTILE)),
                'beta': Agent(TestModel(call_tools=[], custom_output_text='Beta answer')),
                'lines': Agent(TestModel(call_tools=[], custom_output_text=LINES)),
                'gamma': Agent(TestModel(call_tools=[])),
                'delta': failing_agent(),
                'epsilon': forbidden_writer(),
            },
        )
        try:
            alpha = fleet.new('alpha')
            beta = fleet.new('beta')
            lines = fleet.new('lines')
            gamma = fleet.new('gamma')
            delta = fleet.new('delta')
            epsilon = fleet.new('epsilon')
            for name, conversation, title, messages in (
                ('alpha', alpha, 'Browser Alpha', []),
                ('beta', beta, 'Browser Beta', []),
                ('lines', lines, 'Browser Lines', []),
                ('gamma', gamma, 'Browser Gamma', WRITE),
                ('delta', delta, 'Browser Delta', []),
                ('epsilon', epsilon, 'Browser Epsilon', []),
            ):
                portal.call(fleet.runtimes[name].session.commit_messages, messages)
                response = fleet.post(f'/api/sessions/{conversation}/rename', {'title': title})
                assert response.status_code == 204, response.text
            busy = portal.call(
                lambda: fleet.launch.sessions.store.save(
                    summary=ConversationSummary(
                        workspace=str(fleet.folder('beta')),
                        title='Browser Busy',
                        outcome='running',
                        owner_pid=os.getppid(),
                    ),
                    messages=[],
                )
            )
            removed = portal.call(
                lambda: fleet.launch.sessions.store.save(
                    summary=ConversationSummary(workspace=str(root / 'gone'), title='Browser Removed'), messages=[]
                )
            )
            summary = ConversationSummary(workspace=str(fleet.folder('gamma')), title='Browser Saved')
            history: list[ModelMessage] = [
                ModelRequest(parts=[UserPromptPart(content='Saved prompt')]),
                ModelResponse(parts=[TextPart(content='Saved answer')]),
            ]
            saved = portal.call(lambda: fleet.launch.process.conversations.save(summary=summary, messages=history))
            moved_summary = ConversationSummary(workspace=str(fleet.folder('gamma')), title='Browser Moved')
            moved = portal.call(lambda: fleet.launch.process.conversations.save(summary=moved_summary, messages=[]))
            with serve(fleet) as url:
                yield WebUI(
                    url=url,
                    alpha=alpha,
                    beta=beta,
                    lines=lines,
                    gamma=gamma,
                    delta=delta,
                    epsilon=epsilon,
                    busy=busy.id,
                    removed=removed.id,
                    saved=saved.id,
                    moved=moved.id,
                    fleet=fleet,
                )
        finally:
            fleet.exit()


async def test_hostile_markdown_and_external_links(page: Page, web_ui: WebUI) -> None:
    dialogs: list[str] = []

    async def dismiss_dialog(dialog: Dialog) -> None:
        dialogs.append(dialog.message)
        await dialog.dismiss()

    page.on('dialog', dismiss_dialog)
    await page.goto(f'{web_ui.url}/?token={TOKEN}')
    await page.locator('.sidebar').get_by_text('Browser Alpha', exact=True).click()
    await expect(page.locator('.session.current .session-title')).to_have_text('Browser Alpha')
    await expect(page.locator('.session[aria-busy="true"]')).to_have_count(0)
    composer = page.get_by_role('combobox', name='Message CLAI')
    await expect(composer).to_be_enabled()
    await composer.fill('Render the hostile example')
    async with page.expect_response(f'**/api/sessions/{web_ui.alpha}/input') as sent:
        await composer.press('Enter')
    assert (await sent.value).ok
    markdown = page.locator('.markdown').filter(has_text='Markdown survived').last
    await expect(markdown).to_be_visible()
    assert (
        await markdown.locator(
            'script, iframe, style, form, input, [onclick], [onerror], [onload], [onfocus], '
            'a[href^="javascript:"], a[href^="data:"]'
        ).count()
        == 0
    )
    assert dialogs == []

    link = markdown.get_by_role('link', name='Safe external', exact=True)
    await expect(link).to_have_attribute('target', '_blank')
    await expect(link).to_have_attribute('rel', 'noreferrer noopener')
    requests: list[dict[str, str]] = []

    async def external(route: Route) -> None:
        requests.append(route.request.headers)
        assert TOKEN not in route.request.url
        await route.fulfill(status=200, content_type='text/html', body='<h1>External page</h1>')

    await page.context.route('https://example.invalid/**', external)
    async with page.expect_popup() as popup:
        await link.click()
    opened = await popup.value
    await opened.wait_for_load_state()
    assert await opened.evaluate('window.opener === null')
    assert len(requests) == 1
    assert 'referer' not in requests[0]
    assert 'x-clai-token' not in requests[0]


async def test_single_line_breaks_survive_like_the_tui(page: Page, web_ui: WebUI) -> None:
    await page.goto(f'{web_ui.url}/?token={TOKEN}')
    await page.locator('.sidebar').get_by_text('Browser Lines', exact=True).click()
    await expect(page.locator('.session.current .session-title')).to_have_text('Browser Lines')
    await expect(page.locator('.session[aria-busy="true"]')).to_have_count(0)
    composer = page.get_by_role('combobox', name='Message CLAI')
    await expect(composer).to_be_enabled()
    await composer.fill('Print one item per line')
    async with page.expect_response(f'**/api/sessions/{web_ui.lines}/input') as sent:
        await composer.press('Enter')
    assert (await sent.value).ok
    markdown = page.locator('.markdown').filter(has_text='One per line').last
    await expect(markdown).to_be_visible()
    assert await markdown.locator('p').inner_text() == 'One per line\n2\n3'
    assert await markdown.locator('pre code').inner_text() == 'code a\ncode b'
    assert await markdown.locator('pre br').count() == 0


async def test_browser_token_host_and_websocket_origin(page: Page, web_ui: WebUI) -> None:
    for path in ('/', '/?token=wrong', '/api/state?token=' + TOKEN):
        response = await page.goto(web_ui.url + path)
        assert response is not None and response.status == 403
    response = await page.goto(web_ui.url + '/assets/app.js')
    assert response is not None and response.status == 200, 'the public build needs no token'

    await page.goto(f'{web_ui.url}/?token={TOKEN}')
    await expect(page.get_by_title('Connected', exact=True)).to_be_visible()
    assert await page.evaluate("fetch('/api/state').then(response => response.status)") == 403
    assert (
        await page.evaluate(
            "fetch('/api/state', {headers: {'x-clai-token': 'wrong'}}).then(response => response.status)"
        )
        == 403
    )
    for suffix in ('', '?token=wrong'):
        assert (
            await page.evaluate(
                """suffix => new Promise(resolve => {
              const socket = new WebSocket(`ws://${location.host}/ws${suffix}`);
              socket.onopen = () => { socket.close(); resolve('opened'); };
              socket.onclose = () => resolve('refused');
            })""",
                suffix,
            )
            == 'refused'
        )

    # The same loopback server reached by another hostname supplies a real foreign origin.
    response = await page.goto(web_ui.url.replace('127.0.0.1', 'localhost') + f'/?token={TOKEN}')
    assert response is not None and response.status == 403
    assert (
        await page.evaluate(
            """url => new Promise(resolve => {
          const socket = new WebSocket(url);
          socket.onopen = () => { socket.close(); resolve('opened'); };
          socket.onclose = () => resolve('refused');
        })""",
            web_ui.url.replace('http:', 'ws:') + '/ws?token=' + TOKEN,
        )
        == 'refused'
    )


async def test_sidebar_search_accessible_name(page: Page, web_ui: WebUI) -> None:
    await page.goto(f'{web_ui.url}/?token={TOKEN}')
    sidebar = page.locator('.sidebar')
    search = sidebar.get_by_role('textbox', name='Search sessions', exact=True)
    await expect(search).to_be_visible()
    await expect(sidebar.get_by_role('button', name='Archived', exact=True)).to_be_visible()
    await search.fill('Beta')
    await expect(sidebar.get_by_text('Browser Beta', exact=True)).to_be_visible()
    await expect(sidebar.get_by_text('Browser Alpha', exact=True)).to_have_count(0)


async def test_removed_worktrees_fold_after_live_ones_and_archive_as_a_group(page: Page, web_ui: WebUI) -> None:
    """Sessions of removed worktrees wait, closed, in one group after the live ones; a branch archives at once."""

    def sidebar_session(session_id: str, title: str) -> dict[str, object]:
        return {
            'id': session_id,
            'title': title,
            'workspace': f'/wt/{session_id}',
            'updated_at': '2026-01-01T00:00:00Z',
            'created_at': '2026-01-01T00:00:00Z',
            'state': 'ready',
            'pid': None,
            'messages': 2,
            'model': None,
            'current': False,
            'archived': False,
            'openable': True,
        }

    worktrees = [
        {
            'root': '/wt/live',
            'label': 'live',
            'current': False,
            'missing': False,
            'sessions': [sidebar_session('l', 'Live work')],
        },
        {
            'root': '/wt/gone',
            'label': 'gone',
            'current': False,
            'missing': True,
            'sessions': [sidebar_session('g1', 'Gone one'), sidebar_session('g2', 'Gone two')],
        },
    ]

    async def sessions(route: Route) -> None:
        await route.fulfill(json=[{'key': 'repo', 'name': 'repo', 'current': True, 'worktrees': worktrees}])

    async def archive(route: Route) -> None:
        await route.fulfill(status=204)

    await page.route('**/api/sessions?*', sessions)
    await page.route('**/api/sessions/archive', archive)
    await page.goto(f'{web_ui.url}/?token={TOKEN}')
    sidebar = page.locator('.sidebar')
    await sidebar.get_by_role('button', name='Projects', exact=True).click()
    await expect(sidebar.get_by_text('Live work', exact=True)).to_be_visible()
    fold = sidebar.get_by_role('button', name='Removed worktrees (2)')
    await expect(fold).to_be_visible()
    await expect(sidebar.get_by_text('Gone one', exact=True)).to_have_count(0)
    await fold.click()
    gone = sidebar.locator('.session').filter(has_text='Gone one')
    await expect(gone).to_be_visible()
    await gone.hover()
    await gone.get_by_role('button', name='Session actions').click()
    async with page.expect_request('**/api/sessions/archive') as request:
        await sidebar.get_by_role('button', name='Archive all in gone').click()
    assert (await request.value).post_data_json == {'ids': ['g1', 'g2']}
    # A search lists only a branch's matching sessions, so it cannot archive all of them.
    await sidebar.get_by_role('textbox', name='Search sessions', exact=True).fill('Gone one')
    await gone.hover()
    await gone.get_by_role('button', name='Session actions').click()
    await expect(sidebar.get_by_role('button', name='Archive all in gone')).to_be_disabled()


async def test_session_switch_and_reconnect(page: Page, web_ui: WebUI) -> None:
    # Capture real browser sockets while leaving their transport and handlers intact.
    await page.add_init_script("""
      window.claiTestSockets = [];
      const OriginalWebSocket = window.WebSocket;
      window.WebSocket = class extends OriginalWebSocket {
        constructor(...args) { super(...args); window.claiTestSockets.push(this); }
      };
    """)
    await page.goto(f'{web_ui.url}/?token={TOKEN}')
    await page.locator('.sidebar').get_by_text('Browser Beta', exact=True).click()
    await expect(page.locator('.session.current .session-title')).to_have_text('Browser Beta')
    await expect(page.locator('.session[aria-busy="true"]')).to_have_count(0)
    composer = page.get_by_role('combobox', name='Message CLAI')
    await expect(composer).to_be_enabled()
    await composer.fill('Only Beta should receive this')
    async with page.expect_response(f'**/api/sessions/{web_ui.beta}/input') as sent:
        await composer.press('Enter')
    assert (await sent.value).ok
    await expect(page.locator('.markdown').filter(has_text='Beta answer')).to_be_visible()
    await page.locator('.sidebar').get_by_text('Browser Alpha', exact=True).click()
    await expect(page.locator('.session.current .session-title')).to_have_text('Browser Alpha')
    await expect(page.locator('.session[aria-busy="true"]')).to_have_count(0)
    await expect(composer).to_be_enabled()
    await expect(page.locator('.markdown').filter(has_text='Beta answer', visible=True)).to_have_count(0)
    await page.clock.install()
    await page.evaluate('window.claiTestSockets[0].close()')
    await expect(page.get_by_title('Reconnecting…', exact=True)).to_be_visible()
    await page.clock.run_for(500)
    await page.clock.resume()
    await expect(page.get_by_title('Connected', exact=True)).to_be_visible()
    assert await page.evaluate('window.claiTestSockets.length') == 2
    await composer.fill('Alpha after reconnect')
    async with page.expect_request(f'**/api/sessions/{web_ui.alpha}/input') as request:
        await composer.press('Enter')
    assert (await request.value).post_data_json == {'text': 'Alpha after reconnect'}
    await expect(page.locator('.markdown').filter(has_text='Markdown survived').last).to_be_visible()
    await expect(page.locator('.markdown').filter(has_text='Beta answer', visible=True)).to_have_count(0)


async def test_file_change_row_folds_its_diff(page: Page, web_ui: WebUI) -> None:
    await page.goto(f'{web_ui.url}/?token={TOKEN}')
    await page.locator('.sidebar').get_by_text('Browser Gamma', exact=True).click()
    await expect(page.locator('.session.current .session-title')).to_have_text('Browser Gamma')
    row = page.get_by_role('button', name='Wrote notes.py')
    card = page.locator('.tool').filter(has=row)
    diff = card.get_by_text('second line')
    result = card.get_by_text('Wrote 23 chars (2 lines) to notes.py.')
    await expect(row).to_have_attribute('aria-expanded', 'true')
    await expect(diff).to_be_visible()
    await expect(result).to_be_visible()
    await row.click()
    await expect(row).to_have_attribute('aria-expanded', 'false')
    await expect(diff).to_have_count(0)
    await expect(result).to_have_count(0)
    await row.click()
    await expect(row).to_have_attribute('aria-expanded', 'true')
    await expect(diff).to_be_visible()


async def test_composer_recalls_sent_prompts(page: Page, web_ui: WebUI) -> None:
    await page.goto(f'{web_ui.url}/?token={TOKEN}')
    await page.locator('.sidebar').get_by_text('Browser Beta', exact=True).click()
    await expect(page.locator('.session.current .session-title')).to_have_text('Browser Beta')
    await expect(page.locator('.session[aria-busy="true"]')).to_have_count(0)
    composer = page.get_by_role('combobox', name='Message CLAI')
    await expect(composer).to_be_enabled()
    for prompt in ('Recall the first prompt', 'Recall the second prompt\non two lines'):
        await composer.fill(prompt)
        async with page.expect_response(f'**/api/sessions/{web_ui.beta}/input') as sent:
            await composer.press('Enter')
        assert (await sent.value).ok
        await expect(page.locator('.user-bubble').filter(has_text=prompt.split('\n')[0]).last).to_be_visible()
        await expect(composer).to_have_value('')

    # As in the terminal: Up recalls the newest prompt, moves within its lines first, and Down returns to the draft.
    await composer.fill('unsent draft')
    await composer.press('Shift+ArrowUp')
    await expect(composer).to_have_value('unsent draft')
    await composer.press('End')
    await composer.press('ArrowUp')
    await expect(composer).to_have_value('Recall the second prompt\non two lines')
    await composer.press('ArrowUp')
    await expect(composer).to_have_value('Recall the second prompt\non two lines')
    assert await composer.evaluate('area => area.value.slice(0, area.selectionStart).includes("\\n")') is False
    await composer.press('ArrowUp')
    await expect(composer).to_have_value('Recall the first prompt')
    await composer.press('ArrowDown')
    await expect(composer).to_have_value('Recall the second prompt\non two lines')
    await composer.press('ArrowDown')
    await expect(composer).to_have_value('unsent draft')

    # With command suggestions open, the arrows still pick a suggestion instead of recalling a prompt.
    await composer.fill('/')
    suggestions = page.get_by_role('listbox').get_by_role('option')
    await expect(suggestions.first).to_have_attribute('aria-selected', 'true')
    await composer.press('ArrowDown')
    await expect(suggestions.nth(1)).to_have_attribute('aria-selected', 'true')
    await composer.press('ArrowUp')
    await expect(suggestions.first).to_have_attribute('aria-selected', 'true')
    await expect(composer).to_have_value('/')

    # Completions that arrive after a recall are for the draft it replaced, so they never show over the recalled prompt.
    await composer.fill('')
    await expect(suggestions).to_have_count(0)
    held = asyncio.Event()

    async def hold(route: Route) -> None:
        await held.wait()
        await route.continue_()

    await page.route('**/complete?*', hold)
    async with page.expect_response('**/complete?*') as completed:
        await composer.fill('/')
        await composer.press('ArrowUp')
        await expect(composer).to_have_value('Recall the second prompt\non two lines')
        held.set()
    await (await completed.value).body()
    await page.unroute('**/complete?*', hold)
    # Give the page a task to render what that response set: without the fix, the stale suggestions show by then.
    await page.evaluate('new Promise(resolve => setTimeout(resolve, 50))')
    await expect(suggestions).to_have_count(0)


@pytest.mark.parametrize(
    ('prompt', 'summary', 'traceback', 'rest'),
    [
        pytest.param('Resume the task', RUNNING, True, None, id='traceback'),
        pytest.param(
            'Check, then resume the task',
            RUNNING,
            True,
            '[stdout before error]\nchecking the task\n[/stdout before error]',
            id='printed-then-traceback',
        ),
        pytest.param(
            'Run broken code', 'error[invalid-syntax]: unexpected EOF while parsing', False, None, id='no-traceback'
        ),
    ],
)
async def test_a_failed_tool_leads_with_its_error(
    page: Page, web_ui: WebUI, prompt: str, summary: str, traceback: bool, rest: str | None
) -> None:
    await page.goto(f'{web_ui.url}/?token={TOKEN}')
    await page.locator('.sidebar').get_by_text('Browser Delta', exact=True).click()
    await expect(page.locator('.session.current .session-title')).to_have_text('Browser Delta')
    await expect(page.locator('.session[aria-busy="true"]')).to_have_count(0)
    composer = page.get_by_role('combobox', name='Message CLAI')
    await expect(composer).to_be_enabled()
    await composer.fill(prompt)
    async with page.expect_response(f'**/api/sessions/{web_ui.delta}/input') as sent:
        await composer.press('Enter')
    assert (await sent.value).ok
    await expect(page.locator('.markdown').filter(has_text=f'Answered: {prompt}')).to_be_visible()

    tool = page.locator('.tool.error').last
    row = tool.locator('.tool-row')
    await expect(row.locator('.badge.error')).to_have_text('failed')
    # The collapsed row says why, not only that it failed.
    await expect(row.locator('.error-summary')).to_have_text(summary)

    await row.click()
    body = tool.locator('.tool-body')
    # The exception leads the body.
    await expect(body.locator(':scope > *').first).to_contain_text(summary)
    folded = body.locator('details.traceback')
    if traceback:
        # The traceback is folded, without its `~~~` underline, and holds only the traceback.
        await expect(folded).not_to_have_attribute('open', '')
        await folded.locator('summary').click()
        await expect(folded.locator('pre')).to_contain_text('Traceback (most recent call last):')
        await expect(folded.locator('pre')).not_to_contain_text('~~~')
        await expect(folded.locator('pre')).not_to_contain_text('checking the task')
    else:
        await expect(folded).to_have_count(0)
    others = body.locator(':scope > pre.result:not(.error)')
    if rest is None:
        await expect(others).to_have_count(0)
    else:
        await expect(others).to_have_text(rest)


async def test_file_change_row_mounts_a_diff_preview(page: Page, web_ui: WebUI) -> None:
    await page.goto(f'{web_ui.url}/?token={TOKEN}')
    await page.locator('.sidebar').get_by_text('Browser Gamma', exact=True).click()
    await expect(page.locator('.session.current .session-title')).to_have_text('Browser Gamma')
    row = page.get_by_role('button', name='Wrote long.py')
    card = page.locator('.tool').filter(has=row)
    await expect(row).to_have_attribute('aria-expanded', 'true')
    await expect(card.locator('.stats')).to_have_text(f'+{LONG_LINES} −0')
    # Open by default, a long diff mounts only its first lines until the user asks for the rest.
    lines = card.locator('.diff tr')
    await expect(lines).to_have_count(PREVIEW_LINES)
    await card.get_by_role('button', name=f'{LONG_LINES - PREVIEW_LINES} more lines').click()
    await expect(lines).to_have_count(LONG_LINES)
    await expect(card.get_by_role('button', name=f'{LONG_LINES - PREVIEW_LINES} more lines')).to_have_count(0)
    await row.click()
    await expect(lines).to_have_count(0)


async def test_switching_back_to_a_watched_session_shows_it_at_once(page: Page, web_ui: WebUI) -> None:
    """A session shown before stays watched, its transcript kept: showing it again waits on no request or snapshot."""
    await page.add_init_script("""
      window.claiTestSnapshots = [];
      const OriginalWebSocket = window.WebSocket;
      window.WebSocket = class extends OriginalWebSocket {
        constructor(...args) {
          super(...args);
          this.addEventListener('message', (event) => {
            const message = JSON.parse(event.data);
            if (message.type === 'snapshot') window.claiTestSnapshots.push(message.session.id);
          });
        }
      };
    """)
    await page.goto(f'{web_ui.url}/?token={TOKEN}')
    sidebar = page.locator('.sidebar')
    composer = page.get_by_role('combobox', name='Message CLAI')
    for title, conversation in (('Browser Beta', web_ui.beta), ('Browser Alpha', web_ui.alpha)):
        await sidebar.get_by_text(title, exact=True).click()
        await expect(page.locator('.session.current .session-title')).to_have_text(title)
        await expect(page.locator('.session[aria-busy="true"]')).to_have_count(0)
        await expect(composer).to_be_enabled()
        await composer.fill(f'Keep {title}')
        async with page.expect_response(f'**/api/sessions/{conversation}/input') as sent:
            await composer.press('Enter')
        assert (await sent.value).ok
        await expect(page.locator('.user-bubble').filter(has_text=f'Keep {title}', visible=True)).to_be_visible()
    beta = page.locator('.transcript').filter(has_text='Keep Browser Beta')
    await expect(beta).to_be_hidden()
    await beta.evaluate('element => { element.dataset.kept = "yes" }')
    snapshots = await page.evaluate('window.claiTestSnapshots.length')

    # Opening it again only makes it the default once more, so a slow answer must not hold it up.
    held: list[Route] = []

    async def hold(route: Route) -> None:
        held.append(route)

    await page.route(f'**/api/sessions/{web_ui.beta}/open', hold)
    async with page.expect_request(f'**/api/sessions/{web_ui.beta}/open'):
        await sidebar.get_by_text('Browser Beta', exact=True).click()
    await expect(page.locator('.transcript[data-kept="yes"]')).to_be_visible()
    await expect(page.locator('.transcript').filter(has_text='Keep Browser Alpha')).to_be_hidden()
    await expect(page.locator('.session[aria-busy="true"]')).to_have_count(0)
    assert await page.evaluate('window.claiTestSnapshots.length') == snapshots, 'it was watched all along'
    [route] = held
    await route.continue_()
    await expect(page.locator('.session.current .session-title')).to_have_text('Browser Beta')


async def test_a_watched_session_switching_conversation_in_the_background_stays_hidden(
    page: Page, web_ui: WebUI
) -> None:
    """`/new` in a session kept in the background replaces its kept chat; the shown session keeps the main pane."""
    await page.add_init_script("""
      window.claiTestSnapshots = [];
      const OriginalWebSocket = window.WebSocket;
      window.WebSocket = class extends OriginalWebSocket {
        constructor(...args) {
          super(...args);
          this.addEventListener('message', (event) => {
            const message = JSON.parse(event.data);
            if (message.type === 'snapshot') window.claiTestSnapshots.push(message.replaces ?? null);
          });
        }
      };
    """)
    await page.goto(f'{web_ui.url}/?token={TOKEN}')
    sidebar = page.locator('.sidebar')
    for title in ('Browser Alpha', 'Browser Beta'):
        await sidebar.get_by_text(title, exact=True).click()
        await expect(page.locator('.session.current .session-title')).to_have_text(title)
        await expect(page.locator('.session[aria-busy="true"]')).to_have_count(0)
    async with page.expect_response(f'**/api/sessions/{web_ui.alpha}/input') as sent:
        await page.evaluate(
            """([id, token]) => fetch(`/api/sessions/${id}/input`, {
              method: 'POST', headers: {'x-clai-token': token, 'content-type': 'application/json'},
              body: JSON.stringify({text: '/new'}),
            })""",
            [web_ui.alpha, TOKEN],
        )
    assert (await sent.value).ok
    await page.wait_for_function('id => window.claiTestSnapshots.includes(id)', arg=web_ui.alpha)
    await expect(page.locator('.topbar .title')).to_have_text('Browser Beta')
    await expect(page.locator('.session.current .session-title')).to_have_text('Browser Beta')


async def test_failed_write_shows_the_attempt_and_the_error(page: Page, web_ui: WebUI) -> None:
    await page.goto(f'{web_ui.url}/?token={TOKEN}')
    await page.locator('.sidebar').get_by_text('Browser Epsilon', exact=True).click()
    await expect(page.locator('.session.current .session-title')).to_have_text('Browser Epsilon')
    await expect(page.locator('.session[aria-busy="true"]')).to_have_count(0)
    composer = page.get_by_role('combobox', name='Message CLAI')
    await expect(composer).to_be_enabled()
    await composer.fill('Write the forbidden file')
    await composer.press('Enter')
    row = page.locator('.tool-row').filter(has_text='forbidden.py')
    await expect(row.locator('.badge.error')).to_have_text('failed')
    await expect(row).to_have_attribute('aria-expanded', 'false')
    await row.click()
    body = page.locator('.tool').filter(has=row).locator('.tool-body')
    await expect(body.locator('.result.error:first-child')).to_contain_text('Permission denied: /proc/forbidden.py')
    await expect(body.locator('.code-small')).to_have_count(0)
    await expect(body).to_contain_text('line_8 = 8')
    assert '\\n' not in await body.inner_text()


async def test_a_refused_update_shows_no_saved_badge(page: Page, web_ui: WebUI) -> None:
    """Settings > About in a session the browser opened refuses Update…; its row must not claim it saved anything."""
    await page.goto(f'{web_ui.url}/?token={TOKEN}')
    await page.locator('.sidebar').get_by_text('Browser Beta', exact=True).click()
    await expect(page.locator('.session.current .session-title')).to_have_text('Browser Beta')
    await expect(page.locator('.session[aria-busy="true"]')).to_have_count(0)
    await page.evaluate("location.hash = '#settings/about'")
    settings = page.get_by_role('dialog', name='Settings')
    version = settings.locator('.setting').filter(has=page.locator('.setting-label', has_text='Version'))
    async with page.expect_response(f'**/api/sessions/{web_ui.beta}/commands/update/fields/version/action') as acted:
        await version.get_by_role('button', name='Update Version').click()
    assert (await acted.value).status == 400
    # Read with the toast in one page call: a badge renders with it, and a retry would pass once its timer clears it.
    badge = await page.wait_for_function(
        """() => {
          if (!document.querySelector('.toast.error')?.textContent?.includes('/reload and /update restart CLAI.')) return null
          const row = [...document.querySelectorAll('.setting')]
            .find((setting) => setting.querySelector('.setting-label')?.textContent === 'Version')
          return row.querySelector('.setting-state')?.textContent ?? 'none'
        }"""
    )
    assert await badge.json_value() == 'none'


async def test_switching_sessions_focuses_the_composer(page: Page, web_ui: WebUI) -> None:
    await page.goto(f'{web_ui.url}/?token={TOKEN}')
    sidebar = page.locator('.sidebar')
    current = page.locator('.session.current .session-title')
    composer = page.get_by_role('combobox', name='Message CLAI')
    # Whichever session an earlier case left shown, Gamma shows first, so picking Beta switches.
    await expect(current).to_have_count(1)
    await sidebar.get_by_text('Browser Gamma', exact=True).click()
    await expect(current).to_have_text('Browser Gamma')
    await sidebar.get_by_text('Browser Beta', exact=True).click()
    await expect(current).to_have_text('Browser Beta')
    await expect(page.locator('.session[aria-busy="true"]')).to_have_count(0)
    await expect(composer).to_be_focused()
    # Typing right away lands in the prompt box.
    await page.keyboard.type('typed after switching')
    await expect(composer).to_have_value('typed after switching')
    await composer.fill('')
    # Focus the reader moved elsewhere while the session opened stays put.
    release = asyncio.Event()

    async def hold(route: Route) -> None:
        await release.wait()
        await route.continue_()

    await page.route(f'**/api/sessions/{web_ui.delta}/open', hold)
    await sidebar.get_by_text('Browser Delta', exact=True).click()
    await expect(page.locator('.session[aria-busy="true"]')).to_have_count(1)
    # The row picked is the selection at once, while its session still opens, and shows no spinner of its own.
    await expect(current).to_have_text('Browser Delta')
    await expect(page.locator('.session[aria-busy="true"] .spinner')).to_have_count(0)
    search = page.get_by_placeholder('Search sessions')
    await search.focus()
    release.set()
    await expect(current).to_have_text('Browser Delta')
    await expect(page.locator('.session[aria-busy="true"]')).to_have_count(0)
    await expect(search).to_be_focused()
    # A watched session shows at once without opening, and takes focus all the same.
    await sidebar.get_by_text('Browser Gamma', exact=True).click()
    await expect(current).to_have_text('Browser Gamma')
    await expect(composer).to_be_focused()


async def test_at_mentions_pick_files_from_the_session_folder(page: Page, web_ui: WebUI) -> None:
    await page.goto(f'{web_ui.url}/?token={TOKEN}')
    state = await page.request.get(f'{web_ui.url}/api/sessions/{web_ui.lines}/state', headers={HEADER: TOKEN})
    folder = Path((await state.json())['session']['workspace'])
    for name in ('README.md', 'docs/README.md', 'src/package/module.py'):
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        (folder / name).write_text(name)
    await page.locator('.sidebar').get_by_text('Browser Lines', exact=True).click()
    await expect(page.locator('.session.current .session-title')).to_have_text('Browser Lines')
    await expect(page.locator('.session[aria-busy="true"]')).to_have_count(0)
    composer = page.get_by_role('combobox', name='Message CLAI')
    await expect(composer).to_be_enabled()
    options = page.get_by_role('listbox').get_by_role('option')
    names = options.locator('.suggestion-name')

    # `@` at the start of the box offers the folder's files, best first, and Enter takes the one selected.
    await composer.fill('@READ')
    await expect(names).to_have_text(['README.md', 'README.md'])
    await expect(options.nth(1).locator('.suggestion-meta')).to_have_text('docs')
    await expect(options.first).to_have_attribute('aria-selected', 'true')
    await composer.press('Enter')
    await expect(composer).to_have_value('@README.md ')
    await expect(options).to_have_count(0)

    # Mid-text too: a folder stays open, offering its own files next, with the folder shown beside each.
    await composer.fill('foo @src/pa')
    await expect(names).to_have_text(['package/', 'module.py'])
    await expect(options.nth(1).locator('.suggestion-meta')).to_have_text('src/package')
    await composer.press('ArrowDown')
    await expect(options.nth(1)).to_have_attribute('aria-selected', 'true')
    await composer.press('ArrowUp')
    await composer.press('Enter')
    await expect(composer).to_have_value('foo @src/package/')
    await expect(names).to_have_text(['module.py'])
    await composer.press('Tab')
    await expect(composer).to_have_value('foo @src/package/module.py ')
    await expect(options).to_have_count(0)

    # A mention typed before more text replaces only itself, the caret stays after it, and the list stays closed.
    await composer.fill('see  and more')
    await composer.evaluate('area => area.setSelectionRange(4, 4)')
    await page.keyboard.type('@REA')
    await expect(names.first).to_have_text('README.md')
    await composer.press('Tab')
    await expect(composer).to_have_value('see @README.md and more')
    assert await composer.evaluate('area => area.selectionStart') == len('see @README.md')
    await expect(options).to_have_count(0)

    # With the caret moved into a mention, the part before it is completed and the rest of the word replaced.
    async with page.expect_response('**/files?*'):
        await composer.fill('@src/pazzz')
    await expect(options).to_have_count(0)
    for _ in 'zzz':
        await composer.press('ArrowLeft')
    await expect(names.first).to_have_text('package/')
    await composer.press('Tab')
    await expect(composer).to_have_value('@src/package/')

    # Moving the caret off the mention closes its list.
    await composer.fill('see @REA')
    await expect(names.first).to_have_text('README.md')
    await composer.press('Home')
    await expect(options).to_have_count(0)
    await composer.press('End')
    await expect(names.first).to_have_text('README.md')

    # Keys pressed before the list catches up with the draft wait for it, instead of taking a stale suggestion.
    await composer.fill('@src/')
    await expect(names).to_have_text(['package/', 'module.py'])
    held = asyncio.Event()

    async def hold(route: Route) -> None:
        await held.wait()
        await route.continue_()

    await page.route('**/files?*', hold)
    await composer.press('m')
    await composer.press('Tab')
    await options.first.click()
    await expect(composer).to_have_value('@src/m')
    held.set()
    await expect(names).to_have_text(['module.py'])
    await page.unroute('**/files?*', hold)
    await composer.press('Tab')
    await expect(composer).to_have_value('@src/package/module.py ')

    # Esc while newer files are on their way keeps the list closed when they arrive.
    await composer.fill('@READ')
    await expect(names.first).to_have_text('README.md')
    late = asyncio.Event()

    async def hold_late(route: Route) -> None:
        await late.wait()
        await route.continue_()

    await page.route('**/files?*', hold_late)
    async with page.expect_response('**/files?*') as arrived:
        await composer.press('M')
        await composer.press('Escape')
        await expect(options).to_have_count(0)
        late.set()
    await (await arrived.value).body()
    await page.unroute('**/files?*', hold_late)
    await composer.press('ArrowLeft')
    await expect(composer).to_have_value('@READM')
    await expect(options).to_have_count(0)

    # A file typed in full is offered first, and Enter sends it instead of swapping it for another of that name.
    await composer.fill('see @README.md')
    await expect(names).to_have_text(['README.md', 'README.md'])
    await expect(options.first.locator('.suggestion-meta')).to_have_count(0)
    async with page.expect_request(f'**/api/sessions/{web_ui.lines}/input') as typed_in_full:
        await composer.press('Enter')
    assert (await typed_in_full.value).post_data_json == {'text': 'see @README.md'}
    # The first message moves the composer from the new-session hero to the bottom bar; type there.
    await expect(page.locator('.user-bubble').filter(has_text='see @README.md')).to_be_visible()
    await expect(composer).to_have_value('')
    # A longer word under the caret is not the file typed in full, so Enter takes the suggestion instead.
    await composer.fill('@README.mdx')
    await composer.press('ArrowLeft')
    await expect(names.first).to_have_text('README.md')
    await composer.press('Enter')
    await expect(composer).to_have_value('@README.md ')

    # Esc closes the list, and Enter then sends the mention as typed, for the agent to read with its tools.
    await composer.fill('read @READ')
    await expect(names.first).to_have_text('README.md')
    await composer.press('Escape')
    await expect(options).to_have_count(0)
    async with page.expect_request(f'**/api/sessions/{web_ui.lines}/input') as sent:
        await composer.press('Enter')
    assert (await sent.value).post_data_json == {'text': 'read @READ'}


async def test_a_session_another_clai_runs_shows_until_forked(page: Page, web_ui: WebUI) -> None:
    """Clicking it shows it read-only and saves nothing; only the banner's button continues it, in a fork."""
    await page.goto(f'{web_ui.url}/?token={TOKEN}')
    sidebar = page.locator('.sidebar')
    # This CLAI does not follow sessions' folders, so it continues only those of the session it shows: Beta's.
    await sidebar.get_by_text('Browser Beta', exact=True).click()
    await expect(page.locator('.session.current .session-title')).to_have_text('Browser Beta')
    async with page.expect_response(f'**/api/sessions/{web_ui.busy}') as shown:
        await sidebar.get_by_text('Browser Busy', exact=True).click()
    assert (await shown.value).ok
    banner = page.locator('.preview-banner')
    await expect(banner).to_contain_text(f'Running in another CLAI (pid {os.getppid()}).')
    await expect(sidebar.get_by_text('Browser Busy (fork)', exact=True)).to_have_count(0)
    await banner.get_by_role('button', name='Fork to continue here').click()
    await expect(page.locator('.session.current .session-title')).to_have_text('Browser Busy (fork)')
    await expect(banner).to_have_count(0)


async def list_gamma_folder_as_openable(page: Page, web_ui: WebUI) -> None:
    """Opening Gamma makes it the default session, so the sidebar lists the sessions saved in its folder as openable."""
    await page.goto(f'{web_ui.url}/?token={TOKEN}')
    await page.locator('.sidebar').get_by_text('Browser Gamma', exact=True).click()
    await expect(page.locator('.session.current .session-title')).to_have_text('Browser Gamma')
    await expect(page.locator('.session[aria-busy="true"]')).to_have_count(0)
    await page.reload()
    await expect(page.locator('.session.current .session-title')).to_have_text('Browser Gamma')


async def test_a_saved_session_opens_from_its_row_named_at_once(page: Page, web_ui: WebUI) -> None:
    """A saved session's row says it opens live: a click opens it without fetching its transcript first.

    The header names it from the row at once, and never as a new session while its snapshot is on its way.
    """
    await page.add_init_script("""
      window.claiTestTitles = [];
      new MutationObserver(() => {
        const title = document.querySelector('.topbar .title')?.textContent;
        if (title !== undefined && window.claiTestTitles.at(-1) !== title) window.claiTestTitles.push(title);
      }).observe(document, { subtree: true, childList: true, characterData: true });
    """)
    await list_gamma_folder_as_openable(page, web_ui)
    requested: list[str] = []
    page.on('request', lambda request: requested.append(request.url))
    held: list[Route] = []

    async def hold(route: Route) -> None:
        held.append(route)

    await page.route(f'**/api/sessions/{web_ui.saved}/open', hold)
    async with page.expect_request(f'**/api/sessions/{web_ui.saved}/open'):
        await page.locator('.sidebar').get_by_text('Browser Saved', exact=True).click()
    await expect(page.locator('.topbar .title')).to_have_text('Browser Saved')
    await expect(page).to_have_title('Browser Saved · CLAI')
    [route] = held
    await route.continue_()
    await expect(page.locator('.session.current .session-title')).to_have_text('Browser Saved')
    await expect(page.locator('.user-bubble').filter(has_text='Saved prompt', visible=True)).to_be_visible()
    assert f'{web_ui.url}/api/sessions/{web_ui.saved}' not in requested, 'the row said it opens'
    titles = await page.evaluate('window.claiTestTitles')
    assert 'New session' not in titles[titles.index('Browser Saved') :], titles


async def test_a_row_that_says_it_opens_shows_read_only_when_it_no_longer_can(page: Page, web_ui: WebUI) -> None:
    """A row listed before the session's folder went: opening fails, so it shows read-only, as its transcript says."""
    await list_gamma_folder_as_openable(page, web_ui)
    await page.route(
        f'**/api/sessions/{web_ui.moved}/open',
        lambda route: route.fulfill(status=400, json={'detail': 'The folder is gone.'}),
    )
    transcript: dict[str, object] = {
        'id': web_ui.moved,
        'title': 'Browser Moved',
        'workspace': '/gone',
        'items': [],
        'openable': False,
        'missing': True,
        'owner_pid': None,
    }
    await page.route(f'**/api/sessions/{web_ui.moved}', lambda route: route.fulfill(json=transcript))
    # The row says it opens, so the click asks to open it before anything else.
    async with page.expect_request(f'**/api/sessions/{web_ui.moved}/open'):
        await page.locator('.sidebar').get_by_text('Browser Moved', exact=True).click()
    await expect(page.locator('.preview-banner')).to_contain_text('which no longer exists')
    await expect(page.locator('.topbar .title')).to_have_text('Browser Moved')
    await expect(page.get_by_role('alert')).to_have_count(0)


async def test_a_pick_over_a_read_only_view_names_the_picked_session(page: Page, web_ui: WebUI) -> None:
    """While a session opens over a read-only view, the header names the session picked, as the tab title does."""
    await list_gamma_folder_as_openable(page, web_ui)
    await page.route(
        f'**/api/sessions/{web_ui.moved}/open',
        lambda route: route.fulfill(status=400, json={'detail': 'The folder is gone.'}),
    )
    transcript: dict[str, object] = {
        'id': web_ui.moved,
        'title': 'Browser Moved',
        'workspace': '/gone',
        'items': [],
        'openable': False,
        'missing': True,
        'owner_pid': None,
    }
    await page.route(f'**/api/sessions/{web_ui.moved}', lambda route: route.fulfill(json=transcript))
    sidebar = page.locator('.sidebar')
    await sidebar.get_by_text('Browser Moved', exact=True).click()
    await expect(page.locator('.preview-banner')).to_contain_text('which no longer exists')
    held: list[Route] = []

    async def hold(route: Route) -> None:
        held.append(route)

    await page.route(f'**/api/sessions/{web_ui.saved}/open', hold)
    async with page.expect_request(f'**/api/sessions/{web_ui.saved}/open'):
        await sidebar.get_by_text('Browser Saved', exact=True).click()
    await expect(page.locator('.topbar .title')).to_have_text('Browser Saved')
    await expect(page).to_have_title('Browser Saved · CLAI')
    [route] = held
    await route.continue_()
    await expect(page.locator('.session.current .session-title')).to_have_text('Browser Saved')


async def test_the_websocket_opens_before_the_page_renders(page: Page, web_ui: WebUI) -> None:
    """The first snapshot is on its way while React renders the page, not only once it has mounted."""
    await page.add_init_script("""
      window.claiTestRendered = [];
      const OriginalWebSocket = window.WebSocket;
      window.WebSocket = class extends OriginalWebSocket {
        constructor(...args) {
          super(...args);
          window.claiTestRendered.push(document.getElementById('root')?.childElementCount ?? 0);
        }
      };
    """)
    await page.goto(f'{web_ui.url}/?token={TOKEN}')
    await expect(page.get_by_title('Connected', exact=True)).to_be_visible()
    assert await page.evaluate('window.claiTestRendered') == [0]


async def test_opening_a_session_that_resolves_to_the_shown_one_ends_the_wait(page: Page, web_ui: WebUI) -> None:
    """A busy session opens as its fork; when that fork already shows, the click's wait ends with the answer."""
    await page.goto(f'{web_ui.url}/?token={TOKEN}')
    sidebar = page.locator('.sidebar')
    current = page.locator('.session.current .session-title')
    await expect(current).to_have_count(1)
    # The page keeps watching the session it started on and would show it at once, without opening it.
    started = await current.inner_text()
    clicked, clicked_id = (
        ('Browser Lines', web_ui.lines) if started != 'Browser Lines' else ('Browser Epsilon', web_ui.epsilon)
    )
    shown = 'Browser Beta'
    await sidebar.get_by_text(shown, exact=True).click()
    await expect(current).to_have_text(shown)
    await expect(page.locator('.session[aria-busy="true"]')).to_have_count(0)

    async def to_shown(route: Route) -> None:
        await route.fulfill(json={'id': web_ui.beta})

    await page.route(f'**/api/sessions/{clicked_id}/open', to_shown)
    async with page.expect_response(f'**/api/sessions/{clicked_id}/open'):
        await sidebar.get_by_text(clicked, exact=True).click()
    # Well before the 15 second fallback that ends a wait for a session that never shows.
    await expect(page.locator('.opening-overlay')).to_have_count(0, timeout=2_000)
    await expect(page.locator('.session[aria-busy="true"]')).to_have_count(0, timeout=2_000)
    await expect(current).to_have_text(shown)


async def opened(page: Page, title: str) -> None:
    """Wait for the session picked to be the one shown, not only the row selected."""
    await expect(page.locator('.session.current .session-title')).to_have_text(title)
    await expect(page.locator('.session[aria-busy="true"]')).to_have_count(0)


async def send(page: Page, conversation: str, text: str) -> None:
    """Prompt a session that the page does not show, as another tab or the terminal would."""
    async with page.expect_response(f'**/api/sessions/{conversation}/input') as sent:
        await page.evaluate(
            """([id, token, text]) => fetch(`/api/sessions/${id}/input`, {
              method: 'POST', headers: {'x-clai-token': token, 'content-type': 'application/json'},
              body: JSON.stringify({text}),
            })""",
            [conversation, TOKEN, text],
        )
    assert (await sent.value).ok


async def test_sidebar_rows_keep_their_places(page: Page, web_ui: WebUI) -> None:
    """Activity, renames, and selection move no row: each keeps its height in the list, however far it is scrolled."""
    await page.goto(f'{web_ui.url}/?token={TOKEN}')
    sidebar = page.locator('.sidebar')
    await sidebar.get_by_role('button', name='Projects', exact=True).click()
    await expect(sidebar.get_by_text('Browser Epsilon', exact=True)).to_be_visible()
    await sidebar.get_by_text('Browser Beta', exact=True).click()
    await opened(page, 'Browser Beta')
    rows = sidebar.locator('.session')

    async def places() -> list[tuple[str, float]]:
        return await rows.evaluate_all(
            'rows => rows.map(row => [row.dataset.session, Math.round(row.getBoundingClientRect().top - row.closest(".projects").getBoundingClientRect().top + row.closest(".projects").scrollTop)])'
        )

    before = await places()
    # Projects are listed by name, whichever is current, and removed folders fold away last.
    assert [title for title in await sidebar.locator('.project-name').all_inner_texts()][-1] == 'Removed worktrees (1)'

    # A turn in a session not shown saves it, which used to move its row to the top.
    await send(page, web_ui.gamma, 'Activity elsewhere')
    gamma = sidebar.locator(f'.session[data-session="{web_ui.gamma}"]')
    await expect(gamma).to_have_class(re.compile(r'\bunread\b'))
    assert await places() == before

    response = await page.evaluate(
        """([id, token]) => fetch(`/api/sessions/${id}/rename`, {
          method: 'POST', headers: {'x-clai-token': token, 'content-type': 'application/json'},
          body: JSON.stringify({title: 'Browser Delta renamed'}),
        }).then(response => response.status)""",
        [web_ui.delta, TOKEN],
    )
    assert response == 204
    await expect(sidebar.get_by_text('Browser Delta renamed', exact=True)).to_be_visible()
    assert await places() == before

    await sidebar.get_by_text('Browser Alpha', exact=True).click()
    await opened(page, 'Browser Alpha')
    assert await places() == before
    # Opening the unread session marks it read.
    await gamma.locator('.session-main').click()
    await opened(page, 'Browser Gamma')
    await expect(gamma).not_to_have_class(re.compile(r'\bunread\b'))
    assert await places() == before

    # Recent lists every project's sessions newest created first.
    await sidebar.get_by_role('button', name='Recent', exact=True).click()
    created = [web_ui.removed, web_ui.busy, web_ui.epsilon, web_ui.delta, web_ui.gamma, web_ui.lines]
    listed = [conversation for conversation, _ in await places() if conversation in created]
    assert listed == [web_ui.busy, web_ui.epsilon, web_ui.delta, web_ui.gamma, web_ui.lines], (
        'the removed one is folded away'
    )
    await page.evaluate(
        """([id, token]) => fetch(`/api/sessions/${id}/rename`, {
          method: 'POST', headers: {'x-clai-token': token, 'content-type': 'application/json'},
          body: JSON.stringify({title: 'Browser Delta'}),
        })""",
        [web_ui.delta, TOKEN],
    )
    await expect(sidebar.get_by_text('Browser Delta', exact=True)).to_be_visible()
    await sidebar.get_by_role('button', name='Recent', exact=True).click()
    await expect(sidebar.get_by_role('button', name='Projects', exact=True)).to_have_attribute('aria-pressed', 'true')


async def test_a_session_another_clai_runs_shows_a_ring(page: Page, web_ui: WebUI) -> None:
    await page.goto(f'{web_ui.url}/?token={TOKEN}')
    sidebar = page.locator('.sidebar')
    row = sidebar.locator(f'.session[data-session="{web_ui.busy}"]')
    ring = row.locator('.state.busy')
    await expect(ring).to_have_attribute('title', f'Running in clai2 pid {os.getppid()}')
    await expect(row.locator('.spinner')).to_have_count(0)
    # It is running, so the attention view lists it, and the archived view does not.
    await sidebar.get_by_role('button', name=re.compile('^Attention')).click()
    await expect(row).to_be_visible()
    await expect(sidebar.locator('.session').filter(has_text='Browser Beta')).to_have_count(0)
    search = sidebar.get_by_role('textbox', name='Search sessions', exact=True)
    await search.fill('nothing is called this')
    await expect(sidebar.get_by_text('No matching sessions', exact=True)).to_be_visible()
    await search.fill('')
    await sidebar.get_by_role('button', name='Archived', exact=True).click()
    await expect(row).to_have_count(0)
    await sidebar.get_by_role('button', name='Projects', exact=True).click()
    await expect(row).to_be_visible()


async def test_removed_worktrees_fold_away(page: Page, web_ui: WebUI) -> None:
    await page.goto(f'{web_ui.url}/?token={TOKEN}')
    sidebar = page.locator('.sidebar')
    bucket = sidebar.get_by_role('button', name='Removed worktrees (1)')
    await expect(bucket).to_have_attribute('aria-expanded', 'false')
    removed = sidebar.get_by_text('Browser Removed', exact=True)
    await expect(removed).to_have_count(0)
    await bucket.click()
    await expect(removed).to_be_visible()
    # What is opened or folded is kept across reloads.
    await page.reload()
    await expect(removed).to_be_visible()
    await bucket.click()
    await expect(removed).to_have_count(0)


async def test_sidebar_keyboard(page: Page, web_ui: WebUI) -> None:
    """Search moves to its results with Down and opens the first with Enter; the arrows move between rows."""
    await page.goto(f'{web_ui.url}/?token={TOKEN}')
    sidebar = page.locator('.sidebar')
    search = sidebar.get_by_role('textbox', name='Search sessions', exact=True)
    await search.fill('Browser Gamma')
    gamma = sidebar.locator(f'.session[data-session="{web_ui.gamma}"] .session-main')
    await expect(sidebar.locator('.session')).to_have_count(1)
    await expect(gamma).to_be_visible()
    await search.press('ArrowDown')
    await expect(gamma).to_be_focused()
    # Up goes back through the project's heading to the search.
    await page.keyboard.press('ArrowUp')
    gamma_project = sidebar.locator('.project', has=page.locator(f'[data-session="{web_ui.gamma}"]'))
    await expect(gamma_project.locator('.project-title')).to_be_focused()
    await page.keyboard.press('ArrowUp')
    await expect(search).to_be_focused()
    await search.fill('Browser Lines')
    lines = sidebar.locator(f'.session[data-session="{web_ui.lines}"] .session-main')
    await expect(sidebar.locator('.session')).to_have_count(1)
    await expect(lines).to_be_visible()
    await search.press('Enter')
    await opened(page, 'Browser Lines')
    # Enter right after typing opens the first result for what was typed, not for the search before.
    await search.fill('Browser Epsilon')
    await search.press('Enter')
    await opened(page, 'Browser Epsilon')
    # So does Enter typed while the list shows no match at all.
    await search.fill('no such session anywhere')
    await expect(sidebar.locator('.session')).to_have_count(0)
    await search.fill('Browser Gamma')
    await search.press('Enter')
    await opened(page, 'Browser Gamma')
    await search.fill('Browser Lines')
    await expect(sidebar.locator('.session')).to_have_count(1)
    await search.fill('')
    await expect(sidebar.get_by_text('Browser Alpha', exact=True)).to_be_visible()
    await lines.focus()
    await page.keyboard.press('ArrowDown')
    await expect(lines).not_to_be_focused()
    await page.keyboard.press('ArrowUp')
    await expect(lines).to_be_focused()
    # Left moves to the project's heading, which Left folds and Right opens again.
    await page.keyboard.press('ArrowLeft')
    heading = sidebar.locator('.project-title:has(.project-name:text-is("lines"))')
    await expect(heading).to_be_focused()
    await page.keyboard.press('ArrowLeft')
    await expect(heading).to_have_attribute('aria-expanded', 'false')
    await page.keyboard.press('ArrowRight')
    await expect(heading).to_have_attribute('aria-expanded', 'true')


async def test_sidebar_ignores_saved_state_of_another_shape(page: Page, web_ui: WebUI) -> None:
    """A cached list, view, or folding of the wrong shape, as another version might leave, does not break the page."""
    await page.add_init_script("""
      localStorage.setItem('clai-sidebar-cache', JSON.stringify({projects: []}));
      localStorage.setItem('clai-sidebar-view', JSON.stringify('nonsense'));
      localStorage.setItem('clai-sidebar-collapsed', JSON.stringify([1, 2]));
    """)
    await page.goto(f'{web_ui.url}/?token={TOKEN}')
    sidebar = page.locator('.sidebar')
    await expect(sidebar.get_by_role('button', name='Projects', exact=True)).to_have_attribute('aria-pressed', 'true')
    await expect(sidebar.get_by_text('Browser Alpha', exact=True)).to_be_visible()


def git(*args: str) -> str:
    return subprocess.run(['git', *args], check=True, capture_output=True, text=True).stdout.strip()


async def stay(workspace: str) -> None:
    """A session's relocation that `clai2 --web` wires, which this test never reaches; only the dialog depends on it."""


async def test_new_session_checks_out_a_new_worktree(page: Page, web_ui: WebUI, tmp_path: Path) -> None:
    repo = (tmp_path / 'repo').resolve()
    git('init', '-b', 'main', str(repo))
    git(
        '-C',
        str(repo),
        '-c',
        'user.name=Test',
        '-c',
        'user.email=test@example.com',
        'commit',
        '--allow-empty',
        '-m',
        'init',
    )
    opened = await page.request.post(
        f'{web_ui.url}/api/sessions/new', headers={HEADER: TOKEN}, data={'workspace': str(repo)}
    )
    assert opened.ok, await opened.text()
    root = repo / '.worktrees' / 'feature-x'
    # As under `clai2 --web`, so New session asks where; the fixture's sessions stay where they are.
    runtimes = web_ui.fleet.runtimes
    runtimes['repo'].sessions.relocate = stay
    try:
        await page.goto(f'{web_ui.url}/?token={TOKEN}')
        sidebar = page.locator('.sidebar')
        dialog = page.get_by_role('dialog', name='New session')
        new_worktree = dialog.get_by_role('button', name='New worktree of repo · main…')
        await sidebar.get_by_role('button', name='New session').click()
        await expect(dialog.get_by_role('option', name='repo · main')).to_be_visible()
        await dialog.get_by_role('option', name='repo · main').hover()
        await new_worktree.click()
        await dialog.get_by_label('Branch of the new worktree of repo · main').fill('feature-x')
        await dialog.get_by_label('Base').fill('main')
        # A session that fails to start leaves the worktree made, and trying again starts one in it.
        # Esc while the request runs keeps the form, as the disabled Back button does.
        release = asyncio.Event()

        async def busy(route: Route) -> None:
            await release.wait()
            await route.fulfill(status=409, json={'detail': 'Busy'})

        await page.route('**/api/sessions/new', busy)
        await dialog.get_by_role('button', name='Create and start').click()
        await expect(dialog.get_by_role('button', name='Back')).to_be_disabled()
        await page.keyboard.press('Escape')
        await expect(dialog.get_by_label('Base')).to_be_visible()
        release.set()
        await expect(dialog.get_by_role('alert')).to_have_text('Busy')
        await page.unroute('**/api/sessions/new')
        await dialog.get_by_role('button', name='Create and start').click()
        await expect(dialog).to_have_count(0)
        await expect(page.locator(f'.workspace[title="{root}"]')).to_be_visible()
        assert git('-C', str(root), 'branch', '--show-current') == 'feature-x'
        # The next dialog lists it, offers no worktree of a folder outside Git, and refuses the same branch inline.
        runtimes['feature-x'].sessions.relocate = stay
        await page.reload()
        await sidebar.get_by_role('button', name='New session').click()
        await expect(dialog.get_by_role('option', name='repo · feature-x')).to_be_visible()
        await dialog.get_by_role('option', name='alpha').hover()
        await expect(dialog.get_by_role('button', name=re.compile('^New worktree'))).to_have_count(0)
        await dialog.get_by_role('option', name='repo · main').hover()
        await new_worktree.click()
        # Esc goes back to the list, and Esc there closes the dialog.
        await page.keyboard.press('Escape')
        await expect(dialog.get_by_label('Another directory')).to_be_visible()
        await new_worktree.click()
        await dialog.get_by_label('Branch of the new worktree of repo · main').fill('feature-x')
        await dialog.get_by_role('button', name='Create and start').click()
        await expect(dialog.get_by_role('alert')).to_have_text(
            'Cannot create worktree: branch feature-x already exists. Pick another name.'
        )
    finally:
        for name in ('repo', 'feature-x'):
            if name in runtimes:
                runtimes[name].sessions.relocate = None
