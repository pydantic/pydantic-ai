"""Git in a real browser: the header's chip and dialog (commit, push, pull request) and the Changes panel's Git scopes."""

import json
import re
import sys
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from pathlib import Path

import anyio
import pytest
from anyio.from_thread import BlockingPortal
from playwright.async_api import Page, expect

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, UserPromptPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai_harness.file_system import WRITE_FILE_KIND
from tests.clai2.browser.conftest import Serve
from tests.clai2.test_git_actions import PR_URL, Drafter, agents, calls, gh_state, git, repo
from tests.clai2.test_web import TOKEN
from tests.clai2.test_web_sessions import MakeFleet, make_fleet

__all__ = ['gh_state', 'make_fleet', 'repo']

pytestmark = pytest.mark.skipif(
    sys.platform == 'win32', reason='the stand-in `gh` and the hooks are POSIX shell scripts'
)


@dataclass
class GitUI:
    url: str
    conversation: str
    drafter: Drafter


@pytest.fixture
def git_ui(make_fleet: MakeFleet, serve: Serve, repo: Path, gh_state: Path) -> Iterator[GitUI]:
    """A live session in `repo`, on a `feature` branch where the agent edited three files, served to the browser."""
    drafter = Drafter()
    fleet = make_fleet(agents(drafter, 'repo'))
    conversation = fleet.new('repo')
    # Saved, so it can be renamed, as the sidebar lists it.
    fleet.portal.call(fleet.runtimes['repo'].session.commit_messages, list[ModelMessage]())
    assert fleet.post(f'/api/sessions/{conversation}/rename', {'title': 'Git Repo'}).status_code == 204
    git(repo, 'checkout', '-b', 'feature')
    for name in ('notes.txt', 'second.txt', 'third.txt'):
        (repo / name).write_text('edited by the agent\n')
    with serve(fleet) as url:
        yield GitUI(url=url, conversation=conversation, drafter=drafter)


async def test_commit_push_and_open_a_pull_request_from_the_header(
    page: Page, git_ui: GitUI, repo: Path, gh_state: Path
) -> None:
    await page.goto(f'{git_ui.url}/?token={TOKEN}')
    await page.locator('.sidebar').get_by_text('Git Repo', exact=True).click()
    await expect(page.locator('.session.current .session-title')).to_have_text('Git Repo')
    await expect(page.locator('.session[aria-busy="true"]')).to_have_count(0)
    chip = page.locator('.topbar .git-chip')
    await expect(chip).to_have_text('feature3 changed, 0 ahead')

    await chip.click()
    dialog = page.get_by_role('dialog', name='Git')
    files = dialog.get_by_role('group', name='Changed files').get_by_role('checkbox')
    await expect(files).to_have_count(3)
    for index in range(3):
        await expect(files.nth(index)).to_be_checked()
    message = dialog.get_by_role('textbox', name='Commit message')
    # The session's model drafts the message.
    await expect(message).to_have_value('Edit the notes\n\nAnd add one.')

    # A hook that refuses the commit says why, in the dialog.
    hook = repo / '.git' / 'hooks' / 'pre-commit'
    hook.write_text('#!/bin/sh\necho "lint: third.txt is not formatted" >&2\nexit 1\n')
    hook.chmod(0o755)
    await dialog.get_by_role('button', name='Commit all').click()
    await expect(dialog.get_by_role('alert')).to_have_text('lint: third.txt is not formatted')
    hook.unlink()

    await message.fill('Edit three files')
    # Looking at the pull request and coming back keeps the message, and drafts nothing again.
    await dialog.get_by_role('button', name='Create pull request…').click()
    await expect(dialog.get_by_role('textbox', name='Title')).to_have_value('Add a feature')
    await dialog.get_by_role('button', name='Back').click()
    await expect(message).to_have_value('Edit three files')
    await dialog.get_by_role('button', name='Commit all').click()
    await expect(dialog.get_by_role('status')).to_contain_text('Edit three files')
    await expect(chip).to_have_text('feature0 changed, 1 ahead')
    assert git(repo, 'log', '-1', '--format=%s').strip() == 'Edit three files'
    assert sorted(git(repo, 'show', '--name-only', '--format=', 'HEAD').split()) == [
        'notes.txt',
        'second.txt',
        'third.txt',
    ]
    await expect(dialog.get_by_text('No uncommitted changes.')).to_be_visible()

    await dialog.get_by_role('button', name='Push -u origin').click()
    await expect(dialog.get_by_role('status')).to_have_text('Pushed feature to origin.')
    await expect(chip).to_have_text('feature0 changed, 0 ahead')
    await expect(dialog.get_by_role('button', name='Push', exact=True)).to_be_enabled()

    await dialog.get_by_role('button', name='Create pull request…').click()
    title = dialog.get_by_role('textbox', name='Title')
    await expect(title).to_have_value('Add a feature')
    await expect(dialog.get_by_role('textbox', name='Description')).to_have_value('It adds one.')
    await title.fill('Edit three files')
    await dialog.get_by_role('button', name='Open pull request').click()
    link = dialog.get_by_role('link', name=PR_URL)
    await expect(link).to_have_attribute('href', PR_URL)
    # The chip asks `gh` again once the pull request is open, so its call may come after.
    assert 'pr create --head=feature --title=Edit three files --body-file=-' in calls(gh_state)

    # Esc closes the dialog, not the page's turn.
    await page.keyboard.press('Escape')
    await expect(dialog).to_have_count(0)

    # On the default branch, pushing needs a confirmation and no pull request can be opened.
    git(repo, 'checkout', 'main')
    await page.evaluate('window.dispatchEvent(new Event("focus"))')
    await expect(chip).to_have_text('main0 changed, 0 ahead')
    await chip.click()
    push = dialog.get_by_role('button', name='Push', exact=True)
    await expect(push).to_be_disabled()
    await dialog.get_by_role('checkbox', name='Push to main, the default branch').check()
    await expect(push).to_be_enabled()
    await expect(dialog.get_by_role('button', name='Create pull request…')).to_be_disabled()
    await expect(
        dialog.get_by_text('main is the default branch: open a pull request from another branch.')
    ).to_be_visible()

    # Without a signed-in `gh`, the reason shows instead.
    git(repo, 'checkout', 'feature')
    (gh_state / 'token').unlink()
    await dialog.get_by_role('button', name='Close').click()
    await page.evaluate('window.dispatchEvent(new Event("focus"))')
    await expect(chip).to_have_text('feature0 changed, 0 ahead')
    await chip.click()
    await expect(dialog.get_by_role('button', name='Create pull request…')).to_be_disabled()
    await expect(dialog.get_by_text('The GitHub CLI is not signed in to github.com')).to_be_visible()


async def test_unchecking_a_file_marks_the_drafted_message_stale(page: Page, git_ui: GitUI) -> None:
    await page.goto(f'{git_ui.url}/?token={TOKEN}')
    await page.locator('.sidebar').get_by_text('Git Repo', exact=True).click()
    await expect(page.locator('.session[aria-busy="true"]')).to_have_count(0)
    await page.locator('.topbar .git-chip').click()
    dialog = page.get_by_role('dialog', name='Git')
    message = dialog.get_by_role('textbox', name='Commit message')
    await expect(message).to_have_value('Edit the notes\n\nAnd add one.')
    stale = dialog.get_by_text('This message was drafted for 3 files; 2 are selected.')
    await expect(stale).to_have_count(0)

    await dialog.get_by_role('checkbox', name='third.txt').uncheck()
    await expect(stale).to_be_visible()
    # Checking it again makes the draft match the selection.
    await dialog.get_by_role('checkbox', name='third.txt').check()
    await expect(stale).to_have_count(0)

    await dialog.get_by_role('checkbox', name='third.txt').uncheck()
    redraft = dialog.locator('.git-message-label button')
    await redraft.click()
    # Once drafted, the message matches the selection again.
    await expect(redraft).to_have_text('Draft again')
    await expect(stale).to_have_count(0)
    assert 'third.txt' not in git_ui.drafter.prompts[-1]
    await expect(dialog.get_by_role('button', name='Commit 2 of 3')).to_be_enabled()

    # A message the user typed is theirs: no warning about it.
    await message.fill('Edit two files')
    await dialog.get_by_role('checkbox', name='second.txt').uncheck()
    await expect(dialog.get_by_text('This message was drafted for')).to_have_count(0)


async def test_changes_panel_git_scopes_say_which_files_they_leave_out(
    page: Page, make_fleet: MakeFleet, serve: Serve, repo: Path
) -> None:
    fleet = make_fleet(agents(Drafter(), 'repo'))
    conversation = fleet.new('repo')
    (repo / '.git' / 'info' / 'exclude').write_text('*.log\n')
    (repo / 'debug.log').write_text('ignored\n')
    # Saved, so it can be renamed, as the sidebar lists it.
    fleet.portal.call(fleet.runtimes['repo'].session.commit_messages, list[ModelMessage]())
    assert fleet.post(f'/api/sessions/{conversation}/rename', {'title': 'Clean Repo'}).status_code == 204
    with serve(fleet) as url:
        await page.goto(f'{url}/?token={TOKEN}')
        await page.locator('.sidebar').get_by_text('Clean Repo', exact=True).click()
        await expect(page.locator('.session.opening')).to_have_count(0)
        await page.get_by_role('button', name=re.compile(r'^Changes')).click()
        panel = page.get_by_role('complementary', name='Changes')
        await panel.get_by_role('tab', name='Branch vs base').click()
        await expect(panel.locator('.changes-empty')).to_have_text(
            'No tracked file differs from where this branch left its base. Untracked files are not shown.'
        )
        await panel.get_by_role('tab', name='Uncommitted').click()
        await expect(panel.locator('.changes-empty')).to_have_text(
            'No file differs from the last commit. Ignored files are not shown.'
        )

        (repo / 'new.txt').write_text('hi\n')
        await panel.get_by_role('button', name='Refresh').click()
        await expect(panel.get_by_role('list', name='Changed files').get_by_role('listitem')).to_have_text(
            ['Anew.txt+1 −0']
        )


def mid_turn_editor(repo: Path, released: anyio.Event) -> Agent[None, str]:
    """An agent whose turn writes `old.txt` in `repo`, then runs until `released` is set."""

    async def turn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        if isinstance(messages[-1].parts[-1], UserPromptPart):
            arguments = json.dumps({'path': 'old.txt', 'content': 'edited mid-turn\n'})
            yield {0: DeltaToolCall(name='write_file', json_args=arguments, tool_call_id='edit')}
        else:
            await released.wait()
            yield 'Done'

    agent = Agent(FunctionModel(stream_function=turn))

    @agent.tool_plain(tool_kind=WRITE_FILE_KIND)
    def write_file(path: str, content: str) -> str:
        (repo / path).write_text(content)
        return f'Wrote {path}.'

    return agent


async def test_a_running_turns_edits_show_before_it_ends(
    page: Page, make_fleet: MakeFleet, serve: Serve, repo: Path
) -> None:
    released: list[anyio.Event] = []

    def editors(portal: BlockingPortal) -> dict[str, Agent[None, str]]:
        released.append(portal.call(anyio.Event))
        return {'repo': mid_turn_editor(repo, released[0])}

    fleet = make_fleet(editors)
    conversation = fleet.new('repo')
    fleet.portal.call(fleet.runtimes['repo'].session.commit_messages, list[ModelMessage]())
    assert fleet.post(f'/api/sessions/{conversation}/rename', {'title': 'Git Repo'}).status_code == 204
    (repo / 'notes.txt').write_text('edited by the user\n')
    with serve(fleet) as url:
        await page.goto(f'{url}/?token={TOKEN}')
        await page.locator('.sidebar').get_by_text('Git Repo', exact=True).click()
        await expect(page.locator('.session.opening')).to_have_count(0)
        chip = page.locator('.topbar .git-chip')
        await expect(chip).to_have_text('main1 changed, 0 ahead')
        await page.get_by_role('button', name=re.compile(r'^Changes')).click()
        panel = page.get_by_role('complementary', name='Changes')
        await panel.get_by_role('tab', name='Uncommitted').click()
        files = panel.get_by_role('list', name='Changed files').get_by_role('listitem')
        await expect(files).to_have_text(['Mnotes.txt+1 −1'])

        composer = page.get_by_role('combobox', name='Message CLAI')
        await composer.fill('edit old.txt, then run a long command')
        await composer.press('Enter')
        activity = page.locator('.topbar-status .activity')
        try:
            await expect(page.locator('.tool').filter(has_text='old.txt')).to_have_count(1)
            # Each finished tool call asks again, so the chip, the header's count, and the panel follow the turn.
            await expect(chip).to_have_text('main2 changed, 0 ahead', timeout=2_000)
            await expect(files).to_have_text(['Mnotes.txt+1 −1', 'Mold.txt+1 −1'], timeout=2_000)
            await expect(page.get_by_role('button', name='Changes: 1 file changed')).to_be_visible(timeout=2_000)
            await expect(activity).not_to_have_text('ready')
        finally:
            fleet.portal.call(released[0].set)
        await expect(activity).to_have_text('ready')
