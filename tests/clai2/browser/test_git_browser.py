"""The header's git chip and its dialog in a real browser: commit, push, and open a pull request."""

import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from playwright.async_api import Page, expect

from pydantic_ai.messages import ModelMessage
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
        yield GitUI(url=url, conversation=conversation)


async def test_commit_push_and_open_a_pull_request_from_the_header(
    page: Page, git_ui: GitUI, repo: Path, gh_state: Path
) -> None:
    await page.goto(f'{git_ui.url}/?token={TOKEN}')
    await page.locator('.sidebar').get_by_text('Git Repo', exact=True).click()
    await expect(page.locator('.session.current .session-title')).to_have_text('Git Repo')
    await expect(page.locator('.session.opening')).to_have_count(0)
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
    assert calls(gh_state)[-1] == 'pr create --head=feature --title=Edit three files --body-file=-'

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
