"""`/commit`, `/push`, `/pr`, and the web UI's git routes, against real repositories and a stand-in `gh`."""

import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
from anyio.from_thread import BlockingPortal

from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_clai2 import gh_cli
from pydantic_clai2.commands import Command, Commands
from pydantic_clai2.runtime import git_actions
from pydantic_clai2.runtime.git_actions import (
    ChangedFile,
    GitError,
    GitStatus,
    commit,
    draft_commit_message,
    draft_pull_request,
    git_commands,
    git_status,
    github_host,
    open_pull_request,
    push,
    summary,
)
from tests.clai2.test_web_sessions import MakeFleet, make_fleet

__all__ = ['make_fleet']

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(sys.platform == 'win32', reason='the stand-in `gh` and the hooks are POSIX shell scripts'),
]

GITHUB = 'https://github.com/octo/repo.git'
PR_URL = 'https://github.com/octo/repo/pull/7'
FAKE_GH = f"""#!/bin/sh
echo "$@" >> "$FAKE_GH_STATE/calls"
if [ "$1 $2" = "auth token" ]; then
  [ -f "$FAKE_GH_STATE/token" ] && cat "$FAKE_GH_STATE/token" && exit 0
  echo "no oauth token" >&2
  exit 1
fi
if [ "$1 $2" = "pr create" ]; then
  cat > "$FAKE_GH_STATE/body"
  [ -f "$FAKE_GH_STATE/fail" ] && cat "$FAKE_GH_STATE/fail" >&2 && exit 1
  [ -f "$FAKE_GH_STATE/silent" ] && exit 0
  echo "Creating pull request for feature into main in octo/repo"
  echo "{PR_URL}"
  exit 0
fi
exit 2
"""


def git(repo: Path, *args: str) -> str:
    ran = subprocess.run(['git', '-C', str(repo), *args], check=False, capture_output=True, text=True)
    assert ran.returncode == 0, ran.stderr
    return ran.stdout


@pytest.fixture
def gh_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A signed-in stand-in `gh`, and a Git that reads no user or system configuration."""
    state = tmp_path / 'gh-state'
    state.mkdir()
    (state / 'token').write_text('gho_test')
    gh = tmp_path / 'gh'
    gh.write_text(FAKE_GH)
    gh.chmod(0o755)
    monkeypatch.setenv('FAKE_GH_STATE', str(state))
    # A shell script, unlike the autouse fake's Python one, starts no interpreter.
    monkeypatch.setattr(gh_cli, 'gh_command', lambda: [str(gh)])
    config = tmp_path / 'gitconfig'
    config.write_text('')
    monkeypatch.setenv('GIT_CONFIG_GLOBAL', str(config))
    monkeypatch.setenv('GIT_CONFIG_NOSYSTEM', '1')
    for role in ('AUTHOR', 'COMMITTER'):
        monkeypatch.setenv(f'GIT_{role}_NAME', 'Test')
        monkeypatch.setenv(f'GIT_{role}_EMAIL', 'test@example.com')
    return state


@pytest.fixture
def repo(tmp_path: Path, gh_state: Path) -> Path:
    """`repo` on `main`, pushed to a GitHub `origin` whose pushes land in a local bare repository."""
    remote = tmp_path / 'remote.git'
    subprocess.run(['git', 'init', '--bare', '-b', 'main', str(remote)], check=True, capture_output=True)
    work = tmp_path / 'repo'
    subprocess.run(['git', 'init', '-b', 'main', str(work)], check=True, capture_output=True)
    git(work, 'remote', 'add', 'origin', GITHUB)
    git(work, 'config', f'url.{remote}.pushInsteadOf', GITHUB)
    (work / 'notes.txt').write_text('one\n')
    (work / 'old.txt').write_text('old\n')
    (work / 'gone.txt').write_text('gone\n')
    git(work, 'add', '.')
    git(work, 'commit', '-m', 'init')
    git(work, 'push', '-u', 'origin', 'main')
    return work


def calls(state: Path) -> list[str]:
    path = state / 'calls'
    return path.read_text().splitlines() if path.exists() else []


async def test_status_outside_a_repository_or_without_its_directory(tmp_path: Path, gh_state: Path) -> None:
    assert await git_status(tmp_path) is None
    assert summary(None) == 'Not in a Git repository.'
    with pytest.raises(LookupError, match='no longer exists'):
        await git_status(tmp_path / 'missing')
    with pytest.raises(ValueError, match='is not in a Git repository'):
        await commit(tmp_path, message='nothing')


async def test_status_lists_each_kind_of_change(repo: Path) -> None:
    (repo / 'notes.txt').write_text('one\ntwo\n')
    (repo / 'new.txt').write_text('new\n')
    (repo / 'gone.txt').unlink()
    git(repo, 'mv', 'old.txt', 'moved.txt')
    (repo / 'staged.txt').write_text('staged\n')
    git(repo, 'add', 'staged.txt')
    status = await git_status(repo)
    assert status is not None
    assert sorted(status.files, key=lambda file: file.path) == [
        ChangedFile(path='gone.txt', status='D'),
        ChangedFile(path='moved.txt', status='R', original='old.txt'),
        ChangedFile(path='new.txt', status='?'),
        ChangedFile(path='notes.txt', status='M'),
        ChangedFile(path='staged.txt', status='A'),
    ]
    assert (status.branch, status.upstream, status.ahead, status.behind) == ('main', 'origin/main', 0, 0)
    assert status.default_branch
    assert status.push is None
    assert status.pull_request == 'main is the default branch: open a pull request from another branch.'


async def test_a_conflict_is_unmerged(repo: Path) -> None:
    git(repo, 'checkout', '-b', 'other')
    (repo / 'notes.txt').write_text('other\n')
    git(repo, 'commit', '-am', 'other')
    git(repo, 'checkout', 'main')
    (repo / 'notes.txt').write_text('main\n')
    git(repo, 'commit', '-am', 'main')
    subprocess.run(['git', '-C', str(repo), 'merge', 'other'], check=False, capture_output=True)
    status = await git_status(repo)
    assert status is not None
    assert status.files == [ChangedFile(path='notes.txt', status='U')]
    assert status.ahead == 1, 'main is ahead of origin/main'


async def test_commit_only_the_chosen_files_then_the_chip_counts_the_commit(repo: Path) -> None:
    """The issue's acceptance: after the agent edits files, committing them leaves `0 changed, 1 ahead`."""
    git(repo, 'checkout', '-b', 'feature')
    (repo / 'notes.txt').write_text('one\ntwo\n')
    (repo / 'new.txt').write_text('new\n')
    (repo / 'gone.txt').unlink()
    (repo / 'later.txt').write_text('later\n')
    git(repo, 'add', 'later.txt')
    status = await git_status(repo)
    assert status is not None
    assert summary(status) == 'feature: 4 changed, 0 ahead'
    assert (status.upstream, status.default_branch, status.push, status.pull_request) == (None, False, None, None)

    done = await commit(repo, message='  Edit the notes\n', files=['notes.txt', 'new.txt', 'gone.txt', 'new.txt'])
    assert done.startswith('Committed ') and done.endswith(' Edit the notes')
    assert git(repo, 'show', '--name-status', '--format=%B', 'HEAD').split() == [
        'Edit',
        'the',
        'notes',
        'D',
        'gone.txt',
        'A',
        'new.txt',
        'M',
        'notes.txt',
    ]
    status = await git_status(repo)
    assert status is not None
    assert status.files == [ChangedFile(path='later.txt', status='A')], 'a file staged but not chosen stays out'
    await commit(repo, message='Later')
    status = await git_status(repo)
    assert status is not None
    assert summary(status) == 'feature: 0 changed, 2 ahead'


async def test_commit_staged_renames_deletions_and_awkward_names_from_a_subdirectory(repo: Path) -> None:
    """Paths are the repository's, whatever directory the session works in, and none is read as an option."""
    git(repo, 'mv', 'old.txt', 'moved.txt')
    git(repo, 'rm', '--quiet', 'gone.txt')
    (repo / '-dash name.txt').write_text('dash\n')
    inside = repo / 'sub'
    inside.mkdir()
    (inside / 'deep.txt').write_text('deep\n')
    status = await git_status(inside)
    assert status is not None
    assert sorted(file.path for file in status.files) == ['-dash name.txt', 'gone.txt', 'moved.txt', 'sub/deep.txt']
    await commit(inside, message='Move, remove, and add')
    assert sorted(git(repo, 'show', '--name-status', '--format=', 'HEAD').splitlines()) == [
        'A\t-dash name.txt',
        'A\tsub/deep.txt',
        'D\tgone.txt',
        'R100\told.txt\tmoved.txt',
    ]
    status = await git_status(inside)
    assert status is not None and status.files == []
    git(repo, 'rm', '--quiet', 'notes.txt')
    await commit(repo, message='Only a removal')
    assert git(repo, 'show', '--name-status', '--format=', 'HEAD').splitlines() == ['D\tnotes.txt']


async def test_an_upstream_branch_gone_from_the_remote_is_pushed_again(repo: Path) -> None:
    git(repo, 'checkout', '-b', 'feature')
    (repo / 'notes.txt').write_text('feature\n')
    await commit(repo, message='Feature')
    await push(repo)
    git(repo, 'update-ref', '-d', 'refs/remotes/origin/feature')
    status = await git_status(repo)
    assert status is not None
    assert (status.upstream, status.ahead) == (None, 1)


async def test_the_default_branch_may_be_master(tmp_path: Path, gh_state: Path) -> None:
    work = tmp_path / 'old'
    subprocess.run(['git', 'init', '-b', 'master', str(work)], check=True, capture_output=True)
    git(work, 'commit', '--allow-empty', '-m', 'init')
    git(work, 'remote', 'add', 'origin', GITHUB)
    git(work, 'update-ref', 'refs/remotes/origin/master', 'HEAD')
    status = await git_status(work)
    assert status is not None
    assert status.default_branch
    assert status.pull_request == 'master is the default branch: open a pull request from another branch.'
    # A clone records the remote's default branch, whatever its name.
    git(work, 'update-ref', 'refs/remotes/origin/trunk', 'HEAD')
    git(work, 'symbolic-ref', 'refs/remotes/origin/HEAD', 'refs/remotes/origin/trunk')
    git(work, 'checkout', '-q', '-b', 'trunk')
    status = await git_status(work)
    assert status is not None
    assert (status.branch, status.default_branch) == ('trunk', True)


def test_new_files_are_read_as_git_commits_them(tmp_path: Path) -> None:
    read_start = git_actions._read_start  # pyright: ignore[reportPrivateUsage]
    assert read_start(tmp_path) == '(not a regular file)'
    locked = tmp_path / 'locked.txt'
    locked.write_text('secret')
    locked.chmod(0)
    try:
        assert read_start(locked).startswith('(unreadable: ') or os.geteuid() == 0
    finally:
        locked.chmod(0o600)


async def test_commit_refuses_what_it_cannot_commit(repo: Path) -> None:
    with pytest.raises(ValueError, match='Write a commit message first'):
        await commit(repo, message='  ')
    with pytest.raises(ValueError, match='nothing to commit'):
        await commit(repo, message='Nothing')
    with pytest.raises(ValueError, match=r'notes.txt has no changes to commit'):
        await commit(repo, message='Nothing', files=['notes.txt'])


async def test_a_failing_hook_says_why(repo: Path) -> None:
    hook = repo / '.git' / 'hooks' / 'pre-commit'
    hook.write_text('#!/bin/sh\necho "lint: notes.txt is not formatted" >&2\nexit 1\n')
    hook.chmod(0o755)
    (repo / 'notes.txt').write_text('changed\n')
    with pytest.raises(GitError, match=r'lint: notes.txt is not formatted'):
        await commit(repo, message='Change the notes')


async def test_push_confirms_the_default_branch_and_sets_the_upstream(repo: Path) -> None:
    (repo / 'notes.txt').write_text('changed\n')
    await commit(repo, message='Change the notes')
    with pytest.raises(RuntimeError, match='main is the default branch: confirm to push to it'):
        await push(repo)
    assert await push(repo, confirmed=True) == 'Pushed main to origin.'

    git(repo, 'checkout', '-b', 'feature')
    (repo / 'notes.txt').write_text('feature\n')
    await commit(repo, message='Feature')
    assert await push(repo) == 'Pushed feature to origin.'
    status = await git_status(repo)
    assert status is not None
    assert (status.upstream, status.ahead) == ('origin/feature', 0)

    git(repo, 'checkout', '--detach')
    status = await git_status(repo)
    assert status is not None
    assert status.branch is None
    assert summary(status) == 'detached HEAD: 0 changed, 0 ahead'
    with pytest.raises(RuntimeError, match='HEAD is detached'):
        await push(repo)
    with pytest.raises(RuntimeError, match='HEAD is detached'):
        await open_pull_request(repo, title='Detached', body='')


async def test_push_needs_an_origin(tmp_path: Path, gh_state: Path) -> None:
    subprocess.run(['git', 'init', '-b', 'feature', str(tmp_path / 'local')], check=True, capture_output=True)
    status = await git_status(tmp_path / 'local')
    assert status is not None
    assert status.push == status.pull_request == 'This repository has no origin remote to push to.'
    assert (status.ahead, status.default_branch) == (0, False)


async def test_pull_requests_need_github_and_a_signed_in_gh(
    repo: Path, gh_state: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    git(repo, 'checkout', '-b', 'feature')
    (gh_state / 'token').unlink()
    status = await git_status(repo)
    assert status is not None
    assert status.push is None
    assert (
        status.pull_request == 'The GitHub CLI is not signed in to github.com: run gh auth login --hostname github.com.'
    )
    with pytest.raises(RuntimeError, match='not signed in'):
        await open_pull_request(repo, title='Feature', body='')

    def locked(host: str) -> str | None:
        raise UserError('gh auth token did not answer within 10 seconds.')

    (gh_state / 'token').write_text('gho_test')
    monkeypatch.setattr(gh_cli, 'gh_token', locked)
    status = await git_status(repo)
    assert status is not None
    assert status.pull_request == 'gh auth token did not answer within 10 seconds.'
    assert await push(repo) == 'Pushed feature to origin.', 'pushing never asks gh'

    monkeypatch.setattr(gh_cli, 'gh_command', lambda: None)
    status = await git_status(repo)
    assert status is not None
    assert status.pull_request is not None and status.pull_request.startswith('The GitHub CLI (gh) is not installed.')

    git(repo, 'remote', 'set-url', 'origin', str(repo.parent / 'remote.git'))
    status = await git_status(repo)
    assert status is not None
    assert (status.push, status.pull_request) == (None, 'origin is not a GitHub remote.')


def test_github_host() -> None:
    assert github_host('https://github.com/octo/repo.git') == 'github.com'
    assert github_host('ssh://git@github.example.com/octo/repo.git') == 'github.example.com'
    assert github_host('git@github.com:octo/repo.git') == 'github.com'
    assert github_host('/srv/git/repo.git') is None


async def test_open_a_pull_request(repo: Path, gh_state: Path) -> None:
    with pytest.raises(ValueError, match='Write a title'):
        await open_pull_request(repo, title=' ', body='')
    git(repo, 'checkout', '-b', 'feature')
    (repo / 'notes.txt').write_text('feature\n')
    await commit(repo, message='Feature')

    assert await open_pull_request(repo, title=' Add a feature ', body='It adds one.\n') == PR_URL
    assert calls(gh_state)[-1] == 'pr create --head=feature --title=Add a feature --body-file=-'
    assert (gh_state / 'body').read_text() == 'It adds one.'
    status = await git_status(repo)
    assert status is not None
    assert (status.upstream, status.ahead) == ('origin/feature', 0), 'the branch was pushed first'

    # Pushed already: `gh` is asked at once, and what it prints when it fails is the error.
    (gh_state / 'fail').write_text('a pull request for branch "feature" already exists')
    with pytest.raises(GitError, match='already exists'):
        await open_pull_request(repo, title='Again', body='')
    (gh_state / 'fail').unlink()
    (gh_state / 'silent').write_text('')
    with pytest.raises(GitError, match='printed no pull request URL'):
        await open_pull_request(repo, title='Again', body='')


@pytest.mark.skipif(sys.platform == 'darwin', reason='macOS file systems refuse names that are not UTF-8')
async def test_a_file_name_that_is_not_utf8_is_shown_and_committed(repo: Path) -> None:
    raw = os.fsdecode(b'latin\xe9.txt')
    (repo / raw).write_text('latin-1 name\n')
    (repo / 'notes.txt').write_text('changed\n')
    status = await git_status(repo)
    assert status is not None
    assert sorted(file.path for file in status.files) == ['latin\\xe9.txt', 'notes.txt']
    drafter = Drafter()
    await draft_commit_message(repo, model=drafter.resolved, files=['latin\\xe9.txt'])
    assert drafter.prompts[-1] == 'New file latin\\xe9.txt:\nlatin-1 name\n'
    await commit(repo, message='Add the file', files=['latin\\xe9.txt'])
    assert git(repo, 'show', '--name-only', '--format=', 'HEAD').strip() == '"latin\\351.txt"'
    await commit(repo, message='Change the notes')
    status = await git_status(repo)
    assert status is not None and status.files == []


async def test_a_workspace_gone_while_git_runs_is_reported_gone(tmp_path: Path) -> None:
    run = git_actions._run  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(LookupError, match='no longer exists'):
        await run(tmp_path / 'missing', 'git', 'status')
    (tmp_path / 'file').write_text('')
    with pytest.raises(LookupError, match='no longer exists'):
        await run(tmp_path / 'file', 'git', 'status')
    program = tmp_path / 'not-executable'
    program.write_text('')
    with pytest.raises(PermissionError):
        await run(tmp_path, str(program))


async def test_run_failures(tmp_path: Path) -> None:
    with pytest.raises(GitError, match=r'sleep 5 did not finish within 0.01 seconds'):
        await git_actions._run(tmp_path, 'sleep', '5', timeout=0.01)  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(GitError, match='no-such-command-here is not installed'):
        await git_actions._run(tmp_path, 'no-such-command-here', 'x')  # pyright: ignore[reportPrivateUsage]


class Drafter:
    """A model that answers a commit message in a code fence, or a pull request, and keeps what it was shown."""

    def __init__(self, *, fenced: bool = True) -> None:
        self.fenced = fenced
        self.prompts: list[str] = []
        self.model = FunctionModel(self.respond)

    def respond(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prompt = messages[-1].parts[-1]
        assert isinstance(prompt, UserPromptPart) and isinstance(prompt.content, str)
        self.prompts.append(prompt.content)
        if info.output_tools:
            return ModelResponse(
                parts=[ToolCallPart(info.output_tools[0].name, {'title': 'Add a feature', 'body': 'It adds one.'})]
            )
        message = 'Edit the notes\n\nAnd add one.'
        return ModelResponse(parts=[TextPart(f'```text\n{message}\n```' if self.fenced else f'  {message}\n')])

    async def resolved(self) -> FunctionModel:
        return self.model


async def no_model() -> None:
    return None


async def test_draft_a_commit_message_from_the_changes(repo: Path) -> None:
    drafter = Drafter()
    with pytest.raises(ValueError, match='nothing to commit'):
        await draft_commit_message(repo, model=drafter.resolved)
    (repo / 'notes.txt').write_text('one\ntwo\n')
    (repo / 'new.txt').write_text('fresh content\n')
    (repo / 'big.txt').write_text('x' * (git_actions.NEW_FILE_LIMIT + 10))
    (repo / 'image.bin').write_bytes(b'\x89PNG\0\0')
    (repo / 'dangling').symlink_to(repo / 'nowhere')
    with pytest.raises(RuntimeError, match='Choose a model with /model'):
        await draft_commit_message(repo, model=no_model)

    assert await draft_commit_message(repo, model=drafter.resolved) == 'Edit the notes\n\nAnd add one.'
    [shown] = drafter.prompts
    assert '+two' in shown
    assert 'New file new.txt:\nfresh content' in shown
    assert '[... the rest is cut]' in shown
    assert 'New file image.bin:\n(binary)' in shown
    assert f'New file dangling:\n(symbolic link to {repo / "nowhere"})' in shown

    await draft_commit_message(repo, model=drafter.resolved, files=['new.txt'])
    assert drafter.prompts[-1] == 'New file new.txt:\nfresh content\n'


async def test_draft_a_commit_message_before_the_first_commit(tmp_path: Path, gh_state: Path) -> None:
    fresh = tmp_path / 'fresh'
    subprocess.run(['git', 'init', '-b', 'main', str(fresh)], check=True, capture_output=True)
    (fresh / 'first.txt').write_text('first\n')
    git(fresh, 'add', 'first.txt')
    drafter = Drafter(fenced=False)
    assert await draft_commit_message(fresh, model=drafter.resolved) == 'Edit the notes\n\nAnd add one.'
    assert '+first' in drafter.prompts[-1]


async def test_draft_a_pull_request_from_the_branch(repo: Path, tmp_path: Path) -> None:
    drafter = Drafter()
    git(repo, 'checkout', '-b', 'feature')
    (repo / 'notes.txt').write_text('feature\n')
    await commit(repo, message='Add a feature')
    draft = await draft_pull_request(repo, model=drafter.resolved)
    assert (draft.title, draft.body) == ('Add a feature', 'It adds one.')
    assert drafter.prompts[-1].startswith('Commits:\nAdd a feature\n')
    assert '+feature' in drafter.prompts[-1]

    # Without the remote's branch, the commits alone.
    git(repo, 'remote', 'remove', 'origin')
    await draft_pull_request(repo, model=drafter.resolved)
    assert 'init' in drafter.prompts[-1] and drafter.prompts[-1].endswith('Diff:\n')
    with pytest.raises(LookupError):
        await draft_pull_request(tmp_path / 'missing', model=drafter.resolved)


def test_summary_counts_commits_behind() -> None:
    status = GitStatus(
        branch='main',
        upstream='origin/main',
        ahead=1,
        behind=2,
        files=[],
        default_branch=True,
        push=None,
        pull_request=None,
    )
    assert summary(status) == 'main: 0 changed, 1 ahead, 2 behind'


async def test_commands(repo: Path, gh_state: Path) -> None:
    drafter = Drafter()
    commands = {command.name: command for command in git_commands(workspace=lambda: repo, model=drafter.resolved)}
    assert commands['push'].complete([]) == ('confirm',)
    assert commands['push'].complete(['confirm', 'x']) == ()

    async def run(name: str, *args: str) -> str:
        result = commands[name].handler(list(args))
        assert not isinstance(result, str), 'every git command waits on git'
        return await result

    (repo / 'notes.txt').write_text('two\n')
    assert (await run('commit', 'Change the notes')).endswith('Change the notes\nmain: 0 changed, 1 ahead')
    (repo / 'notes.txt').write_text('three\n')
    assert (await run('commit')).endswith('Edit the notes\nmain: 0 changed, 2 ahead')

    with pytest.raises(ValueError, match='Usage: /push'):
        await run('push', 'now')
    with pytest.raises(ValueError, match=r'main is the default branch: /push confirm pushes to it'):
        await run('push')
    assert await run('push', 'confirm') == 'Pushed main to origin.\nmain: 0 changed, 0 ahead'

    git(repo, 'checkout', '-b', 'feature')
    (repo / 'notes.txt').write_text('feature\n')
    await run('commit', 'Feature')
    assert await run('push') == 'Pushed feature to origin.\nfeature: 0 changed, 0 ahead'
    assert await run('pr') == f'Opened {PR_URL}'
    assert calls(gh_state)[-1] == 'pr create --head=feature --title=Add a feature --body-file=-'
    assert await run('pr', 'My title') == f'Opened {PR_URL}'
    assert calls(gh_state)[-1] == 'pr create --head=feature --title=My title --body-file=-'


def test_a_plugins_own_git_command_takes_over() -> None:
    commands = Commands()
    built_in = git_commands(workspace=Path.cwd, model=no_model)
    commands.register_many(built_in)
    own = Command(name='pr', description='Open a pull request my way', handler=lambda args: 'mine')
    assert commands.register_many([own]) == [built_in[2]]
    assert commands.get('pr') is own


def agents(drafter: Drafter, *names: str) -> Callable[[BlockingPortal], dict[str, Agent[None, str]]]:
    """A fleet's agents: one drafting with `drafter` in each folder of `names`."""
    return lambda _portal: {name: Agent(drafter.model) for name in names}


def test_the_web_ui_commits_pushes_and_opens_a_pull_request(
    make_fleet: MakeFleet, tmp_path: Path, repo: Path, gh_state: Path
) -> None:
    drafter = Drafter()
    fleet = make_fleet(agents(drafter, 'repo', 'plain'))
    conversation = fleet.new('repo')
    plain = fleet.new('plain')
    git(repo, 'checkout', '-b', 'feature')
    for name in ('notes.txt', 'second.txt', 'third.txt'):
        (repo / name).write_text('edited by the agent\n')
    assert 'commit' in fleet.runtimes['repo'].commands

    assert fleet.get(f'/api/sessions/{plain}/git').json() is None
    assert fleet.get('/api/sessions/unknown/git').status_code == 404
    base = f'/api/sessions/{conversation}/git'
    status = fleet.get(base).json()
    assert (status['branch'], len(status['files']), status['ahead'], status['push'], status['pull_request']) == (
        'feature',
        3,
        0,
        None,
        None,
    )

    assert fleet.post(f'{base}/commit/draft', {'files': ['notes.txt']}).json() == {
        'message': 'Edit the notes\n\nAnd add one.'
    }
    assert fleet.post(f'{base}/commit/draft', {'files': ['missing.txt']}).status_code == 400
    response = fleet.post(f'{base}/commit', {'message': 'Edit three files', 'files': None})
    assert response.status_code == 200, response.text
    done = response.json()
    assert done['message'].endswith(' Edit three files')
    assert (len(done['status']['files']), done['status']['ahead']) == (0, 1)

    hook = repo / '.git' / 'hooks' / 'pre-commit'
    hook.write_text('#!/bin/sh\necho "hook says no" >&2\nexit 1\n')
    hook.chmod(0o755)
    (repo / 'notes.txt').write_text('again\n')
    refused = fleet.post(f'{base}/commit', {'message': 'Again'})
    assert (refused.status_code, refused.json()['detail']) == (409, 'hook says no')
    hook.unlink()
    git(repo, 'checkout', 'HEAD', '--', 'notes.txt')

    pushed = fleet.post(f'{base}/push', {}).json()
    assert (pushed['message'], pushed['status']['upstream']) == ('Pushed feature to origin.', 'origin/feature')
    assert fleet.post(f'{base}/pr/draft').json() == {'title': 'Add a feature', 'body': 'It adds one.'}
    assert fleet.post(f'{base}/pr', {'title': 'Add a feature', 'body': 'It adds one.'}).json() == {'url': PR_URL}

    git(repo, 'checkout', 'main')
    refused = fleet.post(f'{base}/push', {'confirmed': False})
    assert (refused.status_code, refused.json()['detail']) == (
        409,
        'main is the default branch: confirm to push to it.',
    )
    assert fleet.post(f'{base}/push', {'confirmed': True}).status_code == 200

    # A session whose folder is gone runs nothing there.
    shutil.rmtree(repo)
    assert fleet.get(base).status_code == 404
    assert fleet.post(f'{base}/commit', {'message': 'Gone'}).status_code == 404
