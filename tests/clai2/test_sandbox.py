"""`/sandbox` moves a new conversation into a cloud sandbox holding a copy of the checkout.

A fake provider makes each sandbox a local folder, so the real seed script rebuilds the checkout there
with real Git. Its global Git config sends the GitHub URL to a local bare repository, as the provider's
proxy would send it to GitHub.
"""

import base64
import importlib.machinery
import io
import itertools
import json
import subprocess
import sys
import uuid
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from dataclasses import dataclass, field
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, cast

import anyio
import pytest
from anyio import to_thread
from rich.console import Console

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.workspaces import LocalWorkspaceBackend, WorkspaceRef
from pydantic_clai2 import Session
from pydantic_clai2.builtin_plugins import sandbox
from pydantic_clai2.builtin_plugins.sandbox import SandboxPlugin, history_sandbox
from pydantic_clai2.builtin_plugins.sandbox.providers import (
    GITHUB_HOSTS,
    GitHubCredential,
    boxd as boxd_provider,
    e2b as e2b_provider,
)
from pydantic_clai2.builtin_plugins.sandbox.providers.base import SANDBOX_ENV
from pydantic_clai2.builtin_plugins.sandbox.providers.e2b import E2BProvider
from pydantic_clai2.builtin_plugins.sandbox.seed import SandboxBackend, Snapshot, github_clone_url, seed, snapshot
from pydantic_clai2.config.api_keys import load_keys, save_key
from pydantic_clai2.plugins import (
    ConversationChanged,
    LoadedPlugin,
    PluginHost,
    SessionEnd,
    Transcript,
    load_plugin,
)
from pydantic_clai2.runtime.imported_history import imported_kind
from pydantic_clai2.runtime.rewind import rewind_points
from tests.clai2.dialog_script import Back, Pick, ScriptedDialogs, Type

if TYPE_CHECKING:
    # Imported at run time only by the fixtures that need them, so collecting this file loads no SDK.
    from boxd import Machine, Secret

    from tests.harness.boxd_sandbox.fake_boxd import FakeBoxd, FakeClient
    from tests.harness.e2b_sandbox.fake_e2b import FakeE2B

CLONE_URL = 'https://github.com/owner/repo.git'


def git(directory: Path, *args: str) -> str:
    result = subprocess.run(['git', '-C', str(directory), *args], capture_output=True, text=True, check=True)
    return result.stdout.rstrip()


class FakeSandboxBackend(LocalWorkspaceBackend):
    """A local folder that reports the fake provider's reference, as a cloud sandbox's backend would."""

    def __init__(self, ref: WorkspaceRef, directory: Path, *, env: Mapping[str, str]) -> None:
        super().__init__(directory, env=env)
        self._sandbox_ref = ref

    @property
    def ref(self) -> WorkspaceRef:
        return self._sandbox_ref


@dataclass
class FakeProvider:
    """Sandboxes as folders under `root`, whose Git sends `https://github.com/owner/` to `upstream`'s folder."""

    root: Path
    upstream: Path
    name: str = 'fake'
    title: str = 'Fake'
    working_dir: str = '/home/user/workspace'
    requirement: str | None = None
    created: list[GitHubCredential | None] = field(default_factory=list[GitHubCredential | None])
    reopened: list[WorkspaceRef] = field(default_factory=list[WorkspaceRef])
    deleted: list[WorkspaceRef] = field(default_factory=list[WorkspaceRef])
    closed: list[WorkspaceRef] = field(default_factory=list[WorkspaceRef])
    creating: anyio.Event | None = None
    """When set, `create` waits for it after starting, as a slow provider would."""
    delete_error: Exception | None = None
    envs: list[tuple[WorkspaceRef, str, str]] = field(default_factory=list[tuple[WorkspaceRef, str, str]])
    env_error: RuntimeError | None = None

    def missing(self) -> str | None:
        return self.requirement

    async def create(self, *, github: GitHubCredential | None) -> tuple[WorkspaceRef, FakeSandboxBackend]:
        self.created.append(github)
        if self.creating is not None:
            await self.creating.wait()
        env = {**SANDBOX_ENV, 'GH_TOKEN': 'stand-in'} if github is not None else SANDBOX_ENV
        ref = WorkspaceRef(provider=self.name, id=f'sbx-{len(self.created)}')
        folder = anyio.Path(self.folder(ref))
        await folder.mkdir()
        # A fresh global config per sandbox, as a new sandbox has.
        rewrite = f'[url "{self.upstream.parent}/"]\n\tinsteadOf = https://github.com/owner/\n'
        await anyio.Path(f'{folder}.gitconfig').write_text(rewrite, encoding='utf-8')
        return ref, self._backend(ref, env)

    def reopen(self, ref: WorkspaceRef) -> FakeSandboxBackend:
        self.reopened.append(ref)
        return self._backend(ref, {})

    async def set_env(self, backend: FakeSandboxBackend, name: str, value: str) -> None:
        if self.env_error is not None:
            raise self.env_error
        self.envs.append((backend.ref, name, value))

    async def delete(self, ref: WorkspaceRef) -> None:
        if self.delete_error is not None:
            raise self.delete_error
        self.deleted.append(ref)

    async def close(self, backend: FakeSandboxBackend) -> None:
        self.closed.append(backend.ref)

    def folder(self, ref: WorkspaceRef) -> Path:
        return self.root / ref.id

    def _backend(self, ref: WorkspaceRef, env: Mapping[str, str]) -> FakeSandboxBackend:
        git_env = {'GIT_CONFIG_GLOBAL': f'{self.folder(ref)}.gitconfig', 'GIT_CONFIG_NOSYSTEM': '1'}
        return FakeSandboxBackend(ref, self.folder(ref), env={**env, **git_env})


@pytest.fixture(autouse=True)
def isolated_git_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the machine's Git settings out, such as CI's rewrite of GitHub URLs to carry a token.

    CI sets that rewrite with `GIT_CONFIG_COUNT` and its `GIT_CONFIG_KEY_n`/`GIT_CONFIG_VALUE_n`, which Git reads
    after every config file, so emptying the files alone leaves it in place.
    """
    config = tmp_path / 'global.gitconfig'
    config.touch()
    monkeypatch.setenv('GIT_CONFIG_GLOBAL', str(config))
    monkeypatch.setenv('GIT_CONFIG_NOSYSTEM', '1')
    monkeypatch.delenv('GIT_CONFIG_COUNT', raising=False)
    monkeypatch.delenv('GIT_CONFIG_PARAMETERS', raising=False)


@pytest.fixture
def upstream(tmp_path: Path) -> Path:
    """A bare repository standing in for `CLONE_URL` on GitHub, with one commit on `main`."""
    seedling = tmp_path / 'seedling'
    seedling.mkdir()
    git(seedling, 'init', '--quiet', '--initial-branch=main')
    (seedling / 'README.md').write_text('hello\n', encoding='utf-8')
    git(seedling, 'add', '.')
    git(seedling, '-c', 'user.name=Up', '-c', 'user.email=up@example.com', 'commit', '--quiet', '-m', 'Initial')
    bare = tmp_path / 'github' / 'repo.git'
    git(tmp_path, 'clone', '--quiet', '--bare', str(seedling), str(bare))
    return bare


@pytest.fixture
def checkout(tmp_path: Path, upstream: Path) -> Path:
    """A clone of `upstream` whose `origin` names GitHub, with an unpushed commit and uncommitted files."""
    directory = tmp_path / 'checkout'
    git(tmp_path, 'clone', '--quiet', str(upstream), str(directory))
    git(directory, 'remote', 'set-url', 'origin', 'git@github.com:owner/repo.git')
    git(directory, 'config', 'user.name', 'Ada')
    git(directory, 'config', 'user.email', 'ada@example.com')
    git(directory, 'checkout', '--quiet', '-b', 'feature')
    (directory / 'committed.txt').write_text('committed\n', encoding='utf-8')
    git(directory, 'add', 'committed.txt')
    git(directory, 'commit', '--quiet', '-m', 'Unpushed')
    (directory / 'README.md').write_text('hello, edited\n', encoding='utf-8')
    (directory / 'untracked.txt').write_text('new\n', encoding='utf-8')
    return directory


@pytest.fixture
def provider(tmp_path: Path, upstream: Path, monkeypatch: pytest.MonkeyPatch) -> FakeProvider:
    sandboxes = tmp_path / 'sandboxes'
    sandboxes.mkdir()
    fake = FakeProvider(sandboxes, upstream)
    monkeypatch.setattr(sandbox, 'PROVIDERS', {fake.name: fake})

    def gh_token(hostname: str) -> str:
        return 'gho_secret'

    def gh_user(hostname: str) -> str:
        return 'Ada'

    monkeypatch.setattr(sandbox, 'gh_token', gh_token)
    monkeypatch.setattr(sandbox, 'gh_user', gh_user)
    return fake


def load(directory: Path, conversation: Transcript | None = None, *, terminal: bool = False) -> LoadedPlugin[None]:
    host = PluginHost[None](
        name='sandbox',
        console=Console(file=io.StringIO(), force_terminal=terminal),
        settings={},
        conversation=conversation or Transcript(),
        workspace=lambda: directory,
    )
    return load_plugin(SandboxPlugin, host)


def plugin_of(loaded: LoadedPlugin[None]) -> SandboxPlugin:
    assert isinstance(loaded.plugin, SandboxPlugin)
    return loaded.plugin


def worked_in(ref: WorkspaceRef) -> list[ModelRequest | ModelResponse]:
    """A history whose one turn ran in the workspace `ref` names."""
    return [
        ModelRequest(parts=[UserPromptPart('go')]),
        ModelResponse(parts=[TextPart('done')], workspace_ref=ref),
    ]


def on_screen(
    loaded: LoadedPlugin[None], checkout: Path, *, model: TestModel | FunctionModel | None = None
) -> Session[None, str]:
    """A session over `loaded`'s capabilities, put on screen as the conversation the plugin's commands act on."""
    agent = Agent(model or TestModel(call_tools=[], custom_output_text='done'), deps_type=type(None))
    session = Session(agent, deps=None, plugins=[*loaded.capabilities], workspace=checkout)
    loaded.host.conversation = session
    return session


def status(directory: Path) -> str:
    return git(directory, 'status', '--porcelain', '--branch')


class TestSeed:
    @pytest.mark.parametrize(
        ('remote', 'expected'),
        [
            ('https://github.com/owner/repo.git', CLONE_URL),
            ('https://github.com/owner/repo', CLONE_URL),
            ('git@github.com:owner/repo.git', CLONE_URL),
            ('ssh://git@github.com/owner/repo.git/', CLONE_URL),
            ('https://gitlab.com/owner/repo.git', None),
            ('https://github.com/owner', None),
        ],
    )
    def test_github_clone_url(self, remote: str, expected: str | None) -> None:
        assert github_clone_url(remote) == expected

    async def test_outside_git(self, tmp_path: Path) -> None:
        assert await to_thread.run_sync(snapshot, tmp_path) == Snapshot()

    async def test_github_checkout(self, checkout: Path, provider: FakeProvider) -> None:
        before = status(checkout)
        taken = await to_thread.run_sync(snapshot, checkout)
        assert taken.clone_url == CLONE_URL
        assert (taken.branch, taken.author_name, taken.author_email) == ('feature', 'Ada', 'ada@example.com')
        assert taken.changed
        ref, backend = await provider.create(github=None)
        await seed(backend, taken)

        copy = provider.folder(ref)
        assert git(copy, 'rev-parse', 'HEAD') == git(checkout, 'rev-parse', 'HEAD')
        assert git(copy, 'status', '--porcelain') == ' M README.md\n?? untracked.txt'
        assert (copy / 'README.md').read_text(encoding='utf-8') == 'hello, edited\n'
        assert git(copy, 'remote', 'get-url', 'origin') == CLONE_URL
        assert git(copy, 'config', '--file', f'{copy}.gitconfig', 'user.name') == 'Ada'
        assert git(copy, 'for-each-ref', 'refs/clai-seed') == ''
        # Reading the checkout changed nothing there.
        assert status(checkout) == before
        assert git(checkout, 'for-each-ref', 'refs/clai-seed') == ''

    async def test_pushed_and_clean(self, checkout: Path, provider: FakeProvider) -> None:
        git(checkout, 'checkout', '--quiet', '.')
        (checkout / 'untracked.txt').unlink()
        git(checkout, 'checkout', '--quiet', 'main')
        taken = await to_thread.run_sync(snapshot, checkout)
        assert not taken.changed
        ref, backend = await provider.create(github=None)
        await seed(backend, taken)
        assert status(provider.folder(ref)) == '## main...origin/main'

    async def test_not_on_github(self, checkout: Path, provider: FakeProvider) -> None:
        git(checkout, 'remote', 'remove', 'origin')
        git(checkout, 'checkout', '--quiet', '--detach')
        taken = await to_thread.run_sync(snapshot, checkout)
        assert (taken.clone_url, taken.branch) == (None, None)
        ref, backend = await provider.create(github=None)
        await seed(backend, taken)
        copy = provider.folder(ref)
        assert git(copy, 'rev-parse', 'HEAD') == git(checkout, 'rev-parse', 'HEAD')
        assert git(copy, 'status', '--porcelain') == ' M README.md\n?? untracked.txt'
        assert git(copy, 'remote') == ''

    async def test_branch_deleted_on_github(self, checkout: Path, provider: FakeProvider) -> None:
        # `origin/gone` still names the unpushed commit here, as after its branch was deleted on GitHub.
        git(checkout, 'update-ref', 'refs/remotes/origin/gone', 'HEAD')
        taken = await to_thread.run_sync(snapshot, checkout)
        ref, backend = await provider.create(github=None)
        await seed(backend, taken)
        assert git(provider.folder(ref), 'rev-parse', 'HEAD') == git(checkout, 'rev-parse', 'HEAD')

    async def test_two_copies_at_once(self, checkout: Path, provider: FakeProvider) -> None:
        taken = await to_thread.run_sync(snapshot, checkout)
        (first, first_backend), (second, second_backend) = [await provider.create(github=None) for _ in range(2)]
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(seed, first_backend, taken)
            tasks.start_soon(seed, second_backend, taken)
        for ref in (first, second):
            assert git(provider.folder(ref), 'rev-parse', 'HEAD') == git(checkout, 'rev-parse', 'HEAD')
        assert git(checkout, 'for-each-ref', 'refs/clai-seed') == ''

    async def test_no_commits_yet(self, tmp_path: Path, provider: FakeProvider) -> None:
        directory = tmp_path / 'fresh'
        directory.mkdir()
        git(directory, 'init', '--quiet', '--initial-branch=start')
        assert await to_thread.run_sync(snapshot, directory) == Snapshot(branch='start')
        (directory / 'notes.txt').write_text('draft\n', encoding='utf-8')
        taken = await to_thread.run_sync(snapshot, directory)
        assert (taken.head, taken.branch) == (None, 'start')
        ref, backend = await provider.create(github=None)
        await seed(backend, taken)
        copy = provider.folder(ref)
        assert status(copy) == '## No commits yet on start\n?? notes.txt'
        assert (copy / 'notes.txt').read_text(encoding='utf-8') == 'draft\n'

    async def test_failure_names_git_output(self, checkout: Path, provider: FakeProvider) -> None:
        git(checkout, 'remote', 'set-url', 'origin', 'https://github.com/owner/missing.git')
        ref, backend = await provider.create(github=None)
        del ref
        with pytest.raises(ValueError, match=r'Could not copy the checkout into the sandbox: .*missing\.git'):
            await seed(backend, await to_thread.run_sync(snapshot, checkout))


def test_github_credential() -> None:
    github = GitHubCredential(login='Ada', token='gho_secret')
    assert base64.b64decode(github.basic.removeprefix('Basic ')) == b'x-access-token:gho_secret'
    assert github.bearer == 'token gho_secret'
    assert [github.authorization(host) for host in GITHUB_HOSTS] == [github.basic, github.bearer, github.bearer]
    assert 'gho_secret' not in repr(github)


class TestCommand:
    async def test_start_in_a_new_conversation(self, checkout: Path, provider: FakeProvider) -> None:
        loaded = load(checkout)
        plugin = plugin_of(loaded)
        assert await plugin.command([]) == (
            'This conversation runs on this machine. /sandbox fake moves a new conversation into a cloud sandbox.'
        )
        message = await plugin.command(['fake'])
        head = git(checkout, 'rev-parse', '--short=7', 'HEAD')
        assert message == (
            f'This conversation now runs in Fake sandbox sbx-1. /home/user/workspace holds feature at {head}, '
            'with your uncommitted changes.\n'
            'git and gh there act as @Ada, on every repository that login reaches. '
            "Fake's proxy, outside the sandbox, holds the token until the sandbox is deleted.\n"
            'Your files here stay as they are: ask the agent to push, then pull here. ! commands still run here.'
        )
        assert provider.created == [GitHubCredential(login='Ada', token='gho_secret')]
        # The history records the sandbox before any prompt, so every copy of the conversation finds it.
        request, response = loaded.host.conversation.messages
        assert isinstance(request, ModelRequest)
        [part] = request.parts
        # CLAI's note, not a typed prompt, so `/rewind` never cuts the history back past the sandbox.
        assert imported_kind(part) == 'sandbox'
        # The model reads this reply as its own, so it says where it works, not what the user should do.
        assert isinstance(response, ModelResponse)
        assert response.parts == [TextPart('Working in Fake sandbox sbx-1, at /home/user/workspace.')]
        assert response.workspace_ref == WorkspaceRef(provider='fake', id='sbx-1')
        assert [segment() for segment in loaded.status_segments] == ['fake sandbox']
        assert await plugin.command([]) == (
            'This conversation runs in Fake sandbox sbx-1, in /home/user/workspace. /sandbox delete deletes it.'
        )
        assert await plugin.command(['fake']) == 'This conversation already runs in Fake sandbox sbx-1.'

    async def test_refused_once_the_conversation_has_run(self, checkout: Path, provider: FakeProvider) -> None:
        history = worked_in(WorkspaceRef(provider='local', id=str(checkout)))
        plugin = plugin_of(load(checkout, Transcript(messages=history)))
        with pytest.raises(ValueError, match=r'already run on this machine.*Run /new, then /sandbox fake'):
            await plugin.command(['fake'])
        assert provider.created == []

    async def test_refused_in_another_providers_sandbox(
        self, checkout: Path, provider: FakeProvider, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        other = FakeProvider(tmp_path, provider.upstream, name='other', title='Other')
        monkeypatch.setattr(sandbox, 'PROVIDERS', {provider.name: provider, other.name: other})
        plugin = plugin_of(load(checkout))
        await plugin.command(['fake'])
        with pytest.raises(ValueError, match=r'^This conversation runs in Fake sandbox sbx-1\. Run /new first\.$'):
            await plugin.command(['other'])

    async def test_missing_requirement(self, checkout: Path, provider: FakeProvider) -> None:
        provider.requirement = 'Add your Fake key in /keys.'
        with pytest.raises(ValueError, match=r'^Add your Fake key in /keys\.$'):
            await plugin_of(load(checkout)).command(['fake'])

    @pytest.mark.parametrize('args', [['nope'], ['fake', 'extra']])
    async def test_usage(self, checkout: Path, provider: FakeProvider, args: list[str]) -> None:
        with pytest.raises(ValueError, match=r'^Usage: /sandbox \[fake \| delete\]$'):
            await plugin_of(load(checkout)).command(args)

    def test_complete(self, checkout: Path, provider: FakeProvider) -> None:
        plugin = plugin_of(load(checkout))
        assert list(plugin.complete([])) == list(plugin.complete(['f'])) == ['fake', 'delete']
        assert list(plugin.complete(['fake', ''])) == []

    def test_complete_offers_the_shipped_providers(self, checkout: Path) -> None:
        assert list(plugin_of(load(checkout)).complete([])) == ['e2b', 'boxd', 'delete']

    @pytest.mark.parametrize('gap', ['no gh', 'no login', 'token refused'])
    async def test_without_a_github_login(
        self, checkout: Path, provider: FakeProvider, monkeypatch: pytest.MonkeyPatch, gap: str
    ) -> None:
        def no_gh(hostname: str) -> str:
            raise UserError('gh is not installed')

        def nothing(hostname: str) -> None:
            return None

        if gap == 'no gh':
            monkeypatch.setattr(sandbox, 'gh_token', no_gh)
        elif gap == 'no login':
            monkeypatch.setattr(sandbox, 'gh_token', nothing)
        else:
            monkeypatch.setattr(sandbox, 'gh_user', nothing)
        git(checkout, 'remote', 'remove', 'origin')
        message = await plugin_of(load(checkout)).command(['fake'])
        assert 'gh has no GitHub login here, so the sandbox cannot push or reach private repositories.' in message
        assert provider.created == [None]

    async def test_origin_not_on_github(self, checkout: Path, provider: FakeProvider) -> None:
        git(checkout, 'remote', 'set-url', 'origin', 'https://gitlab.com/owner/repo.git')
        message = await plugin_of(load(checkout)).command(['fake'])
        assert 'act as @Ada' in message
        assert 'origin is not on GitHub, so pushing it needs credentials the sandbox does not have.' in message

    async def test_outside_git(self, tmp_path: Path, provider: FakeProvider) -> None:
        folder = tmp_path / 'plain'
        folder.mkdir()
        message = await plugin_of(load(folder)).command(['fake'])
        assert 'holds an empty folder: this folder is not in a Git repository with files.' in message

    async def test_detached_and_clean(self, checkout: Path, provider: FakeProvider) -> None:
        git(checkout, 'stash', '--quiet', '--include-untracked')
        git(checkout, 'checkout', '--quiet', '--detach')
        message = await plugin_of(load(checkout)).command(['fake'])
        assert '/home/user/workspace holds this checkout.\n' in message

    async def test_failed_copy_deletes_the_sandbox(
        self, checkout: Path, provider: FakeProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        git(checkout, 'remote', 'set-url', 'origin', 'https://github.com/owner/missing.git')
        plugin = plugin_of(load(checkout))
        named: list[WorkspaceRef | None] = []

        async def seeding(backend: SandboxBackend, checkout: Snapshot) -> None:
            # Copying can take minutes, so the history names the sandbox first: a CLAI that dies now leaves it findable.
            named.append(history_sandbox(plugin.host.conversation.messages))
            await seed(backend, checkout)

        monkeypatch.setattr(sandbox, 'seed', seeding)
        with pytest.raises(ValueError, match='Could not copy the checkout'):
            await plugin.command(['fake'])
        assert named == [WorkspaceRef(provider='fake', id='sbx-1')]
        assert provider.deleted == [WorkspaceRef(provider='fake', id='sbx-1')]
        assert plugin.current() is None
        assert plugin.host.conversation.messages == []

    async def test_a_sandbox_whose_delete_failed_stays_named(self, checkout: Path, provider: FakeProvider) -> None:
        git(checkout, 'remote', 'set-url', 'origin', 'https://github.com/owner/missing.git')
        provider.delete_error = RuntimeError('network down')
        plugin = plugin_of(load(checkout))
        with pytest.raises(RuntimeError, match='network down'):
            await plugin.command(['fake'])
        # Still in the history, so `/sandbox delete` can try again.
        assert history_sandbox(plugin.host.conversation.messages) == WorkspaceRef(provider='fake', id='sbx-1')
        assert provider.closed

    async def test_a_slow_github_is_not_a_missing_login(
        self, checkout: Path, provider: FakeProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def slow(hostname: str) -> str:
            raise UserError('GitHub did not answer gh within 10 seconds.')

        monkeypatch.setattr(sandbox, 'gh_user', slow)
        with pytest.raises(UserError, match='GitHub did not answer'):
            await plugin_of(load(checkout)).command(['fake'])
        assert provider.created == []


class TestLifecycle:
    async def test_delete_keeps_the_history(self, checkout: Path, provider: FakeProvider) -> None:
        loaded = load(checkout)
        plugin = plugin_of(loaded)
        assert await plugin.command(['delete']) == 'This conversation has no sandbox.'
        await plugin.command(['fake'])
        started = list(loaded.host.conversation.messages)
        # Even with nothing run yet: a copy of the conversation in another session may be using the sandbox.
        assert await plugin.command(['delete']) == 'Deleted Fake sandbox sbx-1. Run /new to start another conversation.'
        assert provider.deleted == [WorkspaceRef(provider='fake', id='sbx-1')]
        assert loaded.host.conversation.messages == started

    async def test_renaming_the_conversation_keeps_its_unused_sandbox(
        self, checkout: Path, provider: FakeProvider
    ) -> None:
        loaded = load(checkout, Transcript(conversation_id='first'))
        await plugin_of(loaded).command(['fake'])
        await loaded.dispatch(ConversationChanged(conversation_id='first', title='renamed'))
        assert provider.deleted == []
        assert plugin_of(loaded).current() is not None

    async def test_a_failed_delete_is_retried(self, checkout: Path, provider: FakeProvider) -> None:
        loaded = load(checkout, Transcript(conversation_id='first'))
        plugin = plugin_of(loaded)
        await plugin.command(['fake'])
        provider.delete_error = OSError('boxd is down')
        with pytest.raises(OSError, match='boxd is down'):
            await plugin.command(['delete'])
        # Still known, so `/sandbox delete` can try again.
        provider.delete_error = None
        assert await plugin.command(['delete']) == 'Deleted Fake sandbox sbx-1. Run /new to start another conversation.'
        assert provider.deleted == provider.closed == [WorkspaceRef(provider='fake', id='sbx-1')]

    async def test_cancelling_while_creating_deletes_the_sandbox(self, checkout: Path, provider: FakeProvider) -> None:
        plugin = plugin_of(load(checkout))
        provider.creating = anyio.Event()
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(plugin.command, ['fake'])
            while not provider.created:
                await anyio.sleep(0)
            tasks.cancel_scope.cancel()
            # The provider finishes creating after `/sandbox` was cancelled.
            provider.creating.set()
        assert provider.deleted == provider.closed == [WorkspaceRef(provider='fake', id='sbx-1')]
        assert plugin.current() is None

    async def test_a_used_sandbox_stays_open_until_the_session_ends(
        self, checkout: Path, provider: FakeProvider
    ) -> None:
        loaded = load(checkout)
        session = on_screen(loaded, checkout)
        await plugin_of(loaded).command(['fake'])
        await session.prompt('go')
        # A background `/fork` may still run there after `/new`.
        loaded.host.conversation = Transcript(conversation_id='second')
        await loaded.dispatch(ConversationChanged(conversation_id='second', title=None))
        assert (provider.deleted, provider.closed) == ([], [])
        await loaded.dispatch(SessionEnd(reason='exit'))
        assert (provider.deleted, provider.closed) == ([], [WorkspaceRef(provider='fake', id='sbx-1')])

    async def test_a_used_sandbox_stays(self, checkout: Path, provider: FakeProvider) -> None:
        loaded = load(checkout)
        plugin = plugin_of(loaded)
        session = on_screen(loaded, checkout)
        await plugin.command(['fake'])
        await session.prompt('go')
        await loaded.dispatch(ConversationChanged(conversation_id='other', title=None))
        await loaded.dispatch(SessionEnd(reason='exit'))
        assert provider.deleted == []

    async def test_delete_fails_the_next_run_closed(self, checkout: Path, provider: FakeProvider) -> None:
        loaded = load(checkout)
        plugin = plugin_of(loaded)
        session = on_screen(loaded, checkout)
        await plugin.command(['fake'])
        await session.prompt('go')
        assert await plugin.command(['delete']) == (
            'Deleted Fake sandbox sbx-1. Run /new to start another conversation.'
        )
        assert provider.deleted == [WorkspaceRef(provider='fake', id='sbx-1')]
        assert await plugin.command([]) == ('This conversation ran in Fake sandbox sbx-1, now deleted. Run /new.')
        current = plugin.current()
        assert current is not None
        with pytest.raises(
            UserError, match=r"^This conversation's Fake sandbox was deleted\. Run /new to start another\.$"
        ):
            current.open()

    @pytest.mark.parametrize('leave', ['change', 'end'])
    async def test_leaving_keeps_an_unused_sandbox(self, checkout: Path, provider: FakeProvider, leave: str) -> None:
        conversation = Transcript(conversation_id='first')
        loaded = load(checkout, conversation)
        await plugin_of(loaded).command(['fake'])
        if leave == 'change':
            # The shell swaps the conversation before it tells plugins.
            loaded.host.conversation = Transcript(conversation_id='second')
            await loaded.dispatch(ConversationChanged(conversation_id='second', title=None))
        else:
            await loaded.dispatch(SessionEnd(reason='exit'))
        # The saved history names the sandbox, so resuming the conversation must find it.
        assert provider.deleted == []
        restarted = plugin_of(load(checkout, Transcript(messages=conversation.messages)))
        assert restarted.current() is not None
        assert (
            await restarted.command(['delete']) == 'Deleted Fake sandbox sbx-1. Run /new to start another conversation.'
        )
        assert provider.deleted == [WorkspaceRef(provider='fake', id='sbx-1')]

    async def test_rewind_never_drops_the_sandbox(self, checkout: Path, provider: FakeProvider) -> None:
        loaded = load(checkout)
        session = on_screen(loaded, checkout)
        await plugin_of(loaded).command(['fake'])
        await session.prompt('go')
        assert [point.message_index for point in rewind_points(session.messages)] == [2]

    async def test_history_reopens_its_sandbox(self, checkout: Path, provider: FakeProvider) -> None:
        ref = WorkspaceRef(provider='fake', id='sbx-9')
        provider.folder(ref).mkdir()
        loaded = load(checkout, Transcript(messages=worked_in(ref)))
        plugin = plugin_of(loaded)
        assert [segment() for segment in loaded.status_segments] == ['fake sandbox']
        current = plugin.current()
        assert current is not None
        backend = current.open()
        assert current.open() is backend
        assert backend.ref == ref
        assert provider.reopened == [ref]
        await loaded.dispatch(SessionEnd(reason='exit'))
        assert provider.deleted == []

    async def test_a_turn_here_keeps_running_here(self, checkout: Path, provider: FakeProvider) -> None:
        loaded = load(checkout)
        await on_screen(loaded, checkout).prompt('go')
        assert plugin_of(loaded).current() is None

    async def test_a_failed_first_turn_keeps_its_sandbox(self, checkout: Path, provider: FakeProvider) -> None:
        async def fail(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
            raise RuntimeError('the model is down')
            yield  # pragma: no cover - makes this a stream function

        loaded = load(checkout)
        session = on_screen(loaded, checkout, model=FunctionModel(stream_function=fail))
        await plugin_of(loaded).command(['fake'])
        with pytest.raises(RuntimeError, match='the model is down'):
            await session.prompt('go')
        # The turn opened the sandbox, so leaving keeps it, and the saved history still names it after a restart.
        await loaded.dispatch(ConversationChanged(conversation_id='second', title=None))
        assert provider.deleted == []
        restarted = plugin_of(load(checkout, Transcript(messages=session.messages)))
        assert (current := restarted.current()) is not None and current.ref == WorkspaceRef(provider='fake', id='sbx-1')

    async def test_history_in_an_unknown_workspace_runs_here(self, checkout: Path, provider: FakeProvider) -> None:
        loaded = load(checkout, Transcript(messages=worked_in(WorkspaceRef(provider='retired', id='sb-1'))))
        assert plugin_of(loaded).current() is None
        assert [segment() for segment in loaded.status_segments] == ['']

    async def test_reopen_needs_the_requirement(self, checkout: Path, provider: FakeProvider) -> None:
        provider.requirement = 'Add your Fake key in /keys.'
        plugin = plugin_of(load(checkout, Transcript(messages=worked_in(WorkspaceRef(provider='fake', id='x')))))
        current = plugin.current()
        assert current is not None
        with pytest.raises(
            UserError, match=r'^This conversation runs in a Fake sandbox\. Add your Fake key in /keys\.$'
        ):
            current.open()


async def test_runs_use_the_sandbox(checkout: Path, provider: FakeProvider) -> None:
    working_dirs: list[str] = []

    class Probe(AbstractCapability[None]):
        async def before_run(self, ctx: RunContext[None]) -> None:
            working_dirs.append(await ctx.workspace.working_dir())

    loaded = load(checkout)
    plugin = plugin_of(loaded)
    [capability] = loaded.capabilities
    agent = Agent(TestModel(call_tools=[], custom_output_text='done'), deps_type=type(None))
    session = Session(agent, deps=None, plugins=[capability, Probe()], workspace=checkout)
    loaded.host.conversation = session
    await session.prompt('here')
    # A conversation with history refuses `/sandbox`, so start the sandboxed one fresh.
    session.replace_messages([])
    await plugin.command(['fake'])
    result = await session.prompt('there')
    assert working_dirs == [str(checkout.resolve()), str(provider.folder(WorkspaceRef(provider='fake', id='sbx-1')))]
    assert result.response.workspace_ref == WorkspaceRef(provider='fake', id='sbx-1')
    [request] = [message for message in result.new_messages() if isinstance(message, ModelRequest)]
    assert 'Your shell and file tools run in a Fake sandbox.' in (request.instructions or '')

    # A background `/fork` keeps its sandbox after `/new` puts a local conversation on screen.
    fork = Session(
        agent, deps=None, plugins=[capability, Probe()], workspace=checkout, message_history=session.messages
    )
    assert fork.conversation_id != session.conversation_id
    loaded.host.conversation = Transcript(conversation_id='shown')
    result = await fork.prompt('in the background')
    assert working_dirs[-1] == str(provider.folder(WorkspaceRef(provider='fake', id='sbx-1')))
    assert result.response.workspace_ref == WorkspaceRef(provider='fake', id='sbx-1')


async def test_a_fork_before_the_first_answer_runs_in_the_sandbox(checkout: Path, provider: FakeProvider) -> None:
    loaded = load(checkout)
    session = on_screen(loaded, checkout)
    await plugin_of(loaded).command(['fake'])
    # `/fork` copies the history, which already names the sandbox, then `/new` puts a local conversation on screen.
    agent = Agent(TestModel(call_tools=[], custom_output_text='done'), deps_type=type(None))
    fork = Session(
        agent, deps=None, plugins=[*loaded.capabilities], workspace=checkout, message_history=session.messages
    )
    loaded.host.conversation = Transcript(conversation_id='shown')
    result = await fork.prompt('in the background')
    assert result.response.workspace_ref == WorkspaceRef(provider='fake', id='sbx-1')
    # The fork's run opened the sandbox, so leaving the conversation `/sandbox` made it for keeps it.
    await loaded.dispatch(ConversationChanged(conversation_id='shown', title=None))
    assert provider.deleted == []


def asks_for_a_key(name: str = 'ANTHROPIC_API_KEY') -> tuple[FunctionModel, list[str], list[list[str]]]:
    """A model that calls `request_env` for `name` once, then answers with what it got; and the tools each request offered."""
    returns: list[str] = []
    offered: list[list[str]] = []

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        offered.append([tool.name for tool in info.function_tools])
        last = messages[-1].parts[-1]
        if 'request_env' not in offered[-1]:
            yield 'no tool'
        elif isinstance(last, ToolReturnPart):
            returns.append(str(last.content))
            yield 'done'
        else:
            args = json.dumps({'name': name, 'reason': 'run the eval script'})
            yield {0: DeltaToolCall(name='request_env', json_args=args)}

    return FunctionModel(stream_function=respond), returns, offered


def scripted(dialogs: ScriptedDialogs) -> Callable[[object], ScriptedDialogs]:
    """Stands in for `TerminalDialogs`, whatever runners it is given."""
    return lambda runners: dialogs


class TestRequestEnv:
    """`request_env` asks for a key with the `/keys` picker and sets it in the sandbox, mid-turn."""

    async def test_a_typed_key_is_saved_and_set(
        self, checkout: Path, provider: FakeProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dialogs = ScriptedDialogs([Type(text='sk-typed')])
        monkeypatch.setattr(sandbox, 'TerminalDialogs', scripted(dialogs))
        loaded = load(checkout, terminal=True)
        model, returns, _ = asks_for_a_key()
        session = on_screen(loaded, checkout, model=model)
        await plugin_of(loaded).command(['fake'])
        result = await session.prompt('run the evals')
        ref = WorkspaceRef(provider='fake', id='sbx-1')
        assert provider.envs == [(ref, 'ANTHROPIC_API_KEY', 'sk-typed')]
        assert returns == [
            'ANTHROPIC_API_KEY is set for every command started from now on. Restart a server started before it.'
        ]
        [asked] = dialogs.asked
        assert asked.title == (
            'The agent asks for ANTHROPIC_API_KEY to run the eval script. Anything run in the sandbox can read it.'
        )
        # Saved in `/keys`, and never in the conversation.
        assert load_keys()['ANTHROPIC_API_KEY'].get_secret_value() == 'sk-typed'
        assert 'sk-typed' not in str(result.all_messages())

    async def test_a_saved_key_under_another_name(
        self, checkout: Path, provider: FakeProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        save_key(name='WORK_ANTHROPIC', value='sk-saved')
        monkeypatch.setattr(sandbox, 'TerminalDialogs', scripted(ScriptedDialogs([Pick(value='WORK_ANTHROPIC')])))
        loaded = load(checkout, terminal=True)
        model, _, _ = asks_for_a_key()
        session = on_screen(loaded, checkout, model=model)
        await plugin_of(loaded).command(['fake'])
        await session.prompt('run the evals')
        assert provider.envs == [(WorkspaceRef(provider='fake', id='sbx-1'), 'ANTHROPIC_API_KEY', 'sk-saved')]

    @pytest.mark.parametrize(
        ('name', 'env_error', 'expected'),
        [
            ('ANTHROPIC_API_KEY', None, 'The user did not give ANTHROPIC_API_KEY.'),
            ('api key', None, "'api key' is not an environment variable name: use capitals, digits, and underscores."),
            # `$` alone would let a trailing newline through, splitting E2B's profile line.
            (
                'API_KEY\n',
                None,
                "'API_KEY\\n' is not an environment variable name: use capitals, digits, and underscores.",
            ),
            ('GH_TOKEN', None, 'CLAI sets GH_TOKEN in the sandbox itself, so it cannot be replaced.'),
            (
                'ANTHROPIC_API_KEY',
                RuntimeError('this sandbox takes no variables'),
                'ANTHROPIC_API_KEY could not be set: this sandbox takes no variables.',
            ),
        ],
        ids=['declined', 'bad name', 'trailing newline', 'GitHub stand-in', 'provider refused'],
    )
    async def test_nothing_is_set(
        self,
        checkout: Path,
        provider: FakeProvider,
        monkeypatch: pytest.MonkeyPatch,
        name: str,
        env_error: RuntimeError | None,
        expected: str,
    ) -> None:
        answer = Back() if env_error is None else Type(text='sk-typed')
        monkeypatch.setattr(sandbox, 'TerminalDialogs', scripted(ScriptedDialogs([answer])))
        provider.env_error = env_error
        loaded = load(checkout, terminal=True)
        model, returns, _ = asks_for_a_key(name)
        session = on_screen(loaded, checkout, model=model)
        await plugin_of(loaded).command(['fake'])
        await session.prompt('run the evals')
        assert returns == [expected]
        assert provider.envs == []

    async def test_a_key_deleted_while_picking(
        self, checkout: Path, provider: FakeProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        save_key(name='WORK_ANTHROPIC', value='sk-saved')
        monkeypatch.setattr(sandbox, 'TerminalDialogs', scripted(ScriptedDialogs([Pick(value='WORK_ANTHROPIC')])))
        monkeypatch.setattr(sandbox, 'load_keys', dict)  # `/keys` emptied after the picker read it.
        loaded = load(checkout, terminal=True)
        model, returns, _ = asks_for_a_key()
        session = on_screen(loaded, checkout, model=model)
        await plugin_of(loaded).command(['fake'])
        await session.prompt('run the evals')
        assert returns == ['The user did not give ANTHROPIC_API_KEY.']
        assert provider.envs == []

    @pytest.mark.parametrize('terminal', [True, False], ids=['on this machine', 'no terminal'])
    async def test_not_offered(self, checkout: Path, provider: FakeProvider, terminal: bool) -> None:
        loaded = load(checkout, terminal=terminal)
        model, _, offered = asks_for_a_key()
        session = on_screen(loaded, checkout, model=model)
        if not terminal:
            await plugin_of(loaded).command(['fake'])
        await session.prompt('run the evals')
        assert offered == [[]]


@pytest.fixture
def fake_e2b(monkeypatch: pytest.MonkeyPatch) -> Iterator['FakeE2B']:
    """The harness's fake E2B SDK as `e2b`; each client it makes is named by its API key."""
    from tests.harness.e2b_sandbox.fake_e2b import FakeE2B

    control = FakeE2B()
    module = control.module
    module.__spec__ = importlib.machinery.ModuleSpec('e2b', None)
    module.E2B = lambda api_key: control.client(api_key)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, 'e2b', module)
    monkeypatch.setattr('pydantic_ai_harness.e2b_sandbox._backend.e2b', module)
    monkeypatch.delenv(e2b_provider.KEY_NAME, raising=False)
    try:
        yield control
    finally:
        control.close()


@pytest.mark.skipif(find_spec('e2b') is None, reason='e2b not installed')
class TestE2B:
    """E2B, through the SDK fake the harness's own E2B tests use: a live run needs a billed account."""

    def test_missing(self, fake_e2b: 'FakeE2B', monkeypatch: pytest.MonkeyPatch) -> None:
        e2b = E2BProvider()
        assert e2b.missing() == (
            'Add your E2B API key in /keys as E2B_API_KEY. Get one at https://e2b.dev/dashboard?tab=keys.'
        )
        monkeypatch.setenv('E2B_API_KEY', 'from-env')
        assert e2b.missing() is None

        def not_installed(name: str) -> None:
            return None

        monkeypatch.setattr(e2b_provider, 'find_spec', not_installed)
        assert e2b.missing() == (
            "E2B support is not installed. Install CLAI with it: uv tool install 'pydantic-clai2[e2b]'."
        )

    async def test_create_reopen_delete(self, fake_e2b: 'FakeE2B', monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('E2B_API_KEY', 'from-env')
        save_key(name='E2B_API_KEY', value='from-keys')
        e2b = E2BProvider()
        github = GitHubCredential(login='Ada', token='gho_x')
        ref, backend = await e2b.create(github=github)
        assert ref == backend.ref == WorkspaceRef(provider='e2b', id='sbx-1')
        [call] = fake_e2b.create_calls
        assert call.envs == {'GIT_TERMINAL_PROMPT': '0', 'GH_TOKEN': 'placeholder'}
        # The token is only in this sandbox's rules, so it ends with the sandbox, and its URLs stay private.
        assert call.network == {
            'rules': {
                'github.com': [{'transform': {'headers': {'Authorization': github.basic}}}],
                'api.github.com': [{'transform': {'headers': {'Authorization': github.bearer}}}],
                'uploads.github.com': [{'transform': {'headers': {'Authorization': github.bearer}}}],
            },
            'allow_public_traffic': False,
        }
        await e2b.create(github=None)
        # No GitHub rules and no stand-in, but outside traffic is still refused.
        assert fake_e2b.create_calls[-1].network == {'allow_public_traffic': False}
        assert fake_e2b.create_calls[-1].envs == SANDBOX_ENV
        reopened = e2b.reopen(ref)
        assert reopened.ref == ref
        # A login shell runs the file for every later command; a value the command sets itself wins.
        await e2b.set_env(reopened, 'API_KEY', "it's $x")
        made = fake_e2b.sandboxes[0]
        assert made.files.files['/etc/profile.d/clai-env-API_KEY.sh'] == (
            b"[ -n \"${API_KEY+set}\" ] || export API_KEY='it'\"'\"'s $x'\n"
        )
        assert made.files.users[-1] == 'root'
        await e2b.delete(ref)
        assert fake_e2b.sandboxes[0].killed
        # `/keys` wins over the environment, for every call.
        assert set(fake_e2b.clients) == {'from-keys'}

    async def test_a_sandbox_whose_setup_failed_is_deleted(
        self, fake_e2b: 'FakeE2B', monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tests.harness.e2b_sandbox.fake_e2b import FakeFilesystem

        async def refuse(*args: object, **kwargs: object) -> bool:
            raise OSError('envd is down')

        monkeypatch.setenv('E2B_API_KEY', 'from-env')
        monkeypatch.setattr(FakeFilesystem, 'make_dir', refuse)
        with pytest.raises(OSError, match='envd is down'):
            await E2BProvider().create(github=GitHubCredential(login='Ada', token='gho_x'))
        # The sandbox was made before its working folder failed, and its rules held the token.
        [made] = fake_e2b.sandboxes
        assert made.killed


@dataclass
class FakeBoxdAccount:
    """boxd's secrets and machine specs, which the harness's boxd fake leaves out."""

    secrets: dict[str, tuple[str, str, list[str]]] = field(default_factory=dict[str, tuple[str, str, list[str]]])
    specs: dict[str, dict[str, object]] = field(default_factory=dict[str, dict[str, object]])
    clients: list['FakeClient'] = field(default_factory=list['FakeClient'])
    delete_error: Exception | None = None
    fake: 'FakeBoxd' = field(default=None)  # pyright: ignore[reportAssignmentType]

    async def set(self, name: str, value: str, *, scope: str, domains: list[str]) -> None:
        self.secrets[name] = (value, scope, domains)

    async def delete(self, name: str, *, scope: str) -> None:
        if self.delete_error is not None:
            raise self.delete_error
        assert self.secrets.pop(name)[1] == scope

    async def update_spec(self, machine_id: str, patch: dict[str, object]) -> None:
        import boxd

        if self.fake.find(machine_id).create_kwargs['spec'] is None:
            raise boxd.ConflictError(
                "this machine wasn't created from a spec, so it has no spec to update", code='failed_precondition'
            )
        # boxd merges a patch into the spec, maps included.
        env = {
            **cast(dict[str, str], self.specs.get(machine_id, {}).get('env', {})),
            **cast(dict[str, str], patch['env']),
        }
        self.specs[machine_id] = {**patch, 'env': env}

    async def list(self) -> 'list[Secret]':
        from boxd import Secret

        def bound(name: str) -> int:
            return sum(
                name in machine.create_kwargs['spec'].get('secrets', [])
                for machine in self.fake.machines.values()
                if machine.status != 'destroyed'
            )

        return [
            Secret(name=name, scope=scope, domains=domains, bound_machines=bound(name))
            for name, (_, scope, domains) in self.secrets.items()
        ]


@pytest.fixture
async def fake_boxd(monkeypatch: pytest.MonkeyPatch) -> 'AsyncIterator[tuple[FakeBoxd, FakeBoxdAccount]]':
    """The harness's boxd fake, whose commands run here, with `FakeBoxdAccount` on every client."""
    import boxd

    from tests.harness.boxd_sandbox.fake_boxd import FakeBoxd

    fake = FakeBoxd()
    account = FakeBoxdAccount(fake=fake)

    def client(api_key: str | None = None) -> 'FakeClient':
        made = fake.client()
        account.clients.append(made)
        made.secrets = account  # type: ignore[attr-defined]
        made.machines.update_spec = account.update_spec  # type: ignore[attr-defined]
        get = made.machines.get

        async def get_with_spec(machine_id: str) -> 'Machine':
            # The fake's records leave out the spec a machine was created with.
            return (await get(machine_id)).model_copy(update={'spec': fake.find(machine_id).create_kwargs['spec']})

        made.machines.get = get_with_spec
        return made

    monkeypatch.setattr('pydantic_ai_harness.boxd_sandbox._backend.boxd', boxd)
    monkeypatch.setattr(boxd, 'AsyncBoxd', client)
    monkeypatch.setattr('pydantic_ai_harness.boxd_sandbox._backend._POLL_INTERVAL', 0)
    yield fake, account
    await fake.reap()


@pytest.mark.skipif(find_spec('boxd') is None, reason='boxd not installed')
class TestBoxd:
    """boxd, through the SDK fake the harness's own boxd tests use; the swap itself was proven live."""

    def test_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(boxd_provider.KEY_NAME, raising=False)
        provider = boxd_provider.BoxdProvider()
        assert (
            provider.missing() == 'Add your boxd API key in /keys as BOXD_API_KEY. Make one with boxd api-keys create.'
        )
        monkeypatch.setenv('BOXD_API_KEY', 'bk-1')
        assert provider.missing() is None

        def not_installed(name: str) -> None:
            return None

        monkeypatch.setattr(boxd_provider, 'find_spec', not_installed)
        assert provider.missing() == (
            "boxd support is not installed. Install CLAI with it: uv tool install 'pydantic-clai2[boxd]'."
        )

    async def test_create_reopen_delete(
        self, fake_boxd: 'tuple[FakeBoxd, FakeBoxdAccount]', monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        fake, account = fake_boxd
        fake.host_root = tmp_path.resolve()
        secret = 'CLAI_GITHUB_' + '0' * 32
        numbers = itertools.count()
        monkeypatch.setattr(uuid, 'uuid4', lambda: uuid.UUID(int=next(numbers)))  # the secret's name is the first
        # boxd sets the secret's variable to its stand-in; the fake's commands run here, so this stands in for that.
        monkeypatch.setenv(secret, 'bxds_stand-in')
        provider = boxd_provider.BoxdProvider(working_dir=str(tmp_path / 'workspace'))
        ref, backend = await provider.create(github=GitHubCredential(login='Ada', token='gho_x'))
        assert account.secrets == {secret: ('gho_x', 'private-unattached', list(GITHUB_HOSTS))}
        [machine] = fake.machines.values()
        assert machine.isolated
        assert machine.create_kwargs['spec']['secrets'] == [secret]
        # The spec keeps the stand-in as `GH_TOKEN` for every later command, after a reopen too.
        assert account.specs == {ref.id: {'env': {'GH_TOKEN': 'bxds_stand-in'}}}
        assert backend.ref == provider.reopen(ref).ref == ref
        assert (await provider.reopen(ref).run(['printenv', 'GIT_TERMINAL_PROMPT'])).stdout == '0\n'
        # A key the agent asks for goes into the same spec.
        await provider.set_env(backend, 'API_KEY', 'sk-1')
        env = cast(dict[str, str], account.specs[ref.id]['env'])
        assert list(env) == ['GH_TOKEN', 'API_KEY'] and env['API_KEY'] == 'sk-1'
        # Deleting the machine deletes its secret, so the token ends with it.
        await provider.delete(ref)
        assert machine.status == 'destroyed'
        assert account.secrets == {}
        # Without a GitHub login: no secret, and nothing for the spec to keep.
        ref, backend = await provider.create(github=None)
        assert account.secrets == {}
        assert ref.id not in account.specs
        # boxd updates only a spec a machine was made from.
        with pytest.raises(RuntimeError, match='made without a GitHub login'):
            await provider.set_env(backend, 'API_KEY', 'sk-1')
        reopened = provider.reopen(ref)
        await provider.delete(ref)
        # The clients made for workspaces stay open until clai is done with them.
        await provider.close(backend)
        await provider.close(reopened)
        assert all(client.closed for client in account.clients[-3:])

    async def test_a_failed_secret_delete_is_retried(
        self, fake_boxd: 'tuple[FakeBoxd, FakeBoxdAccount]', monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        fake, account = fake_boxd
        fake.host_root = tmp_path.resolve()
        monkeypatch.setattr(uuid, 'uuid4', lambda: uuid.UUID(int=0))
        monkeypatch.setenv('CLAI_GITHUB_' + '0' * 32, 'bxds_stand-in')
        provider = boxd_provider.BoxdProvider(working_dir=str(tmp_path / 'workspace'))
        ref, backend = await provider.create(github=GitHubCredential(login='Ada', token='gho_x'))
        await provider.close(backend)
        account.delete_error = OSError('boxd is down')
        with pytest.raises(OSError, match='boxd is down'):
            await provider.delete(ref)
        # The machine is gone, and nothing names the secret now, so trying again finds it.
        [machine] = fake.machines.values()
        assert machine.status == 'destroyed' and account.secrets
        account.delete_error = None
        await provider.delete(ref)
        assert account.secrets == {}

    async def test_a_secret_whose_machine_boxd_deleted_goes_too(
        self, fake_boxd: 'tuple[FakeBoxd, FakeBoxdAccount]', monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        fake, account = fake_boxd
        fake.host_root = tmp_path.resolve()
        numbers = itertools.count()
        monkeypatch.setattr(uuid, 'uuid4', lambda: uuid.UUID(int=next(numbers)))
        for number in range(20):
            monkeypatch.setenv(f'CLAI_GITHUB_{uuid.UUID(int=number).hex.upper()}', 'bxds_stand-in')
        provider = boxd_provider.BoxdProvider(working_dir=str(tmp_path / 'workspace'))
        github = GitHubCredential(login='Ada', token='gho_x')
        gone, backend = await provider.create(github=github)
        await provider.close(backend)
        kept, backend = await provider.create(github=github)
        await provider.close(backend)
        account.secrets['USER_OWN'] = ('x', 'private-unattached', [])
        # boxd deleted the first machine itself, so only its secret is left.
        from pydantic_ai_harness.boxd_sandbox import BoxdSandbox

        await BoxdSandbox[None](client=fake.client()).destroy(gone)  # pyright: ignore[reportArgumentType]
        await provider.delete(gone)
        [kept_secret] = fake.find(kept.id).create_kwargs['spec']['secrets']
        assert set(account.secrets) == {kept_secret, 'USER_OWN'}
        await provider.delete(kept)
        assert set(account.secrets) == {'USER_OWN'}

    @pytest.mark.parametrize('failure', ['machine refused', 'stand-in is the token'])
    async def test_a_failed_create_leaves_nothing_behind(
        self,
        fake_boxd: 'tuple[FakeBoxd, FakeBoxdAccount]',
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        failure: str,
    ) -> None:
        import boxd

        fake, account = fake_boxd
        fake.host_root = tmp_path.resolve()
        monkeypatch.setattr(uuid, 'uuid4', lambda: uuid.UUID(int=0))
        if failure == 'machine refused':
            fake.create_errors.append(boxd.APIStatusError('quota exceeded', code='resource_exhausted', grpc_code=8))
            raised = boxd.APIStatusError
        else:
            # boxd put the token itself in the machine instead of a stand-in.
            monkeypatch.setenv('CLAI_GITHUB_' + '0' * 32, 'gho_x')
            raised = RuntimeError
        provider = boxd_provider.BoxdProvider(working_dir=str(tmp_path / 'workspace'))
        with pytest.raises(raised):
            await provider.create(github=GitHubCredential(login='Ada', token='gho_x'))
        assert account.secrets == {}
        assert all(machine.status == 'destroyed' for machine in fake.machines.values())
        assert all(client.closed for client in account.clients)
