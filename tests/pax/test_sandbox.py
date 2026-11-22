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
import os
import re
import shlex
import stat
import subprocess
import sys
import uuid
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from dataclasses import dataclass, field
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

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
from pydantic_ai.workspaces import (
    CommandResult,
    LocalWorkspaceBackend,
    SupportsFilesystem,
    WorkspaceCommand,
    WorkspaceRef,
    WorkspaceUnavailableError,
)
from pydantic_ai_harness import Shell
from pydantic_ai_pax import Session
from pydantic_ai_pax.builtin_plugins import sandbox
from pydantic_ai_pax.builtin_plugins.sandbox import SandboxPlugin, history_sandbox
from pydantic_ai_pax.builtin_plugins.sandbox.providers import (
    GITHUB_HOSTS,
    GitHubCredential,
    boxd as boxd_provider,
    daytona as daytona_provider,
    e2b as e2b_provider,
    kubernetes as kubernetes_provider,
    local as local_provider,
    modal as modal_provider,
    opensandbox as opensandbox_provider,
    render as render_provider,
    sprites as sprites_provider,
)
from pydantic_ai_pax.builtin_plugins.sandbox.providers.base import SANDBOX_ENV, give_github_login, saved_key
from pydantic_ai_pax.builtin_plugins.sandbox.providers.e2b import E2BProvider
from pydantic_ai_pax.builtin_plugins.sandbox.seed import (
    SandboxBackend,
    Snapshot,
    clone_script,
    github_clone_url,
    seed,
    snapshot,
)
from pydantic_ai_pax.config import Settings
from pydantic_ai_pax.config.api_keys import load_keys, save_key
from pydantic_ai_pax.config.project_settings import ProjectSettings
from pydantic_ai_pax.config.settings_store import SettingsStore
from pydantic_ai_pax.plugins import (
    ConversationChanged,
    LoadedPlugin,
    PluginHost,
    SessionEnd,
    Transcript,
    load_plugin,
)
from pydantic_ai_pax.runtime.imported_history import imported_kind
from pydantic_ai_pax.runtime.rewind import rewind_points
from pydantic_ai_pax.runtime.session_runtime import SessionRuntime, create_shell
from tests.pax.dialog_script import Back, Pick, ScriptedDialogs, Type

if TYPE_CHECKING:
    # Imported at run time only by the fixtures that need them, so collecting this file loads no SDK.
    from boxd import Machine, Secret

    from tests.harness.boxd_sandbox.fake_boxd import FakeBoxd, FakeClient
    from tests.harness.daytona_sandbox.fake_daytona import FakeDaytona, FakeSandbox as FakeDaytonaSandbox
    from tests.harness.e2b_sandbox.fake_e2b import FakeE2B
    from tests.harness.kubernetes_sandbox.fake_sandboxd import FakeCluster
    from tests.harness.modal_sandbox.fake_modal import FakeModal
    from tests.harness.opensandbox_sandbox.fake_opensandbox import FakeOpenSandbox
    from tests.harness.render_sandbox.workspace_helpers import WorkspaceHarness
    from tests.harness.smol_sandbox.fake_smolvm import FakeSmolvm
    from tests.harness.sprites_sandbox.fake_sprites import SpriteTransport

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
    token_in_sandbox: bool = False
    requirement: str | None = None
    created: list[GitHubCredential | None] = field(default_factory=list[GitHubCredential | None])
    reopened: list[WorkspaceRef] = field(default_factory=list[WorkspaceRef])
    deleted: list[WorkspaceRef] = field(default_factory=list[WorkspaceRef])
    closed: list[WorkspaceRef] = field(default_factory=list[WorkspaceRef])
    creating: anyio.Event | None = None
    """When set, `create` waits for it after starting, as a slow provider would."""
    delete_error: Exception | None = None

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
    loaded: LoadedPlugin[None],
    checkout: Path,
    *,
    model: TestModel | FunctionModel | None = None,
    shell: bool = False,
) -> Session[None, str]:
    """A session over `loaded`'s capabilities, put on screen as the conversation the plugin's commands act on.

    With `shell`, the agent also has harness `Shell`'s tools, which run in the run's workspace.
    """
    agent = Agent(model or TestModel(call_tools=[], custom_output_text='done'), deps_type=type(None))
    plugins = [*loaded.capabilities, *([Shell[None]()] if shell else [])]
    session = Session(agent, deps=None, plugins=plugins, workspace=checkout)
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


def test_a_saved_key_is_trimmed_and_a_malformed_one_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('E2B_API_KEY', ' e2b_key\n')
    assert saved_key('E2B_API_KEY') == 'e2b_key'
    monkeypatch.setenv('E2B_API_KEY', '   ')
    assert saved_key('E2B_API_KEY') is None
    # An SDK would send these in a header, and the HTTP library refusing it would quote the key in its error.
    for malformed in ('e2b key', 'e2b\nkey', 'e2b\x01key'):
        monkeypatch.setenv('E2B_API_KEY', malformed)
        with pytest.raises(UserError) as caught:
            saved_key('E2B_API_KEY')
        assert str(caught.value) == 'E2B_API_KEY contains a space or control character; save it again in /keys.'
        assert 'e2b' not in str(caught.value).replace('E2B_API_KEY', '')


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
            'This conversation runs on this machine. /sandbox fake moves a new conversation into a sandbox.'
        )
        message = await plugin.command(['fake'])
        head = git(checkout, 'rev-parse', '--short=7', 'HEAD')
        assert message == (
            f'This conversation now runs in Fake sandbox sbx-1. /home/user/workspace holds feature at {head}, '
            'with your uncommitted changes.\n'
            'git and gh there act as @Ada, on every repository that login reaches. '
            "Fake's proxy, outside the sandbox, holds the token until the sandbox is deleted.\n"
            'Your files here stay as they are: ask the agent to push, then pull here. ! commands run in the sandbox too.'
        )
        assert provider.created == [GitHubCredential(login='Ada', token='gho_secret')]
        # The history records the sandbox before any prompt, so every copy of the conversation finds it.
        request, response = loaded.host.conversation.messages
        assert isinstance(request, ModelRequest)
        [part] = request.parts
        # CLAI's note, not a typed prompt, so `/rewind` never cuts the history back past the sandbox.
        assert imported_kind(part) == 'sandbox'
        # The response only names the sandbox: with no parts, the model is not made to have said anything.
        assert isinstance(response, ModelResponse)
        assert (response.parts, response.workspace_ref) == ([], WorkspaceRef(provider='fake', id='sbx-1'))
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
        assert list(plugin_of(load(checkout)).complete([])) == [
            'local',
            'e2b',
            'boxd',
            'daytona',
            'opensandbox',
            'modal',
            'render',
            'sprites',
            'kubernetes',
            'delete',
        ]

    @pytest.mark.parametrize('origin', [None, 'https://github.com/owner/private.git'])
    @pytest.mark.parametrize('gap', ['no gh', 'no login', 'token refused'])
    async def test_without_a_github_login(
        self, checkout: Path, provider: FakeProvider, monkeypatch: pytest.MonkeyPatch, gap: str, origin: str | None
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
        if origin is None:
            git(checkout, 'remote', 'remove', 'origin')
        else:
            # A repository on GitHub the sandbox cannot clone without a login, as the fake has no `private.git`.
            git(checkout, 'remote', 'set-url', 'origin', origin)
        message = await plugin_of(load(checkout)).command(['fake'])
        assert (
            'gh has no GitHub login here, so the sandbox cannot push or reach private repositories. '
            'Run /sandbox delete and gh auth login, then /new and /sandbox fake again.'
        ) in message
        assert provider.created == [None]
        # The whole history went over in the bundle instead of a clone.
        copy = provider.folder(WorkspaceRef(provider='fake', id='sbx-1'))
        assert git(copy, 'rev-parse', 'HEAD') == git(checkout, 'rev-parse', 'HEAD')

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

    async def test_a_failed_login_whose_delete_failed_stays_named(
        self, checkout: Path, provider: FakeProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The login runs after the sandbox is named, so a sandbox that may hold the token is never lost.
        provider.token_in_sandbox = True
        provider.delete_error = RuntimeError('network down')
        monkeypatch.setenv('HOME', 'relative')
        plugin = plugin_of(load(checkout))
        with pytest.raises(RuntimeError, match='network down'):
            await plugin.command(['fake'])
        assert history_sandbox(plugin.host.conversation.messages) == WorkspaceRef(provider='fake', id='sbx-1')
        provider.delete_error = None
        assert await plugin.command(['delete']) == 'Deleted Fake sandbox sbx-1. Run /new to start another conversation.'

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


def sandboxed_shell(
    loaded: LoadedPlugin[None], checkout: Path, tmp_path: Path
) -> tuple[SessionRuntime[None, str], io.StringIO]:
    """A headless shell on `checkout` with the plugin's capability, whose conversation the plugin's commands act on."""
    output = io.StringIO()
    shell = create_shell(
        Agent(TestModel(), deps_type=type(None)),
        deps=None,
        plugins=loaded.capabilities,
        usage_limits=None,
        console=Console(file=output, force_terminal=False, width=200),
        settings=Settings(model='test'),
        store=SettingsStore(tmp_path / 'config.db'),
        builtin_plugins=(),
        project=ProjectSettings(),
        headless=True,
        workspace=checkout,
    )
    loaded.host.conversation = shell.session
    return shell, output


def saved_prompts(shell: SessionRuntime[None, str]) -> list[str]:
    return [
        part.content
        for message in shell.session.messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart) and isinstance(part.content, str)
    ]


@pytest.mark.skipif(sys.platform == 'win32', reason='uses POSIX shell syntax')
async def test_shell_lines_run_in_the_sandbox(checkout: Path, provider: FakeProvider, tmp_path: Path) -> None:
    """A `!` line in a sandboxed conversation runs in the sandbox, where the agent's commands run, and says so."""
    loaded = load(checkout)
    shell, output = sandboxed_shell(loaded, checkout, tmp_path)
    await plugin_of(loaded).command(['fake'])
    folder = provider.folder(WorkspaceRef(provider='fake', id='sbx-1')).resolve()
    clipboard = "!printf '\\033]52;c;aGk=\\007'"  # A file the sandbox prints could set this machine's clipboard.
    for line in ('!pwd -P', "!head -c 100001 /dev/zero | tr '\\0' x >&2; exit 3", clipboard, '!!vim', '/exit'):
        shell.send_input(line)
    assert await shell.run() == 'exit'

    text = output.getvalue()
    assert f'Runs in the Fake sandbox; command and output saved for the next prompt\n{folder}\nDone (' in text
    assert str(checkout.resolve()) not in text
    assert '!! needs this terminal, which the Fake sandbox does not have. Nothing ran.' in text
    cut = f'{"x" * 100_000}\n[Output truncated after 100000 characters]\n'
    assert '\x1b' not in text and '\\x1b]52;c;aGk=\\x07' in text
    assert '\x1b]52;c;aGk=\x07' in saved_prompts(shell)[-1]
    assert saved_prompts(shell)[-3:-1] == [
        f'The user ran a shell command in the Fake sandbox (not an agent tool call):\n'
        f'$ pwd -P\nExit code 0\n\nstdout:\n{folder}\n\n\nstderr:\n',
        f'The user ran a shell command in the Fake sandbox (not an agent tool call):\n'
        f"$ head -c 100001 /dev/zero | tr '\\0' x >&2; exit 3\nExit code 3\n\nstdout:\n\n\nstderr:\n{cut}",
    ]


@pytest.mark.parametrize('failure', ['stopped', 'deleted', 'unreachable'])
async def test_a_shell_line_reports_a_sandbox_it_cannot_use(
    checkout: Path,
    provider: FakeProvider,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: Literal['stopped', 'deleted', 'unreachable'],
) -> None:
    """A sandbox that fails the command, was deleted, or cannot be reached is reported; the shell goes on.

    The `!` line never falls back to this machine.
    """

    async def stopped(self: FakeSandboxBackend, command: WorkspaceCommand, **kwargs: object) -> CommandResult:
        raise WorkspaceUnavailableError('The Fake sandbox stopped.')

    def unreachable(self: object, ref: WorkspaceRef) -> object:
        raise RuntimeError('The Fake API returned 502.')

    loaded = load(checkout)
    shell, output = sandboxed_shell(loaded, checkout, tmp_path)
    plugin = plugin_of(loaded)
    await plugin.command(['fake'])
    if failure == 'deleted':
        await plugin.command(['delete'])
    elif failure == 'unreachable':
        monkeypatch.setattr(sandbox._ConversationSandbox, '_workspace', unreachable)  # pyright: ignore[reportPrivateUsage]
    else:
        monkeypatch.setattr(FakeSandboxBackend, 'run', stopped)
    saved = saved_prompts(shell)
    shell.send_input('!pwd')
    shell.send_input('/exit')
    assert await shell.run() == 'exit'

    text = output.getvalue()
    if failure == 'deleted':
        assert "This conversation's Fake sandbox was deleted. Run /new to start another." in text
        assert saved_prompts(shell) == saved
    elif failure == 'unreachable':
        assert 'The Fake API returned 502.' in text
        assert saved_prompts(shell) == saved
    else:
        assert 'Shell error: The Fake sandbox stopped.' in text
        assert saved_prompts(shell) == [
            *saved,
            'The user ran a shell command in the Fake sandbox (not an agent tool call):\n'
            '$ pwd\nShell error: The Fake sandbox stopped.\n\nstdout:\n\n\nstderr:\n',
        ]
    assert str(checkout.resolve()) not in text


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


Step = tuple[str, dict[str, str]] | str
"""A tool call the model makes, by name and arguments, or a text answer that ends the turn."""


def steps(*script: Step) -> tuple[FunctionModel, list[str]]:
    """A model that takes `script`'s steps in order, one per request, across turns; and the tool returns it read."""
    pending = list(script)
    returns: list[str] = []

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        last = messages[-1].parts[-1]
        if isinstance(last, ToolReturnPart):
            returns.append(str(last.content))
        step = pending.pop(0)
        if isinstance(step, str):
            yield step
        else:
            name, args = step
            yield {0: DeltaToolCall(name=name, json_args=json.dumps(args))}

    return FunctionModel(stream_function=respond), returns


ASK = ('request_env', {'name': 'ANTHROPIC_API_KEY', 'reason': 'run the eval script'})


def write_key(file: str) -> Step:
    """A shell call that writes the command's `ANTHROPIC_API_KEY` to `file` in the sandbox, never to the model."""
    return ('run_command', {'command': f'printf %s "${{ANTHROPIC_API_KEY-unset}}" > {file}'})


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
        model, returns = steps(write_key('before'), ASK, write_key('same-turn'), 'done', write_key('next-turn'), 'done')
        session = on_screen(loaded, checkout, model=model, shell=True)
        await plugin_of(loaded).command(['fake'])
        await session.prompt('run the evals')
        result = await session.prompt('and again')
        folder = provider.folder(WorkspaceRef(provider='fake', id='sbx-1'))
        # What each command's own environment held: a later command in the same turn and the next turn get it.
        assert [(folder / file).read_text() for file in ('before', 'same-turn', 'next-turn')] == [
            'unset',
            'sk-typed',
            'sk-typed',
        ]
        assert returns[1] == (
            'ANTHROPIC_API_KEY is set for every command started from now on, until CLAI restarts; ask again if a '
            'command then finds it missing. Restart a server started before it.'
        )
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
        model, _ = steps(ASK, write_key('seen'), 'done')
        session = on_screen(loaded, checkout, model=model, shell=True)
        await plugin_of(loaded).command(['fake'])
        await session.prompt('run the evals')
        assert (provider.folder(WorkspaceRef(provider='fake', id='sbx-1')) / 'seen').read_text() == 'sk-saved'

    async def test_control_characters_in_the_reason_are_shown_escaped(
        self, checkout: Path, provider: FakeProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dialogs = ScriptedDialogs([Back()])
        monkeypatch.setattr(sandbox, 'TerminalDialogs', scripted(dialogs))
        loaded = load(checkout, terminal=True)
        reason = 'run evals\r\x1b[2KCheck configuration\x1b[8m'
        model, returns = steps(('request_env', {'name': 'ANTHROPIC_API_KEY', 'reason': reason}), 'done')
        session = on_screen(loaded, checkout, model=model, shell=True)
        await plugin_of(loaded).command(['fake'])
        await session.prompt('run the evals')
        [asked] = dialogs.asked
        assert asked.title == (
            'The agent asks for ANTHROPIC_API_KEY to run evals\\x0d\\x1b[2KCheck configuration\\x1b[8m. '
            'Anything run in the sandbox can read it.'
        )
        assert returns[0] == 'The user did not give ANTHROPIC_API_KEY.'

    async def test_a_background_command_gets_the_key(
        self, checkout: Path, provider: FakeProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`start_command` launches its job with the workspace's `run`, so the job starts with the key."""
        launched: list[Mapping[str, str] | None] = []

        async def run(
            self: FakeSandboxBackend,
            command: WorkspaceCommand,
            *,
            shell: bool = False,
            env: Mapping[str, str] | None = None,
            timeout: float | None = None,
        ) -> CommandResult:
            launched.append(env)
            return await LocalWorkspaceBackend.run(self, command, shell=shell, env=env, timeout=timeout)

        monkeypatch.setattr(FakeSandboxBackend, 'run', run)
        monkeypatch.setattr(sandbox, 'TerminalDialogs', scripted(ScriptedDialogs([Type(text='sk-typed')])))
        loaded = load(checkout, terminal=True)
        model, _ = steps(ASK, ('start_command', {'command': 'true'}), 'done')
        session = on_screen(loaded, checkout, model=model, shell=True)
        await plugin_of(loaded).command(['fake'])
        launched.clear()  # `/sandbox`'s own setup commands.
        await session.prompt('start the server')
        assert launched and all(env is not None and env['ANTHROPIC_API_KEY'] == 'sk-typed' for env in launched)

    @pytest.mark.parametrize(
        ('name', 'expected'),
        [
            ('ANTHROPIC_API_KEY', 'The user did not give ANTHROPIC_API_KEY.'),
            ('api key', "'api key' is not an environment variable name: use capitals, digits, and underscores."),
            # A trailing newline is not part of a name.
            (
                'API_KEY\n',
                "'API_KEY\\n' is not an environment variable name: use capitals, digits, and underscores.",
            ),
            ('GH_TOKEN', 'CLAI sets GH_TOKEN in the sandbox itself, so it cannot be replaced.'),
        ],
        ids=['declined', 'bad name', 'trailing newline', 'GitHub stand-in'],
    )
    async def test_nothing_is_set(
        self,
        checkout: Path,
        provider: FakeProvider,
        monkeypatch: pytest.MonkeyPatch,
        name: str,
        expected: str,
    ) -> None:
        monkeypatch.setattr(sandbox, 'TerminalDialogs', scripted(ScriptedDialogs([Back()])))
        loaded = load(checkout, terminal=True)
        model, returns, _ = asks_for_a_key(name)
        session = on_screen(loaded, checkout, model=model)
        await plugin_of(loaded).command(['fake'])
        await session.prompt('run the evals')
        assert returns == [expected]
        assert plugin_of(loaded).sandbox(WorkspaceRef(provider='fake', id='sbx-1')).env == {}

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
        assert plugin_of(loaded).sandbox(WorkspaceRef(provider='fake', id='sbx-1')).env == {}

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
            "E2B support is not installed. Install CLAI with it: uv tool install 'pydantic-ai-pax[e2b]'."
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
            "boxd support is not installed. Install CLAI with it: uv tool install 'pydantic-ai-pax[boxd]'."
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
        # Deleting the machine deletes its secret, so the token ends with it.
        await provider.delete(ref)
        assert machine.status == 'destroyed'
        assert account.secrets == {}
        # Without a GitHub login: no secret, and nothing for the spec to keep.
        ref, backend = await provider.create(github=None)
        assert account.secrets == {}
        assert ref.id not in account.specs
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


@pytest.mark.parametrize('gh_token', [None, 'stand-in'])
def test_the_credential_helper(tmp_path: Path, gh_token: str | None) -> None:
    """A proxy provider's stand-in in `GH_TOKEN` wins; without one, git reads the token saved in the sandbox."""
    [line] = [line for line in clone_script(Snapshot()).splitlines() if '.helper ' in line]
    helper = shlex.split(line)[-1].removeprefix('!')
    home = tmp_path / 'sandbox-home'
    (home / '.config/clai').mkdir(parents=True)
    (home / '.config/clai/github-token').write_text('gho_saved', encoding='utf-8')
    env = {'HOME': str(home), 'PATH': os.environ['PATH'], **({} if gh_token is None else {'GH_TOKEN': gh_token})}
    result = subprocess.run(['sh', '-c', f'{helper} get'], env=env, capture_output=True, text=True, check=True)
    assert result.stdout == f'username=x-access-token\npassword={gh_token or "gho_saved"}\n'


async def test_a_provider_that_puts_the_token_in_the_sandbox_says_so(checkout: Path, provider: FakeProvider) -> None:
    provider.token_in_sandbox = True
    loaded = load(checkout)
    message = await plugin_of(loaded).command(['fake'])
    # `/sandbox` wrote the login into the sandbox's home, which the fake shares with the test.
    assert (Path(os.environ['HOME']) / '.config/clai/github-token').read_text(encoding='utf-8') == 'gho_secret'
    assert (
        'The token is saved in the sandbox, where anything run there can read it, until the sandbox is deleted.'
        in message
    )
    agent = Agent(TestModel(call_tools=[], custom_output_text='done'), deps_type=type(None))
    session = Session(agent, deps=None, plugins=list(loaded.capabilities), workspace=checkout)
    session.replace_messages(loaded.host.conversation.messages)
    result = await session.prompt('there')
    [request] = [message for message in result.new_messages() if isinstance(message, ModelRequest)]
    assert (
        'signed in to GitHub as the user, with a token saved in the sandbox, which anything run there can read; '
        'leave GH_TOKEN unset. Otherwise the copy has no GitHub access and no origin remote. Never run `gh auth login`.'
    ) in (request.instructions or '')


GITHUB = GitHubCredential(login='Ada', token='gho_x')

SIGNED_IN = {
    '.config/clai/github-token': ('gho_x', 0o600),
    '.config/gh/hosts.yml': ('github.com:\n    oauth_token: gho_x\n    user: Ada\n    git_protocol: https\n', 0o600),
}


def signed_in(home: Path) -> dict[str, tuple[str, int]]:
    """The GitHub login `give_github_login` wrote under `home`, with each file's mode."""
    assert stat.S_IMODE((home / '.config/clai').stat().st_mode) == 0o700
    return {
        name: ((home / name).read_text(encoding='utf-8'), stat.S_IMODE((home / name).stat().st_mode))
        for name in SIGNED_IN
        if (home / name).exists()
    }


def not_installed(name: str) -> None:
    return None


def installed(name: str) -> str:
    return f'/usr/bin/{name}'


@dataclass
class CommandOnlyBackend:
    """A sandbox that only runs commands, with no file API, as some providers offer."""

    commands: FakeSandboxBackend

    # Required by the backend protocol; seeding and the login never read them.
    @property
    def ref(self) -> WorkspaceRef:  # pragma: no cover
        return self.commands.ref

    async def working_dir(self) -> str:  # pragma: no cover
        return await self.commands.working_dir()

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        return await self.commands.run(command, shell=shell, env=env, timeout=timeout)


async def test_a_sandbox_that_only_runs_commands(checkout: Path, provider: FakeProvider, tmp_path: Path) -> None:
    """Seeding and the in-sandbox GitHub login write their files with shell commands when there is no file API."""
    home = tmp_path / 'sandbox-home'
    home.mkdir()
    ref, _ = await provider.create(github=None)
    copy = provider.folder(ref)
    env = {**SANDBOX_ENV, 'GIT_CONFIG_GLOBAL': f'{copy}.gitconfig', 'GIT_CONFIG_NOSYSTEM': '1', 'HOME': str(home)}
    backend = CommandOnlyBackend(FakeSandboxBackend(ref, copy, env=env))
    assert not isinstance(backend, SupportsFilesystem)

    await seed(backend, await to_thread.run_sync(snapshot, checkout))
    assert git(copy, 'rev-parse', 'HEAD') == git(checkout, 'rev-parse', 'HEAD')
    assert git(copy, 'status', '--porcelain') == ' M README.md\n?? untracked.txt'

    # An existing `~/.config/gh` is made private too.
    (home / '.config/gh').mkdir(parents=True, mode=0o755)
    await give_github_login(backend, GITHUB)
    assert signed_in(home) == SIGNED_IN
    assert stat.S_IMODE((home / '.config/gh').stat().st_mode) == 0o700


@pytest.mark.parametrize('home', ['none', 'a file'])
async def test_a_login_with_no_home_folder_fails(
    provider: FakeProvider, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, home: str
) -> None:
    # No home folder to sign in under, or one where the login's folders cannot be made.
    if home == 'none':
        monkeypatch.setenv('HOME', '')
        raised = 'no home folder'
    else:
        (tmp_path / 'file').touch()
        monkeypatch.setenv('HOME', str(tmp_path / 'file'))
        raised = r'^Could not sign the sandbox in to GitHub: mkdir: .*'
    _, backend = await provider.create(github=None)
    with pytest.raises(RuntimeError, match=raised):
        await give_github_login(backend, GITHUB)


@dataclass
class FakeDaytonaSecrets:
    """Daytona's organization secrets, which the harness's Daytona fake leaves out."""

    secrets: dict[str, tuple[str, str, list[str]]] = field(default_factory=dict[str, tuple[str, str, list[str]]])
    """Each secret's name, value, and hosts, by id."""
    numbers: Iterator[int] = field(default_factory=lambda: itertools.count(1))

    async def create(self, params: object) -> object:
        from types import SimpleNamespace

        number = next(self.numbers)
        name, value, hosts = getattr(params, 'name'), getattr(params, 'value'), getattr(params, 'hosts')
        self.secrets[f'sec-{number}'] = (name, value, hosts)
        return SimpleNamespace(id=f'sec-{number}', name=name, placeholder=f'dtn_secret_{number}', hosts=hosts)

    delete_error: Exception | None = None

    async def delete(self, secret_id: str) -> None:
        from daytona_api_client_async.exceptions import NotFoundException

        if self.delete_error is not None:
            raise self.delete_error
        if self.secrets.pop(secret_id, None) is None:
            raise NotFoundException(status=404, reason='Not Found')


@pytest.fixture
def fake_daytona(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> 'Iterator[tuple[FakeDaytona, FakeDaytonaSecrets]]':
    """The harness's Daytona fake, whose commands run here, with secrets and labels added."""
    import daytona

    from tests.harness.daytona_sandbox.fake_daytona import FakeClient, FakeDaytona

    fake = FakeDaytona()
    fake.host_root = tmp_path.resolve()
    account = FakeDaytonaSecrets()
    clients: list[FakeClient] = []

    def client(config: object = None) -> FakeClient:
        made = fake.client(config)
        clients.append(made)
        made.secret = account  # type: ignore[attr-defined]
        create = made.create

        async def create_with_labels(
            params: 'daytona.CreateSandboxFromSnapshotParams', *, timeout: float = 60
        ) -> 'FakeDaytonaSandbox':
            sandbox = await create(params, timeout=timeout)
            sandbox.labels = dict(params.labels or {})  # type: ignore[attr-defined]

            async def set_labels(labels: dict[str, str]) -> dict[str, str]:
                sandbox.labels = dict(labels)  # type: ignore[attr-defined]
                return labels

            sandbox.set_labels = set_labels  # type: ignore[attr-defined]
            return sandbox

        made.create = create_with_labels  # type: ignore[method-assign]
        return made

    monkeypatch.setattr('pydantic_ai_harness.daytona_sandbox._backend.daytona', daytona)
    monkeypatch.setattr(daytona, 'AsyncDaytona', client)
    monkeypatch.delenv(daytona_provider.KEY_NAME, raising=False)
    yield fake, account
    assert all(made.closed for made in clients)


@pytest.mark.skipif(find_spec('daytona') is None, reason='daytona not installed')
class TestDaytona:
    """Daytona, through the SDK fake the harness's own Daytona tests use; the swap itself was proven live."""

    def test_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(daytona_provider.KEY_NAME, raising=False)
        provider = daytona_provider.DaytonaProvider()
        assert provider.missing() == (
            'Add your Daytona API key in /keys as DAYTONA_API_KEY. Make one at https://app.daytona.io/dashboard/keys.'
        )
        save_key(name='DAYTONA_API_KEY', value='dtn_key')
        assert provider.missing() is None
        monkeypatch.setattr(daytona_provider, 'find_spec', not_installed)
        assert provider.missing() == (
            "Daytona support is not installed. Install CLAI with it: uv tool install 'pydantic-ai-pax[daytona]'."
        )

    async def test_create_reopen_delete(
        self,
        fake_daytona: 'tuple[FakeDaytona, FakeDaytonaSecrets]',
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        fake, account = fake_daytona
        monkeypatch.setattr(uuid, 'uuid4', lambda: uuid.UUID(int=0))
        # Daytona sets `GH_TOKEN` to the secret's stand-in; the fake's commands run here, so this stands in for that.
        monkeypatch.setenv('GH_TOKEN', 'dtn_secret_1')
        provider = daytona_provider.DaytonaProvider(working_dir=str(tmp_path / 'workspace'))
        ref, backend = await provider.create(github=GITHUB)
        secret = 'CLAI_GITHUB_' + '0' * 32
        assert account.secrets == {'sec-1': (secret, 'gho_x', list(GITHUB_HOSTS))}
        [params] = fake.create_params
        # Only the secret's name reaches the sandbox; Daytona's proxy holds the token.
        assert params.env_vars == SANDBOX_ENV
        assert params.secrets == {'GH_TOKEN': secret}
        [made] = fake.sandboxes
        assert made.labels == {'created-by': 'pydantic-ai', 'clai-github-secret': 'sec-1'}  # type: ignore[attr-defined]
        assert ref == backend.ref == WorkspaceRef(provider='daytona', id='sb-1')
        reopened = provider.reopen(ref)
        assert reopened.ref == ref
        # `request_env`'s keys reach a command as its own `env`.
        assert (await reopened.run(['printenv', 'API_KEY'], env={'API_KEY': 'sk-1'})).stdout == 'sk-1\n'
        await provider.close(reopened)
        await provider.close(backend)
        await provider.close(backend)  # Closing again has nothing left to close.
        # Deleting the sandbox deletes its secret, so the token ends with it.
        await provider.delete(ref)
        assert account.secrets == {}
        # Deleting again finds neither.
        fake.purge(made)
        await provider.delete(ref)
        # Without a GitHub login there is no secret to make.
        ref, backend = await provider.create(github=None)
        assert account.secrets == {}
        assert fake.create_params[-1].secrets is None
        await provider.close(backend)
        await provider.delete(ref)

    async def test_a_secret_deleted_already_is_fine(
        self,
        fake_daytona: 'tuple[FakeDaytona, FakeDaytonaSecrets]',
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _, account = fake_daytona
        monkeypatch.setenv('GH_TOKEN', 'dtn_secret_1')
        provider = daytona_provider.DaytonaProvider(working_dir=str(tmp_path / 'workspace'))
        ref, backend = await provider.create(github=GITHUB)
        await provider.close(backend)
        account.secrets.clear()
        await provider.delete(ref)

    async def test_a_failed_secret_delete_keeps_the_sandbox_for_a_retry(
        self,
        fake_daytona: 'tuple[FakeDaytona, FakeDaytonaSecrets]',
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        from daytona_api_client_async.exceptions import ApiException

        fake, account = fake_daytona
        monkeypatch.setenv('GH_TOKEN', 'dtn_secret_1')
        provider = daytona_provider.DaytonaProvider(working_dir=str(tmp_path / 'workspace'))
        ref, backend = await provider.create(github=GITHUB)
        await provider.close(backend)
        # What the secrets API raises, unwrapped by the SDK.
        account.delete_error = ApiException(status=503)
        with pytest.raises(ApiException):
            await provider.delete(ref)
        # The sandbox, and the label naming the secret, are still there, so a retry deletes both.
        assert any(sandbox.state == 'started' for sandbox in fake.sandboxes)
        account.delete_error = None
        await provider.delete(ref)
        assert account.secrets == {}

    async def test_a_failed_create_whose_sandbox_delete_fails_still_deletes_the_secret(
        self,
        fake_daytona: 'tuple[FakeDaytona, FakeDaytonaSecrets]',
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _, account = fake_daytona
        # Daytona put the token itself in the sandbox, so `create` fails after making the sandbox and secret.
        monkeypatch.setenv('GH_TOKEN', 'gho_x')

        async def destroy(self: object, client: object, ref: WorkspaceRef) -> None:
            raise OSError('Daytona is down')

        monkeypatch.setattr(daytona_provider.DaytonaProvider, '_destroy', destroy)
        provider = daytona_provider.DaytonaProvider(working_dir=str(tmp_path / 'workspace'))
        with pytest.raises(OSError, match='Daytona is down'):
            await provider.create(github=GITHUB)
        # The token does not outlive the failed create, even though the sandbox does.
        assert account.secrets == {}

    @pytest.mark.parametrize(
        'failure', ['sandbox refused', 'sandbox refused without a login', 'stand-in is the token', 'no stand-in']
    )
    async def test_a_failed_create_leaves_nothing_behind(
        self,
        fake_daytona: 'tuple[FakeDaytona, FakeDaytonaSecrets]',
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        failure: str,
    ) -> None:
        import daytona

        fake, account = fake_daytona
        if failure.startswith('sandbox refused'):
            fake.create_error = daytona.DaytonaError('quota exceeded')
            raised: type[Exception] = daytona.DaytonaError
        else:
            # Daytona put the token itself in the sandbox, or nothing, instead of a stand-in.
            if failure == 'stand-in is the token':
                monkeypatch.setenv('GH_TOKEN', 'gho_x')
            else:
                monkeypatch.delenv('GH_TOKEN', raising=False)
            raised = RuntimeError
        provider = daytona_provider.DaytonaProvider(working_dir=str(tmp_path / 'workspace'))
        with pytest.raises(raised):
            await provider.create(github=None if failure.endswith('without a login') else GITHUB)
        assert account.secrets == {}
        assert all(sandbox.state != 'started' for sandbox in fake.sandboxes)


@pytest.fixture
async def fake_modal(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> 'AsyncIterator[FakeModal]':
    """The harness's Modal fake as `modal`, whose commands run here."""
    from tests.harness.modal_sandbox.fake_modal import FakeModal

    control = FakeModal()
    control.host_root = tmp_path.resolve()
    control.module.__spec__ = importlib.machinery.ModuleSpec('modal', None)
    monkeypatch.setitem(sys.modules, 'modal', control.module)
    yield control
    for made in control.sandboxes:
        await made.terminate.aio()


@pytest.mark.skipif(find_spec('modal') is None, reason='modal not installed')
class TestModal:
    """Modal, through the SDK fake the harness's own Modal tests use."""

    def test_missing(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        config = tmp_path / 'modal.toml'
        monkeypatch.setenv('MODAL_CONFIG_PATH', str(config))
        monkeypatch.delenv('MODAL_TOKEN_ID', raising=False)
        monkeypatch.delenv('MODAL_TOKEN_SECRET', raising=False)
        provider = modal_provider.ModalProvider()
        assert provider.missing() == 'Sign in to Modal first: run uvx modal token new, then try again.'
        monkeypatch.setenv('MODAL_TOKEN_ID', 'ak-1')
        assert provider.missing() is not None
        monkeypatch.setenv('MODAL_TOKEN_SECRET', 'as-1')
        assert provider.missing() is None
        monkeypatch.delenv('MODAL_TOKEN_ID')
        config.write_text('[default]\n', encoding='utf-8')
        assert provider.missing() is None
        monkeypatch.setattr(modal_provider, 'find_spec', not_installed)
        assert provider.missing() == (
            "Modal support is not installed. Install CLAI with it: uv tool install 'pydantic-ai-pax[modal]'."
        )

    async def test_create_reopen_delete(self, fake_modal: 'FakeModal', tmp_path: Path) -> None:
        workspace = tmp_path / 'workspace'
        workspace.mkdir()  # Modal makes the sandbox's working folder itself.
        provider = modal_provider.ModalProvider(working_dir=str(workspace))
        ref, backend = await provider.create(github=GITHUB)
        assert ref == backend.ref == WorkspaceRef(provider='modal', id='sb-owned')
        [kwargs] = fake_modal.create_kwargs
        # Modal's longest lifetime, and no idle timeout, which would end the sandbox rather than pause it.
        assert (kwargs['timeout'], kwargs['idle_timeout']) == (86_400, None)
        assert kwargs['env'] == SANDBOX_ENV
        await give_github_login(backend, GITHUB)
        assert signed_in(Path(os.environ['HOME'])) == SIGNED_IN
        [made] = fake_modal.sandboxes
        # The token went by the file API, never in a command.
        assert not any('gho_x' in ' '.join(call.argv) for call in made.exec_calls)
        reopened = provider.reopen(ref)
        # `request_env`'s keys reach a command as its own `env`.
        assert (await reopened.run(['printenv', 'API_KEY'], env={'API_KEY': 'sk-1'})).stdout == 'sk-1\n'
        await provider.close(reopened)
        await provider.delete(ref)
        assert made.shutting_down
        ref, _ = await provider.create(github=None)
        assert ref.id == 'sb-owned-2'

    async def test_a_failed_create_leaves_nothing_behind(self, fake_modal: 'FakeModal', tmp_path: Path) -> None:
        workspace = tmp_path / 'workspace'
        workspace.mkdir()
        fake_modal.create_error = OSError('Modal is down')
        with pytest.raises(OSError, match='Modal is down'):
            await modal_provider.ModalProvider(working_dir=str(workspace)).create(github=GITHUB)


@pytest.fixture
def fake_render(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> 'Iterator[WorkspaceHarness]':
    """The harness's Render fake, whose commands run here; every client CLAI makes is wired to it."""
    import render

    from tests.harness.render_sandbox.workspace_helpers import client_is_closed, make_workspace_harness

    harness = make_workspace_harness(monkeypatch, tmp_path)
    made: list[render.RenderAsync] = []
    real = render.RenderAsync

    def client(*, token: str | None, owner_id: str | None) -> render.RenderAsync:
        assert (token, owner_id) == ('rnd_key', 'tea-1')
        opened = real(token=token, owner_id=owner_id)
        harness.install(opened)
        made.append(opened)
        return opened

    monkeypatch.setattr(render, 'RenderAsync', client)
    monkeypatch.delenv(render_provider.KEY_NAME, raising=False)
    monkeypatch.delenv(render_provider.WORKSPACE_KEY_NAME, raising=False)
    yield harness
    assert all(client_is_closed(opened) for opened in made)


@pytest.mark.skipif(find_spec('render') is None, reason='render not installed')
class TestRender:
    """Render, through the SDK fake the harness's own Render tests use."""

    def test_missing(self, fake_render: 'WorkspaceHarness', monkeypatch: pytest.MonkeyPatch) -> None:
        provider = render_provider.RenderProvider()
        message = (
            'Add your Render API key in /keys as RENDER_API_KEY, and the tea-... ID of a workspace with sandboxes '
            'as RENDER_WORKSPACE_ID. Make a key as https://render.com/docs/api#1-create-an-api-key says.'
        )
        assert provider.missing() == message
        save_key(name='RENDER_API_KEY', value='rnd_key')
        assert provider.missing() == message
        monkeypatch.setenv('RENDER_WORKSPACE_ID', 'tea-1')
        assert provider.missing() is None
        monkeypatch.setattr(render_provider, 'find_spec', not_installed)
        assert provider.missing() == (
            "Render support is not installed. Install CLAI with it: uv tool install 'pydantic-ai-pax[render]'."
        )

    async def test_create_reopen_delete(self, fake_render: 'WorkspaceHarness', tmp_path: Path) -> None:
        save_key(name='RENDER_API_KEY', value='rnd_key')
        save_key(name='RENDER_WORKSPACE_ID', value='tea-1')
        provider = render_provider.RenderProvider(working_dir=str(tmp_path / 'workspace'))
        ref, backend = await provider.create(github=GITHUB)
        assert ref == backend.ref == WorkspaceRef(provider='render', id='sb-1')
        # Render's longest lifetime, and the backend made the working folder.
        assert fake_render.create_calls['sb-1'].timeout_seconds == 86_400
        assert (tmp_path / 'workspace').is_dir()
        assert (await backend.run(['printenv', 'GIT_TERMINAL_PROMPT'])).stdout == '0\n'
        await give_github_login(backend, GITHUB)
        assert signed_in(Path(os.environ['HOME'])) == SIGNED_IN
        # The token went by file transfer, never in a command.
        assert not any('gho_x' in str(command) for command in fake_render.commands)
        reopened = provider.reopen(ref)
        # `request_env`'s keys reach a command as its own `env`.
        assert (await reopened.run(['printenv', 'API_KEY'], env={'API_KEY': 'sk-1'})).stdout == 'sk-1\n'
        await provider.close(reopened)
        await provider.close(backend)
        await provider.close(backend)  # Closing again has nothing left to close.
        await provider.delete(ref)
        assert fake_render.terminated == ['sb-1']

    async def test_a_failed_create_leaves_nothing_behind(self, fake_render: 'WorkspaceHarness', tmp_path: Path) -> None:
        save_key(name='RENDER_API_KEY', value='rnd_key')
        save_key(name='RENDER_WORKSPACE_ID', value='tea-1')
        # Render made the sandbox, but the working folder cannot be made there.
        (tmp_path / 'file').touch()
        provider = render_provider.RenderProvider(working_dir=str(tmp_path / 'file/workspace'))
        with pytest.raises(WorkspaceUnavailableError, match='is unavailable in the sandbox'):
            await provider.create(github=GITHUB)
        assert fake_render.terminated == ['sb-1']
        fake_render.create_error = OSError('Render is down')
        with pytest.raises(OSError, match='Render is down'):
            await provider.create(github=GITHUB)
        assert fake_render.created == ['sb-1']


@pytest.fixture
async def fake_sprites(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> 'AsyncIterator[SpriteTransport]':
    """The harness's Sprites fake, as the harness's Sprites tests set it up, whose commands run here."""
    import sprites
    from sprites import AsyncSprite, AsyncSpritesClient
    from sprites.async_filesystem import AsyncSpritePath
    from sprites.types import FileStat

    from pydantic_ai_harness.sprites_sandbox._backend import _LookupCache  # pyright: ignore[reportPrivateUsage]
    from tests.harness.sprites_sandbox.fake_sprites import SpriteTransport

    transport = SpriteTransport(tmp_path)
    monkeypatch.delenv(sprites_provider.KEY_NAME, raising=False)
    monkeypatch.setattr('pydantic_ai_harness.sprites_sandbox._backend._HANDSHAKE_RETRY_DELAYS', (0.0, 0.0))
    monkeypatch.setattr('pydantic_ai_harness.sprites_sandbox._backend._lookups', _LookupCache())
    monkeypatch.setattr('sprites.websocket.connect', transport.connect)
    monkeypatch.setattr(sprites, 'AsyncSpritesClient', transport.client)

    async def get(client: AsyncSpritesClient, name: str) -> AsyncSprite:
        return await transport.get(client, name)

    async def create(client: AsyncSpritesClient, name: str, *, runtime: str | None) -> AsyncSprite:
        return await transport.create(client, name, runtime=runtime)

    async def close(client: AsyncSpritesClient) -> None:
        await transport.close(client)

    async def destroy(client: AsyncSpritesClient, name: str) -> None:
        await transport.destroy(client, name)

    async def fs_stat(path: AsyncSpritePath) -> FileStat:
        return await to_thread.run_sync(transport.fs_stat, path)

    async def fs_write(path: AsyncSpritePath, data: bytes, mode: int = 0o644, mkdir_parents: bool = True) -> None:
        await to_thread.run_sync(transport.fs_write, path, data, mode)

    monkeypatch.setattr(AsyncSpritesClient, 'get_sprite', get)
    monkeypatch.setattr(AsyncSpritesClient, 'create_sprite', create)
    monkeypatch.setattr(AsyncSpritesClient, 'aclose', close)
    monkeypatch.setattr(AsyncSpritesClient, 'destroy_sprite', destroy)
    monkeypatch.setattr(AsyncSpritePath, 'stat', fs_stat)
    monkeypatch.setattr(AsyncSpritePath, 'write_bytes', fs_write)
    yield transport
    # Each exec's thread posts its last frame to this loop, so it must finish while the loop runs.
    for socket in transport.execs:
        await to_thread.run_sync(socket.thread.join, 10)
    assert transport.close_calls == len(transport.clients)


@pytest.mark.skipif(find_spec('sprites') is None, reason='sprites-py not installed')
class TestSprites:
    """Sprites, through the SDK fake the harness's own Sprites tests use."""

    def test_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(sprites_provider.KEY_NAME, raising=False)
        provider = sprites_provider.SpritesProvider()
        assert provider.missing() == (
            'Add your Sprites token in /keys as SPRITE_TOKEN. Make one at https://sprites.dev/account.'
        )
        monkeypatch.setenv('SPRITE_TOKEN', 'spr-1')
        assert provider.missing() is None
        monkeypatch.setattr(sprites_provider, 'find_spec', not_installed)
        assert provider.missing() == (
            "Sprites support is not installed. Install CLAI with it: uv tool install 'pydantic-ai-pax[sprites]'."
        )

    async def test_create_reopen_delete(self, fake_sprites: 'SpriteTransport', tmp_path: Path) -> None:
        save_key(name='SPRITE_TOKEN', value='spr-1')
        provider = sprites_provider.SpritesProvider(working_dir=str(tmp_path / 'workspace'))
        ref, backend = await provider.create(github=GITHUB)
        [name] = fake_sprites.created
        assert ref == backend.ref == WorkspaceRef(provider='sprites', id=name)
        assert {client.token for client in fake_sprites.clients} == {'spr-1'}
        await give_github_login(backend, GITHUB)
        assert signed_in(Path(os.environ['HOME'])) == SIGNED_IN
        # The token went by the filesystem API, never in an exec URL.
        assert not any('gho_x' in socket.query_url for socket in fake_sprites.execs)
        reopened = provider.reopen(ref)
        # `request_env`'s keys reach a command as its own `env`.
        assert (await reopened.run(['printenv', 'API_KEY'], env={'API_KEY': 'sk-1'})).stdout == 'sk-1\n'
        await provider.close(reopened)
        await provider.close(backend)
        await provider.close(backend)  # Closing again has nothing left to close.
        await provider.delete(ref)
        assert fake_sprites.names == set()
        ref, backend = await provider.create(github=None)
        await provider.close(backend)
        await provider.delete(ref)

    async def test_a_failed_create_leaves_nothing_behind(self, fake_sprites: 'SpriteTransport', tmp_path: Path) -> None:
        save_key(name='SPRITE_TOKEN', value='spr-1')
        provider = sprites_provider.SpritesProvider(working_dir=str(tmp_path / 'workspace'))
        fake_sprites.create_error = OSError('Sprites is down')
        with pytest.raises(OSError, match='Sprites is down'):
            await provider.create(github=GITHUB)


@pytest.fixture
async def fake_opensandbox(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> 'AsyncIterator[FakeOpenSandbox]':
    """The harness's OpenSandbox fake server, behind every transport the SDK opens, at the address in `/keys`."""
    import httpx

    from tests.harness.opensandbox_sandbox.fake_opensandbox import (
        API_KEY,
        DOMAIN,
        FakeOpenSandbox,
        OwnedTransport,
        reap,
    )

    server = FakeOpenSandbox(pid_dir=str(tmp_path / 'pids'))

    def transport(**_: object) -> OwnedTransport:
        return OwnedTransport(server)

    monkeypatch.setattr(httpx, 'AsyncHTTPTransport', transport)
    # The fake's commands get only the environment they are sent; a real sandbox has a home of its own.
    monkeypatch.setattr(opensandbox_provider, 'SANDBOX_ENV', {**SANDBOX_ENV, 'HOME': os.environ['HOME']})
    monkeypatch.setenv('OPENSANDBOX_DISABLE_METRICS', '1')
    save_key(name='OPEN_SANDBOX_DOMAIN', value=DOMAIN)
    save_key(name='OPEN_SANDBOX_API_KEY', value=API_KEY)
    yield server
    await reap(server)


@pytest.mark.skipif(find_spec('opensandbox') is None, reason='opensandbox not installed')
class TestOpenSandbox:
    """OpenSandbox, through the fake server the harness's own OpenSandbox tests use."""

    def test_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('OPEN_SANDBOX_DOMAIN', raising=False)
        provider = opensandbox_provider.OpenSandboxProvider()
        assert provider.missing() == (
            "Add your OpenSandbox server's address in /keys as OPEN_SANDBOX_DOMAIN, such as localhost:8080 or "
            'https://sandbox.example.com (an address without a scheme uses HTTPS unless it is this machine), '
            'and its API key as OPEN_SANDBOX_API_KEY if it needs one.'
        )
        monkeypatch.setenv('OPEN_SANDBOX_DOMAIN', 'https://sandbox.example.com')
        assert provider.missing() is None
        monkeypatch.setattr(opensandbox_provider, 'find_spec', not_installed)
        assert provider.missing() == (
            "OpenSandbox support is not installed. Install CLAI with it: uv tool install 'pydantic-ai-pax[opensandbox]'."
        )

    @pytest.mark.parametrize(
        ('domain', 'scheme'),
        [
            pytest.param('https://opensandbox.invalid', 'https', id='https'),
            pytest.param('http://opensandbox.invalid', 'http', id='http'),
            pytest.param('opensandbox.invalid:8080', 'https', id='no-scheme-remote'),
            pytest.param('localhost:8080', 'http', id='no-scheme-localhost'),
            pytest.param('[::1]:8080', 'http', id='no-scheme-ipv6-loopback'),
        ],
    )
    async def test_the_server_scheme(
        self,
        fake_opensandbox: 'FakeOpenSandbox',
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        domain: str,
        scheme: str,
    ) -> None:
        """Plain HTTP would carry the API key and the GitHub token readable, so only this machine gets it unasked."""
        import httpx

        from tests.harness.opensandbox_sandbox.fake_opensandbox import OwnedTransport

        urls: list[httpx.URL] = []

        class Recording(OwnedTransport):
            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                urls.append(request.url)
                return await super().handle_async_request(request)

        def transport(**_: object) -> Recording:
            return Recording(fake_opensandbox)

        monkeypatch.setattr(httpx, 'AsyncHTTPTransport', transport)
        save_key(name='OPEN_SANDBOX_DOMAIN', value=domain)
        provider = opensandbox_provider.OpenSandboxProvider(working_dir=str(tmp_path.resolve()))
        ref, backend = await provider.create(github=GITHUB)
        assert (await backend.run(['echo', 'ok'])).stdout == 'ok\n'
        await provider.close(backend)
        await provider.delete(ref)
        # The API, the credential vault and the commands alike.
        assert {url.path.rsplit('/', 1)[-1] for url in urls} >= {'sandboxes', 'credential-vault', 'command'}
        assert {url.scheme for url in urls} == {scheme}

    async def test_create_reopen_delete(self, fake_opensandbox: 'FakeOpenSandbox', tmp_path: Path) -> None:
        provider = opensandbox_provider.OpenSandboxProvider(working_dir=str(tmp_path.resolve()))
        ref, backend = await provider.create(github=GITHUB)
        made = fake_opensandbox.sandboxes[ref.id]
        assert ref == backend.ref == WorkspaceRef(provider='opensandbox', id=made.id)
        # The proxy needs a sandbox that reaches only the hosts it allows.
        assert made.network_policy == {
            'defaultAction': 'deny',
            'egress': [{'action': 'allow', 'target': host} for host in GITHUB_HOSTS],
        }
        assert made.credential_proxy == {'enabled': True}
        assert made.credential_vault == {
            'credentials': [
                {'name': 'github-0', 'source': {'value': GITHUB.basic, 'type': 'inline'}},
                {'name': 'github-1', 'source': {'value': GITHUB.bearer, 'type': 'inline'}},
            ],
            'bindings': [
                {
                    'name': host.replace('.', '-'),
                    'match': {'schemes': ['https'], 'hosts': [host]},
                    'auth': {
                        'type': 'customHeaders',
                        'headers': [{'name': 'Authorization', 'credential': credential}],
                    },
                }
                for host, credential in zip(GITHUB_HOSTS, ['github-0', 'github-1', 'github-1'])
            ],
        }
        # The sandbox's `git` and `gh` hold only the stand-in the proxy replaces.
        stand_in = GitHubCredential(login='Ada', token='placeholder')
        assert signed_in(Path(os.environ['HOME'])) == {
            '.config/clai/github-token': ('placeholder', 0o600),
            '.config/gh/hosts.yml': (
                f'github.com:\n    oauth_token: {stand_in.token}\n    user: Ada\n    git_protocol: https\n',
                0o600,
            ),
        }
        reopened = provider.reopen(ref)
        # `request_env`'s keys reach a command as its own `env`.
        assert (await reopened.run(['printenv', 'API_KEY'], env={'API_KEY': 'sk-1'})).stdout == 'sk-1\n'
        await provider.close(reopened)
        await provider.close(backend)
        await provider.delete(ref)
        assert fake_opensandbox.sandboxes == {}
        assert fake_opensandbox.opened_transports == fake_opensandbox.closed_transports

        ref, backend = await provider.create(github=None)
        made = fake_opensandbox.sandboxes[ref.id]
        # Without a login, the server's own network policy applies.
        assert (made.network_policy, made.credential_proxy, made.credential_vault) == (None, None, None)
        await provider.close(backend)
        await provider.delete(ref)

    async def test_a_failed_create_leaves_nothing_behind(
        self, fake_opensandbox: 'FakeOpenSandbox', monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(opensandbox_provider, 'SANDBOX_ENV', {**SANDBOX_ENV, 'HOME': 'relative'})
        provider = opensandbox_provider.OpenSandboxProvider(working_dir=str(tmp_path.resolve()))
        with pytest.raises(RuntimeError, match='no home folder'):
            await provider.create(github=GITHUB)
        assert fake_opensandbox.sandboxes == {}
        assert fake_opensandbox.opened_transports == fake_opensandbox.closed_transports
        fake_opensandbox.refuse = (500, 'DOCKER::DOWN')
        with pytest.raises(Exception, match='DOWN'):
            await provider.create(github=GITHUB)


@pytest.fixture
async def fake_cluster(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> 'AsyncIterator[FakeCluster]':
    """The harness's fake agent-sandbox cluster, whose claims run commands here, with its warm pool in `/keys`."""
    from tests.harness.kubernetes_sandbox.fake_sandboxd import FakeCluster

    cluster = FakeCluster(tmp_path / 'cluster')
    monkeypatch.setattr('pydantic_ai_harness.kubernetes_sandbox._backend.AsyncSandboxClient', cluster.client)
    save_key(name='CLAI_KUBERNETES_WARM_POOL', value='pool')
    try:
        yield cluster
    finally:
        await cluster.close()


@pytest.mark.skipif(
    find_spec('k8s_agent_sandbox') is None or find_spec('grpc') is None, reason='k8s-agent-sandbox not installed'
)
class TestKubernetes:
    """Kubernetes, through the fake cluster the harness's own Kubernetes tests use."""

    def test_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('CLAI_KUBERNETES_WARM_POOL', raising=False)
        monkeypatch.setattr(kubernetes_provider.shutil, 'which', installed)
        provider = kubernetes_provider.KubernetesProvider()
        assert provider.missing() == (
            'Add the agent-sandbox warm pool to claim sandboxes from in /keys as CLAI_KUBERNETES_WARM_POOL, '
            "and its namespace as CLAI_KUBERNETES_NAMESPACE if it is not 'default'. "
            "kubectl's current context names the cluster."
        )
        monkeypatch.setenv('CLAI_KUBERNETES_WARM_POOL', 'pool')
        assert provider.missing() is None
        monkeypatch.setattr(kubernetes_provider.shutil, 'which', not_installed)
        assert provider.missing() == 'Install kubectl: CLAI reaches the sandbox pods through kubectl port-forward.'
        monkeypatch.setattr(kubernetes_provider, 'find_spec', not_installed)
        assert provider.missing() == (
            "Kubernetes support is not installed. Install CLAI with it: uv tool install 'pydantic-ai-pax[kubernetes]'."
        )

    async def test_create_reopen_delete(self, fake_cluster: 'FakeCluster', monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('CLAI_KUBERNETES_NAMESPACE', 'sandboxes')
        provider = kubernetes_provider.KubernetesProvider(working_dir=str(fake_cluster.base / 'checkout'))
        ref, backend = await provider.create(github=GITHUB)
        [(namespace, claim)] = fake_cluster.claims
        assert ref == backend.ref == WorkspaceRef(provider='kubernetes', id=f'{namespace}/{claim}')
        # No time limit: the claim lives until `/sandbox delete`.
        assert fake_cluster.created == [
            ('pool', 'sandboxes', {'shutdown_after_seconds': None, 'sandbox_ready_timeout': 180})
        ]
        await give_github_login(backend, GITHUB)
        assert signed_in(Path(os.environ['HOME'])) == SIGNED_IN
        reopened = provider.reopen(ref)
        # `request_env`'s keys reach a command as its own `env`.
        assert (await reopened.run(['printenv', 'API_KEY'], env={'API_KEY': 'sk-1'})).stdout == 'sk-1\n'
        await provider.close(reopened)
        await provider.close(backend)
        await provider.delete(ref)
        assert fake_cluster.claims == {}
        ref, backend = await provider.create(github=None)
        await provider.close(backend)
        await provider.delete(ref)

    async def test_a_failed_create_leaves_nothing_behind(self, fake_cluster: 'FakeCluster') -> None:
        provider = kubernetes_provider.KubernetesProvider(working_dir=str(fake_cluster.base / 'checkout'))
        fake_cluster.create_error = OSError('the cluster is down')
        with pytest.raises(OSError, match='the cluster is down'):
            await provider.create(github=GITHUB)
        assert fake_cluster.claims == {}
        assert all(client.closed for client in fake_cluster.clients)


LocalEngine = Literal['smolmachines', 'smolvm CLI', 'boxlite']


@dataclass
class LocalHost:
    """A host `/sandbox local` runs on, the engine it runs the VM with, and its VMs' states, by name."""

    engine: LocalEngine
    provider: type[local_provider.SmolProvider] | type[local_provider.BoxLiteProvider]
    harness: str
    """The provider name the harness gives its own VMs' refs, which `/sandbox` must not take for its own."""
    states: Callable[[], dict[str, str]]
    stop: Callable[[str], None]
    """Stop a VM, as it is after this machine restarts."""

    @property
    def words(self) -> list[str]:
        """What the user types after `/sandbox`."""
        return self.provider().name.split()


@pytest.fixture
def fake_smolvm(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> 'FakeSmolvm':
    """The harness's fake `smolvm` CLI, in an absolute `PATH` directory, as its own tests set it up."""
    from tests.harness.smol_sandbox.fake_smolvm import install_fake_smolvm

    return install_fake_smolvm(monkeypatch, tmp_path)


@pytest.fixture(
    params=[
        pytest.param(('linux', 'smolmachines'), id='smol-linux'),
        pytest.param(('darwin', 'smolmachines'), id='smol-macos'),
        pytest.param(('win32', 'smolvm CLI'), id='smol-windows'),
        pytest.param(('linux', 'smolvm CLI'), id='smol-linux-cli'),
        pytest.param(('linux', 'boxlite'), id='boxlite-linux'),
        pytest.param(('darwin', 'boxlite'), id='boxlite-macos'),
    ]
)
def local_host(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_smolvm: 'FakeSmolvm'
) -> Iterator[LocalHost]:
    """A host of each OS each `/sandbox local` engine runs on: smolmachines has no Windows wheel, BoxLite no Windows build.

    smolmachines' and BoxLite's VMs are the harness's fakes, whose commands run here; the `smolvm` CLI is `fake_smolvm`.
    """
    platform, engine = cast('tuple[str, LocalEngine]', request.param)
    monkeypatch.setattr(local_provider.sys, 'platform', platform)
    # The fakes' commands get only the environment they are sent; a real VM has a home of its own.
    monkeypatch.setattr(local_provider, 'SANDBOX_ENV', {**SANDBOX_ENV, 'HOME': os.environ['HOME']})
    if engine == 'boxlite':
        # BoxLite has wheels for macOS and Linux only, up to Python 3.14, so some test jobs lack it.
        if find_spec('boxlite') is None:  # pragma: lax no cover
            pytest.skip('boxlite not installed')
        from tests.harness.boxlite_sandbox.fake_boxlite import FakeBoxLite

        boxes = FakeBoxLite(tmp_path.resolve())
        monkeypatch.setattr('pydantic_ai_harness.boxlite_sandbox._backend.boxlite', boxes.module)
        yield LocalHost(
            engine,
            local_provider.BoxLiteProvider,
            'boxlite',
            lambda: {name: box.status for name, box in boxes.boxes.items()},
            boxes.stop,
        )
        boxes.close()
        return
    if engine == 'smolmachines':
        # smolmachines has wheels for macOS and Linux only, so some test jobs lack it.
        if find_spec('smol') is None:  # pragma: lax no cover
            pytest.skip('smolmachines not installed')
        from tests.harness.smol_sandbox.fake_smol import FakeSmol

        control = FakeSmol(tmp_path.resolve())
        for module in ('_backend', '_capability'):
            monkeypatch.setattr(f'pydantic_ai_harness.smol_sandbox.{module}.smol', control.module)

        def stop(name: str) -> None:
            control.vms[name].state = 'stopped'

        yield LocalHost(
            engine,
            local_provider.SmolProvider,
            'smol',
            lambda: {name: vm.state for name, vm in control.vms.items()},
            stop,
        )
        return
    monkeypatch.setattr(local_provider, 'find_spec', not_installed)
    yield LocalHost(
        engine,
        local_provider.SmolProvider,
        'smol',
        lambda: dict(fake_smolvm.machines),
        lambda name: fake_smolvm.machines.update({name: 'stopped'}),
    )


VM_NAME = r'pydantic-ai-[0-9a-f]{32}'

LOCAL_CHOICE = (
    '/sandbox local runs a new conversation in a VM on this machine. Pick its engine: '
    '/sandbox local smol (SmolVM) or /sandbox local boxlite (BoxLite).'
)


class TestLocal:
    """`/sandbox local smol` and `/sandbox local boxlite`: a VM on this machine, on Linux, macOS, and Windows."""

    async def test_offered_with_a_choice_of_engine(self, checkout: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        smol, boxlite = sandbox.PROVIDERS['local smol'], sandbox.PROVIDERS['local boxlite']
        assert (smol.name, smol.title, smol.token_in_sandbox) == ('local smol', 'SmolVM', True)
        assert (boxlite.name, boxlite.title, boxlite.token_in_sandbox) == ('local boxlite', 'BoxLite', True)
        assert 'local' not in sandbox.PROVIDERS
        plugin = plugin_of(load(checkout))
        [command] = plugin.get_commands()
        assert command.description == (
            'Run a new conversation in a sandbox (local smol, local boxlite, e2b, boxd, daytona, opensandbox, modal, '
            'render, sprites, kubernetes; local ones run a VM on this machine), or show where it runs'
        )
        assert await plugin.command([]) == (
            'This conversation runs on this machine. /sandbox local smol or /sandbox local boxlite or /sandbox e2b or '
            '/sandbox boxd or /sandbox daytona or /sandbox opensandbox or /sandbox modal or /sandbox render or '
            '/sandbox sprites or /sandbox kubernetes moves a new conversation into a sandbox.'
        )
        assert list(plugin.complete(['local', ''])) == ['smol', 'boxlite']
        assert list(plugin.complete(['e2b', ''])) == []
        assert list(plugin.complete(['local', 'smol', ''])) == []
        # `/sandbox local` alone asks which engine, saying what each still needs here, and moves nothing.
        monkeypatch.setattr(local_provider, 'find_spec', installed)
        assert await plugin.command(['local']) == LOCAL_CHOICE
        monkeypatch.setattr(local_provider, 'find_spec', not_installed)
        monkeypatch.setattr(local_provider.sys, 'platform', 'win32')
        monkeypatch.setenv('PATH', '')
        assert await plugin.command(['local']) == '\n'.join(
            [LOCAL_CHOICE, f'SmolVM: {local_provider.SmolProvider().missing()}', 'BoxLite: ' + BOXLITE_ON_WINDOWS]
        )
        assert plugin.host.conversation.messages == []
        with pytest.raises(
            ValueError, match=r'^Usage: /sandbox \[local smol \| local boxlite \| e2b \| .* \| delete\]$'
        ):
            await plugin.command(['local', 'smolvm'])
        with pytest.raises(ValueError, match=r'^Usage: '):
            await plugin.command(['smol'])

    @pytest.mark.parametrize(
        ('platform', 'machine'), [('linux', 'x86_64'), ('darwin', 'arm64'), ('darwin', 'x86_64'), ('win32', 'AMD64')]
    )
    def test_missing(self, platform: str, machine: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr(local_provider.sys, 'platform', platform)
        monkeypatch.setattr(local_provider.platform, 'machine', lambda: machine)
        smol, boxlite = local_provider.SmolProvider(), local_provider.BoxLiteProvider()
        empty = tmp_path / 'empty'
        empty.mkdir()
        monkeypatch.setenv('PATH', str(empty))
        # Neither needs a key.
        monkeypatch.setattr(local_provider, 'find_spec', installed)
        assert smol.missing() is None
        assert boxlite.missing() is None
        monkeypatch.setattr(local_provider, 'find_spec', not_installed)
        cli = tmp_path / 'bin'
        cli.mkdir()
        for name in ('smolvm', 'smolvm.exe'):
            (cli / name).touch(mode=0o755)
        monkeypatch.setenv('PATH', f'{empty}{os.pathsep}{cli}')
        assert smol.missing() is None
        # Never one in the current directory, such as a checkout's, nor on a relative `PATH` entry.
        monkeypatch.chdir(tmp_path / 'bin')
        monkeypatch.setenv('PATH', f'.{os.pathsep}{empty}')
        on_intel_mac = machine == 'x86_64' and platform == 'darwin'
        assert smol.missing() == (
            'Install SmolVM: unzip the windows-x86_64 release from https://github.com/smol-machines/smolvm/releases '
            'into a folder on PATH. It needs the Windows Hypervisor Platform. '
            '/sandbox local smol on Windows has not been tested on a Windows host yet.'
            if platform == 'win32'
            else 'SmolVM runs only on Macs with Apple silicon.'
            if on_intel_mac
            else "SmolVM support is not installed. Install CLAI with it: uv tool install 'pydantic-ai-pax[local]'."
        )
        assert boxlite.missing() == (
            BOXLITE_ON_WINDOWS
            if platform == 'win32'
            else 'BoxLite runs only on Macs with Apple silicon.'
            if on_intel_mac
            else "BoxLite support is not installed. Install CLAI with it: uv tool install 'pydantic-ai-pax[local]'. "
            'It needs Python 3.14 or older.'
        )

    async def test_boxlite_deletes_nothing_without_boxlite(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(local_provider.sys, 'platform', 'win32')
        monkeypatch.setattr(local_provider, 'find_spec', not_installed)
        with pytest.raises(WorkspaceUnavailableError, match='BoxLite does not run on Windows'):
            await local_provider.BoxLiteProvider().delete(WorkspaceRef(provider='local-boxlite', id='x'))

    async def test_create_reopen_delete(self, local_host: LocalHost, fake_smolvm: 'FakeSmolvm', tmp_path: Path) -> None:
        provider = local_host.provider(working_dir=str(tmp_path.resolve() / 'workspace'))
        assert provider.missing() is None
        ref, backend = await provider.create(github=GITHUB)
        # Not `local`, which names this machine's own folder, nor the harness's, which its capabilities make too.
        assert ref.provider == f'local-{local_host.words[1]}'
        assert re.fullmatch(VM_NAME, ref.id)
        assert (tmp_path / 'workspace').is_dir()
        if local_host.engine == 'smolvm CLI':
            # The VM pulls an image with bash and git, which the copy of the checkout needs, and reaches GitHub.
            assert fake_smolvm.created == [['--name', ref.id, '--net', '--image', local_provider.IMAGE]]
        else:
            assert fake_smolvm.created == []
        assert (await backend.run(['printenv', 'GIT_TERMINAL_PROMPT'])).stdout == '0\n'
        await give_github_login(backend, GITHUB)
        assert signed_in(Path(os.environ['HOME'])) == SIGNED_IN
        reopened = provider.reopen(ref)
        # `request_env`'s keys reach a command as its own `env`.
        assert (await reopened.run(['printenv', 'API_KEY'], env={'API_KEY': 'sk-1'})).stdout == 'sk-1\n'
        await provider.close(reopened)
        await provider.close(backend)
        await provider.delete(ref)
        await provider.delete(ref)  # One already gone is fine.
        assert local_host.states() == {}
        with pytest.raises(WorkspaceUnavailableError, match='no longer exists'):
            await provider.reopen(ref).run(['true'])
        # Closing a deleted VM's workspace, as `/sandbox delete` does, has nothing left to write back.
        await provider.close(reopened)

    async def test_closing_boots_no_vm(self, local_host: LocalHost, tmp_path: Path) -> None:
        provider = local_host.provider(working_dir=str(tmp_path.resolve() / 'workspace'))
        ref, backend = await provider.create(github=None)
        await provider.close(backend)
        # As after CLAI restarts: a conversation reopened and closed without a command leaves its VM stopped.
        local_host.stop(ref.id)
        await provider.close(provider.reopen(ref))
        assert local_host.states() == {ref.id: 'stopped'}
        await provider.delete(ref)
        # Nor one whose operations could not reach the VM, deleted since.
        gone = provider.reopen(ref)
        with pytest.raises(WorkspaceUnavailableError, match='no longer exists'):
            await gone.working_dir()
        with pytest.raises(WorkspaceUnavailableError, match='no longer exists'):
            await gone.run(['true'])
        await provider.close(gone)
        assert local_host.states() == {}

    async def test_a_conversation_runs_in_the_vm(
        self, local_host: LocalHost, checkout: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        git(checkout, 'remote', 'remove', 'origin')
        Path(os.environ['HOME']).mkdir()
        provider = local_host.provider(working_dir=str(tmp_path.resolve() / 'workspace'))
        monkeypatch.setattr(sandbox, 'PROVIDERS', {provider.name: provider})
        monkeypatch.setattr(sandbox, 'gh_token', not_installed)
        loaded = load(checkout)
        session = on_screen(loaded, checkout, shell=True)
        message = await plugin_of(loaded).command(local_host.words)
        ref = history_sandbox(session.messages)
        assert ref is not None and ref.provider == f'local-{local_host.words[1]}'
        assert message.startswith(f'This conversation now runs in {provider.title} sandbox {ref.id}. ')
        copy = tmp_path / 'workspace'
        assert git(copy, 'rev-parse', 'HEAD') == git(checkout, 'rev-parse', 'HEAD')
        result = await session.prompt('where?')
        assert result.response.workspace_ref == ref
        assert plugin_of(loaded).get_status_segments()[0]() == f'{provider.name} sandbox'
        # A VM of the user's own `SmolSandbox` or `BoxLiteSandbox` is not a `/sandbox` sandbox.
        assert history_sandbox(worked_in(WorkspaceRef(provider=local_host.harness, id=ref.id))) is None
        assert await plugin_of(loaded).command(['delete']) == (
            f'Deleted {provider.title} sandbox {ref.id}. Run /new to start another conversation.'
        )


BOXLITE_ON_WINDOWS = 'BoxLite does not run on Windows: use /sandbox local smol.'


@pytest.fixture
def windows(monkeypatch: pytest.MonkeyPatch, fake_smolvm: 'FakeSmolvm') -> 'FakeSmolvm':
    """A Windows host, where `/sandbox local smol` drives the `smolvm` CLI."""
    monkeypatch.setattr(local_provider.sys, 'platform', 'win32')
    monkeypatch.setattr(local_provider, 'find_spec', not_installed)
    monkeypatch.setattr(local_provider, 'SANDBOX_ENV', {**SANDBOX_ENV, 'HOME': os.environ['HOME']})
    return fake_smolvm


class TestLocalCli:
    """What `/sandbox local smol` adds to the harness's `SmolvmCliBackend`, where smolmachines has no wheel, as on Windows."""

    async def test_close_stops_the_vm(self, windows: 'FakeSmolvm', tmp_path: Path) -> None:
        provider = local_provider.SmolProvider(working_dir=str(tmp_path.resolve() / 'workspace'))
        ref, backend = await provider.create(github=GITHUB)
        await give_github_login(backend, GITHUB)
        # The token reached the VM on a command's stdin, never on a command line of this machine.
        token = base64.b64encode(b'gho_x').decode()
        assert any(token in script for script in windows.scripts)
        assert not any(token in argument for command in windows.commands for argument in command)
        # A workspace that never reached the VM stops nothing.
        await provider.close(provider.reopen(ref))
        [name] = windows.machines
        assert windows.machines[name] == 'running'
        # The VM would otherwise outlive CLAI; its next command starts it again.
        await provider.close(backend)
        assert windows.machines[name] == 'stopped'
        assert (await provider.reopen(ref).run(['true'])).exit_code == 0
        await provider.delete(ref)
        assert windows.machines == {}

    async def test_a_failed_stop_does_not_stop_closing(
        self, windows: 'FakeSmolvm', tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        provider = local_provider.SmolProvider(working_dir=str(tmp_path.resolve() / 'workspace'))
        (first, first_backend), (second, second_backend) = [await provider.create(github=None) for _ in range(2)]
        windows.failures['stop'] = ('SMOLVM_ERROR', 'stuck')
        # As when a session ends: one VM that will not stop leaves the next to close, and says so.
        await provider.close(first_backend)
        assert 'Could not stop or sync SmolVM sandbox' in caplog.text
        del windows.failures['stop']
        await provider.close(second_backend)
        assert windows.machines == {first.id: 'running', second.id: 'stopped'}
        await provider.delete(first)
        await provider.delete(second)

    async def test_without_smolvm(self, windows: 'FakeSmolvm', tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Never one in the current directory, such as a checkout's, nor on a relative `PATH` entry: Windows would run it.
        monkeypatch.chdir(tmp_path)
        (tmp_path / 'smolvm.exe').touch(mode=0o755)
        monkeypatch.setenv('PATH', f'.{os.pathsep}{tmp_path / "nowhere"}')
        provider = local_provider.SmolProvider(working_dir=str(tmp_path.resolve() / 'workspace'))
        assert provider.missing() == (
            'Install SmolVM: unzip the windows-x86_64 release from https://github.com/smol-machines/smolvm/releases '
            'into a folder on PATH. It needs the Windows Hypervisor Platform. '
            '/sandbox local smol on Windows has not been tested on a Windows host yet.'
        )
        with pytest.raises(WorkspaceUnavailableError, match='smolvm was not found on PATH'):
            await provider.create(github=None)
        assert windows.commands == []

    async def test_a_failed_create_leaves_nothing_behind(self, windows: 'FakeSmolvm', tmp_path: Path) -> None:
        provider = local_provider.SmolProvider(working_dir=str(tmp_path.resolve() / 'workspace'))
        windows.failures['start'] = ('HYPERVISOR_UNAVAILABLE', 'the Windows Hypervisor Platform is not enabled')
        with pytest.raises(WorkspaceUnavailableError, match=r'cannot run a local smol VM \(HYPERVISOR_UNAVAILABLE\)'):
            await provider.create(github=GITHUB)
        # smolvm made the machine before it failed to start it.
        assert len(windows.created) == 1
        assert windows.machines == {}

    def test_only_its_own_vms(self, windows: 'FakeSmolvm') -> None:
        # A conversation's history could otherwise open any VM of this user, such as one with this machine's folders.
        with pytest.raises(ValueError, match='only uses VMs it created'):
            local_provider.SmolProvider().reopen(WorkspaceRef(provider='local-smol', id='dev'))


class TestCompaction:
    """Compacting a sandboxed conversation must not move it back to this machine."""

    @pytest.mark.parametrize('strategy', ['summarization', 'truncation'])
    @pytest.mark.parametrize('trailing', ['! command', 'cancelled turn', 'none'])
    async def test_compact_keeps_the_sandbox(
        self, checkout: Path, provider: FakeProvider, strategy: str, trailing: str
    ) -> None:
        from pydantic_ai_pax.builtin_plugins.compaction import CompactionPlugin

        working_dirs: list[str] = []

        class Probe(AbstractCapability[None]):
            async def before_run(self, ctx: RunContext[None]) -> None:
                working_dirs.append(await ctx.workspace.working_dir())

        answers = iter(['first answer ' * 200, 'hang', 'after compaction'])
        waiting = anyio.Event()

        async def respond(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
            answer = next(answers)
            if answer == 'hang':
                waiting.set()
                await anyio.sleep_forever()
            yield answer

        loaded = load(checkout)
        agent = Agent(FunctionModel(stream_function=respond), deps_type=type(None))
        session = Session(agent, deps=None, plugins=[*loaded.capabilities, Probe()], workspace=checkout)
        loaded.host.conversation = session
        await plugin_of(loaded).command(['fake'])
        await session.prompt('a long question ' * 200)
        if trailing == '! command':
            # What `!cat log` saves: a request with no response after it.
            await session.commit_messages(
                [*session.messages, ModelRequest(parts=[UserPromptPart('ran cat log: ' + 'a log line\n' * 2000)])]
            )
            next(answers)
        elif trailing == 'cancelled turn':
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(session.prompt, 'please read this log: ' + 'a log line\n' * 2000)
                await waiting.wait()
                tasks.cancel_scope.cancel()
        else:
            next(answers)
        compaction = load_plugin(
            CompactionPlugin,
            PluginHost[None](
                name='compaction',
                console=Console(file=io.StringIO()),
                settings={'strategy': strategy, 'summarization_model': 'test'},
                conversation=session,
            ),
        )
        notice = await compaction.commands.execute_async('/compact')
        assert notice is not None and notice.startswith('Compacted'), notice
        ref = WorkspaceRef(provider='fake', id='sbx-1')
        # A CLAI restarted on the saved history finds the sandbox the same way.
        restarted = plugin_of(load(checkout, Transcript(messages=session.messages))).current()

        result = await session.prompt('and now?')
        assert (history_sandbox(session.messages), restarted is not None, working_dirs[-1]) == (
            ref,
            True,
            str(provider.folder(ref)),
        )
        assert result.response.workspace_ref == ref

    @pytest.mark.parametrize('ending', ['failed', 'cancelled'])
    async def test_a_turn_ending_unanswered_after_compacting_keeps_the_sandbox(
        self, checkout: Path, provider: FakeProvider, ending: str
    ) -> None:
        from pydantic_ai_harness.compaction import SlidingWindowCompaction
        from pydantic_ai_harness.step_persistence.conversations import SqliteConversationStore

        working_dirs: list[str] = []

        class Probe(AbstractCapability[None]):
            async def before_run(self, ctx: RunContext[None]) -> None:
                working_dirs.append(await ctx.workspace.working_dir())

        waiting = anyio.Event()
        answers = iter(['first answer', 'unanswered', 'third answer'])
        sent: list[int] = []

        async def respond(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
            sent.append(len(messages))
            if next(answers) == 'unanswered':
                if ending == 'failed':
                    raise RuntimeError('the provider is down')
                waiting.set()
                await anyio.sleep_forever()
            yield 'ok'

        loaded = load(checkout)
        agent = Agent(FunctionModel(stream_function=respond), deps_type=type(None))
        # Compacts the second turn's request down to its prompt alone, before the model is called.
        compaction = SlidingWindowCompaction[None](max_messages=4, keep_messages=1)
        session = Session(
            agent,
            deps=None,
            plugins=[*loaded.capabilities, compaction, Probe()],
            workspace=checkout,
            conversations=SqliteConversationStore(database=checkout.parent / 'sessions.db'),
        )
        loaded.host.conversation = session
        await plugin_of(loaded).command(['fake'])
        await session.prompt('first question')
        if ending == 'failed':
            with pytest.raises(RuntimeError):
                await session.prompt('second question')
        else:
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(session.prompt, 'second question')
                await waiting.wait()
                tasks.cancel_scope.cancel()

        await session.prompt('third question')
        ref = WorkspaceRef(provider='fake', id='sbx-1')
        assert sent[1] == 1
        assert working_dirs[-1] == str(provider.folder(ref))
