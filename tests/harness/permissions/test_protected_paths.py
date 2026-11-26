"""Tests for `Permissions.protected_paths` and `Permissions.protected_commands`."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from pydantic_ai import Agent, DeferredToolRequests
from pydantic_ai.capabilities import AbstractCapability, HandleDeferredToolCalls, LocalWorkspace
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.workspaces import Workspace, WorkspaceRef, WorkspaceUnavailableError
from pydantic_ai_harness.approval_reviewer import ApprovalReviewer
from pydantic_ai_harness.file_system import FileSystem
from pydantic_ai_harness.permissions import (
    PermissionDecision,
    PermissionMode,
    PermissionRequest,
    PermissionRule,
    Permissions,
    approval_ask,
    permission_requests,
)
from pydantic_ai_harness.permissions._protected import command_locations, matches, write_locations
from pydantic_ai_harness.shell import Shell

PROTECTED = ('.pax', '.pax/*', '.git/hooks/*')


def _calls(*calls: tuple[str, Mapping[str, object]]) -> FunctionModel:
    """A model that makes `calls` one per step, then answers with the last tool result it got."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        step = sum(isinstance(message, ModelResponse) for message in messages)
        if step < len(calls):
            name, args = calls[step]
            return ModelResponse(parts=[ToolCallPart(name, dict(args), tool_call_id=f'call{step}')])
        last = messages[-1]
        assert isinstance(last, ModelRequest)
        returned = [part for part in last.parts if isinstance(part, ToolReturnPart)]
        return ModelResponse(parts=[TextPart(str(returned[0].content) if returned else 'done')])

    return FunctionModel(respond)


class _Person:
    """Denies every request, and keeps what it was asked."""

    def __init__(self) -> None:
        self.asked: list[PermissionRequest] = []

    async def __call__(self, request: PermissionRequest) -> PermissionDecision:
        self.asked.append(request)
        return PermissionDecision(approved=False)


def _safe_reviewer() -> FunctionModel:
    """A reviewer model that says every call is safe, so it approves whatever it reviews."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {'response': 'no'})])

    return FunctionModel(respond)


async def _run(
    tmp_path: Path,
    *calls: tuple[str, Mapping[str, object]],
    mode: PermissionMode = 'auto',
    rules: Sequence[PermissionRule[None]] = (),
    protected_commands: Sequence[str] = (),
    extra: Sequence[AbstractCapability[None]] = (),
) -> _Person:
    person = _Person()
    agent = Agent[None, 'str | DeferredToolRequests'](
        _calls(*calls),
        deps_type=type(None),
        output_type=[str, DeferredToolRequests],
        capabilities=[
            LocalWorkspace(tmp_path),
            FileSystem(),
            Shell(denied_commands=[]),
            Permissions(mode, rules=rules, protected_paths=PROTECTED, protected_commands=protected_commands),
            *extra,
            HandleDeferredToolCalls[None](approval_ask(person)),
        ],
    )
    await agent.run('go')
    return person


class TestProtectedPaths:
    @pytest.mark.parametrize('path', ['.pax/settings.json', 'sub/../.pax/x', '.git/hooks/pre-commit', '.pax'])
    async def test_auto_asks_before_a_write_to_a_protected_path(self, tmp_path: Path, path: str) -> None:
        tool = 'create_directory' if path == '.pax' else 'write_file'
        person = await _run(
            tmp_path, (tool, {'path': path, 'content': 'x'} if tool == 'write_file' else {'path': path})
        )
        [asked] = person.asked
        assert asked.protected and asked.action == 'write' and asked.subject == path
        assert asked.reason == 'This file write changes a protected path, so only a person can approve it.'
        assert not (tmp_path / path).exists()

    async def test_a_link_into_a_protected_path_asks(self, tmp_path: Path) -> None:
        (tmp_path / '.pax').mkdir()
        (tmp_path / 'link').symlink_to('.pax')
        person = await _run(tmp_path, ('write_file', {'path': 'link/settings.json', 'content': 'x'}))
        assert [asked.subject for asked in person.asked] == ['link/settings.json']

    async def test_other_writes_and_reads_run(self, tmp_path: Path) -> None:
        (tmp_path / '.pax').mkdir()
        (tmp_path / '.pax' / 'settings.json').write_text('{}')
        person = await _run(
            tmp_path,
            ('write_file', {'path': 'notes.pax', 'content': 'x'}),
            ('read_file', {'path': '.pax/settings.json'}),
            ('run_command', {'command': 'cat .pax/settings.json && ls .git/hooks'}),
        )
        assert person.asked == []
        assert (tmp_path / 'notes.pax').read_text() == 'x'

    async def test_full_lets_a_protected_write_run(self, tmp_path: Path) -> None:
        (tmp_path / '.pax').mkdir()
        person = await _run(tmp_path, ('write_file', {'path': '.pax/settings.json', 'content': 'x'}), mode='full')
        assert person.asked == []
        assert (tmp_path / '.pax' / 'settings.json').read_text() == 'x'

    async def test_an_allow_rule_does_not_let_a_protected_write_run(self, tmp_path: Path) -> None:
        person = await _run(
            tmp_path,
            ('write_file', {'path': '.pax/settings.json', 'content': 'x'}),
            mode='read-only',
            rules=[PermissionRule('allow', tools=['write_file'])],
        )
        assert [asked.protected for asked in person.asked] == [True]

    async def test_a_deny_rule_still_refuses_a_protected_write(self, tmp_path: Path) -> None:
        (tmp_path / '.pax').mkdir()
        person = await _run(
            tmp_path,
            ('write_file', {'path': '.pax/settings.json', 'content': 'x'}),
            rules=[PermissionRule('deny', tools=['write_file'])],
        )
        assert person.asked == []
        assert not (tmp_path / '.pax' / 'settings.json').exists()

    @pytest.mark.parametrize(
        'command',
        [
            'echo x > .pax/settings.json',
            'cat a >> .pax/settings.json',
            'cp hook .git/hooks/pre-commit',
            'rm .pax',
            'sed -i s/a/b/ "./.pax/settings.json"',
            'tool --out=.pax/settings.json',
        ],
    )
    async def test_auto_asks_before_a_command_naming_a_protected_path(self, tmp_path: Path, command: str) -> None:
        person = await _run(tmp_path, ('run_command', {'command': command}))
        [asked] = person.asked
        assert asked.protected and asked.action == 'command' and asked.subject == command
        assert asked.reason == 'This command may change a protected path, so only a person can approve it.'
        assert not (tmp_path / '.pax').exists()

    async def test_a_protected_command_asks(self, tmp_path: Path) -> None:
        commands = ['uv run settings set model x', 'echo $PATH', 'settings get model']
        person = await _run(
            tmp_path, *(('run_command', {'command': c}) for c in commands), protected_commands=['settings set']
        )
        assert [(asked.subject, asked.protected) for asked in person.asked] == [(commands[0], True)]

    async def test_a_reviewer_never_approves_a_protected_call(self, tmp_path: Path) -> None:
        """A reviewer that says every call is safe approves the `git push`, but the protected write goes to the person."""
        reviewer = ApprovalReviewer[None](_safe_reviewer(), question='Could this call do harm?')
        (tmp_path / '.pax').mkdir()
        person = await _run(
            tmp_path,
            ('write_file', {'path': '.pax/settings.json', 'content': 'x'}),
            ('run_command', {'command': 'git push --dry-run nowhere || true'}),
            extra=[reviewer],
        )
        assert [asked.subject for asked in person.asked] == ['.pax/settings.json']
        assert not (tmp_path / '.pax' / 'settings.json').exists()

    async def test_without_a_handler_the_request_says_it_is_protected(self, tmp_path: Path) -> None:
        agent = Agent[None, 'str | DeferredToolRequests'](
            _calls(('write_file', {'path': '.pax/settings.json', 'content': 'x'})),
            deps_type=type(None),
            output_type=[str, DeferredToolRequests],
            capabilities=[LocalWorkspace(tmp_path), FileSystem(), Permissions('auto', protected_paths=PROTECTED)],
        )
        result = await agent.run('go')
        assert isinstance(result.output, DeferredToolRequests)
        assert [request.protected for request in permission_requests(result.output)] == [True]

    def test_from_spec_takes_the_protected_fields(self) -> None:
        permissions = Permissions[None].from_spec('auto', protected_paths=['.pax/*'], protected_commands=['pax config'])
        assert (permissions.protected_paths, permissions.protected_commands) == (['.pax/*'], ['pax config'])


class TestMatching:
    def test_a_relative_glob_matches_the_end_of_a_path(self) -> None:
        assert matches('/repo/.pax/settings.json', ['.pax/*'], posix=True)
        assert not matches('/repo/notes.pax', ['.pax', '.pax/*'], posix=True)
        assert matches('/home/me/.config/pax/config.db', ['/home/me/.config/pax/*'], posix=True)

    def test_windows_paths_ignore_case_and_separators(self) -> None:
        assert matches('C:\\Repo\\.PAX\\Settings.json', ['.pax/*'], posix=False)
        assert matches('C:\\Users\\me\\pax\\x', ['c:\\users\\ME\\pax\\*'], posix=False)

    def test_a_very_long_path_counts_as_protected(self) -> None:
        assert matches('/' + 'x' * 5000, ['.pax/*'], posix=True)

    async def test_without_a_workspace_paths_are_only_normalized(self) -> None:
        workspace = Workspace(_NoWorkspace())
        assert await write_locations(workspace, 'a/../.pax/x') == ['.pax/x']
        assert await command_locations(workspace, 'echo x > a/../.pax/x', posix=True) == ['echo', 'x', '.pax/x']

    async def test_a_link_that_cannot_be_followed_counts_as_written(self, tmp_path: Path) -> None:
        workspace = Workspace(_NoRealpath(tmp_path))
        assert await write_locations(workspace, '.pax/x') == [str(tmp_path / '.pax' / 'x')]

    async def test_home_and_variables_are_expanded(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('HOME', '/home/me')
        monkeypatch.setenv('PAX_DIR', '/home/me/pax')
        workspace = Workspace(_NoWorkspace())
        assert await command_locations(workspace, 'touch ~/.pax/x $PAX_DIR/y', posix=True) == [
            'touch',
            '/home/me/.pax/x',
            '/home/me/pax/y',
        ]

    async def test_a_command_that_only_reads_names_no_path(self) -> None:
        workspace = Workspace(_NoWorkspace())
        assert await command_locations(workspace, 'cat .pax/x | grep a', posix=True) == []
        assert await command_locations(workspace, 'cat .pax/x > out', posix=True) != []
        assert await command_locations(workspace, 'cat .pax/x', posix=False) != []
        assert await command_locations(workspace, 'cat $X .pax/x', posix=True) != []
        assert await command_locations(workspace, 'cat ' + 'x' * 70_000, posix=True) != []


class _NoWorkspace:
    """A backend with no working directory, as a run without a workspace has."""

    ref: WorkspaceRef | None = None

    async def working_dir(self) -> str:
        raise WorkspaceUnavailableError('no workspace')


class _NoRealpath:
    """A local backend whose links cannot be followed."""

    ref: WorkspaceRef | None = None

    def __init__(self, root: Path) -> None:
        self.root = root

    async def working_dir(self) -> str:
        return str(self.root)

    async def realpath(self, path: str) -> str:
        raise OSError('cannot follow')
