"""Tests for `Permissions.protected_paths` and `Permissions.protected_commands`."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults, RunContext, Tool
from pydantic_ai.capabilities import AbstractCapability, HandleDeferredToolCalls, LocalWorkspace
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
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
    RememberedApproval,
    approval_ask,
    ask_permissions,
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
        assert asked.person_only and asked.action == 'write' and asked.subject == path
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
        assert [asked.person_only for asked in person.asked] == [True]

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
        assert asked.person_only and asked.action == 'command' and asked.subject == command
        assert asked.reason == 'This command may change a protected path, so only a person can approve it.'
        assert not (tmp_path / '.pax').exists()

    async def test_a_protected_command_asks(self, tmp_path: Path) -> None:
        commands = ['uv run settings set model x', 'echo $PATH', 'settings get model']
        person = await _run(
            tmp_path, *(('run_command', {'command': c}) for c in commands), protected_commands=['settings set']
        )
        assert [(asked.subject, asked.person_only) for asked in person.asked] == [(commands[0], True)]

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
        assert [request.person_only for request in permission_requests(result.output)] == [True]

    def test_from_spec_takes_the_protected_fields(self) -> None:
        permissions = Permissions[None].from_spec('auto', protected_paths=['.pax/*'], protected_commands=['pax config'])
        assert (permissions.protected_paths, permissions.protected_commands) == (['.pax/*'], ['pax config'])


def _approval_required_shell(ran: list[str]) -> Tool[None]:
    """A shell tool registered with `requires_approval=True`, so core defers it before any `before_tool_execute` hook."""

    def run_command(command: str) -> str:
        ran.append(command)
        return 'ran'

    return Tool(
        run_command,
        requires_approval=True,
        metadata={'code_arg_language': 'shell', 'code_arg_name': 'command'},
    )


@dataclass
class _KeepingResults(AbstractCapability[None]):
    """Hands deferred calls to `reviewer`, and keeps what it answered."""

    reviewer: ApprovalReviewer[None]
    results: list[DeferredToolResults | None] = field(default_factory=list[DeferredToolResults | None])

    async def handle_deferred_tool_calls(
        self, ctx: RunContext[None], *, requests: DeferredToolRequests
    ) -> DeferredToolResults | None:
        result = await self.reviewer.handle_deferred_tool_calls(ctx, requests=requests)
        self.results.append(result)
        return result


class TestProtectedCallsDeferredBeforePermissions:
    """A protected call deferred by `requires_approval=True` is still for a person only."""

    async def test_a_reviewer_leaves_it_to_the_person(self, tmp_path: Path) -> None:
        ran: list[str] = []
        person = _Person()
        agent = Agent[None, str](
            _calls(('run_command', {'command': 'echo x > .pax/settings.json'})),
            deps_type=type(None),
            tools=[_approval_required_shell(ran)],
            capabilities=[
                LocalWorkspace(tmp_path),
                Permissions('auto', protected_paths=PROTECTED),
                ApprovalReviewer[None](_safe_reviewer(), question='Could this call do harm?'),
                HandleDeferredToolCalls[None](approval_ask(person)),
            ],
        )
        await agent.run('go')
        assert [(asked.subject, asked.person_only) for asked in person.asked] == [('echo x > .pax/settings.json', True)]
        assert ran == []

    async def test_an_unprotected_call_still_goes_to_the_reviewer(self, tmp_path: Path) -> None:
        ran: list[str] = []
        person = _Person()
        agent = Agent[None, str](
            _calls(('run_command', {'command': 'echo x > notes.md'})),
            deps_type=type(None),
            tools=[_approval_required_shell(ran)],
            capabilities=[
                LocalWorkspace(tmp_path),
                Permissions('auto', protected_paths=PROTECTED),
                ApprovalReviewer[None](_safe_reviewer(), question='Could this call do harm?'),
                HandleDeferredToolCalls[None](approval_ask(person)),
            ],
        )
        await agent.run('go')
        assert person.asked == []
        assert ran == ['echo x > notes.md']

    async def test_a_reviewer_approval_on_resume_still_asks_a_person(self, tmp_path: Path) -> None:
        """Results a reviewer gave for the call where nothing marked it protected, such as before the path was."""
        command = 'echo x > .pax/settings.json'
        reviewer = _KeepingResults(ApprovalReviewer[None](_safe_reviewer(), question='Could this call do harm?'))
        unprotected = Agent[None, str](
            _calls(('run_command', {'command': command})),
            deps_type=type(None),
            tools=[_approval_required_shell([])],
            capabilities=[reviewer],
        )
        await unprotected.run('go')
        [approvals] = reviewer.results
        assert approvals is not None
        assert list(approvals.approvals) == ['call0']

        ran: list[str] = []
        protected = Agent[None, 'str | DeferredToolRequests'](
            _calls(('run_command', {'command': command})),
            deps_type=type(None),
            output_type=[str, DeferredToolRequests],
            tools=[_approval_required_shell(ran)],
            capabilities=[LocalWorkspace(tmp_path), Permissions('auto', protected_paths=PROTECTED)],
        )
        held = await protected.run('go')
        resumed = await protected.run(message_history=held.all_messages(), deferred_tool_results=approvals)
        assert isinstance(resumed.output, DeferredToolRequests)
        assert [(r.subject, r.person_only) for r in permission_requests(resumed.output)] == [(command, True)]
        assert ran == []

    async def test_a_person_approval_on_resume_runs_it(self, tmp_path: Path) -> None:
        ran: list[str] = []
        command = 'echo x > .pax/settings.json'
        agent = Agent[None, 'str | DeferredToolRequests'](
            _calls(('run_command', {'command': command})),
            deps_type=type(None),
            output_type=[str, DeferredToolRequests],
            tools=[_approval_required_shell(ran)],
            capabilities=[LocalWorkspace(tmp_path), Permissions('auto', protected_paths=PROTECTED)],
        )
        held = await agent.run('go')
        assert isinstance(held.output, DeferredToolRequests)
        assert [r.person_only for r in permission_requests(held.output)] == [True]
        resumed = await agent.run(
            message_history=held.all_messages(), deferred_tool_results=DeferredToolResults(approvals={'call0': True})
        )
        assert resumed.output == 'ran'
        assert ran == [command]


class _DBOSDurability(AbstractCapability[None]):
    in_durable_context = True


_DBOSDurability.__module__ = 'pydantic_ai.durable_exec.dbos'


class _Remembering:
    """Approves every request for the rest of the conversation, keeping what it was asked."""

    def __init__(self) -> None:
        self.asked: list[PermissionRequest] = []

    async def __call__(self, request: PermissionRequest) -> PermissionDecision:
        self.asked.append(request)
        return PermissionDecision(approved=True, remember=True)


class TestProtectedApprovalsAreNotRemembered:
    async def test_each_protected_write_asks_again(self, tmp_path: Path) -> None:
        (tmp_path / '.pax').mkdir()
        write = ('write_file', {'path': '.pax/settings.json', 'content': 'x'})
        other = ('write_file', {'path': 'notes.md', 'content': 'x'})
        person = _Remembering()
        remembered: set[RememberedApproval] = set()
        agent = Agent[None, str](
            _calls(write, write, other, other),
            deps_type=type(None),
            capabilities=[
                LocalWorkspace(tmp_path),
                FileSystem(),
                Permissions('read-only', protected_paths=PROTECTED, remembered=remembered),
                HandleDeferredToolCalls[None](approval_ask(person)),
            ],
        )
        await agent.run('go', conversation_id='c')
        assert [asked.subject for asked in person.asked] == ['.pax/settings.json', '.pax/settings.json', 'notes.md']
        assert remembered == {RememberedApproval('c', 'write_file', 'notes.md')}

    async def test_an_approval_remembered_before_does_not_cover_a_protected_write(self, tmp_path: Path) -> None:
        """Such as one remembered while the path was not protected, in a `remembered` set the caller keeps."""
        (tmp_path / '.pax').mkdir()
        person = _Remembering()
        remembered = {RememberedApproval('c', 'write_file', '.pax/settings.json')}
        agent = Agent[None, str](
            _calls(('write_file', {'path': '.pax/settings.json', 'content': 'x'})),
            deps_type=type(None),
            capabilities=[
                LocalWorkspace(tmp_path),
                FileSystem(),
                Permissions('auto', protected_paths=PROTECTED, remembered=remembered),
                HandleDeferredToolCalls[None](approval_ask(person)),
            ],
        )
        await agent.run('go', conversation_id='c')
        assert [asked.subject for asked in person.asked] == ['.pax/settings.json']

    async def test_inside_durable_execution_no_approval_is_kept_in_history(self) -> None:
        def run_command(command: str) -> str:
            return 'ran'

        command = ('run_command', {'command': 'echo x > .pax/settings.json'})
        person = _Remembering()
        agent = Agent[None, str](
            _calls(command, command),
            deps_type=type(None),
            tools=[Tool(run_command, metadata={'code_arg_language': 'shell', 'code_arg_name': 'command'})],
            capabilities=[
                _DBOSDurability(),
                Permissions('auto', protected_paths=PROTECTED),
                HandleDeferredToolCalls[None](handler=lambda ctx, requests: ask_permissions(requests, person)),
            ],
        )
        result = await agent.run('go', conversation_id='c')
        assert len(person.asked) == 2
        returns = [
            part for message in result.all_messages() for part in message.parts if isinstance(part, ToolReturnPart)
        ]
        assert [part.metadata for part in returns] == [None, None]


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


class TestCommandWords:
    """A protected path a command names is seen however the shell spells it."""

    @pytest.mark.parametrize(
        'command',
        [
            'echo x > "/home/u/my dir/.pax/settings.json"',
            "echo x > '/home/u/my dir/.pax/settings.json'",
            'echo x > /home/u/my\\ dir/.pax/settings.json',
            'echo x >"/home/u/my dir/.pax/settings.json";true',
            'tool --out="/home/u/my dir/.pax/settings.json"',
            'echo `cat a`>/home/u/my\\ dir/.pax/settings.json',
            'echo "x > /home/u/my dir/.pax/settings.json',
        ],
        ids=['double-quotes', 'single-quotes', 'escaped-space', 'operators', 'option-value', 'backticks', 'unbalanced'],
    )
    async def test_posix_spellings(self, command: str) -> None:
        assert await _names_protected(Workspace(_NoWorkspace()), command, posix=True)

    @pytest.mark.parametrize(
        'command',
        [
            'Set-Content .pax\\settings.json x',
            'Set-Content C:\\Users\\u\\.pax\\settings.json x',
            'echo x > "C:\\Users\\Alice Smith\\.pax\\settings.json"',
            "Set-Content 'C:\\Users\\Alice Smith\\.PAX\\Settings.json' x",
            'copy hook C:\\repo\\.git\\hooks\\pre-commit',
        ],
        ids=['relative', 'absolute', 'quoted-space', 'single-quoted-case', 'hooks'],
    )
    async def test_windows_spellings(self, command: str) -> None:
        assert await _names_protected(Workspace(_WindowsWorkspace()), command, posix=False)

    async def test_a_quoted_path_is_one_word(self) -> None:
        workspace = Workspace(_NoWorkspace())
        assert await command_locations(workspace, 'echo x > "my dir/.pax/x"', posix=True) == [
            'echo',
            'x',
            'my dir/.pax/x',
        ]
        assert await command_locations(workspace, 'tool --out=a', posix=True) == ['tool', '--out=a', '--out', 'a']

    async def test_windows_paths_resolve_against_the_working_directory(self) -> None:
        workspace = Workspace(_WindowsWorkspace())
        assert await command_locations(workspace, 'Set-Content ".pax\\my settings.json"', posix=False) == [
            'C:\\Repo\\Set-Content',
            'C:\\Repo\\.pax\\my settings.json',
        ]

    async def test_other_commands_name_no_protected_path(self) -> None:
        assert not await _names_protected(Workspace(_NoWorkspace()), 'echo ".pax is a folder" > notes.md', posix=True)
        assert not await _names_protected(Workspace(_WindowsWorkspace()), 'Set-Content notes.pax x', posix=False)

    async def test_case_is_ignored_on_posix_too(self) -> None:
        """macOS file systems ignore case by default, so `.PAX` there is `.pax`."""
        assert await _names_protected(Workspace(_NoWorkspace()), 'cp x .PAX/settings.json', posix=True)
        assert matches('/repo/.Git/Hooks/pre-commit', PROTECTED, posix=True)

    async def test_a_very_long_command_counts_as_protected(self, tmp_path: Path) -> None:
        """Too long to split in reasonable time, though none of its short words is a protected path."""
        person = await _run(tmp_path, ('run_command', {'command': 'echo' + ' x' * 40_000}))
        assert [asked.person_only for asked in person.asked] == [True]

    async def test_a_very_long_command_without_protection_is_not_held(self, tmp_path: Path) -> None:
        person = _Person()
        agent = Agent[None, str](
            _calls(('run_command', {'command': 'echo' + ' x' * 40_000})),
            deps_type=type(None),
            capabilities=[
                LocalWorkspace(tmp_path),
                Shell(denied_commands=[]),
                Permissions('auto'),
                HandleDeferredToolCalls[None](approval_ask(person)),
            ],
        )
        await agent.run('go')
        assert person.asked == []


async def _names_protected(workspace: Workspace, command: str, *, posix: bool) -> bool:
    locations = await command_locations(workspace, command, posix=posix)
    return any(matches(location, PROTECTED, posix=posix) for location in locations)


class _WindowsWorkspace:
    """A Windows backend working in `C:\\Repo`."""

    ref: WorkspaceRef | None = None
    workspace_platform = 'windows'

    async def working_dir(self) -> str:
        return 'C:\\Repo'


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
