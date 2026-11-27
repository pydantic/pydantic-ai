"""Tests for `Permissions`, its `protected_paths` and `protected_commands`, `max_mode` and `managed_rules` included."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pydantic_ai import Agent, DeferredToolRequests, Tool
from pydantic_ai.capabilities import AbstractCapability, HandleDeferredToolCalls, LocalWorkspace, ValidatedToolArgs
from pydantic_ai.exceptions import ApprovalRequired, UserError
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturn,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import DeferredToolResults, RunContext, ToolApproved, ToolDefinition
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.workspaces import CommandResult, Workspace, WorkspaceBackend, WorkspaceCommand, WorkspaceRef
from pydantic_ai_harness.approval_reviewer import ApprovalReviewer
from pydantic_ai_harness.file_system import WRITE_FILE_KIND, FileSystem
from pydantic_ai_harness.permissions import (
    DEFAULT_ASK_COMMANDS,
    PermissionDecision,
    PermissionMode,
    PermissionRequest,
    PermissionRule,
    Permissions,
    PermissionVerdict,
    RememberedApproval,
    approval_ask,
    ask_permissions,
    permission_requests,
)
from pydantic_ai_harness.shell import Shell
from pydantic_ai_harness.sub_agents import SubAgents
from tests.conftest import try_import

with try_import() as prefect_installed:
    from pydantic_ai.durable_exec.prefect._cache_policies import PrefectAgentInputs


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


class _DBOSDurability(AbstractCapability[None]):
    in_durable_context = True


_DBOSDurability.__module__ = 'pydantic_ai.durable_exec.dbos'


def _return_metadata(messages: list[ModelMessage]) -> list[object]:
    return [part.metadata for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]


class _Person:
    """Answers each request with the next decision, repeating the last, and keeps what it was asked.

    Without decisions it refuses every request.
    """

    def __init__(self, *decisions: PermissionDecision) -> None:
        self.decisions = list(decisions) or [PermissionDecision(approved=False)]
        self.asked: list[PermissionRequest] = []

    async def __call__(self, request: PermissionRequest) -> PermissionDecision:
        self.asked.append(request)
        return self.decisions.pop(0) if len(self.decisions) > 1 else self.decisions[0]


def _agent(
    tmp_path: Path,
    model: FunctionModel,
    mode: PermissionMode | Callable[[RunContext[None]], PermissionMode] = 'full',
    person: _Person | None = None,
    *,
    permissions: Permissions[None] | None = None,
    extra: Sequence[AbstractCapability[None]] = (),
) -> Agent[None, str | DeferredToolRequests]:
    """An agent with a workspace, files and a shell, under `permissions` or else `mode`, then `extra`, then `person`."""
    if permissions is None:
        permissions = Permissions[None](mode)
    handler = [] if person is None else [HandleDeferredToolCalls[None](approval_ask(person))]
    return Agent[None, 'str | DeferredToolRequests'](
        model,
        deps_type=type(None),
        output_type=[str, DeferredToolRequests],
        capabilities=[
            LocalWorkspace(tmp_path),
            FileSystem(),
            Shell(denied_commands=[]),
            permissions,
            *extra,
            *handler,
        ],
    )


class TestPermissions:
    async def test_full_runs_everything_without_asking(self, tmp_path: Path) -> None:
        (tmp_path / 'doomed').mkdir()
        person = _Person()
        agent = _agent(tmp_path, _calls(('run_command', {'command': 'rm -rf doomed'})), 'full', person)

        await agent.run('go')

        assert not (tmp_path / 'doomed').exists()
        assert person.asked == []

    async def test_auto_asks_before_a_matching_command_and_deny_keeps_the_directory(self, tmp_path: Path) -> None:
        (tmp_path / 'kept').mkdir()
        person = _Person(PermissionDecision(approved=False, reason='keep it'))
        agent = _agent(tmp_path, _calls(('run_command', {'command': 'mkdir -p kept && rm -rf kept'})), 'auto', person)

        result = await agent.run('go')

        assert (tmp_path / 'kept').is_dir()
        [request] = person.asked
        assert (request.action, request.subject, request.call.tool_name) == (
            'command',
            'mkdir -p kept && rm -rf kept',
            'run_command',
        )
        assert 'Auto mode' in request.reason
        assert result.output == 'The user denied this command, so it did not run. Their reason: keep it'

    async def test_auto_approve_runs_the_command(self, tmp_path: Path) -> None:
        (tmp_path / 'doomed').mkdir()
        person = _Person(PermissionDecision(approved=True))
        agent = _agent(tmp_path, _calls(('run_command', {'command': 'rm -rf doomed'})), 'auto', person)

        await agent.run('go')

        assert not (tmp_path / 'doomed').exists()
        assert len(person.asked) == 1

    async def test_auto_runs_other_commands_and_file_writes(self, tmp_path: Path) -> None:
        person = _Person()
        model = _calls(
            ('run_command', {'command': 'echo hi > note.txt'}),
            ('write_file', {'path': 'other.txt', 'content': 'x'}),
        )

        await _agent(tmp_path, model, 'auto', person).run('go')

        assert (tmp_path / 'note.txt').read_text() == 'hi\n'
        assert (tmp_path / 'other.txt').read_text() == 'x'
        assert person.asked == []

    async def test_read_only_asks_before_writes_and_commands(self, tmp_path: Path) -> None:
        person = _Person(PermissionDecision(approved=False), PermissionDecision(approved=False))
        model = _calls(
            ('write_file', {'path': 'note.txt', 'content': 'x'}),
            ('run_command', {'command': 'echo hi'}),
            ('read_file', {'path': 'missing.txt'}),
        )

        await _agent(tmp_path, model, 'read-only', person).run('go')

        assert not (tmp_path / 'note.txt').exists()
        assert [(request.action, request.subject) for request in person.asked] == [
            ('write', 'note.txt'),
            ('command', 'echo hi'),
        ]
        assert 'Read-only mode' in person.asked[0].reason

    async def test_remember_skips_asking_again_in_the_conversation_only(self, tmp_path: Path) -> None:
        person = _Person(PermissionDecision(approved=True, remember=True), PermissionDecision(approved=False))
        command = ('run_command', {'command': 'echo hi'})
        agent = _agent(
            tmp_path, _calls(command, command, ('run_command', {'command': 'echo other'})), 'read-only', person
        )

        await agent.run('go', conversation_id='first')

        assert [request.subject for request in person.asked] == ['echo hi', 'echo other']
        person.decisions = [PermissionDecision(approved=True)]
        await agent.run('go', conversation_id='second', model=_calls(command))
        assert len(person.asked) == 3

    async def test_mode_function_is_read_on_every_call(self, tmp_path: Path) -> None:
        modes: list[PermissionMode] = ['full', 'read-only']
        person = _Person(PermissionDecision(approved=False))

        def mode(ctx: RunContext[None]) -> PermissionMode:
            return modes.pop(0)

        command = ('run_command', {'command': 'echo hi'})
        await _agent(tmp_path, _calls(command, command), mode, person).run('go')

        assert len(person.asked) == 1

    async def test_a_shell_tool_without_code_arg_name_is_not_checked(self) -> None:
        """Commands are read from the `code_arg_name` argument; a tool that declares none is not a command."""

        def shell(command: str) -> str:
            return command

        tool = Tool(shell, metadata={'code_arg_language': 'shell'})
        agent = Agent[None, 'str | DeferredToolRequests'](
            _calls(('shell', {'command': 'rm -rf x'})),
            deps_type=type(None),
            output_type=[str, DeferredToolRequests],
            tools=[tool],
            capabilities=[Permissions('read-only')],
        )

        assert (await agent.run('go')).output == 'rm -rf x'

    async def test_without_a_handler_the_call_is_deferred(self, tmp_path: Path) -> None:
        agent = _agent(tmp_path, _calls(('run_command', {'command': 'git push'})), 'auto')

        result = await agent.run('go')

        assert isinstance(result.output, DeferredToolRequests)
        [request] = permission_requests(result.output)
        assert (request.action, request.subject) == ('command', 'git push')

    async def test_other_approvals_are_left_for_the_next_handler(self, tmp_path: Path) -> None:
        requests = DeferredToolRequests(approvals=[ToolCallPart('other', {}, tool_call_id='x')])

        assert await ask_permissions(requests, _Person()) is None
        assert permission_requests(requests) == []

    async def test_approval_ask_leaves_other_approvals_pending(self, tmp_path: Path) -> None:
        """`approval_ask` answers an empty result rather than `None`, so the handler does not wait on its token."""
        person = _Person()
        agent = _agent(tmp_path, _calls(('other', {})), 'read-only', person)
        other = Tool(lambda: 'never runs', name='other', requires_approval=True)

        result = await agent.run('go', toolsets=[FunctionToolset([other])])

        assert isinstance(result.output, DeferredToolRequests)
        assert [call.tool_name for call in result.output.approvals] == ['other']
        assert person.asked == []

    async def test_approved_by_another_gate_runs_without_asking(self, tmp_path: Path) -> None:
        async def approve_all(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults:
            return DeferredToolResults(approvals={call.tool_call_id: ToolApproved() for call in requests.approvals})

        (tmp_path / 'doomed').mkdir()
        agent = Agent[None, str](
            _calls(('run_command', {'command': 'rm -rf doomed'})),
            deps_type=type(None),
            capabilities=[
                LocalWorkspace(tmp_path),
                Shell(denied_commands=[]),
                Permissions('auto'),
                HandleDeferredToolCalls[None](handler=approve_all),
            ],
        )

        await agent.run('go')

        assert not (tmp_path / 'doomed').exists()

    @pytest.mark.parametrize('shared', [False, True])
    async def test_sub_agents_are_checked_when_permissions_reaches_them(self, tmp_path: Path, shared: bool) -> None:
        """A delegate runs only what reaches it: `SubAgents(shared_capabilities=[...])` brings `Permissions` along."""
        (tmp_path / 'doomed').mkdir()
        person = _Person(PermissionDecision(approved=False))

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            last = messages[-1]
            if isinstance(last, ModelRequest) and any(isinstance(part, ToolReturnPart) for part in last.parts):
                return ModelResponse(parts=[TextPart('done')])
            if 'delegate_task' in [tool.name for tool in info.function_tools]:
                return ModelResponse(parts=[ToolCallPart('delegate_task', {'agent_name': 'self', 'task': 'clean up'})])
            return ModelResponse(parts=[ToolCallPart('run_command', {'command': 'rm -rf doomed'})])

        permissions = Permissions[None]('read-only')
        handler = HandleDeferredToolCalls[None](handler=lambda ctx, requests: ask_permissions(requests, person))
        agent = Agent[None, str](
            FunctionModel(respond),
            deps_type=type(None),
            capabilities=[
                LocalWorkspace(tmp_path),
                Shell(denied_commands=[]),
                SubAgents(
                    include_self=True,
                    max_depth=2,
                    agent_folders=None,
                    shared_capabilities=[permissions, handler] if shared else [],
                ),
            ],
        )

        await agent.run('go', capabilities=[permissions, handler])

        assert (tmp_path / 'doomed').is_dir() is shared
        assert [request.subject for request in person.asked] == (['rm -rf doomed'] if shared else [])

    async def test_a_caller_owned_memory(self, tmp_path: Path) -> None:
        """`remembered` belongs to the caller, who can read, share, or clear it."""
        remembered: set[RememberedApproval] = set()
        person = _Person(PermissionDecision(approved=True, remember=True))
        command = ('run_command', {'command': 'echo hi'})
        agent = Agent[None, str](
            _calls(command, command),
            deps_type=type(None),
            capabilities=[
                LocalWorkspace(tmp_path),
                Shell(denied_commands=[]),
                Permissions('read-only', remembered=remembered),
                HandleDeferredToolCalls[None](handler=lambda ctx, requests: ask_permissions(requests, person)),
            ],
        )

        await agent.run('go', conversation_id='c')

        assert remembered == {RememberedApproval('c', 'run_command', 'echo hi')}
        assert len(person.asked) == 1

    @pytest.mark.parametrize(
        ('returned', 'metadata', 'asks'),
        [
            ('hi', {'permissions': {'conversation_id': 'c', 'subject': 'echo hi'}}, 2),
            (
                ToolReturn('hi', metadata={'kept': True}),
                {'kept': True, 'permissions': {'conversation_id': 'c', 'subject': 'echo hi'}},
                2,
            ),
            (ToolReturn('hi', metadata='note'), 'note', 3),
            (ToolReturn('hi', metadata={'permissions': {'admin': True}}), {'permissions': {'admin': True}}, 3),
        ],
    )
    async def test_remember_inside_durable_execution_is_kept_in_history(
        self, returned: object, metadata: object, asks: int
    ) -> None:
        """In-process memory would not survive a replay, so inside a durable workflow the approval goes in history."""

        def run_command(command: str) -> object:
            return returned

        remembered: set[RememberedApproval] = set()
        person = _Person(*[PermissionDecision(approved=True, remember=True)] * asks)
        command = ('run_command', {'command': 'echo hi'})
        agent = Agent[None, str](
            _calls(command, ('run_command', {'command': 'echo bye'}), command),
            deps_type=type(None),
            tools=[Tool(run_command, metadata={'code_arg_language': 'shell', 'code_arg_name': 'command'})],
            capabilities=[
                _DBOSDurability(),
                Permissions('read-only', remembered=remembered),
                HandleDeferredToolCalls[None](handler=lambda ctx, requests: ask_permissions(requests, person)),
            ],
        )

        result = await agent.run('go', conversation_id='c')

        assert remembered == set()
        assert len(person.asked) == asks
        assert _return_metadata(result.all_messages())[0] == metadata

    async def test_remember_inside_durable_execution_keeps_the_nested_tool_return_error(self) -> None:
        def run_command(command: str) -> object:
            return [ToolReturn('x')]

        person = _Person(PermissionDecision(approved=True, remember=True))
        agent = Agent[None, str](
            _calls(('run_command', {'command': 'echo hi'})),
            deps_type=type(None),
            tools=[Tool(run_command, metadata={'code_arg_language': 'shell', 'code_arg_name': 'command'})],
            capabilities=[
                _DBOSDurability(),
                Permissions('read-only'),
                HandleDeferredToolCalls[None](handler=lambda ctx, requests: ask_permissions(requests, person)),
            ],
        )

        with pytest.raises(UserError, match='invalid nested `ToolReturn`'):
            await agent.run('go', conversation_id='c')

    async def test_remember_inside_durable_execution_holds_in_its_conversation_only(self, tmp_path: Path) -> None:
        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if any(isinstance(part, ToolReturnPart) for part in messages[-1].parts):
                return ModelResponse(parts=[TextPart('done')])
            return ModelResponse(parts=[ToolCallPart('run_command', {'command': 'echo hi'})])

        person = _Person(*[PermissionDecision(approved=True, remember=True)] * 2)
        agent = Agent[None, str](
            FunctionModel(respond),
            deps_type=type(None),
            capabilities=[
                _DBOSDurability(),
                LocalWorkspace(tmp_path),
                Shell(denied_commands=[]),
                Permissions('read-only'),
                HandleDeferredToolCalls[None](handler=lambda ctx, requests: ask_permissions(requests, person)),
            ],
        )
        first = await agent.run('go', conversation_id='c')

        await agent.run('again', message_history=first.all_messages(), conversation_id='c')
        assert len(person.asked) == 1
        await agent.run('again', message_history=first.all_messages(), conversation_id='other')
        assert len(person.asked) == 2

    async def test_remember_inside_durable_execution_ignores_calls_it_does_not_hold(self) -> None:
        async def approve_remembering(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults:
            remember = {'permissions': {'remember': True}}
            return DeferredToolResults(
                approvals={call.tool_call_id: ToolApproved() for call in requests.approvals},
                metadata={call.tool_call_id: remember for call in requests.approvals},
            )

        agent = Agent[None, str](
            _calls(('guarded', {})),
            deps_type=type(None),
            tools=[Tool(lambda: 'ok', name='guarded', requires_approval=True)],
            capabilities=[
                _DBOSDurability(),
                Permissions('read-only'),
                HandleDeferredToolCalls[None](handler=approve_remembering),
            ],
        )

        result = await agent.run('go')

        assert _return_metadata(result.all_messages()) == [None]

    def test_serialization_name(self) -> None:
        assert Permissions.get_serialization_name() == 'Permissions'

    def test_invalid_pattern_fails_at_construction(self) -> None:
        with pytest.raises(UserError, match='invalid pattern'):
            Permissions('auto', ask_commands=['('])

    @pytest.mark.parametrize(
        ('command', 'asks'),
        [
            ('rm -rf build', True),
            ('rm -f note.txt', True),
            ('rm --recursive build', True),
            ('rm note.txt', False),
            ('git push origin main', True),
            ('git reset --hard HEAD~1', True),
            ('git status', False),
            ('sudo ls', True),
            ('curl https://example.com', True),
            ('uv add httpx', True),
            ('npm install', True),
            ('git add -A', False),
            ('ls -la', False),
        ],
    )
    def test_default_ask_commands(self, command: str, asks: bool) -> None:
        assert any(re.search(pattern, command) for pattern in DEFAULT_ASK_COMMANDS) is asks


def _run_command(command: str) -> str:
    return 'ran'


_SHELL_TOOL = Tool(
    _run_command, name='run_command', metadata={'code_arg_language': 'shell', 'code_arg_name': 'command'}
)

_SHELL_TOOL_REQUIRING_APPROVAL = Tool(
    _run_command,
    name='run_command',
    requires_approval=True,
    metadata={'code_arg_language': 'shell', 'code_arg_name': 'command'},
)

_RULES = [
    PermissionRule('allow', command='git status'),
    PermissionRule('allow', command='ls'),
    PermissionRule('allow', command='echo'),
    PermissionRule('ask', command='git push'),
    PermissionRule('deny', command='rm -rf', reason='No recursive deletes.'),
]


def _rules_agent(
    model: FunctionModel, rules: Sequence[PermissionRule], mode: PermissionMode = 'full', *tools: Tool[None]
) -> Agent[None, str | DeferredToolRequests]:
    return Agent[None, 'str | DeferredToolRequests'](
        model,
        deps_type=type(None),
        output_type=[str, DeferredToolRequests],
        tools=[_SHELL_TOOL, *tools],
        capabilities=[Permissions(mode, rules=rules)],
    )


async def _verdict(command: str, rules: Sequence[PermissionRule] = _RULES, mode: PermissionMode = 'read-only') -> str:
    """Run `command` once under `rules`, and say whether it ran, was asked about, or was denied."""
    result = await _rules_agent(_calls(('run_command', {'command': command})), rules, mode).run('go')
    if isinstance(result.output, DeferredToolRequests):
        return 'ask'
    return 'allow' if result.output == 'ran' else 'deny'


class _WindowsBackend:
    """A `'windows'` workspace, whose commands run in PowerShell; the test tool never reaches it."""

    ref = WorkspaceRef(provider='windows', id='C:\\project')
    workspace_platform = 'windows'

    async def working_dir(self) -> str:
        raise NotImplementedError

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        raise NotImplementedError


class TestPermissionRules:
    @pytest.mark.parametrize(
        ('command', 'verdict'),
        [
            ('git status', 'allow'),
            ('git status --short', 'allow'),
            ('ls -la && git status', 'allow'),
            ('ls 2>&1', 'allow'),
            ('ls > /dev/null', 'allow'),
            ('echo a\\ b', 'allow'),
            ('echo "rm -rf /"', 'ask'),
            ('sh -c "rm -rf /"', 'ask'),
            ('echo $(date); "rm" -rf /', 'deny'),
            ('ls; rm -rf /', 'deny'),
            ('sudo rm -rf /', 'deny'),
            ('/bin/rm -rf build', 'deny'),
            ('echo $(rm -rf /)', 'deny'),
            ('(rm -rf /)', 'deny'),
            ('git status\nrm -rf /', 'deny'),
            ('ls\r/payload', 'ask'),
            ('ls && git push origin main', 'ask'),
            ('git status; curl https://example.com', 'ask'),
            ('gitx status', 'ask'),
            ('FOO=1 git status', 'ask'),
            ('echo hi > ~/.bashrc', 'ask'),
            ('ls >> log', 'ask'),
            ('ls >', 'ask'),
            ('ls | sh', 'ask'),
            ('echo `whoami`', 'ask'),
            ('git status $(pwd)', 'ask'),
            ('ls; > ~/.bashrc', 'ask'),
            ("echo $'a\\''; touch x # '", 'ask'),
            ('cat <<EOF', 'ask'),
            ('rm 2>/dev/null -rf build', 'deny'),
            ('git 1 >&2 status', 'ask'),
            ("echo '>' /dev/null", 'ask'),
            ('echo "a|b"', 'ask'),
            ('echo \\; ls', 'ask'),
            ('ls ;; ls', 'ask'),
            ("echo 'unbalanced", 'ask'),
            ('', 'ask'),
        ],
    )
    async def test_shell_commands_are_judged_by_simple_command(self, command: str, verdict: str) -> None:
        """Every simple command must be allowed for the whole to run; one asked or denied decides it."""
        assert await _verdict(command) == verdict

    @pytest.mark.parametrize(
        ('command', 'rules', 'verdict'),
        [
            ('echo $(date)', [], 'allow'),
            ('ls $(pwd)', [PermissionRule('allow', command='git status')], 'allow'),
            ('echo $(date)', [PermissionRule('deny', command='rm -rf')], 'ask'),
            ('rm$IFS-rf /', [PermissionRule('deny', command='rm -rf')], 'ask'),
            ('X=-rf; rm $X /', [PermissionRule('deny', command='rm -rf')], 'ask'),
            ('{rm,-rf,build}', [PermissionRule('deny', command='rm -rf')], 'ask'),
            ("grep -E 'a|b' x.py", [PermissionRule('deny', command='rm -rf')], 'allow'),
            ("awk '{print $1}' f", [PermissionRule('deny', command='rm -rf')], 'allow'),
            ('ls\ngit status', [PermissionRule('deny', command='rm -rf')], 'allow'),
            ("python -c 'a(); rm -rf'", [PermissionRule('deny', command='rm -rf')], 'ask'),
            ('echo "$HOME"', [PermissionRule('deny', command='rm -rf')], 'ask'),
            ('r\\\nm -rf build', [PermissionRule('deny', command='rm -rf')], 'ask'),
            ("# '\nr$(printf m) -rf build", [PermissionRule('deny', command='rm -rf')], 'ask'),
            ('sudo /bin/r[m] -rf build', [PermissionRule('deny', command='rm -rf')], 'deny'),
            ('FOO=1 /bin/r? -r[f] build', [PermissionRule('deny', command='rm -rf')], 'deny'),
            ('env /usr/bin/gi[t] push', [PermissionRule('ask', command='git push')], 'ask'),
            ('chmod 777 >/dev/null f', [PermissionRule('deny', command='chmod 777')], 'deny'),
            ('rm -fr build', [PermissionRule('deny', command='rm -rf')], 'deny'),
            ('rm -Rfv build', [PermissionRule('deny', command='rm -rf')], 'deny'),
            ('rm -r -f build', [PermissionRule('deny', command='rm -rf')], 'deny'),
            ('rm -r build', [PermissionRule('deny', command='rm -rf')], 'allow'),
            ('git -C repo push', [PermissionRule('ask', command='git push')], 'ask'),
            ('git -c user.name=x push origin', [PermissionRule('ask', command='git push')], 'ask'),
            ('ls *', [PermissionRule('deny', command='curl')], 'allow'),
            ('git commit -m "Add * support"', [PermissionRule('deny', command='curl')], 'allow'),
            ('[ -f setup.py ] && ls', [PermissionRule('deny', command='rm -rf')], 'allow'),
            ('/bin/rm -rf build', [PermissionRule('deny', command='/bin/rm -rf')], 'deny'),
            ('rm >/dev/null -rf build\n', [PermissionRule('deny', command='rm -rf')], 'deny'),
            ('rm -{rf,} build', [PermissionRule('deny', command='rm -rf')], 'ask'),
            ('/bin/r[m] -rf build', [PermissionRule('deny', command='rm -rf')], 'deny'),
            ('/bin/r[m] -f build', [PermissionRule('deny', command='rm -rf')], 'ask'),
            ('ls *.py', [PermissionRule('deny', command='rm -rf')], 'allow'),
            ('echo $(date)', [PermissionRule('allow', tools=['run_command'])], 'allow'),
            (
                'sh -c "rm -rf /; $(x)"',
                [PermissionRule('allow', tools=['run_command']), PermissionRule('deny', command='rm -rf')],
                'ask',
            ),
            ('rm -rf / # $(x)', [PermissionRule('deny', command='rm -rf')], 'deny'),
            ("/bin/rm '' -rf build", [PermissionRule('deny', command='rm -rf')], 'deny'),
            ("'rm' -rf /\ntrue", [PermissionRule('deny', command='rm -rf')], 'deny'),
            ('r\\m -rf /\ntrue', [PermissionRule('deny', command='rm -rf')], 'deny'),
            ('git commit -m "push the fix"', [PermissionRule('ask', command='git push')], 'allow'),
            ('git commit -m "stop using rm -rf in CI"', [PermissionRule('deny', command='rm -rf')], 'ask'),
            ('grep -rn "rm -rf" .', [PermissionRule('deny', command='rm -rf')], 'ask'),
        ],
    )
    async def test_unsplittable_commands_ask_when_command_rules_apply(
        self, command: str, rules: list[PermissionRule], verdict: str
    ) -> None:
        """With ask or deny rules for the tool, a command they cannot split asks unless one denies it; else the mode decides."""
        assert await _verdict(command, rules, 'full') == verdict

    async def test_powershell_commands_are_never_split(self) -> None:
        """PowerShell quoting differs from `sh`, so on a `'windows'` workspace an allow rule cannot vouch for a command."""
        rules = [PermissionRule('allow', command='echo')]
        agent = _rules_agent(_calls(('run_command', {'command': 'echo hi'})), rules, 'read-only')

        result = await agent.run('go', workspace=Workspace(_WindowsBackend()))

        assert isinstance(result.output, DeferredToolRequests)

    async def test_powershell_command_names_ignore_case(self) -> None:
        rules = [PermissionRule('deny', command='Remove-Item')]
        agent = _rules_agent(_calls(('run_command', {'command': 'remove-item -LiteralPath x'})), rules)

        result = await agent.run('go', workspace=Workspace(_WindowsBackend()))

        assert result.output == 'A permission rule denied this command, so it did not run. Do not retry it.'

    async def test_a_tool_wide_allow_needs_no_split(self) -> None:
        """An allow rule without a `command` doesn't look at the command, so it holds even where commands can't be split."""
        rules = [PermissionRule('allow', tools=['run_command'])]
        agent = _rules_agent(_calls(('run_command', {'command': 'echo hi'})), rules, 'read-only')

        result = await agent.run('go', workspace=Workspace(_WindowsBackend()))

        assert result.output == 'ran'

    async def test_a_quoted_argument_in_a_deny_rule_matches_the_same_command(self) -> None:
        rules = [PermissionRule('deny', command='git commit -m "release build"')]

        assert await _verdict('git commit -m "release build"', rules, 'full') == 'deny'

    @pytest.mark.parametrize(
        ('rule', 'held'),
        [
            (PermissionRule('allow', tools=['write_file'], args={'path': 'notes/*'}), True),
            (PermissionRule('deny', tools=['write_file'], args={'path': 'secrets/*'}), False),
        ],
    )
    async def test_path_globs_match_the_normalized_path_too(
        self, tmp_path: Path, rule: PermissionRule, held: bool
    ) -> None:
        """`notes/../secrets/key` writes `secrets/key`, so a `notes/*` allow cannot vouch for it and `secrets/*` denies it."""
        (tmp_path / 'notes').mkdir()
        agent = Agent[None, 'str | DeferredToolRequests'](
            _calls(('write_file', {'path': 'notes/../secrets/key', 'content': 'x'})),
            deps_type=type(None),
            output_type=[str, DeferredToolRequests],
            capabilities=[LocalWorkspace(tmp_path), FileSystem(), Permissions('read-only', rules=[rule])],
        )

        result = await agent.run('go')

        assert not (tmp_path / 'secrets').exists()
        assert isinstance(result.output, DeferredToolRequests) is held

    @pytest.mark.parametrize(
        ('path', 'windows'),
        [
            ('/project/secrets/key', False),
            ('../secrets/key', False),
            ('SECRETS\\key', True),
            ('C:\\p\\secrets\\k', True),
        ],
    )
    async def test_deny_path_globs_match_any_spelling_of_the_path(self, path: str, windows: bool) -> None:
        """A deny glob also matches the end of a longer path, and on Windows ignores case and separator style."""

        def save(path: str) -> str:
            raise NotImplementedError

        rules = [PermissionRule('deny', tools=['save'], args={'path': 'secrets/*'})]
        agent = _rules_agent(_calls(('save', {'path': path})), rules, 'full', Tool(save))

        result = await agent.run('go', workspace=Workspace(_WindowsBackend()) if windows else None)

        assert result.output == 'A permission rule denied this tool call, so it did not run. Do not retry it.'

    @pytest.mark.parametrize(('path', 'runs'), [('Notes\\a.txt', True), ('notes\\..\\secrets\\k', False)])
    async def test_windows_allow_path_globs_need_every_spelling_to_match(self, path: str, runs: bool) -> None:
        def save(path: str, content: str) -> str:
            return 'saved'

        rules = [PermissionRule('allow', tools=['save'], args={'path': 'notes/*'})]
        tool = Tool(save, tool_kind=WRITE_FILE_KIND)
        agent = _rules_agent(_calls(('save', {'path': path, 'content': 'x'})), rules, 'read-only', tool)

        result = await agent.run('go', workspace=Workspace(_WindowsBackend()))

        assert (result.output == 'saved') is runs

    async def test_an_allow_rule_globbing_the_command_argument_never_matches(self) -> None:
        """A glob would let `git status; python evil.py` through `git *`, so only `command` can allow commands."""
        rules = [PermissionRule('allow', tools=['run_command'], args={'command': 'git *'})]

        assert await _verdict('git status; python evil.py', rules) == 'ask'
        assert await _verdict('git status', rules) == 'ask'

    @pytest.mark.parametrize(('verdict', 'expected'), [('deny', 'ask'), ('allow', 'allow')])
    async def test_long_commands_are_not_parsed(
        self, monkeypatch: pytest.MonkeyPatch, verdict: PermissionVerdict, expected: str
    ) -> None:
        """A huge command could block the event loop to parse: ask and deny rules make it ask, else the mode decides."""
        monkeypatch.setattr('pydantic_ai_harness.permissions._rules.COMMAND_LIMIT', 8)

        assert await _verdict('echo 123456789', [PermissionRule(verdict, command='curl')], 'full') == expected

    async def test_long_argument_values_are_not_globbed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Globbing a huge value could take quadratic time, so ask and deny rules match it and allow rules don't."""
        monkeypatch.setattr('pydantic_ai_harness.permissions._rules._GLOB_LIMIT', 8)

        def save(path: str, content: str) -> str:
            raise NotImplementedError

        tool = Tool(save, tool_kind=WRITE_FILE_KIND)
        call = ('save', {'path': 'a/b/c/d/e', 'content': 'x'})
        deny = [PermissionRule('deny', tools=['save'], args={'path': 'secrets/*'})]
        allow = [PermissionRule('allow', tools=['save'], args={'path': 'a/*'})]

        denied = await _rules_agent(_calls(call), deny, 'full', tool).run('go')
        held = await _rules_agent(_calls(call), allow, 'read-only', tool).run('go')

        assert denied.output == 'A permission rule denied this file write, so it did not run. Do not retry it.'
        assert isinstance(held.output, DeferredToolRequests)

    async def test_an_allow_rule_skips_read_only_asks_but_not_ask_commands(self) -> None:
        """`ask_commands` still asks about a command an allow rule matches; remove a pattern from it to opt out."""
        allow_git = [PermissionRule('allow', command='git')]

        assert await _verdict('git status', allow_git, 'read-only') == 'allow'
        assert await _verdict('git push', allow_git, 'read-only') == 'ask'
        assert await _verdict('git push', allow_git, 'auto') == 'ask'

    async def test_deny_returns_a_denied_tool_result_to_the_model(self) -> None:
        result = await _rules_agent(_calls(('run_command', {'command': 'rm -rf /'})), _RULES).run('go')

        [part] = [
            part for message in result.all_messages() for part in message.parts if isinstance(part, ToolReturnPart)
        ]
        assert (part.outcome, part.content) == (
            'denied',
            'A permission rule denied this command, so it did not run. Do not retry it. Reason: No recursive deletes.',
        )

    async def test_deny_beats_allow_and_an_earlier_approval(self) -> None:
        """A tool approved by another gate is still refused when a rule denies it, whatever the rules' order."""

        async def approve_all(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults:
            return DeferredToolResults(approvals={call.tool_call_id: ToolApproved() for call in requests.approvals})

        guarded = Tool(_run_command, name='guarded', requires_approval=True)
        agent = Agent[None, str](
            _calls(('guarded', {'command': 'x'})),
            deps_type=type(None),
            tools=[guarded],
            capabilities=[
                Permissions(
                    rules=[PermissionRule('allow'), PermissionRule('deny', tools=['guarded']), PermissionRule('deny')]
                ),
                HandleDeferredToolCalls[None](handler=approve_all),
            ],
        )

        assert (
            await agent.run('go')
        ).output == 'A permission rule denied this tool call, so it did not run. Do not retry it.'

    async def test_ask_rule_on_any_tool_by_arguments(self) -> None:
        def deploy(env: str, replicas: int = 1) -> str:
            return f'deployed {env}'

        rules = [
            PermissionRule('ask', tools=['deploy'], args={'env': 'prod*'}, reason='Production needs a look.'),
            PermissionRule('ask', tools=['deploy'], args={'replicas': '*'}),
        ]
        agent = _rules_agent(
            _calls(('deploy', {'env': 'staging', 'replicas': 3}), ('deploy', {'env': 'production'})),
            rules,
            'full',
            Tool(deploy),
        )

        held = await agent.run('go')

        assert isinstance(held.output, DeferredToolRequests)
        [call] = held.output.approvals
        assert (call.args_as_dict(), permission_requests(held.output)) == ({'env': 'production'}, [])
        assert held.output.metadata == {call.tool_call_id: {'reason': 'Production needs a look.'}}
        resumed = await agent.run(
            message_history=held.all_messages(),
            deferred_tool_results=DeferredToolResults(approvals={call.tool_call_id: True}),
        )
        assert resumed.output == 'deployed production'

    async def test_allow_rule_overrides_the_mode_and_ask_rule_names_the_written_path(self, tmp_path: Path) -> None:
        rules = [
            PermissionRule('allow', tools=['write_file'], args={'path': 'notes/*'}),
            PermissionRule('ask', tools=['write_file'], args={'path': 'secrets/*'}, reason='Secrets need a look.'),
        ]
        (tmp_path / 'notes').mkdir()
        agent = Agent[None, 'str | DeferredToolRequests'](
            _calls(
                ('write_file', {'path': 'notes/a.txt', 'content': 'x'}),
                ('write_file', {'path': 'secrets/key', 'content': 'x'}),
            ),
            deps_type=type(None),
            output_type=[str, DeferredToolRequests],
            capabilities=[LocalWorkspace(tmp_path), FileSystem(), Permissions('read-only', rules=rules)],
        )

        result = await agent.run('go')

        assert (tmp_path / 'notes' / 'a.txt').read_text() == 'x'
        assert isinstance(result.output, DeferredToolRequests)
        [request] = permission_requests(result.output)
        assert (request.action, request.subject, request.reason) == ('write', 'secrets/key', 'Secrets need a look.')

    @pytest.mark.parametrize('command', ['', '   ', "'unbalanced", "';'"])
    def test_invalid_command_fails_at_construction(self, command: str) -> None:
        with pytest.raises(UserError, match='must be a shell command'):
            PermissionRule('deny', command=command)

    def test_agent_spec_builds_rules(self) -> None:
        agent = Agent.from_spec(
            {
                'model': 'test',
                'capabilities': [
                    {
                        'Permissions': {
                            'mode': 'auto',
                            'rules': [{'verdict': 'deny', 'tools': ['run_command'], 'command': 'git push'}],
                        }
                    }
                ],
            },
            custom_capability_types=[Permissions],
        )

        capability = agent.root_capability.capabilities[-1]
        assert isinstance(capability, Permissions)
        assert (capability.mode, list(capability.rules)) == (
            'auto',
            [PermissionRule('deny', tools=['run_command'], command='git push')],
        )


PROTECTED = ('.pax', '.pax/*', '.git/hooks/*')


class _Denier:
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


async def _protected_run(
    tmp_path: Path,
    *calls: tuple[str, Mapping[str, object]],
    mode: PermissionMode = 'auto',
    rules: Sequence[PermissionRule[None]] = (),
    protected_commands: Sequence[str] = (),
    protected_paths: Sequence[str] = PROTECTED,
    extra: Sequence[AbstractCapability[None]] = (),
) -> _Denier:
    person = _Denier()
    agent = Agent[None, 'str | DeferredToolRequests'](
        _calls(*calls),
        deps_type=type(None),
        output_type=[str, DeferredToolRequests],
        capabilities=[
            LocalWorkspace(tmp_path),
            FileSystem(),
            Shell(denied_commands=[]),
            Permissions(mode, rules=rules, protected_paths=protected_paths, protected_commands=protected_commands),
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
        person = await _protected_run(
            tmp_path, (tool, {'path': path, 'content': 'x'} if tool == 'write_file' else {'path': path})
        )
        [asked] = person.asked
        assert asked.person_only and asked.action == 'write' and asked.subject == path
        assert asked.reason == 'This file write changes a protected path, so only a person can approve it.'
        assert not (tmp_path / path).exists()

    async def test_a_link_into_a_protected_path_asks(self, tmp_path: Path) -> None:
        (tmp_path / '.pax').mkdir()
        (tmp_path / 'link').symlink_to('.pax')
        person = await _protected_run(tmp_path, ('write_file', {'path': 'link/settings.json', 'content': 'x'}))
        assert [asked.subject for asked in person.asked] == ['link/settings.json']

    async def test_other_writes_and_reads_run(self, tmp_path: Path) -> None:
        (tmp_path / '.pax').mkdir()
        (tmp_path / '.pax' / 'settings.json').write_text('{}')
        person = await _protected_run(
            tmp_path,
            ('write_file', {'path': 'notes.pax', 'content': 'x'}),
            ('read_file', {'path': '.pax/settings.json'}),
            ('run_command', {'command': 'cat .pax/settings.json && ls .git/hooks'}),
        )
        assert person.asked == []
        assert (tmp_path / 'notes.pax').read_text() == 'x'

    async def test_full_lets_a_protected_write_run(self, tmp_path: Path) -> None:
        (tmp_path / '.pax').mkdir()
        person = await _protected_run(
            tmp_path, ('write_file', {'path': '.pax/settings.json', 'content': 'x'}), mode='full'
        )
        assert person.asked == []
        assert (tmp_path / '.pax' / 'settings.json').read_text() == 'x'

    async def test_an_allow_rule_does_not_let_a_protected_write_run(self, tmp_path: Path) -> None:
        person = await _protected_run(
            tmp_path,
            ('write_file', {'path': '.pax/settings.json', 'content': 'x'}),
            mode='read-only',
            rules=[PermissionRule('allow', tools=['write_file'])],
        )
        assert [asked.person_only for asked in person.asked] == [True]

    async def test_a_deny_rule_still_refuses_a_protected_write(self, tmp_path: Path) -> None:
        (tmp_path / '.pax').mkdir()
        person = await _protected_run(
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
        person = await _protected_run(tmp_path, ('run_command', {'command': command}))
        [asked] = person.asked
        assert asked.person_only and asked.action == 'command' and asked.subject == command
        assert asked.reason == 'This command may change a protected path, so only a person can approve it.'
        assert not (tmp_path / '.pax').exists()

    async def test_a_protected_command_asks(self, tmp_path: Path) -> None:
        commands = ['uv run settings set model x', 'echo $PATH', 'settings get model']
        person = await _protected_run(
            tmp_path, *(('run_command', {'command': c}) for c in commands), protected_commands=['settings set']
        )
        assert [(asked.subject, asked.person_only) for asked in person.asked] == [(commands[0], True)]

    async def test_protected_commands_alone_leave_paths_unchecked(self, tmp_path: Path) -> None:
        """With no `protected_paths`, only the protected command asks; a write the paths would cover runs."""
        commands = ['settings set model x', 'mkdir -p .pax && echo x > .pax/settings.json']
        person = await _protected_run(
            tmp_path,
            *(('run_command', {'command': c}) for c in commands),
            protected_commands=['settings set'],
            protected_paths=(),
        )
        assert [asked.subject for asked in person.asked] == [commands[0]]
        assert (tmp_path / '.pax' / 'settings.json').exists()

    async def test_a_reviewer_never_approves_a_protected_call(self, tmp_path: Path) -> None:
        """A reviewer that says every call is safe approves the `git push`, but the protected write goes to the person."""
        reviewer = ApprovalReviewer[None](_safe_reviewer(), question='Could this call do harm?')
        (tmp_path / '.pax').mkdir()
        person = await _protected_run(
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


def _approval_required_shell(ran: list[str], *, requires_approval: bool = True) -> Tool[None]:
    """A shell tool registered with `requires_approval=True`, so core defers it before any `before_tool_execute` hook."""

    def run_command(command: str) -> str:
        ran.append(command)
        return 'ran'

    return Tool(
        run_command,
        requires_approval=requires_approval,
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


async def _reviewer_approved_call(*capabilities: AbstractCapability[None]) -> RunContext[None]:
    """Run an unprotected call a reviewer approves, and return the context its tool ran with."""
    seen: list[RunContext[None]] = []

    def run_command(ctx: RunContext[None], command: str) -> str:
        seen.append(ctx)
        return 'ran'

    agent = Agent[None, str](
        _calls(('run_command', {'command': 'echo x > notes.md'})),
        deps_type=type(None),
        tools=[Tool(run_command, requires_approval=True)],
        capabilities=[*capabilities, ApprovalReviewer[None](_safe_reviewer(), question='Could this call do harm?')],
    )
    await agent.run('go')
    [ctx] = seen
    return ctx


def _prefect_key(ctx: RunContext[None]) -> str | None:
    """The key Prefect caches a tool call under, from its `RunContext` and arguments."""
    return PrefectAgentInputs().compute_key(
        task_ctx=MagicMock(), inputs={'ctx': ctx, 'args': {'command': 'echo x > notes.md'}}, flow_parameters={}
    )


class _DefersFirst(AbstractCapability[None]):
    """Defers every call from `after_tool_validate`, before `Permissions` sees it, without saying it is protected."""

    async def after_tool_validate(
        self, ctx: RunContext[None], *, call: ToolCallPart, tool_def: ToolDefinition, args: ValidatedToolArgs
    ) -> ValidatedToolArgs:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return args


class TestProtectedCallsDeferredBeforePermissions:
    """A protected call deferred by `requires_approval=True` is still for a person only."""

    async def test_a_reviewer_leaves_it_to_the_person(self, tmp_path: Path) -> None:
        ran: list[str] = []
        person = _Denier()
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

    async def test_a_reviewer_approval_of_a_call_another_capability_deferred_first_holds_it_for_a_person(
        self, tmp_path: Path
    ) -> None:
        """That request doesn't say the call is protected, so the reviewer approves it, which doesn't let it run."""
        ran: list[str] = []
        command = 'echo x > .pax/settings.json'
        agent = Agent[None, 'str | DeferredToolRequests'](
            _calls(('run_command', {'command': command})),
            deps_type=type(None),
            output_type=[str, DeferredToolRequests],
            tools=[_approval_required_shell(ran, requires_approval=False)],
            capabilities=[
                LocalWorkspace(tmp_path),
                Permissions('auto', protected_paths=PROTECTED),
                _DefersFirst(),
                ApprovalReviewer[None](_safe_reviewer(), question='Could this call do harm?'),
            ],
        )
        result = await agent.run('go')
        assert isinstance(result.output, DeferredToolRequests)
        assert [(r.subject, r.person_only) for r in permission_requests(result.output)] == [(command, True)]
        assert ran == []

    async def test_an_unprotected_call_still_goes_to_the_reviewer(self, tmp_path: Path) -> None:
        ran: list[str] = []
        person = _Denier()
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
        """Results a reviewer gave for the call where nothing marked it protected, such as before `.pax` was."""
        command = 'echo x > .pax/settings.json'
        reviewer = _KeepingResults(ApprovalReviewer[None](_safe_reviewer(), question='Could this call do harm?'))
        unprotected = Agent[None, str](
            _calls(('run_command', {'command': command})),
            deps_type=type(None),
            tools=[_approval_required_shell([])],
            capabilities=[Permissions('auto', protected_paths=['.git/hooks/*']), reviewer],
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

    @pytest.mark.parametrize(
        'permissions',
        [
            pytest.param([], id='no-permissions'),
            pytest.param([Permissions[None]('auto')], id='nothing-protected'),
        ],
    )
    async def test_without_protection_a_reviewer_approval_leaves_the_call_as_it_was(
        self, permissions: list[AbstractCapability[None]]
    ) -> None:
        """No marker in its `tool_call_metadata`, so neither the tool nor a durable cache key sees a change."""
        seen = await _reviewer_approved_call(*permissions)
        assert seen.tool_call_approved and seen.tool_call_metadata is None
        if prefect_installed():  # pragma: no branch
            assert _prefect_key(seen) == _prefect_key(replace(seen, tool_call_metadata=None))

    @pytest.mark.parametrize(
        'permissions',
        [
            pytest.param(Permissions[None]('auto', protected_paths=['.git/hooks/*']), id='protected-paths'),
            pytest.param(Permissions[None]('auto', protected_commands=['pax config']), id='protected-commands'),
            pytest.param(Permissions[None]('auto', protected_paths=['.git/hooks/*']).prefix_tools('p'), id='wrapped'),
        ],
    )
    async def test_with_protection_a_reviewer_approval_is_marked(self, permissions: AbstractCapability[None]) -> None:
        seen = await _reviewer_approved_call(permissions)
        assert seen.tool_call_metadata == {'approval_reviewer': {'approved': True}}

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


def _write_file(path: str, content: str) -> str:
    return 'wrote'


_WRITE_TOOL = Tool(_write_file, name='write_file', tool_kind=WRITE_FILE_KIND)


def _command(command: str) -> tuple[str, Mapping[str, object]]:
    return ('run_command', {'command': command})


async def _held(
    *calls: tuple[str, Mapping[str, object]],
    workspace: WorkspaceBackend | None = None,
    paths: Sequence[str] = PROTECTED,
    commands: Sequence[str] = (),
    durable: bool = False,
) -> list[str]:
    """Make `calls` in `auto` under `protected_paths=paths`, and say which a person was asked about."""
    person = _Denier()
    agent = Agent[None, str](
        _calls(*calls),
        deps_type=type(None),
        tools=[_SHELL_TOOL, _WRITE_TOOL],
        capabilities=[
            *([_DBOSDurability()] if durable else []),
            Permissions('auto', protected_paths=paths, protected_commands=commands),
            HandleDeferredToolCalls[None](handler=lambda ctx, requests: ask_permissions(requests, person)),
        ],
    )
    await agent.run('go', workspace=None if workspace is None else Workspace(workspace))
    assert all(asked.person_only for asked in person.asked)
    return [asked.subject for asked in person.asked]


class TestDurableRunsDoNotExpandFromTheProcess:
    """A replay may run on a worker whose environment differs, so a durable run never expands `~` or a variable."""

    CONFIG = ('/home/me/.config/pydantic-ai-pax', '/home/me/.config/pydantic-ai-pax/*')

    @pytest.mark.parametrize(
        'command',
        ['echo x > ~/.config/pydantic-ai-pax/config.db', 'cp x $CONFIG_HOME/pydantic-ai-pax/plugins/x.py'],
    )
    async def test_the_same_command_asks_whatever_the_environment(
        self, monkeypatch: pytest.MonkeyPatch, command: str
    ) -> None:
        decisions: list[list[str]] = []
        for home in ('/home/me', '/home/elsewhere'):
            monkeypatch.setenv('HOME', home)
            monkeypatch.setenv('CONFIG_HOME', f'{home}/.config')
            decisions.append(await _held(_command(command), paths=self.CONFIG, durable=True))
        assert decisions == [[command], [command]]

    async def test_with_only_protected_commands_a_word_to_expand_runs(self) -> None:
        """No `protected_paths` to match, so a `~` word needs no expanding, even in a durable run."""
        command = _command('echo x > ~/notes.txt')
        assert await _held(command, paths=(), commands=['pax config'], durable=True) == []

    async def test_outside_a_durable_run_the_environment_is_expanded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        command = 'echo x > ~/.config/pydantic-ai-pax/config.db'
        monkeypatch.setenv('HOME', '/home/me')
        assert await _held(_command(command), paths=self.CONFIG) == [command]
        monkeypatch.setenv('HOME', '/home/elsewhere')
        assert await _held(_command(command), paths=self.CONFIG) == []

    async def test_reading_and_unexpanded_commands_still_run(self) -> None:
        reading = _command('cat ~/.config/pydantic-ai-pax/config.db')
        assert await _held(reading, _command('echo x > notes.md'), paths=self.CONFIG, durable=True) == []

    async def test_a_windows_variable_counts_as_needing_expansion(self) -> None:
        command = 'copy x %APPDATA%\\pax'
        assert await _held(_command(command), workspace=_WindowsRepo(), durable=True) == [command]
        assert await _held(_command('copy x 100%'), durable=True) == []


class TestProtectedPathMatching:
    async def test_a_relative_glob_matches_the_end_of_a_path(self) -> None:
        commands = [
            'echo x > /repo/.pax/settings.json',
            'echo x > /repo/notes.pax',
            'echo x > /home/me/.config/pax/config.db',
        ]
        paths = ['.pax', '.pax/*', '/home/me/.config/pax/*']
        assert await _held(*map(_command, commands), paths=paths) == [commands[0], commands[2]]

    async def test_windows_paths_ignore_case_and_separators(self) -> None:
        commands = ['Set-Content C:\\Repo\\.PAX\\Settings.json x', 'Set-Content C:\\Users\\me\\pax\\x y']
        paths = [*PROTECTED, 'c:\\users\\ME\\pax\\*']
        assert await _held(*map(_command, commands), workspace=_WindowsRepo(), paths=paths) == commands

    async def test_a_very_long_path_counts_as_protected(self) -> None:
        command = 'echo x > /' + 'x' * 5000
        assert await _held(_command(command)) == [command]

    async def test_without_a_workspace_paths_are_only_normalized(self) -> None:
        calls = [
            ('write_file', {'path': 'a/../.pax/x', 'content': 'x'}),
            ('write_file', {'path': '.pax/../notes.md', 'content': 'x'}),
            _command('echo x > a/../.pax/x'),
            _command('echo x > .pax/../notes.md'),
        ]
        assert await _held(*calls) == ['a/../.pax/x', 'echo x > a/../.pax/x']

    async def test_a_link_that_cannot_be_followed_counts_as_written(self, tmp_path: Path) -> None:
        write = ('write_file', {'path': '.pax/x', 'content': 'x'})
        assert await _held(write, workspace=_NoRealpath(tmp_path)) == ['.pax/x']

    async def test_home_and_variables_are_expanded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('HOME', '/home/me')
        monkeypatch.setenv('PAX_DIR', '/home/me/pax')
        commands = ['touch ~/.pax/x', 'touch $PAX_DIR/y', 'touch pax/z']
        paths = ['/home/me/.pax/*', '/home/me/pax/*']
        assert await _held(*map(_command, commands), paths=paths) == commands[:2]

    async def test_a_command_that_only_reads_names_no_path(self) -> None:
        commands = ['cat .pax/x | grep a', 'cat .pax/x > out', 'cat $X .pax/x', './cat .pax/x', '/usr/bin/cat .pax/x']
        assert await _held(*map(_command, commands)) == commands[1:]

    async def test_a_read_only_command_on_windows_is_checked(self) -> None:
        command = 'cat .pax\\x'
        assert await _held(_command(command), workspace=_WindowsRepo()) == [command]


class TestProtectedCommandWords:
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
            'echo x#y > /home/u/my\\ dir/.pax/settings.json',
        ],
        ids=[
            'double-quotes',
            'single-quotes',
            'escaped-space',
            'operators',
            'option-value',
            'backticks',
            'hash-in-word',
        ],
    )
    async def test_posix_spellings(self, command: str) -> None:
        assert await _held(_command(command), paths=['/home/u/my dir/.pax/*']) == [command]

    async def test_unbalanced_quotes_count_every_reading(self) -> None:
        command = 'echo "x > /home/u/my dir/.pax/settings.json'
        assert await _held(_command(command)) == [command]

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
        assert await _held(_command(command), workspace=_WindowsRepo()) == [command]

    async def test_a_script_a_shell_or_eval_runs_is_checked(self) -> None:
        commands = [
            "sh -c 'echo x > .pax/settings.json'",
            'bash -lc "echo x > .pax/settings.json"',
            'sh -c "sh -c \'echo x > .pax/settings.json\'"',
            "env /bin/sh -c 'echo x > .pax/settings.json'",
            "eval 'echo x > .pax/settings.json'",
            "sh -c 'pax config set model x'",
            'eval eval eval eval echo hi',
            "bash -c -- 'echo x > .pax/settings.json'",
            "sh -e -c 'echo x > .pax/settings.json'",
            "bash -o pipefail -c 'echo x > .pax/settings.json'",
            "bash --norc -c 'echo x > .pax/settings.json'",
            "bash -c -e 'echo x > .pax/settings.json'",
            "sh -c 'echo hi'",
            "bash -c -- 'echo hi'",
            'eval eval eval echo hi',
            'sh notes.sh',
        ]
        assert await _held(*map(_command, commands), commands=['pax config']) == commands[:12]

    async def test_an_eval_after_a_shell_without_a_script_is_checked(self) -> None:
        commands = [
            "sh /dev/null; eval 'echo x > .pax/settings.json'",
            "bash build.sh && sh x; eval 'echo x > .pax/settings.json'",
        ]
        assert await _held(*map(_command, commands)) == commands

    @pytest.mark.parametrize(
        'command',
        [
            'cp payload .git/hook?/pre-commit',
            'cp payload .git/hook*/pre-commit',
            'echo x > .pa[x]/settings.json',
            'echo x > .pa[!y]/settings.json',
            'echo x > .pa[]x]/settings.json',
            'tee .pa{x,y}/settings.json < payload',
            'cp payload .p{a{x,y},b}/settings.json',
            'cp payload .pa{w..z}/settings.json',
            'cp payload notes/.*/x',
            'cp payload notes/[.]?/x',
            'touch x{a,b}{a,b}{a,b}{a,b}{a,b}{a,b}{a,b}',
        ],
        ids=[
            'question-mark',
            'star',
            'bracket',
            'negated-bracket',
            'bracket-holding-a-bracket',
            'braces',
            'nested-braces',
            'brace-sequence',
            'could-be-a-dot',
            'bracket-could-be-a-dot',
            'too-many-braces',
        ],
    )
    async def test_a_pattern_that_could_expand_to_a_protected_path(self, command: str) -> None:
        assert await _held(_command(command)) == [command]

    async def test_a_windows_wildcard_that_could_name_a_protected_path(self) -> None:
        commands = ['Set-Content .pa?\\settings.json x', 'Set-Content notes\\.*\\x y', 'Set-Content *.txt x']
        assert await _held(*map(_command, commands), workspace=_WindowsRepo()) == commands[:2]

    @pytest.mark.parametrize(
        'command',
        [
            'sort -o.pax/settings.json payload',
            'cp payload -t.git/hooks',
            'install -D payload -t.git/hooks',
            'cp -vt.git/hooks payload',
        ],
    )
    async def test_a_value_attached_to_a_short_option(self, command: str) -> None:
        assert await _held(_command(command), paths=[*PROTECTED, '.git/hooks']) == [command]

    async def test_ordinary_patterns_options_and_shells_still_run(self) -> None:
        commands = [
            'cp *.py dest/',
            "find . -name '*.py' -delete",
            'touch f{1..3}.txt',
            'mv config{,.bak}',
            'echo {} a,b} {x',
            'cp src/[a-z]?.txt .git/hooks-backup/',
            'echo x > notes/[a/b]',
            'echo x > notes/[unclosed',
            'sort -o out.txt in.txt',
            'cp -vt dest/ a.txt',
            "python -c 'print([1, 2])'",
            "git commit -m 'why?'",
            "sh build.sh; eval 'echo hi'",
        ]
        assert await _held(*map(_command, commands), paths=[*PROTECTED, '.git/hooks']) == []

    async def test_a_quoted_path_is_one_word(self) -> None:
        commands = ['echo x > "my dir/.pax/x"', 'echo x > my dir/.pax/x', 'tool --out=my/a']
        assert await _held(*map(_command, commands), paths=['my dir/.pax/*', 'my/a']) == [commands[0], commands[2]]

    async def test_windows_paths_resolve_against_the_working_directory(self) -> None:
        command = 'Set-Content ".pax\\my settings.json"'
        paths = ['C:\\Repo\\.pax\\my settings.json']
        assert await _held(_command(command), workspace=_WindowsRepo(), paths=paths) == [command]

    async def test_other_commands_name_no_protected_path(self) -> None:
        assert await _held(_command('echo ".pax is a folder" > notes.md')) == []
        assert await _held(_command('Set-Content notes.pax x'), workspace=_WindowsRepo()) == []

    async def test_case_is_ignored_on_posix_too(self) -> None:
        """macOS file systems ignore case by default, so `.PAX` there is `.pax`."""
        commands = ['cp x .PAX/settings.json', 'cp x /repo/.Git/Hooks/pre-commit']
        assert await _held(*map(_command, commands)) == commands

    async def test_a_very_long_command_counts_as_protected(self, tmp_path: Path) -> None:
        """Too long to split in reasonable time, though none of its short words is a protected path."""
        person = await _protected_run(tmp_path, ('run_command', {'command': 'echo' + ' x' * 40_000}))
        assert [asked.person_only for asked in person.asked] == [True]

    async def test_a_very_long_command_without_protection_is_not_held(self, tmp_path: Path) -> None:
        person = _Denier()
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


class _WindowsRepo:
    """A Windows backend working in `C:\\Repo`; the test tools never reach it."""

    ref: WorkspaceRef | None = None
    workspace_platform = 'windows'

    async def working_dir(self) -> str:
        return 'C:\\Repo'


class _NoRealpath:
    """A local backend whose links cannot be followed."""

    ref: WorkspaceRef | None = None

    def __init__(self, root: Path) -> None:
        self.root = root

    async def working_dir(self) -> str:
        return str(self.root)

    async def realpath(self, path: str) -> str:
        raise OSError('cannot follow')


MODES: tuple[PermissionMode, ...] = ('full', 'auto', 'read-only')
_LEVEL: dict[PermissionMode, int] = {'full': 0, 'auto': 1, 'read-only': 2}

WRITE = ('write_file', {'path': 'notes.txt', 'content': 'x'})
ECHO = ('run_command', {'command': 'echo hi'})
DELETE = ('run_command', {'command': 'rm -rf doomed'})
ASKED_FROM: dict[str, PermissionMode] = {'notes.txt': 'read-only', 'echo hi': 'read-only', 'rm -rf doomed': 'auto'}
"""The least strict mode that asks about each call."""


async def _approve_all(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults:
    return DeferredToolResults(approvals={call.tool_call_id: ToolApproved() for call in requests.approvals})


def _returns(messages: list[ModelMessage]) -> list[ToolReturnPart]:
    return [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]


def _approving_reviewer() -> ApprovalReviewer[None]:
    """An `ApprovalReviewer` that approves whatever it reviews."""
    return ApprovalReviewer[None](_safe_reviewer(), question='Could this call do harm?')


async def _run(
    tmp_path: Path,
    *calls: tuple[str, Mapping[str, object]],
    permissions: Permissions[None],
    person: _Person | None = None,
    extra: Sequence[AbstractCapability[None]] = (),
) -> _Person:
    """Run `calls` under `permissions`, asking `person`, who refuses every call unless given decisions."""
    person = person or _Person()
    agent = _agent(tmp_path, _calls(*calls), person=person, permissions=permissions, extra=extra)
    await agent.run('go', conversation_id='c')
    return person


class TestMaxMode:
    @pytest.mark.parametrize('max_mode', MODES)
    @pytest.mark.parametrize('mode', MODES)
    async def test_the_stricter_of_the_two_modes_asks(
        self, tmp_path: Path, mode: PermissionMode, max_mode: PermissionMode
    ) -> None:
        """What `max_mode` asks about is for a person only; what only the person's own mode asks about isn't."""
        person = await _run(tmp_path, WRITE, ECHO, DELETE, permissions=Permissions(mode, max_mode=max_mode))

        effective = max(_LEVEL[mode], _LEVEL[max_mode])
        assert [(asked.subject, asked.person_only) for asked in person.asked] == [
            (subject, _LEVEL[max_mode] >= _LEVEL[least])
            for subject, least in ASKED_FROM.items()
            if effective >= _LEVEL[least]
        ]

    async def test_the_reason_names_the_maximum_mode(self, tmp_path: Path) -> None:
        person = await _run(tmp_path, WRITE, ECHO, DELETE, permissions=Permissions('full', max_mode='read-only'))
        assert [asked.reason for asked in person.asked] == [
            "The maximum permission mode, 'read-only', asks before every file write.",
            "The maximum permission mode, 'read-only', asks before every command.",
            "The maximum permission mode, 'read-only', asks before every command.",
        ]
        person = await _run(tmp_path, DELETE, permissions=Permissions('full', max_mode='auto'))
        assert [asked.reason for asked in person.asked] == [
            "The maximum permission mode, 'auto', asks before commands that delete, push, install, or reach the "
            'network.'
        ]

    async def test_mode_for_is_held_to_max_mode_and_a_function_is_read_on_every_call(self) -> None:
        ceilings: list[PermissionMode] = ['auto', 'read-only', 'full']
        permissions = Permissions[None]('full', max_mode=lambda ctx: ceilings[ctx.run_step - 1])
        seen: list[PermissionMode] = []

        def mode(ctx: RunContext[None]) -> str:
            seen.append(permissions.mode_for(ctx))
            return 'ok'

        call: tuple[str, Mapping[str, object]] = ('mode', {})
        agent = Agent[None, str](_calls(call, call, call), deps_type=type(None), tools=[Tool(mode, takes_ctx=True)])
        await agent.run('go', capabilities=[permissions])
        assert seen == ['auto', 'read-only', 'full']

    async def test_a_rules_ask_under_it_is_still_for_a_person_only(self, tmp_path: Path) -> None:
        """The person's own ask rule doesn't turn the ceiling's hold into one a reviewer or memory may answer."""
        agent = Agent[None, 'str | DeferredToolRequests'](
            _calls(ECHO),
            deps_type=type(None),
            output_type=[str, DeferredToolRequests],
            capabilities=[
                LocalWorkspace(tmp_path),
                Shell(denied_commands=[]),
                Permissions('full', max_mode='read-only', rules=[PermissionRule('ask', command='echo')]),
                _approving_reviewer(),
            ],
        )
        held = await agent.run('go')
        assert isinstance(held.output, DeferredToolRequests)
        assert [(request.subject, request.person_only) for request in permission_requests(held.output)] == [
            ('echo hi', True)
        ]

    async def test_a_function_is_read_on_every_gated_call(self, tmp_path: Path) -> None:
        ceilings: list[PermissionMode] = ['full', 'read-only', 'full']
        person = await _run(
            tmp_path,
            ('run_command', {'command': 'echo one'}),
            ('run_command', {'command': 'echo two'}),
            ('run_command', {'command': 'echo three'}),
            permissions=Permissions('full', max_mode=lambda ctx: ceilings[ctx.run_step - 1]),
            person=_Person(PermissionDecision(approved=True)),
        )
        assert [(asked.subject, asked.person_only) for asked in person.asked] == [('echo two', True)]

    async def test_a_rules_allow_does_not_lift_a_call_over_it_and_a_managed_allow_does(self, tmp_path: Path) -> None:
        allow = [PermissionRule[None]('allow', command='echo')]
        person = await _run(tmp_path, ECHO, permissions=Permissions('full', max_mode='read-only', rules=allow))
        assert [(asked.subject, asked.person_only) for asked in person.asked] == [('echo hi', True)]

        person = await _run(tmp_path, ECHO, permissions=Permissions('full', max_mode='read-only', managed_rules=allow))
        assert person.asked == []

    async def test_a_managed_allow_does_not_loosen_the_persons_own_choices(self, tmp_path: Path) -> None:
        managed = [PermissionRule[None]('allow', command='echo')]
        stricter_mode = await _run(tmp_path, ECHO, permissions=Permissions('read-only', managed_rules=managed))
        own_ask = await _run(
            tmp_path,
            ECHO,
            permissions=Permissions(managed_rules=managed, rules=[PermissionRule('ask', command='echo')]),
        )
        assert [(asked.subject, asked.person_only) for asked in [*stricter_mode.asked, *own_ask.asked]] == [
            ('echo hi', False),
            ('echo hi', False),
        ]

        agent = Agent[None, str](
            _calls(ECHO),
            deps_type=type(None),
            capabilities=[
                LocalWorkspace(tmp_path),
                Shell(denied_commands=[]),
                Permissions(managed_rules=managed, rules=[PermissionRule('deny', command='echo')]),
            ],
        )
        assert (await agent.run('go')).output.startswith('A permission rule denied this command')

    async def test_protected_paths_ask_under_it_when_the_persons_mode_is_full(self, tmp_path: Path) -> None:
        (tmp_path / '.pax').mkdir()
        person = await _run(
            tmp_path,
            ('write_file', {'path': '.pax/settings.json', 'content': 'x'}),
            permissions=Permissions('full', max_mode='auto', protected_paths=['.pax/*']),
        )
        assert [(asked.subject, asked.person_only) for asked in person.asked] == [('.pax/settings.json', True)]
        assert not (tmp_path / '.pax' / 'settings.json').exists()

    async def test_another_permissions_on_the_agent_cannot_loosen_it(self, tmp_path: Path) -> None:
        """Each `Permissions` checks a call on its own, so a `'full'` one beside it changes nothing."""
        person = await _run(
            tmp_path, ECHO, permissions=Permissions('full', max_mode='read-only'), extra=[Permissions('full')]
        )
        assert [asked.subject for asked in person.asked] == ['echo hi']


class TestManagedRules:
    async def test_a_managed_deny_refuses_an_approved_call_whatever_the_rules_allow(self, tmp_path: Path) -> None:
        (tmp_path / '.pax').mkdir()
        agent = Agent[None, str](
            _calls(('write_file', {'path': '.pax/settings.json', 'content': 'x'})),
            deps_type=type(None),
            capabilities=[
                LocalWorkspace(tmp_path),
                FileSystem(),
                Permissions(
                    'read-only',
                    rules=[PermissionRule('allow', tools=['write_file'])],
                    managed_rules=[PermissionRule('deny', tools=['write_file'], reason='Settings are managed.')],
                    protected_paths=['.pax/*'],
                ),
                HandleDeferredToolCalls[None](handler=_approve_all),
            ],
        )
        result = await agent.run('go')
        assert result.output == (
            'A managed permission rule denied this file write, so it did not run. Do not retry it. '
            'Reason: Settings are managed.'
        )
        assert not (tmp_path / '.pax' / 'settings.json').exists()

    @pytest.mark.parametrize(
        'command',
        ["sh -c 'touch made'", 'bash -c "touch made"', "echo 'x'; eval 'touch made'", 'sh -c "touch made; $(true)"'],
    )
    async def test_a_managed_deny_inside_quotes_is_still_a_deny(self, tmp_path: Path, command: str) -> None:
        """A person's own deny matched only in quoted text asks, as it may be data; a managed one stays a deny."""
        person = _Person(PermissionDecision(approved=True))
        await _run(
            tmp_path,
            ('run_command', {'command': command}),
            permissions=Permissions(managed_rules=[PermissionRule('deny', command='touch')]),
            person=person,
        )
        assert person.asked == []
        assert not (tmp_path / 'made').exists()

    async def test_a_persons_deny_inside_quotes_still_asks(self, tmp_path: Path) -> None:
        person = _Person(PermissionDecision(approved=True))
        await _run(
            tmp_path,
            ('run_command', {'command': "sh -c 'touch made'"}),
            permissions=Permissions(rules=[PermissionRule('deny', command='touch')]),
            person=person,
        )
        assert [request.subject for request in person.asked] == ["sh -c 'touch made'"]
        assert (tmp_path / 'made').exists()

    async def test_every_deny_is_checked_before_a_managed_ask(self, tmp_path: Path) -> None:
        """Approving a managed ask must not run a call the person's own rule denies."""
        person = _Person(PermissionDecision(approved=True))
        await _run(
            tmp_path,
            ECHO,
            permissions=Permissions(
                managed_rules=[PermissionRule('ask', command='echo')], rules=[PermissionRule('deny', command='echo')]
            ),
            person=person,
        )
        assert person.asked == []

    async def test_a_managed_ask_holds_what_a_rule_allows_and_no_reviewer_answers_it(self, tmp_path: Path) -> None:
        """A reviewer that says every call is safe approves the `rm`, but the managed ask goes to the person."""
        (tmp_path / 'doomed').mkdir()
        person = await _run(
            tmp_path,
            ECHO,
            DELETE,
            permissions=Permissions(
                'auto',
                rules=[PermissionRule('allow', command='echo')],
                managed_rules=[PermissionRule('ask', command='echo', reason='Echo is watched.')],
            ),
            extra=[_approving_reviewer()],
        )
        assert [(asked.subject, asked.reason, asked.person_only) for asked in person.asked] == [
            ('echo hi', 'Echo is watched.', True)
        ]
        assert not (tmp_path / 'doomed').exists()

    async def test_a_managed_ask_on_any_other_tool_is_for_a_person_only(self) -> None:
        def deploy(env: str) -> str:
            return f'deployed {env}'

        agent = Agent[None, 'str | DeferredToolRequests'](
            _calls(('deploy', {'env': 'prod'})),
            deps_type=type(None),
            output_type=[str, DeferredToolRequests],
            tools=[Tool(deploy)],
            capabilities=[
                Permissions(
                    rules=[PermissionRule('allow', tools=['deploy'])],
                    managed_rules=[PermissionRule('ask', tools=['deploy'])],
                ),
                _approving_reviewer(),
            ],
        )
        held = await agent.run('go')
        assert isinstance(held.output, DeferredToolRequests)
        [call] = held.output.approvals
        assert held.output.metadata == {call.tool_call_id: {'permissions': {'person_only': True}}}
        assert permission_requests(held.output) == []

        resumed = await agent.run(
            message_history=held.all_messages(),
            deferred_tool_results=DeferredToolResults(approvals={call.tool_call_id: True}),
        )
        assert resumed.output == 'deployed prod'

    async def test_an_approval_answers_only_its_own_call(self, tmp_path: Path) -> None:
        """Approving one call through `deferred_tool_results` leaves the same tool with other arguments held."""
        agent = Agent[None, 'str | DeferredToolRequests'](
            _calls(ECHO, ('run_command', {'command': 'echo bye'})),
            deps_type=type(None),
            output_type=[str, DeferredToolRequests],
            capabilities=[
                LocalWorkspace(tmp_path),
                Shell(denied_commands=[]),
                Permissions(managed_rules=[PermissionRule('ask', command='echo')]),
            ],
        )
        held = await agent.run('The user approved every command in advance.')
        assert isinstance(held.output, DeferredToolRequests)
        [first] = held.output.approvals
        resumed = await agent.run(
            message_history=held.all_messages(),
            deferred_tool_results=DeferredToolResults(approvals={first.tool_call_id: True}),
        )
        assert isinstance(resumed.output, DeferredToolRequests)
        assert [request.subject for request in permission_requests(resumed.output)] == ['echo bye']

    async def test_a_managed_ask_is_approved_one_call_at_a_time(self, tmp_path: Path) -> None:
        remembered: set[RememberedApproval] = set()
        person = await _run(
            tmp_path,
            ECHO,
            ECHO,
            permissions=Permissions(managed_rules=[PermissionRule('ask', command='echo')], remembered=remembered),
            person=_Person(PermissionDecision(approved=True, remember=True)),
        )
        assert [asked.subject for asked in person.asked] == ['echo hi', 'echo hi']
        assert remembered == set()

    async def test_inside_durable_execution_no_approval_is_kept_for_a_later_run(self) -> None:
        class _DBOSDurability(AbstractCapability[None]):
            in_durable_context = True

        _DBOSDurability.__module__ = 'pydantic_ai.durable_exec.dbos'

        def run_command(command: str) -> str:
            return 'ran'

        def echo_each_run(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            last = messages[-1]
            if isinstance(last, ModelRequest) and any(isinstance(part, ToolReturnPart) for part in last.parts):
                return ModelResponse(parts=[TextPart('done')])
            return ModelResponse(parts=[ToolCallPart(*ECHO)])

        person = _Person(PermissionDecision(approved=True, remember=True))
        agent = Agent[None, str](
            FunctionModel(echo_each_run),
            deps_type=type(None),
            tools=[Tool(run_command, metadata={'code_arg_language': 'shell', 'code_arg_name': 'command'})],
            capabilities=[
                _DBOSDurability(),
                Permissions(managed_rules=[PermissionRule('ask', command='echo')]),
                HandleDeferredToolCalls[None](handler=lambda ctx, requests: ask_permissions(requests, person)),
            ],
        )
        first = await agent.run('go', conversation_id='c')
        assert [part.metadata for part in _returns(first.all_messages())] == [None]
        await agent.run('again', message_history=first.all_messages(), conversation_id='c')
        assert len(person.asked) == 2

    async def test_a_delegate_is_held_and_the_parents_approval_answers_none_of_its_calls(self, tmp_path: Path) -> None:
        (tmp_path / 'doomed').mkdir()
        person = _Person()

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            last = messages[-1]
            if isinstance(last, ModelRequest) and any(isinstance(part, ToolReturnPart) for part in last.parts):
                return ModelResponse(parts=[TextPart('done')])
            if 'delegate_task' in [tool.name for tool in info.function_tools]:
                return ModelResponse(parts=[ToolCallPart('delegate_task', {'agent_name': 'self', 'task': 'clean up'})])
            return ModelResponse(parts=[ToolCallPart('run_command', {'command': 'rm -rf doomed'})])

        permissions = Permissions[None](
            managed_rules=[PermissionRule('ask', tools=['delegate_task']), PermissionRule('ask', command='rm')]
        )

        async def approve_delegations(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults:
            """Approves the plain approval of the delegate call, and asks the person about what `Permissions` holds."""
            asked = await ask_permissions(requests, person)
            approvals = {call.tool_call_id: ToolApproved() for call in requests.approvals}
            if asked is None:
                return DeferredToolResults(approvals=dict(approvals))
            return DeferredToolResults(approvals={**approvals, **asked.approvals}, metadata=asked.metadata)

        handler = HandleDeferredToolCalls[None](handler=approve_delegations)
        agent = Agent[None, str](
            FunctionModel(respond),
            deps_type=type(None),
            capabilities=[
                LocalWorkspace(tmp_path),
                Shell(denied_commands=[]),
                SubAgents(
                    include_self=True, max_depth=2, agent_folders=None, shared_capabilities=[permissions, handler]
                ),
            ],
        )
        await agent.run('go', capabilities=[permissions, handler])
        assert [request.subject for request in person.asked] == ['rm -rf doomed']
        assert (tmp_path / 'doomed').is_dir()

    def test_from_spec_takes_the_managed_fields(self) -> None:
        permissions = Permissions[None].from_spec(
            'full', max_mode='auto', managed_rules=[{'verdict': 'deny', 'command': 'git push'}]
        )
        assert (permissions.max_mode, permissions.managed_rules) == (
            'auto',
            [PermissionRule[None]('deny', command='git push')],
        )


def _read_only(ctx: RunContext[None]) -> PermissionMode:
    return 'read-only'


class TestReviewerApprovals:
    @pytest.mark.parametrize('max_mode', ['read-only', _read_only])
    async def test_a_reviewers_approval_of_an_earlier_hold_does_not_answer_a_person_only_one(
        self, tmp_path: Path, max_mode: PermissionMode | Callable[[RunContext[None]], PermissionMode]
    ) -> None:
        """The first `Permissions` holds the call for anyone and the reviewer approves it; the ceiling holds it again.

        The run's handlers have answered this round, so the call ends the run held, for a person only.
        """
        agent = Agent[None, 'str | DeferredToolRequests'](
            _calls(ECHO),
            deps_type=type(None),
            output_type=[str, DeferredToolRequests],
            capabilities=[
                LocalWorkspace(tmp_path),
                Shell(denied_commands=[]),
                Permissions('read-only'),
                Permissions('full', max_mode=max_mode),
                _approving_reviewer(),
            ],
        )
        held = await agent.run('go')
        assert isinstance(held.output, DeferredToolRequests)
        assert [(request.subject, request.person_only) for request in permission_requests(held.output)] == [
            ('echo hi', True)
        ]
        assert _returns(held.all_messages()) == []

    async def test_a_managed_ask_on_a_plain_tool_with_requires_approval_goes_past_the_reviewer(self) -> None:
        """Held before core defers the tool, the request says `person_only`, so the reviewer leaves it to the person."""
        asked: list[object] = []

        def deploy(env: str) -> str:
            return f'deployed {env}'

        async def person(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults:
            asked.extend(requests.metadata.values())
            return await _approve_all(ctx, requests)

        agent = Agent[None, str](
            _calls(('deploy', {'env': 'prod'})),
            deps_type=type(None),
            tools=[Tool(deploy, requires_approval=True)],
            capabilities=[
                Permissions(managed_rules=[PermissionRule('ask', tools=['deploy'])]),
                _approving_reviewer(),
                HandleDeferredToolCalls[None](handler=person),
            ],
        )
        assert (await agent.run('go')).output == 'deployed prod'
        assert asked == [{'permissions': {'person_only': True}}]

    @pytest.mark.parametrize(
        'permissions',
        [
            Permissions[None](managed_rules=[PermissionRule('ask', command='echo')]),
            Permissions[None](max_mode='read-only'),
        ],
        ids=['managed-ask', 'max-mode'],
    )
    async def test_a_person_only_hold_on_a_command_with_requires_approval_goes_to_the_person(
        self, permissions: Permissions[None]
    ) -> None:
        person = _Person(PermissionDecision(approved=True))
        agent = Agent[None, str](
            _calls(ECHO),
            deps_type=type(None),
            tools=[_SHELL_TOOL_REQUIRING_APPROVAL],
            capabilities=[permissions, _approving_reviewer(), HandleDeferredToolCalls[None](approval_ask(person))],
        )
        assert (await agent.run('go')).output == 'ran'
        assert [(request.subject, request.person_only) for request in person.asked] == [('echo hi', True)]

    async def test_an_approval_from_any_other_handler_counts_as_the_persons(self, tmp_path: Path) -> None:
        """Only `ApprovalReviewer` marks its approvals; a handler that approves in code is trusted as the person."""
        agent = Agent[None, str](
            _calls(ECHO),
            deps_type=type(None),
            capabilities=[
                LocalWorkspace(tmp_path),
                Shell(denied_commands=[]),
                Permissions(managed_rules=[PermissionRule('ask', command='echo')]),
                HandleDeferredToolCalls[None](handler=_approve_all),
            ],
        )
        assert (await agent.run('go')).output == '[stdout]\nhi\n'

    async def test_a_quoted_managed_deny_refuses_a_protected_command_instead_of_holding_it(
        self, tmp_path: Path
    ) -> None:
        """The protected command ask also matches inside the quotes, but no one is asked about a call that is refused."""
        person = await _run(
            tmp_path,
            ('run_command', {'command': "sh -c 'touch made'"}),
            permissions=Permissions(
                'auto', protected_commands=['touch'], managed_rules=[PermissionRule('deny', command='touch')]
            ),
            person=_Person(PermissionDecision(approved=True)),
        )
        assert person.asked == []
        assert not (tmp_path / 'made').exists()

    async def test_a_managed_deny_refuses_a_protected_write_instead_of_holding_it(self, tmp_path: Path) -> None:
        agent = Agent[None, 'str | DeferredToolRequests'](
            _calls(('write_file', {'path': '.pax/settings.json', 'content': 'x'})),
            deps_type=type(None),
            output_type=[str, DeferredToolRequests],
            capabilities=[
                LocalWorkspace(tmp_path),
                FileSystem(),
                Permissions(
                    'auto', managed_rules=[PermissionRule('deny', tools=['write_file'])], protected_paths=['.pax/*']
                ),
            ],
        )
        assert (await agent.run('go')).output == (
            'A managed permission rule denied this file write, so it did not run. Do not retry it.'
        )
