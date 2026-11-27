"""Tests for `Permissions`."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import pytest

from pydantic_ai import Agent, DeferredToolRequests, Tool
from pydantic_ai.capabilities import AbstractCapability, HandleDeferredToolCalls, LocalWorkspace
from pydantic_ai.exceptions import UserError
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
from pydantic_ai.tools import DeferredToolResults, RunContext, ToolApproved
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.workspaces import CommandResult, Workspace, WorkspaceCommand, WorkspaceRef
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
    parse_permission_rule,
    parse_permission_rules,
    permission_requests,
)
from pydantic_ai_harness.shell import Shell
from pydantic_ai_harness.sub_agents import SubAgents


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
    """Answers each request with the next decision, and keeps what it was asked."""

    def __init__(self, *decisions: PermissionDecision) -> None:
        self.decisions = list(decisions)
        self.asked: list[PermissionRequest] = []

    async def __call__(self, request: PermissionRequest) -> PermissionDecision:
        self.asked.append(request)
        return self.decisions.pop(0)


def _agent(
    tmp_path: Path,
    model: FunctionModel,
    mode: PermissionMode | Callable[[RunContext[None]], PermissionMode],
    person: _Person | None = None,
    *,
    rules: Sequence[PermissionRule[None]] = (),
) -> Agent[None, str | DeferredToolRequests]:
    handler = [] if person is None else [HandleDeferredToolCalls[None](approval_ask(person))]
    return Agent[None, 'str | DeferredToolRequests'](
        model,
        deps_type=type(None),
        output_type=[str, DeferredToolRequests],
        capabilities=[
            LocalWorkspace(tmp_path),
            FileSystem(),
            Shell(denied_commands=[]),
            Permissions(mode, rules=rules),
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
        monkeypatch.setattr('pydantic_ai_harness.permissions._rules._COMMAND_LIMIT', 8)

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


class TestParsePermissionRule:
    @pytest.mark.parametrize(
        ('rule', 'verdict', 'expected'),
        [
            ('Bash(git push:*)', 'allow', PermissionRule('allow', command='git push')),
            ('Shell(git push:*)', 'deny', PermissionRule('deny', command='git push')),
            ('  Bash( git push :* )  ', 'allow', PermissionRule('allow', command='git push')),
            ('Bash(git push)', 'ask', PermissionRule('ask', command='git push')),
            ('Bash(rm -rf)', 'deny', PermissionRule('deny', command='rm -rf')),
            ('Bash(echo (x):*)', 'allow', PermissionRule('allow', command='echo (x)')),
            ('Bash', 'ask', PermissionRule('ask', tools={'code_arg_language': 'shell'})),
            ('write_file', 'deny', PermissionRule('deny', tools=['write_file'])),
            ('write_file(docs/**)', 'allow', PermissionRule('allow', tools=['write_file'], args={'path': 'docs/**'})),
            ('write_file(a=b/c)', 'deny', PermissionRule('deny', tools=['write_file'], args={'a': 'b/c'})),
            ('write_file(docs/a=b)', 'deny', PermissionRule('deny', tools=['write_file'], args={'path': 'docs/a=b'})),
            (
                'web_fetch(url=https://example.com/*)',
                'ask',
                PermissionRule('ask', tools=['web_fetch'], args={'url': 'https://example.com/*'}),
            ),
            ('mcp__github__create_issue', 'ask', PermissionRule('ask', tools=['mcp__github__create_issue'])),
            ('read', 'deny', PermissionRule('deny', tools=['read'])),
        ],
    )
    def test_compiles_to_a_rule(self, rule: str, verdict: PermissionVerdict, expected: PermissionRule[None]) -> None:
        assert parse_permission_rule(rule, verdict) == expected

    def test_keeps_the_reason(self) -> None:
        assert parse_permission_rule('Bash(rm -rf)', 'deny', reason='No recursive deletes.') == PermissionRule(
            'deny', command='rm -rf', reason='No recursive deletes.'
        )

    def test_an_exact_command_cannot_be_allowed(self) -> None:
        with pytest.raises(UserError, match=r'write `Bash\(git status:\*\)` to allow them'):
            parse_permission_rule('Bash(git status)', 'allow')

    @pytest.mark.parametrize(
        ('name', 'instead'),
        [
            ('Read', 'read_file'),
            ('Edit', 'edit_file'),
            ('MultiEdit', 'edit_file'),
            ('Write', 'write_file'),
            ('Glob', 'find_files'),
            ('Grep', 'grep'),
            ('WebFetch', 'web_fetch'),
            ('WebSearch', 'web_search'),
        ],
    )
    def test_claude_code_tool_names_name_the_tool_to_use(self, name: str, instead: str) -> None:
        with pytest.raises(UserError, match=rf"Claude Code tool name '{name}': name `{instead}` instead"):
            parse_permission_rule(f'{name}(.env)', 'deny')
        with pytest.raises(UserError, match=rf"Claude Code tool name '{name}'"):
            parse_permission_rule(name, 'deny')

    @pytest.mark.parametrize(
        'rule', ['', '   ', 'write_file()', 'Bash(  )', 'write file', 'write_file(docs', '(x)', 'write_file)']
    )
    def test_malformed_strings_raise(self, rule: str) -> None:
        with pytest.raises(UserError, match=re.escape(f'Permission rule {rule!r} must be a tool name')):
            parse_permission_rule(rule, 'deny')

    @pytest.mark.parametrize('rule', ['shell(rm -rf:*)', 'run_command(git push:*)', 'bash(rm:*)'])
    def test_a_command_pattern_on_another_name_raises(self, rule: str) -> None:
        with pytest.raises(UserError, match=re.escape(f'Permission rule {rule!r} uses `:*`')):
            parse_permission_rule(rule, 'deny')

    @pytest.mark.parametrize('rule', ['bash(rm -rf)', 'SHELL(rm -rf)', 'run_command(rm -rf)', 'shell(rm)'])
    def test_a_path_pattern_on_a_shell_tool_name_raises(self, rule: str) -> None:
        with pytest.raises(UserError, match=re.escape(f'Permission rule {rule!r} matches a path')):
            parse_permission_rule(rule, 'deny')

    def test_a_named_argument_on_a_shell_tool_name_stays(self) -> None:
        assert parse_permission_rule('run_command(command=rm *)', 'deny') == PermissionRule(
            'deny', tools=['run_command'], args={'command': 'rm *'}
        )

    @pytest.mark.parametrize(
        ('rule', 'args'),
        [
            ('web_fetch(url=http://localhost:*)', {'url': 'http://localhost:*'}),
            ('deploy(target=prod:*)', {'target': 'prod:*'}),
            ('web_fetch(url = https://example.com/*)', {'url': 'https://example.com/*'}),
        ],
    )
    def test_a_named_argument_glob_matches_as_written(self, rule: str, args: dict[str, str]) -> None:
        assert parse_permission_rule(rule, 'deny').args == args

    def test_text_before_an_equals_sign_that_is_no_argument_name_is_part_of_the_path(self) -> None:
        assert parse_permission_rule('write_file(9=x)', 'deny') == PermissionRule(
            'deny', tools=['write_file'], args={'path': '9=x'}
        )

    @pytest.mark.parametrize('verdict', ['allow', 'deny'])
    def test_a_trailing_space_star_is_the_same_as_colon_star(self, verdict: PermissionVerdict) -> None:
        assert parse_permission_rule('Bash(git push *)', verdict) == parse_permission_rule('Bash(git push:*)', verdict)
        assert parse_permission_rule('Bash(*)', verdict) == parse_permission_rule('Bash', verdict)

    def test_a_star_inside_a_command_raises(self) -> None:
        with pytest.raises(UserError, match='has a `\\*` that is not at its end'):
            parse_permission_rule('Bash(git * main)', 'deny')

    @pytest.mark.parametrize('rule', ['read_file(./.env)', 'read_file(././.env)', 'read_file(path=./.env)'])
    def test_a_leading_dot_slash_is_dropped(self, rule: str) -> None:
        assert parse_permission_rule(rule, 'deny') == PermissionRule('deny', tools=['read_file'], args={'path': '.env'})

    @pytest.mark.parametrize('rule', ['list_dir(./)', 'list_dir(**/)', 'list_dir(path=./)'])
    def test_a_path_of_only_dot_slash_is_everything(self, rule: str) -> None:
        assert parse_permission_rule(rule, 'deny').args == {'path': '**'}

    @pytest.mark.parametrize(
        ('rule', 'hint'),
        [
            ('write_file(secrets/)', 'write `secrets/\\*\\*` .* add `secrets`'),
            ('read_file(./.git/)', 'write `.git/\\*\\*` .* add `.git`'),
            ('read_file(**/.git/)', 'write `.git/\\*\\*` .* add `.git`'),
        ],
    )
    def test_a_trailing_slash_raises(self, rule: str, hint: str) -> None:
        """`secrets/**` misses a call on `secrets` itself, so a directory rule asks for both globs instead."""
        with pytest.raises(UserError, match=f'ends its path with `/`: {hint}'):
            parse_permission_rule(rule, 'deny')

    async def test_a_directory_and_what_is_under_it_are_both_denied(self, tmp_path: Path) -> None:
        (tmp_path / 'secrets').mkdir()
        (tmp_path / 'secrets' / 'key').write_text('x')
        rules = parse_permission_rules({'deny': ['list_directory(secrets)', 'list_directory(secrets/**)']})
        calls = [('list_directory', {'path': 'secrets'}), ('list_directory', {'path': 'secrets/inner'})]
        for call in calls:
            result = await _agent(tmp_path, _calls(call), 'full', rules=rules).run('go')
            assert result.output == 'A permission rule denied this tool call, so it did not run. Do not retry it.'

    def test_other_arguments_keep_their_glob_as_written(self) -> None:
        assert parse_permission_rule('web_fetch(url=./x)', 'deny').args == {'url': './x'}

    @pytest.mark.parametrize('rule', ['read_file(/secrets/**)', 'read_file(~/.ssh/*)', 'read_file(path=/etc/*)'])
    def test_an_absolute_or_home_path_raises(self, rule: str) -> None:
        with pytest.raises(UserError, match='starts its path with'):
            parse_permission_rule(rule, 'deny')

    @pytest.mark.parametrize('rule', ['mcp__github__*', 'mcp__github__create_?', 'tool[s]', 'mcp__*(x)'])
    def test_a_wildcard_in_a_tool_name_raises(self, rule: str) -> None:
        with pytest.raises(UserError, match='has a wildcard in its tool name'):
            parse_permission_rule(rule, 'deny')

    @pytest.mark.parametrize('rule', ['mcp__github', 'mcp__github(x)'])
    def test_a_claude_code_mcp_server_rule_raises(self, rule: str) -> None:
        with pytest.raises(UserError, match='names an MCP server the way Claude Code does'):
            parse_permission_rule(rule, 'deny')

    def test_a_full_mcp_tool_name_stays_literal(self) -> None:
        assert parse_permission_rule('mcp__github__create_issue', 'deny') == PermissionRule(
            'deny', tools=['mcp__github__create_issue']
        )

    @pytest.mark.parametrize(
        ('rule', 'glob'), [('read_file(**/.env)', '.env'), ('read_file(**/**/secrets/*)', 'secrets/*')]
    )
    def test_a_leading_double_star_is_dropped_for_ask_and_deny(self, rule: str, glob: str) -> None:
        assert parse_permission_rule(rule, 'deny').args == {'path': glob}
        assert parse_permission_rule(rule, 'ask').args == {'path': glob}

    def test_an_allow_keeps_its_leading_double_star(self) -> None:
        """Kept, it allows less, not more: a call at the top isn't allowed, so it asks."""
        assert parse_permission_rule('read_file(**/.env)', 'allow').args == {'path': '**/.env'}
        assert parse_permission_rule('read_file(./**/.env)', 'allow').args == {'path': '**/.env'}

    def test_an_unknown_verdict_raises(self) -> None:
        with pytest.raises(UserError, match="has the verdict 'denny'"):
            parse_permission_rule('read_file', 'denny')  # pyright: ignore[reportArgumentType]

    def test_an_empty_command_raises(self) -> None:
        with pytest.raises(UserError, match='must be a shell command'):
            parse_permission_rule('Bash(:*)', 'allow')


class TestParsePermissionRules:
    def test_parses_each_verdict(self) -> None:
        assert parse_permission_rules(
            {
                'allow': ['Bash(git status:*)', 'write_file(docs/**)'],
                'ask': ['Bash(git push)'],
                'deny': ['Bash(rm -rf)'],
            }
        ) == [
            PermissionRule('allow', command='git status'),
            PermissionRule('allow', tools=['write_file'], args={'path': 'docs/**'}),
            PermissionRule('ask', command='git push'),
            PermissionRule('deny', command='rm -rf'),
        ]

    def test_an_unknown_key_raises(self) -> None:
        with pytest.raises(UserError, match="has the verdict 'denny'"):
            parse_permission_rules({'denny': ['read_file']})  # pyright: ignore[reportArgumentType]
        with pytest.raises(UserError, match="has the verdict 'denny'"):
            parse_permission_rules({'denny': []})  # pyright: ignore[reportArgumentType]

    @pytest.mark.parametrize('strings', ['read_file', [42], [None], None])
    def test_anything_but_a_list_of_strings_raises(self, strings: object) -> None:
        with pytest.raises(UserError, match="for 'deny' must be a list of strings"):
            parse_permission_rules({'deny': strings})  # pyright: ignore[reportArgumentType]

    async def test_a_recursive_deny_refuses_a_file_at_the_top(self, tmp_path: Path) -> None:
        (tmp_path / 'app').mkdir()
        for path in (tmp_path / '.env', tmp_path / 'app' / '.env'):
            path.write_text('SECRET=1')
        rules = parse_permission_rules({'deny': ['read_file(**/.env)']})
        for path in ('.env', 'app/.env', str(tmp_path / '.env')):
            result = await _agent(tmp_path, _calls(('read_file', {'path': path})), 'read-only', rules=rules).run('go')
            assert result.output == 'A permission rule denied this tool call, so it did not run. Do not retry it.'

    async def test_rules_decide_real_shell_and_file_calls(self, tmp_path: Path) -> None:
        (tmp_path / 'docs').mkdir()
        (tmp_path / 'kept').mkdir()
        rules = parse_permission_rules(
            {'allow': ['Bash(echo:*)', 'write_file(docs/**)'], 'deny': ['Bash(rm -rf)', 'write_file(secrets/*)']}
        )

        ran = await _agent(tmp_path, _calls(('run_command', {'command': 'echo hi'})), 'read-only', rules=rules).run(
            'go'
        )
        assert isinstance(ran.output, str) and 'hi' in ran.output

        denied = await _agent(
            tmp_path, _calls(('run_command', {'command': 'rm -rf kept'})), 'read-only', rules=rules
        ).run('go')
        assert denied.output == 'A permission rule denied this command, so it did not run. Do not retry it.'
        assert (tmp_path / 'kept').is_dir()

        await _agent(
            tmp_path, _calls(('write_file', {'path': 'docs/a.md', 'content': 'x'})), 'read-only', rules=rules
        ).run('go')
        assert (tmp_path / 'docs' / 'a.md').read_text() == 'x'

        refused = await _agent(
            tmp_path, _calls(('write_file', {'path': 'secrets/key', 'content': 'x'})), 'read-only', rules=rules
        ).run('go')
        assert refused.output == 'A permission rule denied this file write, so it did not run. Do not retry it.'

        held = await _agent(
            tmp_path, _calls(('write_file', {'path': 'notes.md', 'content': 'x'})), 'read-only', rules=rules
        ).run('go')
        assert isinstance(held.output, DeferredToolRequests)
