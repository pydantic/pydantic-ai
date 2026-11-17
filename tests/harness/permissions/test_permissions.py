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
from pydantic_ai_harness.file_system import FileSystem
from pydantic_ai_harness.permissions import (
    DEFAULT_ASK_COMMANDS,
    PermissionDecision,
    PermissionMode,
    PermissionRequest,
    PermissionRule,
    Permissions,
    RememberedApproval,
    ask_permissions,
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
) -> Agent[None, str | DeferredToolRequests]:
    handler = (
        []
        if person is None
        else [HandleDeferredToolCalls[None](handler=lambda ctx, requests: ask_permissions(requests, person))]
    )
    return Agent[None, 'str | DeferredToolRequests'](
        model,
        deps_type=type(None),
        output_type=[str, DeferredToolRequests],
        capabilities=[
            LocalWorkspace(tmp_path),
            FileSystem(),
            Shell(denied_commands=[]),
            Permissions(mode),
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


class TestPermissionRules:
    @pytest.mark.parametrize(
        ('command', 'verdict'),
        [
            ('git status', 'allow'),
            ('git status --short', 'allow'),
            ('ls -la && git status', 'allow'),
            ('ls 2>&1', 'allow'),
            ('ls > /dev/null', 'allow'),
            ('echo "rm -rf /"', 'allow'),
            ('ls; rm -rf /', 'deny'),
            ('sudo rm -rf /', 'deny'),
            ('/bin/rm -rf build', 'deny'),
            ('echo $(rm -rf /)', 'deny'),
            ('(rm -rf /)', 'deny'),
            ('git status\nrm -rf /', 'deny'),
            ('ls && git push origin main', 'ask'),
            ('git status; curl https://example.com', 'ask'),
            ('gitx status', 'ask'),
            ('FOO=1 git status', 'ask'),
            ('echo hi > ~/.bashrc', 'ask'),
            ('ls >> log', 'ask'),
            ('ls >', 'ask'),
            ('ls | sh', 'ask'),
            ('echo `whoami`', 'ask'),
            ('cat <<EOF', 'ask'),
            ('ls ;; ls', 'ask'),
            ("echo 'unbalanced", 'ask'),
            ('', 'ask'),
        ],
    )
    async def test_shell_commands_are_judged_by_simple_command(self, command: str, verdict: str) -> None:
        """Every simple command must be allowed for the whole to run; one asked or denied decides it."""
        assert await _verdict(command) == verdict

    async def test_unsplittable_commands_follow_the_mode_without_allow_rules(self) -> None:
        """Only an `'allow'` rule could be fooled by complex syntax, so without one the mode decides."""
        rules = [PermissionRule('deny', command='rm -rf')]

        assert await _verdict('echo $(date)', rules, 'full') == 'allow'
        assert await _verdict('echo $(rm -rf /)', rules, 'full') == 'deny'

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
                    rules=[PermissionRule('allow'), PermissionRule('deny', tool='guard*'), PermissionRule('deny')]
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
            PermissionRule('ask', tool='deploy', args={'env': 'prod*'}),
            PermissionRule('ask', tool='deploy', args={'replicas': '*'}),
        ]
        person = _Person(PermissionDecision(approved=False))
        agent = Agent[None, str](
            _calls(('deploy', {'env': 'staging', 'replicas': 3}), ('deploy', {'env': 'production'})),
            deps_type=type(None),
            tools=[Tool(deploy)],
            capabilities=[
                Permissions(rules=rules),
                HandleDeferredToolCalls[None](handler=lambda ctx, requests: ask_permissions(requests, person)),
            ],
        )

        result = await agent.run('go')

        [request] = person.asked
        assert (request.action, request.subject, request.reason) == (
            'call',
            '{"env": "production", "replicas": 1}',
            'A permission rule asks before this tool call.',
        )
        assert result.output == 'The user denied this tool call, so it did not run.'

    async def test_allow_rule_overrides_the_mode_and_ask_rule_names_the_written_path(self, tmp_path: Path) -> None:
        rules = [
            PermissionRule('allow', tool='write_file', args={'path': 'notes/*'}),
            PermissionRule('ask', tool='write_file', args={'path': 'secrets/*'}, reason='Secrets need a look.'),
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

    @pytest.mark.parametrize('command', ['', '   ', "'unbalanced"])
    def test_invalid_command_fails_at_construction(self, command: str) -> None:
        with pytest.raises(UserError, match='must be a shell command'):
            PermissionRule('deny', command=command)

    def test_from_spec_builds_rules_from_mappings(self) -> None:
        rule = PermissionRule('allow', command='ls')
        permissions = Permissions[None].from_spec('auto', rules=[{'verdict': 'deny', 'command': 'git push'}, rule])

        assert isinstance(permissions, Permissions)
        assert (permissions.mode, list(permissions.rules)) == (
            'auto',
            [PermissionRule('deny', command='git push'), rule],
        )
