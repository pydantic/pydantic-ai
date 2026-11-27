"""Tests for `Permissions.max_mode` and `Permissions.managed_rules`: limits the person running an agent can't loosen."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import pytest

from pydantic_ai import Agent, DeferredToolRequests, Tool
from pydantic_ai.capabilities import AbstractCapability, HandleDeferredToolCalls, LocalWorkspace
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import DeferredToolResults, RunContext, ToolApproved
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
from pydantic_ai_harness.shell import Shell
from pydantic_ai_harness.sub_agents import SubAgents

MODES: tuple[PermissionMode, ...] = ('full', 'auto', 'read-only')
_LEVEL: dict[PermissionMode, int] = {'full': 0, 'auto': 1, 'read-only': 2}

WRITE = ('write_file', {'path': 'notes.txt', 'content': 'x'})
ECHO = ('run_command', {'command': 'echo hi'})
DELETE = ('run_command', {'command': 'rm -rf doomed'})
ASKED_FROM: dict[str, PermissionMode] = {'notes.txt': 'read-only', 'echo hi': 'read-only', 'rm -rf doomed': 'auto'}
"""The least strict mode that asks about each call."""


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
    """Answers every request the same way, and keeps what it was asked."""

    def __init__(self, decision: PermissionDecision = PermissionDecision(approved=False)) -> None:
        self.decision = decision
        self.asked: list[PermissionRequest] = []

    async def __call__(self, request: PermissionRequest) -> PermissionDecision:
        self.asked.append(request)
        return self.decision


def _safe_reviewer() -> ApprovalReviewer[None]:
    """A reviewer whose model says every call is safe, so it approves whatever it reviews."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {'response': 'no'})])

    return ApprovalReviewer[None](FunctionModel(respond), question='Could this call do harm?')


async def _approve_all(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults:
    return DeferredToolResults(approvals={call.tool_call_id: ToolApproved() for call in requests.approvals})


def _returns(messages: list[ModelMessage]) -> list[ToolReturnPart]:
    return [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]


async def _run(
    tmp_path: Path,
    *calls: tuple[str, Mapping[str, object]],
    permissions: Permissions[None],
    person: _Person | None = None,
    extra: Sequence[AbstractCapability[None]] = (),
) -> _Person:
    person = person or _Person()
    agent = Agent[None, 'str | DeferredToolRequests'](
        _calls(*calls),
        deps_type=type(None),
        output_type=[str, DeferredToolRequests],
        capabilities=[
            LocalWorkspace(tmp_path),
            FileSystem(),
            Shell(denied_commands=[]),
            permissions,
            *extra,
            HandleDeferredToolCalls[None](approval_ask(person)),
        ],
    )
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
                _safe_reviewer(),
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
            extra=[_safe_reviewer()],
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
                _safe_reviewer(),
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
                _safe_reviewer(),
            ],
        )
        held = await agent.run('go')
        assert isinstance(held.output, DeferredToolRequests)
        assert [(request.subject, request.person_only) for request in permission_requests(held.output)] == [
            ('echo hi', True)
        ]
        assert _returns(held.all_messages()) == []

    async def test_a_reviewers_approval_does_not_answer_a_managed_ask_on_a_plain_tool(self) -> None:
        def deploy(env: str) -> str:
            raise AssertionError('the held call must not run')  # pragma: no cover

        agent = Agent[None, 'str | DeferredToolRequests'](
            _calls(('deploy', {'env': 'prod'})),
            deps_type=type(None),
            output_type=[str, DeferredToolRequests],
            tools=[Tool(deploy, requires_approval=True)],
            capabilities=[Permissions(managed_rules=[PermissionRule('ask', tools=['deploy'])]), _safe_reviewer()],
        )
        held = await agent.run('go')
        assert isinstance(held.output, DeferredToolRequests)
        [call] = held.output.approvals
        assert held.output.metadata == {call.tool_call_id: {'permissions': {'person_only': True}}}

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
