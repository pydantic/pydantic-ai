"""`/permissions`: the `run.permissions` mode on every run, its approval question, and the denial of last resort."""

import io
from pathlib import Path

import anyio
import pytest
from rich.console import Console

from pydantic_ai import Agent, DeferredToolRequests, Tool
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo
from pydantic_ai_harness.ask_user import AskUserAnswer, AskUserResponse
from pydantic_ai_harness.permissions import PermissionDecision, PermissionRequest
from pydantic_ai_harness.shell import Shell
from pydantic_ai_harness.subagents import SubAgents
from pydantic_clai2.builtin_plugins.ask_user_menu import QuestionMenu, TerminalAnswerer, terminal_approver
from pydantic_clai2.builtin_plugins.web import WebPlugin
from pydantic_clai2.builtin_plugins.web.live import BrowserAnswerer
from pydantic_clai2.builtin_plugins.web.protocol import QuestionAsked, QuestionClosed
from pydantic_clai2.config import Settings
from pydantic_clai2.config.project_settings import ProjectSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.plugins import PluginHost, bare_screen
from pydantic_clai2.runtime.permissions import (
    APPROVE,
    APPROVE_SESSION,
    DENY,
    HEADER,
    UNANSWERED,
    approval_handler,
    approval_question,
    decision,
    proposed_diff,
    session_permissions,
)
from pydantic_clai2.runtime.process import ClaiProcess
from pydantic_clai2.runtime.session_runtime import SessionRuntime, create_shell
from pydantic_clai2.runtime.speculation import SpeculationCounters
from pydantic_clai2.runtime.speculative_mode import speculative_capabilities
from tests.clai2.test_speculative_mode import streamed

COMMAND = 'rm -rf doomed'


def request(action: str = 'command', subject: str = COMMAND, **args: object) -> PermissionRequest:
    name = 'run_command' if action == 'command' else 'edit_file'
    call = ToolCallPart(name, args or {'command': subject}, tool_call_id='call1')
    return PermissionRequest(
        call=call, action='command' if action == 'command' else 'write', subject=subject, reason='Asks.'
    )


def removing_agent(folder: Path, ran: list[str]) -> Agent[None, str]:
    """An agent that runs `COMMAND` once with a shell tool declared as harness `Shell` declares its own."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        returned = [part for part in messages[-1].parts if isinstance(part, ToolReturnPart)]
        if returned:
            return ModelResponse(parts=[TextPart(f'Tool said: {returned[0].content}')])
        return ModelResponse(parts=[ToolCallPart('run_command', {'command': COMMAND})])

    def run_command(command: str) -> str:
        ran.append(command)
        return f'ran {command}'

    tool = Tool(run_command, metadata={'code_arg_name': 'command', 'code_arg_language': 'shell'})
    return Agent(streamed(respond), deps_type=type(None), tools=[tool])


def runtime(tmp_path: Path, agent: Agent[None, str]) -> tuple[SessionRuntime[None, str], io.StringIO]:
    output = io.StringIO()
    shell = create_shell(
        agent,
        deps=None,
        plugins=(),
        usage_limits=None,
        console=Console(file=output, force_terminal=False, width=200),
        settings=Settings(model=None, code_mode=False),
        store=None,
        builtin_plugins=(),
        project=ProjectSettings(),
        headless=True,
        workspace=tmp_path,
        process=ClaiProcess.open(SettingsStore(tmp_path / 'config.db')),
        signals=False,
    )
    return shell, output


@pytest.mark.parametrize(('mode', 'runs'), [('full', True), ('auto', False)])
async def test_a_session_no_one_can_be_asked_in_denies_held_calls(tmp_path: Path, mode: str, runs: bool) -> None:
    """Without a terminal or browser to ask, `auto` denies the `rm -rf` and says how to let it run; `full` runs it."""
    ran: list[str] = []
    shell, output = runtime(tmp_path, removing_agent(tmp_path, ran))
    shell.send_input(f'/permissions {mode}')
    shell.send_input('clean up')
    shell.send_input('/exit')
    assert await shell.run() == 'exit'
    assert ran == ([COMMAND] if runs else [])
    if not runs:
        assert f'The user denied this command, so it did not run. {UNANSWERED}' in output.getvalue()


async def test_permissions_command_shows_and_sets_the_mode(tmp_path: Path) -> None:
    shell, output = runtime(tmp_path, removing_agent(tmp_path, []))
    shell.send_input('/permissions')
    shell.send_input('/permissions read-only')
    shell.send_input('/permissions sometimes')
    shell.send_input('/exit')
    await shell.run()
    text = output.getvalue()
    assert 'Permissions: Full access: everything runs without asking.' in text
    assert '/permissions auto' in text
    assert 'Permissions: Read only: every file write and command asks first.' in text
    assert 'Usage: /permissions [full|auto|read-only]' in text
    assert shell.context.settings.permissions == 'read-only'
    assert shell.commands.get('permissions').complete([]) == ('full', 'auto', 'read-only')  # pyright: ignore[reportOptionalMemberAccess]


def test_approval_question_offers_the_three_choices() -> None:
    question = approval_question(request(subject='echo\ta \x1b[31mred\x1b[0m\u2028\u2029x\u00a0y'))
    [asked] = question.questions
    assert question.id == 'call1'
    assert asked.header == HEADER
    assert [option.label for option in asked.options] == [APPROVE, APPROVE_SESSION, DENY]
    assert asked.question == 'Asks. Run this command?\necho    a [31mred[0mx\u00a0y'
    assert 'this file' in approval_question(request('write', 'a.py')).questions[0].options[1].description  # pyright: ignore[reportOperatorIssue]
    assert approval_question(request('write', 'a.py')).questions[0].question == 'Asks. Write a.py?'
    long = approval_question(request(subject='x' * 600)).questions[0].question
    assert len(long) == 500 and long.endswith('…')


@pytest.mark.parametrize(
    ('response', 'expected'),
    [
        (AskUserResponse(cancelled=True), PermissionDecision(approved=False)),
        (AskUserResponse(), PermissionDecision(approved=False)),
        (
            AskUserResponse(answers=(AskUserAnswer(header=HEADER, selected=(APPROVE,)),)),
            PermissionDecision(approved=True),
        ),
        (
            AskUserResponse(answers=(AskUserAnswer(header=HEADER, selected=(APPROVE_SESSION,)),)),
            PermissionDecision(approved=True, remember=True),
        ),
        (
            AskUserResponse(answers=(AskUserAnswer(header=HEADER, selected=(DENY,)),)),
            PermissionDecision(approved=False),
        ),
        (
            AskUserResponse(answers=(AskUserAnswer(header=HEADER, custom_answer='not now'),)),
            PermissionDecision(approved=False, reason='not now'),
        ),
    ],
)
def test_decision(response: AskUserResponse, expected: PermissionDecision) -> None:
    assert decision(response) == expected


def test_proposed_diff() -> None:
    assert proposed_diff(request()) is None
    assert proposed_diff(request('write', 'a.py', path='a.py')) is None
    assert proposed_diff(request('write', 'a.py', path='a.py', content=3)) is None
    assert (
        proposed_diff(request('write', 'a.py', path='a.py', content='one\n'))
        == '--- a/a.py\n+++ b/a.py\n@@ -0,0 +1 @@\n+one\n'
    )
    edited = proposed_diff(
        request(
            'write',
            'a.py',
            path='a.py',
            old_text='a\n',
            new_text='b\n',
            replacements=[{'old_text': 'c\n', 'new_text': 'd\n'}],
        )
    )
    assert edited is not None and '-a\n+b\n' in edited and '-c\n+d\n' in edited


@pytest.mark.parametrize(
    ('held', 'printed'),
    [
        pytest.param(request(subject='ls'), 'ls', id='command'),
        pytest.param(
            request(subject='echo ready\n' + '# padding\n' * 12 + 'rm -rf valuable'),
            'rm -rf valuable',
            id='multi-line-command',
        ),
        pytest.param(request(subject='echo ' + 'x' * 600 + ' && rm -rf doomed'), 'rm -rf doomed', id='long-command'),
        pytest.param(request('write', 'a.py', path='a.py', content='x'), None, id='path'),
        pytest.param(request('write', 'd/' * 300 + 'a.py', path='a.py', content='x'), 'd/d/a.py', id='long-path'),
    ],
)
async def test_terminal_approver_prints_what_the_picker_may_cut(held: PermissionRequest, printed: str | None) -> None:
    """The picker cuts its question to the rows the terminal leaves it, so a command is always printed in full first."""
    output = io.StringIO()
    frames: list[tuple[str, ...]] = []

    def pick(menu: QuestionMenu) -> tuple[str, ...]:
        frames.append(menu.frame(width=80, height=24))
        return (APPROVE_SESSION,)

    answerer = TerminalAnswerer(full_screen=bare_screen, console=Console(file=output, width=1000), runner=pick)
    assert await terminal_approver(answerer)(held) == PermissionDecision(approved=True, remember=True)
    if printed is None:
        assert output.getvalue() == ''
    else:
        assert printed in output.getvalue()
    if held.subject.startswith('echo ready'):
        # Why it must be printed: a 24-row terminal's picker shows the first rows only.
        assert not any('rm -rf valuable' in row for row in frames[0])


async def test_the_browser_shows_an_approval_card() -> None:
    published: list[QuestionAsked | QuestionClosed] = []
    answerer = BrowserAnswerer(publish=lambda message, _: published.append(message), terminal=None)
    decisions: list[PermissionDecision] = []
    held = request('write', 'a.py', path='a.py', content='one\n')
    async with anyio.create_task_group() as group:

        async def approve() -> None:
            decisions.append(await answerer.approve(held))

        group.start_soon(approve)
        await anyio.wait_all_tasks_blocked()
        [questions] = answerer.questions
        assert questions.approval is not None
        assert (questions.approval.tool, questions.approval.action, questions.approval.subject) == (
            'edit_file',
            'write',
            'a.py',
        )
        assert questions.approval.diff is not None and '+one' in questions.approval.diff.diff
        answerer.reply(questions.id, [AskUserAnswer(header=HEADER, custom_answer='no')])
    assert decisions == [PermissionDecision(approved=False, reason='no')]
    assert isinstance(published[0], QuestionAsked)
    assert published[0].questions.approval is not None


async def test_a_command_card_has_no_diff() -> None:
    answerer = BrowserAnswerer(publish=lambda message, _: None, terminal=None)
    async with anyio.create_task_group() as group:
        group.start_soon(answerer.approve, request())
        await anyio.wait_all_tasks_blocked()
        [questions] = answerer.questions
        assert questions.approval is not None and questions.approval.diff is None
        answerer.reply(questions.id, None)


@pytest.mark.parametrize(
    ('saved', 'shown'), [({'run.permissions': 'auto'}, 'auto'), ({'run.permissions': 3}, None), ({}, None)]
)
def test_the_status_names_the_mode(saved: dict[str, object], shown: str | None) -> None:
    def setting(key: str) -> object:
        if key not in saved:
            raise KeyError(key)
        return saved[key]

    host = PluginHost[None](name='web', console=Console(file=io.StringIO()), settings={}, setting=setting)  # pyright: ignore[reportArgumentType]
    assert WebPlugin.from_host(host).service.default().status().permissions == shown


async def test_approval_handler_answers_held_calls_and_leaves_others(tmp_path: Path) -> None:
    ran: list[str] = []
    asked: list[PermissionRequest] = []

    async def approve(held: PermissionRequest) -> PermissionDecision:
        asked.append(held)
        return PermissionDecision(approved=True)

    agent = removing_agent(tmp_path, ran)
    result = await agent.run('go', capabilities=[*session_permissions(lambda: 'auto'), approval_handler(approve)])
    assert ran == [COMMAND] and [held.subject for held in asked] == [COMMAND]
    assert result.output == f'Tool said: ran {COMMAND}'

    other = DeferredToolRequests(approvals=[ToolCallPart('other', {}, tool_call_id='x')])
    [_, unanswered] = session_permissions(lambda: 'auto')
    assert await unanswered.handle_deferred_tool_calls(None, requests=other) is None  # pyright: ignore[reportArgumentType]


async def test_code_mode_holds_a_shell_call_from_run_code(tmp_path: Path) -> None:
    """Code mode folds `shell` into `run_code` by dropping its `code_arg_name`; its commands are still held."""
    (tmp_path / 'doomed').mkdir()
    asked: list[PermissionRequest] = []

    async def deny(held: PermissionRequest) -> PermissionDecision:
        asked.append(held)
        return PermissionDecision(approved=False)

    calls = iter([ToolCallPart('run_code', {'code': "await shell(command='rm -rf doomed')"})])

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[next(calls, TextPart('done'))])

    granted = (Shell[object](tools=['shell'], denied_commands=[]), *session_permissions(lambda: 'auto'))
    sandbox = speculative_capabilities(SpeculationCounters(), granted)
    capabilities = [LocalWorkspace(tmp_path), *granted, approval_handler(deny), *sandbox]
    await Agent(streamed(respond), capabilities=capabilities).run('go')
    assert [held.subject for held in asked] == ['rm -rf doomed']
    assert (tmp_path / 'doomed').is_dir()


async def test_the_terminal_can_answer_the_browsers_approval_first() -> None:
    async def terminal(held: PermissionRequest) -> AskUserResponse:
        return AskUserResponse(answers=(AskUserAnswer(header=HEADER, selected=(APPROVE,)),))

    answerer = BrowserAnswerer(publish=lambda message, _: None, terminal=None, terminal_approval=terminal)
    assert await answerer.approve(request()) == PermissionDecision(approved=True)
    assert answerer.questions == []


async def test_a_supplied_agents_self_delegation_is_held(tmp_path: Path) -> None:
    """A supplied agent gets `/permissions` per run, which its delegate to `self` would not carry, so the delegation is held."""
    ran: list[str] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        returned = [part for part in messages[-1].parts if isinstance(part, ToolReturnPart)]
        if returned:
            return ModelResponse(parts=[TextPart(f'Tool said: {returned[0].content}')])
        if 'delegate_task' in [tool.name for tool in info.function_tools]:
            return ModelResponse(parts=[ToolCallPart('delegate_task', {'agent_name': 'self', 'task': 'clean up'})])
        return ModelResponse(parts=[ToolCallPart('run_command', {'command': COMMAND})])

    def run_command(command: str) -> str:
        ran.append(command)
        return f'ran {command}'

    tool = Tool(run_command, metadata={'code_arg_name': 'command', 'code_arg_language': 'shell'})
    agent = Agent(
        streamed(respond),
        deps_type=type(None),
        tools=[tool],
        capabilities=[SubAgents[None](include_self=True, max_depth=2, agent_folders=None)],
    )
    shell, output = runtime(tmp_path, agent)
    shell.send_input('/permissions auto')
    shell.send_input('clean up')
    shell.send_input('/exit')
    assert await shell.run() == 'exit'
    assert ran == []
    assert f'The user denied this delegation, so it did not run. {UNANSWERED}' in output.getvalue()
