"""The built-in `computer_use` plugin: registration, inline approvals, and its transcript line.

CLAI's CI installs no extras, so `LocalComputer`'s module is replaced with a stand-in; the harness
suite covers the real one.
"""

import io
import sys
import types
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import pytest
from prompt_toolkit.input import PipeInput, create_pipe_input
from pydantic import TypeAdapter
from rich.console import Console, RenderableType
from rich.text import Text

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import HandleDeferredToolCalls
from pydantic_ai.messages import ModelMessage, ModelResponse, PartStartEvent, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import DeferredToolRequests, ToolDenied
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.ask_user import AskUserAnswer, AskUserRequest, AskUserResponse
from pydantic_ai_harness.computer_use import ComputerAction, ComputerActionsEvent, ComputerUse
from pydantic_clai2 import DEFAULT_PLUGINS
from pydantic_clai2.builtin_plugins.computer_use import (
    ALLOW,
    ALLOW_SESSION,
    DENY,
    INSTALL_HINT,
    Approver,
    ComputerUsePlugin,
    describe_actions,
    render_actions,
)
from pydantic_clai2.plugins import PluginHost, bare_screen, load_plugin

LOCAL_MODULE = 'pydantic_ai_harness.computer_use._local'


@dataclass
class StandInComputer:
    monitor: int = 1


@pytest.fixture
def local_computer(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType(LOCAL_MODULE)
    module.LocalComputer = StandInComputer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, LOCAL_MODULE, module)


def _host(settings: dict[str, Any]) -> PluginHost[None]:
    return PluginHost(name='computer_use', console=Console(file=io.StringIO()), settings=settings)


def _ctx() -> RunContext[None]:
    return RunContext(deps=None, model=TestModel(), usage=RunUsage())


def _call(tool_call_id: str, *actions: dict[str, Any], tool_name: str = 'computer') -> ToolCallPart:
    return ToolCallPart(tool_name, {'actions': list(actions)}, tool_call_id=tool_call_id)


CLICK = {'type': 'click', 'x': 10, 'y': 20}


class Script:
    """Answer each question from a list, and keep the questions for inspection."""

    def __init__(self, *responses: AskUserResponse) -> None:
        self.responses = list(responses)
        self.questions: list[str] = []

    async def __call__(self, request: AskUserRequest) -> AskUserResponse:
        [question] = request.questions
        self.questions.append(question.question)
        return self.responses.pop(0)


class Shown:
    """Collect what the approver prints above the picker."""

    def __init__(self) -> None:
        self.listings: list[str] = []

    async def append_async(self, listing: RenderableType) -> None:
        assert isinstance(listing, Text)
        self.listings.append(listing.plain)


def picked(label: str) -> AskUserResponse:
    return AskUserResponse(answers=(AskUserAnswer(header='Computer', selected=(label,)),))


def _plugin(host: PluginHost[None]) -> ComputerUsePlugin:
    plugin = load_plugin(ComputerUsePlugin, host).plugin
    assert isinstance(plugin, ComputerUsePlugin)
    return plugin


class TestPlugin:
    @pytest.mark.usefixtures('local_computer')
    def test_registers_approvals_and_a_renderer_by_default(self) -> None:
        plugin = _plugin(_host({'monitor': 2}))

        computer_use, approvals = plugin.get_capabilities()
        assert isinstance(computer_use, ComputerUse)
        assert computer_use.require_approval
        assert computer_use.computer == StandInComputer(monitor=2)
        assert 'this ' in computer_use.get_instructions()
        assert isinstance(approvals, HandleDeferredToolCalls)
        assert plugin.render(_event([CLICK], performed=1)) is not None
        assert plugin.render(PartStartEvent(index=0, part=TextPart('hi'))) is None

    @pytest.mark.usefixtures('local_computer')
    def test_approval_can_be_turned_off(self) -> None:
        [computer_use] = _plugin(_host({'require_approval': False})).get_capabilities()

        assert isinstance(computer_use, ComputerUse)
        assert not computer_use.require_approval

    def test_a_missing_extra_explains_how_to_install_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, LOCAL_MODULE, None)

        with pytest.raises(ImportError, match='computer-use'):
            ComputerUsePlugin.from_host(_host({}))
        assert 'pydantic-clai2[computer-use]' in INSTALL_HINT

    def test_ships_disabled(self) -> None:
        [declaration] = [plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'computer_use']
        assert not declaration.enabled
        assert declaration.factory == 'pydantic_clai2.builtin_plugins.computer_use'


class TestApprover:
    async def test_allow_approves_one_call_and_keeps_asking(self) -> None:
        script = Script(picked(ALLOW), picked(ALLOW))
        shown = Shown()
        approver = Approver(script, shown.append_async, bare_screen)
        requests = DeferredToolRequests(approvals=[_call('a', CLICK), _call('b', {'type': 'type', 'text': 'hi'})])

        results = await approver(_ctx(), requests)

        assert results is not None
        assert results.approvals == {'a': True, 'b': True}
        assert script.questions == ['Allow the agent to click (10, 20)?', "Allow the agent to type 'hi'?"]

    async def test_allow_for_the_session_stops_asking(self) -> None:
        script = Script(picked(ALLOW_SESSION))
        shown = Shown()
        approver = Approver(script, shown.append_async, bare_screen)

        first = await approver(_ctx(), DeferredToolRequests(approvals=[_call('a', CLICK), _call('b', CLICK)]))
        later = await approver(_ctx(), DeferredToolRequests(approvals=[_call('c', CLICK)]))

        assert first is not None and later is not None
        assert first.approvals == {'a': True, 'b': True}
        assert later.approvals == {'c': True}
        assert len(script.questions) == 1

    @pytest.mark.parametrize(
        ('response', 'message'),
        [
            (picked(DENY), 'The user denied this computer action. Ask them how to proceed.'),
            (AskUserResponse(cancelled=True), 'The user declined this computer action. Ask them how to proceed.'),
            (
                AskUserResponse(answers=(AskUserAnswer(header='Computer', custom_answer='use the menu instead'),)),
                'The user denied this computer action and said: use the menu instead',
            ),
        ],
    )
    async def test_refusals_reach_the_model_as_denials(self, response: AskUserResponse, message: str) -> None:
        approver = Approver(Script(response), Shown().append_async, bare_screen)

        results = await approver(_ctx(), DeferredToolRequests(approvals=[_call('a', CLICK)]))

        assert results is not None
        assert results.approvals == {'a': ToolDenied(message)}

    async def test_other_tools_are_left_to_someone_else(self) -> None:
        approver = Approver(Script(), Shown().append_async, bare_screen)

        results = await approver(_ctx(), DeferredToolRequests(approvals=[_call('a', CLICK, tool_name='shell')]))

        assert results is None

    async def test_inline_requests_print_nothing_extra(self) -> None:
        shown = Shown()

        await Approver(Script(picked(ALLOW)), shown.append_async, bare_screen)(
            _ctx(), DeferredToolRequests(approvals=[_call('a', CLICK)])
        )

        assert shown.listings == []

    async def test_long_requests_are_listed_in_full_before_asking(self) -> None:
        script = Script(picked(DENY))
        shown = Shown()
        typed = 'x' * 300 + '; rm -rf ~'
        actions = [*([CLICK] * 5), {'type': 'type', 'text': typed}, {'type': 'keypress', 'keys': ['enter']}]

        await Approver(script, shown.append_async, bare_screen)(
            _ctx(), DeferredToolRequests(approvals=[_call('a', *actions)])
        )

        [listing] = shown.listings
        assert listing.splitlines() == [
            'The agent wants to run 7 computer actions:',
            *(f'  {number}. click (10, 20)' for number in range(1, 6)),
            f'  6. type {typed!r}',
            '  7. press enter',
        ]
        assert script.questions == ['Allow the agent to run the 7 computer actions listed above?']

    async def test_without_a_terminal_the_call_is_denied_without_asking(self) -> None:
        script = Script()
        shown = Shown()

        @asynccontextmanager
        async def headless_screen() -> AsyncGenerator[None]:
            raise RuntimeError('User interaction is unavailable in headless mode')
            yield  # pragma: no cover

        approver = Approver(script, shown.append_async, headless_screen)
        actions = [{'type': 'type', 'text': 'y' * 400}]

        results = await approver(_ctx(), DeferredToolRequests(approvals=[_call('a', *actions)]))

        assert results is not None
        assert results.approvals == {
            'a': ToolDenied('Computer actions need approval, which is unavailable in this headless session.')
        }
        assert (script.questions, shown.listings) == ([], [])


def test_actions_are_described_in_order() -> None:
    actions = [
        {'type': 'click', 'x': 1, 'y': 2, 'button': 'right', 'count': 2, 'modifiers': ['cmd']},
        {'type': 'click', 'x': 1, 'y': 2, 'count': 3},
        {'type': 'move', 'x': 3, 'y': 4},
        {'type': 'drag', 'path': [{'x': 0, 'y': 0}, {'x': 1, 'y': 1}, {'x': 5, 'y': 6}]},
        {'type': 'scroll', 'x': 7, 'y': 8, 'direction': 'up', 'amount': 5},
        {'type': 'type', 'text': 'a\x1b[31mb'},
        {'type': 'keypress', 'keys': ['ctrl', 'c']},
        {'type': 'wait', 'seconds': 1.5},
        {'type': 'screenshot'},
    ]
    event = _event(actions, performed=len(actions))

    assert describe_actions(event.actions) == (
        'cmd+right-double-click (1, 2), triple-click (1, 2), move to (3, 4), drag (0, 0) to (1, 1) to (5, 6), '
        "scroll up 5 at (7, 8), type 'a\\x1b[31mb', press ctrl+c, wait 1.5s, screenshot"
    )


def _event(actions: list[dict[str, Any]], *, performed: int, error: str | None = None) -> ComputerActionsEvent:
    """Validate actions the way the tool does, so the event carries the typed models."""
    return ComputerActionsEvent(
        tool_call_id='a',
        actions=tuple(TypeAdapter(list[ComputerAction]).validate_python(actions)),
        performed=performed,
        error=error,
        screenshot_size=(1280, 800),
    )


def test_the_transcript_shows_what_ran_and_why_it_stopped() -> None:
    event = _event([CLICK, {'type': 'keypress', 'keys': ['hyper']}], performed=1, error='Unknown key\n')
    rendered = render_actions(event)

    assert isinstance(rendered, Text)
    assert rendered.plain == '  click (10, 20)\n  failed: Unknown key\\x0a'


def test_the_transcript_says_when_nothing_ran() -> None:
    rendered = render_actions(_event([CLICK], performed=0, error='broke'))

    assert isinstance(rendered, Text)
    assert rendered.plain.startswith('  nothing ran\n')


def test_the_transcript_lists_a_clean_run_on_one_line() -> None:
    rendered = render_actions(_event([CLICK, {'type': 'screenshot'}], performed=2))

    assert isinstance(rendered, Text)
    assert rendered.plain == '  click (10, 20), screenshot'


@pytest.fixture
def question_pipe(monkeypatch: pytest.MonkeyPatch) -> Generator[PipeInput]:
    with create_pipe_input() as pipe:
        monkeypatch.setattr('pydantic_clai2.ui.prompt.question_input.create_input', lambda: pipe)
        yield pipe


@pytest.mark.usefixtures('local_computer')
async def test_a_run_lists_a_long_request_in_the_terminal_then_asks(question_pipe: PipeInput) -> None:
    output = io.StringIO()
    host: PluginHost[None] = PluginHost(
        name='computer_use', console=Console(file=output, width=200, height=40), settings={}
    )
    plugin = _plugin(host)
    typed = 'z' * 320

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[_call('a', {'type': 'type', 'text': typed})])
        return ModelResponse(parts=[TextPart('stopped')])

    question_pipe.send_text('3')
    result = await Agent(FunctionModel(respond), deps_type=type(None), capabilities=plugin.get_capabilities()).run('go')

    [denial] = [part for part in result.all_messages()[2].parts if isinstance(part, ToolReturnPart)]
    assert denial.content == 'The user denied this computer action. Ask them how to proceed.'
    assert f"1.type'{typed}'" in ''.join(output.getvalue().split())
