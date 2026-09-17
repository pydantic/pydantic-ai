"""The built-in `ask_user` plugin, driven headless."""

import asyncio
import io
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import pytest
from menu_script import Script
from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness.ask_user import (
    DECLINED,
    AskUser,
    AskUserAnswer,
    AskUserAnsweredEvent,
    AskUserRequest,
    AskUserResponse,
    Question,
    QuestionOption,
)
from rich.console import Console
from termflow.tui import MenuItem  # pyright: ignore[reportMissingTypeStubs]
from termflow.tui.menu import Menu, MenuResult  # pyright: ignore[reportMissingTypeStubs]

from pydantic_clai2 import DEFAULT_PLUGINS
from pydantic_clai2.ask_user_menu import QuestionMenu, TerminalAnswerer, activate, build_question_menu, render_answer
from pydantic_clai2.field_menu import TERMINAL, Runners
from pydantic_clai2.plugins import PluginHost


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


APPROACH = Question(
    header='Approach',
    question='How should we do it?',
    options=(QuestionOption(label='Refactor', description='Rewrite the module'), QuestionOption(label='Patch')),
)
TARGETS = Question(
    header='Targets',
    question='Which files?',
    options=(QuestionOption(label='api.py'), QuestionOption(label='db.py')),
    multi_select=True,
)


def chosen(*labels: str) -> MenuResult:
    items = [MenuItem(label, value=label) for label in labels]
    return MenuResult(item=items[0], items=items)


class ScreenLog:
    """A `FullScreen` that records when the terminal was taken and given back."""

    def __init__(self) -> None:
        self.events: list[str] = []

    @asynccontextmanager
    async def __call__(self) -> AsyncGenerator[None]:
        self.events.append('taken')
        try:
            yield
        finally:
            self.events.append('released')


def test_menu_parts_single_select() -> None:
    menu = QuestionMenu(question=APPROACH, position=1, total=1)
    assert menu.title == 'Approach'
    assert menu.hint == 'Enter select - Esc decline'
    assert [item.label for item in menu.items()] == ['Refactor', 'Patch']
    refactor, patch = menu.items()
    assert menu.preview(refactor) == 'How should we do it?\n\nRewrite the module'
    assert menu.preview(patch) == 'How should we do it?\n\n(no description)'
    assert isinstance(menu.build(), Menu)


def test_menu_parts_multi_select_numbered() -> None:
    menu = QuestionMenu(question=TARGETS, position=2, total=3)
    assert menu.title == 'Targets (question 2 of 3)'
    assert menu.hint == 'Space toggle - Enter confirm - Esc decline'
    assert isinstance(build_question_menu(TARGETS, position=2, total=3), Menu)


async def test_answers_every_question_on_a_settled_screen() -> None:
    script = Script(lists=[], choices=[chosen('Patch'), chosen('api.py', 'db.py')], texts=[])
    screen = ScreenLog()
    answerer = TerminalAnswerer(full_screen=screen, runners=script.runners)
    response = await answerer(AskUserRequest(questions=(APPROACH, TARGETS)))
    assert response == AskUserResponse(
        answers=(
            AskUserAnswer(header='Approach', selected=('Patch',)),
            AskUserAnswer(header='Targets', selected=('api.py', 'db.py')),
        )
    )
    assert script.opened == ['choice', 'choice']
    assert screen.events == ['taken', 'released']


async def test_parallel_requests_take_the_terminal_one_at_a_time() -> None:
    screen = ScreenLog()
    started = asyncio.Event()
    release = asyncio.Event()

    def slow_choice(menu: Menu) -> MenuResult:
        started.set()
        asyncio.run_coroutine_threadsafe(release.wait(), loop).result()
        return chosen('Patch')

    loop = asyncio.get_running_loop()
    runners = Runners(run_list=slow_choice, run_choice=slow_choice, run_text=TERMINAL.run_text)
    answerer = TerminalAnswerer(full_screen=screen, runners=runners)
    first = asyncio.create_task(answerer(AskUserRequest(questions=(APPROACH,))))
    await started.wait()
    second = asyncio.create_task(answerer(AskUserRequest(questions=(APPROACH,))))
    await asyncio.sleep(0.05)
    assert screen.events == ['taken']
    release.set()
    assert (await first).answers == (AskUserAnswer(header='Approach', selected=('Patch',)),)
    assert (await second).answers == (AskUserAnswer(header='Approach', selected=('Patch',)),)
    assert screen.events == ['taken', 'released', 'taken', 'released']


async def test_escape_declines_the_whole_request() -> None:
    script = Script(lists=[], choices=[chosen('Patch'), MenuResult(cancelled=True)], texts=[])
    screen = ScreenLog()
    answerer = TerminalAnswerer(full_screen=screen, runners=script.runners)
    response = await answerer(AskUserRequest(questions=(APPROACH, TARGETS)))
    assert response == AskUserResponse(cancelled=True)
    assert screen.events == ['taken', 'released']


def test_render_answer_lists_picks_or_the_decline() -> None:
    answered = AskUserAnsweredEvent(
        request_id='r1',
        response=AskUserResponse(
            answers=(
                AskUserAnswer(header='Approach', selected=('Patch',)),
                AskUserAnswer(header='Targets', selected=('api.py', 'db.py')),
            )
        ),
    )
    output = io.StringIO()
    console = Console(file=output, width=80)
    console.print(render_answer(answered))
    console.print(render_answer(AskUserAnsweredEvent(request_id='r2', response=AskUserResponse(cancelled=True))))
    text = output.getvalue()
    assert '● Approach: Patch\n● Targets: api.py, db.py\n' in text
    assert '● You declined to answer' in text


async def test_activate_registers_capability_and_renderer() -> None:
    host: PluginHost[None] = PluginHost(name='ask_user', console=Console(file=io.StringIO()), settings={})
    activate(host)
    (capability,) = host.capabilities
    assert isinstance(capability, AskUser)
    (renderer,) = host.renderers
    assert renderer(AskUserAnsweredEvent(request_id='r', response=AskUserResponse(cancelled=True))) is not None
    assert any(plugin.id == 'ask_user' for plugin in DEFAULT_PLUGINS)


async def test_declining_reaches_the_model_through_the_plugin() -> None:
    """The capability's declined result survives the round trip through the real tool call."""
    script = Script(lists=[], choices=[MenuResult(cancelled=True)], texts=[])
    capabilities: list[AbstractCapability[None]] = [
        AskUser(answerer=TerminalAnswerer(full_screen=ScreenLog(), runners=script.runners))
    ]

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            questions = [APPROACH.model_dump(), TARGETS.model_dump()]
            return ModelResponse(parts=[ToolCallPart('ask_user_question', {'questions': questions})])
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(respond), deps_type=type(None), capabilities=capabilities)
    result = await agent.run('go')
    assert result.output == 'done'
    returns = [part for message in result.all_messages() for part in message.parts if isinstance(part, ToolReturnPart)]
    assert len(returns) == 1 and returns[0].content == DECLINED
