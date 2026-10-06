"""The built-in `ask_user` plugin, driven headless."""

import asyncio
import io
import json
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Generator
from contextlib import asynccontextmanager, contextmanager
from threading import Event

import anyio
import pytest
from prompt_toolkit.input import PipeInput, create_pipe_input
from rich.cells import cell_len
from rich.console import Console
from rich.text import Text

from pydantic_ai import Agent, FunctionToolCallEvent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage, ModelResponse, PartStartEvent, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
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
from pydantic_ai_harness.subagents import DelegationTasks, SubAgent, SubAgents
from pydantic_clai2 import DEFAULT_PLUGINS, StreamRenderer
from pydantic_clai2.builtin_plugins.ask_user_menu import (
    AskUserPlugin,
    QuestionMenu,
    TerminalAnswerer,
    render_answer,
)
from pydantic_clai2.plugins import PluginHost, load_plugin
from pydantic_clai2.runtime.sandbox_calls import SandboxCallStartedEvent
from pydantic_clai2.ui.menus.menu_worker import menu_key
from pydantic_clai2.ui.prompt.prompt_surface import ENTER, LEAVE, MODES_OFF, PromptSurface
from pydantic_clai2.ui.prompt.question_input import Paste
from tests.clai2.surface_terminal import SurfaceTerminal


@pytest.fixture
def question_pipe(monkeypatch: pytest.MonkeyPatch) -> Generator[PipeInput]:
    with create_pipe_input() as pipe:
        monkeypatch.setattr('pydantic_clai2.ui.prompt.question_input.create_input', lambda: pipe)
        yield pipe


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


def test_single_select_and_number_shortcuts() -> None:
    menu = QuestionMenu(question=APPROACH, position=1, total=1)
    assert menu.title == 'Approach'
    assert menu.choose('up') is None
    assert menu.choose('up') is None
    assert menu.choose('enter') == ('Patch',)
    assert menu.choose('1') == ('Refactor',)
    assert menu.choose('down') is None
    assert menu.choose('tab') is None
    assert menu.choose('tab') is None
    assert menu.choose('enter') == ('Refactor',)


def test_multi_select_enter_toggles_and_done_submits() -> None:
    menu = QuestionMenu(question=TARGETS, position=2, total=3)
    assert menu.title == 'Targets (question 2 of 3)'
    assert menu.choose('3') is None  # Empty answers cannot be submitted.
    assert menu.choose('1') is None
    assert menu.choose('enter') is None
    assert menu.selected == set()
    assert menu.choose('2') is None
    assert menu.choose('1') is None
    assert menu.choose(' ') is None
    assert menu.choose('9') is None
    assert menu.choose('3') == ('api.py', 'db.py')


def test_inline_frame_shows_context_selection_and_navigation() -> None:
    menu = QuestionMenu(question=TARGETS, position=1, total=1)
    menu.choose('1')
    frame = '\n'.join(menu.frame(width=100, height=24))
    assert 'Targets' in frame
    assert '> 1. [x] api.py' in frame
    assert '2. [ ] db.py' in frame
    assert '3. Done' in frame
    assert 'Space' not in frame
    menu.choose('down')
    frame = menu.frame(width=24, height=10)
    assert any('db.py' in row for row in frame)
    assert len(frame) <= 5
    assert all(len(Text.from_ansi(row).plain) <= 24 for row in frame)


@pytest.mark.parametrize(
    'keys, expected',
    [
        (['', 'down', 'enter'], ('Patch',)),
        (['escape'], None),
        (['ctrl-c'], None),
    ],
)
def test_inline_terminal_preserves_transcript(keys: list[str], expected: tuple[str, ...] | None) -> None:
    output = io.StringIO()
    console = Console(file=output, width=80, height=24)
    console.print('Previous conversation stays here')
    menu = QuestionMenu(question=APPROACH, position=1, total=1)
    assert menu.run(console=console, key_source=iter(keys).__next__) == expected
    rendered = output.getvalue()
    assert rendered.startswith('Previous conversation stays here')
    assert 'Rewrite the module' in rendered
    # Without an editor, the picker opens a live panel for the question, then leaves the
    # question in the terminal's own scrollback.
    assert rendered.rindex(LEAVE) > rendered.rindex(ENTER)
    assert 'How should we do it?' in rendered.rsplit(LEAVE, 1)[1]
    assert '\x1b[3J' not in rendered
    assert '\x1b[?25h' in rendered


async def test_answers_every_question_on_a_settled_screen() -> None:
    answers = iter([('Patch',), ('api.py', 'db.py')])
    screen = ScreenLog()
    answerer = TerminalAnswerer(full_screen=screen, runner=lambda menu: next(answers))
    response = await answerer(AskUserRequest(questions=(APPROACH, TARGETS)))
    assert response == AskUserResponse(
        answers=(
            AskUserAnswer(header='Approach', selected=('Patch',)),
            AskUserAnswer(header='Targets', selected=('api.py', 'db.py')),
        )
    )
    assert screen.events == ['taken', 'released']


async def test_parallel_requests_take_the_terminal_one_at_a_time() -> None:
    screen = ScreenLog()
    started = asyncio.Event()
    release = asyncio.Event()

    def slow_choice(menu: QuestionMenu) -> tuple[str, ...]:
        loop.call_soon_threadsafe(started.set)
        asyncio.run_coroutine_threadsafe(release.wait(), loop).result()
        return ('Patch',)

    loop = asyncio.get_running_loop()
    answerer = TerminalAnswerer(full_screen=screen, runner=slow_choice)
    first = asyncio.create_task(answerer(AskUserRequest(questions=(APPROACH,))))
    await started.wait()
    second_started = asyncio.Event()

    async def another() -> AskUserResponse:
        second_started.set()
        return await answerer(AskUserRequest(questions=(APPROACH,)))

    second = asyncio.create_task(another())
    await second_started.wait()
    assert screen.events == ['taken']
    release.set()
    assert (await first).answers == (AskUserAnswer(header='Approach', selected=('Patch',)),)
    assert (await second).answers == (AskUserAnswer(header='Approach', selected=('Patch',)),)
    assert screen.events == ['taken', 'released', 'taken', 'released']


async def test_escape_declines_the_whole_request() -> None:
    answers = iter([('Patch',), None])
    screen = ScreenLog()
    answerer = TerminalAnswerer(full_screen=screen, runner=lambda menu: next(answers))
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


@pytest.mark.parametrize('sandboxed', [False, True])
async def test_call_header_names_the_questions_without_raw_json(sandboxed: bool) -> None:
    """The header once showed `questions=[{"header":...`; the picker below shows each question in full."""
    output = io.StringIO()
    host: PluginHost[None] = PluginHost(name='ask_user', console=Console(file=io.StringIO()), settings={})
    plugin = load_plugin(AskUserPlugin, host).plugin
    renderer = StreamRenderer(Console(file=output, width=80), stop_loading=lambda: None, renderers=[plugin.render])
    questions = {'questions': [APPROACH.model_dump(mode='json'), TARGETS.model_dump(mode='json')]}
    if sandboxed:
        call = ToolCallPart('ask_user_question', questions, tool_call_id='code__1')
        await renderer.on_stream_event(SandboxCallStartedEvent(tool_call_id='code__1', call=call))
        assert output.getvalue() == '● run_code\n\n● ask_user_question Approach, Targets\n\n'
    else:
        await renderer.on_stream_event(FunctionToolCallEvent(ToolCallPart('ask_user_question', questions)))
        assert output.getvalue() == '● ask_user_question Approach, Targets\n\n'


@pytest.mark.parametrize('args', ['{"questions": [{"head', {'questions': 'none'}, {}])
async def test_call_header_without_readable_questions_shows_only_the_tool_name(args: str | dict[str, object]) -> None:
    output = io.StringIO()
    console = Console(file=output, width=80)
    host: PluginHost[None] = PluginHost(name='ask_user', console=console, settings={})
    renderable = load_plugin(AskUserPlugin, host).plugin.render(
        FunctionToolCallEvent(ToolCallPart('ask_user_question', args))
    )
    console.print(renderable)
    assert output.getvalue() == '● ask_user_question\n'
    assert load_plugin(AskUserPlugin, host).plugin.render(FunctionToolCallEvent(ToolCallPart('other', args))) is None


async def test_call_header_with_many_questions_stays_on_one_row() -> None:
    output = io.StringIO()
    console = Console(file=output, width=40)
    host: PluginHost[None] = PluginHost(name='ask_user', console=console, settings={})
    questions = [APPROACH.model_copy(update={'header': f'Q{index}'}).model_dump() for index in range(10)]
    console.print(
        load_plugin(AskUserPlugin, host).plugin.render(
            FunctionToolCallEvent(ToolCallPart('ask_user_question', {'questions': questions}))
        )
    )
    assert output.getvalue() == '● ask_user_question Q0, Q1, Q2, Q3, Q4,…\n'


async def test_call_header_escapes_line_breaks_in_question_headers() -> None:
    """The header renders before validation, so a forged `\\n● ...` row must stay on the header's line."""
    output = io.StringIO()
    console = Console(file=output, width=80)
    host: PluginHost[None] = PluginHost(name='ask_user', console=console, settings={})
    questions = [APPROACH.model_copy(update={'header': 'Real\n● forged call'}).model_dump()]
    console.print(
        load_plugin(AskUserPlugin, host).plugin.render(
            FunctionToolCallEvent(ToolCallPart('ask_user_question', {'questions': questions}))
        )
    )
    assert output.getvalue() == '● ask_user_question Real\\x0a● forged call\n'


async def test_plugin_declares_capability_and_renderer() -> None:
    host: PluginHost[None] = PluginHost(name='ask_user', console=Console(file=io.StringIO()), settings={})
    loaded = load_plugin(AskUserPlugin, host)
    (capability,) = loaded.capabilities
    assert isinstance(capability, AskUser)
    assert loaded.plugin.has_render
    assert (
        loaded.plugin.render(AskUserAnsweredEvent(request_id='r', response=AskUserResponse(cancelled=True))) is not None
    )
    assert loaded.plugin.render(PartStartEvent(index=0, part=TextPart('hi'))) is None
    assert any(plugin.id == 'ask_user' for plugin in DEFAULT_PLUGINS)


async def test_declining_reaches_the_model_through_the_plugin() -> None:
    """The capability's declined result survives the round trip through the real tool call."""
    answers = iter([None])
    capabilities: list[AbstractCapability[None]] = [
        AskUser(answerer=TerminalAnswerer(full_screen=ScreenLog(), runner=lambda menu: next(answers)))
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


async def test_a_delegated_task_names_itself_before_asking() -> None:
    """The user did not prompt a child task, so its question first says which task is asking."""
    answers = iter([('Patch',)])
    output = io.StringIO()
    capabilities: list[AbstractCapability[object]] = [
        AskUser(
            answerer=TerminalAnswerer(
                full_screen=ScreenLog(), console=Console(file=output), runner=lambda menu: next(answers)
            )
        )
    ]

    async def child(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        if len(messages) == 1:
            questions = json.dumps({'questions': [APPROACH.model_dump(mode='json')]})
            yield {0: DeltaToolCall(name='ask_user_question', json_args=questions)}
        else:
            yield 'child done'

    def parent(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('delegate_task', {'agent_name': 'asker', 'task': 'ask'})])
        return ModelResponse(parts=[TextPart('parent done')])

    asker = Agent(FunctionModel(stream_function=child), deps_type=object, name='asker', capabilities=capabilities)
    agent = Agent(
        FunctionModel(parent),
        deps_type=object,
        capabilities=[SubAgents(agents=[SubAgent(asker)], agent_folders=None)],
    )
    owner = DelegationTasks()
    async with owner.opened():
        with owner.bind():
            assert (await agent.run('go', deps=None)).output == 'parent done'
    (record,) = owner.records.values()
    assert record.output == 'child done'
    assert output.getvalue() == f'Task [{record.id[:8]}] requests your input\n'


async def test_default_runner_reuses_editor_surface(question_pipe: PipeInput) -> None:
    output = io.StringIO()
    surface = PromptSurface(output=output, size=lambda: (80, 24))
    surface.write('Earlier conversation\n')
    console = Console(file=surface, width=80, height=24)
    question_pipe.send_text('2')
    response = await TerminalAnswerer(full_screen=ScreenLog(), console=console)(AskUserRequest(questions=(APPROACH,)))
    assert response.answers == (AskUserAnswer(header='Approach', selected=('Patch',)),)
    assert 'Earlier conversation' in output.getvalue()
    assert 'How should we do it?' in '\n'.join(surface.transcript.frame(width=80, height=24).rows)
    assert LEAVE not in output.getvalue(), "the editor's panel stays on screen"
    assert output.getvalue().count(ENTER) == 1


def test_terminal_cleanup_on_input_failure() -> None:
    output = io.StringIO()
    console = Console(file=output, width=80, height=24)

    def fail() -> str:
        raise OSError('input closed')

    with pytest.raises(OSError, match='input closed'):
        QuestionMenu(question=APPROACH, position=1, total=1).run(console=console, key_source=fail)
    assert '\x1b[?25h' in output.getvalue()
    assert 'How should we do it?' in output.getvalue().rsplit(LEAVE, 1)[1]


async def test_cancellation_joins_question_reader_before_releasing_screen(monkeypatch: pytest.MonkeyPatch) -> None:
    started = asyncio.Event()
    stopping = asyncio.Event()
    stopped = Event()
    input_ready = Event()
    allow_exit = Event()
    loop = asyncio.get_running_loop()
    screen = ScreenLog()

    def read_key(*, timeout: float) -> str:
        loop.call_soon_threadsafe(started.set)
        assert input_ready.wait(10)
        return ''

    def run(menu: QuestionMenu) -> tuple[str, ...] | None:
        try:
            while menu_key() != 'ctrl-c':
                pass
        finally:
            loop.call_soon_threadsafe(stopping.set)
            assert allow_exit.wait(10)
            assert screen.events == ['taken']
            stopped.set()
        return None

    async def cancel(scope: anyio.CancelScope) -> None:
        try:
            await started.wait()
            scope.cancel()
            input_ready.set()
            await stopping.wait()
            assert screen.events == ['taken']
        finally:
            input_ready.set()
            allow_exit.set()

    monkeypatch.setattr('pydantic_clai2.ui.menus.menu_worker.read_key', read_key)
    answerer = TerminalAnswerer(full_screen=screen, runner=run)
    with anyio.fail_after(10):
        async with anyio.create_task_group() as tasks:
            scope = anyio.CancelScope()
            tasks.start_soon(cancel, scope)
            with scope:
                await answerer(AskUserRequest(questions=(APPROACH,)))
            assert stopped.is_set()
            assert screen.events == ['taken', 'released']


def test_wrapped_descriptions_and_literal_markup() -> None:
    question = Question(
        header='[red]Literal',
        question='Literal question',
        options=(
            QuestionOption(label='First', description='This description needs several lines to remain readable.'),
            QuestionOption(label='[red]Second'),
        ),
    )
    menu = QuestionMenu(question=question, position=1, total=1)
    rows = menu.frame(width=30, height=30)
    text = '\n'.join(Text.from_ansi(row).plain for row in rows)
    assert 'remain readable.' in text
    assert '[red]Literal' in text
    assert '[red]Second' in text
    assert all(len(Text.from_ansi(row).plain) <= 30 for row in rows)
    menu.choose('down')
    assert any('> 2.' in row for row in menu.frame(width=12, height=6))


def test_option_description_preserves_explicit_line_breaks() -> None:
    question = Question(
        header='Steps',
        question='Which sequence?',
        options=(
            QuestionOption(label='First', description='First step\n\nSecond step'),
            QuestionOption(label='Other'),
        ),
    )
    menu = QuestionMenu(question=question, position=1, total=1)
    rows = [Text.from_ansi(row).plain for row in menu.frame(width=80, height=24)]
    assert rows[1:4] == ['> 1. First - First step', '  ', '  Second step']


def test_wide_characters_wrap_without_losing_choice_text() -> None:
    label = '中文選項需要完整顯示'
    description = '確認修復結果並保留所有文字 🎉🎉🎉'
    question = Question(
        header='Choices',
        question='Which one?',
        options=(QuestionOption(label=label, description=description), QuestionOption(label='Other')),
    )
    menu = QuestionMenu(question=question, position=1, total=1)
    rows = [Text.from_ansi(row).plain for row in menu.frame(width=16, height=80)]
    assert all(cell_len(row) <= 16 for row in rows)
    choices = ''.join(row[2:].strip() for row in rows[1:-1])
    assert label in choices
    assert description.replace(' ', '') in choices.replace(' ', '')


@pytest.mark.parametrize('question, shortcut', [(APPROACH, '3'), (TARGETS, '4')])
def test_custom_answer_editing(question: Question, shortcut: str) -> None:
    menu = QuestionMenu(question=question, position=1, total=1)
    assert menu.choose(shortcut) is None
    assert menu.editing_custom
    assert menu.choose('enter') is None
    for key in (' ', '中', 'x', 'left', 'delete', '文', 'home', 'delete', 'end', '!'):
        assert menu.choose(key) is None
    assert menu.choose('enter') == '中文!'
    rows = menu.frame(width=20, height=8)
    assert len(rows) <= 4
    assert all(cell_len(Text.from_ansi(row).plain) <= 20 for row in rows)
    assert '中文!' in ''.join(rows)
    assert 'Other (type answer)' in ''.join(menu.frame(width=80, height=24))


def test_custom_back_preserves_picks_and_draft() -> None:
    menu = QuestionMenu(question=TARGETS, position=1, total=1)
    menu.choose('1')
    menu.choose('4')
    menu.choose('x')
    menu.choose('escape')
    assert not menu.editing_custom
    assert menu.selected == {0}
    assert menu.choose('3') == ('api.py',)
    menu.choose('4')
    assert menu.choose('enter') == 'x'


@pytest.mark.parametrize(
    'keys, expected',
    [
        (['3', 'x', 'enter'], 'x'),
        (['3', 'x', 'escape', '2'], ('Patch',)),
        (['3', 'escape', 'escape'], None),
        (['3', 'ctrl-c'], None),
    ],
)
def test_custom_inline_lifecycle(keys: list[str], expected: tuple[str, ...] | str | None) -> None:
    output = io.StringIO()
    surface = PromptSurface(output=output, size=lambda: (80, 24))
    surface.write('Previous conversation\n')
    menu = QuestionMenu(question=APPROACH, position=1, total=1)
    assert menu.run(console=Console(file=surface), key_source=iter(keys).__next__) == expected
    assert LEAVE not in output.getvalue()
    assert 'Previous conversation' in output.getvalue()
    assert MODES_OFF in output.getvalue(), 'input modes return to the editor, the screen stays'


@pytest.mark.parametrize('enter', ['\r', '\n'])
async def test_custom_answer_reaches_model_through_inline_picker(question_pipe: PipeInput, enter: str) -> None:
    question_pipe.send_text(f'3Use another approach{enter}')
    screen = ScreenLog()
    output = io.StringIO()
    answerer = TerminalAnswerer(full_screen=screen, console=Console(file=output))

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('ask_user_question', {'questions': [APPROACH.model_dump()]})])
        return ModelResponse(parts=[TextPart('done')])

    with anyio.fail_after(5):
        result = await Agent(FunctionModel(respond), capabilities=[AskUser(answerer=answerer)]).run('go')
    returns = [p for m in result.all_messages() for p in m.parts if isinstance(p, ToolReturnPart)]
    assert returns[0].content == {'Approach': ['Use another approach']}
    assert screen.events == ['taken', 'released']
    console = Console(file=output)
    console.print(
        render_answer(
            AskUserAnsweredEvent(
                request_id='custom',
                response=AskUserResponse(
                    answers=(AskUserAnswer(header='Approach', custom_answer='[red]Custom answer'),)
                ),
            )
        )
    )
    assert 'Approach: [red]Custom answer' in output.getvalue()


async def test_bracketed_paste_is_atomic_and_does_not_answer_next_question(question_pipe: PipeInput) -> None:
    question_pipe.send_text('3\x1b[200~first\r\nsecond\t中文\x1b[201~\r2')
    response = await TerminalAnswerer(full_screen=ScreenLog(), console=Console(file=io.StringIO()))(
        AskUserRequest(questions=(APPROACH, APPROACH.model_copy(update={'header': 'Next'})))
    )
    assert response.answers == (
        AskUserAnswer(header='Approach', custom_answer='first\nsecond    中文'),
        AskUserAnswer(header='Next', selected=('Patch',)),
    )


def test_pasted_shortcuts_are_literal_and_history_search_is_disabled() -> None:
    menu = QuestionMenu(question=APPROACH, position=1, total=1)
    assert menu.choose(Paste(text='3\n')) is None
    assert not menu.editing_custom
    menu.choose('3')
    menu.choose('ctrl-r')
    assert menu.custom.search is None
    menu.choose(Paste(text='first\nsecond'))
    assert menu.choose('enter') == 'first\nsecond'


async def test_question_decoder_eof_declines(question_pipe: PipeInput) -> None:
    question_pipe.close()
    response = await TerminalAnswerer(full_screen=ScreenLog(), console=Console(file=io.StringIO()))(
        AskUserRequest(questions=(APPROACH,))
    )
    assert response.cancelled


async def test_custom_decoder_cancellation_detaches_before_editor_resumes(
    question_pipe: PipeInput, monkeypatch: pytest.MonkeyPatch
) -> None:
    screen = ScreenLog()
    painted = asyncio.Event()
    loop = asyncio.get_running_loop()
    paint = PromptSurface.paint
    frames = 0
    attach = question_pipe.attach

    @contextmanager
    def tracked_attach(callback: Callable[[], None]) -> Generator[None]:
        with attach(callback):
            screen.events.append('attached')
            try:
                yield
            finally:
                screen.events.append('detached')

    def track_paint(surface: PromptSurface, rows: tuple[str, ...]) -> None:
        nonlocal frames
        paint(surface, rows)
        frames += 1
        # The third frame follows selection and an idle input poll in the custom editor.
        if frames == 3:
            loop.call_soon_threadsafe(painted.set)

    async def cancel(scope: anyio.CancelScope) -> None:
        await painted.wait()
        assert screen.events == ['taken', 'attached']
        scope.cancel()

    monkeypatch.setattr(question_pipe, 'attach', tracked_attach)
    monkeypatch.setattr(PromptSurface, 'paint', track_paint)
    question_pipe.send_text('3')
    with anyio.fail_after(10):
        async with anyio.create_task_group() as tasks:
            scope = anyio.CancelScope()
            tasks.start_soon(cancel, scope)
            with scope:
                await TerminalAnswerer(full_screen=screen, console=Console(file=io.StringIO()))(
                    AskUserRequest(questions=(APPROACH,))
                )
            assert scope.cancelled_caught
    assert screen.events == ['taken', 'attached', 'detached', 'released']


async def test_wheel_and_page_keys_scroll_the_transcript_while_answering(question_pipe: PipeInput) -> None:
    terminal = SurfaceTerminal(width=80, height=24)
    surface = PromptSurface(output=terminal, size=lambda: (80, 24))
    for index in range(60):
        surface.write(f'line {index}\n')
    # Wheel up, a click, a drag that copies, then a page back and forth: none of them type into the answer.
    question_pipe.send_text('3\x1b[<64;10;5M\x1b[<0;10;5M\x1b[<0;1;1M\x1b[<32;7;1M\x1b[<0;7;1m\x1b[5~\x1b[6~x\r')
    # The fake terminal is a TTY, so an auto colour system would downgrade, and Rich caches that
    # downgrade on the shared parsed style, leaking into later truecolor tests on this worker.
    console = Console(file=surface, width=80, height=24, color_system=None)
    response = await TerminalAnswerer(full_screen=ScreenLog(), console=console)(AskUserRequest(questions=(APPROACH,)))
    assert response.answers == (AskUserAnswer(header='Approach', custom_answer='x'),)
    assert surface.view.anchor is not None, 'the view stays where the user scrolled'
    assert '\x1b]52;c;' in terminal.getvalue(), 'the drag copied the transcript'
    assert 'How should we do it?' not in terminal.lines()
    assert 'line 50' in terminal.lines()
