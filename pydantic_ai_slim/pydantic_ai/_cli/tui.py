from __future__ import annotations

from asyncio import Queue
from dataclasses import dataclass
from pathlib import Path
from string import Template

from prompt_toolkit.history import FileHistory
from textual import containers, getters, on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.geometry import clamp
from textual.message import Message
from textual.reactive import reactive, var
from textual.screen import Screen
from textual.suggester import SuggestFromList
from textual.widget import Widget
from textual.widgets import Footer, Input, Label, Markdown, Static, TextArea
from textual.widgets.input import Selection

from pydantic_ai._run_context import AgentDepsT
from pydantic_ai.agent import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.output import OutputDataT

DEFAULT_THEME = 'nord'


class CLAIApp(App[None]):
    """The CLA TUI app."""

    BINDING_GROUP_TITLE = 'App'
    CSS_PATH = 'clai.tcss'

    BINDINGS = [Binding('ctrl+c', 'app.quit', 'Exit', priority=True)]

    def __init__(
        self,
        agent: Agent[AgentDepsT, OutputDataT],
        history_path: Path,
        prompt: str | None = None,
        title: str | None = None,
        *,
        # TODO(Marcelo)We need to find a way to expose a way to create the deps object.
        _deps: AgentDepsT = None,
    ):
        super().__init__()
        self._agent = agent
        self.history_path = history_path
        self.title = title or 'Pydantic AI CLI'
        self._prompt = prompt
        self._deps = _deps

    def on_load(self) -> None:
        """Called before application mode."""
        # Set the default theme here to avoid flash of different theme
        self.theme = DEFAULT_THEME

    def get_default_screen(self) -> MainScreen:
        return MainScreen(self._agent, self.history_path, self.title, prompt=self._prompt, deps=self._deps)


HELP = Template("""\
## $title

- **Powered by Pydantic AI**

    The Python agent framework designed to make it less painful to build production grade applications with Generative AI.

| Command | Purpose |
| --- | --- |
| `/markdown` | Show markdown output of last question. |
| `/multiline` |  Enable multiline mode. |
| `/exit` | Exit CLAI. |
""")


class ErrorMessage(Static):
    """An error message for the user."""


class Response(Markdown):
    """Response from the agent."""


class UserText(containers.HorizontalGroup):
    """Copy of what the user prompted."""

    def __init__(self, prompt: str) -> None:
        self._prompt = prompt
        super().__init__()

    def compose(self) -> ComposeResult:
        yield Label('clai ➤', id='prompt')
        yield Label(self._prompt, id='message')


class PromptInput(Input):
    """Custom prompt to disable maximize."""

    BINDING_GROUP_TITLE = 'Prompt'
    ALLOW_MAXIMIZE = False


class PromptTextArea(TextArea):
    """A custom textarea."""

    BINDING_GROUP_TITLE = 'Prompt'


class Prompt(containers.HorizontalGroup, can_focus=False):
    """Takes input from the user."""

    BINDINGS = [
        Binding('shift+up', 'history(-1)', 'History up', priority=True),
        Binding('shift+down', 'history(+1)', 'History down', priority=True),
        Binding('ctrl+j', 'submit', 'Submit prompt', key_display='shift+⏎', priority=True),
        Binding('escape', 'escape', 'Exit multiline'),
    ]

    history_position = var(0, bindings=True)
    multiline = reactive(False)
    input = getters.query_one('#prompt-input', Input)
    text_area = getters.query_one('#prompt-textarea', TextArea)

    @dataclass
    class Submitted(Message):
        """Prompt text was submitted."""

        value: str

    def __init__(self, history: FileHistory, id: str | None = None) -> None:
        self.history = history
        self.history_strings: list[str] = []
        self.edit_prompt = ''
        super().__init__(id=id)

    def compose(self) -> ComposeResult:
        yield Label('clai ➤', id='prompt')
        yield PromptInput(
            id='prompt-input',
            placeholder='Ask me anything',
            suggester=SuggestFromList(
                [
                    '/markdown',
                    '/multiline',
                    '/exit',
                ]
            ),
        )
        yield PromptTextArea(
            id='prompt-textarea',
            language='markdown',
            highlight_cursor_line=False,
        )

    def watch_multiline(self, multiline: bool) -> None:
        if multiline:
            self.input.display = False
            self.text_area.display = True
            self.text_area.load_text(self.input.value)
            self.text_area.focus()
        else:
            self.input.display = True
            self.text_area.display = False
            self.input.value = self.text_area.text.partition('\n')[0]
            self.input.focus()

    @property
    def value(self) -> str:
        """Value of prompt."""
        if self.multiline:
            return self.text_area.text
        else:
            return self.input.value

    @value.setter
    def value(self, value: str) -> None:
        multiline = '\n' in value
        self.multiline = multiline
        if multiline:
            self.text_area.load_text(value)
        else:
            self.input.value = value
            self.input.selection = Selection.cursor(len(value))

    def clear(self) -> None:
        with self.prevent(Input.Changed):
            self.input.clear()
        with self.prevent(TextArea.Changed):
            self.text_area.load_text('')

    async def action_history(self, direction: int) -> None:
        if self.history_position == 0:
            self.history_strings.clear()
            async for prompt in self.history.load():
                if prompt.strip():
                    self.history_strings.append(prompt)
            self.history_strings.reverse()
        self.history_position = self.history_position + direction

    def action_submit(self) -> None:
        self.post_message(self.Submitted(self.text_area.text))
        self.clear()
        self.action_escape()
        self.history_position = 0

    def action_escape(self) -> None:
        self.history_position = 0
        self.multiline = False

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action == 'history':
            if parameters[0] == +1 and self.history_position == 0:
                return None
            if parameters[0] == -1 and self.history_strings and self.history_position == -len(self.history_strings):
                return None
        if action in ('submit', 'escape'):
            return self.multiline
        return True

    def validate_history_position(self, history_position: int) -> int:
        return clamp(history_position, -len(self.history_strings), 0)

    async def watch_history_position(self, previous_position: int, position: int) -> None:
        if previous_position == 0:
            self.edit_prompt = self.value
        if position == 0:
            self.value = self.edit_prompt
        elif position < 0:
            self.value = self.history_strings[position]

    @on(Input.Submitted)
    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.post_message(self.Submitted(event.value))
        self.clear()
        self.history_position = 0


class Contents(containers.VerticalScroll):
    """The conversation contents."""

    BINDING_GROUP_TITLE = 'Conversation'
    BINDINGS = [Binding('tab', 'screen.focus-next', 'Focus prompt')]


class Conversation(containers.Vertical):
    """The conversation with the AI."""

    contents = getters.query_one('#contents', containers.VerticalScroll)
    prompt = getters.query_one(Prompt)

    def __init__(self, history: FileHistory, title: str) -> None:
        self.history = history
        self.title = title
        super().__init__()

    def compose(self) -> ComposeResult:
        yield Contents(id='contents')
        yield Prompt(self.history, id='prompt')

    def get_last_markdown_source(self) -> str | None:
        """Get the source of the last markdown response, or `None` if there is no markdown response."""
        for child in reversed(self.contents.children):
            if isinstance(child, Markdown):
                return child.source
        return None

    async def on_mount(self) -> None:
        await self.post(Response(HELP.safe_substitute(title=self.title)))

    async def post(self, widget: Widget) -> None:
        await self.contents.mount(widget)
        self.contents.anchor()

    async def post_prompt(self, prompt: str) -> None:
        await self.post(UserText(prompt))


class MainScreen(Screen[None]):
    """Main screen containing conversation."""

    app: CLAIApp

    BINDING_GROUP_TITLE = 'Screen'
    AUTO_FOCUS = 'Conversation Prompt Input'

    conversation = getters.query_one(Conversation)

    def __init__(
        self,
        agent: Agent[AgentDepsT, OutputDataT],
        history_path: Path,
        title: str,
        *,
        prompt: str | None = None,
        deps: AgentDepsT = None,
    ):
        self.agent = agent
        self.prompt = prompt
        self.messages: list[ModelMessage] = []
        self.history = FileHistory(history_path)
        self.deps = deps
        super().__init__()
        self.title = title

    def compose(self) -> ComposeResult:
        yield Conversation(self.history, self.title or 'Pydantic AI CLI')
        yield Footer()

    async def on_mount(self) -> None:
        """Runs when the widget is mounted."""
        # Initialize the prompt queue
        self.prompt_queue: Queue[str | None] = Queue(maxsize=10)
        self.run_response_queue()
        if self.prompt:
            # Send initial prompt
            await self.conversation.post_prompt(self.prompt)
            await self.ask_agent(self.prompt)

    async def on_unmount(self) -> None:
        """Called when the app exits."""
        # Tell the response queue task to finish up
        await self.prompt_queue.put(None)

    @on(Prompt.Submitted)
    async def on_conversation_prompt(self, event: Prompt.Submitted) -> None:
        """Called when the user submits a prompt."""
        prompt = event.value.strip()
        if not prompt:
            self.app.bell()
            return
        self.history.append_string(prompt)
        if prompt.startswith('/'):
            await self.process_slash(prompt)
        else:
            await self.conversation.post_prompt(prompt)
            await self.ask_agent(prompt)

    async def process_slash(self, prompt: str) -> None:
        prompt = prompt.strip()
        if prompt == '/markdown':
            markdown = self.conversation.get_last_markdown_source()
            if not markdown:
                await self.conversation.post(ErrorMessage('No markdown to display'))
            else:
                await self.conversation.post(Static(markdown))
        elif prompt == '/multiline':
            self.conversation.prompt.multiline = not self.conversation.prompt.multiline
        elif prompt == '/exit':
            self.app.exit()
        else:
            await self.conversation.post(ErrorMessage(f'Unknown command: {prompt!r}'))

    async def ask_agent(self, prompt: str) -> None:
        """Send the prompt to the agent."""
        await self.prompt_queue.put(prompt)

    async def post_response(self) -> Response:
        """Post a response, returns a callable to append markdown."""
        response = Response()
        response.display = False
        await self.conversation.post(response)
        return response

    @work
    async def run_response_queue(self) -> None:
        """Listens to the prompt queue, posts prompts, and streams the response."""
        while (prompt := await self.prompt_queue.get()) is not None:
            response = await self.post_response()
            markdown_stream = Markdown.get_stream(response)
            try:
                async with self.agent.iter(prompt, message_history=self.messages, deps=self.deps) as agent_run:
                    async for node in agent_run:
                        if Agent.is_model_request_node(node):
                            async with node.stream(agent_run.ctx) as handle_stream:
                                async for fragment in handle_stream.stream_text(delta=True, debounce_by=None):
                                    await markdown_stream.write(fragment)
                                    response.display = True
                    assert agent_run.result is not None
                    self.messages[:] = agent_run.result.all_messages()
            finally:
                await markdown_stream.stop()
