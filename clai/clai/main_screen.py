from __future__ import annotations

import asyncio
from asyncio import Queue
from dataclasses import dataclass

from textual import containers, getters, on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.message import Message
from textual.screen import Screen
from textual.suggester import SuggestFromList
from textual.widget import Widget
from textual.widgets import Footer, Input, Label, Markdown

from pydantic_ai import __version__
from pydantic_ai._run_context import AgentDepsT
from pydantic_ai.agent import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.output import OutputDataT

HELP = f"""\
## Pydantic AI TUI **v{__version__}**


| Prompt | Purpose |
| --- | --- |
| `/markdown` | Show markdown output of last question. |
|`/multiline` |  Enable multiline mode. |
| `/exit` | Exit CLAI. |


"""


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


class Prompt(containers.HorizontalGroup):
    """Takes input from the user."""

    def compose(self) -> ComposeResult:
        yield Label('clai ➤', id='prompt')
        yield PromptInput(
            id='prompt-input',
            suggester=SuggestFromList(
                [
                    '/markdown',
                    '/multiline',
                    '/exit',
                ]
            ),
        )


class Contents(containers.VerticalScroll):
    BINDING_GROUP_TITLE = 'Conversation'

    BINDINGS = [Binding('tab', 'screen.focus-next', 'Focus prompt')]


class Conversation(containers.Vertical):
    """The conversation with the AI."""

    contents = getters.query_one('#contents', containers.VerticalScroll)

    @dataclass
    class Prompt(Message):
        """A prompt from the user."""

        prompt: str

    def compose(self) -> ComposeResult:
        with Contents(id='contents'):
            pass

        yield Prompt(id='prompt')

    async def on_mount(self) -> None:
        await self.post_help()

    async def post_help(self) -> None:
        await self.post(Response(HELP))

    async def post(self, widget: Widget) -> None:
        await self.contents.mount(widget)
        self.contents.anchor()

    async def post_prompt(self, prompt: str) -> None:
        await self.post(UserText(prompt))

    @on(Input.Submitted)
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        self.post_message(self.Prompt(event.input.value))
        event.input.clear()


class MainScreen(Screen[None]):
    """Main screen containing conversation."""

    BINDING_GROUP_TITLE = 'Screen'
    AUTO_FOCUS = 'Conversation Prompt Input'

    conversation = getters.query_one(Conversation)

    def __init__(self, agent: Agent[AgentDepsT, OutputDataT], prompt: str | None = None):
        self.agent = agent
        self.prompt = prompt
        self.messages: list[ModelMessage] = []
        super().__init__()

    def compose(self) -> ComposeResult:
        yield Conversation()
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

    @on(Conversation.Prompt)
    async def on_conversation_prompt(self, event: Conversation.Prompt) -> None:
        """Called when the user submits a prompt."""
        prompt = event.prompt
        await self.conversation.post_prompt(prompt)
        await self.ask_agent(prompt)

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
                async with self.agent.iter(prompt, message_history=self.messages) as agent_run:
                    async for node in agent_run:
                        if Agent.is_model_request_node(node):
                            async with node.stream(agent_run.ctx) as handle_stream:
                                async for fragment in handle_stream.stream_text(delta=True, debounce_by=None):
                                    await markdown_stream.write(fragment)
                                    response.display = True
                    self.messages[:] = agent_run.result.all_messages()
            finally:
                await markdown_stream.stop()
