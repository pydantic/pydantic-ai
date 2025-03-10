from __future__ import annotations as _annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import cast

from typing_inspection.introspection import get_literal_values

from pydantic_ai.exceptions import UserError
from pydantic_ai.models import KnownModelName
from pydantic_graph.nodes import End

try:
    import argcomplete
    from prompt_toolkit import PromptSession
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory, Suggestion
    from prompt_toolkit.buffer import Buffer
    from prompt_toolkit.document import Document
    from prompt_toolkit.history import FileHistory
    from rich.console import Console, ConsoleOptions, RenderResult
    from rich.live import Live
    from rich.markdown import CodeBlock, Markdown
    from rich.status import Status
    from rich.syntax import Syntax
    from rich.text import Text
except ImportError as _import_error:
    raise ImportError(
        'Please install `rich`, `prompt-toolkit` and `argcomplete` to use the PydanticAI CLI, '
        "you can use the `cli` optional group — `pip install 'pydantic-ai-slim[cli]'`"
    ) from _import_error

from pydantic_ai.agent import Agent
from pydantic_ai.messages import ModelMessage, PartDeltaEvent, TextPartDelta

__version__ = version('pydantic-ai')


class SimpleCodeBlock(CodeBlock):
    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:  # pragma: no cover
        code = str(self.text).rstrip()
        yield Text(self.lexer_name, style='dim')
        yield Syntax(code, self.lexer_name, theme=self.theme, background_color='default', word_wrap=True)
        yield Text(f'/{self.lexer_name}', style='dim')


Markdown.elements['fence'] = SimpleCodeBlock


def cli(args_list: Sequence[str] | None = None) -> int:  # noqa: C901  # pragma: no cover
    parser = argparse.ArgumentParser(
        prog='pai',
        description=f"""\
PydanticAI CLI v{__version__}\n\n

Special prompt:
* `/exit` - exit the interactive mode
* `/markdown` - show the last markdown output of the last question
* `/multiline` - toggle multiline mode
""",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('prompt', nargs='?', help='AI Prompt, if omitted fall into interactive mode')
    parser.add_argument(
        '--model',
        nargs='?',
        help='Model to use, it should be "<provider>:<model>" e.g. "openai:gpt-4o". If omitted it will default to "openai:gpt-4o"',
        default='openai:gpt-4o',
    ).completer = argcomplete.ChoicesCompleter(list(get_literal_values(KnownModelName)))  # type: ignore[reportPrivateUsage]
    parser.add_argument('--no-stream', action='store_true', help='Whether to stream responses from OpenAI')
    parser.add_argument('--version', action='store_true', help='Show version and exit')

    argcomplete.autocomplete(parser)
    args = parser.parse_args(args_list)

    console = Console()
    console.print(f'pai - PydanticAI CLI v{__version__}', style='green bold', highlight=False)
    if args.version:
        return 0

    now_utc = datetime.now(timezone.utc)
    tzname = now_utc.astimezone().tzinfo.tzname(now_utc)  # type: ignore
    try:
        agent = Agent(
            model=args.model or 'openai:gpt-4o',
            system_prompt=f"""\
    Help the user by responding to their request, the output should be concise and always written in markdown.
    The current date and time is {datetime.now()} {tzname}.
    The user is running {sys.platform}.""",
        )
    except UserError:
        console.print(f'[red]Invalid model "{args.model}"[/red]')
        return 1

    stream = not args.no_stream

    if prompt := cast(str, args.prompt):
        try:
            asyncio.run(ask_agent(agent, prompt, stream, console))
        except KeyboardInterrupt:
            pass
        return 0

    history = Path.home() / '.pai-prompt-history.txt'
    session = PromptSession(history=FileHistory(str(history)))  # type: ignore
    multiline = False
    messages: list[ModelMessage] = []

    while True:
        try:
            auto_suggest = CustomAutoSuggest(['/markdown', '/multiline', '/exit'])
            text = cast(str, session.prompt('pai ➤ ', auto_suggest=auto_suggest, multiline=multiline))
        except (KeyboardInterrupt, EOFError):
            return 0

        if not text.strip():
            continue

        ident_prompt = text.lower().strip(' ').replace(' ', '-').lstrip(' ')
        if ident_prompt == '/markdown':
            try:
                parts = messages[-1].parts
            except IndexError:
                console.print('[dim]No markdown output available.[/dim]')
                continue
            for part in parts:
                if part.part_kind == 'text':
                    last_content = part.content
                    console.print('[dim]Last markdown output of last question:[/dim]\n')
                    console.print(Syntax(last_content, lexer='markdown', background_color='default'))

            continue
        if ident_prompt == '/multiline':
            multiline = not multiline
            if multiline:
                console.print(
                    'Enabling multiline mode. '
                    '[dim]Press [Meta+Enter] or [Esc] followed by [Enter] to accept input.[/dim]'
                )
            else:
                console.print('Disabling multiline mode.')
            continue
        if ident_prompt == '/exit':
            console.print('[dim]Exiting…[/dim]')
            return 0

        try:
            messages = asyncio.run(ask_agent(agent, text, stream, console, messages))
        except KeyboardInterrupt:
            return 0


async def ask_agent(
    agent: Agent,
    prompt: str,
    stream: bool,
    console: Console,
    messages: list[ModelMessage] | None = None,
) -> list[ModelMessage]:  # pragma: no cover
    status: None | Status = Status('[dim]Working on it…[/dim]', console=console)
    live = Live('', refresh_per_second=15, console=console)
    status.start()

    async with agent.iter(prompt, message_history=messages) as agent_run:
        console.print('\nResponse:', style='green')

        content: str = ''
        interrupted = False
        try:
            node = agent_run.next_node
            while not isinstance(node, End):
                node = await agent_run.next(node)
                if Agent.is_model_request_node(node):
                    async with node.stream(agent_run.ctx) as handle_stream:
                        # NOTE(Marcelo): It took me a lot of time to figure out how to stop `status` and start `live`
                        # in a context manager, so I had to do it manually with `stop` and `start` methods.
                        # PR welcome to simplify this code.
                        if status is not None:
                            status.stop()
                            status = None
                        if not live.is_started:
                            live.start()
                        async for event in handle_stream:
                            if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                                if stream:
                                    content += event.delta.content_delta
                                    live.update(Markdown(content))
        except KeyboardInterrupt:
            interrupted = True
        finally:
            live.stop()

        if interrupted:
            console.print('[dim]Interrupted[/dim]')

        assert agent_run.result
        if not stream:
            content = agent_run.result.data
            console.print(Markdown(content))
        return agent_run.result.all_messages()


class CustomAutoSuggest(AutoSuggestFromHistory):
    def __init__(self, special_suggestions: list[str] | None = None):  # pragma: no cover
        super().__init__()
        self.special_suggestions = special_suggestions or []

    def get_suggestion(self, buffer: Buffer, document: Document) -> Suggestion | None:  # pragma: no cover
        # Get the suggestion from history
        suggestion = super().get_suggestion(buffer, document)

        # Check for custom suggestions
        text = document.text_before_cursor.strip()
        for special in self.special_suggestions:
            if special.startswith(text):
                return Suggestion(special[len(text) :])
        return suggestion


def app():  # pragma: no cover
    sys.exit(cli())
