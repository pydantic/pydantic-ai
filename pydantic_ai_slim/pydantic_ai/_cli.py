from __future__ import annotations as _annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from contextlib import ExitStack
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import cast

from typing_inspection.introspection import get_literal_values

from pydantic_ai.agent import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, PartDeltaEvent, TextPartDelta
from pydantic_ai.models import KnownModelName

try:
    import argcomplete
    from prompt_toolkit import PromptSession
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory, Suggestion
    from prompt_toolkit.buffer import Buffer
    from prompt_toolkit.document import Document
    from prompt_toolkit.history import FileHistory
    from rich.console import Console, ConsoleOptions, RenderResult
    from rich.live import Live
    from rich.markdown import CodeBlock, Heading, Markdown
    from rich.status import Status
    from rich.style import Style
    from rich.syntax import Syntax
    from rich.text import Text
except ImportError as _import_error:
    raise ImportError(
        'Please install `rich`, `prompt-toolkit` and `argcomplete` to use the PydanticAI CLI, '
        'you can use the `cli` optional group — `pip install "pydantic-ai-slim[cli]"`'
    ) from _import_error


__version__ = version('pydantic-ai-slim')


class SimpleCodeBlock(CodeBlock):
    """Customised code blocks in markdown.

    This avoids a background color which messes up copy-pasting and sets the language name as dim prefix and suffix.
    """

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:  # pragma: no cover
        code = str(self.text).rstrip()
        yield Text(self.lexer_name, style='dim')
        yield Syntax(code, self.lexer_name, theme=self.theme, background_color='default', word_wrap=True)
        yield Text(f'/{self.lexer_name}', style='dim')


class LeftHeading(Heading):
    """Customised headings in markdown to stop centering and prepend markdown style hashes."""

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:  # pragma: no cover
        # note we use `Style(bold=True)` not `self.style_name` here to disable underlining which is ugly IMHO
        yield Text(f'{"#" * int(self.tag[1:])} {self.text.plain}', style=Style(bold=True))


Markdown.elements.update(
    fence=SimpleCodeBlock,
    heading_open=LeftHeading,
)


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
    arg = parser.add_argument(
        '-m',
        '--model',
        nargs='?',
        help='Model to use, in format "<provider>:<model>" e.g. "openai:gpt-4o". Defaults to "openai:gpt-4o".',
        default='openai:gpt-4o',
    )
    # we don't want to autocomplete or list models that don't include the provider,
    # e.g. we want to show `openai:gpt-4o` but not `gpt-4o`
    qualified_model_names = [n for n in get_literal_values(KnownModelName.__value__) if ':' in n]
    arg.completer = argcomplete.ChoicesCompleter(qualified_model_names)  # type: ignore[reportPrivateUsage]
    parser.add_argument(
        '-l',
        '--list-models',
        action='store_true',
        help='List all available models and exit',
    )
    parser.add_argument(
        '-t',
        '--code-theme',
        nargs='?',
        help='Which colors to use for code, can be "dark", "light" or any theme from pygments.org/styles/. Defaults to "monokai".',
        default='monokai',
    )
    parser.add_argument('--no-stream', action='store_true', help='Whether to stream responses from the model')
    parser.add_argument('--version', action='store_true', help='Show version and exit')

    argcomplete.autocomplete(parser)
    args = parser.parse_args(args_list)

    console = Console()
    console.print(
        f'[green]pai - PydanticAI CLI v{__version__} using[/green] [magenta]{args.model}[/magenta]', highlight=False
    )
    if args.version:
        return 0
    if args.list_models:
        console.print('Available models:', style='green bold')
        for model in qualified_model_names:
            console.print(f'  {model}', highlight=False)
        return 0

    now_utc = datetime.now(timezone.utc)
    tzname = now_utc.astimezone().tzinfo.tzname(now_utc)  # type: ignore
    try:
        agent = Agent(
            model=args.model,
            system_prompt=f"""\
    Help the user by responding to their request, the output should be concise and always written in markdown.
    The current date and time is {datetime.now()} {tzname}.
    The user is running {sys.platform}.""",
        )
    except UserError:
        console.print(f'[red]Invalid model "{args.model}"[/red]')
        return 1

    stream = not args.no_stream
    if args.code_theme == 'light':
        code_theme = 'default'
    elif args.code_theme == 'dark':
        code_theme = 'monokai'
    else:
        code_theme = args.code_theme

    if prompt := cast(str, args.prompt):
        try:
            asyncio.run(ask_agent(agent, prompt, stream, console, code_theme))
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
        if ident_prompt.startswith('/'):
            if ident_prompt == '/markdown':
                try:
                    parts = messages[-1].parts
                except IndexError:
                    console.print('[dim]No markdown output available.[/dim]')
                    continue
                console.print('[dim]Markdown output of last question:[/dim]\n')
                for part in parts:
                    if part.part_kind == 'text':
                        console.print(
                            Syntax(
                                part.content,
                                lexer='markdown',
                                theme=code_theme,
                                word_wrap=True,
                                background_color='default',
                            )
                        )

            elif ident_prompt == '/multiline':
                multiline = not multiline
                if multiline:
                    console.print(
                        'Enabling multiline mode. '
                        '[dim]Press [Meta+Enter] or [Esc] followed by [Enter] to accept input.[/dim]'
                    )
                else:
                    console.print('Disabling multiline mode.')
            elif ident_prompt == '/exit':
                console.print('[dim]Exiting…[/dim]')
                return 0
            else:
                console.print(f'[red]Unknown command[/red] [magenta]`{ident_prompt}`[/magenta]')
        else:
            try:
                messages = asyncio.run(ask_agent(agent, text, stream, console, code_theme, messages))
            except KeyboardInterrupt:
                console.print('[dim]Interrupted[/dim]')
                messages = []


async def ask_agent(
    agent: Agent,
    prompt: str,
    stream: bool,
    console: Console,
    code_theme: str,
    messages: list[ModelMessage] | None = None,
) -> list[ModelMessage]:  # pragma: no cover
    status = Status('[dim]Working on it…[/dim]', console=console)

    if not stream:
        with status:
            result = await agent.run(prompt, message_history=messages)
        content = result.data
        console.print(Markdown(content, code_theme=code_theme))
        return result.all_messages()

    with status, ExitStack() as stack:
        async with agent.iter(prompt, message_history=messages) as agent_run:
            live = Live('', refresh_per_second=15, console=console, vertical_overflow='visible')
            content: str = ''
            async for node in agent_run:
                if Agent.is_model_request_node(node):
                    async with node.stream(agent_run.ctx) as handle_stream:
                        status.stop()  # stopping multiple times is idempotent
                        stack.enter_context(live)  # entering multiple times is idempotent

                        async for event in handle_stream:
                            if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                                content += event.delta.content_delta
                                live.update(Markdown(content, code_theme=code_theme))

        assert agent_run.result is not None
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
