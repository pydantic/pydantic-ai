from __future__ import annotations as _annotations

import argparse
import asyncio
import importlib
import os
import sys
from asyncio import CancelledError
from collections.abc import Sequence
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from typing_inspection.introspection import get_literal_values

from . import __version__
from ._run_context import AgentDepsT
from .agent import AbstractAgent, Agent
from .exceptions import UserError
from .messages import ModelMessage, ModelResponse
from .models import KnownModelName, infer_model
from .output import OutputDataT

try:
    import argcomplete
    import pyperclip
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
        'Please install `rich`, `prompt-toolkit`, `pyperclip` and `argcomplete` to use the Pydantic AI CLI, '
        'you can use the `cli` optional group — `pip install "pydantic-ai-slim[cli]"`'
    ) from _import_error


__all__ = 'cli', 'cli_exit'


PYDANTIC_AI_HOME = Path.home() / '.pydantic-ai'
"""The home directory for Pydantic AI CLI.

This folder is used to store the prompt history and configuration.
"""

PROMPT_HISTORY_FILENAME = 'prompt-history.txt'


class SimpleCodeBlock(CodeBlock):
    """Customized code blocks in markdown.

    This avoids a background color which messes up copy-pasting and sets the language name as dim prefix and suffix.
    """

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        code = str(self.text).rstrip()
        yield Text(self.lexer_name, style='dim')
        yield Syntax(code, self.lexer_name, theme=self.theme, background_color='default', word_wrap=True)
        yield Text(f'/{self.lexer_name}', style='dim')


class LeftHeading(Heading):
    """Customized headings in markdown to stop centering and prepend markdown style hashes."""

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        # note we use `Style(bold=True)` not `self.style_name` here to disable underlining which is ugly IMHO
        yield Text(f'{"#" * int(self.tag[1:])} {self.text.plain}', style=Style(bold=True))


Markdown.elements.update(
    fence=SimpleCodeBlock,
    heading_open=LeftHeading,
)


cli_agent = Agent()


@cli_agent.system_prompt
def cli_system_prompt() -> str:
    now_utc = datetime.now(timezone.utc)
    tzinfo = now_utc.astimezone().tzinfo
    tzname = tzinfo.tzname(now_utc) if tzinfo else ''
    return f"""\
Help the user by responding to their request, the output should be concise and always written in markdown.
The current date and time is {datetime.now()} {tzname}.
The user is running {sys.platform}."""


def cli_exit(prog_name: str = 'pai'):  # pragma: no cover
    """Run the CLI and exit."""
    sys.exit(cli(prog_name=prog_name))


def cli(  # noqa: C901
    args_list: Sequence[str] | None = None, *, prog_name: str = 'pai', default_model: str = 'openai:gpt-5'
) -> int:
    """Run the CLI and return the exit code for the process."""
    parser = argparse.ArgumentParser(
        prog=prog_name,
        description=f"""\
Pydantic AI CLI v{__version__}\n\n

Special prompts:
* `/exit` - exit the interactive mode (ctrl-c and ctrl-d also work)
* `/markdown` - show the last markdown output of the last question
* `/multiline` - toggle multiline mode
* `/cp` - copy the last response to clipboard
""",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('prompt', nargs='?', help='AI Prompt, if omitted fall into interactive mode')
    arg = parser.add_argument(
        '-m',
        '--model',
        nargs='?',
        help=f'Model to use, in format "<provider>:<model>" e.g. "openai:gpt-5" or "anthropic:claude-sonnet-4-5". Defaults to "{default_model}".',
    )
    # we don't want to autocomplete or list models that don't include the provider,
    # e.g. we want to show `openai:gpt-4o` but not `gpt-4o`
    qualified_model_names = [n for n in get_literal_values(KnownModelName.__value__) if ':' in n]
    arg.completer = argcomplete.ChoicesCompleter(qualified_model_names)  # type: ignore[reportPrivateUsage]
    parser.add_argument(
        '-a',
        '--agent',
        help='Custom Agent to use, in format "module:variable", e.g. "mymodule.submodule:my_agent"',
    )
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
        help='Which colors to use for code, can be "dark", "light" or any theme from pygments.org/styles/. Defaults to "dark" which works well on dark terminals.',
        default='dark',
    )
    parser.add_argument('--no-stream', action='store_true', help='Disable streaming from the model')
    parser.add_argument('--version', action='store_true', help='Show version and exit')

    argcomplete.autocomplete(parser)
    args = parser.parse_args(args_list)

    console = Console()
    name_version = f'[green]{prog_name} - Pydantic AI CLI v{__version__}[/green]'
    if args.version:
        console.print(name_version, highlight=False)
        return 0
    if args.list_models:
        console.print(f'{name_version}\n\n[green]Available models:[/green]')
        for model in qualified_model_names:
            console.print(f'  {model}', highlight=False)
        return 0

    agent: Agent[None, str] = cli_agent
    if args.agent:
        sys.path.append(os.getcwd())
        try:
            module_path, variable_name = args.agent.split(':')
        except ValueError:
            console.print('[red]Error: Agent must be specified in "module:variable" format[/red]')
            return 1

        module = importlib.import_module(module_path)
        agent = getattr(module, variable_name)
        if not isinstance(agent, Agent):
            console.print(f'[red]Error: {args.agent} is not an Agent instance[/red]')
            return 1

    model_arg_set = args.model is not None
    if agent.model is None or model_arg_set:
        try:
            agent.model = infer_model(args.model or default_model)
        except UserError as e:
            console.print(f'Error initializing [magenta]{args.model}[/magenta]:\n[red]{e}[/red]')
            return 1

    model_name = agent.model if isinstance(agent.model, str) else f'{agent.model.system}:{agent.model.model_name}'
    if args.agent and model_arg_set:
        console.print(
            f'{name_version} using custom agent [magenta]{args.agent}[/magenta] with [magenta]{model_name}[/magenta]',
            highlight=False,
        )
    elif args.agent:
        console.print(f'{name_version} using custom agent [magenta]{args.agent}[/magenta]', highlight=False)
    else:
        console.print(f'{name_version} with [magenta]{model_name}[/magenta]', highlight=False)

    stream = not args.no_stream
    if args.code_theme == 'light':
        code_theme = 'default'
    elif args.code_theme == 'dark':
        code_theme = 'monokai'
    else:
        code_theme = args.code_theme  # pragma: no cover

    if prompt := cast(str, args.prompt):
        try:
            asyncio.run(ask_agent(agent, prompt, stream, console, code_theme))
        except KeyboardInterrupt:
            pass
        return 0

    try:
        return asyncio.run(run_chat(stream, agent, console, code_theme, prog_name))
    except KeyboardInterrupt:  # pragma: no cover
        return 0


async def run_chat(
    stream: bool,
    agent: AbstractAgent[AgentDepsT, OutputDataT],
    console: Console,
    code_theme: str,
    prog_name: str,
    config_dir: Path | None = None,
    deps: AgentDepsT = None,
    message_history: Sequence[ModelMessage] | None = None,
) -> int:
    prompt_history_path = (config_dir or PYDANTIC_AI_HOME) / PROMPT_HISTORY_FILENAME
    prompt_history_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_history_path.touch(exist_ok=True)
    session: PromptSession[Any] = PromptSession(history=FileHistory(str(prompt_history_path)))

    multiline = False
    messages: list[ModelMessage] = list(message_history) if message_history else []

    while True:
        try:
            auto_suggest = CustomAutoSuggest(['/markdown', '/multiline', '/exit', '/cp'])
            text = await session.prompt_async(f'{prog_name} ➤ ', auto_suggest=auto_suggest, multiline=multiline)
        except (KeyboardInterrupt, EOFError):  # pragma: no cover
            return 0

        if not text.strip():
            continue

        ident_prompt = text.lower().strip().replace(' ', '-')
        if ident_prompt.startswith('/'):
            exit_value, multiline = handle_slash_command(ident_prompt, messages, multiline, console, code_theme)
            if exit_value is not None:
                return exit_value
        else:
            try:
                messages = await ask_agent(agent, text, stream, console, code_theme, deps, messages)
            except CancelledError:  # pragma: no cover
                console.print('[dim]Interrupted[/dim]')
            except Exception as e:  # pragma: no cover
                cause = getattr(e, '__cause__', None)
                console.print(f'\n[red]{type(e).__name__}:[/red] {e}')
                if cause:
                    console.print(f'[dim]Caused by: {cause}[/dim]')


async def ask_agent(
    agent: AbstractAgent[AgentDepsT, OutputDataT],
    prompt: str,
    stream: bool,
    console: Console,
    code_theme: str,
    deps: AgentDepsT = None,
    messages: Sequence[ModelMessage] | None = None,
) -> list[ModelMessage]:
    status = Status('[dim]Working on it…[/dim]', console=console)

    if not stream:
        with status:
            result = await agent.run(prompt, message_history=messages, deps=deps)
        content = str(result.output)
        console.print(Markdown(content, code_theme=code_theme))
        return result.all_messages()

    with status, ExitStack() as stack:
        async with agent.iter(prompt, message_history=messages, deps=deps) as agent_run:
            live = Live('', refresh_per_second=15, console=console, vertical_overflow='ellipsis')
            async for node in agent_run:
                if Agent.is_model_request_node(node):
                    async with node.stream(agent_run.ctx) as handle_stream:
                        status.stop()  # stopping multiple times is idempotent
                        stack.enter_context(live)  # entering multiple times is idempotent

                        async for content in handle_stream.stream_output(debounce_by=None):
                            live.update(Markdown(str(content), code_theme=code_theme))

        assert agent_run.result is not None
        return agent_run.result.all_messages()


class CustomAutoSuggest(AutoSuggestFromHistory):
    def __init__(self, special_suggestions: list[str] | None = None):
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


def handle_slash_command(
    ident_prompt: str, messages: list[ModelMessage], multiline: bool, console: Console, code_theme: str
) -> tuple[int | None, bool]:
    if ident_prompt == '/markdown':
        try:
            parts = messages[-1].parts
        except IndexError:
            console.print('[dim]No markdown output available.[/dim]')
        else:
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
                'Enabling multiline mode. [dim]Press [Meta+Enter] or [Esc] followed by [Enter] to accept input.[/dim]'
            )
        else:
            console.print('Disabling multiline mode.')
        return None, multiline
    elif ident_prompt == '/exit':
        console.print('[dim]Exiting…[/dim]')
        return 0, multiline
    elif ident_prompt == '/cp':
        if not messages or not isinstance(messages[-1], ModelResponse):
            console.print('[dim]No output available to copy.[/dim]')
        else:
            text_to_copy = messages[-1].text
            if text_to_copy and (text_to_copy := text_to_copy.strip()):
                pyperclip.copy(text_to_copy)
                console.print('[dim]Copied last output to clipboard.[/dim]')
            else:
                console.print('[dim]No text content to copy.[/dim]')
    else:
        console.print(f'[red]Unknown command[/red] [magenta]`{ident_prompt}`[/magenta]')
    return None, multiline
