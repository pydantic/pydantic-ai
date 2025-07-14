from __future__ import annotations as _annotations

import asyncio
import importlib
import os
import sys
from asyncio import CancelledError
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
from click import ClickException
from typing_inspection.introspection import get_literal_values

from . import __version__
from ._run_context import AgentDepsT
from .agent import Agent
from .exceptions import UserError
from .messages import ModelMessage
from .models import KnownModelName, Model, infer_model
from .output import OutputDataT

try:
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


__all__ = 'cli', 'cli_exit'


PYDANTIC_AI_HOME = Path.home() / '.pydantic-ai'
"""The home directory for PydanticAI CLI.

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
        yield Syntax(
            code,
            self.lexer_name,
            theme=self.theme,
            background_color='default',
            word_wrap=True,
        )
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


# we don't want to autocomplete or list models that don't include the provider,
# e.g. we want to show `openai:gpt-4o` but not `gpt-4o`
qualified_model_names = [n for n in get_literal_values(KnownModelName.__value__) if ':' in n]


# Custom option to hide the huge list of model names from the ``--help`` output
class _ModelOption(click.Option):
    """Click ``Option`` that strips the automatically appended choices list from the help text.

    We still want the underlying :class:`click.Choice` type for validation & completion, but we
    *don't* want Click to append the full list of ~300 model names to the help output (it makes
    ``pai --help`` almost unusable).
    """

    def get_help_record(self, ctx: click.Context) -> tuple[str, str] | None:
        # Get Click's default help record first.
        record = super().get_help_record(ctx)
        if record is None:
            return None

        option_str, help_str = record

        # Click shows choices in brackets like [choice1|choice2|choice3...]
        # We need to strip this massive list. Look for the pattern [xxx|xxx|xxx...]
        import re

        # Remove the entire choices section that starts with [ and contains | separators
        help_str = re.sub(r'\s*\[([^[\]]*\|[^[\]]*)+\]', '', help_str)

        return option_str, help_str


def _setup_agent(
    agent_path: str | None,
    model_name: str | None,
) -> Agent:
    """Set up the agent based on command line arguments."""
    agent: Agent[None, str] = cli_agent
    if agent_path:
        sys.path.append(os.getcwd())
        try:
            module_path, variable_name = agent_path.split(':')
        except ValueError:
            raise ClickException(click.style('Agent must be specified in "module:variable" format', fg='red'))

        module = importlib.import_module(module_path)
        agent = getattr(module, variable_name)
        if not isinstance(agent, Agent):
            raise ClickException(click.style(f'{agent_path} is not an Agent instance'))

    model_arg_set = model_name is not None
    if agent.model is None or model_arg_set:
        try:
            agent.model = infer_model(model_name or 'openai:gpt-4o')
        except UserError as e:
            raise ClickException(
                f'Error initializing {click.style(model_name, fg="magenta")}:\n{click.style(str(e), fg="red")}'
            )

    return agent


def _print_agent_info(
    console: Console,
    name_version: str,
    agent: Agent[None, str],
    agent_path: str | None,
    model_name: str | None,
) -> None:
    """Print agent and model information."""
    if isinstance(agent.model, str):
        model_display = agent.model
    elif isinstance(agent.model, Model):
        model_display = f'{agent.model.system}:{agent.model.model_name}'
    else:
        model_display = 'unknown'

    if agent_path and model_name is not None:
        console.print(
            f'{name_version} using custom agent [magenta]{agent_path}[/magenta] with [magenta]{model_display}[/magenta]',
            highlight=False,
        )
    elif agent_path:
        console.print(
            f'{name_version} using custom agent [magenta]{agent_path}[/magenta]',
            highlight=False,
        )
    else:
        console.print(
            f'{name_version} with [magenta]{model_display}[/magenta]',
            highlight=False,
        )


def _handle_prompt(
    console: Console,
    prog_name: str,
    prompt: tuple[str],
    agent: Agent[None, str],
    stream: bool,
    code_theme: str,
) -> int:
    """Handle prompt input and execution."""
    if prompt:
        # If prompt is provided, run it and exit
        prompt_str = ' '.join(prompt)

        try:
            asyncio.run(ask_agent(agent, prompt_str, stream, console, code_theme))
        except KeyboardInterrupt:
            return 0
    else:
        # Otherwise, start interactive mode
        try:
            asyncio.run(run_chat(stream, agent, console, code_theme, prog_name))
        except KeyboardInterrupt:
            return 0

    return 0


@click.command(
    name='pai',
    help=f"""
PydanticAI CLI v{__version__}

Special prompts:
* `/exit` - exit the interactive mode (ctrl-c and ctrl-d also work)
* `/markdown` - show the last markdown output of the last question
* `/multiline` - toggle multiline mode
""",
    context_settings={'help_option_names': ['-h', '--help']},
)
@click.argument('prompt', nargs=-1)
@click.option(
    '-m',
    '--model',
    'model_name',
    type=click.Choice(qualified_model_names),
    cls=_ModelOption,
    help='Model to use, in format "<provider>:<model>" e.g. "openai:gpt-4o" or "anthropic:claude-3-7-sonnet-latest". Defaults to "openai:gpt-4o".',
)
@click.option(
    '-a',
    '--agent',
    'agent_path',
    help='Custom Agent to use, in format "module:variable", e.g. "mymodule.submodule:my_agent"',
)
@click.option(
    '-l',
    '--list-models',
    is_flag=True,
    help='List all available models and exit',
)
@click.option(
    '-t',
    '--code-theme',
    type=click.Choice(['dark', 'light']),
    default='dark',
    help='Which colors to use for code, can be "dark", "light" or any theme from pygments.org/styles/. Defaults to "dark" which works well on dark terminals.',
)
@click.option('--no-stream', is_flag=True, help='Disable streaming from the model')
@click.option('--version', is_flag=True, help='Show version and exit')
@click.pass_context
def cli(
    ctx: click.Context,
    prompt: tuple[str],
    model_name: str | None,
    agent_path: str | None,
    list_models: bool,
    code_theme: str,
    no_stream: bool,
    version: bool,
) -> int:
    """Run the CLI and return the exit code for the process."""
    console = Console()
    prog_name = ctx.find_root().info_name or 'pai'
    name_version = f'{prog_name} - PydanticAI CLI v{__version__}'

    # Handle version and list-models flags
    if version:
        console.print(name_version, highlight=False)
        return 0

    if list_models:
        console.print(f'{name_version}\n\n[green]Available models:[/green]')
        for model in qualified_model_names:
            console.print(f'  {model}', highlight=False)
        return 0

    stream = not no_stream
    if code_theme == 'light':
        code_theme = 'default'
    elif code_theme == 'dark':
        code_theme = 'monokai'
    else:
        code_theme = code_theme  # pragma: no cover

    # Set up the agent
    agent = _setup_agent(agent_path, model_name)

    # Print agent info
    _print_agent_info(console, name_version, agent, agent_path, model_name)

    # Handle prompt or start interactive mode
    return _handle_prompt(console, prog_name, prompt, agent, stream, code_theme)


async def run_chat(
    stream: bool,
    agent: Agent[AgentDepsT, OutputDataT],
    console: Console,
    code_theme: str,
    prog_name: str,
    config_dir: Path | None = None,
    deps: AgentDepsT = None,
) -> int:
    prompt_history_path = (config_dir or PYDANTIC_AI_HOME) / PROMPT_HISTORY_FILENAME
    prompt_history_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_history_path.touch(exist_ok=True)
    session: PromptSession[Any] = PromptSession(history=FileHistory(str(prompt_history_path)))

    multiline = False
    messages: list[ModelMessage] = []

    while True:
        try:
            auto_suggest = CustomAutoSuggest(['/markdown', '/multiline', '/exit'])
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
    agent: Agent[AgentDepsT, OutputDataT],
    prompt: str,
    stream: bool,
    console: Console,
    code_theme: str,
    deps: AgentDepsT = None,
    messages: list[ModelMessage] | None = None,
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
    ident_prompt: str,
    messages: list[ModelMessage],
    multiline: bool,
    console: Console,
    code_theme: str,
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
    else:
        console.print(f'[red]Unknown command[/red] [magenta]`{ident_prompt}`[/magenta]')
    return None, multiline
