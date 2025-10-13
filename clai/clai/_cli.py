from __future__ import annotations as _annotations

import argparse
import importlib
import os
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from importlib.metadata import version as _metadata_version
from pathlib import Path

from typing_inspection.introspection import get_literal_values

from pydantic_ai.agent import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import KnownModelName, infer_model

__version__ = _metadata_version('clai')

try:
    import argcomplete
    from rich.console import Console
except ImportError as _import_error:
    raise ImportError(
        'Please install `rich`, `prompt-toolkit` and `argcomplete` to use the Pydantic AI CLI, '
        'you can use the `cli` optional group — `pip install "pydantic-ai-slim[cli]"`'
    ) from _import_error


__all__ = 'cli', 'cli_exit'


PYDANTIC_AI_HOME = Path.home() / '.pydantic-ai'
"""The home directory for Pydantic AI CLI.

This folder is used to store the prompt history and configuration.
"""

PROMPT_HISTORY_FILENAME = 'prompt-history.txt'


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


def cli(
    args_list: Sequence[str] | None = None,
    *,
    prog_name: str = 'pai',
    default_model: str = 'openai:gpt-4.1',
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
""",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('prompt', nargs='?', help='AI Prompt, if omitted fall into interactive mode')
    arg = parser.add_argument(
        '-m',
        '--model',
        nargs='?',
        help=f'Model to use, in format "<provider>:<model>" e.g. "openai:gpt-4.1" or "anthropic:claude-sonnet-4-0". Defaults to "{default_model}".',
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
    title = name_version = f'{prog_name} - Pydantic AI CLI v{__version__}'
    if args.agent and model_arg_set:
        title = f'{name_version} using custom agent **{args.agent}** with `{model_name}`'

    elif args.agent:
        title = f'{name_version} using custom agent **{args.agent}**'

    else:
        title = f'{name_version} with **{model_name}**'

    from clai.tui import CLAIApp

    app = CLAIApp(agent, PYDANTIC_AI_HOME / PROMPT_HISTORY_FILENAME, prompt=args.prompt, title=title)
    app.run()
    return 0
