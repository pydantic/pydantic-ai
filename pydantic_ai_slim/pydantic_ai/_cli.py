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

from pydantic_ai.result import OutputDataT
from pydantic_ai.tools import AgentDepsT

from . import __version__
from .agent import Agent
from .exceptions import UserError
from .messages import ModelMessage
from .models import KnownModelName, infer_model

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
    import httpx
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
DISCOVERY_CONFIG_FILENAME = 'discovery.json'


def save_discovery_config(endpoint: str, model_name: str) -> None:
    """Save the last used discovery configuration to a file."""
    try:
        PYDANTIC_AI_HOME.mkdir(parents=True, exist_ok=True)
        config_file = PYDANTIC_AI_HOME / DISCOVERY_CONFIG_FILENAME
        import json

        config = {
            "endpoint": endpoint,
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
        }
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
    except Exception:
        # Silently ignore errors when saving config
        pass


def load_discovery_config() -> tuple[str, str, str] | None:
    """Load the last used discovery configuration from a file."""
    try:
        config_file = PYDANTIC_AI_HOME / DISCOVERY_CONFIG_FILENAME
        if config_file.exists():
            import json

            with open(config_file, "r") as f:
                config = json.load(f)
            return (
                config.get("endpoint"),
                config.get("model_name"),
                config.get("timestamp"),
            )
    except Exception:
        # Silently ignore errors when loading config
        pass
    return None


async def discover_local_models(base_url: str) -> list[str]:
    """Discover models from a local OpenAI-compatible API endpoint.

    Args:
        base_url: The complete API base URL (e.g., 'http://localhost:11434/v1' or 'http://localhost:1234/v1')

    Returns:
        List of discovered model names in alphabetical order
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            models_url = f"{base_url.rstrip('/')}/models"
            response = await client.get(models_url)
            response.raise_for_status()

            data = response.json()
            if "data" in data and isinstance(data["data"], list):
                models = [model["id"] for model in data["data"] if "id" in model]
                return sorted(models)  # Return alphabetically sorted
            else:
                return []
    except Exception as e:
        raise UserError(f"Failed to discover models from {base_url}: {e}")


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


def cli_exit(prog_name: str = 'pai') -> None:  # pragma: no cover
    """Run the CLI and exit."""
    sys.exit(cli(prog_name=prog_name))


def cli(args_list: Sequence[str] | None = None, *, prog_name: str = 'pai') -> int:  # noqa: C901
    """Run the CLI and return the exit code for the process."""
    parser = argparse.ArgumentParser(
        prog=prog_name,
        description=f"""\
PydanticAI CLI v{__version__}\n\n

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
        help='Model to use, in format "<provider>:<model>" e.g. "openai:gpt-4o" or "anthropic:claude-3-7-sonnet-latest". Defaults to "openai:gpt-4o".',
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
        "-d",
        "--discover-models",
        nargs="?",
        const="",
        metavar="API_URL",
        help='Discover models from a local OpenAI-compatible API endpoint and interactively select one to chat with. '
             'Requires complete API URL including version (e.g., "http://localhost:11434/v1" for Ollama or "http://localhost:1234/v1" for LM Studio). '
             'If no URL provided, uses the last discovered endpoint.',
    )
    parser.add_argument(
        "-dd",
        "--discover-direct",
        action="store_true",
        help='Directly connect to the last used model and endpoint without any prompts. Equivalent to "clai -d" + pressing Enter twice.',
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
    name_version = f'[green]{prog_name} - PydanticAI CLI v{__version__}[/green]'
    if args.version:
        console.print(name_version, highlight=False)
        return 0
    if args.list_models:
        console.print(f'{name_version}\n\n[green]Available models:[/green]')
        for model in qualified_model_names:
            console.print(f'  {model}', highlight=False)

        # Show last discovery configuration if available
        last_config = load_discovery_config()
        if last_config and last_config[0]:
            try:
                timestamp = datetime.fromisoformat(last_config[2])
                time_str = timestamp.strftime("%Y-%m-%d %H:%M")
                console.print(
                    f"\n[dim]Last discovery: {last_config[1]} at {last_config[0]} ({time_str})[/dim]"
                )
                console.print(f'[dim]Use "clai -d" to reconnect to last endpoint[/dim]')
            except (ValueError, TypeError):
                console.print(
                    f"\n[dim]Last discovery: {last_config[1]} at {last_config[0]}[/dim]"
                )
                console.print(f'[dim]Use "clai -d" to reconnect to last endpoint[/dim]')
        return 0

    if args.discover_direct:
        # Direct connection to last used model and endpoint
        last_config = load_discovery_config()
        if not last_config or not last_config[0] or not last_config[1]:
            console.print(f"{name_version}\n\n[red]No previous discovery found.[/red]")
            console.print(
                '[dim]Use "clai -d <api_url>" to discover models first (e.g., clai -d http://localhost:1234/v1)[/dim]'
            )
            return 1

        endpoint, model_name, timestamp_str = last_config
        try:
            ts = datetime.fromisoformat(timestamp_str)
            time_str = ts.strftime("%Y-%m-%d %H:%M")
            console.print(
                f"{name_version}\n\n[green]Connecting directly to: {model_name} at {endpoint} (last used {time_str})[/green]"
            )
        except (ValueError, TypeError):
            console.print(
                f"{name_version}\n\n[green]Connecting directly to: {model_name} at {endpoint}[/green]"
            )

        # Configure the agent with the saved model and provider
        try:
            from .providers.openai import OpenAIProvider
            from .models.openai import OpenAIModel

            provider = OpenAIProvider(base_url=endpoint)
            model = OpenAIModel(model_name, provider=provider)
            cli_agent.model = model

        except Exception as e:
            console.print(f"[red]Error connecting to {endpoint}: {e}[/red]")
            console.print('[dim]Try "clai -d" to rediscover models[/dim]')
            return 1

    if args.discover_models is not None:
        # Determine the endpoint to use
        endpoint = args.discover_models
        if not endpoint:
            # No endpoint provided, try to use the last saved one
            last_config = load_discovery_config()
            if last_config and last_config[0]:
                endpoint = last_config[0]
                console.print(
                    f"{name_version}\n\n[dim]Using last discovered endpoint: {endpoint}[/dim]"
                )
            else:
                console.print(
                    f"{name_version}\n\n[red]No endpoint provided and no previous discovery found.[/red]"
                )
                console.print(
                    "[dim]Usage: clai -d <api_url> (e.g., clai -d http://localhost:1234/v1)[/dim]"
                )
                return 1

        console.print(f"[green]Discovering models from {endpoint}...[/green]")
        try:
            # Load last used configuration
            last_config = load_discovery_config()
            if last_config and last_config[0] == endpoint:
                try:
                    timestamp = datetime.fromisoformat(last_config[2])
                    time_str = timestamp.strftime("%Y-%m-%d %H:%M")
                    console.print(
                        f"[dim]Last used: {last_config[1]} ({time_str})[/dim]"
                    )
                except (ValueError, TypeError):
                    console.print(f"[dim]Last used: {last_config[1]}[/dim]")

            discovered_models = asyncio.run(discover_local_models(endpoint))
            if discovered_models:
                console.print(
                    f"\n[green]Found {len(discovered_models)} models:[/green]"
                )

                # Check if last used model is still available
                default_index = None
                if (
                    last_config
                    and last_config[0] == endpoint
                    and last_config[1] in discovered_models
                ):
                    default_index = discovered_models.index(last_config[1])

                for i, model in enumerate(discovered_models, 1):
                    if default_index is not None and i - 1 == default_index:
                        console.print(
                            f"  {i}. {model} [dim](last used)[/dim]", highlight=False
                        )
                    else:
                        console.print(f"  {i}. {model}", highlight=False)

                # Interactive model selection
                if default_index is not None:
                    console.print(
                        f"\n[cyan]Select a model to chat with (1-{len(discovered_models)}) or press Enter for default ({default_index + 1}):[/cyan]"
                    )
                else:
                    console.print(
                        f"\n[cyan]Select a model to chat with (1-{len(discovered_models)}) or press Enter to exit:[/cyan]"
                    )

                try:
                    choice = input().strip()
                    if not choice:
                        if default_index is not None:
                            model_index = default_index
                            console.print(
                                f"[dim]Using default: {discovered_models[model_index]}[/dim]"
                            )
                        else:
                            console.print("[dim]Exiting...[/dim]")
                            return 0
                    else:
                        model_index = int(choice) - 1

                    if 0 <= model_index < len(discovered_models):
                        selected_model = discovered_models[model_index]
                        console.print(
                            f"[green]Selected model: {selected_model}[/green]"
                        )

                        # Save the selection for next time
                        save_discovery_config(endpoint, selected_model)

                        # Configure the agent with the selected model and local provider
                        from .providers.openai import OpenAIProvider
                        from .models.openai import OpenAIModel

                        # Use the endpoint as provided by the user
                        provider = OpenAIProvider(base_url=endpoint)
                        model = OpenAIModel(selected_model, provider=provider)
                        cli_agent.model = model

                        console.print(
                            f"[green]Starting chat with {selected_model} at {endpoint}[/green]"
                        )
                    else:
                        console.print(
                            "[red]Invalid selection. Please choose a number from the list.[/red]"
                        )
                        return 1
                except ValueError:
                    console.print("[red]Invalid input. Please enter a number.[/red]")
                    return 1
                except KeyboardInterrupt:
                    console.print("\n[dim]Exiting...[/dim]")
                    return 0
            else:
                console.print(
                    "[yellow]No models found at the specified endpoint.[/yellow]"
                )
                return 1
        except UserError as e:
            console.print(f"[red]Error: {e}[/red]")
            return 1

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
            agent.model = infer_model(args.model or 'openai:gpt-4o')
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
    else:
        console.print(f'[red]Unknown command[/red] [magenta]`{ident_prompt}`[/magenta]')
    return None, multiline
