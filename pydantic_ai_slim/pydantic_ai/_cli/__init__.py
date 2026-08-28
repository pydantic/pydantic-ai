from __future__ import annotations as _annotations

import argparse
import functools
import json
import sys
from collections.abc import Sequence
from contextlib import AsyncExitStack, ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anyio
from pydantic import ImportString, TypeAdapter, ValidationError

from .. import __version__, models, usage as _usage
from .._run_context import AgentDepsT
from ..agent import AbstractAgent, Agent
from ..exceptions import UserError
from ..messages import FunctionToolCallEvent, FunctionToolResultEvent, ModelMessage, ModelResponse, ToolReturnPart
from ..models import infer_model, known_model_names
from ..native_tools import NATIVE_TOOLS_REQUIRING_CONFIG, SUPPORTED_NATIVE_TOOLS
from ..output import OutputDataT
from ..settings import ModelSettings
from ..toolsets import AbstractToolset

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

SUPPORTED_CLI_TOOL_IDS = sorted(
    bint.kind for bint in SUPPORTED_NATIVE_TOOLS if bint not in NATIVE_TOOLS_REQUIRING_CONFIG
)


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

_import_string_adapter: TypeAdapter[Any] = TypeAdapter(ImportString)


def load_agent(agent_path: str) -> Agent[Any, Any] | None:
    """Load an agent from a module path or a YAML/JSON spec file.

    Supports two formats:
    - Module path in uvicorn style: `'module:variable'`, e.g. `'test_agent:my_agent'`
    - File path to a YAML or JSON agent spec: e.g. `'agent.yml'`, `'agent.yaml'`, `'agent.json'`

    Args:
        agent_path: Module path or file path to load the agent from.

    Returns:
        Agent instance or None if loading fails.
    """
    path = Path(agent_path)
    if path.suffix in ('.yaml', '.yml', '.json'):  # pragma: no cover
        if not path.is_file():
            return None
        return Agent.from_file(path)

    sys.path.insert(0, str(Path.cwd()))
    try:
        obj = _import_string_adapter.validate_python(agent_path)
        if not isinstance(obj, Agent):
            return None
        return obj  # pyright: ignore[reportUnknownVariableType]
    except ValidationError:
        return None


@cli_agent.system_prompt
def cli_system_prompt() -> str:
    now_utc = datetime.now(timezone.utc)
    tzinfo = now_utc.astimezone().tzinfo
    tzname = tzinfo.tzname(now_utc) if tzinfo else ''
    return f"""\
Help the user by responding to their request, the output should be concise and always written in markdown.
The current date and time is {datetime.now()} {tzname}.
The user is running {sys.platform}."""


def cli_exit(prog_name: str = 'clai'):  # pragma: no cover
    """Run the CLI and exit."""
    sys.exit(cli(prog_name=prog_name))


def cli(args_list: Sequence[str] | None = None, *, prog_name: str = 'clai', default_model: str = 'openai:gpt-5') -> int:
    """Run the CLI and return the exit code for the process."""
    # we don't want to autocomplete or list models that don't include the provider,
    # e.g. we want to show `openai:gpt-5.2` but not `gpt-5.2`
    qualified_model_names = [n for n in known_model_names() if ':' in n]
    args_list = list(args_list) if args_list is not None else sys.argv[1:]

    # Check if this is a web command - route to web parser if so
    # This allows positional prompt arg in main parser without conflicting with subcommands
    if args_list and args_list[0] == 'web':
        return _cli_web(args_list[1:], prog_name, default_model, qualified_model_names)

    return _cli_chat(args_list, prog_name, default_model, qualified_model_names)


def _cli_web(args_list: list[str], prog_name: str, default_model: str, qualified_model_names: list[str]) -> int:
    """Handle the web subcommand."""
    parser = argparse.ArgumentParser(
        prog=f'{prog_name} web',
        description='Start a web-based chat interface for a generic or specified agent',
    )
    parser.add_argument(
        '--agent',
        '-a',
        help='Agent to serve: a module path like "module:variable" or a YAML/JSON spec file like "agent.yml". '
        'If omitted, creates a generic agent with the first specified model as default.',
    )
    model_arg = parser.add_argument(
        '-m',
        '--model',
        action='append',
        dest='models',
        help='Model to make available (can be repeated, e.g., -m openai:gpt-5 -m anthropic:claude-sonnet-4-6). '
        'Format: "provider:model_name". First model is preselected in UI; additional models appear as options.',
    )
    model_arg.completer = argcomplete.ChoicesCompleter(qualified_model_names)  # type: ignore[reportPrivateUsage]
    parser.add_argument(
        '-t',
        '--tool',
        choices=SUPPORTED_CLI_TOOL_IDS,
        action='append',
        dest='tools',
        help=f'Builtin tool to make available in the UI (can be repeated, e.g., -t web_search -t code_execution). '
        f'Available: {", ".join(SUPPORTED_CLI_TOOL_IDS)}.',
    )
    parser.add_argument(
        '-i',
        '--instructions',
        help="System instructions. When `--agent` is specified, these are additional to the agent's existing instructions "
        'and will be passed as extra instructions to each run.',
    )
    parser.add_argument(
        '--html-source',
        help='URL or file path for the chat UI HTML. If not specified, the UI is downloaded from a CDN.',
    )
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind server (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=7932, help='Port to bind server (default: 7932)')
    parser.add_argument(
        '--allowed-host',
        action='append',
        dest='allowed_hosts',
        help='Hostname to answer to in addition to IP addresses and localhost, which are always '
        'allowed (can be repeated). Needed only when reaching the UI through a name, e.g. behind a '
        'reverse proxy or a tunnel. Supports "*.example.com" and "*" for any host.',
    )
    argcomplete.autocomplete(parser)
    args = parser.parse_args(args_list)

    from .web import run_web_command

    return run_web_command(
        agent_path=args.agent,
        host=args.host,
        port=args.port,
        models=args.models or [],
        tools=args.tools or [],
        instructions=args.instructions,
        default_model=default_model,
        html_source=args.html_source,
        allowed_hosts=args.allowed_hosts or [],
    )


def _cli_chat(args_list: list[str], prog_name: str, default_model: str, qualified_model_names: list[str]) -> int:
    """Handle the chat command (default)."""
    parser = argparse.ArgumentParser(
        prog=prog_name,
        description=f"""\
Pydantic AI CLI v{__version__}

subcommands:
  web           Start a web-based chat interface for an agent
                Run "clai web --help" for more information
""",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        '-l',
        '--list-models',
        action='store_true',
        help='List all available models and exit',
    )
    parser.add_argument('--version', action='store_true', help='Show version and exit')

    # Chat arguments
    parser.add_argument(
        'prompt',
        nargs='?',
        help='AI prompt for one-shot mode. If omitted, starts interactive mode.',
    )
    model_arg = parser.add_argument(
        '-m',
        '--model',
        help=f'Model to use, in format "<provider>:<model>" e.g. "openai:gpt-5" or "anthropic:claude-sonnet-4-6". Defaults to "{default_model}".',
    )
    model_arg.completer = argcomplete.ChoicesCompleter(qualified_model_names)  # type: ignore[reportPrivateUsage]
    parser.add_argument(
        '-a',
        '--agent',
        help='Custom Agent to use: a module path like "module:variable" or a YAML/JSON spec file like "agent.yml"',
    )
    parser.add_argument(
        '-t',
        '--code-theme',
        help='Which colors to use for code, can be "dark", "light" or any theme from pygments.org/styles/. Defaults to "dark" which works well on dark terminals.',
        default='dark',
    )
    parser.add_argument('--no-stream', action='store_true', help='Disable streaming from the model')
    parser.add_argument(
        '--mcp-config',
        help='Path to MCP servers configuration file (JSON, using the same mcpServers shape as Claude Desktop, Claude Code, and Cursor).',
    )
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

    # Default to chat command
    return _run_chat_command(args, console, name_version, default_model, prog_name)


def _load_mcp_toolsets_for_cli(config_path: str, console: Console) -> Sequence[AbstractToolset[Any]] | None:
    """Load the `--mcp-config` toolsets, or print a friendly error and return `None`."""
    # An empty value (`--mcp-config=`, or `--mcp-config="$UNSET_VAR"` in a script) is falsy, so a
    # truthiness check would skip MCP entirely and leave the user thinking their servers loaded.
    if not config_path:
        console.print('[red]Error: --mcp-config needs a path to a configuration file[/red]')
        return None

    from ..mcp import load_mcp_toolsets

    try:
        return load_mcp_toolsets(config_path)
    # `OSError` rather than `FileNotFoundError`: a path that exists but can't be read as a file
    # (a directory, or one without read permission) raises a sibling `OSError` subclass.
    except (OSError, ValidationError, ValueError) as e:
        console.print(f'[red]Error: Could not load MCP config from {config_path}:\n{e}[/red]')
        return None


def _run_chat_command(
    args: argparse.Namespace, console: Console, name_version: str, default_model: str, prog_name: str
) -> int:
    """Handle the chat command."""
    agent: Agent[object, str] = cli_agent
    if args.agent:
        loaded = load_agent(args.agent)
        if loaded is None:
            console.print(f'[red]Error: Could not load agent from {args.agent}[/red]')
            return 1
        agent = loaded

    toolsets: Sequence[AbstractToolset[Any]] | None = None
    if args.mcp_config is not None:
        toolsets = _load_mcp_toolsets_for_cli(args.mcp_config, console)
        if toolsets is None:
            return 1

    model_arg_set = args.model is not None
    if agent.model is None or model_arg_set:
        try:
            agent.model = infer_model(args.model or default_model)
        except UserError as e:
            console.print(f'Error initializing [magenta]{args.model}[/magenta]:\n[red]{e}[/red]')
            return 1

    model_name = agent.model if isinstance(agent.model, str) else agent.model.model_id
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

    if args.prompt:
        try:
            anyio.run(functools.partial(ask_agent, agent, args.prompt, stream, console, code_theme, toolsets=toolsets))
        except KeyboardInterrupt:
            pass
        return 0

    try:
        return anyio.run(functools.partial(run_chat, stream, agent, console, code_theme, prog_name, toolsets=toolsets))
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
    model: models.Model | models.KnownModelName | str | None = None,
    model_settings: ModelSettings | None = None,
    usage_limits: _usage.UsageLimits | None = None,
    toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
) -> int:
    prompt_history_path = (config_dir or PYDANTIC_AI_HOME) / PROMPT_HISTORY_FILENAME
    prompt_history_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_history_path.touch(exist_ok=True)
    session: PromptSession[Any] = PromptSession(history=FileHistory(str(prompt_history_path)))

    multiline = False
    messages: list[ModelMessage] = list(message_history) if message_history else []
    session_usage = _usage.RunUsage()
    session_turns = 0

    async with AsyncExitStack() as toolset_stack:
        # Hold the toolsets open for the whole chat rather than per turn. Each run still enters
        # them, but `MCPToolset` ref-counts, so those become no-ops instead of a fresh subprocess
        # and `tools/list` handshake on every message, and server-side session state survives the
        # turn. An unreachable server therefore fails at startup rather than on the first message.
        for toolset in toolsets or ():
            await toolset_stack.enter_async_context(toolset)

        while True:
            try:
                auto_suggest = CustomAutoSuggest(['/markdown', '/multiline', '/usage', '/exit', '/cp'])
                text = await session.prompt_async(f'{prog_name} ➤ ', auto_suggest=auto_suggest, multiline=multiline)
            except (KeyboardInterrupt, EOFError):  # pragma: no cover
                return 0

            if not text.strip():
                continue

            ident_prompt = text.lower().strip().replace(' ', '-')
            if ident_prompt.startswith('/'):
                exit_value, multiline = handle_slash_command(
                    ident_prompt, messages, multiline, console, code_theme, usage=session_usage, turns=session_turns
                )
                if exit_value is not None:
                    return exit_value
            else:
                try:
                    messages = await ask_agent(
                        agent,
                        text,
                        stream,
                        console,
                        code_theme,
                        deps=deps,
                        messages=messages,
                        model=model,
                        model_settings=model_settings,
                        usage_limits=usage_limits,
                        toolsets=toolsets,
                        usage=session_usage,
                    )
                    session_turns += 1
                except anyio.get_cancelled_exc_class():
                    console.print('[dim]Interrupted[/dim]')  # pragma: no cover
                except Exception as e:
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
    model: models.Model | models.KnownModelName | str | None = None,
    model_settings: ModelSettings | None = None,
    usage_limits: _usage.UsageLimits | None = None,
    toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
    *,
    usage: _usage.RunUsage | None = None,
) -> list[ModelMessage]:
    status = Status('[dim]Working on it…[/dim]', console=console)

    # Count this turn into a fresh `RunUsage` so `usage_limits` stays per-run, then merge it into the
    # session total in a `finally` so a turn that fails after a billed request is still counted.
    turn_usage = _usage.RunUsage()
    try:
        if not stream:
            with status:
                result = await agent.run(
                    prompt,
                    message_history=messages,
                    deps=deps,
                    model=model,
                    model_settings=model_settings,
                    usage_limits=usage_limits,
                    toolsets=toolsets,
                    usage=turn_usage,
                )
            content = str(result.output)
            console.print(Markdown(content, code_theme=code_theme))
            return result.all_messages()

        with status, ExitStack() as stack:
            async with agent.iter(
                prompt,
                message_history=messages,
                deps=deps,
                model=model,
                model_settings=model_settings,
                usage_limits=usage_limits,
                toolsets=toolsets,
                usage=turn_usage,
            ) as agent_run:
                live = Live('', refresh_per_second=15, console=console, vertical_overflow='ellipsis')
                content_pieces: list[str] = []
                # Tool calls run concurrently and can return out of order, so in-flight calls are
                # keyed by call id — rendering only the latest would erase the others' indicators.
                pending_calls: dict[str, str] = {}
                updated_content = ''
                live_started = False

                async for node in agent_run:
                    if Agent.is_model_request_node(node):
                        async with node.stream(agent_run.ctx) as handle_stream:
                            # Inside the `async with`, so the spinner survives request preparation
                            # and time-to-first-byte rather than handing the user a blank display.
                            # The first `ModelRequestNode` always precedes any tool-call node, so
                            # `live` is entered before any tool-call node needs it — and the flag
                            # enters it exactly once.
                            if not live_started:
                                status.stop()
                                stack.enter_context(live)
                                live_started = True

                            async for content in handle_stream.stream_output(debounce_by=None):
                                updated_content = str(content)
                                display = '\n\n'.join([*content_pieces, updated_content])
                                live.update(Markdown(display, code_theme=code_theme))

                    elif Agent.is_call_tools_node(node):
                        # Freeze the text streamed so far so tool-call lines append below it rather
                        # than overwriting it on the next model request node.
                        if updated_content:
                            content_pieces.append(updated_content)
                            updated_content = ''

                        async with node.stream(agent_run.ctx) as handle_stream:
                            async for event in handle_stream:
                                if isinstance(event, FunctionToolCallEvent):
                                    pending_calls[event.tool_call_id] = event.part.tool_name
                                elif isinstance(event, FunctionToolResultEvent):
                                    # Pop on any result, not just a `ToolReturnPart`: a call that
                                    # comes back as a `RetryPromptPart` would otherwise stay pending
                                    # and pin its indicator for the rest of the run.
                                    pending_calls.pop(event.tool_call_id, None)
                                    if isinstance(event.part, ToolReturnPart):
                                        content_pieces.append(f'> Called tool `{event.part.tool_name}`.')
                                calling = [f'> _Calling tool `{name}`…_' for name in pending_calls.values()]
                                live.update(Markdown('\n\n'.join([*content_pieces, *calling]), code_theme=code_theme))

            assert agent_run.result is not None
            return agent_run.result.all_messages()
    finally:
        if usage is not None:
            usage.incr(turn_usage)


class CustomAutoSuggest(AutoSuggestFromHistory):
    def __init__(self, special_suggestions: list[str] | None = None):
        super().__init__()
        self.special_suggestions = special_suggestions or []

    def get_suggestion(self, buffer: Buffer, document: Document) -> Suggestion | None:
        # Get the suggestion from history
        suggestion = super().get_suggestion(buffer, document)

        # Check for custom suggestions
        text = document.text_before_cursor.strip()
        for special in self.special_suggestions:
            if special.startswith(text):
                return Suggestion(special[len(text) :])
        return suggestion


def format_usage(usage: _usage.RunUsage, turns: int, *, as_json: bool = False) -> str:
    """Render cumulative session usage for the `/usage` slash command.

    Args:
        usage: The accumulated usage for the session.
        turns: The number of turns (prompts answered by the agent) so far.
        as_json: If set, render a single-line JSON object for scripting instead of the human-readable summary.
    """
    if as_json:
        return json.dumps(
            {
                'turns': turns,
                'input_tokens': usage.input_tokens,
                'output_tokens': usage.output_tokens,
                'total_tokens': usage.total_tokens,
                'requests': usage.requests,
                'tool_calls': usage.tool_calls,
            }
        )
    return (
        'clai usage (session total)\n\n'
        f'Turns:      {turns:,}\n'
        f'Tokens:     {usage.total_tokens:,}\n'
        f'  Input:    {usage.input_tokens:,}\n'
        f'  Output:   {usage.output_tokens:,}\n'
        f'Requests:   {usage.requests:,}\n'
        f'Tool calls: {usage.tool_calls:,}'
    )


def handle_slash_command(
    ident_prompt: str,
    messages: list[ModelMessage],
    multiline: bool,
    console: Console,
    code_theme: str,
    *,
    usage: _usage.RunUsage | None = None,
    turns: int = 0,
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
    elif ident_prompt == '/usage' or ident_prompt.startswith('/usage-'):
        # A flag is separated by a space, which is replaced with `-` upstream, so `/usage --json`
        # arrives as `/usage---json`. Requiring the `/usage-` prefix keeps `/usagex` an unknown command.
        option = ident_prompt[len('/usage') :].strip('-')
        if option in ('', 'json'):
            # `soft_wrap` keeps the JSON on a single line for piping; the text has no markup to render.
            console.print(format_usage(usage or _usage.RunUsage(), turns, as_json=option == 'json'), soft_wrap=True)
        else:
            console.print(f'[red]Unknown `/usage` option[/red] [magenta]`{ident_prompt}`[/magenta]')
    else:
        console.print(f'[red]Unknown command[/red] [magenta]`{ident_prompt}`[/magenta]')
    return None, multiline
