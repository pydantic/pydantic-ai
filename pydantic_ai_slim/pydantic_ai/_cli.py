from __future__ import annotations as _annotations

import argparse
import asyncio
import os
import shlex
import subprocess
import sys
from asyncio import CancelledError
from collections.abc import Coroutine, Sequence
from contextlib import ExitStack
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable, NotRequired, ParamSpec, cast

from pydantic import TypeAdapter
from pydantic.type_adapter import R
from typing_extensions import TypedDict
from typing_inspection.introspection import get_literal_values

from pydantic_ai.agent import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.mcp import MCPServer, MCPServerHTTP, MCPServerStdio
from pydantic_ai.messages import FunctionToolCallEvent, FunctionToolResultEvent, ModelMessage, ToolReturnPart
from pydantic_ai.models import KnownModelName, infer_model

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


# logfire.configure()
# logfire.log_slow_async_callbacks()

__version__ = version('pydantic-ai-slim')

PYDANTIC_AI_HOME = Path.home() / '.pydantic-ai'
"""The home directory for PydanticAI.

This folder is used to store the prompt history and configuration.
"""

PYDANTIC_AI_MCP_SERVERS_FILE = PYDANTIC_AI_HOME / 'mcp_servers.jsonc'
"""The MCP servers configuration file."""


class SimpleCodeBlock(CodeBlock):
    """Customised code blocks in markdown.

    This avoids a background color which messes up copy-pasting and sets the language name as dim prefix and suffix.
    """

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        code = str(self.text).rstrip()
        yield Text(self.lexer_name, style='dim')
        yield Syntax(code, self.lexer_name, theme=self.theme, background_color='default', word_wrap=True)
        yield Text(f'/{self.lexer_name}', style='dim')


class LeftHeading(Heading):
    """Customised headings in markdown to stop centering and prepend markdown style hashes."""

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


def cli(args_list: Sequence[str] | None = None) -> int:
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
    parser.add_argument('--edit-mcp-servers', action='store_true', help='Open an editor to configure MCP servers')

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
    if args.edit_mcp_servers:
        return edit_mcp_servers(console)

    mcp_servers = mcp_servers_from_config()
    # TODO(Marcelo): We should allow extending the list of MCP servers publicly.
    cli_agent._mcp_servers.extend(mcp_servers)  # type: ignore[reportPrivateUsage]

    try:
        cli_agent.model = infer_model(args.model)
    except UserError as e:
        console.print(f'Error initializing [magenta]{args.model}[/magenta]:\n[red]{e}[/red]')
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
            asyncio.run(run_mcp_servers(ask_agent)(prompt, stream, console, code_theme))
        except KeyboardInterrupt:
            pass
        return 0

    history = PYDANTIC_AI_HOME / 'prompt-history.txt'
    # doing this instead of `PromptSession[Any](history=` allows mocking of PromptSession in tests
    session: PromptSession[Any] = PromptSession(history=FileHistory(str(history)))
    try:
        return asyncio.run(run_mcp_servers(run_chat)(session, stream, console, code_theme))
    except KeyboardInterrupt:  # pragma: no cover
        return 0


def run_mcp_servers(func: Callable[P, Coroutine[Any, Any, R]]) -> Callable[P, Coroutine[Any, Any, R]]:
    """Run the MCP servers before calling the function.

    This is convenient for testing, as it allows us to not use the MCP servers in tests.
    """

    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        async with cli_agent.run_mcp_servers():
            return await func(*args, **kwargs)

    return wrapper


async def run_chat(session: PromptSession[Any], stream: bool, console: Console, code_theme: str) -> int:
    multiline = False
    messages: list[ModelMessage] = []

    while True:
        try:
            auto_suggest = CustomAutoSuggest(['/markdown', '/multiline', '/exit'])
            text = await session.prompt_async('pai ➤ ', auto_suggest=auto_suggest, multiline=multiline)
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
                messages = await ask_agent(text, stream, console, code_theme, messages)
            except CancelledError:  # pragma: no cover
                console.print('[dim]Interrupted[/dim]')


async def ask_agent(
    prompt: str,
    stream: bool,
    console: Console,
    code_theme: str,
    messages: list[ModelMessage] | None = None,
) -> list[ModelMessage]:
    status = Status('[dim]Working on it…[/dim]', console=console)

    if not stream:
        with status:
            result = await cli_agent.run(prompt, message_history=messages)
        content = result.data
        console.print(Markdown(content, code_theme=code_theme))
        return result.all_messages()

    with status, ExitStack() as stack:
        async with cli_agent.iter(prompt, message_history=messages) as agent_run:
            live = Live('', refresh_per_second=15, console=console, vertical_overflow='visible')
            # Keep track of all content pieces
            content_pieces: list[str] = []
            updated_content = ''

            async for node in agent_run:
                status.stop()  # stopping multiple times is idempotent
                stack.enter_context(live)  # entering multiple times is idempotent

                if Agent.is_model_request_node(node):
                    async with node.stream(agent_run.ctx) as handle_stream:
                        async for content in handle_stream.stream_output():
                            updated_content = content
                            # Show the current content plus all previous pieces
                            display_content = '\n\n'.join(content_pieces + [updated_content])
                            live.update(Markdown(display_content, code_theme=code_theme))

                elif Agent.is_call_tools_node(node):
                    # If there was model content before this tool call, save it
                    if updated_content:
                        content_pieces.append(updated_content)
                        updated_content = ''

                    async with node.stream(agent_run.ctx) as handle_stream:
                        async for event in handle_stream:
                            if isinstance(event, FunctionToolCallEvent):
                                # Show all previous content plus the tool call indicator
                                display_content = '\n\n'.join(content_pieces + ['']) if content_pieces else ''
                                if display_content:
                                    live.update(Markdown(display_content, code_theme=code_theme))
                                # Show the "Calling..." message in the same format
                                tool_name = event.part.tool_name
                                display_content = '\n\n'.join(
                                    content_pieces + [f'> 🔧 _Calling tool `{tool_name}`..._']
                                )
                                live.update(Markdown(display_content, code_theme=code_theme))
                            elif isinstance(event, FunctionToolResultEvent) and isinstance(
                                event.result, ToolReturnPart
                            ):
                                content = event.result
                                tool_name = event.result.tool_name
                                content_pieces.append(f'> 🔧 Called tool `{tool_name}`.')

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


class _MCPServerHTTP(TypedDict):
    url: str


class _MCPServerStdio(TypedDict):
    command: str
    args: list[str]
    env: NotRequired[dict[str, str]]


class _MCPServers(TypedDict):
    mcpServers: dict[str, _MCPServerHTTP | _MCPServerStdio]


_mcp_servers_ta = TypeAdapter(_MCPServers)


def edit_mcp_servers(console: Console) -> int:
    """Open an editor to configure MCP servers.

    Args:
        console: The console to print messages to.

    Returns:
        0 on success, 1 on error.
    """
    PYDANTIC_AI_HOME.mkdir(parents=True, exist_ok=True)

    default_content = """\
/* This file is managed by PydanticAI CLI.

You can include MCP servers in the following format:

    ```jsonc
    {
        "mcpServers": {
            "my-http-server": {
            "url": "http://localhost:3000",
        },
        "my-stdio-server": {
            "command": "uv",
            "args": ["run", "my_script.py"]
        }
    }
    ```

For more information, visit: https://ai.pydantic.dev/cli/mcp
*/
{
  "mcpServers": {}
}
"""

    if not PYDANTIC_AI_MCP_SERVERS_FILE.exists():
        PYDANTIC_AI_MCP_SERVERS_FILE.write_text(default_content)
        console.print(f'Created new MCP servers configuration at [cyan]{PYDANTIC_AI_MCP_SERVERS_FILE}[/cyan]')

    editor = os.environ.get('EDITOR', 'vim')
    try:
        subprocess.run(shlex.split(editor) + [str(PYDANTIC_AI_MCP_SERVERS_FILE)], check=True)
        console.print(f'Successfully edited MCP servers configuration at [cyan]{PYDANTIC_AI_MCP_SERVERS_FILE}[/cyan]')
        return 0
    except subprocess.CalledProcessError as e:
        console.print(f'[red]Error editing MCP servers configuration: {e}[/red]')
        return 1


def mcp_servers_from_config() -> list[MCPServer]:
    if not PYDANTIC_AI_MCP_SERVERS_FILE.exists():
        return []

    file_content = PYDANTIC_AI_MCP_SERVERS_FILE.read_text()
    # Remove multiline comments
    cleaned_content = file_content
    while True:
        start = cleaned_content.find('/*')
        if start == -1:
            break
        end = cleaned_content.find('*/', start)
        if end == -1:
            break
        cleaned_content = cleaned_content[:start] + cleaned_content[end + 2 :]

    mcp_server_config = _mcp_servers_ta.validate_json(cleaned_content)

    mcp_servers: list[MCPServer] = []
    for server in mcp_server_config['mcpServers'].values():
        if 'url' in server:
            mcp_servers.append(MCPServerHTTP(server['url']))
        elif 'command' in server:
            mcp_servers.append(MCPServerStdio(server['command'], server['args'], server.get('env', {})))
    return mcp_servers


P = ParamSpec('P')


def app():  # pragma: no cover
    sys.exit(cli())
