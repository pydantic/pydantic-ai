import argparse
import asyncio
import os
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Literal, TypedDict, cast

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from rich.console import Console, ConsoleOptions, RenderResult
from rich.live import Live
from rich.markdown import CodeBlock, Markdown
from rich.syntax import Syntax
from rich.text import Text

from pydantic_ai.agent import Agent
from pydantic_ai.messages import ModelMessage, PartDeltaEvent, TextPartDelta

__version__ = version('pydantic-ai')


class SimpleCodeBlock(CodeBlock):
    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        code = str(self.text).rstrip()
        yield Text(self.lexer_name, style='dim')
        yield Syntax(code, self.lexer_name, theme=self.theme, background_color='default', word_wrap=True)
        yield Text(f'/{self.lexer_name}', style='dim')


Markdown.elements['fence'] = SimpleCodeBlock


def cli() -> int:
    parser = argparse.ArgumentParser(
        prog='PydAI',
        description=f"""\
PydanticAI CLI v{__version__}

Special prompt:
* `multiline` - toggle multiline mode
""",
    )
    parser.add_argument('prompt', nargs='?', help='AI Prompt, if omitted fall into interactive mode')
    # Default model, if omitted it will default to openai:gpt-4o.
    parser.add_argument(
        'model', nargs='?', help='Model to use, if omitted it will default to openai:gpt-4o', default='openai:gpt-4o'
    )
    parser.add_argument('--no-stream', action='store_true', help='Whether to stream responses from OpenAI')
    parser.add_argument('--version', action='store_true', help='Show version and exit')

    args = parser.parse_args()

    console = Console()
    console.print(f'PydAI - PydanticAI CLI v{__version__}', style='green bold', highlight=False)
    if args.version:
        return 0

    now_utc = datetime.now(timezone.utc)
    tzname = now_utc.astimezone().tzinfo.tzname(now_utc)  # type: ignore
    agent = Agent(
        model=args.model or 'openai:gpt-4o',
        system_prompt=f"""\
Help the user by responding to their request, the output should be concise and always written in markdown.
The current date and time is {datetime.now()} {tzname}.
The user is running {sys.platform}.""",
    )

    stream = not args.no_stream

    if prompt := cast(str, args.prompt):
        try:
            asyncio.run(ask_agent(agent, prompt, stream, console))
        except KeyboardInterrupt:
            pass
        return 0

    history = Path().home() / '.prompt-history.txt'
    session = PromptSession(history=FileHistory(str(history)))  # type: ignore
    multiline = False
    messages: list[ModelMessage] = []

    while True:
        try:
            text = cast(str, session.prompt('pydai ➤ ', auto_suggest=AutoSuggestFromHistory(), multiline=multiline))
        except (KeyboardInterrupt, EOFError):
            return 0

        if not text.strip():
            continue

        ident_prompt = text.lower().strip(' ').replace(' ', '-')
        if ident_prompt == 'multiline':
            multiline = not multiline
            if multiline:
                console.print(
                    'Enabling multiline mode. '
                    '[dim]Press [Meta+Enter] or [Esc] followed by [Enter] to accept input.[/dim]'
                )
            else:
                console.print('Disabling multiline mode.')
            continue

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
) -> list[ModelMessage]:
    async with agent.iter(prompt, message_history=messages) as agent_run:
        console.print('\nResponse:', style='green')

        content: str = ''
        interrupted = False
        with Live('', refresh_per_second=15, console=console) as live:
            try:
                async for node in agent_run:
                    if Agent.is_model_request_node(node):
                        async with node.stream(agent_run.ctx) as handle_stream:
                            async for event in handle_stream:
                                if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                                    if stream:
                                        content += event.delta.content_delta
                                        live.update(Markdown(content))
            except KeyboardInterrupt:
                interrupted = True

        if interrupted:
            console.print('[dim]Interrupted[/dim]')

        assert agent_run.result
        if not stream:
            content = agent_run.result.data
            console.print(Markdown(content))
        return agent_run.result.all_messages()


class Provider(TypedDict):
    name: Literal['openai']
    api_key: str


def infer_provider() -> Provider:
    """Infer the provider from the environment variables."""
    try:
        openai_api_key = os.environ['OPENAI_API_KEY']
    except KeyError:
        raise ValueError('Only OpenAI is supported at the moment, please set the OPENAI_API_KEY environment variable.')
    return {'name': 'openai', 'api_key': openai_api_key}


def default_model(provider: Literal['openai']) -> Literal['openai:gpt-4o']:
    if provider == 'openai':
        return 'openai:gpt-4o'
    raise ValueError(f'Unknown provider: {provider}')
