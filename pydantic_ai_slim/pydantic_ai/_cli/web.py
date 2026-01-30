from __future__ import annotations

from rich.console import Console

from pydantic_ai import Agent
from pydantic_ai.builtin_tools import BUILTIN_TOOL_TYPES, AbstractBuiltinTool
from pydantic_ai.ui._web import create_web_app

from . import SUPPORTED_CLI_TOOL_IDS, load_agent


def run_web_command(
    agent_path: str | None = None,
    host: str = '127.0.0.1',
    port: int = 7932,
    models: list[str] = [],
    tools: list[str] = [],
    instructions: str | None = None,
    default_model: str = 'openai:gpt-5',
    html_source: str | None = None,
) -> int:
    """Run the web command to serve an agent via web UI.

    If an agent is provided, its model and builtin tools are used as defaults.
    CLI-specified models and tools are added on top. Duplicates are removed.

    Args:
        agent_path: Agent path in 'module:variable' format. If None, creates generic agent.
        host: Host to bind the server to.
        port: Port to bind the server to.
        models: List of model strings (e.g., ['openai:gpt-5', 'anthropic:claude-sonnet-4-5']).
        tools: List of builtin tool IDs (e.g., ['web_search', 'code_execution']).
        instructions: System instructions passed as extra instructions to each agent run.
        default_model: Default model to use when no agent or models are specified.
        html_source: URL or file path for the chat UI HTML.
    """
    console = Console()

    if agent_path:
        agent = load_agent(agent_path)
        if agent is None:
            console.print(f'[red]Error: Could not load agent from {agent_path}[/red]')
            return 1
    else:
        agent = Agent()

    # Use default model if neither agent nor CLI specifies one
    if agent.model is None and not models:
        models = [default_model]

    tool_instances: list[AbstractBuiltinTool] = []
    for tool_id in tools:
        tool_cls = BUILTIN_TOOL_TYPES.get(tool_id)
        if tool_cls is None:
            console.print(f'[yellow]Warning: Unknown tool "{tool_id}", skipping[/yellow]')
            continue
        if tool_id not in SUPPORTED_CLI_TOOL_IDS:
            console.print(
                f'[yellow]Warning: "{tool_id}" requires configuration and cannot be enabled via CLI, skipping[/yellow]'
            )
            continue
        tool_instances.append(tool_cls())

    app = create_web_app(
        agent,
        models=models or None,
        builtin_tools=tool_instances,
        instructions=instructions,
        html_source=html_source,
    )

    agent_desc = agent_path or 'generic agent'
    console.print(f'\n[green]Starting chat UI for {agent_desc}...[/green]')
    console.print(f'Open your browser at: [link=http://{host}:{port}]http://{host}:{port}[/link]')
    console.print('[dim]Press Ctrl+C to stop the server[/dim]\n')

    try:
        import uvicorn

        uvicorn.run(app, host=host, port=port)
        return 0
    except KeyboardInterrupt:  # pragma: no cover
        console.print('\n[dim]Server stopped.[/dim]')
        return 0
    except ImportError:  # pragma: no cover
        console.print('[red]Error: uvicorn is required to run the chat UI[/red]')
        console.print('[dim]Install it with: pip install uvicorn[/dim]')
        return 1
    except Exception as e:  # pragma: no cover
        console.print(f'[red]Error starting server: {e}[/red]')
        return 1
