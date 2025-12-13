from __future__ import annotations

from rich.console import Console

from pydantic_ai import Agent
from pydantic_ai.builtin_tools import (
    BUILTIN_TOOL_TYPES,
    DEPRECATED_BUILTIN_TOOL_KINDS,
    AbstractBuiltinTool,
)
from pydantic_ai.models import Model
from pydantic_ai.ui._web import create_web_app

from . import load_agent

# Tools that require configuration and cannot be enabled via CLI
# (includes deprecated tools plus tools needing config like mcp_server and memory)
UNSUPPORTED_CLI_TOOL_KINDS = DEPRECATED_BUILTIN_TOOL_KINDS | frozenset({'mcp_server', 'memory'})


def run_web_command(  # noqa: C901
    agent_path: str | None = None,
    host: str = '127.0.0.1',
    port: int = 7932,
    models: list[str] | None = None,
    tools: list[str] | None = None,
    instructions: str | None = None,
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
    """
    console = Console()

    if agent_path:
        agent = load_agent(agent_path)
        if agent is None:
            console.print(f'[red]Error: Could not load agent from {agent_path}[/red]')
            return 1
    else:
        agent = Agent()

    if agent.model is None and not models:
        console.print('[red]Error: At least one model (-m) is required when agent has no model[/red]')
        return 1

    # build models dict: agent's model as 'default', plus any CLI models
    models_dict: dict[str, Model | str] | None = None
    if agent.model is not None or models:  # pragma: no branch
        models_dict = {}
        if agent.model is not None:
            # Use 'default' key - create_web_app will use agent.model for the actual model
            models_dict['default'] = agent.model
        if models:
            for m in models:
                if m not in models_dict.values():
                    models_dict[m] = m

    # collect CLI builtin tools only (agent's builtin_tools are handled by create_web_app)
    cli_tool_instances: list[AbstractBuiltinTool] | None = None
    if tools:
        cli_tool_instances = []
        for tool_id in tools:
            if tool_id in UNSUPPORTED_CLI_TOOL_KINDS:
                console.print(
                    f'[yellow]Warning: "{tool_id}" requires configuration and cannot be enabled via CLI, skipping[/yellow]'
                )
                continue
            tool_cls = BUILTIN_TOOL_TYPES.get(tool_id)
            if tool_cls is None:
                console.print(f'[yellow]Warning: Unknown tool "{tool_id}", skipping[/yellow]')
                continue
            cli_tool_instances.append(tool_cls())

    app = create_web_app(
        agent,
        models=models_dict,
        builtin_tools=cli_tool_instances,
        instructions=instructions,
    )

    agent_desc = agent_path if agent_path else 'generic agent'
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
