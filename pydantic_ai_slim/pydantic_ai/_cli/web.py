from __future__ import annotations

from pydantic import ValidationError
from rich.console import Console

from pydantic_ai import Agent
from pydantic_ai.builtin_tools import (
    BUILTIN_TOOL_TYPES,
    DEPRECATED_BUILTIN_TOOL_TYPES,
    AbstractBuiltinTool,
)
from pydantic_ai.mcp import MCPServerSSE, MCPServerStdio, MCPServerStreamableHTTP, load_mcp_servers
from pydantic_ai.models import infer_model
from pydantic_ai.ui._web import create_web_app

from . import load_agent

# Tools that require configuration and cannot be enabled via CLI
# (includes deprecated tools plus tools needing config like mcp_server and memory)
UNSUPPORTED_CLI_TOOLS = DEPRECATED_BUILTIN_TOOL_TYPES | frozenset({'mcp_server', 'memory'})


def run_web_command(  # noqa: C901
    agent_path: str | None = None,
    host: str = '127.0.0.1',
    port: int = 7932,
    models: list[str] | None = None,
    tools: list[str] | None = None,
    instructions: str | None = None,
    mcp: str | None = None,
) -> int:
    """Run the web command to serve an agent via web UI.

    If an agent is provided, its model and builtin tools are used as defaults.
    CLI-specified models and tools are added on top. Duplicates are removed.
    MCP servers are loaded as toolsets if specified.

    Args:
        agent_path: Agent path in 'module:variable' format. If None, creates generic agent.
        host: Host to bind the server to.
        port: Port to bind the server to.
        models: List of model strings (e.g., ['openai:gpt-5', 'anthropic:claude-sonnet-4-5']).
        tools: List of builtin tool IDs (e.g., ['web_search', 'code_execution']).
        instructions: System instructions. For generic agent (no agent_path), these become
            the agent's instructions. With an agent, these are passed as extra instructions
            to each run.
        mcp: Path to JSON file with MCP server configurations.
    """
    console = Console()

    if agent_path:
        agent = load_agent(agent_path)
        if agent is None:
            console.print(f'[red]Error: Could not load agent from {agent_path}[/red]')
            return 1
    else:
        agent = Agent(instructions=instructions)

    if agent.model is None and not models:
        console.print('[red]Error: At least one model (-m) is required when agent has no model[/red]')
        return 1

    # build models list
    if agent.model is not None:
        resolved = infer_model(agent.model)
        agent_model = f'{resolved.system}:{resolved.model_name}'
        models = [agent_model] + [m for m in (models or []) if m != agent_model]

    # collect builtin tools
    all_tool_instances: list[AbstractBuiltinTool] = []

    # agent's own builtin tools first
    all_tool_instances.extend(agent._builtin_tools)  # pyright: ignore[reportPrivateUsage]

    # then CLI tools
    if tools:
        for tool_id in tools:
            if tool_id in UNSUPPORTED_CLI_TOOLS:
                console.print(
                    f'[yellow]Warning: "{tool_id}" requires configuration and cannot be enabled via CLI, skipping[/yellow]'
                )
                continue
            tool_cls = BUILTIN_TOOL_TYPES.get(tool_id)
            if tool_cls is None:
                console.print(f'[yellow]Warning: Unknown tool "{tool_id}", skipping[/yellow]')
                continue
            all_tool_instances.append(tool_cls())

    # Load MCP servers as toolsets if specified
    mcp_servers: list[MCPServerStdio | MCPServerStreamableHTTP | MCPServerSSE] = []
    if mcp:
        try:
            mcp_servers = load_mcp_servers(mcp)
            console.print(f'[dim]Loaded {len(mcp_servers)} MCP server(s) from {mcp}[/dim]')
        except FileNotFoundError as e:
            console.print(f'[red]Error: {e}[/red]')
            return 1
        except ValidationError as e:
            console.print(f'[red]Error parsing MCP config: {e}[/red]')
            return 1
        except ValueError as e:  # pragma: no cover
            console.print(f'[red]Error: {e}[/red]')
            return 1

    # For generic agent, instructions are already in the agent.
    # For provided agent, pass instructions as extra instructions for each run.
    run_instructions = instructions if agent_path else None

    app = create_web_app(
        agent,
        models=models,
        builtin_tools=all_tool_instances or None,
        toolsets=mcp_servers or None,
        instructions=run_instructions,
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
