"""CLI command for launching a web chat UI for agents."""

from __future__ import annotations

import sys
from pathlib import Path

from pydantic import BaseModel, ImportString, ValidationError
from rich.console import Console

from pydantic_ai import Agent
from pydantic_ai.builtin_tools import AbstractBuiltinTool, get_builtin_tool_cls
from pydantic_ai.models import infer_model
from pydantic_ai.ui._web import create_web_app, load_mcp_server_tools

__all__ = ['_run_web_command']


class _AgentLoader(BaseModel):
    """Helper model for loading agents using Pydantic ImportString."""

    agent: ImportString  # type: ignore[valid-type]


def _load_agent(agent_path: str) -> Agent | None:
    """Load an agent from module path in uvicorn style.

    Args:
        agent_path: Path in format 'module:variable', e.g. 'test_agent:my_agent'

    Returns:
        Agent instance or None if loading fails
    """
    sys.path.insert(0, str(Path.cwd()))
    try:
        loader = _AgentLoader(agent=agent_path)
        agent = loader.agent  # pyright: ignore[reportUnknownVariableType,reportUnknownMemberType]
        if not isinstance(agent, Agent):
            return None
        return agent  # pyright: ignore[reportUnknownVariableType]
    except ValidationError:
        return None


def _run_web_command(  # noqa: C901
    agent_path: str | None = None,
    host: str = '127.0.0.1',
    port: int = 7932,
    models: list[str] | None = None,
    tools: list[str] | None = None,
    instructions: str | None = None,
    mcp: str | None = None,
) -> int:
    """Run the web command to serve an agent via web UI.

    Args:
        agent_path: Agent path in 'module:variable' format. If None, creates generic agent.
        host: Host to bind the server to.
        port: Port to bind the server to.
        models: List of model strings (e.g., ['openai:gpt-5', 'claude-sonnet-4-5']).
        tools: List of builtin tool IDs (e.g., ['web_search', 'code_execution']).
        instructions: System instructions for generic agent.
        mcp: Path to JSON file with MCP server configurations.
    """
    console = Console()

    if agent_path:
        agent = _load_agent(agent_path)
        if agent is None:
            console.print(f'[red]Error: Could not load agent from {agent_path}[/red]')
            return 1
    else:
        agent = Agent()

        if instructions:

            @agent.system_prompt
            def system_prompt() -> str:  # pyright: ignore[reportUnusedFunction]
                return instructions  # pragma: no cover

    if agent.model is None and not models:
        console.print('[red]Error: At least one model (-m) is required when agent has no model[/red]')
        return 1

    # If no CLI models provided but agent has a model, use agent's model
    if not models and agent.model is not None:
        resolved_model = infer_model(agent.model)
        models = [f'{resolved_model.system}:{resolved_model.model_name}']

    # Collect builtin tools: agent's own + CLI-provided
    all_tool_instances: list[AbstractBuiltinTool] = []

    # Add agent's own builtin tools first (these are always enabled)
    all_tool_instances.extend(agent._builtin_tools)  # pyright: ignore[reportPrivateUsage]

    # Parse and add CLI tools
    if tools:
        for tool_id in tools:
            tool_cls = get_builtin_tool_cls(tool_id)
            if tool_cls is None or tool_id in ('url_context', 'mcp_server'):
                console.print(f'[yellow]Warning: Unknown tool "{tool_id}", skipping[/yellow]')
                continue
            if tool_id == 'memory':
                console.print('[yellow]Warning: MemoryTool requires agent to have memory configured, skipping[/yellow]')
                continue
            all_tool_instances.append(tool_cls())

    # Load MCP server tools if specified
    if mcp:
        try:
            mcp_tools = load_mcp_server_tools(mcp)
            all_tool_instances.extend(mcp_tools)
            console.print(f'[dim]Loaded {len(mcp_tools)} MCP server(s) from {mcp}[/dim]')
        except FileNotFoundError as e:
            console.print(f'[red]Error: {e}[/red]')
            return 1
        except ValidationError as e:
            console.print(f'[red]Error parsing MCP config: {e}[/red]')
            return 1
        except ValueError as e:  # pragma: no cover
            console.print(f'[red]Error: {e}[/red]')
            return 1

    app = create_web_app(
        agent,
        models=models,
        builtin_tools=all_tool_instances if all_tool_instances else None,
    )

    agent_desc = agent_path if agent_path else 'generic agent'
    console.print(f'\n[green]Starting chat UI for {agent_desc}...[/green]')
    console.print(f'Open your browser at: [link=http://{host}:{port}]http://{host}:{port}[/link]')
    console.print('[dim]Press Ctrl+C to stop the server[/dim]\n')

    try:
        import uvicorn

        uvicorn.run(app, host=host, port=port)
        return 0  # pragma: no cover
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
