"""CLI command for launching a web chat UI for discovered agents."""

from __future__ import annotations

import logging

from pydantic import ValidationError

from pydantic_ai import Agent
from pydantic_ai._cli import _load_agent  # pyright: ignore[reportPrivateUsage]
from pydantic_ai.builtin_tools import (
    BUILTIN_TOOL_CLASSES,
    BUILTIN_TOOL_ID,
    AbstractBuiltinTool,
)
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import infer_model
from pydantic_ai.ui.web import AIModel, create_web_app, format_model_display_name, load_mcp_server_tools


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

    Args:
        agent_path: Agent path in 'module:variable' format. If None, creates generic agent.
        host: Host to bind the server to
        port: Port to bind the server to
        models: List of model strings (e.g., ['openai:gpt-5', 'claude-sonnet-4-5'])
        tools: List of builtin tool IDs (e.g., ['web_search', 'code_execution'])
        instructions: System instructions for generic agent
        mcp: Path to JSON file with MCP server configurations
    """
    if agent_path:
        agent = _load_agent(agent_path)
        if agent is None:
            print(f'Error: Could not load agent from {agent_path}')
            return 1
    else:
        agent = Agent()

        if instructions:

            @agent.system_prompt
            def system_prompt() -> str:  # pyright: ignore[reportUnusedFunction]
                return instructions

    if agent.model is None and not models:
        print('Error: At least one model (-m) is required when agent has no model')
        return 1

    parsed_tools_ids: list[BUILTIN_TOOL_ID] = []
    if tools:
        for tool_id in tools:
            if tool_id not in BUILTIN_TOOL_ID.__args__:
                logging.warning(
                    f'Tool "{tool_id}" is not a valid builtin tool ID. Valid IDs are: {BUILTIN_TOOL_ID.__args__}'
                )
                continue
            parsed_tools_ids.append(tool_id)  # pyright: ignore[reportArgumentType]

    parsed_models: list[AIModel] = []

    if models:
        for model_str in models:
            if ':' in model_str:
                model_id = model_str
                model_name = model_str.split(':', 1)[1]
            else:
                try:
                    inferred = infer_model(model_str)
                    model_id = f'{inferred.system}:{inferred.model_name}'
                    model_name = inferred.model_name
                except UserError:
                    print(f'Error: Model "{model_str}" requires a provider prefix (e.g., "openai:{model_str}")')
                    return 1

            display_name = format_model_display_name(model_name)

            parsed_models.append(
                AIModel(
                    id=model_id,
                    name=display_name,
                    builtin_tools=parsed_tools_ids,
                )
            )

    parsed_tool_instances: list[AbstractBuiltinTool] = []
    if tools:
        for tool_id in tools:
            if tool_id not in BUILTIN_TOOL_CLASSES:
                print(f'Warning: Unknown tool "{tool_id}", skipping')
                continue

            tool_class = BUILTIN_TOOL_CLASSES[tool_id]
            parsed_tool_instances.append(tool_class())

    if mcp:
        try:
            mcp_tools = load_mcp_server_tools(mcp)
            parsed_tool_instances.extend(mcp_tools)
            print(f'Loaded {len(mcp_tools)} MCP server(s) from {mcp}')
        except FileNotFoundError as e:
            print(f'Error: {e}')
            return 1
        except ValidationError as e:
            print(f'Error parsing MCP config: {e}')
            return 1
        except ValueError as e:
            print(f'Error: {e}')
            return 1

    if parsed_models:
        first_model = parsed_models[0]
        agent.model = infer_model(first_model.id)

    app = create_web_app(
        agent,
        models=parsed_models if parsed_models else None,
        builtin_tools=parsed_tool_instances if parsed_tool_instances else None,
    )

    agent_desc = agent_path if agent_path else 'generic agent'
    print(f'\nStarting chat UI for {agent_desc}...')
    print(f'Open your browser at: http://{host}:{port}')
    print('Press Ctrl+C to stop the server\n')

    try:
        import uvicorn

        uvicorn.run(app, host=host, port=port)
        return 0
    except KeyboardInterrupt:
        print('\nServer stopped.')
        return 0
    except ImportError:
        print('Error: uvicorn is required to run the chat UI')
        print('Install it with: pip install uvicorn')
        return 1
    except Exception as e:
        print(f'Error starting server: {e}')
        return 1
