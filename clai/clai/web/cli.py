"""CLI command for launching a web chat UI for discovered agents."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.builtin_tools import AbstractBuiltinTool
from pydantic_ai.ui.web import AIModel, BuiltinTool, create_chat_app


def load_agent_options(
    config_path: Path,
) -> tuple[list[AIModel] | None, dict[str, AbstractBuiltinTool] | None, list[BuiltinTool] | None]:
    """Load agent options from a config file.

    Args:
        config_path: Path to the config file (e.g., agent_options.py)

    Returns:
        Tuple of (models, builtin_tools, builtin_tool_defs) or (None, None, None) if not found
    """
    if not config_path.exists():
        return None, None, None

    try:
        spec = importlib.util.spec_from_file_location('agent_options_config', config_path)
        if spec is None or spec.loader is None:
            print(f'Warning: Could not load config from {config_path}')
            return None, None, None

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        models = getattr(module, 'AI_MODELS', None)
        builtin_tools = getattr(module, 'BUILTIN_TOOLS', None)
        builtin_tool_defs = getattr(module, 'BUILTIN_TOOL_DEFS', None)

        return models, builtin_tools, builtin_tool_defs

    except Exception as e:
        print(f'Warning: Error loading config from {config_path}: {e}')
        return None, None, None


def load_agent(agent_path: str) -> Agent | None:
    """Load an agent from module path in uvicorn style.

    Args:
        agent_path: Path in format 'module:variable', e.g. 'test_agent:my_agent'

    Returns:
        Agent instance or None if loading fails
    """
    sys.path.insert(0, str(Path.cwd()))

    try:
        module_path, variable_name = agent_path.split(':')
    except ValueError:
        print('Error: Agent must be specified in "module:variable" format')
        return None

    try:
        module = importlib.import_module(module_path)
        agent = getattr(module, variable_name, None)

        if agent is None:
            print(f'Error: {variable_name} not found in module {module_path}')
            return None

        if not isinstance(agent, Agent):
            print(f'Error: {variable_name} is not an Agent instance')
            return None

        return agent  # pyright: ignore[reportUnknownVariableType]

    except ImportError as e:
        print(f'Error: Could not import module {module_path}: {e}')
        return None
    except Exception as e:
        print(f'Error loading agent: {e}')
        return None


def run_chat_command(
    agent_path: str,
    host: str = '127.0.0.1',
    port: int = 8000,
    config_path: Path | None = None,
    auto_config: bool = True,
) -> int:
    """Run the chat command to serve an agent via web UI.

    Args:
        agent_path: Agent path in 'module:variable' format, e.g. 'test_agent:my_agent'
        host: Host to bind the server to
        port: Port to bind the server to
        config_path: Path to agent_options.py config file
        auto_config: Auto-discover agent_options.py in current directory
    """
    agent = load_agent(agent_path)
    if agent is None:
        return 1

    models, builtin_tools, builtin_tool_defs = None, None, None
    if config_path:
        print(f'Loading config from {config_path}...')
        models, builtin_tools, builtin_tool_defs = load_agent_options(config_path)
    elif auto_config:
        default_config = Path.cwd() / 'agent_options.py'
        if default_config.exists():
            print(f'Found config file: {default_config}')
            models, builtin_tools, builtin_tool_defs = load_agent_options(default_config)

    app = create_chat_app(agent, models=models, builtin_tools=builtin_tools, builtin_tool_defs=builtin_tool_defs)

    print(f'\nStarting chat UI for {agent_path}...')
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
