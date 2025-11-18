"""Agent discovery using AST parsing to find pydantic_ai.Agent objects."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import NamedTuple


class AgentInfo(NamedTuple):
    """Information about a discovered agent."""

    module_path: str  # e.g., "chat.golden_gate_bridge:agent"
    file_path: Path
    agent_name: str


# Directories to exclude from search
EXCLUDE_DIRS = {
    '.venv',
    'venv',
    'env',
    'node_modules',
    '__pycache__',
    '.pytest_cache',
    '.git',
    '.hg',
    '.svn',
    'dist',
    'build',
    '.eggs',
    '.tox',
    '.nox',
    '.mypy_cache',
    '.ruff_cache',
    '.pydantic-work',
}


def _should_exclude_dir(dir_path: Path) -> bool:
    return dir_path.name in EXCLUDE_DIRS or dir_path.name.startswith('.')


def _file_to_module_path(file_path: Path, root_dir: Path) -> str | None:
    """Convert a file path to a Python module path.

    Args:
        file_path: The file path to convert.
        root_dir: The root directory of the project.

    Returns:
        The module path (e.g., "chat.my_agent") or None if conversion fails.
    """
    try:
        rel_path = file_path.relative_to(root_dir)

        if rel_path.suffix != '.py':
            return None

        parts = list(rel_path.parts[:-1]) + [rel_path.stem]
        module_path = '.'.join(parts)

        return module_path
    except (ValueError, AttributeError):
        return None


def _parse_file_for_agents(file_path: Path, root_dir: Path) -> list[AgentInfo]:  # noqa: C901
    """Parse a Python file for pydantic_ai.Agent instances using AST.

    Args:
        file_path: The Python file to parse.
        root_dir: The root directory of the project.

    Returns:
        List of AgentInfo objects for agents found in the file.
    """
    try:
        content = file_path.read_text(encoding='utf-8')
    except (OSError, UnicodeDecodeError):
        # Skip files we can't read
        return []

    try:
        tree = ast.parse(content, filename=str(file_path))
    except SyntaxError:
        # Skip files with syntax errors
        return []

    # Track imports of Agent class
    agent_names: set[str] = set()

    for node in ast.walk(tree):
        # Handle: from pydantic_ai import Agent
        if isinstance(node, ast.ImportFrom):
            if node.module and 'pydantic_ai' in node.module:
                for alias in node.names:
                    if alias.name == 'Agent':
                        agent_names.add(alias.asname or 'Agent')

        # Handle: import pydantic_ai (then pydantic_ai.Agent)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if 'pydantic_ai' in alias.name:
                    # Track as the alias or original name
                    agent_names.add(alias.asname or alias.name)

    if not agent_names:
        # No pydantic_ai imports, skip this file
        return []

    # Find agent instances
    agents: list[AgentInfo] = []
    module_path = _file_to_module_path(file_path, root_dir)

    if not module_path:
        return []

    for node in ast.walk(tree):
        # Look for assignments like: agent = Agent(...)
        if isinstance(node, ast.Assign):
            # Check if the value is a Call to Agent
            if isinstance(node.value, ast.Call):
                call_name = None

                # Handle direct call: Agent(...)
                if isinstance(node.value.func, ast.Name):
                    if node.value.func.id in agent_names:
                        call_name = node.value.func.id

                # Handle attribute call: pydantic_ai.Agent(...)
                elif isinstance(node.value.func, ast.Attribute):
                    if (
                        node.value.func.attr == 'Agent'
                        and isinstance(node.value.func.value, ast.Name)
                        and node.value.func.value.id in agent_names
                    ):
                        call_name = 'Agent'

                if call_name:
                    # Extract variable names being assigned
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            agent_name = target.id
                            agents.append(
                                AgentInfo(
                                    module_path=f'{module_path}:{agent_name}',
                                    file_path=file_path,
                                    agent_name=agent_name,
                                )
                            )

    return agents


def find_agents(root_dir: Path | None = None) -> list[AgentInfo]:
    """Find all pydantic_ai.Agent instances in the current directory.

    Args:
        root_dir: The root directory to search. Defaults to current working directory.

    Returns:
        List of AgentInfo objects for all discovered agents.
    """
    if root_dir is None:
        root_dir = Path.cwd()

    agents: list[AgentInfo] = []
    visited_files: set[Path] = set()

    for path in root_dir.rglob('*.py'):
        if path in visited_files:
            continue
        visited_files.add(path)

        try:
            # Check if any parent directory should be excluded
            if any(_should_exclude_dir(parent) for parent in path.parents):
                continue
        except (OSError, RuntimeError):
            continue

        file_agents = _parse_file_for_agents(path, root_dir)
        agents.extend(file_agents)

    return agents
