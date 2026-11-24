"""Run the Coder composition with a configurable workspace and model."""

import os
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.models import Model
from pydantic_ai_harness.coder import Coder

DEFAULT_MODEL = os.environ.get('PYDANTIC_AI_MODEL', 'anthropic:claude-opus-5-5')


def build_agent(model: Model | str = DEFAULT_MODEL, workspace: Path | None = None) -> Agent:
    """Build the coding agent for the requested workspace, by default the current directory."""
    return Agent(model, name='coder', capabilities=[LocalWorkspace(workspace or '.'), Coder()])


def main() -> None:
    """Run the coding agent interactively in the current repository."""
    build_agent().to_cli_sync()


if __name__ == '__main__':
    main()
