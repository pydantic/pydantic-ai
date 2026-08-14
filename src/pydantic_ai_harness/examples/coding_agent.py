"""Build a coding agent from the blocks packaged as `Coder`.

Run the packaged equivalent without assembling the blocks:

    uvx --with pydantic-ai-harness clai -a pydantic_ai_harness.coder:coder_agent -m anthropic:claude-fable-5
"""

import os
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.models import Model

from pydantic_ai_harness import (
    LLM_API_KEY_ENV_PATTERNS,
    ClearToolResults,
    FileSystem,
    Planning,
    RepoContext,
    Shell,
    SubAgent,
    SubAgents,
    ToolOutputLimits,
    WarnNearLimits,
)

DEFAULT_MODEL = os.environ.get('PYDANTIC_AI_MODEL', 'anthropic:claude-fable-5')

# Keep this blown-out composition in sync across docs/coder.md, docs/index.md, README.md,
# pydantic_ai_harness/coder/README.md, and examples/coding_agent.py.
ALLOWED_COMMANDS = (
    'git',
    'rg',
    'grep',
    'find',
    'ls',
    'cat',
    'sed',
    'head',
    'tail',
    'python',
    'uv',
    'pytest',
    'ruff',
    'make',
)


def build_agent(model: Model | str = DEFAULT_MODEL, workspace: Path | None = None) -> Agent:
    """Build the coding agent for `workspace` (defaults to the current directory)."""
    workspace = workspace or Path.cwd()
    explorer = SubAgent(
        Agent(
            name='explorer',
            description='Explore the codebase and answer questions without modifying anything',
            instructions='Answer with concrete paths and evidence.',
            capabilities=[
                FileSystem(workspace, read_only=True),
                RepoContext(workspace_dir=workspace),
            ],
        )
    )
    return Agent(
        model,
        name='coder',
        instructions='You are a coding agent built on Pydantic AI.',
        capabilities=[
            FileSystem(workspace),  # read/write/edit/search, path-traversal safe
            Shell(  # allowlisted commands, LLM API keys stripped from their environment
                cwd=workspace,
                allowed_commands=ALLOWED_COMMANDS,
                denied_env_patterns=LLM_API_KEY_ENV_PATTERNS,
            ),
            RepoContext(workspace_dir=workspace),  # loads AGENTS.md/CLAUDE.md + repo structure
            Planning(),  # structured task plans the model maintains
            SubAgents(agents=[explorer], agent_folders=None),  # delegate exploration off the main context
            ClearToolResults(max_fraction=0.7),  # clears old tool results near the limit
            WarnNearLimits(max_context_fraction=0.9),  # warns the model before it hits limits
            ToolOutputLimits(),  # bounds oversized tool results
        ],
    )


def main() -> None:
    """Start an interactive session in the current repository."""
    build_agent().to_cli_sync()


if __name__ == '__main__':
    main()
