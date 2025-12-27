"""Skills toolset for Pydantic AI.

This module provides a standardized, composable framework for building and managing
Agent Skills within the Pydantic AI ecosystem. Agent Skills are modular collections
of instructions, scripts, tools, and resources that enable AI agents to progressively
discover, load, and execute specialized capabilities for domain-specific tasks.

See [skills documentation](../skills.md) for more information.

Key components:
- [`SkillsToolset`][pydantic_ai.toolsets.skills.SkillsToolset]: Main toolset for integrating skills with agents
- [`Skill`][pydantic_ai.toolsets.skills.Skill]: Data class representing a loaded skill with resource/script methods
- [`SkillsDirectory`][pydantic_ai.toolsets.skills.SkillsDirectory]: Filesystem-based skill discovery and management
- [`SkillScriptExecutor`][pydantic_ai.toolsets.skills.SkillScriptExecutor]: Protocol for executing skill scripts

Example:
    ```python
    from pydantic_ai import Agent
    from pydantic_ai.toolsets import SkillsToolset

    # Initialize Skills Toolset with skill directories
    skills_toolset = SkillsToolset(directories=["./skills"])

    # Create agent with skills as a toolset
    # Skills instructions are automatically injected via get_instructions()
    agent = Agent(
        model='openai:gpt-4o',
        instructions="You are a helpful research assistant.",
        toolsets=[skills_toolset]
    )

    # Use agent - skills tools are available for the agent to call
    result = await agent.run(
        "What are the last 3 papers on arXiv about machine learning?"
    )
    print(result.output)
    ```
"""

from pydantic_ai.toolsets.skills._directory import SkillsDirectory
from pydantic_ai.toolsets.skills._exceptions import (
    SkillException,
    SkillNotFoundError,
    SkillResourceLoadError,
    SkillResourceNotFoundError,
    SkillScriptExecutionError,
    SkillValidationError,
)
from pydantic_ai.toolsets.skills._local import (
    CallableSkillScriptExecutor,
    LocalSkill,
    LocalSkillScriptExecutor,
)
from pydantic_ai.toolsets.skills._toolset import SkillsToolset
from pydantic_ai.toolsets.skills._types import Skill, SkillMetadata, SkillResource, SkillScript, SkillScriptExecutor

__all__ = (
    # Main toolset
    'SkillsToolset',
    # Directory discovery
    'SkillsDirectory',
    # Executors
    'SkillScriptExecutor',
    'LocalSkillScriptExecutor',
    'CallableSkillScriptExecutor',
    # Types
    'Skill',
    'LocalSkill',
    'SkillMetadata',
    'SkillResource',
    'SkillScript',
    # Exceptions
    'SkillException',
    'SkillNotFoundError',
    'SkillResourceLoadError',
    'SkillResourceNotFoundError',
    'SkillScriptExecutionError',
    'SkillValidationError',
)
