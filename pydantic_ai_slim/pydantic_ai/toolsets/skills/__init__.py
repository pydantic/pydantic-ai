"""Skills toolset for Pydantic AI.

This module provides a standardized, composable framework for building and managing
Agent Skills within the Pydantic AI ecosystem. Agent Skills are modular collections
of instructions, scripts, tools, and resources that enable AI agents to progressively
discover, load, and execute specialized capabilities for domain-specific tasks.

Example:
    ```python
    from pydantic_ai import Agent, SkillsToolset

    # Initialize Skills Toolset with one or more skill directories
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

from pydantic_ai.toolsets.skills._discovery import discover_skills, parse_skill_md
from pydantic_ai.toolsets.skills._exceptions import (
    SkillException,
    SkillNotFoundError,
    SkillResourceLoadError,
    SkillScriptExecutionError,
    SkillValidationError,
)
from pydantic_ai.toolsets.skills._toolset import SkillsToolset
from pydantic_ai.toolsets.skills._types import Skill, SkillMetadata, SkillResource, SkillScript

__all__ = (
    # Main toolset
    'SkillsToolset',
    # Types
    'Skill',
    'SkillMetadata',
    'SkillResource',
    'SkillScript',
    # Discovery
    'discover_skills',
    'parse_skill_md',
    # Exceptions
    'SkillException',
    'SkillNotFoundError',
    'SkillResourceLoadError',
    'SkillScriptExecutionError',
    'SkillValidationError',
)
