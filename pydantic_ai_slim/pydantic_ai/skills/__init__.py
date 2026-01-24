"""Skills toolset for Pydantic AI.

This module provides a standardized, composable framework for building and managing
Agent Skills within the Pydantic AI ecosystem. Agent Skills are modular collections
of instructions, scripts, tools, and resources that enable AI agents to progressively
discover, load, and execute specialized capabilities for domain-specific tasks.

See [skills documentation](../skills/overview.md) for more information.

Key components:
- [`SkillsToolset`][pydantic_ai.toolsets.skills.SkillsToolset]: Main toolset for integrating skills with agents
- [`Skill`][pydantic_ai.toolsets.skills.Skill]: Data class representing a skill with resource/script decorators
- [`SkillsDirectory`][pydantic_ai.toolsets.skills.SkillsDirectory]: Filesystem-based skill discovery and management

Example - Programmatic skills:
    ```python
    from pydantic_ai import Agent, RunContext
    from pydantic_ai.toolsets.skills import Skill, SkillResource, SkillsToolset

    # Create a skill with static content and resources
    my_skill = Skill(
        name='hr-analytics-skill',
        description='Skill for HR analytics',
        content='Use this skill for HR data analysis...',
        resources=[
            SkillResource(name='table-schemas', content='Schema definitions...')
        ]
    )

    # Add callable resources using decorator
    @my_skill.resource
    def get_db_context() -> str:
        return "Dynamic database context information."

    @my_skill.resource
    async def get_samples(ctx: RunContext[MyDeps]) -> str:
        return await ctx.deps.fetch_samples()

    # Add callable scripts using decorator
    @my_skill.script
    async def load_dataset(ctx: RunContext[MyDeps]) -> str:
        await ctx.deps.load_data()
        return 'Dataset loaded.'

    @my_skill.script
    async def run_query(ctx: RunContext[MyDeps], query: str) -> str:
        result = await ctx.deps.db.execute(query)
        return str(result)

    # Use with agent
    agent = Agent(
        model='openai:gpt-5',
        toolsets=[SkillsToolset(skills=[my_skill])]
    )
    ```

Example - File-based skills:
    ```python
    from pydantic_ai import Agent
    from pydantic_ai.toolsets import SkillsToolset

    # Initialize Skills Toolset with skill directories
    skills_toolset = SkillsToolset(directories=["./skills"])

    # Create agent with skills as a toolset
    # Skills instructions are automatically injected via get_instructions()
    agent = Agent(
        model='openai:gpt-5.2',
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

from ._directory import SkillsDirectory
from ._exceptions import (
    SkillException,
    SkillNotFoundError,
    SkillResourceLoadError,
    SkillResourceNotFoundError,
    SkillScriptExecutionError,
    SkillValidationError,
)
from ._local import (
    CallableSkillScriptExecutor,
    FileBasedSkillResource,
    FileBasedSkillScript,
    LocalSkillScriptExecutor,
)
from ._types import Skill, SkillResource, SkillScript, SkillWrapper, normalize_skill_name

# SkillsToolset is imported from toolsets module in __all__ for convenience
# but not imported here to avoid circular dependencies

__all__ = (
    # Directory discovery
    'SkillsDirectory',
    # Types
    'Skill',
    'SkillResource',
    'SkillScript',
    'SkillWrapper',
    'normalize_skill_name',
    # Filesystem implementations
    'FileBasedSkillResource',
    'FileBasedSkillScript',
    # Executors (for advanced use cases)
    'LocalSkillScriptExecutor',
    'CallableSkillScriptExecutor',
    # Exceptions
    'SkillException',
    'SkillNotFoundError',
    'SkillResourceLoadError',
    'SkillResourceNotFoundError',
    'SkillScriptExecutionError',
    'SkillValidationError',
)
