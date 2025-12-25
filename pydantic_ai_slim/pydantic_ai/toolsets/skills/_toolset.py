"""Skills toolset implementation.

This module provides the main SkillsToolset class that integrates
skill discovery and management with Pydantic AI agents.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

from ..._run_context import RunContext
from ..function import FunctionToolset
from ._directory import SkillsDirectory
from ._exceptions import SkillNotFoundError
from ._types import Skill

# Default instruction template for skills system prompt
DEFAULT_INSTRUCTION_TEMPLATE = """## Skills

You have access to skills that extend your capabilities. Skills are modular packages containing instructions, resources, and scripts for specialized tasks.

### Available Skills

The following skills are available to you. Use them when relevant to the task:

{skills_list}

### How to Use Skills

**Progressive disclosure**: Load skill information only when needed.

1. **When a skill is relevant to the current task**: Use `load_skill(skill_name)` to read the full instructions.
2. **For additional documentation**: Use `read_skill_resource(skill_name, resource_name)` to read FORMS.md, REFERENCE.md, or other resources.
3. **To execute skill scripts**: Use `run_skill_script(skill_name, script_name, args)` with appropriate command-line arguments.

**Best practices**:
- Select skills based on task relevance and descriptions listed above
- Use progressive disclosure: load only what you need, when you need it, starting with load_skill
- Follow the skill's documented usage patterns and examples
"""

# Template used by load_skill
LOAD_SKILL_TEMPLATE = """# Skill: {skill_name}
**Description:** {description}
**Path:** {path}
**Available Resources:**
{resources_list}
**Available Scripts:**
{scripts_list}

---

{content}
"""


class SkillsToolset(FunctionToolset):
    """Pydantic AI toolset for automatic skill discovery and integration.

    See [skills docs](../skills.md) for more information.

    This is the primary interface for integrating skills with Pydantic AI agents.
    It manages skills directly and provides tools for skill interaction.

    Provides the following tools to agents:
    - list_skills(): List all available skills
    - load_skill(skill_name): Load a specific skill's instructions
    - read_skill_resource(skill_name, resource_name): Read a skill resource file
    - run_skill_script(skill_name, script_name, args): Execute a skill script

    Example:
        ```python
        from pydantic_ai import Agent, SkillsToolset

        # Default: uses ./skills directory
        agent = Agent(
            model='openai:gpt-4o',
            instructions="You are a helpful assistant.",
            toolsets=[SkillsToolset()]
        )

        # Multiple directories
        agent = Agent(
            model='openai:gpt-4o',
            toolsets=[SkillsToolset(directories=["./skills", "./more-skills"])]
        )

        # Programmatic skills
        from pydantic_ai.toolsets.skills import Skill, SkillMetadata

        custom_skill = Skill(
            name="my-skill",
            uri="./custom",
            metadata=SkillMetadata(name="my-skill", description="Custom skill"),
            content="Instructions here",
        )
        agent = Agent(
            model='openai:gpt-4o',
            toolsets=[SkillsToolset(skills=[custom_skill])]
        )

        # Combined mode: both programmatic skills and directories
        agent = Agent(
            model='openai:gpt-4o',
            toolsets=[SkillsToolset(
                skills=[custom_skill],
                directories=["./skills"]
            )]
        )

        # Using SkillsDirectory instances directly
        from pydantic_ai.toolsets.skills import SkillsDirectory

        dir1 = SkillsDirectory(path="./skills")
        agent = Agent(
            model='openai:gpt-4o',
            toolsets=[SkillsToolset(directories=[dir1, "./more-skills"])]
        )
        # Skills instructions are automatically injected via get_instructions()
        ```
    """

    def __init__(
        self,
        *,
        skills: list[Skill] | None = None,
        directories: list[str | Path | SkillsDirectory] | None = None,
        validate: bool = True,
        max_depth: int | None = 3,
        id: str | None = None,
        instruction_template: str | None = None,
    ) -> None:
        """Initialize the skills toolset.

        Args:
            skills: List of pre-loaded Skill objects. Can be combined with `directories`.
            directories: List of directories or SkillsDirectory instances to discover skills from.
                Can be combined with `skills`. If both are None, defaults to ["./skills"].
                String/Path entries are converted to SkillsDirectory instances.
            validate: Validate skill structure during discovery (used when creating SkillsDirectory from str/Path).
            max_depth: Maximum depth for skill discovery (None for unlimited, used when creating SkillsDirectory from str/Path).
            id: Unique identifier for this toolset.
            instruction_template: Custom instruction template for skills system prompt.
                Must include `{skills_list}` placeholder. If None, uses default template.

        Example:
            ```python
            # Default: uses ./skills directory
            toolset = SkillsToolset()

            # Multiple directories
            toolset = SkillsToolset(directories=["./skills", "./more-skills"])

            # Programmatic skills
            toolset = SkillsToolset(skills=[skill1, skill2])

            # Combined mode
            toolset = SkillsToolset(
                skills=[skill1, skill2],
                directories=["./skills", skills_dir_instance]
            )

            # Using SkillsDirectory instances directly
            dir1 = SkillsDirectory(path="./skills")
            toolset = SkillsToolset(directories=[dir1])
            ```
        """
        super().__init__(id=id)

        self._instruction_template = instruction_template

        # Initialize the skills dict and directories list (for refresh)
        self._skills: dict[str, Skill] = {}
        self._skill_directories: list[SkillsDirectory] = []
        self._validate = validate
        self._max_depth = max_depth

        # Load programmatic skills first
        if skills is not None:
            for skill in skills:
                if skill.name in self._skills:
                    warnings.warn(
                        f"Duplicate skill '{skill.name}' found. Overriding previous occurrence.",
                        UserWarning,
                        stacklevel=2,
                    )
                self._skills[skill.name] = skill

        # Load directory-based skills
        if directories is not None:
            self._load_directory_skills(directories)
        elif skills is None:
            # Default: ./skills directory (only if no skills provided)
            default_dir = Path('./skills')
            if not default_dir.exists():
                warnings.warn(
                    f"Default skills directory '{default_dir}' does not exist. No skills will be loaded.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                self._load_directory_skills([default_dir])

        # Register tools
        self._register_tools()

    @property
    def skills(self) -> dict[str, Skill]:
        """Get the dictionary of loaded skills.

        Returns:
            Dictionary mapping skill names to Skill objects.
        """
        return self._skills

    def get_skill(self, name: str) -> Skill:
        """Get a specific skill by name.

        Args:
            name: Name of the skill to get.

        Returns:
            The requested Skill object.

        Raises:
            SkillNotFoundError: If skill is not found.
        """
        if name not in self._skills:
            available = ', '.join(sorted(self._skills.keys())) or 'none'
            raise SkillNotFoundError(f"Skill '{name}' not found. Available: {available}")
        return self._skills[name]

    def _load_directory_skills(self, directories: list[str | Path | SkillsDirectory]) -> None:
        """Load skills from configured directories.

        Converts directory specifications to SkillsDirectory instances and
        discovers skills from each directory in a single pass.

        Args:
            directories: List of directory paths or SkillsDirectory instances.
        """
        for directory in directories:
            # Normalize to SkillsDirectory instance
            if isinstance(directory, SkillsDirectory):
                skill_dir = directory
            else:
                skill_dir = SkillsDirectory(
                    path=directory,
                    validate=self._validate,
                    max_depth=self._max_depth,
                )

            # Store for future reference
            self._skill_directories.append(skill_dir)

            # Discover skills from this directory (last one wins)
            for _, skill in skill_dir.get_skills().items():
                skill_name = skill.name
                if skill_name in self._skills:
                    warnings.warn(
                        f"Duplicate skill '{skill_name}' found. Overriding previous occurrence.",
                        UserWarning,
                        stacklevel=3,
                    )
                self._skills[skill_name] = skill

    def _register_tools(self) -> None:
        """Register skill management tools with the toolset.

        This method registers all four skill management tools:
        - list_skills: List available skills
        - load_skill: Load skill instructions
        - read_skill_resource: Read skill resources
        - run_skill_script: Execute skill scripts
        """

        @self.tool
        async def list_skills(_ctx: RunContext[Any]) -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
            """List all available skills with their descriptions.

            Only use this tool if the available skills are not in your system prompt.

            Returns:
                Dictionary mapping skill names to their descriptions.
                    An empty dictionary if no skills are available.
            """
            return {name: skill.metadata.description for name, skill in self._skills.items()}

        @self.tool
        async def load_skill(ctx: RunContext[Any], skill_name: str) -> str:  # pyright: ignore[reportUnusedFunction]
            """Load full instructions for a skill.

            Always load the skill before using read_skill_resource
            or run_skill_script to understand the skill's capabilities, available
            resources, scripts, and their usage patterns.

            Args:
                ctx: Run context (required by toolset protocol).
                skill_name: Name of the skill to load.

            Returns:
                Full skill instructions including available resources and scripts.
            """
            _ = ctx  # Required by Pydantic AI toolset protocol
            if skill_name not in self._skills:
                available = ', '.join(sorted(self._skills.keys())) or 'none'
                raise SkillNotFoundError(f"Skill '{skill_name}' not found. Available: {available}")

            skill = self._skills[skill_name]

            # Build resources list
            if skill.resources:
                resources_list = '\n'.join(f'- {res.name}' for res in skill.resources)
            else:
                resources_list = 'No resources available.'

            # Build scripts list
            if skill.scripts:
                scripts_list = '\n'.join(f'- {scr.name}' for scr in skill.scripts)
            else:
                scripts_list = 'No scripts available.'

            # Format response
            return LOAD_SKILL_TEMPLATE.format(
                skill_name=skill.name,
                description=skill.metadata.description,
                path=skill.uri or 'N/A',
                resources_list=resources_list,
                scripts_list=scripts_list,
                content=skill.content,
            )

        @self.tool
        async def read_skill_resource(  # pyright: ignore[reportUnusedFunction]
            ctx: RunContext[Any],
            skill_name: str,
            resource_name: str,
        ) -> str:
            """Read a resource file from a skill (e.g., FORMS.md, REFERENCE.md).

            Call load_skill first to see which resources are available.

            Args:
                ctx: Run context (required by toolset protocol).
                skill_name: Name of the skill.
                resource_name: The resource filename (e.g., "FORMS.md").

            Returns:
                The resource file content.
            """
            if skill_name not in self._skills:
                raise SkillNotFoundError(f"Skill '{skill_name}' not found.")

            skill = self._skills[skill_name]
            return await skill.read_resource(ctx, resource_name)

        @self.tool
        async def run_skill_script(  # pyright: ignore[reportUnusedFunction]
            ctx: RunContext[Any],
            skill_name: str,
            script_name: str,
            args: list[str] | None = None,
        ) -> str:
            """Execute a skill script with command-line arguments.

            Call load_skill first to understand the script's expected arguments,
            usage patterns, and example invocations. Running scripts without
            loading instructions first will likely fail.

            Args:
                ctx: Run context (required by toolset protocol).
                skill_name: Name of the skill.
                script_name: The script name (without .py extension).
                args: Optional list of command-line arguments (positional args, flags, values).

            Returns:
                The script's output (stdout and stderr combined).
            """
            if skill_name not in self._skills:
                raise SkillNotFoundError(f"Skill '{skill_name}' not found.")

            skill = self._skills[skill_name]
            return await skill.run_script(ctx, script_name, args)

    async def get_instructions(self, ctx: RunContext[Any]) -> str | None:
        """Return instructions to inject into the agent's system prompt.

        Returns the skills system prompt containing all skill metadata
        and usage guidance for the agent.

        Args:
            ctx: The run context for this agent run.

        Returns:
            The skills system prompt, or None if no skills are loaded.
        """
        if not self._skills:
            return None

        # Build skills list
        skills_list_lines: list[str] = []
        for skill in sorted(self._skills.values(), key=lambda s: s.name):
            skills_list_lines.append(f'- **{skill.name}**: {skill.metadata.description}')
        skills_list = '\n'.join(skills_list_lines)

        # Use custom template or default
        template = self._instruction_template if self._instruction_template else DEFAULT_INSTRUCTION_TEMPLATE

        # Format template with skills list
        return template.format(skills_list=skills_list)
