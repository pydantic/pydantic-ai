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
DEFAULT_INSTRUCTION_TEMPLATE = """<skills>
Here is a list of skills that contain domain specific knowledge on a variety of topics.
Each skill comes with a description of the topic and instructions on how to use it.
When a user asks you to perform a task that falls within the domain of a skill, use the `load_skill` tool to acquire the full instructions.

{skills_list}

Use progressive disclosure: load only what you need, when you need it.
- First, use `load_skill(skill_name)` to read the full instructions
- To read additional resources within a skill, use `read_skill_resource(skill_name, resource_name)`
- To execute skill scripts, use `run_skill_script(skill_name, script_name, args)`
</skills>
"""

# Template used by load_skill
LOAD_SKILL_TEMPLATE = """<skill>
<name>{skill_name}</name>
<description>{description}</description>
<path>{path}</path>
<resources>{resources_list}</resources>
<scripts>{scripts_list}</scripts>

<instructions>

{content}

</instructions>

</skill>
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
            for skill in skill_dir.get_skills().values():
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
                Dictionary mapping skill names to brief descriptions.
                Empty dictionary if no skills are available.
            """
            return {name: skill.metadata.description for name, skill in self._skills.items()}

        @self.tool
        async def load_skill(ctx: RunContext[Any], skill_name: str) -> str:  # pyright: ignore[reportUnusedFunction]
            """Load complete instructions and metadata for a specific skill.

            Do NOT infer or guess resource names or script names - they must come from
            the output of this tool.

            Args:
                skill_name: Exact name of the skill.

            Returns:
                Complete skill documentation including:
                - Skill description and purpose
                - List of available resource files (e.g., FORMS.md, REFERENCE.md)
                - List of available scripts with their names
                - Detailed usage instructions and examples
            """
            _ = ctx  # Required by Pydantic AI toolset protocol
            if skill_name not in self._skills:
                available = ', '.join(sorted(self._skills.keys())) or 'none'
                raise SkillNotFoundError(f"Skill '{skill_name}' not found. Available: {available}")

            skill = self._skills[skill_name]

            # Build resources list
            if skill.resources:
                resources_list = '\n'.join(f'<resource>{res.name}</resource>' for res in skill.resources)
            else:
                resources_list = ''

            # Build scripts list
            if skill.scripts:
                scripts_list = '\n'.join(f'<script>{scr.name}</script>' for scr in skill.scripts)
            else:
                scripts_list = ''

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
            """Read a resource file from a skill.

            Do NOT guess or infer resource names, use load_skill first to get the resource names.

            Resources contain supplementary documentation like form templates,
            reference guides, or data schemas.

            Args:
                ctx: Run context (required by toolset protocol).
                skill_name: Exact name of the skill (from list_skills or load_skill).
                resource_name: Exact resource filename as listed in load_skill output
                    (e.g., "FORMS.md", "REFERENCE.md").

            Returns:
                Complete content of the requested resource file.
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
            """Execute a script provided by a skill.

            Do NOT guess or infer script names or arguments.
            Use load_skill first to get the script names and usage instructions.

            Args:
                ctx: Run context (required by toolset protocol).
                skill_name: Exact name of the skill (from list_skills or load_skill).
                script_name: Exact script name as listed in load_skill output (without .py extension).
                args: Command-line arguments as specified in load_skill instructions.
                    Include positional args, flags, and values in order (e.g., ["--format", "json", "input.txt"]).

            Returns:
                Script output including both stdout and stderr.
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
            skills_list_lines.append('<skill>')
            skills_list_lines.append(f'<name>{skill.name}</name>')
            skills_list_lines.append(f'<description>{skill.metadata.description}</description>')
            if skill.uri:
                skills_list_lines.append(f'<path>{skill.uri}</path>')
            skills_list_lines.append('</skill>')
        skills_list = '\n'.join(skills_list_lines)

        # Use custom template or default
        template = self._instruction_template if self._instruction_template else DEFAULT_INSTRUCTION_TEMPLATE

        # Format template with skills list
        return template.format(skills_list=skills_list)
