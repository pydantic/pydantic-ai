"""Skills toolset implementation.

This module provides the main SkillsToolset class that integrates
skill discovery and management with Pydantic AI agents.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import anyio

from ..._run_context import RunContext
from ..function import FunctionToolset
from ._discovery import discover_skills
from ._exceptions import (
    SkillNotFoundError,
    SkillScriptExecutionError,
)
from ._types import Skill

# Default instruction template for skills system prompt
DEFAULT_INSTRUCTION_TEMPLATE = """# Skills

You have access to skills that extend your capabilities. Skills are modular packages containing instructions, resources, and scripts for specialized tasks.

## Available Skills

The following skills are available to you. Use them when relevant to the task:

{skills_list}

## How to Use Skills

**Progressive disclosure**: Load skill information only when needed.

1. **When a skill is relevant to the current task**: Use `load_skill(skill_name)` to read the full instructions.
2. **For additional documentation**: Use `read_skill_resource(skill_name, resource_name)` to read FORMS.md, REFERENCE.md, or other resources.
3. **To execute skill scripts**: Use `run_skill_script(skill_name, script_name, args)` with appropriate command-line arguments.

**Best practices**:
- Select skills based on task relevance and descriptions listed above
- Use progressive disclosure: load only what you need, when you need it, starting with load_skill
- Follow the skill's documented usage patterns and examples
"""


def _is_safe_path(base_path: Path, target_path: Path) -> bool:
    """Check if target_path is safely within base_path (no path traversal).

    Args:
        base_path: The base directory path.
        target_path: The target path to validate.

    Returns:
        True if target_path is within base_path, False otherwise.
    """
    try:
        target_path.resolve().relative_to(base_path.resolve())
        return True
    except ValueError:
        return False


class SkillsToolset(FunctionToolset):
    """Pydantic AI toolset for automatic skill discovery and integration.

    See [skills docs](../skills.md) for more information.

    This is the primary interface for integrating skills with Pydantic AI agents.
    It implements the toolset protocol and automatically discovers, loads, and
    registers skills from specified directories.

    Provides the following tools to agents:
    - list_skills(): List all available skills
    - load_skill(skill_name): Load a specific skill's instructions
    - read_skill_resource(skill_name, resource_name): Read a skill resource file
    - run_skill_script(skill_name, script_name, args): Execute a skill script

    Example:
        ```python
        from pydantic_ai import Agent, SkillsToolset

        skills_toolset = SkillsToolset(directories=["./skills"])

        agent = Agent(
            model='openai:gpt-4o',
            instructions="You are a helpful assistant.",
            toolsets=[skills_toolset]
        )
        # Skills instructions are automatically injected via get_instructions()
        ```
    """

    def __init__(
        self,
        directories: list[str | Path],
        *,
        auto_discover: bool = True,
        validate: bool = True,
        id: str | None = None,
        script_timeout: int = 30,
        python_executable: str | Path | None = None,
        max_depth: int | None = 3,
        instruction_template: str | None = None,
    ) -> None:
        """Initialize the skills toolset.

        Args:
            directories: List of directory paths to search for skills.
            auto_discover: Automatically discover and load skills on init.
            validate: Validate skill structure and metadata on load.
            id: Unique identifier for this toolset.
            script_timeout: Timeout in seconds for script execution (default: 30).
            python_executable: Path to Python executable for running scripts.
                If None, uses sys.executable (default).
            max_depth: Maximum depth to search for SKILL.md files. None for unlimited.
                Default is 3 levels deep to prevent performance issues with large trees.
            instruction_template: Custom instruction template for skills system prompt.
                Must include `{skills_list}` placeholder. If None, uses default template.
        """
        super().__init__(id=id)

        self._directories = [Path(d) for d in directories]
        self._validate = validate
        self._max_depth = max_depth
        self._script_timeout = script_timeout
        self._python_executable = str(python_executable) if python_executable else sys.executable
        self._instruction_template = instruction_template
        self._skills: dict[str, Skill] = {}

        if auto_discover:
            self._discover_skills()

        # Register tools
        self._register_tools()

    def _discover_skills(self) -> None:
        """Discover and load skills from configured directories."""
        skills = discover_skills(
            directories=self._directories,
            validate=self._validate,
            max_depth=self._max_depth,
        )
        self._skills = {skill.name: skill for skill in skills}

    def _register_tools(self) -> None:  # noqa: C901
        """Register skill management tools with the toolset.

        This method registers all four skill management tools:
        - list_skills: List available skills
        - load_skill: Load skill instructions
        - read_skill_resource: Read skill resources
        - run_skill_script: Execute skill scripts
        """

        @self.tool
        async def list_skills(_ctx: RunContext[Any]) -> str:  # pyright: ignore[reportUnusedFunction]
            """List all available skills with their descriptions.

            Only use this tool if the available skills are not in your system prompt.

            Returns:
                Formatted list of available skills with names and descriptions.
            """
            if not self._skills:
                return 'No skills available.'

            lines = ['# Available Skills', '']

            for name, skill in sorted(self._skills.items()):
                lines.append(f'{name}: {skill.metadata.description}')

            return '\n'.join(lines)

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
                return f"Error: Skill '{skill_name}' not found. Available skills: {available}"

            skill = self._skills[skill_name]

            lines = [
                f'# Skill: {skill.name}',
                f'**Description:** {skill.metadata.description}',
                f'**Path:** {skill.path}',
                '',
            ]

            # Add resource list if available
            if skill.resources:
                lines.append('**Available Resources:**')
                for resource in skill.resources:
                    lines.append(f'- {resource.name}')
                lines.append('')

            # Add scripts list if available
            if skill.scripts:
                lines.append('**Available Scripts:**')
                for script in skill.scripts:
                    lines.append(f'- {script.name}')
                lines.append('')

            lines.append('---')
            lines.append('')
            lines.append(skill.content)

            return '\n'.join(lines)

        @self.tool
        async def read_skill_resource(  # noqa: D417  # pyright: ignore[reportUnusedFunction]
            ctx: RunContext[Any],
            skill_name: str,
            resource_name: str,
        ) -> str:
            """Read a resource file from a skill (e.g., FORMS.md, REFERENCE.md).

            Call load_skill first to see which resources are available.

            Args:
                skill_name: Name of the skill.
                resource_name: The resource filename (e.g., "FORMS.md").

            Returns:
                The resource file content.
            """
            _ = ctx  # Required by Pydantic AI toolset protocol
            if skill_name not in self._skills:
                return f"Error: Skill '{skill_name}' not found."

            skill = self._skills[skill_name]

            # Find the resource
            resource = None
            for r in skill.resources:
                if r.name == resource_name:
                    resource = r
                    break

            if resource is None:
                available = [r.name for r in skill.resources]
                return (
                    f"Error: Resource '{resource_name}' not found in skill '{skill_name}'. "
                    f'Available resources: {available}'
                )

            # Security check
            if not _is_safe_path(skill.path, resource.path):
                return 'Error: Resource path escapes skill directory.'

            try:
                content = resource.path.read_text(encoding='utf-8')
                return content
            except OSError as e:
                return f"Error: Failed to read resource '{resource_name}': {e}"

        @self.tool
        async def run_skill_script(  # noqa: D417  # pyright: ignore[reportUnusedFunction]
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
                skill_name: Name of the skill.
                script_name: The script name (without .py extension).
                args: Optional list of command-line arguments (positional args, flags, values).

            Returns:
                The script's output (stdout and stderr combined).
            """
            _ = ctx  # Required by Pydantic AI toolset protocol
            if skill_name not in self._skills:
                return f"Error: Skill '{skill_name}' not found."

            skill = self._skills[skill_name]

            # Find the script
            script = None
            for s in skill.scripts:
                if s.name == script_name:
                    script = s
                    break

            if script is None:
                available = [s.name for s in skill.scripts]
                return (
                    f"Error: Script '{script_name}' not found in skill '{skill_name}'. Available scripts: {available}"
                )

            # Security check
            if not _is_safe_path(skill.path, script.path):
                return 'Error: Script path escapes skill directory.'

            # Build command
            cmd = [self._python_executable, str(script.path)]
            if args:
                cmd.extend(args)

            try:
                # Use anyio.run_process for async-compatible execution
                result = None
                with anyio.move_on_after(self._script_timeout) as scope:
                    result = await anyio.run_process(
                        cmd,
                        check=False,  # We handle return codes manually
                        cwd=str(skill.path),
                    )

                # Check if timeout was reached
                if scope.cancelled_caught:
                    raise SkillScriptExecutionError(
                        f"Script '{script_name}' timed out after {self._script_timeout} seconds"
                    )

                # At this point, result should be set; if not, treat as an execution error
                if result is None:
                    raise SkillScriptExecutionError(
                        f"Script '{script_name}' did not complete execution; no result was returned"
                    )

                # Decode output from bytes to string
                output = result.stdout.decode('utf-8', errors='replace')
                if result.stderr:
                    stderr = result.stderr.decode('utf-8', errors='replace')
                    output += f'\n\nStderr:\n{stderr}'

                if result.returncode != 0:
                    output += f'\n\nScript exited with code {result.returncode}'

                return output.strip() or '(no output)'

            except OSError as e:
                raise SkillScriptExecutionError(f"Failed to execute script '{script_name}': {e}") from e

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
        for name, skill in sorted(self._skills.items()):
            skills_list_lines.append(f'- **{name}**: {skill.metadata.description}')
        skills_list = '\n'.join(skills_list_lines)

        # Use custom template or default
        template = self._instruction_template if self._instruction_template else DEFAULT_INSTRUCTION_TEMPLATE

        # Format template with skills list
        return template.format(skills_list=skills_list)

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
            name: The skill name.

        Returns:
            The Skill object.

        Raises:
            SkillNotFoundError: If the skill is not found.
        """
        if name not in self._skills:
            raise SkillNotFoundError(f"Skill '{name}' not found")
        return self._skills[name]

    def refresh(self) -> None:
        """Re-discover skills from configured directories.

        Call this method to reload skills after changes to the filesystem.
        """
        self._discover_skills()
