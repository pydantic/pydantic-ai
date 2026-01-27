"""Skills toolset implementation.

This module provides the main [`SkillsToolset`][pydantic_ai.toolsets.skills.SkillsToolset]
class that integrates skill discovery and management with Pydantic AI agents.

The toolset provides four main tools for agents:
- list_skills: List all available skills
- load_skill: Load full instructions for a specific skill
- read_skill_resource: Read skill resource files or invoke callable resources
- run_skill_script: Execute skill scripts
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable
from inspect import signature as get_signature
from pathlib import Path
from typing import Any

from .._griffe import doc_descriptions
from .._run_context import RunContext
from ..skills import Skill, SkillResource, SkillScript, SkillsDirectory, SkillWrapper, normalize_skill_name
from ..skills._exceptions import SkillNotFoundError, SkillResourceNotFoundError, SkillValidationError
from ..skills._types import SKILL_NAME_PATTERN
from .function import FunctionToolset

# Default instruction template for skills system prompt
_INSTRUCTION_SKILLS_HEADER = """\
You have access to a collection of skills containing domain-specific knowledge and capabilities.
Each skill provides specialized instructions, resources, and scripts for specific tasks.

<available_skills>
{skills_list}
</available_skills>

When a task falls within a skill's domain:
1. Use `load_skill` to read the complete skill instructions
2. Follow the skill's guidance to complete the task
3. Use any additional skill resources and scripts as needed

Use progressive disclosure: load only what you need, when you need it."""

# Template used by load_skill
LOAD_SKILL_TEMPLATE = """<skill>
<name>{skill_name}</name>
<description>{description}</description>
<uri>{uri}</uri>

<resources>
{resources_list}
</resources>

<scripts>
{scripts_list}
</scripts>

<instructions>
{content}
</instructions>
</skill>
"""


class SkillsToolset(FunctionToolset):
    """Pydantic AI toolset for automatic skill discovery and integration.

    See [skills docs](../skills/overview.md) for more information.

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
            model='openai:gpt-5.2',
            instructions="You are a helpful assistant.",
            toolsets=[SkillsToolset()]
        )

        # Multiple directories
        agent = Agent(
            model='openai:gpt-5.2',
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
            model='openai:gpt-5.2',
            toolsets=[SkillsToolset(skills=[custom_skill])]
        )

        # Combined mode: both programmatic skills and directories
        agent = Agent(
            model='openai:gpt-5.2',
            toolsets=[SkillsToolset(
                skills=[custom_skill],
                directories=["./skills"]
            )]
        )

        # Using SkillsDirectory instances directly
        from pydantic_ai.toolsets.skills import SkillsDirectory

        dir1 = SkillsDirectory(path="./skills")
        agent = Agent(
            model='openai:gpt-5.2',
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
                Tool usage guidance is provided in the tool docstrings themselves.

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
                self._register_skill(skill)

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

    def _build_resource_xml(self, resource: SkillResource) -> str:
        """Build XML representation of a resource.

        Args:
            resource: The resource to format.

        Returns:
            XML string representation of the resource.
        """
        res_xml = f'<resource name="{resource.name}"'
        if resource.description:
            res_xml += f' description="{resource.description}"'
        if resource.function and resource.function_schema:
            params_json = json.dumps(resource.function_schema.json_schema)
            res_xml += f' parameters={json.dumps(params_json)}'
        res_xml += ' />'
        return res_xml

    def _build_script_xml(self, script: SkillScript) -> str:
        """Build XML representation of a script.

        Args:
            script: The script to format.

        Returns:
            XML string representation of the script.
        """
        scr_xml = f'<script name="{script.name}"'
        if script.description:
            scr_xml += f' description="{script.description}"'
        if script.function and script.function_schema:
            params_json = json.dumps(script.function_schema.json_schema)
            scr_xml += f' parameters={json.dumps(params_json)}'
        scr_xml += ' />'
        return scr_xml

    def _find_skill_resource(self, skill: Skill, resource_name: str) -> SkillResource | None:
        """Find a resource in a skill by name.

        Args:
            skill: The skill to search in.
            resource_name: The resource name to find.

        Returns:
            The resource if found, None otherwise.
        """
        if not skill.resources:
            return None
        for r in skill.resources:
            if r.name == resource_name:
                return r
        return None

    def _find_skill_script(self, skill: Skill, script_name: str) -> SkillScript | None:
        """Find a script in a skill by name.

        Args:
            skill: The skill to search in.
            script_name: The script name to find.

        Returns:
            The script if found, None otherwise.
        """
        if not skill.scripts:
            return None
        for s in skill.scripts:
            if s.name == script_name:
                return s
        return None

    def _register_tools(self) -> None:
        """Register all skill management tools with the toolset.

        Registers the following tools:
        - list_skills: List all available skills
        - load_skill: Load a skill's complete instructions
        - read_skill_resource: Access skill resources
        - run_skill_script: Execute skill scripts

        To exclude specific tools, use the filtered() composition method.
        """
        self._register_list_skills()
        self._register_load_skill()
        self._register_read_skill_resource()
        self._register_run_skill_script()

    def _register_list_skills(self) -> None:
        """Register the list_skills tool."""

        @self.tool
        async def list_skills(_ctx: RunContext[Any]) -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
            """Get an overview of all available skills and what they do.

            Use this when you need to discover what skills exist or refresh your knowledge
            of available capabilities. Skills provide domain-specific knowledge and instructions
            for specialized tasks.

            Returns:
                Dictionary mapping skill names to their descriptions.
                Empty dictionary if no skills are available.
            """
            return {name: skill.description for name, skill in self._skills.items()}

    def _register_load_skill(self) -> None:
        """Register the load_skill tool."""

        @self.tool
        async def load_skill(ctx: RunContext[Any], skill_name: str) -> str:  # pyright: ignore[reportUnusedFunction]  # noqa: D417
            """Load complete instructions and capabilities for a specific skill.

            A skill contains detailed instructions, supplementary resources (like templates or
            reference docs), and executable scripts. Load a skill when you need to perform a
            task within its domain.

            Args:
                skill_name: Exact name from your available skills list.
                    Must match exactly (e.g., "data-analysis" not "data analysis").

            Returns:
                Structured documentation containing:
                - Skill name, description, and source location
                - Available resources: supplementary files with their parameters
                - Available scripts: executable programs with their parameters
                - Detailed instructions: step-by-step guidance for using the skill

            Important:
            - Read the entire instructions section before taking action
            - Resource and script names are authoritative - use exact names from the output
            - Do NOT infer or guess resource/script names
            - Check parameter schemas if resources or scripts require arguments
            """
            _ = ctx  # Required by Pydantic AI toolset protocol
            if skill_name not in self._skills:
                available = ', '.join(sorted(self._skills.keys())) or 'none'
                raise SkillNotFoundError(f"Skill '{skill_name}' not found. Available: {available}")

            skill = self._skills[skill_name]

            # Build resources list with schemas for callable resources
            resources_parts: list[str] = []
            if skill.resources:
                for res in skill.resources:
                    resources_parts.append(self._build_resource_xml(res))
            resources_list = '\n'.join(resources_parts) if resources_parts else '<!-- No resources -->'

            # Build scripts list with schemas for callable scripts
            scripts_parts: list[str] = []
            if skill.scripts:
                for scr in skill.scripts:
                    scripts_parts.append(self._build_script_xml(scr))
            scripts_list = '\n'.join(scripts_parts) if scripts_parts else '<!-- No scripts -->'

            # Format response
            return LOAD_SKILL_TEMPLATE.format(
                skill_name=skill.name,
                description=skill.description,
                uri=skill.uri or 'N/A',
                resources_list=resources_list,
                scripts_list=scripts_list,
                content=skill.content,
            )

    def _register_read_skill_resource(self) -> None:
        """Register the read_skill_resource tool."""

        @self.tool
        async def read_skill_resource(  # pyright: ignore[reportUnusedFunction]  # noqa: D417
            ctx: RunContext[Any],
            skill_name: str,
            resource_name: str,
            args: dict[str, Any] | None = None,
        ) -> str:
            """Access supplementary documentation, templates, or data from a skill.

            Resources are additional files that support skill execution. They can be static
            content (markdown docs, templates, schemas) or dynamic callables (functions that
            generate content based on parameters).

            When to use this:
            - When a skill's instructions reference a specific resource
            - To access form templates, reference documentation, or data schemas
            - When you need supplementary information beyond the skill instructions

            Args:
                skill_name: Name of the skill containing the resource.
                resource_name: Exact name of the resource as listed in the skill.
                    Examples: "FORMS.md", "REFERENCE.md", "get_schema"
                    Must match exactly - do not infer or guess.
                args: Arguments for callable resources (optional for static files).
                    Keys must match the parameter names in the resource's schema.

            Returns:
                The resource content as a string.

            Important:
            - Get resource names from the skill's documentation first
            - Use exact resource names - do not modify or guess
            - Check if the resource requires arguments (check its schema)
            - Static files don't need args; callables may require them
            """
            if skill_name not in self._skills:
                raise SkillNotFoundError(f"Skill '{skill_name}' not found.")

            skill = self._skills[skill_name]

            # Find the resource
            resource = self._find_skill_resource(skill, resource_name)

            if resource is None:
                available = [r.name for r in skill.resources] if skill.resources else []
                raise SkillResourceNotFoundError(
                    f"Resource '{resource_name}' not found in skill '{skill_name}'. Available: {available}"
                )

            # Use resource.load() interface - implementation handles the details
            return await resource.load(ctx=ctx, args=args)

    def _register_run_skill_script(self) -> None:
        """Register the run_skill_script tool."""

        @self.tool
        async def run_skill_script(  # pyright: ignore[reportUnusedFunction]  # noqa: D417
            ctx: RunContext[Any],
            skill_name: str,
            script_name: str,
            args: dict[str, Any] | None = None,
        ) -> str:
            """Execute a skill script that performs actions or computations.

            Scripts are executable programs provided by skills that can perform actions
            (API calls, file operations), process data (transformations, analysis), or
            generate outputs (reports, visualizations).

            When to use this:
            - When a skill's instructions tell you to run a specific script
            - To perform automated tasks that the skill provides
            - For data processing, API interactions, or file operations

            Args:
                skill_name: Name of the skill containing the script.
                script_name: Exact name of the script as listed in the skill.
                    Usually includes .py extension: "analyze.py", "process.py"
                    Must match exactly - do not infer or guess.
                args: Arguments required by the script.
                    Keys must match the parameter names in the script's schema.

            Returns:
                Script execution output including stdout and stderr.

            Important:
            - Get script names from the skill's documentation first
            - Use exact script names - do not modify or guess
            - Check the script's parameter schema for required arguments
            - Review skill instructions before running scripts
            - Scripts may modify external state (files, databases, APIs)
            - Execution errors are included in the output
            """
            if skill_name not in self._skills:
                raise SkillNotFoundError(f"Skill '{skill_name}' not found.")

            skill = self._skills[skill_name]

            # Find the script
            script = self._find_skill_script(skill, script_name)

            if script is None:
                available = [s.name for s in skill.scripts] if skill.scripts else []
                raise SkillResourceNotFoundError(
                    f"Script '{script_name}' not found in skill '{skill_name}'. Available: {available}"
                )

            # Use script.run() interface - implementation handles the details
            return await script.run(ctx=ctx, args=args)

    async def get_instructions(self, ctx: RunContext[Any]) -> str | None:
        """Return instructions to inject into the agent's system prompt.

        Returns the skills system prompt containing usage guidance and all skill metadata.

        Args:
            ctx: The run context for this agent run.

        Returns:
            The skills system prompt, or None if no skills are loaded.
        """
        if not self._skills:
            return None

        # Build skills list in XML format
        skills_list_lines: list[str] = []
        for skill in sorted(self._skills.values(), key=lambda s: s.name):
            skills_list_lines.append('<skill>')
            skills_list_lines.append(f'<name>{skill.name}</name>')
            skills_list_lines.append(f'<description>{skill.description}</description>')
            if skill.uri:
                skills_list_lines.append(f'<uri>{skill.uri}</uri>')
            skills_list_lines.append('</skill>')
        skills_list = '\n'.join(skills_list_lines)

        # Use custom template if provided, otherwise use default
        if self._instruction_template:
            return self._instruction_template.format(skills_list=skills_list)

        return _INSTRUCTION_SKILLS_HEADER.format(skills_list=skills_list)

    def skill(
        self,
        func: Callable[[], str] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        license: str | None = None,
        compatibility: str | None = None,
        metadata: dict[str, Any] | None = None,
        resources: list[SkillResource] | None = None,
        scripts: list[SkillScript] | None = None,
    ) -> Any:
        """Decorator to define a skill using a function.

        The decorated function should return a string containing the skill's instructions/content.
        The skill name is derived from the function name (underscores replaced with hyphens)
        unless explicitly provided via the `name` parameter.

        Example:
            ```python
            from pydantic_ai import RunContext
            from pydantic_ai.toolsets.skills import SkillsToolset

            skills = SkillsToolset()

            @skills.skill(resources=[], metadata={'version': '1.0'})
            def data_analyzer() -> str:
                '''Analyze data from various sources.'''
                return '''
                Use this skill for data analysis tasks.
                The skill provides tools for querying and analyzing data.
                '''

            @data_analyzer.resource
            async def get_schema(ctx: RunContext[MyDeps]) -> str:
                return await ctx.deps.database.get_schema()

            @data_analyzer.script
            async def run_analysis(ctx: RunContext[MyDeps], query: str) -> str:
                result = await ctx.deps.database.execute(query)
                return str(result)
            ```

        Args:
            func: The function that returns skill content (must return str).
            name: Skill name (defaults to normalized function name: underscores → hyphens).
            description: Skill description (inferred from docstring if not provided).
            license: Optional license information (e.g., "Apache-2.0").
            compatibility: Optional environment requirements (e.g., "Requires Python 3.10+").
            metadata: Additional metadata fields as a dictionary.
            resources: Initial list of resources to attach to the skill.
            scripts: Initial list of scripts to attach to the skill.

        Returns:
            A SkillWrapper instance that can be used to attach resources and scripts.
        """

        def decorator(f: Callable[[], str]) -> SkillWrapper[Any]:
            # Derive name from function name if not provided
            if name is not None:
                # Explicit name provided - validate it directly without normalization
                skill_name = name
                # Validate the explicit name
                if not SKILL_NAME_PATTERN.match(skill_name):
                    raise SkillValidationError(
                        f"Skill name '{skill_name}' is invalid. "
                        'Skill names must contain only lowercase letters, numbers, and hyphens '
                        '(no consecutive hyphens).'
                    )
                if len(skill_name) > 64:
                    raise SkillValidationError(
                        f"Skill name '{skill_name}' exceeds 64 characters ({len(skill_name)} chars)."
                    )
            else:
                # Derive and normalize from function name
                skill_name = normalize_skill_name(f.__name__)

            # Extract description from docstring if not provided
            skill_description = description
            if skill_description is None:
                sig = get_signature(f)
                desc, _ = doc_descriptions(f, sig, docstring_format='auto')
                skill_description = desc

            # Create the skill wrapper
            wrapper: SkillWrapper[Any] = SkillWrapper(
                function=f,
                name=skill_name,
                description=skill_description,
                license=license,
                compatibility=compatibility,
                metadata=metadata,
                resources=list(resources) if resources else [],
                scripts=list(scripts) if scripts else [],
            )

            # Register the skill immediately (execute function to get content)
            self._register_skill(wrapper)

            # Return the wrapper so resources/scripts can be attached
            return wrapper

        if func is None:
            # Called with arguments: @skills.skill(name="custom")
            return decorator
        else:
            # Called without arguments: @skills.skill
            return decorator(func)

    def _register_skill(self, skill: Skill | SkillWrapper[Any]) -> None:
        """Register a skill with the toolset.

        Converts SkillWrapper instances to Skill dataclasses before registering.
        Warns about duplicate skill names (last occurrence wins).

        Args:
            skill: Skill or SkillWrapper instance to register.
        """
        # Convert SkillWrapper to Skill if needed
        if isinstance(skill, SkillWrapper):
            skill = skill.to_skill()

        # Warn about duplicates
        if skill.name in self._skills:
            warnings.warn(
                f"Duplicate skill '{skill.name}' found. Overriding previous occurrence.",
                UserWarning,
                stacklevel=3,
            )

        # Register the skill
        self._skills[skill.name] = skill
