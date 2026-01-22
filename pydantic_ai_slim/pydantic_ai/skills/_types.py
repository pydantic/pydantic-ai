"""Type definitions for skills toolset.

This module contains dataclass-based type definitions for skills,
their resources, and scripts.

Data classes:
- [`Skill`][pydantic_ai.toolsets.skills.Skill]: A skill instance with metadata, content, resources, and scripts
- [`SkillResource`][pydantic_ai.toolsets.skills.SkillResource]: A resource file or callable within a skill
- [`SkillScript`][pydantic_ai.toolsets.skills.SkillScript]: An executable script within a skill
- [`SkillWrapper`][pydantic_ai.toolsets.skills.SkillWrapper]: Generic wrapper for decorator-based skill creation
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from pydantic.json_schema import GenerateJsonSchema

from .. import _function_schema
from ..tools import GenerateToolJsonSchema
from ._exceptions import SkillValidationError

# Skill name pattern: lowercase letters, numbers, and hyphens (no consecutive hyphens)
_SKILL_NAME_PATTERN = re.compile(r'^[a-z0-9]+(-[a-z0-9]+)*$')

# Generic type variable for dependencies
DepsT = TypeVar('DepsT')


def normalize_skill_name(func_name: str) -> str:
    """Normalize a function name to a valid skill name.

    Converts underscores to hyphens and validates against the skill naming pattern.

    Args:
        func_name: The function name to normalize.

    Returns:
        Normalized skill name (lowercase, underscores replaced with hyphens).

    Raises:
        SkillValidationError: If the name contains invalid characters after normalization.

    Example:
        ```python
        normalize_skill_name('data_analyzer')  # Returns 'data-analyzer'
        normalize_skill_name('my_cool_skill')  # Returns 'my-cool-skill'
        normalize_skill_name('InvalidName')  # Raises SkillValidationError
        ```
    """
    # Replace underscores with hyphens and convert to lowercase
    normalized = func_name.replace('_', '-').lower()

    # Validate against pattern
    if not _SKILL_NAME_PATTERN.match(normalized):
        raise SkillValidationError(
            f"Skill name '{normalized}' (derived from function '{func_name}') is invalid. "
            'Skill names must contain only lowercase letters, numbers, and hyphens '
            '(no consecutive hyphens).'
        )

    # Check length
    if len(normalized) > 64:
        raise SkillValidationError(f"Skill name '{normalized}' exceeds 64 characters ({len(normalized)} chars).")

    return normalized


@dataclass
class SkillResource:
    """A skill resource within a skill (e.g., FORMS.md, REFERENCE.md, or callable function).

    Resources can be either:
    - Static content: Loaded from files or provided directly as strings
    - Dynamic callables: Functions that generate content on-demand, optionally taking RunContext

    Attributes:
        name: Resource name (e.g., "FORMS.md" or "get_samples").
        description: Description of what the resource provides.
        content: Static content string (for file-based or direct content resources).
        function: Callable that generates content dynamically (for programmatic resources).
        takes_ctx: Whether the function takes a RunContext first argument (auto-detected if None).
        function_schema: Function schema for callable resources (auto-generated).
        uri: Optional URI string for file-based resources (internal use only).
    """

    name: str
    description: str | None = None
    content: str | None = None
    function: Callable[..., Any | Awaitable[Any]] | None = None
    takes_ctx: bool = False
    function_schema: _function_schema.FunctionSchema | None = None
    uri: str | None = None

    def __post_init__(self) -> None:
        """Validate that resource has either content, function, or uri.

        For programmatic resources, content or function is required.
        For file-based resources (subclasses), uri is sufficient.
        """
        if self.content is None and self.function is None and self.uri is None:
            raise ValueError(f"Resource '{self.name}' must have either content, function, or uri")
        if self.function is not None and self.function_schema is None:
            raise ValueError(f"Resource '{self.name}' with function must have function_schema")

    async def load(self, ctx: Any, args: dict[str, Any] | None = None) -> Any:
        """Load the resource content.

        Default implementation for programmatic resources with function or static content.
        File-based resource subclasses override this method to load from disk.

        Args:
            ctx: RunContext for accessing dependencies.
            args: Named arguments for callable resources.

        Returns:
            The resource content (can be any type like str, dict, etc.).

        Raises:
            ValueError: If resource has no content or function.
        """
        # Handle callable resource (programmatic)
        if self.function and self.function_schema:
            return await self.function_schema.call(args or {}, ctx)

        # Handle static content
        elif self.content:
            return self.content

        else:
            raise ValueError(f"Resource '{self.name}' has no content or function")


@dataclass
class SkillScript:
    """An executable script within a skill.

    Scripts can be either:
    - Programmatic functions: Python functions with RunContext support
    - File-based scripts: Python scripts executed via subprocess

    Attributes:
        name: Script name (includes .py extension for file-based scripts).
        description: Description of what the script does.
        function: Callable that implements the script logic (for programmatic scripts).
        takes_ctx: Whether the function takes a RunContext first argument (auto-detected).
        function_schema: Function schema for callable scripts (auto-generated).
        uri: Optional URI string for file-based scripts (internal use only).
        skill_name: Optional parent skill name (internal use only).
    """

    name: str
    description: str | None = None
    function: Callable[..., Any] | None = None
    takes_ctx: bool = False
    function_schema: _function_schema.FunctionSchema | None = None
    uri: str | None = None
    skill_name: str | None = None

    def __post_init__(self) -> None:
        """Validate that script has either function or uri.

        For programmatic scripts, function is required.
        For file-based scripts (subclasses), uri is sufficient.
        """
        if self.function is None and self.uri is None:
            raise ValueError(f"Script '{self.name}' must have either function or uri")
        if self.function is not None and self.function_schema is None:
            raise ValueError(f"Script '{self.name}' with function must have function_schema")

    async def run(self, ctx: Any, args: dict[str, Any] | None = None) -> Any:
        """Execute the script.

        Default implementation for programmatic scripts with function.
        File-based script subclasses override this method to execute via subprocess.

        Args:
            ctx: RunContext for accessing dependencies.
            args: Named arguments for the script.

        Returns:
            Script output (can be any type like str, dict, etc.).

        Raises:
            ValueError: If script has no function.
        """
        # Handle callable script (programmatic)
        if self.function and self.function_schema:
            return await self.function_schema.call(args or {}, ctx)

        else:
            raise ValueError(f"Script '{self.name}' has no function")


@dataclass
class Skill:
    """A skill instance with metadata, content, resources, and scripts.

    Skills can be created programmatically or loaded from filesystem directories.

    Example - Programmatic skill with decorators:
        ```python
        from pydantic_ai import RunContext
        from pydantic_ai.toolsets.skills import Skill, SkillResource

        # Create a skill (uri is optional and only for file-based skills)
        my_skill = Skill(
            name='hr-analytics-skill',
            description='Skill for HR analytics',
            content='Use this skill for HR data analysis...',
            resources=[
                SkillResource(name='table-schemas', content='Schema definitions...')
            ]
        )

        # Add callable resources
        @my_skill.resource
        def get_db_context() -> str:
            return "Dynamic database context."

        @my_skill.resource
        async def get_samples(ctx: RunContext[MyDeps]) -> str:
            return await ctx.deps.get_samples()

        # Add callable scripts
        @my_skill.script
        async def load_dataset(ctx: RunContext[MyDeps]) -> str:
            await ctx.deps.load_data()
            return 'Dataset loaded.'

        @my_skill.script
        async def run_query(ctx: RunContext[MyDeps], query: str) -> str:
            result = await ctx.deps.db.execute(query)
            return str(result)
        ```

    Attributes:
        name: Skill name.
        description: Brief description of what the skill does.
        content: Main instructional content for the skill.
        license: Optional license information (e.g., "Apache-2.0" or reference to LICENSE.txt).
        compatibility: Optional environment requirements (e.g., "Requires git, docker, and internet access"). Max 500 chars.
        resources: List of resources (files or callables). Defaults to empty list.
        scripts: List of scripts (functions or file-based). Defaults to empty list.
        uri: Optional URI string for skill's base location (internal use only, for file-based skills).
        metadata: Additional metadata fields.
    """

    name: str
    description: str
    content: str
    license: str | None = None
    compatibility: str | None = None
    resources: list[SkillResource] = field(default_factory=list)
    scripts: list[SkillScript] = field(default_factory=list)
    uri: str | None = None
    metadata: dict[str, Any] | None = None

    def resource(
        self,
        func: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        takes_ctx: bool | None = None,
        docstring_format: _function_schema.DocstringFormat = 'auto',
        schema_generator: type[GenerateJsonSchema] | None = None,
    ) -> Callable[..., Any] | Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator to register a callable as a skill resource.

        The decorated function can optionally take RunContext as its first argument
        for accessing dependencies. This is auto-detected if not specified.

        Example:
            ```python
            @my_skill.resource
            def get_context() -> str:
                return "Static context"

            @my_skill.resource
            async def get_data(ctx: RunContext[MyDeps]) -> str:
                return await ctx.deps.fetch_data()
            ```

        Args:
            func: The function to register as a resource.
            name: Resource name (defaults to function name).
            description: Resource description (inferred from docstring if not provided).
            takes_ctx: Whether function takes RunContext (auto-detected if None).
            docstring_format: Format of the docstring ('auto', 'google', 'numpy', 'sphinx').
            schema_generator: Custom JSON schema generator class.

        Returns:
            The original function (allows use as decorator).
        """

        def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
            resource_name = name or f.__name__
            gen = schema_generator or GenerateToolJsonSchema

            # Generate function schema
            func_schema = _function_schema.function_schema(
                f,
                schema_generator=gen,
                takes_ctx=takes_ctx,
                docstring_format=docstring_format,
                require_parameter_descriptions=False,
            )

            # Create resource and add to skill
            resource = SkillResource(
                name=resource_name,
                description=description or func_schema.description,
                function=f,
                takes_ctx=func_schema.takes_ctx,
                function_schema=func_schema,
            )
            self.resources.append(resource)
            return f

        if func is None:
            # Called with arguments: @my_skill.resource(name="custom")
            return decorator
        else:
            # Called without arguments: @my_skill.resource
            return decorator(func)

    def script(
        self,
        func: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        takes_ctx: bool | None = None,
        docstring_format: _function_schema.DocstringFormat = 'auto',
        schema_generator: type[GenerateJsonSchema] | None = None,
    ) -> Callable[..., Any]:
        """Decorator to register a callable as a skill script.

        The decorated function can optionally take RunContext as its first argument
        for accessing dependencies. This is auto-detected if not specified.

        Scripts accept named arguments (kwargs) matching their function signature.

        Example:
            ```python
            @my_skill.script
            async def load_data(ctx: RunContext[MyDeps]) -> str:
                await ctx.deps.load()
                return 'Loaded'

            @my_skill.script
            async def run_query(ctx: RunContext[MyDeps], query: str, limit: int = 10) -> str:
                result = await ctx.deps.db.execute(query, limit)
                return str(result)
            ```

        Args:
            func: The function to register as a script.
            name: Script name (defaults to function name).
            description: Script description (inferred from docstring if not provided).
            takes_ctx: Whether function takes RunContext (auto-detected if None).
            docstring_format: Format of the docstring ('auto', 'google', 'numpy', 'sphinx').
            schema_generator: Custom JSON schema generator class.

        Returns:
            The original function (allows use as decorator).
        """

        def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
            script_name = name or f.__name__
            gen = schema_generator or GenerateToolJsonSchema

            # Generate function schema
            func_schema = _function_schema.function_schema(
                f,
                schema_generator=gen,
                takes_ctx=takes_ctx,
                docstring_format=docstring_format,
                require_parameter_descriptions=False,
            )

            # Create script and add to skill
            script = SkillScript(
                name=script_name,
                description=description or func_schema.description,
                function=f,
                takes_ctx=func_schema.takes_ctx,
                function_schema=func_schema,
                skill_name=self.name,
            )
            self.scripts.append(script)
            return f

        if func is None:
            # Called with arguments: @my_skill.script(name="custom")
            return decorator
        else:
            # Called without arguments: @my_skill.script
            return decorator(func)


class SkillWrapper(Generic[DepsT]):
    """Generic wrapper for decorator-based skill creation with type-safe dependencies.

    This class wraps a function that returns skill content and provides decorators
    for attaching resources and scripts. The wrapper is generic over the dependency
    type, enabling type-safe RunContext usage in attached resources and scripts.

    This is typically created via the `@skills.skill` decorator on a SkillsToolset instance.

    Example:
        ```python
        from dataclasses import dataclass
        from pydantic_ai import RunContext
        from pydantic_ai.toolsets.skills import SkillsToolset

        @dataclass
        class MyDeps:
            database: DatabaseConn

        skills = SkillsToolset[MyDeps]()

        @skills.skill(resources=[], metadata={'version': '1.0'})
        def data_analyzer(ctx: RunContext[MyDeps]) -> str:
            '''Analyze data from the database.'''
            return 'Use this skill for data analysis...'

        @data_analyzer.resource
        async def get_schema(ctx: RunContext[MyDeps]) -> str:
            return await ctx.deps.database.get_schema()

        @data_analyzer.script
        async def run_analysis(ctx: RunContext[MyDeps], query: str) -> str:
            result = await ctx.deps.database.execute(query)
            return str(result)
        ```

    Attributes:
        function: The function that returns skill content.
        name: Skill name (normalized from function name if not provided).
        description: Brief description (from docstring if not provided).
        license: Optional license information.
        compatibility: Optional environment requirements.
        metadata: Additional metadata fields.
        resources: List of resources to attach to the skill.
        scripts: List of scripts to attach to the skill.
    """

    def __init__(
        self,
        function: Callable[[Any], str],
        name: str,
        description: str | None,
        license: str | None,
        compatibility: str | None,
        metadata: dict[str, Any] | None,
        resources: list[SkillResource],
        scripts: list[SkillScript],
    ) -> None:
        """Initialize the skill wrapper.

        Args:
            function: The function that returns skill content.
            name: Skill name (already normalized).
            description: Skill description.
            license: Optional license information.
            compatibility: Optional environment requirements.
            metadata: Additional metadata fields.
            resources: Initial list of resources.
            scripts: Initial list of scripts.
        """
        self.function = function
        self.name = name
        self.description = description
        self.license = license
        self.compatibility = compatibility
        self.metadata = metadata
        self.resources = list(resources)  # Make a copy to avoid mutations
        self.scripts = list(scripts)  # Make a copy to avoid mutations

    def resource(
        self,
        func: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        takes_ctx: bool | None = None,
        docstring_format: _function_schema.DocstringFormat = 'auto',
        schema_generator: type[GenerateJsonSchema] | None = None,
    ) -> Callable[..., Any] | Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator to attach a callable resource to the skill.

        The decorated function can optionally take RunContext as its first argument
        for accessing dependencies. This is auto-detected if not specified.

        Example:
            ```python
            @my_skill.resource
            def get_context() -> str:
                return "Static context"

            @my_skill.resource
            async def get_data(ctx: RunContext[MyDeps]) -> str:
                return await ctx.deps.fetch_data()

            @my_skill.resource(name="custom_name", description="Custom description")
            async def my_resource(ctx: RunContext[MyDeps], param: str) -> dict:
                return {"result": param}
            ```

        Args:
            func: The function to register as a resource.
            name: Resource name (defaults to function name).
            description: Resource description (inferred from docstring if not provided).
            takes_ctx: Whether function takes RunContext (auto-detected if None).
            docstring_format: Format of the docstring ('auto', 'google', 'numpy', 'sphinx').
            schema_generator: Custom JSON schema generator class.

        Returns:
            The original function (allows use as decorator).
        """

        def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
            resource_name = name or f.__name__
            gen = schema_generator or GenerateToolJsonSchema

            # Generate function schema
            func_schema = _function_schema.function_schema(
                f,
                schema_generator=gen,
                takes_ctx=takes_ctx,
                docstring_format=docstring_format,
                require_parameter_descriptions=False,
            )

            # Create resource and add to skill wrapper
            resource = SkillResource(
                name=resource_name,
                description=description or func_schema.description,
                function=f,
                takes_ctx=func_schema.takes_ctx,
                function_schema=func_schema,
            )
            self.resources.append(resource)
            return f

        if func is None:
            # Called with arguments: @my_skill.resource(name="custom")
            return decorator
        else:
            # Called without arguments: @my_skill.resource
            return decorator(func)

    def script(
        self,
        func: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        takes_ctx: bool | None = None,
        docstring_format: _function_schema.DocstringFormat = 'auto',
        schema_generator: type[GenerateJsonSchema] | None = None,
    ) -> Callable[..., Any] | Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator to attach a callable script to the skill.

        The decorated function can optionally take RunContext as its first argument
        for accessing dependencies. This is auto-detected if not specified.

        Scripts accept named arguments (kwargs) matching their function signature.

        Example:
            ```python
            @my_skill.script
            async def load_data(ctx: RunContext[MyDeps]) -> str:
                await ctx.deps.load()
                return 'Loaded'

            @my_skill.script
            async def run_query(ctx: RunContext[MyDeps], query: str, limit: int = 10) -> str:
                result = await ctx.deps.db.execute(query, limit)
                return str(result)
            ```

        Args:
            func: The function to register as a script.
            name: Script name (defaults to function name).
            description: Script description (inferred from docstring if not provided).
            takes_ctx: Whether function takes RunContext (auto-detected if None).
            docstring_format: Format of the docstring ('auto', 'google', 'numpy', 'sphinx').
            schema_generator: Custom JSON schema generator class.

        Returns:
            The original function (allows use as decorator).
        """

        def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
            script_name = name or f.__name__
            gen = schema_generator or GenerateToolJsonSchema

            # Generate function schema
            func_schema = _function_schema.function_schema(
                f,
                schema_generator=gen,
                takes_ctx=takes_ctx,
                docstring_format=docstring_format,
                require_parameter_descriptions=False,
            )

            # Create script and add to skill wrapper
            script = SkillScript(
                name=script_name,
                description=description or func_schema.description,
                function=f,
                takes_ctx=func_schema.takes_ctx,
                function_schema=func_schema,
                skill_name=self.name,
            )
            self.scripts.append(script)
            return f

        if func is None:
            # Called with arguments: @my_skill.script(name="custom")
            return decorator
        else:
            # Called without arguments: @my_skill.script
            return decorator(func)

    def to_skill(self) -> Skill:
        """Convert the wrapper to a Skill dataclass.

        Calls the wrapped function to generate the skill content and constructs
        a Skill instance with all attached resources and scripts.

        Returns:
            A Skill dataclass instance with all metadata and attached resources/scripts.
        """
        # Call the function to get content (must return str)
        content = self.function()

        # Create and return the Skill instance
        return Skill(
            name=self.name,
            description=self.description or '',
            content=content,
            license=self.license,
            compatibility=self.compatibility,
            resources=self.resources,
            scripts=self.scripts,
            uri=None,  # No URI for programmatic skills
            metadata=self.metadata,
        )
