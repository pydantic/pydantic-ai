"""Type definitions for skills toolset.

This module contains dataclass-based type definitions for skills,
their metadata, resources, and scripts.

Data classes:
- [`Skill`][pydantic_ai.toolsets.skills.Skill]: A loaded skill instance with metadata, content, resources, and scripts
- [`SkillMetadata`][pydantic_ai.toolsets.skills.SkillMetadata]: Skill metadata from SKILL.md frontmatter
- [`SkillResource`][pydantic_ai.toolsets.skills.SkillResource]: A resource file within a skill
- [`SkillScript`][pydantic_ai.toolsets.skills.SkillScript]: An executable script within a skill
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..._run_context import AgentDepsT, RunContext


class SkillScriptExecutor(Protocol):
    """Protocol for executing skill scripts.

    Implementations provide different execution environments:
    - Local execution with Python subprocess
    - Remote execution via API
    - Sandboxed execution (Docker, WASM, etc.)
    """

    async def run(
        self,
        skill: Skill,
        script: SkillScript,
        args: list[str] | None = None,
    ) -> Any:
        """Run a skill script and return its output.

        Args:
            skill: The skill containing the script.
            script: The script to run.
            args: Optional command-line arguments.

        Returns:
            Combined stdout and stderr output.

        Raises:
            SkillScriptExecutionError: If execution fails.
        """
        ...


@dataclass
class SkillMetadata:
    """Skill metadata from SKILL.md frontmatter.

    Only `name` and `description` are required. Other fields
    (version, author, category, tags, etc.) can be added dynamically
    based on frontmatter content.

    Attributes:
        name: The skill identifier.
        description: Brief description of what the skill does.
        extra: Additional metadata fields from frontmatter.
    """

    name: str
    description: str
    extra: dict[str, Any] | None = None


@dataclass
class SkillResource:
    """A skill resource within a skill (e.g., FORMS.md, REFERENCE.md).

    Resources can be either filesystem paths or URLs. The URI field stores
    the resource location as a string.

    Attributes:
        name: Resource filename (e.g., "FORMS.md").
        uri: URI string - either a filesystem path or URL (http://, https://).
        content: Loaded content (lazy-loaded, None until read).
    """

    name: str
    uri: str
    content: str | None = None


@dataclass
class SkillScript:
    """An executable script within a skill.

    Script-based tools: Executable Python scripts in scripts/ directory
    or directly in the skill directory.
    Can be executed via SkillsToolset.run_skill_script() tool.

    Attributes:
        name: Script name without .py extension.
        uri: URI string - either a filesystem path or URL (http://, https://).
        skill_name: Parent skill name.
    """

    name: str
    uri: str
    skill_name: str


@dataclass
class Skill(ABC):
    """Abstract base class for skill instances.

    Skills can be discovered from filesystem directories or created programmatically.
    The uri field stores the skill's base location (directory path or URL) as a string.

    Attributes:
        name: Skill name (from metadata).
        uri: URI string for skill's base location. Can be a filesystem path or URL.
        metadata: Parsed metadata from SKILL.md.
        content: Main content from SKILL.md (without frontmatter).
        resources: Resource files (FORMS.md, etc.). None if no resources.
        scripts: Available scripts in the skill directory or scripts/ subdirectory.
            None if no scripts.
        script_executor: Executor for running scripts. Defaults to LocalSkillScriptExecutor
            if not provided. Can be None for skills without scripts.
    """

    name: str
    uri: str
    metadata: SkillMetadata
    content: str
    resources: list[SkillResource] | None = None
    scripts: list[SkillScript] | None = None
    script_executor: SkillScriptExecutor | None = field(default=None, repr=False)

    @property
    def description(self) -> str:
        """Get skill description from metadata."""
        return self.metadata.description

    @abstractmethod
    async def read_resource(self, ctx: RunContext[AgentDepsT], resource_uri: str) -> str:
        """Read a resource file from this skill.

        Args:
            ctx: The run context for this agent run.
            resource_uri: URI or name of the resource file.

        Returns:
            Resource file content.

        Raises:
            SkillResourceNotFoundError: If resource is not found.
            NotImplementedError: For non-filesystem resources (URLs, etc.).
        """
        ...

    @abstractmethod
    async def run_script(
        self,
        ctx: RunContext[AgentDepsT],
        script_uri: str,
        args: list[str] | None = None,
    ) -> str:
        """Execute a script from this skill.

        Args:
            ctx: The run context for this agent run.
            script_uri: URI or name of the script (without .py extension).
            args: Optional command-line arguments (e.g., "-param value").

        Returns:
            Script output (stdout and stderr combined).

        Raises:
            SkillResourceNotFoundError: If script is not found.
            SkillResourceLoadError: If script path is unsafe.
            SkillScriptExecutionError: If script execution fails.
        """
        ...
