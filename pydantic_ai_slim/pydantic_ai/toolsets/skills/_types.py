"""Type definitions for skills toolset.

This module contains dataclass-based type definitions for skills,
their metadata, resources, and scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillResource:
    """A resource file within a skill (e.g., FORMS.md, REFERENCE.md).

    Attributes:
        name: Resource filename (e.g., "FORMS.md").
        path: Absolute path to the resource file.
        content: Loaded content (lazy-loaded, None until read).
    """

    name: str
    path: Path
    content: str | None = None


@dataclass
class SkillScript:
    """An executable script within a skill.

    Script-based tools: Executable Python scripts in scripts/ directory
    or directly in the skill directory.
    Can be executed via SkillsToolset.run_skill_script() tool.

    Attributes:
        name: Script name without .py extension.
        path: Absolute path to the script file.
        skill_name: Parent skill name.
    """

    name: str
    path: Path
    skill_name: str


@dataclass
class Skill:
    """A loaded skill instance.

    Attributes:
        name: Skill name (from metadata).
        path: Absolute path to skill directory.
        metadata: Parsed metadata from SKILL.md.
        content: Main content from SKILL.md (without frontmatter).
        resources: Optional resource files (FORMS.md, etc.).
        scripts: Available scripts in the skill directory or scripts/ subdirectory.
    """

    name: str
    path: Path
    metadata: SkillMetadata
    content: str
    resources: list[SkillResource] = field(default_factory=list)
    scripts: list[SkillScript] = field(default_factory=list)

    @property
    def description(self) -> str:
        """Get skill description from metadata."""
        return self.metadata.description
