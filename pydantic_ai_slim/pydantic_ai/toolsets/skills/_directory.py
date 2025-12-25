"""Filesystem-based skill discovery and management.

This module provides [`SkillsDirectory`][pydantic_ai.toolsets.skills.SkillsDirectory]
for discovering and loading skills from a filesystem directory.

Supports nested skill directories (e.g., ./skills/data/analysis, ./skills/dev/engineer)
and provides methods for loading skill instructions, reading resource files,
and executing scripts.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from ._exceptions import (
    SkillNotFoundError,
    SkillValidationError,
)
from ._local import CallableSkillScriptExecutor, LocalSkill, LocalSkillScriptExecutor
from ._types import Skill, SkillMetadata, SkillResource, SkillScript, SkillScriptExecutor

__all__ = ['SkillsDirectory']

# Anthropic's naming convention: lowercase letters, numbers, and hyphens only
SKILL_NAME_PATTERN = re.compile(r'^[a-z0-9-]+$')
RESERVED_WORDS = {'anthropic', 'claude'}


def _validate_skill_metadata(
    frontmatter: dict[str, Any],
    instructions: str,
) -> bool:
    """Validate skill metadata against Anthropic's requirements.

    Emits warnings for any validation issues found.

    Args:
        frontmatter: Parsed YAML frontmatter.
        instructions: The skill instructions content.

    Returns:
        True if validation passed with no issues, False if warnings were emitted.
    """
    is_valid = True
    name = frontmatter.get('name', '')
    description = frontmatter.get('description', '')

    # Validate name format
    if name:
        # Check length first to avoid processing excessively long names (good practice)
        if len(name) > 64:
            warnings.warn(
                f"Skill name '{name}' exceeds 64 characters ({len(name)} chars) recommendation. Consider shortening it.",
                UserWarning,
                stacklevel=2,
            )
            is_valid = False
        elif not SKILL_NAME_PATTERN.match(name):
            warnings.warn(
                f"Skill name '{name}' should contain only lowercase letters, numbers, and hyphens",
                UserWarning,
                stacklevel=2,
            )
            is_valid = False
        # Check for reserved words
        for reserved in RESERVED_WORDS:
            if reserved in name:
                warnings.warn(f"Skill name '{name}' contains reserved word '{reserved}'", UserWarning, stacklevel=2)
                is_valid = False

    # Validate description
    if description and len(description) > 1024:
        warnings.warn(
            f'Skill description exceeds 1024 characters ({len(description)} chars)', UserWarning, stacklevel=2
        )
        is_valid = False

    # Validate instructions length (Anthropic recommends under 500 lines)
    lines = instructions.split('\n')
    if len(lines) > 500:
        warnings.warn(
            f'SKILL.md body exceeds recommended 500 lines ({len(lines)} lines). '
            f'Consider splitting into separate resource files.',
            UserWarning,
            stacklevel=2,
        )
        is_valid = False

    return is_valid


def _parse_skill_md(content: str) -> tuple[dict[str, Any], str]:
    """Parse a SKILL.md file into frontmatter and instructions.

    Uses PyYAML for robust YAML parsing.

    Args:
        content: Full content of the SKILL.md file.

    Returns:
        Tuple of (frontmatter_dict, instructions_markdown).

    Raises:
        SkillValidationError: If YAML parsing fails.
    """
    # Match YAML frontmatter between --- delimiters
    frontmatter_pattern = r'^---\s*\n(.*?)^---\s*\n'
    match = re.search(frontmatter_pattern, content, re.DOTALL | re.MULTILINE)

    if not match:
        # No frontmatter, treat entire content as instructions
        return {}, content.strip()

    frontmatter_yaml = match.group(1).strip()
    instructions = content[match.end() :].strip()

    # Handle empty frontmatter
    if not frontmatter_yaml:
        return {}, instructions

    try:
        frontmatter = yaml.safe_load(frontmatter_yaml)
        return frontmatter, instructions
    except yaml.YAMLError as e:
        raise SkillValidationError(f'Failed to parse YAML frontmatter: {e}') from e


def _discover_resources(skill_folder: Path) -> list[SkillResource]:
    """Discover resource files in a skill folder.

    Resources are markdown files other than SKILL.md, plus any files
    in a resources/ subdirectory.

    Args:
        skill_folder: Path to the skill directory.

    Returns:
        List of discovered SkillResource objects.
    """
    resources: list[SkillResource] = []

    # Find .md files other than SKILL.md (FORMS.md, REFERENCE.md, etc.)
    for md_file in skill_folder.glob('*.md'):
        if md_file.name.upper() != 'SKILL.MD':
            resources.append(
                SkillResource(
                    name=md_file.name,
                    uri=str(md_file.resolve()),
                )
            )

    # Find files in resources/ subdirectory if it exists
    resources_dir = skill_folder / 'resources'
    if resources_dir.exists() and resources_dir.is_dir():
        for resource_file in resources_dir.rglob('*'):
            if resource_file.is_file():
                rel_path = resource_file.relative_to(skill_folder)
                resources.append(
                    SkillResource(
                        name=str(rel_path),
                        uri=str(resource_file.resolve()),
                    )
                )

    return resources


def _find_skill_files(root_dir: Path, max_depth: int | None) -> list[Path]:
    """Find SKILL.md files with depth-limited search using optimized glob patterns.

    Args:
        root_dir: Root directory to search from.
        max_depth: Maximum depth to search. None for unlimited.

    Returns:
        List of paths to SKILL.md files.
    """
    if max_depth is None:
        # Unlimited recursive search
        return list(root_dir.glob('**/SKILL.md'))

    # Build explicit glob patterns for each depth level
    # This is much faster than iterdir() while still limiting depth
    skill_files: list[Path] = []

    for depth in range(max_depth + 1):
        if depth == 0:
            pattern = 'SKILL.md'
        else:
            pattern = '/'.join(['*'] * depth) + '/SKILL.md'

        skill_files.extend(root_dir.glob(pattern))

    return skill_files


def _discover_scripts(skill_folder: Path, skill_name: str) -> list[SkillScript]:
    """Discover executable scripts in a skill folder.

    Looks for Python scripts in:
    - Directly in the skill folder (*.py)
    - In a scripts/ subdirectory

    Args:
        skill_folder: Path to the skill directory.
        skill_name: Name of the parent skill.

    Returns:
        List of discovered SkillScript objects.
    """
    scripts: list[SkillScript] = []

    # Find .py files in skill folder root (excluding __init__.py)
    for py_file in skill_folder.glob('*.py'):
        if py_file.name != '__init__.py':
            scripts.append(
                SkillScript(
                    name=py_file.stem,  # filename without .py
                    uri=str(py_file.resolve()),
                    skill_name=skill_name,
                )
            )

    # Find .py files in scripts/ subdirectory
    scripts_dir = skill_folder / 'scripts'
    if scripts_dir.exists() and scripts_dir.is_dir():
        for py_file in scripts_dir.glob('*.py'):
            if py_file.name != '__init__.py':
                scripts.append(
                    SkillScript(
                        name=py_file.stem,
                        uri=str(py_file.resolve()),
                        skill_name=skill_name,
                    )
                )

    return scripts


def _discover_skills(
    path: str | Path,
    validate: bool = True,
    max_depth: int | None = 3,
    script_executor: SkillScriptExecutor | None = None,
) -> list[Skill]:
    """Discover skills from a filesystem directory.

    Searches for SKILL.md files in the given directory and loads
    skill metadata and structure.

    Args:
        path: Directory path to search for skills.
        validate: Whether to validate skill structure (requires name and description).
        max_depth: Maximum depth to search for SKILL.md files. None for unlimited.
            Default is 3 levels deep to prevent performance issues with large trees.
        script_executor: Script executor for running skill scripts.

    Returns:
        List of discovered Skill objects.

    Raises:
        SkillValidationError: If validation is enabled and a skill is invalid.
    """
    skills: list[Skill] = []
    dir_path = Path(path).expanduser().resolve()

    if not dir_path.exists():
        return skills

    if not dir_path.is_dir():
        return skills

    # Find all SKILL.md files (depth-limited search for performance)
    skill_files = _find_skill_files(dir_path, max_depth)
    for skill_file in skill_files:
        try:
            skill_folder = skill_file.parent
            content = skill_file.read_text(encoding='utf-8')
            frontmatter, instructions = _parse_skill_md(content)

            # Get required fields
            name = frontmatter.get('name')
            description = frontmatter.get('description', '')

            # Validation - if name is missing, handle based on validate flag
            if not name:
                if validate:
                    # Skip skill and log warning when validation is enabled
                    warnings.warn(
                        f'Skipping skill at {skill_file}: missing required "name" field.', UserWarning, stacklevel=2
                    )
                    continue
                else:
                    # Use folder name as fallback when validation is disabled
                    name = skill_folder.name

            # Extract extra metadata fields
            extra = {k: v for k, v in frontmatter.items() if k not in ('name', 'description')}

            # Create metadata
            metadata = SkillMetadata(
                name=name,
                description=description,
                extra=extra if extra else None,
            )

            # Validate metadata
            if validate:
                _validate_skill_metadata(frontmatter, instructions)

            # Discover resources and scripts
            resources = _discover_resources(skill_folder)
            scripts = _discover_scripts(skill_folder, name)

            # Create skill
            skill = LocalSkill(
                name=name,
                uri=str(skill_folder.resolve()),
                metadata=metadata,
                content=instructions,
                resources=resources if resources else None,
                scripts=scripts if scripts else None,
                script_executor=script_executor,
            )

            skills.append(skill)

        except SkillValidationError as sve:
            if validate:
                raise
            else:
                warnings.warn(f'Skipping invalid skill at {skill_file}: {sve}', UserWarning, stacklevel=2)
        except Exception as e:
            # Don't re-wrap warnings as exceptions
            if not isinstance(e, Warning):
                raise SkillValidationError(f'Failed to load skill from {skill_file}: {e}') from e

    return skills


class SkillsDirectory:
    """Skill source for a single filesystem directory or programmatic skills.

    Can operate in two modes:
    1. Discovery mode: Discovers and loads skills from a single local directory
    2. Programmatic mode: Accepts pre-built Skill objects

    Scripts are executed using the bound SkillScriptExecutor.
    """

    def __init__(
        self,
        *,
        path: str | Path,
        validate: bool = True,
        max_depth: int | None = 3,
        script_executor: SkillScriptExecutor | Callable[..., Any] | None = None,
    ) -> None:
        """Initialize the skills directory source.

        Args:
            path: Directory path to search for skills (discovery mode).
            validate: Validate skill structure on discovery.
            max_depth: Maximum depth for skill discovery (None for unlimited).
            script_executor: Script executor for running scripts. Can be:
                - A SkillScriptExecutor instance
                - A callable (function) that will be wrapped in CallableSkillScriptExecutor
                - None (uses LocalSkillScriptExecutor with default settings)

        Raises:
            ValueError: If neither or both `path` and `skills` are provided.

        Example:
            ```python
            # Discovery mode - single directory
            source = SkillsDirectory(path="./skills")

            # With custom executor
            from pydantic_ai.toolsets.skills import LocalSkillScriptExecutor

            executor = LocalSkillScriptExecutor(timeout=60)
            source = SkillsDirectory(path="./skills", script_executor=executor)

            # With callable executor
            async def my_executor(skill, script, args=None):
                # Custom execution logic
                return "result"

            source = SkillsDirectory(
                path="./skills",
                script_executor=my_executor
            )
            ```
        """
        self._path = Path(path).expanduser().resolve()
        self._validate = validate
        self._max_depth = max_depth

        # Handle script_executor parameter
        if script_executor is None:
            self._executor: SkillScriptExecutor = LocalSkillScriptExecutor()
        elif callable(script_executor):
            self._executor = CallableSkillScriptExecutor(func=script_executor)
        else:
            self._executor = script_executor

        # Discover skills from directory
        self._skills: dict[str, Skill] = self.get_skills()

    def get_skills(self) -> dict[str, Skill]:
        """Get all skills from this source.

        Returns:
            Dictionary of skill URI to Skill object.
        """
        skills = _discover_skills(
            path=self._path,
            validate=self._validate,
            max_depth=self._max_depth,
            script_executor=self._executor,
        )

        return {skill.uri: skill for skill in skills}

    @property
    def skills(self) -> dict[str, Skill]:
        """Get the dictionary of loaded skills.

        Returns:
            Dictionary mapping skill URI to Skill objects.
        """
        return self._skills

    def load_skill(self, skill_uri: str) -> Skill:
        """Load full instructions for a skill.

        Args:
            skill_uri: URI of the skill to load (skill name for filesystem skills).

        Returns:
            Loaded Skill object.

        Raises:
            SkillNotFoundError: If skill is not found.
        """
        skill = self._skills.get(skill_uri)

        if skill is None:
            raise SkillNotFoundError(f"Skill '{skill_uri}' not found in {self._path.as_posix()}.")

        return skill
