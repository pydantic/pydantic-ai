"""Filesystem-based skill discovery and management.

This module provides [`SkillsDirectory`][pydantic_ai.toolsets.skills.SkillsDirectory]
for discovering and loading skills from a filesystem directory.

Supports nested skill directories with configurable depth limits and provides
internal helper functions for skill validation, metadata parsing, and resource/script discovery.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any

import yaml

from ._exceptions import (
    SkillNotFoundError,
    SkillValidationError,
)
from ._local import (
    CallableSkillScriptExecutor,
    LocalSkillScriptExecutor,
    create_file_based_resource,
    create_file_based_script,
)
from ._types import Skill, SkillResource, SkillScript

__all__ = ['SkillsDirectory']

# agentskills.io naming convention: lowercase letters, numbers, and hyphens only (no consecutive hyphens)
SKILL_NAME_PATTERN = re.compile(r'^[a-z0-9]+(-[a-z0-9]+)*$')
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

    # Validate compatibility (if provided)
    compatibility = frontmatter.get('compatibility', '')
    if compatibility and len(compatibility) > 500:
        warnings.warn(
            f'Skill compatibility exceeds 500 characters ({len(compatibility)} chars)', UserWarning, stacklevel=2
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

    Resources are text files other than SKILL.md in the root directory
    or in any subdirectory (e.g., references/, assets/).

    Supported file types:
    - Markdown: .md
    - JSON: .json
    - YAML: .yaml, .yml
    - CSV: .csv
    - XML: .xml
    - Text: .txt

    Args:
        skill_folder: Path to the skill directory.

    Returns:
        List of discovered SkillResource objects.
    """
    resources: list[SkillResource] = []

    # Supported textual file extensions
    supported_extensions = ['.md', '.json', '.yaml', '.yml', '.csv', '.xml', '.txt']

    # Find all supported files recursively, excluding SKILL.md
    for extension in supported_extensions:
        for resource_file in skill_folder.rglob(f'*{extension}'):
            if resource_file.name.upper() != 'SKILL.MD':
                rel_path = resource_file.relative_to(skill_folder)
                # Always use relative path for consistency and better context
                name = str(rel_path)
                resources.append(
                    create_file_based_resource(
                        name=name,
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


def _discover_scripts(
    skill_folder: Path,
    skill_name: str,
    executor: LocalSkillScriptExecutor | CallableSkillScriptExecutor,
) -> list[SkillScript]:
    """Discover executable scripts in a skill folder.

    Looks for Python scripts in:
    - Directly in the skill folder (*.py)
    - In a scripts/ subdirectory

    Args:
        skill_folder: Path to the skill directory.
        skill_name: Name of the parent skill.
        executor: Executor for running file-based scripts.

    Returns:
        List of discovered SkillScript objects.
    """
    scripts: list[SkillScript] = []

    # Find .py files in skill folder root (excluding __init__.py)
    for py_file in skill_folder.glob('*.py'):
        if py_file.name != '__init__.py':
            rel_path = py_file.relative_to(skill_folder)
            scripts.append(
                create_file_based_script(
                    name=str(rel_path),  # relative path with .py extension
                    uri=str(py_file.resolve()),
                    skill_name=skill_name,
                    executor=executor,
                )
            )

    # Find .py files in scripts/ subdirectory
    scripts_dir = skill_folder / 'scripts'
    if scripts_dir.exists() and scripts_dir.is_dir():
        for py_file in scripts_dir.glob('*.py'):
            if py_file.name != '__init__.py':
                rel_path = py_file.relative_to(skill_folder)
                scripts.append(
                    create_file_based_script(
                        name=str(rel_path),  # relative path with .py extension
                        uri=str(py_file.resolve()),
                        skill_name=skill_name,
                        executor=executor,
                    )
                )

    return scripts


def _discover_skills(
    path: str | Path,
    validate: bool = True,
    max_depth: int | None = 3,
    script_executor: LocalSkillScriptExecutor | CallableSkillScriptExecutor | None = None,
) -> list[Skill]:
    """Discover skills from a filesystem directory.

    Searches for SKILL.md files in the given directory and loads
    skill metadata and structure.

    Args:
        path: Directory path to search for skills.
        validate: Whether to validate skill structure (requires name and description).
        max_depth: Maximum depth to search for SKILL.md files. None for unlimited.
            Default is 3 levels deep to prevent performance issues with large trees.
        script_executor: Optional custom script executor for file-based scripts.

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

            # Extract standard fields and remaining metadata
            license_field = frontmatter.get('license')
            compatibility_field = frontmatter.get('compatibility')
            metadata = {
                k: v for k, v in frontmatter.items() if k not in ('name', 'description', 'license', 'compatibility')
            }

            # Validate metadata
            if validate:
                _validate_skill_metadata(frontmatter, instructions)

            # Discover resources and scripts
            resources = _discover_resources(skill_folder)
            scripts = _discover_scripts(skill_folder, name, script_executor or LocalSkillScriptExecutor())

            # Create skill instance
            skill = Skill(
                name=name,
                description=description,
                content=instructions,
                license=license_field,
                compatibility=compatibility_field,
                uri=str(skill_folder.resolve()),
                resources=resources,
                scripts=scripts,
                metadata=metadata if metadata else None,
            )

            skills.append(skill)

        except SkillValidationError as sve:
            if validate:
                raise
            else:
                warnings.warn(f'Skipping invalid skill at {skill_file}: {sve}', UserWarning, stacklevel=2)
        except (OSError, ValueError, KeyError) as e:
            # Catch specific exceptions that can occur during skill loading
            raise SkillValidationError(f'Failed to load skill from {skill_file}: {e}') from e

    return skills


class SkillsDirectory:
    """Skill source for a single filesystem directory.

    Discovers and loads skills from a local directory by finding SKILL.md files
    and automatically discovering associated resources and scripts.

    File-based scripts are executed using the configured script executor
    (LocalSkillScriptExecutor or CallableSkillScriptExecutor).
    """

    def __init__(
        self,
        *,
        path: str | Path,
        validate: bool = True,
        max_depth: int | None = 3,
        script_executor: LocalSkillScriptExecutor | CallableSkillScriptExecutor | None = None,
    ) -> None:
        """Initialize the skills directory source.

        Args:
            path: Directory path to search for skills.
            validate: Validate skill structure on discovery.
            max_depth: Maximum depth for skill discovery (None for unlimited).
            script_executor: Optional custom script executor for file-based scripts.
                Can be LocalSkillScriptExecutor or CallableSkillScriptExecutor.
                If None, uses LocalSkillScriptExecutor with default settings.

        Example:
            ```python
            # Discovery mode - single directory
            source = SkillsDirectory(path="./skills")

            # With custom executor
            from pydantic_ai.toolsets.skills import LocalSkillScriptExecutor

            executor = LocalSkillScriptExecutor(timeout=60)
            source = SkillsDirectory(path="./skills", script_executor=executor)

            # With callable executor
            from pydantic_ai.toolsets.skills import CallableSkillScriptExecutor

            async def my_executor(script, args=None, skill_uri=None):
                return f"Executed {script.name}"

            executor = CallableSkillScriptExecutor(func=my_executor)
            source = SkillsDirectory(path="./skills", script_executor=executor)
            ```
        """
        self._path = Path(path).expanduser().resolve()
        self._validate = validate
        self._max_depth = max_depth
        self._script_executor = script_executor or LocalSkillScriptExecutor()

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
            script_executor=self._script_executor,
        )

        return {skill.uri: skill for skill in skills if skill.uri is not None}

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
