"""Skill discovery and parsing utilities.

This module provides functions for discovering skills from filesystem directories
and parsing SKILL.md files with YAML frontmatter.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from ._exceptions import SkillValidationError
from ._types import (
    Skill,
    SkillMetadata,
    SkillResource,
    SkillScript,
)

# Anthropic's naming convention: lowercase letters, numbers, and hyphens only
SKILL_NAME_PATTERN = re.compile(r'^[a-z0-9-]+$')
RESERVED_WORDS = {'anthropic', 'claude'}


def validate_skill_metadata(
    frontmatter: dict[str, Any],
    instructions: str,
) -> list[str]:
    """Validate skill metadata against Anthropic's requirements.

    Args:
        frontmatter: Parsed YAML frontmatter.
        instructions: The skill instructions content.

    Returns:
        List of validation warnings (empty if no issues).
    """
    warnings_list: list[str] = []

    name = frontmatter.get('name', '')
    description = frontmatter.get('description', '')

    # Validate name format
    if name:
        # Check length first to prevent regex on excessively long strings
        if len(name) > 64:
            warnings_list.append(f"Skill name '{name}' exceeds 64 characters ({len(name)} chars)")
        # Only run regex if name is reasonable length (defense in depth)
        elif not SKILL_NAME_PATTERN.match(name):
            warnings_list.append(f"Skill name '{name}' should contain only lowercase letters, numbers, and hyphens")
        # Check for reserved words
        for reserved in RESERVED_WORDS:
            if reserved in name:
                warnings_list.append(f"Skill name '{name}' contains reserved word '{reserved}'")

    # Validate description
    if description and len(description) > 1024:
        warnings_list.append(f'Skill description exceeds 1024 characters ({len(description)} chars)')

    # Validate instructions length (Anthropic recommends under 500 lines)
    lines = instructions.split('\n')
    if len(lines) > 500:
        warnings_list.append(
            f'SKILL.md body exceeds recommended 500 lines ({len(lines)} lines). '
            f'Consider splitting into separate resource files.'
        )

    return warnings_list


def parse_skill_md(content: str) -> tuple[dict[str, Any], str]:
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
        frontmatter_data = yaml.safe_load(frontmatter_yaml)
        if frontmatter_data is None:
            frontmatter: dict[str, Any] = {}
        elif isinstance(frontmatter_data, dict):
            frontmatter = frontmatter_data
        else:
            frontmatter = {}
    except yaml.YAMLError as e:
        raise SkillValidationError(f'Failed to parse YAML frontmatter: {e}') from e

    return frontmatter, instructions


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
                    path=md_file.resolve(),
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
                        path=resource_file.resolve(),
                    )
                )

    return resources


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
                    path=py_file.resolve(),
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
                        path=py_file.resolve(),
                        skill_name=skill_name,
                    )
                )

    return scripts


def discover_skills(
    directories: Sequence[str | Path],
    validate: bool = True,
) -> list[Skill]:
    """Discover skills from filesystem directories.

    Searches for SKILL.md files in the given directories and loads
    skill metadata and structure.

    Args:
        directories: List of directory paths to search for skills.
        validate: Whether to validate skill structure (requires name and description).

    Returns:
        List of discovered Skill objects.

    Raises:
        SkillValidationError: If validation is enabled and a skill is invalid.
    """
    skills: list[Skill] = []

    for skill_dir in directories:
        dir_path = Path(skill_dir).expanduser().resolve()

        if not dir_path.exists():
            continue

        if not dir_path.is_dir():
            continue

        # Find all SKILL.md files (recursive search)
        for skill_file in dir_path.glob('**/SKILL.md'):
            try:
                skill_folder = skill_file.parent
                content = skill_file.read_text(encoding='utf-8')
                frontmatter, instructions = parse_skill_md(content)

                # Get required fields
                name = frontmatter.get('name')
                description = frontmatter.get('description', '')

                # Validation
                if validate:
                    if not name:
                        continue

                # Use folder name if name not provided
                if not name:
                    name = skill_folder.name

                # Extract extra metadata fields
                extra = {k: v for k, v in frontmatter.items() if k not in ('name', 'description')}

                # Create metadata
                metadata = SkillMetadata(
                    name=name,
                    description=description,
                    extra=extra,
                )

                # Validate metadata
                if validate:
                    validate_skill_metadata(frontmatter, instructions)

                # Discover resources and scripts
                resources = _discover_resources(skill_folder)
                scripts = _discover_scripts(skill_folder, name)

                # Create skill
                skill = Skill(
                    name=name,
                    path=skill_folder.resolve(),
                    metadata=metadata,
                    content=instructions,
                    resources=resources,
                    scripts=scripts,
                )

                skills.append(skill)

            except SkillValidationError:
                raise
            except OSError:
                continue

    return skills
