#!/usr/bin/env python
"""Check structural integrity of Claude Code skills.

Aligned with official Anthropic skill validator pattern:
https://github.com/anthropics/skills/blob/main/skills/skill-creator/scripts/quick_validate.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import yaml

SKILLS_ROOT = Path('skills')
SKILL_MD_MAX_LINES = 500

# Allowed frontmatter properties (from official spec)
ALLOWED_PROPERTIES = {'name', 'description', 'license', 'allowed-tools', 'metadata', 'compatibility'}


def validate_name(name: str, expected_dir_name: str) -> list[str]:
    """Validate skill name per spec."""
    errors: list[str] = []
    if not isinstance(name, str):
        errors.append(f'Name must be a string, got {type(name).__name__}')
        return errors
    name = name.strip()
    if not name:
        errors.append('Name cannot be empty')
        return errors
    if len(name) > 64:
        errors.append(f'Name is too long ({len(name)} chars). Maximum is 64.')
    if not re.match(r'^[a-z0-9-]+$', name):
        errors.append(f"Name '{name}' must be hyphen-case (lowercase letters, digits, hyphens only)")
    if name.startswith('-') or name.endswith('-'):
        errors.append(f"Name '{name}' cannot start or end with hyphen")
    if '--' in name:
        errors.append(f"Name '{name}' cannot contain consecutive hyphens")

    # Check name matches directory name
    if name != expected_dir_name:
        errors.append(f"Name '{name}' must match directory name '{expected_dir_name}'")

    return errors


def validate_description(desc: str) -> list[str]:
    """Validate skill description per spec."""
    errors: list[str] = []
    if not isinstance(desc, str):
        errors.append(f'Description must be a string, got {type(desc).__name__}')
        return errors
    desc = desc.strip()
    if not desc:
        errors.append('Description cannot be empty')
        return errors
    if len(desc) > 1024:
        errors.append(f'Description is too long ({len(desc)} chars). Maximum is 1024.')
    # Check for XML tags (not just any angle brackets)
    if re.search(r'<[a-zA-Z/][^>]*>', desc):
        errors.append('Description cannot contain XML tags')
    return errors


def validate_frontmatter(content: str, skill_dir_name: str) -> list[str]:
    """Validate SKILL.md YAML frontmatter."""
    errors: list[str] = []

    if not content.startswith('---'):
        errors.append('SKILL.md must start with YAML frontmatter (---)')
        return errors

    match = re.match(r'^---\n(.*?)\n---', content, re.DOTALL)
    if not match:
        errors.append('SKILL.md frontmatter is not closed (missing second ---)')
        return errors

    frontmatter_text = match.group(1)

    try:
        frontmatter: dict[str, Any] = yaml.safe_load(frontmatter_text)
        if not isinstance(frontmatter, dict):
            errors.append('Frontmatter must be a YAML dictionary')
            return errors
    except yaml.YAMLError as e:
        errors.append(f'Invalid YAML in frontmatter: {e}')
        return errors

    # Check for unexpected properties
    unexpected: set[str] = set(frontmatter.keys()) - ALLOWED_PROPERTIES
    if unexpected:
        errors.append(f'Unexpected frontmatter keys: {", ".join(sorted(unexpected))}')

    # Check required fields and validate their values
    if 'name' not in frontmatter:
        errors.append('Missing required field: name')
    else:
        name: str = frontmatter['name']
        errors.extend(validate_name(name, skill_dir_name))

    if 'description' not in frontmatter:
        errors.append('Missing required field: description')
    else:
        desc: str = frontmatter['description']
        errors.extend(validate_description(desc))

    return errors


def validate_skill(skill_md: Path, verbose: bool = False) -> list[str]:
    """Validate a single SKILL.md file."""
    errors: list[str] = []
    skill_dir_name = skill_md.parent.name

    if not skill_md.exists():
        errors.append(f'{skill_md} does not exist')
        return errors

    content = skill_md.read_text(encoding='utf-8')

    # Check line limit
    line_count = len(content.splitlines())
    if line_count > SKILL_MD_MAX_LINES:
        errors.append(f'{skill_md} is {line_count} lines (max {SKILL_MD_MAX_LINES})')
    elif verbose:
        print(f'OK: {skill_md} is {line_count} lines (max {SKILL_MD_MAX_LINES})')

    # Validate frontmatter
    errors.extend(validate_frontmatter(content, skill_dir_name))

    return errors


def discover_skills() -> list[Path]:
    """Discover all skills in the skills directory."""
    if not SKILLS_ROOT.exists():
        return []
    return sorted(SKILLS_ROOT.glob('*/SKILL.md'))


def main() -> int:
    """Validate all skills and report results."""
    verbose = '--verbose' in sys.argv or '-v' in sys.argv
    all_errors: list[tuple[Path, list[str]]] = []

    # Discover all skills
    skill_files = discover_skills()

    if not skill_files:
        print(f'No skills found in {SKILLS_ROOT}/')
        return 1

    if verbose:
        print(f'Found {len(skill_files)} skill(s) in {SKILLS_ROOT}/')

    # Validate each skill
    for skill_md in skill_files:
        if verbose:
            print(f'\nValidating {skill_md}...')
        errors = validate_skill(skill_md, verbose)
        if errors:
            all_errors.append((skill_md, errors))

    # Report results
    if all_errors:
        total_errors = sum(len(errors) for _, errors in all_errors)
        print(f'\nSkills check FAILED with {total_errors} error(s) in {len(all_errors)} skill(s):')
        for skill_md, errors in all_errors:
            print(f'\n  {skill_md}:')
            for error in errors:
                print(f'    - {error}')
        return 1

    print(f'Skills check passed! ({len(skill_files)} skill(s) validated)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
