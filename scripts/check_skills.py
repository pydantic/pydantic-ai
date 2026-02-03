#!/usr/bin/env python
"""Check structural integrity of Claude Code skills.

Aligned with official Anthropic skill validator pattern:
https://github.com/anthropics/skills/blob/main/skills/skill-creator/scripts/quick_validate.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

SKILLS_DIR = Path('skills/pydantic-ai')
SKILL_MD = SKILLS_DIR / 'SKILL.md'
SKILL_MD_MAX_LINES = 500

# Allowed frontmatter properties (from official spec)
ALLOWED_PROPERTIES = {'name', 'description', 'license', 'allowed-tools', 'metadata', 'compatibility'}


def validate_name(name: str) -> list[str]:
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
    if '<' in desc or '>' in desc:
        errors.append('Description cannot contain angle brackets (< or >)')
    return errors


def validate_frontmatter(content: str) -> list[str]:
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
        frontmatter = yaml.safe_load(frontmatter_text)
        if not isinstance(frontmatter, dict):
            errors.append('Frontmatter must be a YAML dictionary')
            return errors
    except yaml.YAMLError as e:
        errors.append(f'Invalid YAML in frontmatter: {e}')
        return errors

    # Check for unexpected properties
    unexpected = set(frontmatter.keys()) - ALLOWED_PROPERTIES
    if unexpected:
        errors.append(f"Unexpected frontmatter keys: {', '.join(sorted(unexpected))}")

    # Check required fields and validate their values
    if 'name' not in frontmatter:
        errors.append('Missing required field: name')
    else:
        errors.extend(validate_name(frontmatter['name']))

    if 'description' not in frontmatter:
        errors.append('Missing required field: description')
    else:
        errors.extend(validate_description(frontmatter['description']))

    return errors


def main() -> int:
    verbose = '--verbose' in sys.argv or '-v' in sys.argv
    errors: list[str] = []

    # 1. Check SKILL.md exists
    if not SKILL_MD.exists():
        errors.append(f'{SKILL_MD} does not exist')
    else:
        content = SKILL_MD.read_text()

        # 2. Check line limit
        line_count = len(content.splitlines())
        if line_count > SKILL_MD_MAX_LINES:
            errors.append(f'{SKILL_MD} is {line_count} lines (max {SKILL_MD_MAX_LINES})')
        elif verbose:
            print(f'OK: {SKILL_MD} is {line_count} lines (max {SKILL_MD_MAX_LINES})')

        # 3. Validate frontmatter
        errors.extend(validate_frontmatter(content))

    # Report results
    if errors:
        print(f'Skills check FAILED with {len(errors)} error(s):')
        for error in errors:
            print(f'  - {error}')
        return 1

    print('Skills check passed!')
    return 0


if __name__ == '__main__':
    sys.exit(main())
