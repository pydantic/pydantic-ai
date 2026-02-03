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
ALLOWED_PROPERTIES = {'name', 'description', 'license', 'allowed-tools', 'metadata', 'user-invocable'}


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

    # Check required fields
    if 'name' not in frontmatter:
        errors.append('Missing required field: name')
    if 'description' not in frontmatter:
        errors.append('Missing required field: description')

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

    # 4. Check VERSION file exists
    version_file = SKILLS_DIR / 'VERSION'
    if not version_file.exists():
        errors.append(f'{version_file} does not exist')
    elif verbose:
        print(f'OK: {version_file} exists')

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
