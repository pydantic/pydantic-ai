#!/usr/bin/env python3
"""Fix default_factory=list/dict/set to use lambda with proper types."""

import re
from pathlib import Path
import sys

def fix_file(file_path: Path) -> tuple[int, list[str]]:
    """Fix default_factory in a single file. Returns (num_changes, changed_lines)."""
    content = file_path.read_text()
    lines = content.split('\n')
    changes = 0
    changed_lines = []

    for i, line in enumerate(lines):
        original = line

        # Pattern: TYPE_ANNOTATION = field(default_factory=list/dict/set)
        # Handles field(), dataclasses.field(), etc.
        match = re.match(
            r'^(\s*)(\w+):\s*(.+?)\s*=\s*((?:dataclasses\.)?field)\(default_factory=(list|dict|set)(?!\[)(.*)\)(.*)$',
            line
        )

        if match:
            indent, var_name, type_annotation, field_func, factory_type, rest_args, comment = match.groups()
            # Replace with typed collection as default_factory
            if rest_args and not rest_args.startswith(','):
                rest_args = ', ' + rest_args.strip()
            lines[i] = f'{indent}{var_name}: {type_annotation} = {field_func}(default_factory={type_annotation}{rest_args}){comment}'
            changes += 1
            changed_lines.append(f'{file_path}:{i+1}')
            continue

        # Pattern: TYPE_ANNOTATION = Field(default_factory=list/dict/set, ...)
        match = re.match(
            r'^(\s*)(\w+):\s*([^\s=]+)\s*=\s*Field\(default_factory=(list|dict|set)(?!\[)(.*)\)(.*)$',
            line
        )

        if match:
            indent, var_name, type_annotation, factory_type, rest_args, comment = match.groups()
            # Replace with typed collection as default_factory
            if rest_args and not rest_args.startswith(','):
                rest_args = ', ' + rest_args.strip()
            lines[i] = f'{indent}{var_name}: {type_annotation} = Field(default_factory={type_annotation}{rest_args}){comment}'
            changes += 1
            changed_lines.append(f'{file_path}:{i+1}')
            continue

    if changes > 0:
        file_path.write_text('\n'.join(lines))

    return changes, changed_lines


def main():
    # Find all Python files
    root = Path('.')
    python_files = list(root.rglob('*.py'))

    total_changes = 0
    all_changed_lines = []

    for file_path in python_files:
        # Skip venv, .git, etc
        if any(part.startswith('.') or part == '__pycache__' or part == 'venv' for part in file_path.parts):
            continue

        try:
            changes, changed_lines = fix_file(file_path)
            if changes > 0:
                total_changes += changes
                all_changed_lines.extend(changed_lines)
                print(f'Fixed {changes} lines in {file_path}')
        except Exception as e:
            print(f'Error processing {file_path}: {e}', file=sys.stderr)

    print(f'\nTotal changes: {total_changes}')
    print(f'\nChanged lines:')
    for line in all_changed_lines[:50]:  # Show first 50
        print(f'  {line}')
    if len(all_changed_lines) > 50:
        print(f'  ... and {len(all_changed_lines) - 50} more')

    return total_changes


if __name__ == '__main__':
    changes = main()
    sys.exit(0 if changes > 0 else 1)
