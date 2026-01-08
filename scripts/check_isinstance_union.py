#!/usr/bin/env python3
"""Check and fix isinstance/issubclass calls to use union syntax.

This script detects and fixes:
1. isinstance(x, (A, B)) → isinstance(x, A | B)
2. isinstance(x, (A | B)) → isinstance(x, A | B) (removes redundant parens)

Usage:
    python check_isinstance_union.py [--fix] [--check] files...

Options:
    --fix   Fix issues in-place (default)
    --check Only check, don't modify files
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path


def find_issues(source: str, tree: ast.Module) -> list[tuple[int, int, int, int, str]]:
    """Find isinstance/issubclass calls with tuple or redundant-paren type args.

    Returns list of (start_line, start_col, end_line, end_col, replacement).
    Positions are 0-indexed.
    """
    issues: list[tuple[int, int, int, int, str]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name):
            continue
        if node.func.id not in ('isinstance', 'issubclass'):
            continue
        if len(node.args) < 2:
            continue

        type_arg = node.args[1]

        if isinstance(type_arg, ast.Tuple):
            # Case 1: isinstance(x, (A, B)) or isinstance(x, (A | B,)) with trailing comma
            # Convert tuple elements to union syntax
            if len(type_arg.elts) == 0:
                continue

            # Generate the replacement by unparsing each element and joining with |
            parts = [ast.unparse(elt) for elt in type_arg.elts]
            replacement = ' | '.join(parts)

            # Get positions (1-indexed in AST, we convert to 0-indexed)
            start_line = type_arg.lineno - 1
            start_col = type_arg.col_offset
            end_line = type_arg.end_lineno - 1 if type_arg.end_lineno else start_line
            end_col = type_arg.end_col_offset if type_arg.end_col_offset else start_col

            issues.append((start_line, start_col, end_line, end_col, replacement))

        else:
            # Case 2: isinstance(x, (A | B)) or isinstance(x, (A)) - check for redundant parens
            # This handles BinOp (unions), Name, Attribute, Subscript (generic types)
            start_line = type_arg.lineno - 1
            start_col = type_arg.col_offset
            end_line = type_arg.end_lineno - 1 if type_arg.end_lineno else start_line
            end_col = type_arg.end_col_offset if type_arg.end_col_offset else start_col

            # Check if there are parentheses around the expression
            lines = source.splitlines(keepends=True)
            # We need to look at the character before start_col
            if start_col > 0 and start_line < len(lines):
                char_before = lines[start_line][start_col - 1]
                if char_before == '(':
                    # Check if there's a matching ) after
                    if end_line < len(lines):
                        line_after = lines[end_line]
                        if end_col < len(line_after) and line_after[end_col] == ')':
                            # Found redundant parens - the replacement is just the inner content
                            replacement = ast.unparse(type_arg)
                            # Expand positions to include the parens
                            issues.append((start_line, start_col - 1, end_line, end_col + 1, replacement))

    return issues


def fix_source(source: str, issues: list[tuple[int, int, int, int, str]]) -> str:
    """Apply fixes to source code. Issues must be processed in reverse order."""
    lines = source.splitlines(keepends=True)

    # Handle case where file doesn't end with newline
    if source and not source.endswith('\n'):
        if lines:
            lines[-1] = lines[-1] + '\n'

    # Sort issues in reverse order (bottom-up, right-to-left)
    sorted_issues = sorted(issues, key=lambda x: (x[0], x[1]), reverse=True)

    for start_line, start_col, end_line, end_col, replacement in sorted_issues:
        if start_line == end_line:
            line = lines[start_line]
            lines[start_line] = line[:start_col] + replacement + line[end_col:]
        else:
            # Multi-line replacement
            first_line = lines[start_line]
            last_line = lines[end_line]
            lines[start_line] = first_line[:start_col] + replacement + last_line[end_col:]
            # Remove intermediate lines
            del lines[start_line + 1 : end_line + 1]

    result = ''.join(lines)
    # Preserve original ending
    if source and not source.endswith('\n') and result.endswith('\n'):
        result = result[:-1]
    return result


def check_file(filepath: Path, fix: bool = False) -> list[str]:
    """Check a single file for issues. Returns list of issue descriptions."""
    try:
        source = filepath.read_text()
    except Exception as e:
        return [f'{filepath}: error reading file: {e}']

    try:
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError as e:
        return [f'{filepath}: syntax error: {e}']

    issues = find_issues(source, tree)
    if not issues:
        return []

    messages: list[str] = []
    for start_line, start_col, end_line, end_col, replacement in issues:
        # Report 1-indexed line numbers for user-friendly output
        if start_line == end_line:
            loc = f'{start_line + 1}:{start_col + 1}'
        else:
            loc = f'{start_line + 1}:{start_col + 1}-{end_line + 1}:{end_col + 1}'
        messages.append(f'{filepath}:{loc}: use union syntax: {replacement}')

    if fix:
        fixed_source = fix_source(source, issues)
        filepath.write_text(fixed_source)
        messages = [msg + ' (fixed)' for msg in messages]

    return messages


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Check isinstance/issubclass calls use union syntax instead of tuples.'
    )
    parser.add_argument('files', nargs='*', type=Path, help='Files to check')
    parser.add_argument('--fix', action='store_true', default=True, help='Fix issues in-place (default)')
    parser.add_argument('--check', action='store_true', help='Only check, do not modify files')

    args = parser.parse_args()

    if args.check:
        args.fix = False

    if not args.files:
        parser.print_help()
        return 0

    all_messages: list[str] = []
    for filepath in args.files:
        if filepath.is_file() and filepath.suffix == '.py':
            messages = check_file(filepath, fix=args.fix)
            all_messages.extend(messages)
        elif filepath.is_dir():
            for pyfile in filepath.rglob('*.py'):
                messages = check_file(pyfile, fix=args.fix)
                all_messages.extend(messages)

    for msg in all_messages:
        print(msg)

    # Return non-zero if issues were found (signals to pre-commit that files changed)
    return 1 if all_messages else 0


if __name__ == '__main__':
    sys.exit(main())
