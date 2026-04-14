#!/usr/bin/env python
"""Check that all cassette files have corresponding tests.

This script verifies that every VCR cassette file in the test suite has a
corresponding test function. Orphaned cassettes (cassettes without tests)
indicate dead code that should be removed.

Usage:
    python scripts/check_cassettes.py [--verbose]
"""

from __future__ import annotations

import ast
import sys
from collections import defaultdict
from pathlib import Path

_FORBIDDEN_CHARS = r"""<>?%*:|"'/\\"""


def _sanitize_cassette_name(name: str) -> str:
    """Replicate pytest-recording's cassette name sanitization."""
    for ch in _FORBIDDEN_CHARS:
        name = name.replace(ch, '-')
    return name


def _has_vcr_marker(decorator_list: list[ast.expr]) -> bool:
    """Check if a decorator list contains pytest.mark.vcr (with or without parens)."""
    for dec in decorator_list:
        # @pytest.mark.vcr or @pytest.mark.vcr()
        if isinstance(dec, ast.Attribute) and dec.attr == 'vcr':
            return True
        if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute) and dec.func.attr == 'vcr':
            return True
    return False


def _has_cassette_marker(decorator_list: list[ast.expr]) -> bool:
    """Check if a decorator list contains pytest.mark.ws_cassette (with or without parens)."""
    for dec in decorator_list:
        if isinstance(dec, ast.Attribute) and dec.attr == 'ws_cassette':
            return True
        if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute) and dec.func.attr == 'ws_cassette':
            return True
    return False


def _has_module_vcr_marker(tree: ast.Module) -> bool:
    """Check if the module has pytestmark = [..., pytest.mark.vcr, ...]."""
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not (isinstance(target, ast.Name) and target.id == 'pytestmark'):
                continue
            return 'vcr' in ast.dump(node.value)
    return False


def _collect_vcr_tests_from_file(path: Path) -> set[str]:
    """Parse a Python test file and return cassette names for VCR or ws_cassette-marked tests."""
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return set()

    module_has_vcr = _has_module_vcr_marker(tree)
    cassette_names: set[str] = set()

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if not node.name.startswith('test_'):
                continue
            if module_has_vcr or _has_vcr_marker(node.decorator_list) or _has_cassette_marker(node.decorator_list):
                cassette_names.add(_sanitize_cassette_name(node.name))

        elif isinstance(node, ast.ClassDef):
            class_has_vcr = _has_vcr_marker(node.decorator_list)
            for method in ast.iter_child_nodes(node):
                if not isinstance(method, ast.FunctionDef | ast.AsyncFunctionDef):
                    continue
                if not method.name.startswith('test_'):
                    continue
                if module_has_vcr or class_has_vcr or _has_vcr_marker(method.decorator_list) or _has_cassette_marker(method.decorator_list):
                    cassette_names.add(_sanitize_cassette_name(f'{node.name}.{method.name}'))

    return cassette_names


def get_all_cassettes() -> dict[str, set[str]]:
    """Return {test_file_stem: set of cassette names (without .yaml)}."""
    cassettes: dict[str, set[str]] = {}

    for cassette_dir in Path('tests').rglob('cassettes'):
        if not cassette_dir.is_dir():
            continue
        for subdir in cassette_dir.iterdir():
            if subdir.is_dir():
                test_stem = subdir.name
                # Handle double extensions like .xai.yaml (xAI uses gRPC/protobuf, not HTTP)
                cassette_names = {f.stem[:-4] if f.stem.endswith('.xai') else f.stem for f in subdir.glob('*.yaml')}
                cassettes.setdefault(test_stem, set()).update(cassette_names)

    return cassettes


def get_all_tests() -> dict[str, set[str]]:
    """Use AST parsing to find all VCR-marked tests and their cassette names."""
    tests: dict[str, set[str]] = defaultdict(set)

    for test_file in Path('tests').rglob('test_*.py'):
        cassette_names = _collect_vcr_tests_from_file(test_file)
        if cassette_names:
            tests[test_file.stem].update(cassette_names)

    return dict(tests)


def main() -> int:
    verbose = '--verbose' in sys.argv or '-v' in sys.argv

    print('Collecting cassettes...')
    cassettes = get_all_cassettes()
    total_cassettes = sum(len(c) for c in cassettes.values())
    print(f'Found {total_cassettes} cassettes in {len(cassettes)} test modules')

    print('Collecting VCR-marked tests...')
    tests = get_all_tests()
    total_tests = sum(len(t) for t in tests.values())
    print(f'Found {total_tests} tests in {len(tests)} test modules')

    orphans: list[str] = []
    matched = 0

    for test_file, cassette_names in sorted(cassettes.items()):
        expected_cassettes = tests.get(test_file, set())

        if not expected_cassettes and verbose:
            print(f'Warning: No tests found for module {test_file}')

        for cassette in sorted(cassette_names):
            # Parametrized tests produce cassettes like test_foo[param].yaml
            # Strip the [param] suffix to match the base test name
            base_name = cassette.split('[')[0]
            if cassette in expected_cassettes or base_name in expected_cassettes:
                matched += 1
                if verbose:
                    print(f'  OK: {test_file}/{cassette}.yaml')
            else:
                orphans.append(f'{test_file}/{cassette}.yaml')

    print()
    print(f'Orphaned cassettes check: {matched} matched, {len(orphans)} orphaned')

    if orphans:
        print()
        print('Orphaned cassettes (no matching test):')
        for orphan in sorted(orphans):
            print(f'  - {orphan}')
        print()
        print('These cassettes have no corresponding test and may be dead code.')
        print('Either add a test or remove the cassette.')
        return 1

    print('All cassettes have matching tests!')
    return 0


if __name__ == '__main__':
    sys.exit(main())
