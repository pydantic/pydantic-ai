#!/usr/bin/env python
"""Check that all cassette files have corresponding tests.

This script verifies that every VCR cassette file in the test suite has a
corresponding test function. Orphaned cassettes (cassettes without tests)
indicate dead code that should be removed.

Usage:
    python scripts/check_cassettes.py [--verbose]
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


def get_all_cassettes() -> dict[str, set[str]]:
    """Return {test_file_stem: set of cassette names (without .yaml)}."""
    cassettes: dict[str, set[str]] = {}

    for cassette_dir in Path('tests').rglob('cassettes'):
        if not cassette_dir.is_dir():
            continue
        for subdir in cassette_dir.iterdir():
            if subdir.is_dir():
                test_stem = subdir.name  # e.g., 'test_bedrock'
                cassette_names = {f.stem for f in subdir.glob('*.yaml')}
                if test_stem in cassettes:
                    cassettes[test_stem].update(cassette_names)
                else:
                    cassettes[test_stem] = cassette_names

    return cassettes


def get_all_tests() -> dict[str, set[str]]:
    """Use pytest --collect-only to get all VCR-marked test names."""
    result = subprocess.run(
        ['uv', 'run', 'pytest', '--collect-only', '-q', '-m', 'vcr', 'tests/'],
        capture_output=True,
        text=True,
    )

    tests: dict[str, set[str]] = {}
    for line in result.stdout.splitlines():
        # Parse lines like: tests/models/test_bedrock.py::test_name[param]
        # or: tests/models/test_bedrock.py::TestClass::test_method[param]
        match = re.match(r'tests/.*/?(test_\w+)\.py::(.+)', line)
        if match:
            test_file, test_name = match.groups()
            tests.setdefault(test_file, set()).add(test_name)

    return tests


def normalize_cassette_name(cassette: str) -> str:
    """Normalize cassette name for comparison with test names.

    Handles:
    - Parametrized tests: test_foo[param] -> test_foo
    - Class-based tests: TestClass.test_method -> TestClass::test_method
    """
    # Strip parameter suffix
    base_name = re.sub(r'\[.*\]$', '', cassette)
    # Convert dot notation to pytest's :: notation for class methods
    base_name = base_name.replace('.', '::')
    return base_name


def find_matching_test(cassette: str, test_names: set[str]) -> bool:
    """Check if a cassette has a matching test."""
    normalized = normalize_cassette_name(cassette)

    for test_name in test_names:
        # Strip parameters from test name for comparison
        test_base = re.sub(r'\[.*\]$', '', test_name)
        if test_base == normalized:
            return True

    return False


def main() -> int:
    verbose = '--verbose' in sys.argv or '-v' in sys.argv

    print('Collecting cassettes...')
    cassettes = get_all_cassettes()
    total_cassettes = sum(len(c) for c in cassettes.values())
    print(f'Found {total_cassettes} cassettes in {len(cassettes)} test modules')

    print('Collecting VCR-marked tests (this may take a moment)...')
    tests = get_all_tests()
    total_tests = sum(len(t) for t in tests.values())
    print(f'Found {total_tests} tests in {len(tests)} test modules')

    orphans: list[str] = []
    matched = 0

    for test_file, cassette_names in sorted(cassettes.items()):
        test_names = tests.get(test_file, set())

        if not test_names and verbose:
            print(f'Warning: No tests found for module {test_file}')

        for cassette in sorted(cassette_names):
            if find_matching_test(cassette, test_names):
                matched += 1
                if verbose:
                    print(f'  OK: {test_file}/{cassette}.yaml')
            else:
                orphans.append(f'{test_file}/{cassette}.yaml')

    print()
    print(f'Results: {matched} matched, {len(orphans)} orphaned')

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
