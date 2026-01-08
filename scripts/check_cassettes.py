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

from pytest_recording.plugin import get_default_cassette_name


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
    """Use pytest --collect-only to get all VCR-marked tests and compute expected cassette names."""
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
            # Extract class name if present (TestClass::test_method[param])
            if '::' in test_name:
                class_name, method_name = test_name.split('::', 1)
                # Create a mock class for get_default_cassette_name
                class_obj = type(class_name, (), {})
                cassette_name = get_default_cassette_name(class_obj, method_name)
            else:
                cassette_name = get_default_cassette_name(None, test_name)
            tests.setdefault(test_file, set()).add(cassette_name)

    return tests


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
        expected_cassettes = tests.get(test_file, set())

        if not expected_cassettes and verbose:
            print(f'Warning: No tests found for module {test_file}')

        for cassette in sorted(cassette_names):
            if cassette in expected_cassettes:
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
