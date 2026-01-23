#!/usr/bin/env python
"""Check that all cassette files have corresponding tests.

This script verifies that every VCR cassette file in the test suite has a
corresponding test function. Orphaned cassettes (cassettes without tests)
indicate dead code that should be removed.

Usage:
    python scripts/check_cassettes.py [--verbose]
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import pytest


class _CollectVcrTests:
    """Pytest plugin that collects cassette names referenced by VCR-marked tests.

    This is a class (not functions) because pytest's plugin system requires objects
    with hook methods, and we need to accumulate state across all test items.
    """

    def __init__(self) -> None:
        self.tests: dict[str, set[str]] = defaultdict(set)

    @staticmethod
    def _remove_yaml_ext(s: str) -> str:
        if s.endswith('.yaml'):
            return s[:-5]
        return s

    def pytest_collection_modifyitems(
        self, session: pytest.Session, config: pytest.Config, items: list[pytest.Item]
    ) -> None:
        # prevents pytest.PytestAssertRewriteWarning: Module already imported so cannot be rewritten; pytest_recording
        from pytest_recording.plugin import get_default_cassette_name

        for item in items:
            if not any(item.iter_markers('vcr')):
                continue

            test_file_stem = Path(item.location[0]).stem

            m = item.get_closest_marker('default_cassette')
            if m and m.args:
                self.tests[test_file_stem].add(self._remove_yaml_ext(m.args[0]))
            else:
                self.tests[test_file_stem].add(
                    self._remove_yaml_ext(get_default_cassette_name(getattr(item, 'cls', None), item.name))
                )

            for vm in item.iter_markers('vcr'):
                for arg in vm.args:
                    self.tests[test_file_stem].add(self._remove_yaml_ext(arg))


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
    """Use pytest collection to get all VCR-marked tests and their cassette names."""
    collector = _CollectVcrTests()
    rc = pytest.main(['--collect-only', '-q', 'tests/'], plugins=[collector])
    if rc not in (pytest.ExitCode.OK, pytest.ExitCode.NO_TESTS_COLLECTED):
        raise SystemExit(rc)
    return dict(collector.tests)


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
