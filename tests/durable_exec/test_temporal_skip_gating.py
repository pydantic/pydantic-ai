"""The Temporal suite has to skip cleanly when a requirement is missing, however it is invoked.

`tests/durable_exec/temporal/` gates itself on `temporalio`, `logfire`, `mcp`, `openai` and Python
< 3.14 with module-level `pytest.skip(..., allow_module_level=True)` calls, which belong in the test
modules only. pytest imports the conftest of every command-line argument's directory up front, in
`PytestPluginManager._set_initial_conftests`, and catches only `Exception` there while `Skipped` is a
`BaseException` — so a gate in `conftest.py` turns `pytest tests/durable_exec/temporal` into a
traceback with exit 1 and no test report. Naming a parent directory hides it, because the temporal
conftest is then imported during collection, which does handle `Skipped`; CI names the parent, so
only the targeted command regresses.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[2]

# Installed before pytest imports anything, so it stands in for an environment that never got the
# `temporal` extra. Masking `temporalio` alone is enough: it is the suite's first gate, and every
# other gate reaches the conftest by the same path.
MASK_TEMPORALIO = """
import sys


class MaskTemporalio:
    def find_spec(self, name, path=None, target=None):
        if name == 'temporalio' or name.startswith('temporalio.'):
            raise ImportError('No module named ' + repr(name))
        return None


sys.meta_path.insert(0, MaskTemporalio())
"""


def test_temporal_suite_skips_when_invoked_on_its_own_directory() -> None:
    """`pytest tests/durable_exec/temporal` reports skips instead of crashing on the conftest import.

    Run in a subprocess because the failure mode is pytest's own start-up sequence: the conftest is
    imported from `pytest_load_initial_conftests`, before any hook this process could install.
    """
    # `-p no:pretty` because pytest-pretty replaces the terminal reporter's summary and drops the
    # `-rs` skip-reason lines this test reads.
    args = [
        'tests/durable_exec/temporal',
        '-rs',
        '--no-header',
        '-p',
        'no:randomly',
        '-p',
        'no:cacheprovider',
        '-p',
        'no:pretty',
    ]
    result = subprocess.run(
        [sys.executable, '-c', f'{MASK_TEMPORALIO}\nimport pytest\nraise SystemExit(pytest.main({args!r}))'],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    output = result.stdout + result.stderr
    # Every test module skipped, so nothing is collected and pytest's exit code says so rather than
    # `OK`. What matters is that each module reported its own skip reason and nothing raised.
    assert result.returncode == pytest.ExitCode.NO_TESTS_COLLECTED, output
    assert 'temporal not installed' in output
    assert 'Skipped: ' not in output
