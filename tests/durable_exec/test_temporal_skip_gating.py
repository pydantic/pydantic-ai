"""The Temporal suite has to skip cleanly when a requirement is missing, however it is invoked.

`tests/durable_exec/temporal/` gates itself on `temporalio`, `logfire`, `mcp`, `openai` and Python
< 3.14 with module-level `pytest.skip(..., allow_module_level=True)` calls, which belong in the test
modules only. pytest loads the conftest of every command-line argument's directory up front, from
`PytestPluginManager._set_initial_conftests`, and `_importconftest` catches only `Exception` while
`Skipped` is a `BaseException` — so a gate in `conftest.py` turns `pytest tests/durable_exec/temporal`
into a traceback with exit 1 and no test report. Naming a parent directory hides it, because the
temporal conftest is then imported during collection, which does handle `Skipped`.

`test-durable-exec` names the parent, which is why CI stayed green. `test-temporal-latest` does not
— it passes node ids under this directory, and `_set_initial_conftests` strips `::` and loads the
containing directory's conftest — so it takes the crashing path and passes only because its install
happens to satisfy every gate.

This file lives one directory above the suite it guards: the subprocess below collects
`tests/durable_exec/temporal`, and a gate-less test inside that directory would be collected by its
own child and recurse.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[2]

# Installed before pytest imports anything, so it stands in for an environment that never got the
# `temporal` extra. `temporalio` is the one requirement that can be masked this way: it registers no
# `pytest11` entry point, whereas masking `logfire` makes pytest's own
# `load_setuptools_entrypoints('pytest11')` raise before it can report anything. So this covers a
# re-added `temporalio` or version gate, and — through the shared conftest-loading path — the shape
# of all five; it does not cover a module-level `import logfire` / `mcp` / `openai` sneaking back in,
# which the `--all-extras` install this suite runs under would satisfy anyway.
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
        timeout=300,
    )

    output = result.stdout + result.stderr
    # Every test module skipped, so nothing is collected and pytest's exit code says so rather than
    # `OK`. It is still the distinguishing signal: a conftest that raises exits 1, not 5.
    assert result.returncode == pytest.ExitCode.NO_TESTS_COLLECTED, output
    assert output.count('temporal not installed') == 4, output
    assert 'Skipped: ' not in output
