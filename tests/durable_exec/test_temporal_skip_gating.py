"""The Temporal suite has to skip cleanly when a requirement is missing, however it is invoked.

`tests/durable_exec/temporal/` gates itself on `temporalio`, `logfire`, `mcp`, `openai` and Python
< 3.14 with module-level `pytest.skip(..., allow_module_level=True)` calls, which belong in the test
modules only. pytest loads the conftest of every command-line argument's directory up front, from
`PytestPluginManager._set_initial_conftests`, and `_importconftest` catches only `Exception` while
`Skipped` is a `BaseException` — so a gate in `conftest.py` turns `pytest tests/durable_exec/temporal`
into a traceback with exit 1 and no test report. Naming a parent directory hides it, because the
temporal conftest is then imported during collection, which does handle `Skipped`.

Both invocations that reach the conftest up front are pinned below. Neither exits 0 — nothing is
collected, so pytest reports `NO_TESTS_COLLECTED` for a directory and `USAGE_ERROR` for a node id
that a skipped module can no longer supply. What the fix buys is a skip report instead of a
traceback, which is what these tests assert.

`test-durable-exec` names the parent directory, which is why CI stayed green. `test-temporal-latest`
does not — it passes node ids under this directory, and `_set_initial_conftests` strips `::` and
loads the containing directory's conftest — so it takes the crashing path and passes only because
its install happens to satisfy every gate.

This file lives one directory above the suite it guards: the subprocess below collects
`tests/durable_exec/temporal`, and a gate-less test inside that directory would be collected by its
own child and recurse.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[2]
TEMPORAL_SUITE = REPO_ROOT / 'tests' / 'durable_exec' / 'temporal'

# Installed before pytest imports anything, so it stands in for an environment that never got the
# `temporal` extra. `temporalio` is the one requirement that can be masked this way: it registers no
# `pytest11` entry point, whereas masking `logfire` makes pytest's own
# `load_setuptools_entrypoints('pytest11')` raise before it can report anything. So this covers a
# re-added `temporalio` or version gate, and — through the shared conftest-loading path — the shape
# of all five; it does not cover a module-level `import logfire` / `mcp` / `openai` sneaking back in,
# which the `--all-extras` install this suite runs under would satisfy anyway.
_BLOCK_TEMPORALIO = """
import sys


class BlockTemporalio:
    def find_spec(self, name, path=None, target=None):
        if name == 'temporalio' or name.startswith('temporalio.'):
            raise ImportError('No module named ' + repr(name))
        return None


sys.meta_path.insert(0, BlockTemporalio())
"""


def _run_masked_pytest(target: str) -> subprocess.CompletedProcess[str]:
    """Collect `target` in a subprocess that cannot import `temporalio`.

    A subprocess because the failure mode is pytest's own start-up sequence: the conftest is
    imported from `pytest_load_initial_conftests`, before any hook an in-process test could install.

    `COVERAGE_*` is scrubbed from the child's environment, matching
    `tests/test_public_interface_contracts.py` and `tests/models/test_openai.py`. With
    `patch = ["subprocess"]` the child would otherwise start coverage and pay a full instrumented
    pytest start-up, and every line it reaches past the mask is a `pragma: lax no cover` import
    guard, so the data it produces is worth nothing and risks colliding with the strict pragma audit.
    """
    # `-p no:pretty` because pytest-pretty replaces the terminal reporter's summary and drops the
    # `-rs` skip-reason lines these tests read.
    args = [target, '-rs', '--no-header', '-p', 'no:randomly', '-p', 'no:cacheprovider', '-p', 'no:pretty']
    return subprocess.run(
        [sys.executable, '-c', f'{_BLOCK_TEMPORALIO}\nimport pytest\nraise SystemExit(pytest.main({args!r}))'],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        env={key: value for key, value in os.environ.items() if not key.startswith('COVERAGE_')},
    )


def test_temporal_suite_skips_when_invoked_on_its_own_directory() -> None:
    """`pytest tests/durable_exec/temporal` reports skips instead of crashing on the conftest import."""
    result = _run_masked_pytest('tests/durable_exec/temporal')

    output = result.stdout + result.stderr
    # Every test module skipped, so nothing is collected and pytest's exit code says so rather than
    # `OK`. It is still the distinguishing signal: a conftest that raises exits 1, not 5.
    assert result.returncode == pytest.ExitCode.NO_TESTS_COLLECTED, output
    # One skip line per collected test module — the whole suite gated out, not just the first file.
    # Derived rather than hardcoded so adding a correctly-gated module doesn't fail this test.
    # `_shared.py` carries the same gate but is never collected, so it contributes no line.
    assert output.count('temporal not installed') == len(list(TEMPORAL_SUITE.glob('test_*.py'))), output
    # `Skipped: ` is the traceback prefix of an escaped `Skipped`, not anything `-rs` prints (its
    # short-summary form is `SKIPPED [1] <file>:<line>: <reason>`).
    assert 'Skipped: ' not in output, output


def test_temporal_suite_skips_when_invoked_on_a_node_id() -> None:
    """A node id under the suite reports the skip too — the form `test-temporal-latest` runs.

    `_set_initial_conftests` strips `::` and loads the containing directory's conftest, so this
    invocation reaches it exactly as the bare directory does. The skipped module can no longer
    supply the node id, so pytest adds `USAGE_ERROR` on top of the skip report; that is a resolved
    argument, not the unhandled `Skipped` this suite regressed on.
    """
    result = _run_masked_pytest(
        'tests/durable_exec/temporal/test_durability.py::test_durability_coerces_activity_config_values'
    )

    output = result.stdout + result.stderr
    assert result.returncode == pytest.ExitCode.USAGE_ERROR, output
    assert 'temporal not installed' in output
    assert 'Skipped: ' not in output, output
