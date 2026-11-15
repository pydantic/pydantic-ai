"""Markers for harness test modules that define Temporal workflows.

Temporal's workflow sandbox re-executes the module a workflow is defined in, together with every
module it imports that isn't passed through. `tests/harness/conftest.py` imports `dirty_equals` and
other sandbox-unsafe modules, so importing these markers from there made the sandbox re-import
`dirty_equals`, which can collide with the host's copy in `typing`'s shared generic cache and fail
the workflow (https://github.com/pydantic/pydantic-ai/issues/9337).

Keep this module's imports to what the workflow test modules already import themselves.
"""

from __future__ import annotations

import sys

import pytest

__all__ = ('ignore_source_reads_left_open', 'skip_temporal_sandbox_on_314')

skip_temporal_sandbox_on_314 = pytest.mark.skipif(
    sys.version_info >= (3, 14),
    reason='temporalio sandbox is incompatible with Python 3.14 '
    '(remove when https://github.com/temporalio/sdk-python/issues/1326 closes)',
)
"""Same gate as core's Temporal suite: the sandbox fails with late-import errors on 3.14."""

# On 3.14 coverage reads a module's source while the test runs; when Temporal's workflow sandbox
# interrupts that read, the file is left for the garbage collector to close. Only `.py` files match.
ignore_source_reads_left_open = pytest.mark.filterwarnings(
    "ignore:unclosed file <_io.BufferedReader name='[^']*\\.py'>:ResourceWarning"
)
