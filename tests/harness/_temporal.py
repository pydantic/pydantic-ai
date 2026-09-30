"""Temporal-specific harness test markers safe to import inside workflow sandboxes."""

from __future__ import annotations

import sys

import pytest

skip_temporal_sandbox_on_314 = pytest.mark.skipif(
    sys.version_info >= (3, 14),
    reason='temporalio sandbox is incompatible with Python 3.14 '
    '(remove when https://github.com/temporalio/sdk-python/issues/1326 closes)',
)
"""Same gate as core's Temporal suite: the sandbox fails with late-import errors on 3.14."""
