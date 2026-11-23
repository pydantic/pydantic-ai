"""The tests no CI cell runs, and why, for `.github/scripts/never_ran.py`.

See "Tests that run in no cell" in `tests/AGENTS.md`.
"""

from __future__ import annotations as _annotations

from .manifests import Exempt

NEVER_RAN_EXEMPT: dict[str, Exempt] = {}
"""Each test, test file, or directory (ending in `/`) whose tests every CI cell skips, with the reason none runs them."""

PATH_GATED_JOBS: dict[str, tuple[str, ...]] = {
    'harness-bubblewrap-integration': ('tests/harness/bubblewrap_sandbox/',),
    'test-clai2-clipboard': ('tests/pax/test_image_input.py', 'tests/pax/test_image_clipboard_native.py'),
}
"""The tests that only a job `ci.yml` runs for some changes runs, by job: exempt on a run that skipped that job."""
