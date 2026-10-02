"""Shared setup for every test that starts a Temporal dev server with `WorkflowEnvironment.start_local`.

Modules that define Temporal workflows import this at module level, so the workflow sandbox
re-imports it: keep its imports to the standard library.
"""

from __future__ import annotations

from pathlib import Path


def temporal_dev_server_cache_dir() -> str:
    """The directory to pass as `start_local(download_dest_dir=...)`.

    `start_local` downloads the dev-server binary to the system temp dir by default, which is empty on
    every CI run, so a CDN hiccup fails the test at setup (#5399). The `test-durable-exec` job restores
    this directory with `actions/cache`, and local runs reuse it across reboots. Resolved at call time
    rather than at module level, because the workflow sandbox restricts `Path.home()`.
    """
    path = Path.home() / '.cache' / 'temporal-dev-server'
    path.mkdir(parents=True, exist_ok=True)
    return str(path)
