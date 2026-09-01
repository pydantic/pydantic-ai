from __future__ import annotations

from collections.abc import Iterator

import pytest


@pytest.fixture(scope='session')
def prefect_test_harness() -> Iterator[None]:
    """Run all Prefect integration tests against one isolated test server."""
    from prefect.settings import PREFECT_SERVER_SERVICES_TASK_RUN_RECORDER_ENABLED, temporary_settings
    from prefect.testing.utilities import prefect_test_harness

    # The task-run recorder is a background writer against the same sqlite file the flows write to.
    # Nothing in these tests reads what it records, and disabling it avoids lock contention in CI.
    with temporary_settings({PREFECT_SERVER_SERVICES_TASK_RUN_RECORDER_ENABLED: False}):
        with prefect_test_harness(server_startup_timeout=60):
            yield
