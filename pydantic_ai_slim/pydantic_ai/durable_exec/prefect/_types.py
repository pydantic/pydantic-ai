from typing import Any

from typing_extensions import TypedDict


class TaskConfig(TypedDict, total=False):
    """Configuration for a task in Prefect.

    These options are passed to the `@task` decorator.
    """

    retries: int
    """Maximum number of retries for the task."""

    retry_delay_seconds: float | list[float]
    """Delay between retries in seconds. Can be a single value or a list for exponential backoff."""

    timeout_seconds: float
    """Maximum time in seconds for the task to complete."""

    cache_policy: Any
    """Prefect cache policy for the task."""

    persist_result: bool
    """Whether to persist the task result."""

    log_prints: bool
    """Whether to log print statements from the task."""
