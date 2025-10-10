from __future__ import annotations

from prefect.cache_policies import CachePolicy
from prefect.results import ResultStorage
from typing_extensions import TypedDict

from pydantic_ai.durable_exec.prefect._cache_policies import DEFAULT_PYDANTIC_AI_CACHE_POLICY


class TaskConfig(TypedDict, total=False):
    """Configuration for a task in Prefect.

    These options are passed to the `@task` decorator.
    """

    retries: int
    """Maximum number of retries for the task."""

    retry_delay_seconds: float | list[float]
    """Delay between retries in seconds. Can be a single value or a list for custom backoff."""

    timeout_seconds: float
    """Maximum time in seconds for the task to complete."""

    cache_policy: CachePolicy
    """Prefect cache policy for the task."""

    persist_result: bool
    """Whether to persist the task result."""

    result_storage: ResultStorage
    """Prefect result storage for the task. Should be a storage block or a block slug like `s3-bucket/my-storage`."""

    log_prints: bool
    """Whether to log print statements from the task."""


default_task_config = TaskConfig(
    retries=0,
    retry_delay_seconds=1.0,
    persist_result=True,
    log_prints=False,
    cache_policy=DEFAULT_PYDANTIC_AI_CACHE_POLICY,
)
