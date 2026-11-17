from __future__ import annotations

from concurrent.futures import Executor
from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING, Any

from pydantic_ai import _utils
from pydantic_ai.tools import AgentDepsT, RunContext

from .abstract import AbstractCapability, WrapRunHandler

if TYPE_CHECKING:
    from pydantic_ai.run import AgentRunResult


@dataclass
class UseThreadExecutor(AbstractCapability[Any]):
    """Use a custom executor for running sync functions in threads.

    By default, sync tool functions and other sync callbacks are run in threads using
    [`anyio.to_thread.run_sync`][anyio.to_thread.run_sync], which creates ephemeral threads.
    In long-running servers (e.g. FastAPI), this can lead to thread accumulation under sustained load.

    This capability provides a bounded [`ThreadPoolExecutor`][concurrent.futures.ThreadPoolExecutor]
    (or any [`Executor`][concurrent.futures.Executor]) to use instead, scoped to agent runs:

    ```python
    from concurrent.futures import ThreadPoolExecutor

    from pydantic_ai import Agent
    from pydantic_ai.capabilities import UseThreadExecutor

    executor = ThreadPoolExecutor(max_workers=16, thread_name_prefix='agent-worker')
    agent = Agent('openai:gpt-5.2', capabilities=[UseThreadExecutor(executor)])
    ```

    To set an executor for all agents globally, use
    [`Agent.using_thread_executor()`][pydantic_ai.agent.AbstractAgent.using_thread_executor].
    """

    executor: Executor
    """The executor to use for running sync functions."""

    _: KW_ONLY

    id: str | None = 'use_thread_executor'
    """One-off: exactly one executor is in effect for a run, so the id is fixed by default.

    `wrap_run` sets a context variable, so a second one nested inside the first shadows it and the
    outer executor is never used. Naming them the same makes that resolution explicit rather than an
    accident of nesting order.
    """

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None

    async def wrap_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        handler: WrapRunHandler,
    ) -> AgentRunResult[Any]:
        with _utils.using_thread_executor(self.executor):
            return await handler()


# TODO(v3): remove the `ThreadExecutor` alias, this `__getattr__`, and the forwarding one in `capabilities/__init__.py`
def __getattr__(name: str) -> object:
    if name == 'ThreadExecutor':
        import warnings

        from pydantic_ai._warnings import PydanticAIDeprecationWarning

        warnings.warn(
            '`ThreadExecutor` has been renamed to `UseThreadExecutor`. '
            'Update your imports; this deprecated alias will be removed in a future release.',
            PydanticAIDeprecationWarning,
            stacklevel=2,
        )
        return UseThreadExecutor
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
