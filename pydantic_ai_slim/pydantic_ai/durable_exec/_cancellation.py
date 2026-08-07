"""Portable, engine-agnostic seam for externally cancelling a durable agent run.

First-party run cancellation (`AgentRun.cancel()`, `RunContext.cancel()`,
[`CancellationToken`][pydantic_ai.CancellationToken]) has no story across a durable-execution
boundary: a `CancellationToken` is a same-process, cross-thread handle that can't be serialized
into a workflow/flow, and a durable run exposes no `AgentRun` for an external actor (a user hitting
"stop") to reach. Cancelling the whole workflow works but is all-or-nothing and doesn't produce a
first-party "the run was cancelled" outcome.

[`DurableRunCancellation`][pydantic_ai.durable_exec.DurableRunCancellation] bridges that gap with a
per-run capability. It captures the run's [`RunCancellation`][pydantic_ai._cancel.RunCancellation]
controller in `before_run`, and its [`cancel`][pydantic_ai.durable_exec.DurableRunCancellation.cancel]
method triggers it. Each durable engine wires its own external-cancellation mechanism to that one
method — for Temporal a `@workflow.signal` handler (a signal runs on the workflow event loop and is
recorded in history, so the resulting cancellation is deterministic on replay) — while the binding
between the trigger and the run stays engine-agnostic here.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

from ..capabilities.abstract import AbstractCapability
from ..tools import AgentDepsT, RunContext

if TYPE_CHECKING:
    from .._cancel import RunCancellation


@dataclass
class DurableRunCancellation(AbstractCapability[AgentDepsT]):
    """A per-run handle for cancelling a durable agent run from outside the run.

    Pass a single instance to one durable `run(capabilities=[...])` call, then call
    [`cancel`][pydantic_ai.durable_exec.DurableRunCancellation.cancel] from the engine's
    external-cancellation handler to cancel *that* run first-party: it ends with
    [`RunCancelled`][pydantic_ai.exceptions.RunCancelled] rather than by tearing down the
    whole workflow or flow.

    For Temporal, wire `cancel()` to a [`@workflow.signal`](https://docs.temporal.io/develop/python/message-passing#signals)
    handler:

    ```python {test="skip"}
    from temporalio import workflow

    from pydantic_ai.durable_exec import DurableRunCancellation

    with workflow.unsafe.imports_passed_through():
        from my_agent import my_temporal_agent


    @workflow.defn
    class MyAgentWorkflow:
        def __init__(self) -> None:
            # A fresh handle per workflow execution; it binds to this run only.
            self.cancellation = DurableRunCancellation()

        @workflow.run
        async def run(self, prompt: str) -> str:
            result = await my_temporal_agent.run(prompt, capabilities=[self.cancellation])
            return result.output

        @workflow.signal
        def cancel(self) -> None:
            self.cancellation.cancel()
    ```

    An external actor then cancels the run by signalling the workflow
    (`await handle.signal(MyAgentWorkflow.cancel)`). DBOS and Prefect bind their own
    external-cancellation mechanisms to the same `cancel()` method.

    A single instance binds to a single run; create a fresh one per durable execution (e.g. in the
    workflow's `__init__`), not a module-level singleton shared across concurrent runs.
    """

    _safe_at_runtime: ClassVar[bool] = True
    """Workflow-side only — introduces no toolsets, native tools, or model wrapping — so it is safe
    to attach per-run even when a durability capability is bound. Internal flag read by the bundled
    durable-execution integrations."""

    _cancellation: RunCancellation | None = field(default=None, init=False, repr=False)
    _cancel_requested: bool = field(default=False, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    @classmethod
    def get_serialization_name(cls) -> str | None:
        # Holds a live, run-scoped controller reference, so it is not spec-constructible.
        return None

    async def before_run(self, ctx: RunContext[AgentDepsT]) -> None:
        # Capture the run's cancellation controller so `cancel()` can trigger it. Read via
        # `__dict__` so a restricted run-context subclass (e.g. `TemporalRunContext`) whose
        # `__getattribute__` rejects absent fields returns `None` instead of raising. `before_run`
        # has no `await`, so it completes before any engine handler (e.g. a Temporal signal) can run
        # on the same loop: the controller is always captured before a `cancel()` can reach it.
        cancellation: RunCancellation | None = ctx.__dict__.get('_cancellation')
        with self._lock:
            self._cancellation = cancellation
            already_requested = self._cancel_requested
        if already_requested and cancellation is not None:
            # A `cancel()` that arrived before the run bound its controller is applied now.
            cancellation.cancel()

    def cancel(self) -> None:
        """Request first-party cancellation of the bound run.

        Idempotent, and safe to call before the run has started (the request is applied once the run
        binds its controller) or after it has finished (a no-op). `RunCancellation.cancel()` marshals
        the request onto the run's own event loop, so this is safe to call from the engine's
        external-cancellation handler.
        """
        with self._lock:
            self._cancel_requested = True
            cancellation = self._cancellation
        if cancellation is not None:
            cancellation.cancel()
