"""One `run_workflow` call: its sandbox sessions, the host functions its scripts call, and nesting.

A `WorkflowRun` lives for one tool call. The script, and every saved workflow it runs inline
through `workflow()`, share its sub-agent budget, concurrency cap, print capture and Monty pool.
Each nested workflow runs in a session of its own, checked out while its caller is suspended.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Generic, cast

from opentelemetry.trace import Span
from pydantic import JsonValue, TypeAdapter, ValidationError
from pydantic_core import to_jsonable_python

from pydantic_ai import RunContext, StructuredDict
from pydantic_ai.exceptions import UsageLimitExceeded, UserError
from pydantic_ai.models import Model
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.usage import UsageLimits

try:
    from pydantic_monty import (
        AsyncMontySession,
        MontyComplete,
        MontyCrashedError,
        MontyRuntimeError,
        MontySyntaxError,
        MontyTypingError,
        ResourceLimits,
    )
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'pydantic-monty is required for DynamicWorkflow. '
        'Install it with: uv add "pydantic-ai-harness[dynamic-workflow]"'
    ) from _import_error

from pydantic_ai_harness._monty_exec import MontyExecutor, MontyRunState, PrintCapture, call_monty, is_sandbox_panic
from pydantic_ai_harness.dynamic_workflow._catalog import WorkflowAgent
from pydantic_ai_harness.dynamic_workflow._events import WorkflowLogEvent, WorkflowPhaseEvent
from pydantic_ai_harness.dynamic_workflow._library import SavedWorkflow, WorkflowLibrary
from pydantic_ai_harness.dynamic_workflow._prelude import (
    ARGS_NAME,
    HOST_AGENT,
    HOST_ASYNC_NAMES,
    HOST_BUDGET,
    HOST_INLINE_NAMES,
    HOST_LOG,
    HOST_PHASE,
    HOST_WORKFLOW,
)
from pydantic_ai_harness.dynamic_workflow._results import BudgetExhausted, CompletedDispatch

# Set while a workflow script is executing, so a sub-agent that itself tries to run a workflow can
# be refused: a sub-agent does not orchestrate. asyncio copies the context into each task
# `asyncio.gather` schedules, so concurrently-dispatched sub-agents inherit this flag (the capability
# is asyncio-only). Saved workflows nest through `workflow()`, which never goes through the tool.
in_workflow: contextvars.ContextVar[bool] = contextvars.ContextVar('pydantic_ai_harness_in_workflow', default=False)

_MODEL_SAFE_EXCEPTION_MESSAGE_TYPES = (UsageLimitExceeded,)
_JSON_OBJECT_ADAPTER: TypeAdapter[dict[str, JsonValue]] = TypeAdapter(dict[str, JsonValue])


@dataclass
class CallBudget:
    """The sub-agent calls one agent run may make, shared by every `run_workflow` call in it."""

    max_calls: int
    used: int = 0


@dataclass(frozen=True)
class Frame:
    """Where a script runs: the saved workflow it is, and the saved workflows it runs inside of."""

    stack: tuple[str, ...] = ()

    @property
    def workflow(self) -> str | None:
        """The saved workflow this script is, or `None` for an ad-hoc script."""
        return self.stack[-1] if self.stack else None


@dataclass(kw_only=True)
class WorkflowRun(Generic[AgentDepsT]):
    """State for one `run_workflow` call and every script it runs."""

    ctx: RunContext[AgentDepsT]
    agents: Mapping[str, WorkflowAgent[AgentDepsT]]
    library: WorkflowLibrary | None
    budget: CallBudget
    default_agent: str | None
    max_concurrent_agents: int
    max_workflow_depth: int
    forward_usage: bool
    inherit_model: bool
    sub_agent_usage_limits: UsageLimits | None
    prelude: str
    stubs: str
    bind_args: bool
    limits: ResourceLimits
    in_temporal: bool

    monty: MontyRunState = field(default_factory=MontyRunState)
    capture: PrintCapture = field(default_factory=PrintCapture)
    completed: list[CompletedDispatch] = field(default_factory=list[CompletedDispatch])
    budget_exhausted: bool = False
    _agent_slots: asyncio.Semaphore = field(init=False)
    _session_slots: dict[int, asyncio.Semaphore] = field(default_factory=dict[int, asyncio.Semaphore])

    def __post_init__(self) -> None:
        self._agent_slots = asyncio.Semaphore(self.max_concurrent_agents)

    # --- Running scripts ---

    async def main_session(self) -> AsyncMontySession:
        """The call's own session, where the tool's script or named workflow runs."""
        return await self.monty.get_session(
            type_check=True, type_check_stubs=self.stubs, limits=self.limits, in_temporal_workflow=self.in_temporal
        )

    async def execute(
        self, session: AsyncMontySession, code: str, *, frame: Frame, args: dict[str, JsonValue]
    ) -> MontyComplete:
        """Feed the helpers, then run `code` in `session`, answering its host calls, until it completes."""
        prelude = await call_monty(self.monty.portal, partial(session.feed_start, self.prelude, skip_type_check=True))
        assert isinstance(prelude, MontyComplete), 'the prelude only defines functions'
        # `_by_name` is not mutated while a script executes (reveals land in `get_tools`, which does
        # not interleave with `call_tool`), so the catalog is stable for the whole call.
        return await MontyExecutor(
            dispatch=partial(self._dispatch, frame),
            valid_names={*self.agents, *HOST_ASYNC_NAMES, *HOST_INLINE_NAMES},
            inline_names=HOST_INLINE_NAMES,
            portal=self.monty.portal,
            max_sleep_secs=self.limits.get('max_feed_duration_secs'),
        ).run(
            partial(
                session.feed_start,
                code,
                inputs={ARGS_NAME: args} if self.bind_args else None,
                print_callback=self.capture.callback,
            )
        )

    def enter_saved(
        self, name: object, args: object, frame: Frame
    ) -> tuple[SavedWorkflow, dict[str, JsonValue], Frame]:
        """Resolve a saved workflow to run inside `frame`, raising `ValueError` saying why it cannot run."""
        if not isinstance(name, str):
            raise TypeError(f'workflow name must be a string, got {type(name).__name__}')
        workflows = self.library.workflows if self.library is not None else {}
        workflow = workflows.get(name)
        if workflow is None:
            available = ', '.join(sorted(workflows)) or 'none are saved'
            raise ValueError(f'unknown saved workflow {name!r}; available: {available}')
        if name in frame.stack:
            raise ValueError(f'workflow {name!r} would call itself: {" -> ".join([*frame.stack, name])}')
        if len(frame.stack) >= self.max_workflow_depth:
            raise ValueError(f'workflow {name!r} would nest deeper than {self.max_workflow_depth} saved workflows')
        try:
            checked = _JSON_OBJECT_ADAPTER.validate_python({} if args is None else args)
        except ValidationError as error:
            raise ValueError(f'workflow {name!r} args must be a JSON object') from error
        if missing := workflow.missing_args(checked):
            raise ValueError(f'workflow {name!r} is missing required args: {", ".join(missing)}')
        return workflow, checked, Frame((*frame.stack, name))

    @contextmanager
    def workflow_span(self, frame: Frame, args: dict[str, JsonValue]) -> Generator[Span]:
        """A `dynamic_workflow.workflow` span around one saved workflow's run."""
        attributes: dict[str, str | int] = {
            'dynamic_workflow.name': frame.workflow or '',
            'dynamic_workflow.depth': len(frame.stack),
        }
        if self.ctx.trace_include_content:
            attributes['dynamic_workflow.args'] = json.dumps(args)
        with self.ctx.tracer.start_as_current_span('dynamic_workflow.workflow', attributes=attributes) as span:
            outcome = 'error'
            try:
                yield span
                outcome = 'ok'
            finally:
                span.set_attribute('dynamic_workflow.outcome', 'budget_exhausted' if self.budget_exhausted else outcome)

    async def _run_nested(self, name: object, args: object, frame: Frame) -> object:
        """Run a saved workflow inline for `workflow()`, in a session of its own."""
        workflow, checked, child = self.enter_saved(name, args, frame)
        # One pool of sessions per depth: a session only ever waits for a deeper one, so a full level
        # cannot deadlock the levels above it.
        slots = self._session_slots.setdefault(len(child.stack), asyncio.Semaphore(self.max_concurrent_agents))
        async with slots:
            with self.workflow_span(child, checked):
                try:
                    async with self.monty.extra_session(
                        type_check=True, type_check_stubs=self.stubs, limits=self.limits
                    ) as session:
                        return (await self.execute(session, workflow.source, frame=child, args=checked)).output
                except (MontyTypingError, MontySyntaxError, MontyRuntimeError) as error:
                    raise RuntimeError(f'workflow {workflow.name!r} failed:\n{error.display()}') from error
                except MontyCrashedError as error:
                    raise RuntimeError(f'workflow {workflow.name!r} crashed the sandbox worker') from error
                except BaseException as error:
                    if not is_sandbox_panic(error):
                        raise
                    raise RuntimeError(f'workflow {workflow.name!r} aborted inside the sandbox') from error

    # --- Host functions ---

    async def _dispatch(self, frame: Frame, name: str, kwargs: dict[str, object]) -> object:
        """Answer one host call from a script running in `frame`."""
        if name == HOST_AGENT:
            return await self._agent_call(frame, kwargs)
        if name == HOST_WORKFLOW:
            return await self._run_nested(kwargs.get('name'), kwargs.get('args'), frame)
        if name == HOST_LOG:
            await self.ctx.emit(WorkflowLogEvent(message=str(kwargs.get('message')), workflow=frame.workflow))
            return None
        if name == HOST_PHASE:
            await self.ctx.emit(WorkflowPhaseEvent(title=str(kwargs.get('title')), workflow=frame.workflow))
            return None
        if name == HOST_BUDGET:
            used = self.budget.used
            return {'max': self.budget.max_calls, 'used': used, 'remaining': self.budget.max_calls - used}
        return await self._catalog_call(name, kwargs)

    async def _catalog_call(self, agent_name: str, kwargs: dict[str, object]) -> object:
        """A sub-agent called by its own name: `reviewer(task=...)`."""
        # The sandbox signature is `(*, task: str)`, but Monty does not validate kwargs against it at
        # runtime, and the static check can be evaded through `Any` (e.g. `json.loads` results) -- so
        # check here: a dropped extra kwarg or a non-string `task` would otherwise run the sub-agent
        # on silently-wrong input. Each raises before the budget is touched.
        if 'task' not in kwargs:
            raise TypeError(f'{agent_name}() missing required keyword argument: task')
        extra = sorted(set(kwargs) - {'task'})
        if extra:
            raise TypeError(f'{agent_name}() got unexpected keyword argument(s): {", ".join(extra)}; only task')
        task = kwargs['task']
        if not isinstance(task, str):
            raise TypeError(f'{agent_name}() task must be a string, got {type(task).__name__}')
        return await self._run_agent(agent_name, task)

    async def _agent_call(self, frame: Frame, kwargs: dict[str, object]) -> object:
        """A sub-agent called through the `agent()` helper."""
        task = kwargs.get('task')
        if not isinstance(task, str):
            raise TypeError(f'agent() task must be a string, got {type(task).__name__}')
        name, model, phase = (_optional_str(kwargs, key) for key in ('name', 'model', 'phase'))
        agent_name = self._resolve_agent_name(name)
        schema = kwargs.get('schema')
        try:
            output_type = None if schema is None else StructuredDict(_JSON_OBJECT_ADAPTER.validate_python(schema))
        except (ValidationError, UserError) as error:
            raise ValueError(f'agent() schema must be a JSON schema of type "object": {error}') from error
        if phase is not None:
            await self.ctx.emit(WorkflowPhaseEvent(title=phase, workflow=frame.workflow))
        return await self._run_agent(agent_name, task, output_type=output_type, model=model)

    def _resolve_agent_name(self, name: str | None) -> str:
        if name is None:
            name = self.default_agent
        if name is None and len(self.agents) == 1:
            name = next(iter(self.agents))
        if name is None:
            raise ValueError(f'agent() needs name=, one of: {", ".join(self.agents)}')
        if name not in self.agents:
            raise ValueError(f'agent() got unknown name {name!r}; available: {", ".join(self.agents)}')
        return name

    async def _run_agent(
        self,
        agent_name: str,
        task: str,
        *,
        output_type: type[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> object:
        """Run one sub-agent against the shared per-run budget and the concurrency cap.

        The budget check + increment must stay suspension-free: there must be no `await`
        between them. asyncio only switches tasks at suspension points, so an await-free
        check-then-increment is atomic across the concurrently-gathered dispatches, which
        is what makes `max_agent_calls` an exact ceiling under fan-out. The concurrency slot
        is acquired only after the increment, so a call waiting for a slot has its budget.

        This exists precisely because `usage_limits` cannot give an exact ceiling here:
        core's own limit check is split from its increment by the model-request `await`
        (a TOCTOU race -- N gathered sub-agents all pass the check before any increments;
        measured ~20x overshoot), and `RunContext` exposes `usage` but not `usage_limits`,
        so the parent's configured limit can't be forwarded to sub-agents at all.
        """
        if self.budget.used >= self.budget.max_calls:
            if not self.budget_exhausted:
                self.ctx.tracer.start_span(
                    'dynamic_workflow.budget_exhausted',
                    attributes={'dynamic_workflow.max_agent_calls': self.budget.max_calls},
                ).end()
            self.budget_exhausted = True
            raise BudgetExhausted(self.budget.max_calls)
        self.budget.used += 1
        # `ctx.model` is an `AbstractModel`; only a request-response `Model` can drive a sub-agent
        # run. A realtime run's model isn't one, so fall back to the sub-agent's own default rather
        # than forwarding a model it cannot run with. An explicit `model=` from the script wins.
        ctx_model = self.ctx.model
        run_model: Model[Any] | str | None = model
        if run_model is None and self.inherit_model and isinstance(ctx_model, Model):
            run_model = cast('Model[Any]', ctx_model)
        async with self._agent_slots:
            try:
                result = await self.agents[agent_name].agent.run(
                    task,
                    output_type=output_type,
                    deps=self.ctx.deps,
                    model=run_model,
                    usage=self.ctx.usage if self.forward_usage else None,
                    usage_limits=self.sub_agent_usage_limits,
                )
                output: object = to_jsonable_python(result.output)
            except Exception as exc:
                # Don't leak host internals (file paths, deps/agent reprs) to the model;
                # surface the failing agent and error type by default.
                message = f'sub-agent {agent_name!r} raised {type(exc).__name__}'
                if isinstance(exc, _MODEL_SAFE_EXCEPTION_MESSAGE_TYPES):
                    message = f'{message}: {exc}'
                raise RuntimeError(message) from exc
        self.completed.append(CompletedDispatch(agent_name=agent_name, task=task, result=output))
        return output


def _optional_str(kwargs: dict[str, object], key: str) -> str | None:
    value = kwargs.get(key)
    if value is not None and not isinstance(value, str):
        raise TypeError(f'agent() {key} must be a string, got {type(value).__name__}')
    return value
