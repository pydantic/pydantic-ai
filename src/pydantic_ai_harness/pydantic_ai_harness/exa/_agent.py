"""Exa agent capability that delegates deep research tasks to the Exa Agent API as deferred tool calls."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from dataclasses import KW_ONLY, dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol
from uuid import uuid4

import httpx
from pydantic import BaseModel, ValidationError

from pydantic_ai.capabilities import AbstractCapability, durable_operation
from pydantic_ai.exceptions import CallDeferred, ModelRetry, UserError
from pydantic_ai.messages import ToolReturn
from pydantic_ai.tools import AgentDepsT, DeferredToolRequests, DeferredToolResults, RunContext
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai_harness._combine import one_per_id
from pydantic_ai_harness._durable import RetryRequest, raise_retry, retry_as_result
from pydantic_ai_harness.exa._toolset import (
    _AUTH_STATUS_RE,  # pyright: ignore[reportPrivateUsage]
    _recoverable,  # pyright: ignore[reportPrivateUsage]
    _source_list,  # pyright: ignore[reportPrivateUsage]
    _with_sources,  # pyright: ignore[reportPrivateUsage]
)

try:
    from exa_py import AsyncExa
    from exa_py.agent.types import AgentEffort, AgentEvent, AgentRun
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'exa-py is required for ExaAgent. Install it with: pip install "pydantic-ai-harness[exa]"'
    ) from _import_error

if TYPE_CHECKING:
    from pydantic_ai._instructions import AgentInstructions

RUN_ID_METADATA_KEY = 'exa_agent_run_id'
"""Key under which the Exa run ID is stored in a deferred call's metadata."""

_OWNER_METADATA_KEY = 'exa_agent_owner_id'
"""Key under which the owning capability's token is stored in a deferred call's metadata.

The inline resolver claims a deferred call by this token rather than by tool
name, so wrapper capabilities that rename or prefix tools (e.g. `PrefixTools`)
do not break resolution, and multiple `ExaAgent` instances in one agent never
claim each other's calls (each has its own `output_schema` and poll settings).
The token is the capability's `id`, which also holds in a process that
recovers a durable run.
"""

_AGENT_TOOL_NAME = 'exa_agent'

_INSTRUCTIONS = (
    'You can delegate open-ended research, list-building, or structured enrichment tasks that '
    'need many searches and reads to `exa_agent`. It returns a cited result and a run ID you '
    'can pass back as `previous_run_id` to ask follow-up questions.'
)


class ExaAgentRuns(Protocol):
    """The subset of the `exa_py` Agent runs API (`AsyncExa().agent.runs`) that `ExaAgent` calls.

    Any object with these two methods can back the capability. Pass one via
    `ExaAgent.runs` to configure authentication explicitly, or to substitute a
    fake in tests. The signatures mirror `exa_py`'s own (`create` is typed with
    the streaming union even though `ExaAgent` never streams), so a real
    `AsyncExa().agent.runs` satisfies the protocol as-is.
    """

    async def create(
        self,
        *,
        query: str,
        system_prompt: str | None = None,
        output_schema: dict[str, object] | type[BaseModel] | None = None,
        effort: AgentEffort | None = None,
        previous_run_id: str | None = None,
    ) -> AgentRun | AsyncGenerator[AgentEvent, None]:
        """Create an agent run."""
        ...  # pragma: no cover

    async def poll_until_finished(
        self, run_id: str, *, poll_interval: int = 1000, timeout_ms: int = 3600000
    ) -> AgentRun:
        """Poll an agent run until it reaches a terminal status."""
        ...  # pragma: no cover


def _default_runs() -> ExaAgentRuns:
    """Build the Agent runs client of an `AsyncExa` from the `EXA_API_KEY` environment variable."""
    try:
        return AsyncExa().agent.runs
    except ValueError as error:
        raise UserError(
            'ExaAgent needs an Exa API key: set the EXA_API_KEY environment variable, '
            'or pass a configured runs client, e.g. ExaAgent(runs=AsyncExa(api_key=...).agent.runs).'
        ) from error


def agent_run_result(
    run: AgentRun, *, output_schema: type[BaseModel] | dict[str, object] | None = None
) -> ToolReturn[str]:
    """Render a terminal Exa agent run as the `exa_agent` tool result.

    Use it when resolving externally executed `exa_agent` calls in a host
    application (building `DeferredToolResults`), so external and inline
    execution produce the same result shape. When `output_schema` is a
    Pydantic model class, a completed run's structured output is validated
    against it; a mismatch raises `ModelRetry`.

    The `ToolReturn.return_value` text is what the model sees; its metadata
    carries the run ID under `RUN_ID_METADATA_KEY` and the citation `sources`
    (`ExaSource` dicts) for the application to use directly.
    """
    if run.status != 'completed':
        message = run.error.message if run.error is not None and run.error.message else 'no error details'
        return ToolReturn(
            f'Exa agent run {run.id} {run.status}: {message}',
            metadata={RUN_ID_METADATA_KEY: run.id, 'sources': []},
        )
    output = run.output
    body = output.text or '' if output is not None else ''
    if output is not None and output.structured is not None:
        structured: object = output.structured
        if isinstance(output_schema, type):
            try:
                body = output_schema.model_validate(structured).model_dump_json()
            except ValidationError as error:
                raise ModelRetry(
                    f'Exa agent run {run.id} output did not match the configured schema: {error}'
                ) from error
        else:
            body = json.dumps(structured)
    if not body:
        body = '(no text output)'
    sources: dict[str, str | None] = {}
    if output is not None and output.grounding:
        sources = {citation.url: citation.title for entry in output.grounding for citation in entry.citations}
    text = f'{body}\n\nRun ID: {run.id} (pass as previous_run_id for follow-ups)'
    return ToolReturn(
        _with_sources(text, sources),
        metadata={RUN_ID_METADATA_KEY: run.id, 'sources': _source_list(sources)},
    )


async def _create_run(
    runs: ExaAgentRuns,
    *,
    query: str,
    system_prompt: str | None,
    output_schema: type[BaseModel] | dict[str, object] | None,
    effort: AgentEffort | None,
    previous_run_id: str | None,
) -> str:
    """Create an Exa agent run and return its ID."""
    run = await runs.create(
        query=query,
        system_prompt=system_prompt,
        output_schema=output_schema,
        effort=effort,
        previous_run_id=previous_run_id,
    )
    assert isinstance(run, AgentRun)
    return run.id


class ExaAgentToolset(FunctionToolset[AgentDepsT]):
    """Provides the `exa_agent` tool: create an Exa agent run and defer the call until it finishes.

    `ExaAgent` passes `create_run` so that creating the Exa run is one of its
    durable operations, whose result durable execution records instead of
    creating another run on recovery.
    """

    def __init__(
        self,
        *,
        runs: ExaAgentRuns,
        effort: AgentEffort | None,
        output_schema: type[BaseModel] | dict[str, object] | None,
        system_prompt: str | None,
        owner_id: str,
        id: str | None = None,
        create_run: Callable[[str, str | None], Awaitable[str | RetryRequest]] | None = None,
    ) -> None:
        super().__init__(id=id)
        self._runs = runs
        self._effort: AgentEffort | None = effort
        self._output_schema = output_schema
        self._system_prompt = system_prompt
        self._owner_id = owner_id
        self._create_run = create_run
        self.add_function(self.exa_agent, name=_AGENT_TOOL_NAME)

    @_recoverable
    async def exa_agent(self, query: str, previous_run_id: str | None = None) -> str:
        """Delegate an open-ended research, list-building, or enrichment task to an Exa agent.

        Args:
            query: The research task or question for the agent.
            previous_run_id: A previous result's run ID, to ask a follow-up
                question in the context of that run.

        Returns:
            The agent's cited result, with its run ID.
        """
        if self._create_run is not None:
            run_id = raise_retry(await self._create_run(query, previous_run_id))
        else:
            run_id = await _create_run(
                self._runs,
                query=query,
                system_prompt=self._system_prompt,
                output_schema=self._output_schema,
                effort=self._effort,
                previous_run_id=previous_run_id,
            )
        raise CallDeferred(metadata={RUN_ID_METADATA_KEY: run_id, _OWNER_METADATA_KEY: self._owner_id})


@dataclass
class ExaAgent(AbstractCapability[AgentDepsT]):
    """Delegation of deep research tasks to the [Exa](https://exa.ai) Agent API.

    Adds one tool, `exa_agent`, which creates an asynchronous Exa agent run
    (queued -> running -> terminal, up to an hour) and defers the tool call.
    By default the capability resolves its own deferred calls inline by
    polling the run to completion. With `execution='external'` the calls
    surface as `DeferredToolRequests` output instead, for the host application
    to resolve out of band (the run ID is in the request metadata under
    `RUN_ID_METADATA_KEY`), including across process restarts.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness.exa import ExaAgent

    agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[ExaAgent()])
    ```

    Authentication comes from the `EXA_API_KEY` environment variable by
    default; pass `runs` to configure it explicitly.

    Creating a run and polling it to completion are durable operations, so
    under durable execution a recovered run reuses the recorded run ID and
    result instead of creating or polling the Exa run again.
    """

    effort: AgentEffort | None = None
    """How much work the Exa agent invests per run; `None` uses the API default."""

    execution: Literal['inline', 'external'] = 'inline'
    """How deferred `exa_agent` calls are resolved.

    With `'inline'`, the capability polls the run to completion during the
    agent run. With `'external'`, calls bubble up as `DeferredToolRequests`
    output for the host application to resolve (see `agent_run_result`), which
    suits durable workers that outlive a single process.
    """

    output_schema: type[BaseModel] | dict[str, object] | None = None
    """Structured output schema for the Exa agent's result. `None` returns prose.

    Accepts a Pydantic model class or a JSON-schema-style dict. A model class
    is forwarded to the API and a completed run's structured output is
    validated against it (a mismatch surfaces as a retry). The dict form skips
    client-side validation and is the serializable shape used by agent specs.
    """

    system_prompt: str | None = None
    """System prompt forwarded to the Exa agent run; `None` uses the API default."""

    poll_interval: int = 1000
    """Milliseconds between polls while resolving a run inline."""

    timeout_ms: int = 3_600_000
    """Milliseconds to wait for a run to finish when resolving inline."""

    guidance: str | None = None
    """Custom delegation guidance for the system prompt.

    Leave as `None` for the default guidance, or set `''` to contribute no
    instructions at all.
    """

    runs: ExaAgentRuns | None = None
    """Exa Agent runs client; when `None`, `exa_py.AsyncExa().agent.runs` is built from `EXA_API_KEY`.

    Any object satisfying the `ExaAgentRuns` protocol works: use it to pass an
    API key explicitly or substitute a fake in tests.
    """

    _: KW_ONLY
    id: str | None = 'exa_agent'
    """Stable identity for durable execution, which records each Exa run's creation and result under it."""

    _instance_token: str = field(default_factory=lambda: uuid4().hex, init=False, repr=False, compare=False)
    """Owner token for a capability constructed with `id=None`, which has no stable one."""

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        """Static delegation guidance: when to hand a task to `exa_agent`, and run ID continuation.

        A non-`None` `guidance` replaces the default; `''` disables
        instructions entirely.
        """
        if self.guidance is not None:
            return self.guidance or None
        return _INSTRUCTIONS

    @classmethod
    def combine(cls, capabilities: Sequence[AbstractCapability[AgentDepsT]]) -> AbstractCapability[AgentDepsT]:
        """Two under one `id` are one configuration stated twice; two that disagree raise rather than merge."""
        return one_per_id(capabilities)

    def get_toolset(self) -> ExaAgentToolset[AgentDepsT]:
        """Build the toolset providing the `exa_agent` tool."""
        return ExaAgentToolset[AgentDepsT](
            runs=self._resolved_runs(),
            effort=self.effort,
            output_schema=self.output_schema,
            system_prompt=self.system_prompt,
            owner_id=self._owner_token,
            id=self.id,
            create_run=self._create_run,
        )

    @durable_operation('create_run')
    async def _create_run(self, query: str, previous_run_id: str | None) -> str | RetryRequest:
        return await retry_as_result(self._start_run(query, previous_run_id))

    @_recoverable
    async def _start_run(self, query: str, previous_run_id: str | None) -> str:
        return await _create_run(
            self._resolved_runs(),
            query=query,
            system_prompt=self.system_prompt,
            output_schema=self.output_schema,
            effort=self.effort,
            previous_run_id=previous_run_id,
        )

    async def handle_deferred_tool_calls(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        requests: DeferredToolRequests,
    ) -> DeferredToolResults | None:
        """Resolve deferred `exa_agent` calls inline by polling the Exa run to completion.

        With `execution='external'` all calls are left unresolved, so they
        bubble up as `DeferredToolRequests` output for the host application to
        resolve; the Exa run ID is available in
        `requests.metadata[tool_call_id][RUN_ID_METADATA_KEY]`.

        Calls are claimed by the owner token in the deferred-call metadata
        rather than by tool name, so tool renaming or prefixing wrappers (e.g.
        `PrefixTools`) do not break inline resolution.
        """
        if self.execution != 'inline':
            return None
        calls: dict[str, ToolReturn[str] | ModelRetry] = {}
        for call in requests.calls:
            metadata = requests.metadata.get(call.tool_call_id, {})
            run_id = metadata.get(RUN_ID_METADATA_KEY)
            if metadata.get(_OWNER_METADATA_KEY) == self._owner_token and isinstance(run_id, str):
                result = await self._resolve_run(run_id)
                calls[call.tool_call_id] = ModelRetry(result.message) if isinstance(result, RetryRequest) else result
        if not calls:
            return None
        return DeferredToolResults(calls=calls)

    @durable_operation('resolve_run')
    async def _resolve_run(self, run_id: str) -> ToolReturn[str] | RetryRequest:
        """Poll one Exa run and render its tool result, mapping failures to retry results.

        Exceptions raised here would abort the whole agent run: the deferred-call
        pipeline only converts a `ModelRetry` *result* (in `DeferredToolResults.calls`)
        into a retry prompt, not a raised one. So a structured-output mismatch is
        returned as a `RetryRequest`, and a polling failure (timeout, network, or
        transient API error) becomes a `RetryRequest` that carries the run ID, since
        the run may still finish server-side and the model can follow up with
        `previous_run_id`. `handle_deferred_tool_calls` turns each into its
        `ModelRetry` result. Auth failures propagate as configuration errors.
        """
        try:
            run = await self._resolved_runs().poll_until_finished(
                run_id, poll_interval=self.poll_interval, timeout_ms=self.timeout_ms
            )
        except (TimeoutError, httpx.HTTPError) as error:
            return RetryRequest(
                f'Waiting for Exa agent run {run_id} failed: {error}. '
                f'The run may still finish server-side; you can follow up by calling the tool '
                f'again with previous_run_id={run_id!r}.'
            )
        except ValueError as error:
            if _AUTH_STATUS_RE.search(str(error)):
                raise
            return RetryRequest(
                f'Waiting for Exa agent run {run_id} failed: {error}. '
                f'The run may still finish server-side; you can follow up by calling the tool '
                f'again with previous_run_id={run_id!r}.'
            )
        try:
            return agent_run_result(run, output_schema=self.output_schema)
        except ModelRetry as retry:
            return RetryRequest(retry.message)

    @property
    def _owner_token(self) -> str:
        """The token stamped into deferred-call metadata so the inline resolver only claims this capability's calls.

        The capability's `id`, which a process that recovers a durable run gives it too, unlike a
        per-instance token: replaying the run there must claim the same calls the original did.
        """
        return self.id if self.id is not None else self._instance_token

    def _resolved_runs(self) -> ExaAgentRuns:
        return self.runs if self.runs is not None else _default_runs()

    @classmethod
    def from_spec(
        cls,
        *,
        effort: AgentEffort | None = None,
        execution: Literal['inline', 'external'] = 'inline',
        output_schema: dict[str, object] | None = None,
        system_prompt: str | None = None,
        poll_interval: int = 1000,
        timeout_ms: int = 3_600_000,
        guidance: str | None = None,
        id: str | None = 'exa_agent',
    ) -> ExaAgent[AgentDepsT]:
        """Construct the capability from serializable spec options.

        The `runs` field is not spec-serializable, so spec-loaded instances
        always build the default client from `EXA_API_KEY`. `output_schema`
        takes the JSON-schema dict form here; Pydantic model classes are only
        available when constructing the capability in Python.
        """
        return cls(
            effort=effort,
            execution=execution,
            output_schema=output_schema,
            system_prompt=system_prompt,
            poll_interval=poll_interval,
            timeout_ms=timeout_ms,
            guidance=guidance,
            id=id,
        )
