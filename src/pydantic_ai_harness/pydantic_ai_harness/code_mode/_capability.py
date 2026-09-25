"""Code mode capability that routes selected tools through a Monty sandbox."""

from __future__ import annotations

from collections.abc import AsyncIterable, Sequence
from dataclasses import KW_ONLY, dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

from pydantic import TypeAdapter, ValidationError
from pydantic_ai import AbstractToolset
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering
from pydantic_ai.capabilities._tool_search import ToolSearch as _ToolSearch
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import AgentStreamEvent, ModelResponse, NativeToolSearchReturnPart, SystemPromptPart
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition, ToolSelector
from typing_extensions import TypedDict

from pydantic_ai_harness.code_mode._eager import EagerCodeModeToolset
from pydantic_ai_harness.code_mode._speculation import (
    MAX_SPECULATIONS_PER_PART,
    SpeculationCoordinator,
    SpeculationStats,
)
from pydantic_ai_harness.code_mode._toolset import (
    CodeModeMount,
    CodeModeOS,
    CodeModeResourceLimits,
    CodeModeToolset,
    as_os_handler,
    in_durable_execution,
)

if TYPE_CHECKING:
    from pydantic_ai.capabilities.abstract import ValidatedToolArgs
    from pydantic_ai.messages import ToolCallPart
    from pydantic_ai.models import ModelRequestContext


_DISCOVERY_ANNOUNCEMENT_PREFIX = (
    'New functions are now available inside `run_code`. Their signatures have been '
    'added to the available-functions catalog in the system prompt'
)


@dataclass
class CodeMode(AbstractCapability[AgentDepsT]):
    """Capability that exposes selected tools as callables inside a `run_code` sandbox.

    By default (`tools='all'`) every eligible regular tool the agent has is wrapped
    behind a single `run_code` tool -- the model writes Python that calls them as
    functions instead of issuing tool calls directly. Framework control tools,
    undiscovered deferred tools, native fallbacks, and other code-execution tools
    remain native.

    Pass a list of tool names or a callable predicate to `tools` to split the
    toolset: matching tools become callables inside the sandbox, and the rest
    stay visible to the model as normal tool calls.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness import CodeMode

    # Sandbox all tools
    agent = Agent('openai:gpt-5', capabilities=[CodeMode()])

    # Sandbox only specific tools
    agent = Agent('openai:gpt-5', capabilities=[CodeMode(tools=['search', 'fetch'])])
    ```

    By default, sandboxed code cannot touch the host -- no filesystem, environment
    variables, or clock. Two parameters open it up:

    - `mount` shares specific host directories: reach for it when the agent reads or
      writes real files.
    - `os_access` routes the sandbox's OS calls to a handler you provide: reach for it
      when the agent needs environment variables, the clock, or filesystem behavior you
      control.

    `mount` exposes selected host directories. The built-in `OSAccess` has an
    isolated filesystem and environment but uses the host clock by default; custom
    OS handlers can expose other host resources.

    ```python
    from pydantic_monty import MountDir

    agent = Agent('openai:gpt-5', capabilities=[CodeMode(mount=MountDir(virtual_path='/work', host_path='/tmp/agent-work'))])
    ```
    """

    tools: ToolSelector[AgentDepsT] = field(default='all')
    """Which wrapped tools should be sandboxed inside `run_code`.

    - `'all'` (default): every eligible regular tool the agent has is sandboxed.
    - `Sequence[str]`: only tools whose names are listed are sandboxed.
    - Callable `(ctx, tool_def) -> bool | Awaitable[bool]`: tools where the
      callable returns `True` are sandboxed; the rest stay as native tool calls.
    """

    max_retries: int = 3
    """Maximum number of retries for the `run_code` tool (syntax errors count as retries)."""

    _: KW_ONLY

    max_tool_calls: int = 100
    """Maximum nested tool calls dispatched by one `run_code` invocation.

    Budget is reserved before each call is scheduled, so a snippet cannot allocate host tasks
    beyond this many. Calls past the budget are refused at the sandbox call site.
    """

    os_access: CodeModeOS | None = None
    """Give sandboxed code environment variables, the clock, and file I/O through a handler you provide; unset, they are unavailable.

    Pass an `AbstractOS` such as `OSAccess`, or a handler called with keyword arguments (see
    `pydantic_monty.OsHandler`). A plain function is called from a Monty thread, not the event
    loop's thread, with the run's contextvars set, so it must not touch asyncio objects; an `async`
    handler is awaited on an event loop instead. Inside a Temporal workflow, clock, environment, and
    randomness calls run on the workflow's own thread, so the handler can use `workflow.now()`.
    The positional `(name, args, kwargs)` form is deprecated.
    """

    mount: CodeModeMount | None = None
    """Host directories to expose to sandboxed `pathlib` code; each mount's `mode` controls whether writes reach the host."""

    resource_limits: CodeModeResourceLimits | Literal['unlimited'] | None = None
    """Sandbox execution limits.

    `None` applies a 30-second execution and 256 MiB heap backstop. `max_duration_secs` is per
    snippet: no single `run_code` snippet runs longer than it, and it is not a run-wide budget.
    `'unlimited'` removes the time and memory caps, but Monty's finite suspension budget still
    applies. Set `max_suspensions` to bound cumulative host interactions across consecutive snippets.
    """

    eager: bool = False
    """Execute complete streamed statements before the `run_code` call finishes.

    Needs asyncio, like the sandbox executor, and is inactive under durable execution. Side
    effects cannot be rolled back and run before hooks on `run_code` see the completed call.
    See the Code Mode guide for the execution and `restart` semantics.
    """

    speculate: Sequence[str] | Literal['declared'] | None = None
    """Launch side-effect-free sandbox calls while the `run_code` arguments are still streaming.

    Calls to eligible functions whose arguments are all keyword literals start as soon as their
    text has streamed; when the completed snippet dispatches the same call, the in-flight result
    is adopted instead of starting cold. Pass the names of tools that are safe to run early, or
    `'declared'` to trust what the tools declare about themselves (`Tool(metadata={'read_only':
    True})` or the MCP `readOnlyHint` annotation). At most `max_tool_calls` (and never more than
    32) calls start early per `run_code` call, and they do not reserve from `max_tool_calls`:
    unclaimed launches are extra bounded work alongside the dispatches the snippet makes.
    Composes with `eager`. Inactive under durable execution and when the run's parallel
    execution mode is sequential. See the Code Mode guide for the mechanics.
    """

    monty_sandbox_url: str | None = None
    """Run sandboxed code on remote Monty workers reached over this `ws://` or `wss://` URL.

    Only execution moves: tool dispatch, mounts, `os_access`, and print capture stay host-side
    over the connection.

    Use `wss://` unless the server is on a network you trust: the connection carries the tool calls
    your agent executes, so anyone who can intercept it can choose what your tools run.
    """

    dynamic_catalog: bool = False
    """Keep the `run_code` tool definition cache-stable as the sandboxed toolset grows.

    By default the signatures of all sandboxed tools are rendered into `run_code`'s
    description, which lives in the prompt-cache-keyed tool-definitions block. When the
    toolset changes mid-run -- e.g. [`ToolSearch`][pydantic_ai.capabilities.ToolSearch]
    reveals a new tool that then gets folded into `run_code` -- the description changes and
    busts the prefix cache from that point on.

    Set `dynamic_catalog=True` to instead:

    - keep only the static base prose (sandbox restrictions, return-value contract) in
      `run_code.description`, so the tool-definitions block stays byte-stable across
      discoveries;
    - move the "available functions" catalog (TypedDict definitions + signatures) into
      agent instructions as a dynamic
      [`InstructionPart`][pydantic_ai.messages.InstructionPart], which providers with
      static/dynamic instruction splitting (Anthropic, Bedrock) place after the cache
      breakpoint;
    - announce newly-discovered tools via a short
      [`SystemPromptPart`][pydantic_ai.messages.SystemPromptPart] enqueued through
      [`RunContext.enqueue`][pydantic_ai.tools.RunContext.enqueue], so the model knows the
      new functions are callable without rewriting the cached description.

    This pays off when paired with [`ToolSearch`][pydantic_ai.capabilities.ToolSearch]: the
    tool-definitions cache survives discoveries at the cost of a larger (but
    cache-friendly) system prompt. With a fixed toolset and no `ToolSearch`, the default
    keeps the system prompt shorter and is the better choice.
    """

    speculation_stats: SpeculationStats = field(default_factory=SpeculationStats, init=False, repr=False)
    """Aggregate launch/adopt/evict counters across this instance's runs, when `speculate` is set."""

    _speculation: SpeculationCoordinator[AgentDepsT] | None = field(default=None, init=False, repr=False)

    _announced_tools: set[str] = field(default_factory=set[str], init=False, repr=False)

    def __post_init__(self) -> None:
        # Converted once here, so the per-run copies and the toolsets built from this do not warn again.
        self.os_access = as_os_handler(self.os_access)
        if isinstance(self.speculate, str) and self.speculate != 'declared':
            raise UserError(
                f"`speculate` accepts a list of tool names or the string 'declared', not {self.speculate!r}. "
                'To allowlist one tool, pass a one-element list.'
            )

    def get_ordering(self) -> CapabilityOrdering:
        """CodeMode wraps around ToolSearch so that search_tools stays native."""
        return CapabilityOrdering(position='outermost', wraps=[_ToolSearch])

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> CodeMode[AgentDepsT]:
        """Return a fresh instance so concurrent runs don't share `_announced_tools` or speculation state."""
        if not self.dynamic_catalog and self.speculate is None:
            return self
        clone = replace(self)
        # `replace` re-runs `__init__`, resetting `init=False` fields: `_announced_tools` starts
        # fresh (intended), and the stats object is rebound so callers holding this instance
        # observe counters accumulated by its per-run clones.
        clone.speculation_stats = self.speculation_stats
        if self.speculate is not None:
            allowlist = 'declared' if isinstance(self.speculate, str) else frozenset(self.speculate)
            clone._speculation = SpeculationCoordinator(
                allowlist=allowlist,
                stats=self.speculation_stats,
                launch_cap=min(MAX_SPECULATIONS_PER_PART, self.max_tool_calls),
            )
        return clone

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT] | None:
        """Wrap the agent's assembled toolset, splitting it into native + sandboxed subsets if needed."""
        if self.eager:
            return EagerCodeModeToolset(
                wrapped=toolset,
                tool_selector=self.tools,
                max_retries=self.max_retries,
                max_tool_calls=self.max_tool_calls,
                resource_limits=self.resource_limits,
                dynamic_catalog=self.dynamic_catalog,
                os_access=self.os_access,
                mount=self.mount,
                monty_sandbox_url=self.monty_sandbox_url,
                capability=self,
                speculation=self._speculation,
            )
        return CodeModeToolset(
            wrapped=toolset,
            tool_selector=self.tools,
            max_retries=self.max_retries,
            max_tool_calls=self.max_tool_calls,
            resource_limits=self.resource_limits,
            dynamic_catalog=self.dynamic_catalog,
            os_access=self.os_access,
            mount=self.mount,
            monty_sandbox_url=self.monty_sandbox_url,
            capability=self,
            speculation=self._speculation,
        )

    @property
    def has_wrap_run_event_stream(self) -> bool:
        """Report the stream hook only when a streamed execution tier is enabled.

        The base class detects a class-level override, which would put every `CodeMode` user in
        streaming mode; gating on the instance keeps plain `CodeMode` runs non-streaming.
        """
        return self.eager or self.speculate is not None

    async def wrap_run_event_stream(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        stream: AsyncIterable[AgentStreamEvent],
    ) -> AsyncIterable[AgentStreamEvent]:
        """Feed streamed `run_code` argument deltas to the eager pump and the speculation launcher.

        Wrapped events pass through unmodified; the watchers act by side effect, enqueueing
        closed statements for the live REPL and launching eligible calls. Inactive under durable
        execution, where overlapping non-deterministic work with the stream has no place in a
        replayed workflow.
        """
        toolset = None if in_durable_execution(ctx) else CodeModeToolset.from_run_context(ctx)
        async for event in stream:
            yield event
            if toolset is not None:
                await toolset.observe_stream_event(event, ctx)

    async def after_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        result: Any,
    ) -> Any:
        """Announce newly-discovered tools from a local `search_tools` return.

        Only active with `dynamic_catalog=True`. The native-search path is handled by
        [`after_model_request`][pydantic_ai_harness.CodeMode.after_model_request] instead
        (server-side search emits a `NativeToolSearchReturnPart` rather than a regular tool
        execute result).
        """
        if self.dynamic_catalog and tool_def.tool_kind == 'tool-search':
            self._announce_newly_discovered(ctx, _extract_discovered_names(result))
        return result

    async def after_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        """Announce newly-discovered tools from a native (server-side) tool-search return.

        Only active with `dynamic_catalog=True`.
        """
        if self.dynamic_catalog:
            for part in response.parts:
                if isinstance(part, NativeToolSearchReturnPart):
                    self._announce_newly_discovered(ctx, _extract_discovered_names(part.content))
        return response

    def _announce_newly_discovered(self, ctx: RunContext[AgentDepsT], names: Sequence[str]) -> None:
        """Enqueue a system-prompt announcement for any names we haven't already announced."""
        fresh = [n for n in names if n not in self._announced_tools]
        if not fresh:
            return
        self._announced_tools.update(fresh)
        listing = ', '.join(f'`{name}`' for name in fresh)
        # Enqueue a `SystemPromptPart` so the announcement is framed as system-level context.
        # Mid-conversation `SystemPromptPart`s are rendered inline (not hoisted to the top-level
        # system prompt) on all providers since pydantic/pydantic-ai#5509, so this is cache-safe.
        ctx.enqueue(SystemPromptPart(content=f'{_DISCOVERY_ANNOUNCEMENT_PREFIX}: {listing}.'))


class _DiscoveredCatalog(TypedDict):
    """Lenient view of a tool-search return: just the entry list, items left unvalidated."""

    discovered_tools: list[object]


class _DiscoveredEntry(TypedDict):
    """Lenient view of one discovered-tool entry: only the name we announce."""

    name: str


_CATALOG_ADAPTER = TypeAdapter(_DiscoveredCatalog)
_ENTRY_ADAPTER = TypeAdapter(_DiscoveredEntry)


def _extract_discovered_names(content: object) -> list[str]:
    """Read newly-discovered tool names from a tool-search return content.

    Carried on both the local `ToolSearchReturnPart` and the native
    `NativeToolSearchReturnPart`. Validated leniently: a malformed catalog yields `[]` and a
    malformed entry is skipped, since the announcement is a courtesy nudge, not load-bearing
    logic.
    """
    try:
        catalog = _CATALOG_ADAPTER.validate_python(content)
    except ValidationError:
        return []
    names: list[str] = []
    for entry in catalog['discovered_tools']:
        try:
            names.append(_ENTRY_ADAPTER.validate_python(entry)['name'])
        except ValidationError:
            continue
    return names
