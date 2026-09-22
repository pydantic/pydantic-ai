"""Speculative launch of sandboxed tool calls while `run_code` arguments stream.

Implements the streaming half of speculative programmatic tool calling (sPTC,
<https://alexzhang13.github.io/blog/2026/spec-ptc/>) for `CodeMode`: while the model is still
emitting the `run_code` tool call, the partial `code` argument is decoded from the accumulated
JSON and scanned for calls to eligible sandbox functions whose arguments are all literals. Those
calls launch immediately as tasks; when the completed snippet later executes and dispatches the
same call, the in-flight task is adopted instead of starting the tool cold.

The other half of the trick, overlapping independent calls with each other during execution,
already exists: `MontyExecutor` defers external calls as futures and only forces them when the
sandbox needs the value.

Scope is deliberately the blog's "Case 1" (all-literal arguments). Calls whose arguments carry
dependencies would need a shadow interpreter to materialize values.

Only eligible tools are ever launched early. Launching early is observationally equivalent to
the normal call only for tools without side effects, so eligibility is an explicit promise: the
user's allowlist, or the tools' own declarations. Speculated calls run through the same nested
`ToolManager` path as cold calls, so tool hooks fire at launch time rather than at adoption time.
"""

from __future__ import annotations

import ast
import asyncio
import re
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Annotated, Any, Generic, Literal

import anyio
from pydantic import Strict, TypeAdapter, ValidationError
from pydantic_ai.messages import (
    AgentStreamEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    ToolCallPart,
    ToolCallPartDelta,
)
from pydantic_ai.tool_manager import ToolManager
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition
from pydantic_ai.toolsets.abstract import AbstractToolset, ToolsetTool
from typing_extensions import TypedDict

from pydantic_ai_harness.code_mode._events import (
    SpeculativeCallClaimedEvent,
    SpeculativeCallEvictedEvent,
    SpeculativeCallLaunchedEvent,
    SpeculativeCallMissedEvent,
    SpeculativeCallSettledEvent,
    SpeculativeCodeUpdateEvent,
)
from pydantic_ai_harness.code_mode._streaming import (
    CANCEL_TIMEOUT_SECONDS,
    MAX_SCAN_CHARS,
    MAX_SCAN_WORK_CHARS,
    closed_statements,
    decode_partial_args,
)
from pydantic_ai_harness.code_mode._toolset import (
    NestedCallOutcome,
    RunCodeExecution,
    global_mode_is_sequential,
    run_nested_call,
)

_RUN_CODE_TOOL_NAME = 'run_code'

MAX_SPECULATIONS_PER_PART = 32
"""Upper bound on early launches per streamed `run_code` part.

Wrong-branch conditionals and rewritten lines make some speculations garbage; the cap bounds
how much garbage one part can start. Adoption is unaffected once a call is in flight.
"""

SpeculationAllowlist = frozenset[str] | Literal['declared']
"""Original tool names the user declared side-effect free, or `'declared'` to trust the tools'
own definitions (see `declares_speculation_safety`)."""


@dataclass(kw_only=True)
class SpeculationStats:
    """Aggregate speculation counters, shared across the runs of one `CodeMode` instance."""

    launched: int = 0
    """Calls started early from partial code."""

    adopted: int = 0
    """Early calls whose result the executing snippet actually claimed."""

    evicted: int = 0
    """Early calls cancelled or discarded without being claimed."""


@dataclass(kw_only=True)
class SpeculativeCall:
    """One in-flight early launch, waiting to be claimed by the executing snippet."""

    sandbox_name: str
    original_name: str
    kwargs: dict[str, object]
    task: asyncio.Task[NestedCallOutcome]
    launch_id: str
    started_at: float
    settled_at: float | None = None
    settle_emitted: bool = False

    def settled_state(self) -> Literal['pending', 'ready', 'failed']:
        """Where this launch currently stands, for settled and eviction events."""
        if not self.task.done() or self.task.cancelled():
            return 'pending'
        if self.task.exception() is not None:  # pragma: no cover - the runner settles, never raises
            return 'failed'
        return 'failed' if self.task.result().error is not None else 'ready'

    def elapsed_ms(self) -> float:
        """Wall-clock from launch until settled, or until now while still running."""
        end = self.settled_at if self.settled_at is not None else time.perf_counter()
        return (end - self.started_at) * 1000


@dataclass(kw_only=True)
class _StepIngredients(Generic[AgentDepsT]):
    """Everything the launcher needs to run a nested call, stashed by the toolset each step."""

    wrapped: AbstractToolset[AgentDepsT]
    wrapped_tools: dict[str, ToolsetTool[AgentDepsT]]
    sanitized_to_original: dict[str, str]
    eligible: frozenset[str]
    """Sandbox (possibly sanitized) names that may launch early this step."""


@dataclass(kw_only=True)
class _PartWatch:
    """Accumulated state for one streamed `run_code` tool call part."""

    tool_call_id: str
    run_step: int
    """The model step that produced the part; launches survive one retry step, then retire."""

    args_text: str = ''
    args_dict: dict[str, Any] | None = None
    halted: bool = False
    """Set once the streamed prefix outgrew the host-side scan budget; the part stays whole
    until dispatch, where the execution prefetch still covers it."""

    scan_work_chars: int = 0
    closed_count: int = 0
    scanned_newlines: int = -1
    """Newline count of the code at the last full scan.

    Statements only close on line boundaries, so the parse work is skipped for the many deltas
    that arrive within a line; they still produce a code-update event, just without reparsing.
    """

    demanded: dict[str, int] = field(default_factory=dict[str, int])
    """Occurrences of each `(function, arguments)` key seen in this part's streamed code.

    Launches are deduplicated against calls already in flight (typically a failed attempt's
    surviving launches), so a retry claims instead of relaunching. Multiplicity within the part
    stays exact: the Nth occurrence launches once N exceeds the in-flight count.
    """

    launched: int = 0
    calls: dict[str, deque[SpeculativeCall]] = field(default_factory=dict[str, deque[SpeculativeCall]])
    """FIFO per canonical key: the k-th identical dispatch claims the k-th launch, so results of a
    nondeterministic tool called twice with the same arguments are never collapsed."""

    hits: int = 0
    misses: int = 0
    hidden_ms: float = 0.0
    """Launch-to-claim wall-clock summed over hits: latency the snippet did not wait for."""


def _canonical(value: Any) -> object:
    """A type-tagged, order-normalized form of a literal value whose `repr` is injective.

    JSON was not enough: it coerces `{1: 'x'}` and `{'1': 'x'}` to the same text and rejects
    tuple keys outright. Tagging every node with its type keeps `1`, `1.0`, `True`, and `'1'`
    apart, and sorting unordered containers by the `repr` of their members makes the form
    independent of construction order while staying comparable across mixed member types.
    """
    # Same idiom as `_preview`: `isinstance` narrows to an unparameterized container, so the
    # elements are read through an unnarrowed alias to keep them typed.
    raw: Any = value
    if isinstance(value, dict):
        items = [(_canonical(key), _canonical(item)) for key, item in raw.items()]
        return ('dict', tuple(sorted(items, key=repr)))
    if isinstance(value, (set, frozenset)):
        return (type(raw).__name__, tuple(sorted((_canonical(member) for member in raw), key=repr)))
    if isinstance(value, (list, tuple)):
        return (type(raw).__name__, tuple(_canonical(element) for element in raw))
    return (type(raw).__name__, value)


def _canonical_key(sandbox_name: str, kwargs: dict[str, object]) -> str:
    """Claim identity for one concrete call: both launch and claim hash through here."""
    return repr((sandbox_name, _canonical(kwargs)))


_StrictBool = Annotated[bool, Strict()]
"""A declaration counts only when it is literally `True`, not a truthy stand-in."""


class _McpSafetyHints(TypedDict, total=False):
    """The MCP tool annotations that vouch for early execution, as they arrive in tool metadata."""

    readOnlyHint: _StrictBool


class _SafetyDeclarations(TypedDict, total=False):
    """The tool metadata keys `speculate='declared'` reads; unrelated keys are ignored."""

    read_only: _StrictBool
    annotations: _McpSafetyHints


_SAFETY_ADAPTER = TypeAdapter(_SafetyDeclarations)


def declares_speculation_safety(tool_def: ToolDefinition) -> bool:
    """Whether a tool's own definition presents evidence that early execution is safe.

    Two channels: first-party tools set `metadata={'read_only': True}` on the `Tool`, and MCP
    servers publish the `readOnlyHint` tool annotation, which arrives under
    `metadata['annotations']`. Idempotence is deliberately not evidence: an idempotent delete
    still deletes, and a launch from an untaken branch would run it. Hints are the server's
    claim, not a proof; `speculate='declared'` extends them the trust an explicit allowlist
    places in the user.

    The key vocabulary tracks pydantic-ai's tool behavior annotations (pydantic/pydantic-ai#6344,
    catalogued in pydantic/pydantic-ai#7955), so a first-class `ToolDefinition` field with this
    name would already be in use here.
    """
    try:
        declared = _SAFETY_ADAPTER.validate_python(tool_def.metadata or {})
    except ValidationError:
        return False
    return bool(declared.get('read_only') or declared.get('annotations', {}).get('readOnlyHint'))


_SKIP_CONTAINERS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


def _iter_calls(node: ast.AST) -> list[ast.Call]:
    """Collect `ast.Call` nodes in execution position, skipping bodies that don't run yet.

    `def`/`class`/`lambda` bodies execute only when later invoked, usually with non-literal
    arguments, so calls inside them are excluded. Conditional and loop bodies are included:
    launching a pure call from a branch that never runs wastes the call but stays correct,
    since unclaimed launches are evicted.
    """
    found: list[ast.Call] = []
    if isinstance(node, ast.Call):
        found.append(node)
    for child in ast.iter_child_nodes(node):
        if isinstance(child, _SKIP_CONTAINERS):
            continue
        found.extend(_iter_calls(child))
    return found


@dataclass(kw_only=True)
class _ExtractedCall:
    """One speculatable call read off the code, with the launching statement's line span."""

    sandbox_name: str
    kwargs: dict[str, object]
    line_start: int
    line_end: int


def _literal_calls(statements: Sequence[ast.stmt], eligible: frozenset[str]) -> list[_ExtractedCall]:
    """Extract eligible sandbox calls whose arguments are entirely keyword literals."""
    out: list[_ExtractedCall] = []
    for statement in statements:
        if isinstance(statement, _SKIP_CONTAINERS):
            # A top-level `def`/`class` statement only defines; its body (and even its
            # decorators, conservatively) runs later, if ever.
            continue
        for call in _iter_calls(statement):
            func = call.func
            if not isinstance(func, ast.Name) or func.id not in eligible:
                continue
            kwargs = _literal_kwargs(call, func.id)
            if kwargs is None:
                continue
            out.append(
                _ExtractedCall(
                    sandbox_name=func.id,
                    kwargs=kwargs,
                    line_start=statement.lineno,
                    line_end=statement.end_lineno or statement.lineno,
                )
            )
    return out


def _literal_kwargs(call: ast.Call, name: str) -> dict[str, object] | None:
    """Return the call's arguments as literal keyword values, or `None` if any are not.

    Positional arguments are never speculated: the sandbox rejects them at execution time, so an
    early launch would run a call the real snippet cannot claim.
    """
    if not isinstance(call.func, ast.Name) or call.func.id != name or call.args:
        return None
    kwargs: dict[str, object] = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            return None
        try:
            kwargs[keyword.arg] = ast.literal_eval(keyword.value)
        except ValueError:
            return None
    return kwargs


def _close_paren(code: str, start: int) -> int | None:
    """Return the index just past the paren that closes `code[start]`, or `None` if still open.

    A small scanner rather than a parse: the enclosing statement is usually incomplete, so only
    the call expression itself can be balanced. Tracks nesting across all bracket kinds and skips
    string literals (with escapes); the extracted span is verified by `ast.parse` afterwards, so
    the scanner only has to find a plausible end, not validate syntax.
    """
    depth = 0
    quote: str | None = None
    i = start
    while i < len(code):
        ch = code[i]
        if quote is not None:
            if ch == '\\':
                i += 2
                continue
            if ch == quote:
                quote = None
        elif ch in '"\'':
            quote = ch
        elif ch in '([{':
            depth += 1
        elif ch in ')]}':
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return None


def _text_literal_calls(code: str, eligible: frozenset[str]) -> tuple[list[_ExtractedCall], int]:
    """Extract complete eligible calls straight from streamed text, closed statement or not.

    This is what makes launches fire as soon as a call finishes streaming: `_literal_calls` only
    sees statements the line-conservative scanner has closed, which for a call inside a compound
    statement (an `if`/`else` arm) can trail the call's own text by many lines. Here a call is
    launchable once its closing paren has streamed, wherever its statement stands.

    Text-level extraction cannot see context, so a call spelled inside a string literal, a
    comment, or a `def` body can launch too. Those launches waste a pure call and are evicted at
    completion; the `def`/attribute lookbehind filters the two cheap-to-catch shapes.

    Also returns the characters the paren scanner walked, so the caller can charge it against
    the scan budget: nested calls make the walks overlap, and the sum is what bounds the work.
    """
    out: list[_ExtractedCall] = []
    walked = 0
    for name in eligible:
        for match in re.finditer(rf'\b{re.escape(name)}\s*\(', code):
            before = code[: match.start()]
            if before.rstrip().endswith('.') or re.search(r'\bdef\s*$', before):
                continue
            end = _close_paren(code, match.end() - 1)
            if end is None:
                # Every later occurrence sits inside this still-open call, so none of them is a
                # launchable top-level call yet; stopping here keeps one scan linear.
                walked += len(code) - match.start()
                break
            walked += end - match.start()
            try:
                expression = ast.parse(code[match.start() : end], mode='eval')
            except (SyntaxError, ValueError, RecursionError, MemoryError):
                # Same set `closed_statements` guards: a NUL character is a `ValueError`, not
                # a `SyntaxError`, and adversarial nesting can exhaust the parser.
                continue
            if not isinstance(expression.body, ast.Call):
                continue  # pragma: no cover - a `name(...)` span that parses is always a Call
            kwargs = _literal_kwargs(expression.body, name)
            if kwargs is None:
                continue
            out.append(
                _ExtractedCall(
                    sandbox_name=name,
                    kwargs=kwargs,
                    line_start=before.count('\n') + 1,
                    line_end=code[:end].count('\n') + 1,
                )
            )
    return out, walked


@dataclass(kw_only=True)
class SpeculationCoordinator(Generic[AgentDepsT]):
    """Per-run speculation store shared between the stream watcher and `run_code` dispatch.

    `CodeMode` creates one per run and hands it to its `CodeModeToolset`. The toolset's
    `observe_stream_event` feeds streamed `run_code` argument deltas in; its dispatch path claims
    matching in-flight calls out. Events are emitted through the `RunContext` each side holds.
    """

    allowlist: SpeculationAllowlist
    stats: SpeculationStats
    launch_cap: int = MAX_SPECULATIONS_PER_PART
    """Most launches one `run_code` part may start; `CodeMode` lowers it to its `max_tool_calls`.

    Launches are host tasks the snippet has not asked for yet, so they cannot reserve from the
    part's nested-call budget; capping them at that budget bounds the total either way.
    """

    _step: _StepIngredients[AgentDepsT] | None = field(default=None, init=False)
    _parts: dict[str, _PartWatch] = field(default_factory=dict[str, _PartWatch], init=False)
    _index_to_part: dict[int, str] = field(default_factory=dict[int, str], init=False)
    _run_step: int | None = field(default=None, init=False)
    _tool_part_indexes: set[int] = field(default_factory=set[int], init=False)

    def stash_step(
        self,
        *,
        wrapped: AbstractToolset[AgentDepsT],
        wrapped_tools: dict[str, ToolsetTool[AgentDepsT]],
        sanitized_to_original: dict[str, str],
        callable_defs: dict[str, ToolDefinition],
    ) -> None:
        """Record this step's dispatch ingredients; called by `CodeModeToolset.get_tools`.

        Eligibility is resolved here: allowlisted names, minus `sequential` tools, whose
        rendering as `def` gives execution an ordering contract that an early launch would break.
        """
        if self.allowlist == 'declared':
            eligible = frozenset(
                name
                for name, tool_def in callable_defs.items()
                if not tool_def.sequential and declares_speculation_safety(tool_def)
            )
        else:
            allowlist = self.allowlist
            eligible = frozenset(
                name
                for name, tool_def in callable_defs.items()
                if not tool_def.sequential and sanitized_to_original.get(name, name) in allowlist
            )
        self._step = _StepIngredients(
            wrapped=wrapped,
            wrapped_tools=wrapped_tools,
            sanitized_to_original=sanitized_to_original,
            eligible=eligible,
        )

    # -- streaming side ---------------------------------------------------------------------

    async def observe(self, event: AgentStreamEvent, ctx: RunContext[AgentDepsT]) -> None:
        """Feed one stream event; launches tasks for any newly speculatable calls."""
        if self._run_step != ctx.run_step:
            self._run_step = ctx.run_step
            self._tool_part_indexes.clear()
            await self._retire_stale(ctx)
        if isinstance(event, PartStartEvent) and not isinstance(event.part, ToolCallPart):
            self._tool_part_indexes.discard(event.index)
        if isinstance(event, PartStartEvent) and isinstance(event.part, ToolCallPart):
            preceded = bool(self._tool_part_indexes - {event.index})
            self._tool_part_indexes.add(event.index)
            if preceded:
                # Earlier calls may mutate the state this snippet reads. Prelaunch at normal
                # dispatch instead, after the tool manager has enforced execution ordering.
                self._index_to_part.pop(event.index, None)
                return
        match event:
            case PartStartEvent(part=ToolCallPart() as part) if part.tool_name == _RUN_CODE_TOOL_NAME:
                watch = _PartWatch(tool_call_id=part.tool_call_id, run_step=ctx.run_step)
                if isinstance(part.args, str):
                    watch.args_text = part.args
                elif isinstance(part.args, dict):
                    watch.args_dict = dict(part.args)
                self._parts[part.tool_call_id] = watch
                self._index_to_part[event.index] = part.tool_call_id
                await self._scan(watch, ctx)
            case PartStartEvent():
                self._index_to_part.pop(event.index, None)
            case PartDeltaEvent(delta=ToolCallPartDelta() as delta):
                watch = self._watch_at(event.index, delta.tool_call_id)
                if watch is not None:
                    if isinstance(delta.args_delta, str):
                        watch.args_text += delta.args_delta
                    elif isinstance(delta.args_delta, dict):
                        watch.args_dict = {**(watch.args_dict or {}), **delta.args_delta}
                    await self._scan(watch, ctx)
            case PartEndEvent():
                watch = self._watch_at(event.index, None)
                if watch is not None:
                    # The arguments are complete: every statement is closed now, including the
                    # trailing ones the line-conservative scanner held back (streamed code rarely
                    # ends with a newline, so without this a snippet's last statements never
                    # launch and their dispatches go cold).
                    await self._scan(watch, ctx, final=True)
            case _:
                pass
        await self._emit_settles(ctx)

    async def _retire_stale(self, ctx: RunContext[AgentDepsT]) -> None:
        """Evict launches no snippet can legitimately claim any more.

        A failed snippet keeps its launches for the retry, which is the next model step. By the
        step after that the retry has either claimed them or gone cold, so anything still queued
        is stale: adopting it later would hand an unrelated snippet a result from minutes ago.
        """
        for part_id, watch in list(self._parts.items()):
            if watch.run_step >= ctx.run_step - 1:
                continue
            del self._parts[part_id]
            await self._evict(replace(ctx, tool_call_id=part_id, tool_name=_RUN_CODE_TOOL_NAME), watch)

    def _watch_at(self, part_index: int, tool_call_id: str | None) -> _PartWatch | None:
        """Route a delta by part index, following a call id the provider rewrites mid-stream.

        Without the re-key, the executed call would look up a part the stream never recorded
        under that id, and the launches would outlive the snippet instead of being evicted.
        """
        indexed_id = self._index_to_part.get(part_index)
        if indexed_id is None:
            return None
        watch = self._parts[indexed_id]
        if tool_call_id is not None and tool_call_id != indexed_id:
            del self._parts[indexed_id]
            self._parts[tool_call_id] = watch
            self._index_to_part[part_index] = tool_call_id
            watch.tool_call_id = tool_call_id
        return watch

    @property
    def _current_step(self) -> _StepIngredients[AgentDepsT]:
        step = self._step
        assert step is not None, '`get_tools` primes the step before the model streams or `run_code` dispatches'
        return step

    @staticmethod
    def _run_is_sequential(ctx: RunContext[AgentDepsT]) -> bool:
        """Whether the run opted every tool call into serial execution, which launches would break."""
        tool_manager = ctx.tool_manager
        return tool_manager is not None and global_mode_is_sequential(tool_manager.get_parallel_execution_mode)

    async def _scan(self, watch: _PartWatch, ctx: RunContext[AgentDepsT], *, final: bool = False) -> None:
        step = self._current_step
        if not step.eligible or watch.halted or self._run_is_sequential(ctx):
            return
        code = self._decode(watch)
        if code is None:
            return
        part_ctx = replace(ctx, tool_call_id=watch.tool_call_id, tool_name=_RUN_CODE_TOOL_NAME)
        newlines = code.count('\n')
        if not final and newlines == watch.scanned_newlines:
            # No line boundary since the last scan: nothing can have closed. Report the grown
            # code for live rendering and skip the parse work.
            await part_ctx.emit(SpeculativeCodeUpdateEvent(code=code, closed_statements=watch.closed_count))
            return
        watch.scanned_newlines = newlines
        watch.closed_count = self._count_closed(code, final=final)
        await part_ctx.emit(SpeculativeCodeUpdateEvent(code=code, closed_statements=watch.closed_count))
        # Launch from the raw text, not the closed statements: a call is ready once its closing
        # paren has streamed, even while its enclosing statement (an `if` arm, a `with` body) is
        # still being generated. Rescans recount every occurrence in the grown prefix, so
        # `demanded` is reconciled to the count rather than incremented.
        extracted_calls, walked = _text_literal_calls(self._ordered_prefix(code), step.eligible)
        if not self._charge_scan_work(watch, walked):
            return
        seen: dict[str, int] = {}
        for extracted in extracted_calls:
            key = _canonical_key(extracted.sandbox_name, extracted.kwargs)
            seen[key] = seen.get(key, 0) + 1
            if seen[key] <= watch.demanded.get(key, 0):
                continue
            if watch.launched >= self.launch_cap:
                return
            watch.demanded[key] = seen[key]
            if watch.demanded[key] <= self._in_flight(key):
                # Already covered, usually by a failed attempt's surviving launch; the snippet
                # claims it at execution. If another part claims it first, the execution
                # prefetch launches the deficit.
                continue
            await self._launch(part_ctx, watch, step, extracted, phase='streaming')

    def _decode(self, watch: _PartWatch) -> str | None:
        """The code streamed so far, or `None` when it is undecodable or over the scan budget."""
        if watch.args_dict is not None:
            code = watch.args_dict.get('code')
            if not isinstance(code, str):
                return None
        else:
            if len(watch.args_text) > MAX_SCAN_CHARS:
                watch.halted = True
                return None
            args = decode_partial_args(watch.args_text)
            if args is None:
                return None
            decoded = args.get('code')
            if not isinstance(decoded, str):
                return None
            code = decoded
        # Every scan rereads the whole prefix, so cumulative work is quadratic in snippet length
        # without this bound.
        return code if self._charge_scan_work(watch, len(code)) else None

    @staticmethod
    def _charge_scan_work(watch: _PartWatch, chars: int) -> bool:
        """Account host-side parsing against the part's budget; a part past it stays whole until dispatch."""
        if chars > MAX_SCAN_WORK_CHARS - watch.scan_work_chars:
            watch.halted = True
            return False
        watch.scan_work_chars += chars
        return True

    @staticmethod
    def _count_closed(code: str, *, final: bool) -> int:
        if not final:
            return len(closed_statements(code))
        try:
            return len(ast.parse(code).body)
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            return len(closed_statements(code))

    def _in_flight(self, key: str) -> int:
        return sum(len(watch.calls.get(key, ())) for watch in self._parts.values())

    async def _launch(
        self,
        ctx: RunContext[AgentDepsT],
        watch: _PartWatch,
        step: _StepIngredients[AgentDepsT],
        extracted: _ExtractedCall,
        *,
        phase: Literal['streaming', 'execution'],
    ) -> None:
        sandbox_name = extracted.sandbox_name
        original_name = step.sanitized_to_original.get(sandbox_name, sandbox_name)
        parent_manager = ctx.tool_manager
        tool_manager = ToolManager(
            toolset=step.wrapped,
            root_capability=parent_manager.root_capability if parent_manager is not None else None,
            ctx=ctx,
            tools=step.wrapped_tools,
        )
        watch.launched += 1
        self.stats.launched += 1
        launch_id = f'{watch.tool_call_id}__spec_{watch.launched}'
        call_part = ToolCallPart(tool_name=original_name, args=extracted.kwargs, tool_call_id=launch_id)
        call = SpeculativeCall(
            sandbox_name=sandbox_name,
            original_name=original_name,
            kwargs=extracted.kwargs,
            task=asyncio.ensure_future(run_nested_call(tool_manager, call_part)),
            launch_id=launch_id,
            started_at=time.perf_counter(),
        )
        call.task.add_done_callback(lambda _task, call=call: setattr(call, 'settled_at', time.perf_counter()))
        watch.calls.setdefault(_canonical_key(sandbox_name, extracted.kwargs), deque()).append(call)
        await ctx.emit(
            SpeculativeCallLaunchedEvent(
                launch_id=launch_id,
                sandbox_function=sandbox_name,
                wrapped_tool_name=original_name,
                arguments=extracted.kwargs,
                line_start=extracted.line_start,
                line_end=extracted.line_end,
                phase=phase,
            )
        )

    async def _emit_settles(self, ctx: RunContext[AgentDepsT]) -> None:
        """Report launches that finished since the watcher last saw stream traffic."""
        for watch in self._parts.values():
            for queue in watch.calls.values():
                for call in queue:
                    if call.settle_emitted or not call.task.done():
                        continue
                    call.settle_emitted = True
                    await replace(ctx, tool_call_id=watch.tool_call_id, tool_name=_RUN_CODE_TOOL_NAME).emit(
                        SpeculativeCallSettledEvent(
                            launch_id=call.launch_id,
                            outcome='failed' if call.settled_state() == 'failed' else 'ready',
                            elapsed_ms=call.elapsed_ms(),
                        )
                    )

    # -- execution side ---------------------------------------------------------------------

    def _ordered_prefix(self, code: str) -> str:
        """Do not speculate beyond a tool whose effects are not declared safe."""
        step = self._current_step
        names = set(step.wrapped_tools) | step.sanitized_to_original.keys()
        for name in names - step.eligible:
            match = re.search(rf'\b{re.escape(name)}\s*\(', code)
            if match is not None:
                code = code[: match.start()]
        return code

    async def prelaunch_for_execution(
        self, ctx: RunContext[AgentDepsT], execution: RunCodeExecution, code: str
    ) -> None:
        """Launch every literal eligible call the snippet holds, before the sandbox takes a step.

        Sequential `await`s execute one statement at a time, so a cold eligible call blocks every
        statement after it. At execution start the code is complete: launching the calls the
        stream watcher never saw (single-chunk argument deltas, provider quirks) means the
        snippet's awaits collect from tasks that are all already running, and wall time
        approaches the longest call instead of the sum. Only the deficit against launches already
        in flight is started, so FIFO multiplicity stays exact.
        """
        step = self._current_step
        if not step.eligible or self._run_is_sequential(ctx) or len(code) > MAX_SCAN_CHARS:
            # Oversized snippets skip host-side parsing entirely, like the stream scan: the
            # sandbox parser applies its own resource limits at dispatch.
            return
        try:
            body = ast.parse(self._ordered_prefix(code)).body
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            return
        extracted_calls = _literal_calls(body, step.eligible)
        if not extracted_calls:
            return
        watch = self._watch_for(ctx, execution)
        demanded: dict[str, int] = {}
        for extracted in extracted_calls:
            key = _canonical_key(extracted.sandbox_name, extracted.kwargs)
            demanded[key] = demanded.get(key, 0) + 1
            if demanded[key] <= self._in_flight(key):
                continue
            if watch.launched >= self.launch_cap:
                return
            await self._launch(ctx, watch, step, extracted, phase='execution')

    def _watch_for(self, ctx: RunContext[AgentDepsT], execution: RunCodeExecution) -> _PartWatch:
        """The watch for an executing part, created when the stream never showed the part."""
        parent_id = execution.parent_tool_call_id
        return self._parts.setdefault(parent_id, _PartWatch(tool_call_id=parent_id, run_step=ctx.run_step))

    def eligible(self, sandbox_name: str) -> bool:
        """Whether this sandbox function may speculate this step; drives miss reporting."""
        return sandbox_name in self._current_step.eligible

    def claim(self, parent_tool_call_id: str, sandbox_name: str, kwargs: dict[str, object]) -> SpeculativeCall | None:
        """Pop the oldest in-flight launch matching this dispatch, if any.

        Prefers the watch recorded under this part's id, then falls back to any other watch
        holding an exact `(function, arguments)` match: some providers re-key a tool call between
        the streamed part and its executed form, which would otherwise turn every launch into a
        miss, and the purity promise makes an identical launch from another part interchangeable.
        """
        key = _canonical_key(sandbox_name, kwargs)
        primary = self._parts.get(parent_tool_call_id)
        watches = [primary] if primary is not None else []
        watches.extend(watch for part_id, watch in self._parts.items() if part_id != parent_tool_call_id)
        for watch in watches:
            queue = watch.calls.get(key)
            if queue:
                return queue.popleft()
        return None

    async def adopt(
        self,
        ctx: RunContext[AgentDepsT],
        execution: RunCodeExecution,
        claimed: SpeculativeCall,
        call_part: ToolCallPart,
    ) -> Any:
        """Resolve a dispatch from a call launched while the snippet was still streaming.

        Budget accounting already happened at dispatch, exactly as for a cold call. The launch ran
        under a provisional tool call id, so the history parts are recorded here under the real
        nested id instead.
        """
        execution.nested_calls[call_part.tool_call_id] = call_part
        ready_at_claim = claimed.task.done()
        outcome = await claimed.task
        self.stats.adopted += 1
        watch = self._watch_for(ctx, execution)
        watch.hits += 1
        watch.hidden_ms += claimed.elapsed_ms()
        await ctx.emit(
            SpeculativeCallClaimedEvent(
                launch_id=claimed.launch_id,
                nested_tool_call_id=call_part.tool_call_id,
                wrapped_tool_name=claimed.original_name,
                ready_at_claim=ready_at_claim,
                elapsed_ms=claimed.elapsed_ms(),
            )
        )
        return execution.finish(call_part, outcome)

    async def report_miss(
        self,
        ctx: RunContext[AgentDepsT],
        execution: RunCodeExecution,
        sandbox_name: str,
        call_part: ToolCallPart,
    ) -> None:
        """Record a speculation-eligible dispatch that found no launch and runs cold."""
        self._watch_for(ctx, execution).misses += 1
        await ctx.emit(
            SpeculativeCallMissedEvent(
                sandbox_function=sandbox_name,
                wrapped_tool_name=call_part.tool_name,
                nested_tool_call_id=call_part.tool_call_id,
            )
        )

    async def evict_part(self, ctx: RunContext[AgentDepsT], parent_tool_call_id: str) -> dict[str, object] | None:
        """Drop unclaimed launches for one `run_code` part that ran to completion, and summarize it.

        Whatever was not claimed (wrong-branch conditionals, rewritten lines) is garbage for this
        part. Failed snippets keep their launches: syntax and type errors fail before any
        dispatch, and the retry claims the survivors under its fresh tool call id.
        """
        watch = self._parts.pop(parent_tool_call_id, None)
        if watch is None:
            return None
        evicted = await self._evict(ctx, watch)
        if not watch.hits and not watch.misses and not evicted:
            return None
        return {
            'hits': watch.hits,
            'hidden_ms': round(watch.hidden_ms, 3),
            'misses': watch.misses,
            'wasted': len(evicted),
        }

    async def close(self) -> None:
        """Run-end cleanup: cancel every launch no snippet ever claimed.

        Shielded: a run cancelled through an anyio scope is level-triggered, so without the
        shield the first ``_cancel_watch`` await below would be re-cancelled and every later
        watch's launches would keep running past the run's end. Each ``_cancel_watch`` await is
        bounded by ``CANCEL_TIMEOUT_SECONDS``, so the shield cannot hold the unwind hostage.
        """
        parts, self._parts = self._parts, {}
        self._index_to_part.clear()
        with anyio.CancelScope(shield=True):
            for watch in parts.values():
                await self._cancel_watch(watch)

    async def _evict(
        self, ctx: RunContext[AgentDepsT], watch: _PartWatch
    ) -> list[tuple[SpeculativeCall, Literal['pending', 'ready', 'failed']]]:
        """Cancel a watch's unclaimed launches and report each one."""
        evicted = await self._cancel_watch(watch)
        for call, state in evicted:
            await ctx.emit(
                SpeculativeCallEvictedEvent(
                    launch_id=call.launch_id,
                    wrapped_tool_name=call.original_name,
                    state=state,
                )
            )
        return evicted

    async def _cancel_watch(
        self, watch: _PartWatch
    ) -> list[tuple[SpeculativeCall, Literal['pending', 'ready', 'failed']]]:
        evicted: list[tuple[SpeculativeCall, Literal['pending', 'ready', 'failed']]] = []
        for queue in watch.calls.values():
            for call in queue:
                # Capture where the launch stood before cancellation rewrites it.
                evicted.append((call, call.settled_state()))
                call.task.cancel()
                self.stats.evicted += 1
        watch.calls.clear()
        if evicted:
            # Wait for the cancellations to unwind before the run moves on, but not forever: a
            # tool that swallows the cancellation would otherwise hold the `run_code` result
            # hostage. Abandoning such a task is safe, it starts no further tool calls.
            await asyncio.wait({call.task for call, _ in evicted}, timeout=CANCEL_TIMEOUT_SECONDS)
        return evicted
