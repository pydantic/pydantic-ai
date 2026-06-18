"""`OverflowingToolOutput` -- reduce oversized tool returns at production time."""

from __future__ import annotations

import warnings
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeGuard

from pydantic_ai import FunctionToolset
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import ToolCallPart, ToolReturn, ToolReturnContent, UserContent
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition, ToolSelector, matches_tool_selector
from pydantic_ai.toolsets import AgentToolset

from pydantic_ai_harness.experimental.overflow._bands import (
    Action,
    Band,
    Passthrough,
    Spill,
    Summarize,
    Truncate,
)
from pydantic_ai_harness.experimental.overflow._payload import (
    is_binary,
    json_sketch,
    measure,
    strip_ansi,
    to_bytes,
    to_text,
    truncate_text,
)
from pydantic_ai_harness.experimental.overflow._store import LocalFileStore, OverflowStore

READ_TOOL_NAME = 'read_tool_result'
"""Name of the registered read-back tool. Its own returns are exempt from reduction."""

_DEFAULT_THRESHOLD = 10_000
"""Default band threshold (characters) -- below this, returns pass through untouched."""

_DEFAULT_SUMMARY_PROMPT = """\
The following output from the `{tool_name}` tool is too large to keep in full. Summarize it \
so the summary carries everything needed to keep working: concrete values, identifiers, \
errors, and structure. Respond ONLY with the summary, no preamble.

<output>
{output}
</output>\
"""


def _default_bands() -> list[Band]:
    """Lossless spill with a bounded truncation fallback: zero LLM cost, no silent drop."""
    return [Band(over=_DEFAULT_THRESHOLD, action=Spill(then=Truncate()))]


@dataclass
class _Unit:
    """One reducible piece of a tool return: its `return_value` or its `content`.

    `suffix` distinguishes the two so they spill to distinct handles for the same call.
    """

    binary: bool
    text: str | None
    data: bytes
    value: ToolReturnContent
    suffix: str


@dataclass
class OverflowingToolOutput(AbstractCapability[AgentDepsT]):
    """Reduce oversized tool returns when they are produced, persisting the reduction.

    A tool can return a payload large enough to dominate the context window. Tool returns
    persist in history, so an oversized one is re-sent on every later request. This
    capability intercepts a return in `after_tool_execute`, reduces it once, and lets the
    reduced form persist -- it is not recomputed per request.

    Three reduction modes, freely combined through an ordered list of size `bands`:

    - `Truncate`: clamp to a character budget. Lossy, zero-cost.
    - `Spill`: persist the full payload, hand the model a `read_tool_result` handle plus a
      preview. Lossless.
    - `Summarize`: size-gated LLM summary. Inherits the run's model by default.

    The first band whose `over` threshold the measured size meets wins; smaller returns pass
    through. `per_tool` replaces the band list for named tools; `tool_filter` scopes which
    tools are touched at all. The default is `Spill(then=Truncate())`: lossless when a store
    accepts the write, a bounded truncation otherwise.

    `ModelRetry` and other errors never reach this hook (they are raised, not returned), so
    error payloads the model needs to recover are never spilled or summarized.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_harness.experimental.overflow import (
            Band,
            OverflowingToolOutput,
            Spill,
            Summarize,
            Truncate,
        )

        agent = Agent(
            'openai:gpt-4o',
            capabilities=[
                OverflowingToolOutput(
                    bands=[
                        Band(over=100_000, action=Spill()),
                        Band(over=20_000, action=Summarize()),
                        Band(over=5_000, action=Truncate()),
                    ],
                )
            ],
        )
        ```
    """

    bands: Sequence[Band] = field(default_factory=_default_bands)
    """Ordered size bands. The first band whose `over` threshold is met wins."""

    per_tool: Mapping[str, Sequence[Band]] = field(default_factory=dict[str, Sequence[Band]])
    """Per-tool band lists that replace `bands` for the named tools."""

    tool_filter: ToolSelector[AgentDepsT] = 'all'
    """Which tools this capability touches. Non-matching tools always pass through."""

    over_tokens: bool = False
    """Measure band thresholds in estimated tokens instead of characters."""

    tokenizer: Callable[[str], int] | None = None
    """Optional `(str) -> int` tokenizer for `over_tokens`. Defaults to a ~4-char heuristic."""

    store: OverflowStore | None = None
    """Backend for spilled payloads. Defaults to a `LocalFileStore`."""

    strip_ansi: bool = False
    """Strip ANSI escape sequences from text returns before measuring and reducing."""

    summary_prompt: str = _DEFAULT_SUMMARY_PROMPT
    """Prompt template for `Summarize`. Must contain `{tool_name}` and `{output}`."""

    _store: OverflowStore = field(init=False, repr=False)
    _bands: list[Band] = field(init=False, repr=False)
    _per_tool: dict[str, list[Band]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._store = self.store if self.store is not None else LocalFileStore()
        self._bands = self._prepare_bands(self.bands)
        self._per_tool = {name: self._prepare_bands(bands) for name, bands in self.per_tool.items()}

    @staticmethod
    def _prepare_bands(bands: Sequence[Band]) -> list[Band]:
        """Validate thresholds and order bands largest-first so first-match means largest-fit."""
        for band in bands:
            if band.over < 0:
                raise ValueError('Band.over must be non-negative.')
        return sorted(bands, key=lambda b: b.over, reverse=True)

    # --- toolset ---

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        """Register the `read_tool_result` tool for reading spilled payloads on demand."""
        store = self._store

        async def read_tool_result(
            ctx: RunContext[AgentDepsT],
            handle: str,
            offset: int = 0,
            limit: int = 200,
            from_end: bool = False,
            pattern: str | None = None,
        ) -> str:
            """Read a slice of a spilled tool result.

            Args:
                ctx: The run context (supplied by the agent).
                handle: The handle from the overflowed tool return.
                offset: Number of matching lines to skip from the start (or end). Must be >= 0.
                limit: Maximum number of lines to return (>= 1; clamped to a built-in cap).
                from_end: Count `offset`/`limit` from the end of the result.
                pattern: Optional literal substring; only lines containing it are returned.
            """
            return await _read_slice(store, handle, offset, limit, from_end, pattern)

        return FunctionToolset([read_tool_result])

    # --- reduction ---

    async def after_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        result: Any,
    ) -> Any:
        """Reduce the tool result -- both `return_value` and model-visible `content`."""
        original: object = result
        if call.tool_name == READ_TOOL_NAME:
            return original
        if not await matches_tool_selector(self.tool_filter, ctx, tool_def):
            return original

        metadata: object
        if isinstance(result, ToolReturn):
            return_value: ToolReturnContent = result.return_value
            content = result.content
            metadata = result.metadata
            wrapped = True
        else:
            return_value = result
            content = None
            metadata = None
            wrapped = False

        if isinstance(return_value, BaseException):
            return original

        bands = self._per_tool.get(call.tool_name, self._bands)
        value_unit = self._make_unit(return_value, suffix='')
        value_text, value_handle = await self._reduce(ctx, call, bands, value_unit)
        content_text, content_handle = await self._reduce_content(ctx, call, bands, content)

        if value_text is None and content_text is None:
            return original

        return self._assemble(
            wrapped=wrapped,
            return_value=return_value,
            content=content,
            metadata=metadata,
            value_unit=value_unit,
            value_text=value_text,
            value_handle=value_handle,
            content_text=content_text,
            content_handle=content_handle,
        )

    def _assemble(
        self,
        *,
        wrapped: bool,
        return_value: ToolReturnContent,
        content: str | Sequence[UserContent] | None,
        metadata: object,
        value_unit: _Unit,
        value_text: str | None,
        value_handle: str | None,
        content_text: str | None,
        content_handle: str | None,
    ) -> object:
        """Rebuild the tool result from the reduced parts, preserving the envelope."""
        if wrapped:
            new_metadata = metadata
            if value_handle is not None or content_handle is not None:
                new_metadata = _with_handles(metadata, value_handle, len(value_unit.data), content_handle)
            wrapped_out: ToolReturn[object] = ToolReturn(
                return_value=value_text if value_text is not None else return_value,
                content=content_text if content_text is not None else content,
                metadata=new_metadata,
            )
            return wrapped_out

        # A plain (non-`ToolReturn`) result has no separate content part.
        if value_handle is not None:
            spilled_out: ToolReturn[object] = ToolReturn(
                return_value=value_text, metadata=_with_handles(None, value_handle, len(value_unit.data))
            )
            return spilled_out
        return value_text

    def _make_unit(self, value: ToolReturnContent, *, suffix: str) -> _Unit:
        """Pre-render a value into the text / bytes the reduction pipeline needs."""
        if is_binary(value):
            return _Unit(binary=True, text=None, data=to_bytes(value), value=value, suffix=suffix)
        text = to_text(value)
        if self.strip_ansi:
            text = strip_ansi(text)
        return _Unit(binary=False, text=text, data=text.encode('utf-8'), value=value, suffix=suffix)

    async def _reduce(
        self,
        ctx: RunContext[AgentDepsT],
        call: ToolCallPart,
        bands: Sequence[Band],
        unit: _Unit,
    ) -> tuple[str | None, str | None]:
        """Select a band for `unit` and apply it. Returns `(replacement, handle)`.

        `replacement` is None when the unit passes through unchanged; `handle` is set only
        when the unit was spilled.
        """
        size = (
            len(unit.data)
            if unit.binary
            else measure(unit.text or '', over_tokens=self.over_tokens, tokenizer=self.tokenizer)
        )
        action = _select_action(bands, size)
        if action is None:
            return None, None
        return await self._apply(ctx, call, action, unit)

    async def _reduce_content(
        self,
        ctx: RunContext[AgentDepsT],
        call: ToolCallPart,
        bands: Sequence[Band],
        content: str | Sequence[UserContent] | None,
    ) -> tuple[str | None, str | None]:
        """Reduce model-visible `content`. Text content is reduced; other content warns."""
        if content is None:
            return None, None
        if isinstance(content, str):
            return await self._reduce(ctx, call, bands, self._make_unit(content, suffix='.content'))

        text = ''.join(part for part in content if isinstance(part, str))
        size = measure(text, over_tokens=self.over_tokens, tokenizer=self.tokenizer)
        action = _select_action(bands, size)
        if action is not None and not isinstance(action, Passthrough):
            warnings.warn(
                f'OverflowingToolOutput: tool {call.tool_name!r} returned large non-text '
                f'content ({len(content)} parts); leaving it unreduced.',
                stacklevel=2,
            )
        return None, None

    async def _apply(
        self,
        ctx: RunContext[AgentDepsT],
        call: ToolCallPart,
        action: Action,
        unit: _Unit,
    ) -> tuple[str | None, str | None]:
        """Apply one action to a unit, falling back to its `then` when it cannot run."""
        if isinstance(action, Passthrough):
            return None, None

        if isinstance(action, Truncate):
            if unit.binary:
                return await self._fallback(ctx, call, action.then, unit)
            assert unit.text is not None
            return truncate_text(unit.text, action.max_chars, action.strategy), None

        if isinstance(action, Spill):
            return await self._spill(ctx, call, action, unit)

        return await self._summarize_action(ctx, call, action, unit)

    async def _fallback(
        self,
        ctx: RunContext[AgentDepsT],
        call: ToolCallPart,
        then: Action | None,
        unit: _Unit,
    ) -> tuple[str | None, str | None]:
        """Run the fallback action, or keep the unit unchanged when there is none."""
        if then is None:
            return None, None
        return await self._apply(ctx, call, then, unit)

    async def _spill(
        self,
        ctx: RunContext[AgentDepsT],
        call: ToolCallPart,
        action: Spill,
        unit: _Unit,
    ) -> tuple[str | None, str | None]:
        key = _handle_key(ctx, call, unit.suffix)
        try:
            handle = await self._store.write(key, unit.data)
        except Exception:
            return await self._fallback(ctx, call, action.then, unit)

        preview = _build_spill_preview(handle, unit, action.preview_chars, over_tokens=self.over_tokens)
        return preview, handle

    async def _summarize_action(
        self,
        ctx: RunContext[AgentDepsT],
        call: ToolCallPart,
        action: Summarize,
        unit: _Unit,
    ) -> tuple[str | None, str | None]:
        if unit.binary:
            return await self._fallback(ctx, call, action.then, unit)
        assert unit.text is not None
        try:
            summary = await self._summarize(ctx, call, action, unit.text)
        except Exception:
            return await self._fallback(ctx, call, action.then, unit)
        return summary, None

    async def _summarize(
        self,
        ctx: RunContext[AgentDepsT],
        call: ToolCallPart,
        action: Summarize,
        text: str,
    ) -> str:
        """Generate the summary via a custom callable or the inherited-model agent."""
        if action.summarize is not None:
            outcome = action.summarize(call.tool_name, text)
            if isinstance(outcome, Awaitable):
                return await outcome
            return outcome

        from pydantic_ai import Agent

        model = action.model if action.model is not None else ctx.model
        prompt = self.summary_prompt.format(tool_name=call.tool_name, output=text)
        agent: Agent[None, str] = Agent(model, instructions='You summarize oversized tool output.')
        run = await agent.run(prompt, usage=ctx.usage)
        return run.output.strip()


def _select_action(bands: Sequence[Band], size: int) -> Action | None:
    """Return the first (largest-threshold) band action whose threshold `size` meets."""
    for band in bands:
        if size >= band.over:
            return band.action
    return None


def _handle_key(ctx: RunContext[AgentDepsT], call: ToolCallPart, suffix: str = '') -> str:
    """Build a per-run, per-call, per-retry key so concurrent and retried calls never clash.

    `suffix` keeps a return's `return_value` and `content` spills on distinct handles.
    """
    run_id = ctx.run_id or 'run'
    call_id = call.tool_call_id or 'call'
    return f'{run_id}/{call_id}.{ctx.retry}{suffix}'


def _is_mapping(value: object) -> TypeGuard[Mapping[object, object]]:
    """`TypeGuard` so a mapping narrows to a known element type, not `Unknown`."""
    return isinstance(value, Mapping)


def _with_handles(
    existing: object,
    value_handle: str | None,
    value_bytes: int,
    content_handle: str | None = None,
) -> dict[str, object]:
    """Stash spill handle(s) in `ToolReturn.metadata` (app-only, costs no model tokens)."""
    base: dict[str, object] = {}
    if _is_mapping(existing):
        base.update(_copy_mapping(existing))
    if value_handle is not None:
        base['overflow_handle'] = value_handle
        base['overflow_bytes'] = value_bytes
    if content_handle is not None:
        base['overflow_content_handle'] = content_handle
    return base


def _copy_mapping(source: Mapping[object, object]) -> dict[str, object]:
    """Copy an arbitrary mapping with stringified keys (tool metadata is app-defined)."""
    return {str(key): source[key] for key in source}


def _build_spill_preview(handle: str, unit: _Unit, preview_chars: int, *, over_tokens: bool) -> str:
    """Compose the model-visible spill stand-in: marker, sketch, and a head/tail preview."""
    if unit.binary:
        size_desc = f'{len(unit.data):,} bytes (binary)'
        body = f'<{len(unit.data):,} bytes of binary data>'
        sketch = ''
    else:
        text = unit.text or ''
        size_unit = 'tokens' if over_tokens else 'chars'
        amount = measure(text, over_tokens=over_tokens, tokenizer=None) if over_tokens else len(text)
        size_desc = f'{amount:,} {size_unit}'
        body = _head_tail_preview(text, preview_chars)
        sketch = json_sketch(unit.value)

    header = (
        f'[Tool output too large ({size_desc}); stored to handle {handle!r}. '
        f'Read it with read_tool_result(handle={handle!r}, offset=0, limit=200, '
        f'from_end=False, pattern=None).]'
    )
    parts = [header]
    if sketch:
        parts.append(f'shape: {sketch}')
    parts.append(body)
    return '\n'.join(parts)


def _head_tail_preview(text: str, preview_chars: int) -> str:
    """Return a head+tail slice of `text` with a middle-elision marker."""
    if len(text) <= preview_chars:
        return text
    head_chars = preview_chars // 2
    tail_chars = preview_chars - head_chars
    omitted = len(text) - head_chars - tail_chars
    return f'{text[:head_chars]}\n...[{omitted:,} chars omitted]...\n{text[-tail_chars:]}'


_MAX_READ_LINES = 1_000
"""Hard cap on lines returned by one `read_tool_result` call."""

_MAX_READ_CHARS = 50_000
"""Hard cap on characters returned by one `read_tool_result` call."""


async def _read_slice(
    store: OverflowStore,
    handle: str,
    offset: int,
    limit: int,
    from_end: bool,
    pattern: str | None,
) -> str:
    """Filter and slice a spilled payload for `read_tool_result`, bounded in both axes.

    `pattern` is a literal substring (not a regex), so a model-supplied value cannot hang
    the host with catastrophic backtracking. `limit` is clamped and the joined output is
    capped, so one call can never return an unbounded amount of text.
    """
    if offset < 0:
        raise ModelRetry('`offset` must be >= 0.')
    if limit < 1:
        raise ModelRetry('`limit` must be >= 1.')
    limit = min(limit, _MAX_READ_LINES)

    try:
        data = await store.read(handle)
    except OSError:
        # Return, not raise: a wrong handle (e.g. the model passing a tool-call id) or a
        # result that is no longer stored must not consume a tool retry and escalate to a
        # fatal `UnexpectedModelBehavior`. Guide the model to a valid handle instead. The
        # exception is intentionally not echoed -- a store's error can carry the resolved
        # filesystem path or other backend detail the model has no need for.
        return (
            f'[No stored tool result for handle {handle!r}. Use the exact handle string from a '
            '"[Tool output too large ... stored to handle ...]" marker; if the result is no longer '
            'available, re-run the original tool.]'
        )

    lines = data.decode('utf-8', errors='replace').splitlines()
    if pattern is not None:
        lines = [line for line in lines if pattern in line]

    total = len(lines)
    if from_end:
        end = max(0, total - offset)
        window = lines[max(0, end - limit) : end]
    else:
        window = lines[offset : offset + limit]

    body = '\n'.join(window)
    capped = ''
    if len(body) > _MAX_READ_CHARS:
        body = body[:_MAX_READ_CHARS]
        capped = ', output capped'
    header = f'[handle {handle!r}: {total:,} matching line(s); showing {len(window)}{capped}]'
    return f'{header}\n{body}' if body else header
