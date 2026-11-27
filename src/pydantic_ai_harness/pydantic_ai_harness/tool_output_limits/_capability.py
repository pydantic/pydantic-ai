"""`ToolOutputLimits` -- reduce oversized tool returns at production time."""

from __future__ import annotations

import base64
import codecs
import logging
import posixpath
import re
import sys
import warnings
from collections import deque
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Iterable, Mapping, Sequence
from contextlib import aclosing
from dataclasses import KW_ONLY, dataclass, field
from typing import Any

from pydantic_ai import FunctionToolset, Tool, ToolKind
from pydantic_ai._framing import second_model_prompt
from pydantic_ai._utils import replace_no_init
from pydantic_ai.capabilities import AbstractCapability, durable_operation
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.messages import (
    LoadCapabilityReturnPart,
    ToolCallPart,
    ToolReturn,
    ToolReturnContent,
    ToolSearchReturnPart,
    UserContent,
    is_multi_modal_content,
)
from pydantic_ai.models import Model
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition, ToolSelector, matches_tool_selector
from pydantic_ai.toolsets import AgentToolset
from pydantic_ai.workspaces import WorkspaceError, WorkspaceReadOnlyError
from pydantic_ai_harness._line_slice import MAX_READ_CHARS, MAX_READ_LINES, ReadUnit
from pydantic_ai_harness._models import is_request_response_model
from pydantic_ai_harness._typeguards import is_mapping, is_object_list, is_object_list_or_tuple
from pydantic_ai_harness._usage import reserved_usage_limits
from pydantic_ai_harness._work_log_kinds import WORK_LOG_READ_KIND, WORK_LOG_SEARCH_KIND
from pydantic_ai_harness._workspace import METADATA_DIR, raise_tool_failure
from pydantic_ai_harness.file_system import READ_FILE_KIND
from pydantic_ai_harness.file_system._tool_kinds import step_aside_for_read_file
from pydantic_ai_harness.tool_output_limits._bands import (
    Action,
    Band,
    Passthrough,
    Spill,
    Summarize,
    Truncate,
)
from pydantic_ai_harness.tool_output_limits._payload import (
    RERUN_NOTE,
    Serializer,
    is_binary,
    json_sketch,
    measure,
    strip_ansi,
    to_bytes,
    to_text,
    truncate_text,
)
from pydantic_ai_harness.tool_output_limits._store import SPILL_DIR, OverflowStore, WorkspaceOverflowStore, iter_payload

logger = logging.getLogger(__name__)

READ_TOOL_NAME = 'read_tool_result'
"""Default name of the registered read-back tool."""

READ_TOOL_RESULT_KIND = ToolKind('tool_output_limits', 'read_tool_result')
"""The kind of the `read_tool_result` tool, whatever it is named. Its returns are exempt from reduction."""

GREP_TOOL_NAME = 'grep_tool_result'
"""Default name of the registered search tool."""

GREP_TOOL_RESULT_KIND = ToolKind('tool_output_limits', 'grep_tool_result')
"""The kind of the `grep_tool_result` tool, whatever it is named. Its returns are exempt from reduction."""

_EXEMPT_KINDS = frozenset(
    {
        READ_TOOL_RESULT_KIND.id,
        GREP_TOOL_RESULT_KIND.id,
        WORK_LOG_READ_KIND.id,
        WORK_LOG_SEARCH_KIND.id,
        LoadCapabilityReturnPart.tool_kind,
        ToolSearchReturnPart.tool_kind,
    }
)
"""Kinds whose returns are never reduced, whatever `tool_filter` says.

The spill query tools and `WorkLog`'s `search_work_log` and `read_work_log` return a bounded page of a
stored result, so reducing one would store the page a second time. The others are core's `load_capability` and
`search_tools`: core re-reads their structured content from history to decide which capabilities are
loaded and which tools are revealed, so a reduced return would leave those locked.
"""

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


def _spills(action: Action | None) -> bool:
    """Whether `action`, or a fallback it chains to, spills to the store."""
    while action is not None:
        if isinstance(action, Spill):
            return True
        action = _next_action(action)
    return False


def _default_bands() -> list[Band]:
    """Spill to the store, falling back to truncation when the store write fails."""
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
class ToolOutputLimits(AbstractCapability[AgentDepsT]):
    """Reduce oversized tool returns when they are produced, persisting the reduction.

    A tool can return a payload large enough to dominate the context window. Tool returns
    persist in history, so an oversized one is re-sent on every later request. This
    capability intercepts a return in `after_tool_execute`, reduces it once, and lets the
    reduced form persist -- it is not recomputed per request.

    Three reduction modes, freely combined through an ordered list of size `bands`:

    - `Truncate`: clamp to a character budget. Lossy, zero-cost.
    - `Spill`: persist the full payload, hand the model a handle plus a preview. Lossless.
      The model reads it back with `read_tool_result`, or, when the spill is in the run's
      workspace and `FileSystem` is on the agent, with its `read_file`.
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
        from pydantic_ai_harness.tool_output_limits import (
            Band,
            LocalFileOverflowStore,
            ToolOutputLimits,
            Spill,
            Summarize,
            Truncate,
        )

        agent = Agent(
            'openai:gpt-4o',
            capabilities=[
                ToolOutputLimits(
                    bands=[
                        Band(over=100_000, action=Spill()),
                        Band(over=20_000, action=Summarize()),
                        Band(over=5_000, action=Truncate()),
                    ],
                    store=LocalFileOverflowStore(),
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
    """Optional `(str) -> int` tokenizer for `over_tokens`. Defaults to the compaction text estimate."""

    store: OverflowStore | WorkspaceOverflowStore | None = None
    """Backend for spilled payloads.

    Defaults to a `WorkspaceOverflowStore` in the run's workspace. When the bands can spill and no store
    has a workspace to write to, the run fails at its start: attach a workspace, or pass
    `store=LocalFileOverflowStore()` to keep spills on this machine.
    """

    strip_ansi: bool = False
    """Strip ANSI escape sequences from text returns before measuring and reducing."""

    summary_prompt: str = _DEFAULT_SUMMARY_PROMPT
    """Prompt template for `Summarize`. Must contain `{tool_name}` and `{output}`."""

    serializer: Serializer | None = None
    """Render a structured (non-string, non-binary) return to the text that is measured,
    previewed, spilled, and read back. When unset, structured returns render as compact
    JSON, which puts the whole value on one line; the `indented_json` and `json_lines`
    presets make large spills pageable by line through `read_tool_result`. A return that
    stays under every band threshold passes through as the original object, so the
    serialized text is only model-visible once a band triggers."""

    # Override the inherited default ID because durable-operation recovery needs a stable identity.
    _: KW_ONLY
    id: str | None = 'tool_output_limits'

    _store: OverflowStore | WorkspaceOverflowStore = field(init=False, repr=False)
    _bands: list[Band] = field(init=False, repr=False)
    _per_tool: dict[str, list[Band]] = field(init=False, repr=False)
    _toolset: AgentToolset[AgentDepsT] | None = field(default=None, init=False, repr=False, compare=False)
    _spilled_calls: dict[str, int] = field(default_factory=dict[str, int], init=False, repr=False, compare=False)
    """Spill name -> the run step that first wrote it, on the per-run instance `for_run` returns.

    A `tool_call_id` is only unique within one response, so a later response can reuse one. The step
    of a name's first spill is what tells the two apart: the call that wrote it first keeps the
    documented `run_id/tool_call_id.retry` handle, and a later call that would land on the same name
    is parked one level down. Kept on the instance rather than read off the history, which a
    capability such as a compaction may rewrite mid-run.
    """

    def __post_init__(self) -> None:
        self._store = self.store if self.store is not None else WorkspaceOverflowStore()
        self._bands = self._prepare_bands(self.bands)
        self._per_tool = {name: self._prepare_bands(bands) for name, bands in self.per_tool.items()}

    @classmethod
    def combine(cls, capabilities: Sequence[AbstractCapability[AgentDepsT]]) -> AbstractCapability[AgentDepsT]:
        """Merge the configuration, with the later band winning a shared threshold, then re-derive from it.

        `_store`, `_bands`, and `_per_tool` are derived in `__post_init__`, which merging skips, so
        they are rebuilt from the merged configuration: a merged band list keeps the largest-first
        order a single instance has, and a stated `store` is the one spills go to.
        """
        limits = [capability for capability in capabilities if isinstance(capability, ToolOutputLimits)]
        assert len(limits) == len(capabilities)
        merged = super().combine(capabilities)
        assert isinstance(merged, cls)
        bands: list[Band] = []
        for capability in limits:
            thresholds = {band.over for band in capability.bands}
            bands = [band for band in bands if band.over not in thresholds] + list(capability.bands)
        merged = replace_no_init(merged, bands=bands)
        merged.__post_init__()
        return merged

    @staticmethod
    def _prepare_bands(bands: Sequence[Band]) -> list[Band]:
        """Validate thresholds and order bands largest-first so first-match means largest-fit."""
        for band in bands:
            if band.over < 0:
                raise ValueError('Band.over must be non-negative.')
        return sorted(bands, key=lambda b: b.over, reverse=True)

    async def before_run(self, ctx: RunContext[AgentDepsT]) -> None:
        """Fail the run at its start when a band can spill but no store has a workspace to write to."""
        store = self._store
        if (
            isinstance(store, WorkspaceOverflowStore)
            and store.workspace is None
            and not ctx.workspace.attached
            and any(_spills(band.action) for bands in (self._bands, *self._per_tool.values()) for band in bands)
        ):
            raise UserError(
                "`ToolOutputLimits` spills oversized tool output to the run's workspace, but none is attached "
                "to this run. Attach one, such as `LocalWorkspace('.')`, or pass `store=LocalFileOverflowStore()` "
                'to keep spills on this machine.'
            )

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> ToolOutputLimits[AgentDepsT]:
        """Return a copy with no spill names recorded, so runs don't share them.

        A shallow copy rather than `replace`, which would rebuild the toolset that `get_toolset`
        builds once for durable execution.
        """
        return replace_no_init(self, _spilled_calls={})

    @property
    def _spills_to_run_workspace(self) -> bool:
        """Whether spills are files in the run's workspace, which the run's file tools can read.

        Known from configuration: the default `WorkspaceOverflowStore` without a `workspace` of its own.
        """
        store = self._store
        return isinstance(store, WorkspaceOverflowStore) and store.workspace is None

    def _store_for(self, ctx: RunContext[AgentDepsT]) -> OverflowStore:
        store = self._store
        return store.bind(ctx.workspace) if isinstance(store, WorkspaceOverflowStore) else store

    # --- toolset ---

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        """Register the `read_tool_result` and `grep_tool_result` tools for querying spilled payloads on demand.

        When spills are files in the run's workspace, `read_tool_result` steps aside for a tool of
        `FileSystem`'s `READ_FILE_KIND`, which reads them too. Built once, so durable execution
        sees the toolset it registered.
        """
        if self._toolset is None:
            self._toolset = self._make_toolset()
        return self._toolset

    def _make_toolset(self) -> AgentToolset[AgentDepsT]:

        async def read_tool_result(
            ctx: RunContext[AgentDepsT],
            handle: str,
            offset: int = 0,
            limit: int = 200,
            from_end: bool = False,
            pattern: str | None = None,
            unit: ReadUnit = 'lines',
        ) -> str:
            """Read a slice of a spilled tool result.

            Args:
                ctx: The run context (supplied by the agent).
                handle: The handle from the overflowed tool return.
                offset: Number of matching lines (or bytes) to skip from the start (or end). Must be >= 0.
                limit: Maximum number of lines (or bytes) to return (>= 1; clamped to a built-in cap).
                from_end: Count `offset`/`limit` from the end of the result.
                pattern: Optional literal substring; only lines containing it are returned.
                unit: `'bytes'` slices raw bytes, for output without useful line breaks such as minified JSON;
                    `offset` and `limit` then count bytes, so pass a larger `limit` (up to 50,000).
            """
            store = self._store_for(ctx)
            return await _read_slice(store, handle, offset, limit, from_end, pattern, unit, _reader_name(ctx))

        async def grep_tool_result(
            ctx: RunContext[AgentDepsT],
            handle: str,
            pattern: str,
            context: int = 2,
            max_matches: int = 20,
            offset: int = 0,
        ) -> str:
            """Find the lines of a spilled tool result that contain a literal substring, with their line numbers.

            Matching lines are prefixed `N:` and context lines `N-`; separate groups are divided by `--`.

            Args:
                ctx: The run context (supplied by the agent).
                handle: The handle from the overflowed tool return.
                pattern: Literal substring to find (not a regex).
                context: Lines to show before and after each match (>= 0; clamped to a built-in cap).
                max_matches: Maximum number of matches to return (>= 1; clamped to a built-in cap).
                offset: Number of matches to skip, to continue a previous search. Must be >= 0.
            """
            store = self._store_for(ctx)
            return await _grep(store, handle, pattern, context, max_matches, offset, _reader_name(ctx))

        prepare = step_aside_for_read_file if self._spills_to_run_workspace else None
        tools = [
            Tool(read_tool_result, tool_kind=READ_TOOL_RESULT_KIND, prepare=prepare),
            Tool(grep_tool_result, tool_kind=GREP_TOOL_RESULT_KIND, prepare=prepare),
        ]
        return FunctionToolset(tools, id=self.id or 'tool_output_limits')

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
        if tool_def.tool_kind in _EXEMPT_KINDS or await _reads_a_spill(ctx, tool_def, args):
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
        new_value: object | None
        items: list[object] = return_value if is_object_list(return_value) else [return_value]
        if any(is_multi_modal_content(item) for item in items):
            new_value, value_handle, value_bytes = await self._reduce_data_beside_files(ctx, call, bands, items)
        else:
            value_unit = self._make_unit(return_value, tool_name=call.tool_name, suffix='')
            new_value, value_handle = await self._reduce(ctx, call, bands, value_unit)
            value_bytes = len(value_unit.data)
        content_text, content_handle = await self._reduce_content(ctx, call, bands, content)

        if new_value is None and content_text is None:
            return original

        return self._assemble(
            wrapped=wrapped,
            return_value=return_value,
            content=content,
            metadata=metadata,
            new_value=new_value,
            value_handle=value_handle,
            value_bytes=value_bytes,
            content_text=content_text,
            content_handle=content_handle,
        )

    async def _reduce_data_beside_files(
        self,
        ctx: RunContext[AgentDepsT],
        call: ToolCallPart,
        bands: Sequence[Band],
        items: list[object],
    ) -> tuple[list[object] | None, str | None, int]:
        """Reduce the data beside the files of a return value, which reach the model as files.

        A lone data item is measured as itself and several as one JSON list. Core sends that data as text, so
        a lone raw `bytes` item is measured as text too, not as a binary unit that `Truncate` would pass over.
        When reduced, the result goes ahead of the files. Returns `(new_items, handle, data_bytes)`, with
        `new_items` None when nothing was reduced.
        """
        files = [item for item in items if is_multi_modal_content(item)]
        data = [item for item in items if not is_multi_modal_content(item)]
        if not data:
            return None, None, 0
        value = data[0] if len(data) == 1 else data
        unit = self._make_unit(to_text(value) if is_binary(value) else value, tool_name=call.tool_name, suffix='')
        reduced, handle = await self._reduce(ctx, call, bands, unit)
        if reduced is None:
            return None, None, 0
        return [reduced, *files], handle, len(unit.data)

    def _assemble(
        self,
        *,
        wrapped: bool,
        return_value: ToolReturnContent,
        content: str | Sequence[UserContent] | None,
        metadata: object,
        new_value: object | None,
        value_handle: str | None,
        value_bytes: int,
        content_text: str | None,
        content_handle: str | None,
    ) -> object:
        """Rebuild the tool result from the reduced parts, preserving the envelope."""
        if wrapped:
            new_metadata = metadata
            if value_handle is not None or content_handle is not None:
                new_metadata = _with_handles(metadata, value_handle, value_bytes, content_handle)
            wrapped_out: ToolReturn[object] = ToolReturn(
                return_value=new_value if new_value is not None else return_value,
                content=content_text if content_text is not None else content,
                metadata=new_metadata,
            )
            return wrapped_out

        # A plain (non-`ToolReturn`) result has no separate content part.
        if value_handle is not None:
            spilled_out: ToolReturn[object] = ToolReturn(
                return_value=new_value, metadata=_with_handles(None, value_handle, value_bytes)
            )
            return spilled_out
        return new_value

    def _make_unit(self, value: ToolReturnContent, *, tool_name: str, suffix: str) -> _Unit:
        """Pre-render a value into the text / bytes the reduction pipeline needs."""
        if is_binary(value):
            return _Unit(binary=True, text=None, data=to_bytes(value), value=value, suffix=suffix)
        if self.serializer is not None and not isinstance(value, str):
            try:
                text = _ensure_text(self.serializer(value))
            except Exception:
                # An after-hook exception would abort the run and lose the tool's output,
                # so a broken serializer degrades to the default rendering instead.
                warnings.warn(
                    f'ToolOutputLimits: tool {tool_name!r} serializer raised or returned non-text; '
                    f'falling back to compact JSON.',
                    stacklevel=2,
                )
                text = to_text(value)
        else:
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
            return await self._reduce(
                ctx, call, bands, self._make_unit(content, tool_name=call.tool_name, suffix='.content')
            )

        text = ''.join(part for part in content if isinstance(part, str))
        size = measure(text, over_tokens=self.over_tokens, tokenizer=self.tokenizer)
        action = _select_action(bands, size)
        if action is not None and not isinstance(action, Passthrough):
            warnings.warn(
                f'ToolOutputLimits: tool {call.tool_name!r} returned large non-text '
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
            return (
                truncate_text(unit.text, action.max_chars, action.strategy, keep_tail_lines=action.keep_tail_lines),
                None,
            )

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

    def _handle_key(self, ctx: RunContext[AgentDepsT], call: ToolCallPart, suffix: str = '') -> str:
        """Build a per-run, per-call, per-retry key so concurrent and retried calls never clash.

        `suffix` keeps a return's `return_value` and `content` spills on distinct handles. A later
        step reusing a name already spilled this run (see `_spilled_calls`) gets
        `run_id/<step>/tool_call_id.retry` instead of the plain `run_id/tool_call_id.retry`.
        """
        run_id = ctx.run_id or 'run'
        name = f'{call.tool_call_id or "call"}.{ctx.retry}{suffix}'
        first_step = self._spilled_calls.setdefault(name, ctx.run_step)
        if first_step == ctx.run_step:
            return f'{run_id}/{name}'
        return f'{run_id}/{ctx.run_step}/{name}'

    async def _spill(
        self,
        ctx: RunContext[AgentDepsT],
        call: ToolCallPart,
        action: Spill,
        unit: _Unit,
    ) -> tuple[str | None, str | None]:
        key = self._handle_key(ctx, call, unit.suffix)
        # Text crosses the durable boundary as is; only binary needs base64 to travel as JSON.
        if unit.binary:
            payload = base64.b64encode(unit.data).decode('ascii')
        else:
            assert unit.text is not None
            payload = unit.text
        handle = await self._write_spill(ctx, call.tool_name, key, payload, unit.binary, _fallback_label(action.then))
        if handle is None:
            return await self._fallback(ctx, call, action.then, unit)

        preview = _build_spill_preview(
            handle, unit, action.preview_chars, over_tokens=self.over_tokens, read_hint=self._read_hint(ctx, handle)
        )
        return preview, handle

    # A durable operation, so a replay returns the recorded outcome instead of writing again: a
    # second write could return a different handle, or succeed where the first failed.
    @durable_operation('spill')
    async def _write_spill(
        self, ctx: RunContext[AgentDepsT], tool_name: str, key: str, payload: str, binary: bool, fallback: str
    ) -> str | None:
        """Write a spill and return its handle, or `None` when the store could not take it.

        `payload` is the text, or base64 when `binary`: a durable engine may carry it to another worker as JSON.
        """
        data = base64.b64decode(payload) if binary else payload.encode('utf-8')
        try:
            return await self._store_for(ctx).write(key, data)
        except (UserError, WorkspaceReadOnlyError) as error:
            # A read-only workspace, or none at all (a deferred-loaded capability skips `before_run`):
            # say so rather than degrading quietly.
            warnings.warn(f'ToolOutputLimits: could not spill a {tool_name!r} result: {error}', stacklevel=3)
        except Exception as exc:
            _log_fallback('Spilling', tool_name, fallback, exc)
        return None

    def _read_hint(self, ctx: RunContext[AgentDepsT], handle: str) -> str:
        """How the model reads a spill back, naming the tools by the names they have in this run."""
        if self._spills_to_run_workspace and (reader := ctx.tool_name_for_kind(READ_FILE_KIND)) is not None:
            return f' Read it with {reader}(path={handle!r}).'
        reader = ctx.tool_name_for_kind(READ_TOOL_RESULT_KIND) or READ_TOOL_NAME
        return f' Read it with {reader}(handle={handle!r}, offset=0, limit=200, from_end=False, pattern=None).'

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
            if action.summarize is not None:
                outcome = action.summarize(call.tool_name, unit.text)
                summary = await outcome if isinstance(outcome, Awaitable) else outcome
            else:
                self._summary_model(ctx, action)
                summary = await self._summarize(ctx, call.tool_name, unit.text, self._summarize_path(action))
        except UserError:
            # A misconfiguration -- e.g. a realtime run with no summarizer `model=` (#585) -- is
            # the user's to fix, not something to mask by silently truncating instead.
            raise
        except Exception as exc:
            _log_fallback('Summarizing', call.tool_name, _fallback_label(action.then), exc)
            return await self._fallback(ctx, call, action.then, unit)
        return f'[summary of {len(unit.text):,} chars of output{RERUN_NOTE}]\n{summary}', None

    @durable_operation('summarize')
    async def _summarize(
        self,
        ctx: RunContext[AgentDepsT],
        tool_name: str,
        text: str,
        action_path: str,
    ) -> str:
        """Generate a summary with configuration recovered from this capability."""
        from pydantic_ai import Agent

        action = self._resolve_summarize(action_path)
        model = self._summary_model(ctx, action)
        prompt = self.summary_prompt.format(tool_name=tool_name, output=text)
        agent: Agent[None, str] = Agent(
            model, name='tool_output_limits', instructions='You summarize oversized tool output.'
        )
        with second_model_prompt():
            run = await agent.run(
                prompt,
                conversation_id=ctx.conversation_id,
                usage=ctx.usage,
                usage_limits=reserved_usage_limits(ctx.usage_limits),
            )
        return run.output.strip()

    @staticmethod
    def _summary_model(ctx: RunContext[AgentDepsT], action: Summarize) -> Model[Any] | str:
        """Resolve and validate the model on both the caller and durable worker paths."""
        model = action.model if action.model is not None else ctx.model
        if isinstance(model, str):
            return model
        if not is_request_response_model(model):
            raise UserError(
                'Summarizing oversized tool output needs a request-response model, but the run '
                f'uses {type(model).__name__}, which is not one. Set a `model=` on the '
                '`Summarize` action to use for summarization.'
            )
        return model

    def _summarize_path(self, target: Summarize) -> str:
        for prefix, bands in [
            ('bands', self._bands),
            *((f'per_tool:{name}', value) for name, value in self._per_tool.items()),
        ]:
            for index, band in enumerate(bands):
                action = band.action
                depth = 0
                while action is not None:
                    if action is target:
                        return f'{prefix}:{index}:{depth}'
                    action = _next_action(action)
                    depth += 1
        raise RuntimeError(  # pragma: no cover - target comes from the configuration walked above
            'Summarize action is not part of this capability configuration.'
        )

    def _resolve_summarize(self, path: str) -> Summarize:
        prefix, index_text, depth_text = path.rsplit(':', 2)
        bands = self._bands if prefix == 'bands' else self._per_tool[prefix.removeprefix('per_tool:')]
        action: Action | None = bands[int(index_text)].action
        for _ in range(int(depth_text)):
            if action is None:  # pragma: no cover - durable inputs originate from `_summarize_path`
                break
            action = _next_action(action)
        if not isinstance(action, Summarize):  # pragma: no cover - durable inputs originate from `_summarize_path`
            raise RuntimeError(f'Durable summarize path no longer identifies a Summarize action: {path!r}.')
        return action


async def _reads_a_spill(ctx: RunContext[Any], tool_def: ToolDefinition, args: dict[str, Any]) -> bool:
    """Whether this is a `READ_FILE_KIND` read of a file under the run workspace's spill directory.

    Its result is the spill itself, or a page of it, so reducing it again would spill the spill.
    Only the path is compared, so a spill written by an earlier run, or another configuration, counts too.
    `read_file` bounds its own output; `FileSystem`'s searches bound only their match count, so their
    results are reduced as usual.
    """
    if tool_def.tool_kind != READ_FILE_KIND.id or not isinstance(path := args.get('path'), str):
        return False
    # Ordinary reads are told apart without asking the workspace for its working directory.
    workspace = ctx.workspace
    if f'{METADATA_DIR}/{SPILL_DIR}/' not in posixpath.normpath(path) or not workspace.attached:
        return False
    spill_dir = posixpath.join(await workspace.working_dir(), METADATA_DIR, SPILL_DIR)
    return (await workspace.resolve(path)).startswith(spill_dir + '/')


def _next_action(action: Action) -> Action | None:
    if isinstance(action, (Truncate, Spill, Summarize)):
        return action.then
    return None


def _ensure_text(rendered: object) -> str:
    """Reject non-`str` serializer output (e.g. `to_json` bytes without `.decode()`)."""
    if not isinstance(rendered, str):
        raise TypeError(f'serializer returned {type(rendered).__name__}')
    return rendered


def _select_action(bands: Sequence[Band], size: int) -> Action | None:
    """Return the first (largest-threshold) band action whose threshold `size` meets."""
    for band in bands:
        if size >= band.over:
            return band.action
    return None


def _fallback_label(then: Action | None) -> str:
    """Name the fallback action for a log line; `None` keeps the output unchanged."""
    return 'the unreduced output' if then is None else type(then).__name__


def _log_fallback(operation: str, tool_name: str, fallback: str, exc: Exception) -> None:
    logger.warning(
        '%s output of tool %r failed (%s); falling back to %s',
        operation,
        tool_name,
        type(exc).__name__,
        fallback,
        exc_info=exc,
    )


def _with_handles(
    existing: object,
    value_handle: str | None,
    value_bytes: int,
    content_handle: str | None = None,
) -> object:
    """Stash spill handle(s) in `ToolReturn.metadata` (app-only, costs no model tokens).

    A list or tuple stays a list, with the handles appended as one dict item: that is the
    shape the AG-UI and Vercel AI adapters read UI events and chunks from, and both skip
    items that are not events, so wrapping it in a dict would silently drop them. A single
    event or chunk becomes a one item list for the same reason.
    """
    handles: dict[str, object] = {}
    if value_handle is not None:
        handles['overflow_handle'] = value_handle
        handles['overflow_bytes'] = value_bytes
    if content_handle is not None:
        handles['overflow_content_handle'] = content_handle

    if is_mapping(existing):
        return {**{str(key): existing[key] for key in existing}, **handles}
    if is_object_list_or_tuple(existing):
        return [*existing, handles]
    if _is_ui_event(existing):
        return [existing, handles]
    if existing is not None:
        return {'original_metadata': existing, **handles}
    return handles


def _is_ui_event(value: object) -> bool:
    """Whether `value` is a single AG-UI event or Vercel AI data chunk the UI adapters pass through."""
    try:
        from ag_ui.core import BaseEvent
    except ImportError:
        pass
    else:
        if isinstance(value, BaseEvent):
            return True
    try:
        from pydantic_ai.ui.vercel_ai._utils import DATA_CHUNK_TYPES
    except ImportError:  # pragma: no cover
        return False
    return isinstance(value, DATA_CHUNK_TYPES)


def _build_spill_preview(handle: str, unit: _Unit, preview_chars: int, *, over_tokens: bool, read_hint: str) -> str:
    """Compose the model-visible spill stand-in: marker, sketch, and a head/tail preview."""
    if unit.binary:
        size_description = f'{len(unit.data):,} bytes (binary)'
        body = f'<{len(unit.data):,} bytes of binary data>'
        sketch = ''
    else:
        text = unit.text or ''
        size_unit = 'tokens' if over_tokens else 'chars'
        amount = measure(text, over_tokens=over_tokens, tokenizer=None) if over_tokens else len(text)
        size_description = f'{amount:,} {size_unit}'
        body = _head_tail_preview(text, preview_chars)
        sketch = json_sketch(unit.value)

    header = f'[Tool output too large ({size_description}); stored to handle {handle!r}.{read_hint}]'
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


_MAX_GREP_MATCHES = 200
"""Hard cap on matches returned by one `grep_tool_result` call."""

_MAX_GREP_CONTEXT = 10
"""Hard cap on context lines shown on each side of a `grep_tool_result` match."""


async def _guarded(handle: str, query: Awaitable[str]) -> str:
    """Run a spill query, or return the message that guides the model to a valid handle."""
    try:
        return await query
    except WorkspaceError as error:
        raise_tool_failure(error)
    except (OSError, UserError):
        # Return, not raise: a wrong handle (e.g. the model passing a tool-call id) or a
        # result that is no longer stored must not consume a tool retry and escalate to a
        # fatal `UnexpectedModelBehavior`. Guide the model to a valid handle instead. The
        # exception is intentionally not echoed -- a store's error can carry the resolved
        # filesystem path or other backend detail the model has no need for. `UserError` is a
        # run with no workspace attached, where no workspace spill can exist.
        return (
            f'[No stored tool result for handle {handle!r}. Use the exact handle string from a '
            '"[Tool output too large ... stored to handle ...]" marker; if the result is no longer '
            'available, re-run the original tool.]'
        )


@dataclass(frozen=True)
class _Entry:
    """A line to render for a spill query."""

    prefix: str
    text: str
    index: int
    """The line's index in the spill."""
    match: bool


_GroupedEntry = _Entry | None
"""An entry, or `None` for a separator between grep groups."""


def _byte_len(text: str) -> int:
    return len(text.encode('utf-8', 'surrogateescape'))


def _display(text: str) -> str:
    """Render a line of `_payload_texts` text for the model, with undecodable bytes as U+FFFD."""
    return text.encode('utf-8', 'surrogateescape').decode('utf-8', errors='replace')


@dataclass
class _Page:
    """What `_fill` rendered within the output cap, and what it left out."""

    lines: list[str] = field(default_factory=list[str])
    shown: int = 0
    """How many matches the page shows."""
    cut: tuple[int, int] | None = None
    """A match cut by the cap: its line index and how many of its bytes were shown."""
    omitted: list[tuple[int, int]] = field(default_factory=list[tuple[int, int]])
    """Ranges of context line indices left out to stay within the cap."""
    size: int = 0
    first: int | None = None
    """The index of the first line offered, to report leading context that was dropped."""
    kept: int | None = None
    """The index of the first line kept."""

    def add(self, text: str, match: bool) -> None:
        self.lines.append(text)
        self.size += len(text) + 1
        self.shown += match


@dataclass(frozen=True)
class _Stop:
    """Why `_place` filled the page: whether context from `start` (or the next entry) is left out."""

    omit: bool
    start: int | None = None


async def _fill(entries: AsyncIterator[_GroupedEntry]) -> _Page:
    """Render `entries` up to the output cap.

    Stops before a line that would cross the cap, so the match count stays true to the body. Until
    the first match is in, leading context is dropped instead, so every page makes progress. A match
    longer than the cap on its own is cut. Context left out either way is recorded in `omitted`.
    """
    page = _Page()
    separate = False
    async for entry in entries:
        if entry is None:
            separate = True
        elif (stop := _place(page, entry, separate)) is None:
            separate = False
        else:
            if stop.omit:
                await _omit(page, entries, stop.start)
            break
    if page.first is not None and page.kept is not None and page.kept > page.first:
        page.omitted.insert(0, (page.first, page.kept - 1))
    return page


def _place(page: _Page, entry: _Entry, separate: bool) -> _Stop | None:
    """Add one line to `page`, returning why it stopped once it is full."""
    if page.first is None:
        page.first = page.kept = entry.index
    text = ('--\n' if separate else '') + entry.prefix + _display(entry.text)
    if page.size + len(text) <= MAX_READ_CHARS:
        page.add(text, entry.match)
        return None
    if page.shown:
        # A line opening the next group belongs to a match the next page shows.
        return _Stop(omit=not (entry.match or separate), start=entry.index)
    page.lines.clear()
    page.size = 0
    page.kept = entry.index if entry.match else entry.index + 1
    if not entry.match:
        return None
    if len(text) <= MAX_READ_CHARS:
        page.add(text, entry.match)
        return None
    kept = entry.text[: MAX_READ_CHARS - len(entry.prefix)]
    page.add(entry.prefix + _display(kept), entry.match)
    page.cut = (entry.index, _byte_len(kept))
    return _Stop(omit=True)


async def _omit(page: _Page, entries: AsyncIterator[_GroupedEntry], start: int | None) -> None:
    """Record the context lines from `start` (or the next entry) up to the next match or group separator."""
    end = start
    async for entry in entries:
        if entry is None or entry.match:
            break
        start = entry.index if start is None else start
        end = entry.index
    if start is not None and end is not None:
        page.omitted.append((start, end))


@dataclass(frozen=True)
class _Reader:
    """The `read_tool_result` call a capped page points the model to, named as the tool is called in the run."""

    handle: str
    name: str

    def call(self, args: str) -> str:
        return f'{self.name}(handle={self.handle!r}, {args})'

    def omitted(self, start: int, end: int) -> str:
        return f'context lines {start + 1}-{end + 1} omitted ({self.call(f"offset={start}, limit={end - start + 1}")})'

    def cut(self, offset: int) -> str:
        args = f"unit='bytes', offset={offset}, limit={MAX_READ_CHARS}"
        return f'the cut line continues at {self.call(args)}'


def _reader_name(ctx: RunContext[Any]) -> str:
    """`read_tool_result` as it is called in this run step, for the hints that point to it."""
    return ctx.tool_name_for_kind(READ_TOOL_RESULT_KIND) or READ_TOOL_NAME


async def _capped(store: OverflowStore, page: _Page, more: str, reader: _Reader) -> str:
    """The header note for output stopped at the cap, saying what was left out and where it continues."""
    notes = [reader.omitted(start, end) for start, end in page.omitted]
    if page.cut is not None:
        index, shown_bytes = page.cut
        notes.append(reader.cut(await _line_start(store, reader.handle, index) + shown_bytes))
    if more:
        notes.append(more)
    return f', output capped at {MAX_READ_CHARS:,} chars; {"; ".join(notes)}' if notes else ''


async def _read_slice(
    store: OverflowStore,
    handle: str,
    offset: int,
    limit: int,
    from_end: bool,
    pattern: str | None,
    unit: ReadUnit,
    reader_name: str,
) -> str:
    """Filter and slice a spilled payload for `read_tool_result`, bounded in both axes.

    `pattern` is a literal substring (not a regex), so a model-supplied value cannot hang
    the host with catastrophic backtracking. `limit` is clamped and the joined output is
    capped, so one call can never return an unbounded amount of text. Every matching line is
    counted, but only the lines the window needs are kept.
    """
    if offset < 0:
        raise ModelRetry('`offset` must be >= 0.')
    if limit < 1:
        raise ModelRetry('`limit` must be >= 1.')
    if unit == 'bytes' and pattern is not None:
        raise ModelRetry("`pattern` filters lines, so it cannot be combined with `unit='bytes'`.")
    if unit == 'bytes':
        return await _guarded(handle, _byte_slice(store, handle, offset, min(limit, MAX_READ_CHARS), from_end))
    limit = min(limit, MAX_READ_LINES)
    reader = _Reader(handle, reader_name)
    return await _guarded(handle, _line_slice(store, handle, offset, limit, from_end, pattern, reader))


async def _line_slice(
    store: OverflowStore, handle: str, offset: int, limit: int, from_end: bool, pattern: str | None, reader: _Reader
) -> str:
    total, window = await _matching_window(store, handle, pattern, offset, limit, from_end=from_end)
    first = max(0, max(0, total - offset) - limit) if from_end else offset
    page = await _fill(_aiter(_Entry('', line, index, True) for index, line in window))
    filtered = '' if pattern is None else f', pattern={pattern!r}'
    next_lines = reader.call(f'offset={first + page.shown}{filtered}')
    more = f'the next lines are at {next_lines}' if page.shown < len(window) else ''
    capped = await _capped(store, page, more, reader)
    return '\n'.join([f'[handle {handle!r}: {total:,} matching line(s); showing {page.shown}{capped}]', *page.lines])


async def _aiter(entries: Iterable[_Entry]) -> AsyncIterator[_GroupedEntry]:
    for entry in entries:
        yield entry


async def _byte_slice(store: OverflowStore, handle: str, offset: int, limit: int, from_end: bool) -> str:
    """Slice raw stored bytes, decoding a multi-byte UTF-8 sequence cut at either edge as U+FFFD.

    A forward read keeps only the slice; a read from the end keeps the last `offset + limit` bytes.
    """
    total = 0
    kept = bytearray()
    async with aclosing(iter_payload(store, handle)) as chunks:
        async for chunk in chunks:
            if from_end:
                kept += chunk
                del kept[: max(0, len(kept) - (offset + limit))]
            else:
                kept += chunk[max(0, offset - total) : max(0, offset + limit - total)]
            total += len(chunk)
    if from_end:
        end = max(0, total - offset)
        start = max(0, end - limit)
        kept = kept[len(kept) - (total - start) : len(kept) - (total - end)]
    else:
        start = min(offset, total)
        end = start + len(kept)
    header = f'[handle {handle!r}: bytes {start}-{end} of {total} (end exclusive)]'
    return '\n'.join([header, kept.decode('utf-8', errors='replace')]) if kept else header


@dataclass
class _Grep:
    """Turns a payload's lines, in order, into the entries of one page of matches.

    The matches whose ordinal is in `page` are shown with `context` lines around each. Windows that
    overlap or touch merge into one group, with a separator entry between groups. Only matches in
    `page` are marked as matches. Every match is counted in `matches`.
    """

    pattern: str
    context: int
    page: range
    matches: int = 0
    end: int = 0
    """The index after the last line shown."""
    after: int = 0
    """How many trailing context lines are still to show."""
    before: deque[tuple[int, str]] = field(default_factory=deque[tuple[int, str]])

    def entries(self, index: int, line: str) -> list[_GroupedEntry]:
        match = self.pattern in line
        entries = self._match(index, line) if match and self.matches in self.page else self._context(index, line)
        self.matches += match
        self.before.append((index, line))
        if len(self.before) > self.context:
            self.before.popleft()
        return entries

    def _match(self, index: int, line: str) -> list[_GroupedEntry]:
        leading = [(n, text) for n, text in self.before if n >= self.end]
        separate = self.end and (leading[0][0] if leading else index) > self.end
        entries: list[_GroupedEntry] = [None] if separate else []
        entries += [_Entry(f'{n + 1}-', text, n, False) for n, text in leading]
        entries.append(_Entry(f'{index + 1}:', line, index, True))
        self.end, self.after = index + 1, self.context
        return entries

    def _context(self, index: int, line: str) -> list[_GroupedEntry]:
        if not self.after:
            return []
        self.end, self.after = index + 1, self.after - 1
        return [_Entry(f'{index + 1}-', line, index, False)]


async def _grep_entries(store: OverflowStore, handle: str, grep: _Grep) -> AsyncGenerator[_GroupedEntry]:
    index = 0
    async with aclosing(_payload_lines(store, handle)) as batches:
        async for batch in batches:
            for line in batch:
                for entry in grep.entries(index, line):
                    yield entry
                index += 1


async def _grep(
    store: OverflowStore,
    handle: str,
    pattern: str,
    context: int,
    max_matches: int,
    offset: int,
    reader_name: str,
) -> str:
    """Find literal `pattern` in a spilled payload for `grep_tool_result`, with 1-based line numbers.

    Context windows that overlap or touch merge into one group. Matches, context, and the joined
    output are all capped, so one call can never return an unbounded amount of text. What the cap
    leaves out is described as a call to the `read_tool_result` named `reader_name`.
    """
    if not pattern:
        raise ModelRetry('`pattern` must not be empty.')
    if context < 0:
        raise ModelRetry('`context` must be >= 0.')
    if max_matches < 1:
        raise ModelRetry('`max_matches` must be >= 1.')
    if offset < 0:
        raise ModelRetry('`offset` must be >= 0.')
    page = range(offset, offset + min(max_matches, _MAX_GREP_MATCHES))
    reader = _Reader(handle, reader_name)
    return await _guarded(handle, _grep_page(store, handle, pattern, min(context, _MAX_GREP_CONTEXT), page, reader))


async def _grep_page(
    store: OverflowStore, handle: str, pattern: str, context: int, page: range, reader: _Reader
) -> str:
    grep = _Grep(pattern, context, page)
    async with aclosing(_grep_entries(store, handle, grep)) as entries:
        filled = await _fill(entries)
        async for _ in entries:
            pass
    more = grep.matches - page.start - filled.shown
    remaining = f', {more:,} more; continue with offset={page.start + filled.shown}' if more > 0 else ''
    capped = await _capped(store, filled, '', reader)
    header = f'[handle {handle!r}: {grep.matches:,} matching line(s); showing {filled.shown}{remaining}{capped}]'
    return '\n'.join([header, *filled.lines])


async def _matching_window(
    store: OverflowStore, handle: str, pattern: str | None, offset: int, limit: int, *, from_end: bool
) -> tuple[int, list[tuple[int, str]]]:
    """Count the lines containing `pattern` and return the count with the window `offset` and `limit` select.

    The window holds each line with its index in the spill. A forward read keeps only the window. A read
    from the end keeps the last `offset + limit` lines, since where its window starts is known only once
    the payload ends.
    """
    total = index = 0
    kept: deque[tuple[int, str]] = deque(maxlen=min(offset + limit, sys.maxsize) if from_end else limit)
    async with aclosing(_payload_lines(store, handle)) as batches:
        async for batch in batches:
            matching = [(index + n, line) for n, line in enumerate(batch) if pattern is None or pattern in line]
            index += len(batch)
            kept.extend(matching if from_end else matching[max(0, offset - total) : max(0, offset + limit - total)])
            total += len(matching)
    window = list(kept)
    return total, window[: max(0, len(window) - offset)] if from_end else window


_LINE_BREAK = re.compile('\r\n|[\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029]')
"""The line boundaries `str.splitlines` uses."""


async def _line_start(store: OverflowStore, handle: str, index: int) -> int:
    """The byte offset line `index` of a stored payload starts at, counting lines as `str.splitlines` does."""
    if index == 0:
        return 0
    position = breaks = 0
    async with aclosing(_payload_texts(store, handle)) as texts:
        async for text in texts:
            for line_break in _LINE_BREAK.finditer(text):
                breaks += 1
                if breaks == index:
                    return position + _byte_len(text[: line_break.end()])
            position += _byte_len(text)
    # The caller only asks for a line it just read from this payload.
    raise AssertionError(f'line {index} not found in {handle!r}')  # pragma: no cover


async def _payload_texts(store: OverflowStore, handle: str) -> AsyncGenerator[str]:
    """Yield a stored payload's text one decoded chunk at a time.

    `surrogateescape` keeps every stored byte recoverable, so byte offsets stay exact for any payload;
    `_display` renders undecodable bytes as U+FFFD. A trailing CR waits for the next chunk, which may
    start with the LF of a CRLF.
    """
    decoder = codecs.getincrementaldecoder('utf-8')(errors='surrogateescape')
    cr = ''
    async with aclosing(iter_payload(store, handle)) as chunks:
        async for chunk in chunks:
            text = cr + decoder.decode(chunk)
            cr = '\r' if text.endswith('\r') else ''
            yield text.removesuffix(cr)
    yield cr + decoder.decode(b'', final=True)


async def _payload_lines(store: OverflowStore, handle: str) -> AsyncGenerator[list[str]]:
    """Yield a stored payload's lines as `str.splitlines` splits its text, one decoded chunk's lines at a time."""
    pieces: list[str] = []
    async with aclosing(_payload_texts(store, handle)) as texts:
        async for text in texts:
            yield _complete_lines(text, pieces)
    if pieces:
        yield [''.join(pieces)]


def _complete_lines(text: str, pieces: list[str]) -> list[str]:
    """Return the lines `text` completes, keeping its unterminated end in `pieces`."""
    lines = text.splitlines()
    # The last line is unterminated when `text` ends with it: `splitlines` yields an empty line only before a break.
    end = lines.pop() if lines and lines[-1] and text.endswith(lines[-1]) else None
    if pieces and lines:
        lines[0] = ''.join(pieces) + lines[0]
        pieces.clear()
    if end is not None:
        pieces.append(end)
    return lines
