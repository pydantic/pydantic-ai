"""Tool search toolset and strategy types.

`ToolSearchToolset` wraps another toolset to support discovery of tools marked with
`defer_loading=True`. It settles the one question that doesn't depend on which model
serves the request — is this tool *searchable*? — and leaves the rest to
[`Model.prepare_request`][pydantic_ai.models.Model.prepare_request], which can't be
decided here because the model isn't known yet (think `FallbackModel`).

The two questions a deferred tool raises are kept apart:

* **Hidden until revealed** — true of every deferred tool. Carried by the authored
  `defer_loading` value, which stays set for the whole run, with current visibility
  travelling separately on `ModelRequestParameters.revealed_tool_names`.
* **Member of the searchable corpus** — carried by `with_native='tool_search'`, and set
  only on deferred tools no on-demand capability gates. A capability-gated tool becomes
  available by loading its capability, never by querying for it.

`Model.prepare_request` then decides, per model, how each hidden tool reaches the wire:

* Where the provider can declare a tool but withhold its schema, hidden tools stay in
  `tools` behind the provider's deferral flag (Anthropic `defer_loading`, OpenAI
  Responses `defer_loading`) and a reveal unlocks them in place, leaving `tools`
  byte-identical across the reveal.
* Otherwise hidden tools are kept off the wire entirely and arrive when they're revealed
  — as a full declaration on providers with a mid-conversation reveal item, or simply as
  a new `tools` entry where there's no such item.

`search_tools`, the local discovery function, carries `unless_native='tool_search'`
and is dropped by the adapter when the builtin is supported. When the capability commits
to a named-native strategy with no local equivalent (`'bm25'`/`'regex'`) the toolset is
constructed with `enable_fallback=False` and `search_tools` is not emitted at all — that
way `_resolve_request_tools` raises on providers that can't honor the builtin, and
the wire stays clean (just the native tool) on those that can.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from functools import cache
from typing import Annotated, Any

from pydantic import Field, TypeAdapter, ValidationError
from typing_extensions import TypedDict, assert_never

from .._run_context import AgentDepsT, RunContext
from .._tool_search import _NO_MATCHES_MESSAGE  # pyright: ignore[reportPrivateUsage]
from ..exceptions import ModelRetry, UserError
from ..messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    NativeToolSearchReturnPart,
    ToolAvailabilityDeltaPart,
    ToolReturnPart,
    ToolSearchReturnPart,
    post_compaction_window,
)
from ..native_tools._tool_search import (
    TOOL_SEARCH_FUNCTION_TOOL_NAME,
    ToolSearchFunc,
    ToolSearchMatch,
    ToolSearchReturnContent,
    ToolSearchTool,
)
from ..tools import Tool, ToolDefinition
from ._capability_owned import is_gated_by_deferred_capability
from .abstract import ToolsetTool
from .wrapper import WrapperToolset

_SEARCH_TOOLS_NAME = TOOL_SEARCH_FUNCTION_TOOL_NAME
_TOOL_SEARCH_BUILTIN_ID = ToolSearchTool.kind

_LEGACY_DISCOVERED_TOOLS_METADATA_KEY = 'discovered_tools'


class _LegacyDiscoveryMetadata(TypedDict):
    """Pre-typed-content metadata sideband shape.

    Earlier versions stashed discovered tool names on
    `ToolReturnPart.metadata['discovered_tools']` instead of on the typed `content`.
    Validating against this shape via Pydantic keeps the legacy reader honest about
    what it accepts; new writes always go through the typed content.
    """

    discovered_tools: list[str]


_LEGACY_METADATA_TA = TypeAdapter(_LegacyDiscoveryMetadata)


_MAX_SEARCH_RESULTS = 10
_SEARCH_TOKEN_RE = re.compile(r'[a-z0-9]+')


def _tokenize(text: str) -> set[str]:
    """Lowercase + extract alphanumeric tokens for keyword matching.

    Used for both the query and the indexed terms (tool name + description) so
    matching is case-insensitive and word-bounded — `me` matches `get_me` but not
    the substring inside `comment`.
    """
    return set(_SEARCH_TOKEN_RE.findall(text.lower()))


def keywords_search_fn(_ctx: RunContext[Any], queries: Sequence[str], tools: Sequence[ToolDefinition]) -> list[str]:
    """Built-in keyword-overlap search algorithm exposed as a [`ToolSearchFunc`][pydantic_ai.capabilities.ToolSearchFunc].

    Score each tool by how many query keywords appear in its name or description, then
    return matching names ordered by descending score. Used both as the default
    algorithm when `ToolSearch` was constructed without an explicit strategy AND as
    the explicit `strategy='keywords'` choice — the difference is that the explicit
    choice routes through the same dispatch path as a user-supplied callable, which
    enables client-executed-native wire on supporting providers (cache benefit).
    """
    terms = _tokenize(' '.join(queries))
    if not terms:
        return []
    scored: list[tuple[int, str]] = []
    for tool_def in tools:
        tool_terms = _tokenize(f'{tool_def.name} {tool_def.description or ""}')
        score = len(terms & tool_terms)
        if score > 0:
            scored.append((score, tool_def.name))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [name for _, name in scored]


_DEFAULT_TOOL_DESCRIPTION = (
    'Search first for a standalone deferred tool when current tools and catalog descriptions do not name the requested operation.'
    ' A capability id used as an ordinary domain word does not request that capability.'
    ' This cannot find capability-owned tools; load a listed capability by id instead.'
    ' If no tools are found, do not retry.'
)


_DEFAULT_PARAMETER_DESCRIPTION = (
    'List of search queries to match against tool names and descriptions.'
    ' Use specific words likely to appear in tool names or descriptions to narrow down relevant tools.'
    ' Each query is independently tokenized; matches across queries are unioned.'
)


# Schema source only, never invoked.
def _search_tools_signature(
    queries: Annotated[list[str], Field(description=_DEFAULT_PARAMETER_DESCRIPTION)],
) -> ToolSearchReturnContent:  # pragma: no cover
    """Source-of-truth signature for the `search_tools` function tool.

    Used by [`Tool`][pydantic_ai.tools.Tool] to derive the JSON schema and validator
    that go on the `ToolDefinition` we hand to the model. Wrapping the function in a
    `Tool` (rather than hand-rolling a `TypedDict` + `TypeAdapter`) keeps schema
    generation aligned with how every other tool in the framework is defined.

    Parameter is `queries: list[str]` to match the cross-provider
    [`ToolSearchArgs`][pydantic_ai.messages.ToolSearchArgs] shape — so the same typed
    [`ToolSearchCallPart`][pydantic_ai.messages.ToolSearchCallPart] represents both
    model-emitted local calls AND cross-provider-synthesized history (Anthropic native
    `bm25`/`regex` and OpenAI Responses `tool_search_call`, both normalized to `queries`).
    """
    raise NotImplementedError


_SEARCH_TOOL_FN_SCHEMA = Tool(_search_tools_signature).function_schema
_SEARCH_TOOL_SCHEMA: dict[str, Any] = _SEARCH_TOOL_FN_SCHEMA.json_schema
_SEARCH_TOOL_VALIDATOR = _SEARCH_TOOL_FN_SCHEMA.validator


@cache
def _build_search_args_schema(parameter_description: str) -> tuple[dict[str, Any], Any]:
    """Reuse the default schema/validator or splice in a custom `queries` description.

    Cached per-description: with the default description the call is a constant-time
    lookup that returns the module-level schema and validator (`Tool(fn).function_schema`
    is the source of truth for both). A custom description gets a per-description rebuild
    whose result is memoized — the framework only pays schema-construction cost on the
    first run with a given override.

    The custom path splices `parameter_description` into the existing JSON schema rather
    than rebuilding from a closure-bound signature: `from __future__ import annotations`
    stringifies the type expression, so a closure-captured `parameter_description` would
    be unresolvable when `Tool` re-evaluates the string at schema-derivation time. The
    validator is unaffected by description, so we can safely reuse the default one.
    """
    if parameter_description == _DEFAULT_PARAMETER_DESCRIPTION:
        return _SEARCH_TOOL_SCHEMA, _SEARCH_TOOL_VALIDATOR

    schema = deepcopy(_SEARCH_TOOL_SCHEMA)
    schema['properties']['queries']['description'] = parameter_description
    return schema, _SEARCH_TOOL_VALIDATOR


def parse_discovered_tools(messages: Sequence[ModelMessage]) -> set[str]:
    """Scan visible message history for previously-discovered tool names.

    Every [`CompactionPart`][pydantic_ai.messages.CompactionPart] resets the derived
    state at its exact position in a response. This is deliberately provider-agnostic:
    over-counting can prevent rediscovery or claim a schema is visible when it is not,
    while under-counting once only permitted a redundant, idempotent search. Now that
    availability gates execution, an under-count also *refuses* the call — see
    [`post_compaction_window`][pydantic_ai.messages.post_compaction_window] for when that
    is wrong and what is tracked to fix it.

    Trusts that any `ToolSearchReturnPart` / `NativeToolSearchReturnPart` in the
    history has a validated `ToolSearchReturnContent`:
    Pydantic's discriminator dispatch promotes from base parts on deserialization,
    and direct construction goes through the typed-class `__init__` (which Pydantic
    validates). No defensive isinstance walks needed.

    Also reads the legacy `metadata['discovered_tools']` sideband (validated against
    a TypedDict) so histories serialized before the typed-content migration continue
    to surface previously-discovered tools.

    Only the [`post_compaction_window`][pydantic_ai.messages.post_compaction_window] is scanned —
    the one definition of the boundary — so locating it costs a cheap reverse
    `isinstance` pass rather than parsing history the boundary would reset anyway.
    """
    return set(discovered_tool_names_in_order(messages))


def discovered_tool_names_in_order(messages: Sequence[ModelMessage]) -> tuple[str, ...]:
    """Return discovered names in first-appearance order for byte-stable provider tool segments.

    Scans only the [`post_compaction_window`][pydantic_ai.messages.post_compaction_window], so both the
    reveal set and the wire ordering derive from what the model can actually see.
    """
    return _discovered_tool_names_in_order(post_compaction_window(messages))


def _discovered_tool_names_in_order(messages: Sequence[ModelMessage]) -> tuple[str, ...]:
    """Parse discovery evidence from an already-selected message window."""
    discovered: dict[str, None] = {}
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, ToolAvailabilityDeltaPart):
                    discovered.update(dict.fromkeys(part.tools_added))
                elif isinstance(part, ToolSearchReturnPart):
                    discovered.update(dict.fromkeys(match['name'] for match in part.discovered_tools))
                elif isinstance(part, ToolReturnPart) and part.tool_name == _SEARCH_TOOLS_NAME:
                    # Legacy histories carry discoveries on `metadata['discovered_tools']`
                    # rather than typed content. Narrowing tool_name + metadata shape avoids
                    # surfacing a user-defined `search_tools` whose metadata has no legacy
                    # shape.
                    try:
                        validated = _LEGACY_METADATA_TA.validate_python(part.metadata)
                    except ValidationError:
                        continue
                    discovered.update(dict.fromkeys(validated['discovered_tools']))
        elif isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, NativeToolSearchReturnPart):
                    discovered.update(dict.fromkeys(match['name'] for match in part.discovered_tools))
        else:
            assert_never(msg)
    return tuple(discovered)


@dataclass(kw_only=True)
class _SearchTool(ToolsetTool[AgentDepsT]):
    """The local `search_tools` function, carrying the corpus it should search over.

    The real `ToolDefinition`s flow through to user-supplied search functions so
    callables can read whatever metadata they need (parameters schema, kind, etc.) — not
    just the name/description pair we'd otherwise expose.
    """

    corpus: list[ToolDefinition]

    discovered_tool_names: set[str]
    """Snapshot of `RunContext.discovered_tool_names` taken when the search tool was built.

    Discovered, not revealed: it is raw history evidence, and no per-request wire state is
    consulted. Inside the corpus the two coincide anyway — capability-gated tools are excluded
    from it, so a corpus tool named by history is also available — but the search only needs the
    weaker fact, that the model has seen this tool before, to sort it behind fresh matches.
    """


@dataclass
class ToolSearchToolset(WrapperToolset[AgentDepsT]):
    """A toolset that enables tool discovery for large toolsets.

    Wraps another toolset and exposes a `search_tools` function that lets the model
    discover tools with `defer_loading=True`. Tools with `defer_loading=True` are
    not initially presented to the model — they become available after the model
    discovers them via search.

    When the model supports the framework-managed tool-search builtin, discovery is
    handled by the provider and the deferred tools are sent to the API with
    `defer_loading=True` on the wire.
    """

    search_fn: ToolSearchFunc[AgentDepsT] | None = None
    """Optional custom search function. If `None`, the default keyword-overlap algorithm is used.

    Receives the run context, the list of search queries, and the full searchable deferred-tool
    corpus, including tools already discovered during this run, and returns the matching tool names
    ordered by relevance. Both sync and async implementations are accepted.
    """

    max_results: int = _MAX_SEARCH_RESULTS
    """Maximum number of matches returned from the default algorithm."""

    tool_description: str | None = None
    """Custom description for the `search_tools` function shown to the model."""

    parameter_description: str | None = None
    """Custom description for the `queries` parameter shown to the model."""

    enable_fallback: bool = True
    """When False, the local `search_tools` function tool is not emitted — used when the
    capability commits to a named-native strategy that has no local equivalent (e.g.
    `'bm25'`, `'regex'`). With no fallback registered, `_resolve_request_tools` raises
    on providers that can't honor the builtin, instead of silently substituting the local
    keyword algorithm; and on providers that DO support it, only the native tool reaches
    the wire (no redundant `search_tools` slot that could confuse the model)."""

    max_retries: int | None = None
    """Maximum number of retries for the local `search_tools` function tool, *after* the
    first attempt.

    When `None`, the agent's tool retry budget applies (`Agent(retries={'tools': N})`),
    following the same `tool.max_retries` -> `toolset.max_retries` -> `ctx.max_retries`
    precedence as [`FunctionToolset`][pydantic_ai.toolsets.FunctionToolset]. The budget is
    consumed by malformed arguments (e.g. a bare string where a list is expected) and by
    blank queries; a search that simply finds no matches returns normally and never spends
    a retry.
    """

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        all_tools = await self.wrapped.get_tools(ctx)

        deferred: dict[str, ToolsetTool[AgentDepsT]] = {}
        visible: dict[str, ToolsetTool[AgentDepsT]] = {}
        for name, tool in all_tools.items():
            if tool.tool_def.defer_loading:
                deferred[name] = tool
            else:
                visible[name] = tool

        if not deferred:
            return all_tools

        if _SEARCH_TOOLS_NAME in all_tools:
            raise UserError(
                f"Tool name '{_SEARCH_TOOLS_NAME}' is reserved for tool search. Rename your tool to avoid conflicts."
            )

        result: dict[str, ToolsetTool[AgentDepsT]] = dict(visible)

        # Every deferred tool is hidden until something reveals it — that's `defer_loading`, which
        # stays set across the run as the author wrote it, with current visibility travelling
        # separately on `ModelRequestParameters.revealed_tool_names`. Only some of them are also
        # *searchable*: a tool an on-demand capability gates is reached by loading that capability,
        # never by querying for it, so marking it a corpus member would claim a searchability it
        # doesn't have — and each adapter would then have to subtract it again on its own terms.
        # `with_native` therefore means corpus membership and nothing else.
        searchable: dict[str, ToolsetTool[AgentDepsT]] = {}
        for name, tool in deferred.items():
            if is_gated_by_deferred_capability(ctx, tool.tool_def):
                result[name] = tool
            else:
                searchable[name] = tool
                result[name] = replace(tool, tool_def=replace(tool.tool_def, with_native=_TOOL_SEARCH_BUILTIN_ID))

        # Emit `search_tools` whenever the searchable corpus is non-empty and a local fallback is
        # enabled. It carries `unless_native='tool_search'` so the adapter drops it on
        # the wire when the builtin is supported (the native path handles discovery
        # server-side); keeping it in the toolset across discovery steps preserves prompt
        # caching, since dropping it once everything is discovered would invalidate the
        # request prefix on the very next turn.
        #
        # With nothing searchable — a run whose deferred tools are all capability-gated — there is
        # no search to offer. Emitting it anyway would spend a tool slot and cache-prefix bytes on
        # every turn for a function that can only ever answer "no matches".
        #
        # When `enable_fallback=False` (named-native strategies `'bm25'`/`'regex'`) we
        # skip emission entirely: there's no local algorithm to fall back to, and emitting
        # it would both register a phantom fallback that suppresses the
        # "unsupported builtin" raise AND leave a redundant function tool on the wire
        # alongside the native builtin on providers that DO support it.
        if self.enable_fallback and searchable:
            result[_SEARCH_TOOLS_NAME] = self._build_search_tool(ctx, searchable)

        return result

    def _build_search_tool(
        self,
        ctx: RunContext[AgentDepsT],
        searchable: dict[str, ToolsetTool[AgentDepsT]],
    ) -> _SearchTool[AgentDepsT]:
        parameter_description = self.parameter_description or _DEFAULT_PARAMETER_DESCRIPTION
        schema, args_validator = _build_search_args_schema(parameter_description)

        # Real `ToolDefinition`s for every searchable tool — what the user's
        # search function sees, and what the local keywords search indexes. Capability-gated
        # tools never get here: they aren't searchable, so they aren't in `searchable`.
        corpus = [tool.tool_def for tool in searchable.values()]

        # `unless_native` tells the adapter to drop this function tool when the native
        # builtin is supported. That's what we want for server-side strategies (the
        # provider handles search entirely). For a custom callable strategy, the native
        # path on both Anthropic (regular function tool with tool_reference result
        # formatting) and OpenAI (`execution='client'`) still needs the local function
        # tool to execute the search, so we leave `unless_native` unset in that case.
        #
        # The `enable_fallback=False` path (named-native `'bm25'`/`'regex'`) never reaches
        # here — `get_tools` skips emitting `search_tools` entirely in that case (see the
        # caller).
        unless_native = _TOOL_SEARCH_BUILTIN_ID if self.search_fn is None else None
        search_tool_def = ToolDefinition(
            name=_SEARCH_TOOLS_NAME,
            description=self.tool_description or _DEFAULT_TOOL_DESCRIPTION,
            parameters_json_schema=schema,
            tool_kind='tool-search',
            unless_native=unless_native,
        )

        return _SearchTool(
            toolset=self,
            tool_def=search_tool_def,
            max_retries=self.max_retries if self.max_retries is not None else ctx.max_retries,
            args_validator=args_validator,
            corpus=corpus,
            discovered_tool_names=set(ctx.discovered_tool_names),
        )

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        if name == _SEARCH_TOOLS_NAME and isinstance(tool, _SearchTool):
            return await self._search_tools(tool_args, ctx, tool)
        return await self.wrapped.call_tool(name, tool_args, ctx, tool)

    @staticmethod
    def _search_terms(name: str, description: str | None) -> set[str]:
        search_terms = set(_SEARCH_TOKEN_RE.findall(name.lower()))
        if description:
            search_terms.update(_SEARCH_TOKEN_RE.findall(description.lower()))
        return search_terms

    async def _search_tools(
        self, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], search_tool: _SearchTool[AgentDepsT]
    ) -> ToolSearchReturnContent:
        """Run the configured search strategy over all searchable deferred tools."""
        queries: list[str] = tool_args['queries']
        if not any(q.strip() for q in queries):
            raise ModelRetry('Please provide at least one non-empty search query.')

        fn = self.search_fn
        if fn is not None:
            return await self._run_search_fn(fn, queries, ctx, search_tool)
        return self._run_keywords_search(queries, search_tool)

    def _run_keywords_search(
        self, queries: Sequence[str], search_tool: _SearchTool[AgentDepsT]
    ) -> ToolSearchReturnContent:
        """Score each tool by how many query tokens appear in its name/description.

        Tokenizes on alphanumeric runs for both the queries and the indexed terms, so the
        top hit for "github profile" is `github_get_me` (two matches) without matching
        substrings inside longer words like `comment` for the query `me`. Tokens from all
        queries are unioned — the same token-overlap score applies across the set.
        """
        terms = self._search_terms(' '.join(queries), None)
        if not terms:
            raise ModelRetry('Please provide at least one non-empty search query.')

        scored_matches: list[tuple[bool, int, ToolSearchMatch]] = []
        for tool_def in search_tool.corpus:
            tool_terms = self._search_terms(tool_def.name, tool_def.description)
            score = len(terms & tool_terms)
            if score == 0:
                continue
            scored_matches.append(
                (tool_def.name not in search_tool.discovered_tool_names, score, {'name': tool_def.name})
            )

        if not scored_matches:
            return self._empty_return()

        # Undiscovered-first is the PRIMARY key, relevance the tiebreak: an already-discovered
        # tool must never displace an undiscovered match when `max_results` trims — it only
        # fills whatever slots are left over.
        scored_matches.sort(key=lambda item: (item[0], item[1]), reverse=True)
        matches = [match for _, _, match in scored_matches[: self.max_results]]
        return self._build_return(matches)

    async def _run_search_fn(
        self,
        fn: ToolSearchFunc[AgentDepsT],
        queries: Sequence[str],
        ctx: RunContext[AgentDepsT],
        search_tool: _SearchTool[AgentDepsT],
    ) -> ToolSearchReturnContent:
        """Invoke a user-provided strategy, validating that the returned names are known."""
        tool_defs_by_name = {tool_def.name: tool_def for tool_def in search_tool.corpus}

        result = fn(ctx, queries, search_tool.corpus)
        if inspect.isawaitable(result):
            result = await result

        matches: list[ToolSearchMatch] = []
        for name in list(result)[: self.max_results]:
            if (tool_def := tool_defs_by_name.get(name)) is not None:
                matches.append({'name': tool_def.name})

        if not matches:
            return self._empty_return()
        return self._build_return(matches)

    @staticmethod
    def _empty_return() -> ToolSearchReturnContent:
        """Shaped "no matches" return: empty discovered_tools list with a user-visible message.

        Sending only the typed
        [`ToolSearchReturnContent`][pydantic_ai.messages.ToolSearchReturnContent] is enough
        — the JSON-serialized return value carries the message to the model, so it doesn't
        retry searching with the same keywords; adapters that need the message on the wire
        (Anthropic custom-callable empty-results path) read it from there too.
        """
        return {
            'discovered_tools': [],
            'message': _NO_MATCHES_MESSAGE,
        }

    @staticmethod
    def _build_return(matches: list[ToolSearchMatch]) -> ToolSearchReturnContent:
        """Shaped matches return: typed [`ToolSearchReturnContent`][pydantic_ai.messages.ToolSearchReturnContent]."""
        return {'discovered_tools': matches}
