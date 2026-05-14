"""Tool search toolset and strategy types.

`ToolSearchToolset` wraps another toolset to support discovery of tools marked with
`defer_loading=True`. Rather than commit to native-vs-local at toolset time (which can't
know which model will actually serve the request — think `FallbackModel`), the toolset
emits one entry per deferred tool with both `with_native='tool_search'` and the
current local visibility on `defer_loading`, then lets
[`Model.prepare_request`][pydantic_ai.models.Model.prepare_request] filter based on the
specific model's support for the framework-managed tool-search builtin:

* On the native path the adapter keeps every corpus member (regardless of local
  discovery state) and applies its provider-specific wire format — e.g. setting
  `defer_loading=True` on the Anthropic / OpenAI Responses tool param so the provider
  drives discovery server-side.
* On the local path corpus members with `defer_loading=True` (still undiscovered) are
  dropped from the wire; discovered ones (`defer_loading=False`) stay so the model can
  call them by their real name.

`search_tools`, the local discovery function, carries `unless_native='tool_search'`
and is dropped by the adapter when the builtin is supported. When the capability commits
to a named-native strategy with no local equivalent (`'bm25'`/`'regex'`) the toolset is
constructed with `enable_fallback=False` and `search_tools` is not emitted at all — that
way `_resolve_native_tool_swap` raises on providers that can't honor the builtin, and
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
from typing_extensions import TypedDict

from .._run_context import AgentDepsT, RunContext
from .._tool_search import _NO_MATCHES_MESSAGE  # pyright: ignore[reportPrivateUsage]
from ..exceptions import ModelRetry, UserError
from ..messages import (
    ModelRequest,
    NativeToolSearchReturnPart,
    ToolReturnPart,
    ToolSearchReturnPart,
)
from ..native_tools._tool_search import (
    TOOL_SEARCH_FUNCTION_TOOL_NAME,
    ToolSearchFunc,
    ToolSearchMatch,
    ToolSearchReturnContent,
    ToolSearchTool,
)
from ..tools import Tool, ToolDefinition
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
    'There are additional tools not yet visible to you.'
    ' When you need a capability not provided by your current tools,'
    ' search here by providing one or more queries to discover and activate relevant tools.'
    ' Each query is tokenized into words; tool names and descriptions are scored by token overlap.'
    ' If no tools are found, they do not exist — do not retry.'
)


_DEFAULT_PARAMETER_DESCRIPTION = (
    'List of search queries to match against tool names and descriptions.'
    ' Use specific words likely to appear in tool names or descriptions to narrow down relevant tools.'
    ' Each query is independently tokenized; matches across queries are unioned.'
)


def _search_tools_signature(
    queries: Annotated[list[str], Field(description=_DEFAULT_PARAMETER_DESCRIPTION)],
) -> ToolSearchReturnContent:  # pragma: no cover - schema source only, never invoked
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


@dataclass(kw_only=True)
class _SearchTool(ToolsetTool[AgentDepsT]):
    """The local `search_tools` function, carrying the corpus it should search over.

    The real `ToolDefinition`s flow through to user-supplied search functions so
    callables can read whatever metadata they need (parameters schema, kind, etc.) — not
    just the name/description pair we'd otherwise expose.
    """

    corpus: list[ToolDefinition]


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

    Receives the run context, the list of search queries, and the deferred tool definitions, and
    returns the matching tool names ordered by relevance. Both sync and async implementations
    are accepted.
    """

    max_results: int = _MAX_SEARCH_RESULTS
    """Maximum number of matches returned from the default algorithm."""

    tool_description: str | None = None
    """Custom description for the `search_tools` function shown to the model."""

    parameter_description: str | None = None
    """Custom description for the `keywords` parameter shown to the model."""

    enable_fallback: bool = True
    """When False, the local `search_tools` function tool is not emitted — used when the
    capability commits to a named-native strategy that has no local equivalent (e.g.
    `'bm25'`, `'regex'`). With no fallback registered, `_resolve_native_tool_swap` raises
    on providers that can't honor the builtin, instead of silently substituting the local
    keyword algorithm; and on providers that DO support it, only the native tool reaches
    the wire (no redundant `search_tools` slot that could confuse the model)."""

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

        discovered = self._parse_discovered_tools(ctx)

        result: dict[str, ToolsetTool[AgentDepsT]] = dict(visible)

        # Single entry per deferred tool, keyed by its real name. `with_native`
        # stays set across the run (the tool is part of the search corpus regardless of
        # current discovery state); `defer_loading` reflects current visibility — flipped
        # to `False` once the tool is discovered. `Model.prepare_request` reads both
        # flags together to decide what reaches the wire (see the four-rule filter in
        # `_resolve_native_tool_swap`).
        for name, tool in deferred.items():
            managed_def = replace(
                tool.tool_def,
                with_native=_TOOL_SEARCH_BUILTIN_ID,
                defer_loading=name not in discovered,
            )
            result[name] = replace(tool, tool_def=managed_def)

        # Emit `search_tools` whenever the corpus is non-empty and a local fallback is
        # enabled. It carries `unless_native='tool_search'` so the adapter drops it on
        # the wire when the builtin is supported (the native path handles discovery
        # server-side); keeping it in the toolset across discovery steps preserves prompt
        # caching, since dropping it once everything is discovered would invalidate the
        # request prefix on the very next turn.
        #
        # When `enable_fallback=False` (named-native strategies `'bm25'`/`'regex'`) we
        # skip emission entirely: there's no local algorithm to fall back to, and emitting
        # it would both register a phantom fallback that suppresses the
        # "unsupported builtin" raise AND leave a redundant function tool on the wire
        # alongside the native builtin on providers that DO support it.
        if self.enable_fallback:
            result[_SEARCH_TOOLS_NAME] = self._build_search_tool(deferred, discovered)

        return result

    def _build_search_tool(
        self,
        deferred: dict[str, ToolsetTool[AgentDepsT]],
        discovered: set[str],
    ) -> _SearchTool[AgentDepsT]:
        parameter_description = self.parameter_description or _DEFAULT_PARAMETER_DESCRIPTION
        schema, args_validator = _build_search_args_schema(parameter_description)

        # Real `ToolDefinition`s for tools still pending discovery — what the user's
        # search function sees, and what the local keywords search indexes.
        corpus = [tool.tool_def for name, tool in deferred.items() if name not in discovered]

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
            max_retries=1,
            args_validator=args_validator,
            corpus=corpus,
        )

    def _parse_discovered_tools(self, ctx: RunContext[AgentDepsT]) -> set[str]:
        """Scan message history for previously-discovered tool names.

        Trusts that any [`ToolSearchReturnPart`][pydantic_ai.messages.ToolSearchReturnPart] /
        [`NativeToolSearchReturnPart`][pydantic_ai.messages.NativeToolSearchReturnPart]
        in the history has a validated [`ToolSearchReturnContent`][pydantic_ai.messages.ToolSearchReturnContent]:
        Pydantic's discriminator dispatch promotes from base parts on deserialization,
        and direct construction goes through the typed-class `__init__` (which Pydantic
        validates). No defensive isinstance walks needed.

        Also reads the legacy `metadata['discovered_tools']` sideband (validated against
        a TypedDict) so histories serialized before the typed-content migration continue
        to surface previously-discovered tools.
        """
        discovered: set[str] = set()
        for msg in ctx.messages:
            if isinstance(msg, ModelRequest):
                for part in msg.parts:
                    if isinstance(part, ToolSearchReturnPart):
                        self._collect_typed(part.content, discovered)
                    elif isinstance(part, ToolReturnPart) and part.tool_name == _SEARCH_TOOLS_NAME:
                        # Legacy histories carry discoveries on `metadata['discovered_tools']`
                        # rather than typed content. Narrowing tool_name + metadata shape avoids
                        # surfacing a user-defined `search_tools` whose metadata has no legacy
                        # shape.
                        self._collect_legacy(part.metadata, discovered)
            else:  # ModelResponse — the only other variant of ModelMessage.
                for part in msg.parts:
                    if isinstance(part, NativeToolSearchReturnPart):
                        self._collect_typed(part.content, discovered)
        return discovered

    @staticmethod
    def _collect_typed(content: ToolSearchReturnContent, discovered: set[str]) -> None:
        """Add discovered tool names from a validated [`ToolSearchReturnContent`][pydantic_ai.messages.ToolSearchReturnContent]."""
        discovered.update(match['name'] for match in content['discovered_tools'])

    @staticmethod
    def _collect_legacy(metadata: Any, discovered: set[str]) -> None:
        """Backward-compat reader for the pre-typed-content metadata sideband."""
        try:
            validated = _LEGACY_METADATA_TA.validate_python(metadata)
        except ValidationError:
            return
        discovered.update(validated['discovered_tools'])

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
        """Run the configured search strategy over the deferred-but-not-yet-discovered tools."""
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

        scored_matches: list[tuple[int, ToolSearchMatch]] = []
        for tool_def in search_tool.corpus:
            tool_terms = self._search_terms(tool_def.name, tool_def.description)
            score = len(terms & tool_terms)
            if score == 0:
                continue
            scored_matches.append((score, {'name': tool_def.name, 'description': tool_def.description}))

        if not scored_matches:
            return self._empty_return()

        scored_matches.sort(key=lambda item: item[0], reverse=True)
        matches = [match for _, match in scored_matches[: self.max_results]]
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
                matches.append({'name': tool_def.name, 'description': tool_def.description})

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
