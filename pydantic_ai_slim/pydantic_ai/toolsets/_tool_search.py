"""Tool search toolset and strategy types.

`ToolSearchToolset` wraps another toolset to support discovery of tools marked with
`defer_loading=True`. Rather than commit to native-vs-local at toolset time (which can't
know which model will actually serve the request — think `FallbackModel`), the toolset
emits one entry per deferred tool with both `with_builtin='tool_search'` and the
current local visibility on `defer_loading`, then lets
[`Model.prepare_request`][pydantic_ai.models.Model.prepare_request] filter based on the
specific model's support for the [`ToolSearchTool`][pydantic_ai.builtin_tools.tool_search.ToolSearchTool]
builtin:

* On the native path the adapter keeps every corpus member (regardless of local
  discovery state) and applies its provider-specific wire format — e.g. setting
  `defer_loading=True` on the Anthropic / OpenAI Responses tool param so the provider
  drives discovery server-side.
* On the local path corpus members with `defer_loading=True` (still undiscovered) are
  dropped from the wire; discovered ones (`defer_loading=False`) stay so the model can
  call them by their real name.

`search_tools`, the local discovery function, carries `unless_builtin='tool_search'`
and is dropped by the adapter when the builtin is supported.
"""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass, replace
from functools import cache
from typing import Annotated, Any

from pydantic import Field, TypeAdapter
from typing_extensions import TypedDict

from .._run_context import AgentDepsT, RunContext
from ..builtin_tools.tool_search import (
    TOOL_SEARCH_FUNCTION_TOOL_NAME,
    ToolSearchFunc,
    ToolSearchMatch,
    ToolSearchReturnContent,
    ToolSearchTool,
)
from ..exceptions import ModelRetry, UserError
from ..messages import (
    BuiltinToolReturnPart,
    BuiltinToolSearchReturnPart,
    ModelRequest,
    ToolReturn,
    ToolReturnPart,
    ToolSearchReturnPart,
)
from ..tools import ToolDefinition
from .abstract import ToolsetTool
from .wrapper import WrapperToolset

_SEARCH_TOOLS_NAME = TOOL_SEARCH_FUNCTION_TOOL_NAME
_TOOL_SEARCH_BUILTIN_ID = ToolSearchTool.kind

_LEGACY_DISCOVERED_TOOLS_METADATA_KEY = 'discovered_tools'
"""Legacy metadata key for previously-discovered tool names.

Pre-typed-content versions of this toolset wrote discovered tool names to
`ToolReturnPart.metadata['discovered_tools']` instead of the typed
[`ToolSearchReturnContent`][pydantic_ai.builtin_tools.tool_search.ToolSearchReturnContent]
on `content`. Read-only — kept for backward-compat parsing of persisted histories. New
writes go to the typed content."""


_MAX_SEARCH_RESULTS = 10
_SEARCH_TOKEN_RE = re.compile(r'[a-z0-9]+')

_DEFAULT_TOOL_DESCRIPTION = (
    'There are additional tools not yet visible to you.'
    ' When you need a capability not provided by your current tools,'
    ' search here by providing specific keywords to discover and activate relevant tools.'
    ' Each keyword is matched independently against tool names and descriptions.'
    ' If no tools are found, they do not exist — do not retry.'
)


@cache
def _build_search_args_schema(parameter_description: str) -> tuple[dict[str, Any], TypeAdapter[Any]]:
    """Build the `search_tools` parameter schema for the given description.

    Cached per-description: the default description is used for every agent step with
    deferred tools, so we only pay for class and adapter construction once per distinct
    description value.
    """

    class _SearchToolArgs(TypedDict):
        keywords: Annotated[str, Field(description=parameter_description)]

    ta = TypeAdapter(_SearchToolArgs)
    schema = ta.json_schema()
    # TypeAdapter doesn't support config= for TypedDict; rename away from the private class.
    schema['title'] = 'SearchToolArgs'
    return schema, ta


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

    When the model supports the [`ToolSearchTool`][pydantic_ai.builtin_tools.tool_search.ToolSearchTool]
    builtin, discovery is handled by the provider and the deferred tools are sent to the API
    with `defer_loading=True` on the wire.
    """

    search_fn: ToolSearchFunc[AgentDepsT] | None = None
    """Optional custom search function. If `None`, the default keyword-overlap algorithm is used.

    Receives the run context, the raw query string, and the deferred tool definitions, and
    returns the matching tool names ordered by relevance. Both sync and async implementations
    are accepted.
    """

    max_results: int = _MAX_SEARCH_RESULTS
    """Maximum number of matches returned from the default algorithm."""

    tool_description: str | None = None
    """Custom description for the `search_tools` function shown to the model."""

    parameter_description: str | None = None
    """Custom description for the `keywords` parameter shown to the model."""

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

        # Single entry per deferred tool, keyed by its real name. `with_builtin`
        # stays set across the run (the tool is part of the search corpus regardless of
        # current discovery state); `defer_loading` reflects current visibility — flipped
        # to `False` once the tool is discovered. `Model.prepare_request` reads both
        # flags together to decide what reaches the wire (see the four-rule filter in
        # `_resolve_builtin_tool_swap`).
        for name, tool in deferred.items():
            managed_def = replace(
                tool.tool_def,
                with_builtin=_TOOL_SEARCH_BUILTIN_ID,
                defer_loading=name not in discovered,
            )
            result[name] = replace(tool, tool_def=managed_def)

        # Emit `search_tools` whenever the corpus is non-empty — we always reach this
        # point with deferred tools to manage. It carries `unless_builtin='tool_search'`
        # so the adapter drops it when the builtin is supported (the native path handles
        # discovery server-side). Keeping it across discovery steps preserves prompt
        # caching: dropping it once everything is discovered would invalidate the
        # request prefix on the very next turn.
        result[_SEARCH_TOOLS_NAME] = self._build_search_tool(deferred, discovered)

        return result

    def _build_search_tool(
        self,
        deferred: dict[str, ToolsetTool[AgentDepsT]],
        discovered: set[str],
    ) -> _SearchTool[AgentDepsT]:
        parameter_description = self.parameter_description or self._DEFAULT_PARAMETER_DESCRIPTION
        schema, args_ta = _build_search_args_schema(parameter_description)

        # Real `ToolDefinition`s for tools still pending discovery — what the user's
        # search function sees, and what the local keywords search indexes.
        corpus = [tool.tool_def for name, tool in deferred.items() if name not in discovered]

        # `unless_builtin` tells the adapter to drop this function tool when the native
        # builtin is supported. That's what we want for server-side strategies (the
        # provider handles search entirely). For a custom callable strategy, the native
        # path on both Anthropic (regular function tool with tool_reference result
        # formatting) and OpenAI (`execution='client'`) still needs the local function
        # tool to execute the search, so we leave `unless_builtin` unset in that case.
        search_tool_def = ToolDefinition(
            name=_SEARCH_TOOLS_NAME,
            description=self.tool_description or _DEFAULT_TOOL_DESCRIPTION,
            parameters_json_schema=schema,
            unless_builtin=_TOOL_SEARCH_BUILTIN_ID if self.search_fn is None else None,
        )

        return _SearchTool(
            toolset=self,
            tool_def=search_tool_def,
            max_retries=1,
            args_validator=args_ta.validator,  # pyright: ignore[reportArgumentType]
            corpus=corpus,
        )

    def _parse_discovered_tools(self, ctx: RunContext[AgentDepsT]) -> set[str]:
        """Scan message history for previously-discovered tool names.

        Reads the typed
        [`ToolSearchReturnContent`][pydantic_ai.builtin_tools.tool_search.ToolSearchReturnContent]
        off of `content` for both the local `search_tools` return (on
        [`ToolSearchReturnPart`][pydantic_ai.messages.ToolSearchReturnPart] in a
        `ModelRequest`) and the provider-native return (on
        [`BuiltinToolSearchReturnPart`][pydantic_ai.messages.BuiltinToolSearchReturnPart]
        / [`BuiltinToolReturnPart`][pydantic_ai.messages.BuiltinToolReturnPart] in a
        `ModelResponse` — each adapter normalizes its provider's wire format into the
        same shape).

        Also reads the legacy `metadata['discovered_tools']` sideband on
        `ToolReturnPart` so message histories serialized on `main` (before the
        typed-content migration) continue to surface previously-discovered tools.
        """
        discovered: set[str] = set()
        for msg in ctx.messages:
            if isinstance(msg, ModelRequest):
                for part in msg.parts:
                    # The local-fallback path produces `ToolSearchReturnPart` (a typed
                    # `ToolReturnPart` subclass), reached via the `isinstance` check.
                    # Direct `ToolReturnPart('search_tools', ...)` constructions
                    # (legacy fixtures, fresh user code) fall through to the legacy
                    # metadata reader on `ToolReturnPart`.
                    if isinstance(part, ToolSearchReturnPart):
                        self._collect_discovered_from_typed_content(part.content, discovered)
                    elif isinstance(part, ToolReturnPart) and part.tool_name == _SEARCH_TOOLS_NAME:
                        promoted = ToolReturnPart.narrow_type(part)
                        # Defensive: registered narrower for `search_tools` always returns
                        # the typed subclass.
                        if isinstance(promoted, ToolSearchReturnPart):  # pragma: no branch
                            self._collect_discovered_from_typed_content(promoted.content, discovered)
                        self._collect_discovered_from_legacy_metadata(part.metadata, discovered)
            else:  # ModelResponse — the only other variant of ModelMessage.
                for part in msg.parts:
                    if isinstance(part, BuiltinToolSearchReturnPart):
                        self._collect_discovered_from_typed_content(part.content, discovered)
                    elif isinstance(part, BuiltinToolReturnPart) and part.tool_name == _TOOL_SEARCH_BUILTIN_ID:
                        # Base `BuiltinToolReturnPart` instances reach this branch when
                        # constructed directly (e.g. legacy test fixtures) — promote and
                        # read off the typed view.
                        promoted = BuiltinToolReturnPart.narrow_type(part)
                        # Defensive: registered narrower for `tool_search` always returns
                        # the typed subclass.
                        if isinstance(promoted, BuiltinToolSearchReturnPart):  # pragma: no branch
                            self._collect_discovered_from_typed_content(promoted.content, discovered)
        return discovered

    @staticmethod
    def _collect_discovered_from_typed_content(
        content: ToolSearchReturnContent | str | None, discovered: set[str]
    ) -> None:
        if not isinstance(content, dict):
            return
        matches = content.get('discovered_tools')
        if not isinstance(matches, list):
            return
        for match in matches:
            if not isinstance(match, dict):
                continue
            name = match.get('name')
            if isinstance(name, str):
                discovered.add(name)

    @staticmethod
    def _collect_discovered_from_legacy_metadata(metadata: Any, discovered: set[str]) -> None:
        """Backward-compat reader for the pre-typed-content metadata sideband.

        Earlier versions stashed discovered tool names on
        `ToolReturnPart.metadata['discovered_tools']` instead of on the typed
        `content`. Persisted histories from those versions still need to surface
        their discoveries on resume.
        """
        if not isinstance(metadata, dict):
            return
        names = metadata.get(_LEGACY_DISCOVERED_TOOLS_METADATA_KEY)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        if not isinstance(names, list):
            return
        discovered.update(name for name in names if isinstance(name, str))  # pyright: ignore[reportUnknownVariableType]

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
    ) -> ToolReturn:
        """Run the configured search strategy over the deferred-but-not-yet-discovered tools."""
        keywords = tool_args['keywords']
        if not keywords:
            raise ModelRetry('Please provide search keywords.')

        if self.search_fn is not None:
            return await self._run_search_fn(keywords, ctx, search_tool)
        return self._run_keywords_search(keywords, search_tool)

    # The `'keywords'` local strategy and its default parameter description live together —
    # this sets up cleanly for a future strategy registry without doing the registry now.
    # Future: extract a `_LOCAL_STRATEGIES` dict if/when a second local strategy lands.
    _DEFAULT_PARAMETER_DESCRIPTION = (
        'Space-separated keywords to match against tool names and descriptions.'
        ' Use specific words likely to appear in tool names or descriptions to narrow down relevant tools.'
    )

    def _run_keywords_search(self, keywords: str, search_tool: _SearchTool[AgentDepsT]) -> ToolReturn:
        """Score each tool by how many query keywords appear in its name/description.

        Tokenizes on alphanumeric runs for both the query and the indexed terms, so the
        top hit for "github profile" is `github_get_me` (two matches) without matching
        substrings inside longer words like `comment` for the query `me`.
        """
        terms = self._search_terms(keywords, None)
        if not terms:
            raise ModelRetry('Please provide search keywords.')

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
        self, keywords: str, ctx: RunContext[AgentDepsT], search_tool: _SearchTool[AgentDepsT]
    ) -> ToolReturn:
        """Invoke a user-provided strategy, validating that the returned names are known."""
        assert self.search_fn is not None

        tool_defs_by_name = {tool_def.name: tool_def for tool_def in search_tool.corpus}

        result = self.search_fn(ctx, keywords, search_tool.corpus)
        if inspect.isawaitable(result):
            result = await result

        matches: list[ToolSearchMatch] = []
        for name in list(result)[: self.max_results]:
            if (tool_def := tool_defs_by_name.get(name)) is not None:
                matches.append({'name': tool_def.name, 'description': tool_def.description})

        if not matches:
            return self._empty_return()
        return self._build_return(matches)

    _NO_MATCHES_MESSAGE = 'No matching tools found. The tools you need may not be available.'

    @classmethod
    def _empty_return(cls) -> ToolReturn:
        """Shaped "no matches" return: empty `discovered_tools` list + a user-visible note.

        The note is sent to the model as separate content (`ToolReturn.content`) so
        the model doesn't retry searching with the same keywords, while
        `return_value` stays as a typed
        [`ToolSearchReturnContent`][pydantic_ai.builtin_tools.tool_search.ToolSearchReturnContent]
        with the `message` slot mirroring the same text for adapters that need it
        on the wire (Anthropic custom-callable empty-results path).
        """
        return_value: ToolSearchReturnContent = {
            'discovered_tools': [],
            'message': cls._NO_MATCHES_MESSAGE,
        }
        return ToolReturn(
            return_value=return_value,
            content=cls._NO_MATCHES_MESSAGE,
        )

    @staticmethod
    def _build_return(matches: list[ToolSearchMatch]) -> ToolReturn:
        """Shaped matches return: typed [`ToolSearchReturnContent`][pydantic_ai.builtin_tools.tool_search.ToolSearchReturnContent]."""
        return_value: ToolSearchReturnContent = {'discovered_tools': matches}
        return ToolReturn(return_value=return_value)
