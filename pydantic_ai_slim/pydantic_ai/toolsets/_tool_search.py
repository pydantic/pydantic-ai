"""Tool search toolset and strategy types.

`ToolSearchToolset` wraps another toolset to support discovery of tools marked with
`defer_loading=True`. Rather than commit to native-vs-local at toolset time (which can't
know which model will actually serve the request — think ``FallbackModel``), the toolset
emits **both** representations of every deferred tool and lets
[`Model.prepare_request`][pydantic_ai.models.Model.prepare_request] filter to one based
on the specific model's support for the
[`ToolSearchTool`][pydantic_ai.builtin_tools.ToolSearchTool] builtin.

Dict keys follow the convention:

* ``{name}~managed:tool_search`` → deferred tool in **managed** form (carries
  ``managed_by_builtin='tool_search'`` on its ``ToolDefinition``). Always present for
  every deferred tool. When the model supports native tool search, the adapter keeps
  this in the request and sets ``defer_loading=True`` on the wire.
* ``{name}`` → deferred tool in **regular** form (no ``managed_by_builtin``). Only
  present when the tool has been discovered — either by our local ``search_tools``
  function or by a previous provider-native search, both of which write
  ``discovered_tools`` into message history. When the model does NOT support native
  tool search, the adapter keeps this variant and drops the managed one.
* ``search_tools`` → the local discovery function with ``prefer_builtin='tool_search'``.
  Dropped by the adapter when the builtin is supported.

The two variants of a deferred tool share the same underlying ``ToolsetTool`` — only the
wrapping ``ToolDefinition`` differs (``managed_by_builtin`` flag set or not). The
model calls the tool by its plain name, and the ``ToolManager`` dispatches by
``tool_def.name`` rather than dict key, so either variant resolves correctly.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from functools import cache
from typing import Annotated, Any, Literal, Union

from pydantic import Field, TypeAdapter
from typing_extensions import TypedDict

from .._run_context import AgentDepsT, RunContext
from ..builtin_tools import TOOL_SEARCH_FUNCTION_TOOL_NAME, ToolSearchNativeStrategy
from ..exceptions import ModelRetry, UserError
from ..messages import BuiltinToolReturnPart, ModelRequest, ToolReturn, ToolReturnPart
from ..tools import ToolDefinition
from .abstract import ToolsetTool
from .wrapper import WrapperToolset

ToolSearchFunc = Callable[[str, Sequence[ToolDefinition]], Sequence[str]]
"""Custom search function for
[`ToolSearch`][pydantic_ai.capabilities.ToolSearch]'s ``strategy`` field.

Takes the natural-language query and the deferred tool definitions, and returns the
matching tool names ordered by relevance.
"""

ToolSearchLocalStrategy = Literal['substring']
"""Named local tool search strategy.

``'substring'`` opts into the built-in token-overlap algorithm explicitly — use this
to lock in the current local algorithm rather than the ``None`` default (which lets
Pydantic AI pick the best algorithm per provider and may change over time).
"""

ToolSearchStrategy = Union[ToolSearchFunc, ToolSearchLocalStrategy, ToolSearchNativeStrategy]  # noqa: UP007
"""Strategy value accepted by [`ToolSearch.strategy`][pydantic_ai.capabilities.ToolSearch.strategy].

* ``None`` (default, on the capability): let Pydantic AI pick the best strategy for the
  current provider — native on supporting models, the default local algorithm elsewhere.
  The choice may change in future versions.
* ``'substring'``: force the local token-overlap algorithm regardless of provider.
* ``'bm25'`` / ``'regex'``: force a specific provider-native strategy (Anthropic). The
  request fails on providers that can't honor the choice.
* Callable ``(query, tools) -> names``: custom search function. Used locally, and also
  by the native "client-executed" surface on providers that support it (Anthropic custom
  tool-reference blocks, OpenAI ``ToolSearchToolParam(execution='client')``).
"""

DISCOVERED_TOOLS_METADATA_KEY = 'discovered_tools'
"""Key on ``ToolReturnPart.metadata`` / ``BuiltinToolReturnPart.metadata`` that carries
the list of tool names discovered by a tool-search turn.

This is the contract between the tool-search toolset and the provider adapters. The
toolset writes it when the local ``search_tools`` function runs; adapters write it when
mapping the provider's native tool-search result block (Anthropic) or output item
(OpenAI). Replay reads it via :func:`extract_discovered_tool_names` to shape the
provider-specific tool-search round-trip format. Stored on the return part (not on the
call part) because the result of the discovery — not the request for it — is what the
next turn needs to act on.
"""


def extract_discovered_tool_names(metadata: Any) -> list[str] | None:
    """Read the discovered-tool-names list off of a tool return's metadata.

    Returns ``None`` when the metadata doesn't carry the convention (not a dict, key
    absent, or value isn't a list of strings). Callers should treat ``None`` as
    "no discovery data"; an empty list means "search ran, nothing matched".
    """
    if not isinstance(metadata, dict):
        return None
    value = metadata.get(DISCOVERED_TOOLS_METADATA_KEY)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    if not isinstance(value, list):
        return None
    return [item for item in value if isinstance(item, str)]  # pyright: ignore[reportUnknownVariableType]


_SEARCH_TOOLS_NAME = TOOL_SEARCH_FUNCTION_TOOL_NAME
_TOOL_SEARCH_BUILTIN_ID = 'tool_search'
_MANAGED_KEY_SUFFIX = f'~managed:{_TOOL_SEARCH_BUILTIN_ID}'


def _managed_key(name: str) -> str:
    """Dict key for the managed-variant entry of a deferred tool in the toolset output."""
    return f'{name}{_MANAGED_KEY_SUFFIX}'


_MAX_SEARCH_RESULTS = 10
_SEARCH_TOKEN_RE = re.compile(r'[a-z0-9]+')

_DEFAULT_TOOL_DESCRIPTION = (
    'There are additional tools not yet visible to you.'
    ' When you need a capability not provided by your current tools,'
    ' search here by providing specific keywords to discover and activate relevant tools.'
    ' Each keyword is matched independently against tool names and descriptions.'
    ' If no tools are found, they do not exist — do not retry.'
)

_DEFAULT_SEARCH_GUIDANCE = (
    'Space-separated keywords to match against tool names and descriptions.'
    ' Use specific words likely to appear in tool names or descriptions to narrow down relevant tools.'
)


@cache
def _build_search_args_schema(search_guidance: str) -> tuple[dict[str, Any], TypeAdapter[Any]]:
    """Build the `search_tools` parameter schema for the given guidance text.

    Cached per-guidance: the default guidance is used for every agent step with deferred
    tools, so we only pay for class and adapter construction once per distinct guidance
    value.
    """

    class _SearchToolArgs(TypedDict):
        keywords: Annotated[str, Field(description=search_guidance)]

    ta = TypeAdapter(_SearchToolArgs)
    schema = ta.json_schema()
    # TypeAdapter doesn't support config= for TypedDict; rename away from the private class.
    schema['title'] = 'SearchToolArgs'
    return schema, ta


@dataclass(kw_only=True)
class _SearchIndexEntry:
    name: str
    description: str | None
    search_terms: set[str]


@dataclass(kw_only=True)
class _SearchTool(ToolsetTool[AgentDepsT]):
    search_index: list[_SearchIndexEntry]


@dataclass
class ToolSearchToolset(WrapperToolset[AgentDepsT]):
    """A toolset that enables tool discovery for large toolsets.

    Wraps another toolset and exposes a ``search_tools`` function that lets the model
    discover tools with ``defer_loading=True``. Tools with ``defer_loading=True`` are
    not initially presented to the model — they become available after the model
    discovers them via search.

    When the model supports the [`ToolSearchTool`][pydantic_ai.builtin_tools.ToolSearchTool]
    builtin, discovery is handled by the provider and the deferred tools are sent to the API
    with ``defer_loading=True`` on the wire.
    """

    search_fn: ToolSearchFunc | None = None
    """Optional custom search function. If ``None``, the default token-overlap algorithm is used.

    Receives the raw query string and the deferred tool definitions, and returns the matching
    tool names ordered by relevance.
    """

    max_results: int = _MAX_SEARCH_RESULTS
    """Maximum number of matches returned from the default algorithm."""

    tool_description: str | None = None
    """Custom description for the ``search_tools`` function shown to the model."""

    search_guidance: str | None = None
    """Custom description for the ``keywords`` parameter shown to the model."""

    local_fallback: bool = True
    """Whether to register the local ``search_tools`` function tool.

    When ``False``, deferred tools appear only in their managed form and discovery must go
    through the provider's native tool search. Used by
    [`ToolSearch`][pydantic_ai.capabilities.ToolSearch] when an explicit named native
    strategy (``'bm25'`` / ``'regex'``) is configured — falling back to a different local
    algorithm would silently ignore the user's choice, so we skip the local tool entirely
    and let ``prepare_request`` raise if the native builtin is unavailable.
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

        discovered = self._parse_discovered_tools(ctx)

        result: dict[str, ToolsetTool[AgentDepsT]] = dict(visible)

        # Emit both representations of every deferred tool so the model adapter
        # (inside `Model.prepare_request` with the concrete model in hand — the decision
        # that can't be made here because of `FallbackModel`) can filter to whichever
        # path the specific model supports:
        #
        # * ``{name}~managed:tool_search`` — always present; carries
        #   ``managed_by_builtin='tool_search'``. Kept by adapters whose model supports
        #   native tool search.
        # * ``{name}`` — only present for already-discovered tools. Kept by adapters
        #   whose model doesn't support native tool search.
        #
        # Both entries share the same underlying dispatch target; the surviving entry
        # determines which `ToolDefinition` is sent to the model. Duplication here stays
        # bounded: at most `N + k` entries where `k` is the number of discovered tools
        # (not `2N`), because undiscovered tools only appear in the managed slot.
        for name, tool in deferred.items():
            managed_def = replace(tool.tool_def, managed_by_builtin=_TOOL_SEARCH_BUILTIN_ID)
            result[_managed_key(name)] = replace(tool, tool_def=managed_def)
            if name in discovered:
                result[name] = tool

        # `search_tools` carries `prefer_builtin='tool_search'` (when fallback-to-local
        # is appropriate) — the adapter drops it when the builtin is supported. Skip
        # emitting it when ``local_fallback=False`` (the capability's named-native mode:
        # we must not silently substitute a different local algorithm for the user's
        # explicit strategy choice) or once every deferred tool is already discovered.
        if self.local_fallback and not discovered.issuperset(deferred):
            result[_SEARCH_TOOLS_NAME] = self._build_search_tool(deferred, discovered)

        return result

    def _build_search_tool(
        self,
        deferred: dict[str, ToolsetTool[AgentDepsT]],
        discovered: set[str],
    ) -> _SearchTool[AgentDepsT]:
        search_guidance = self.search_guidance or _DEFAULT_SEARCH_GUIDANCE
        schema, args_ta = _build_search_args_schema(search_guidance)

        search_index = [
            _SearchIndexEntry(
                name=name,
                description=tool.tool_def.description,
                search_terms=self._search_terms(name, tool.tool_def.description),
            )
            for name, tool in deferred.items()
            if name not in discovered
        ]

        # `prefer_builtin` tells the adapter to drop this function tool when the native
        # builtin is supported. That's what we want for server-side strategies (the
        # provider handles search entirely). For a custom callable strategy, the native
        # path on both Anthropic (regular function tool with tool_reference result
        # formatting) and OpenAI (`execution='client'`) still needs the local function
        # tool to execute the search, so we leave `prefer_builtin` unset in that case.
        search_tool_def = ToolDefinition(
            name=_SEARCH_TOOLS_NAME,
            description=self.tool_description or _DEFAULT_TOOL_DESCRIPTION,
            parameters_json_schema=schema,
            prefer_builtin=_TOOL_SEARCH_BUILTIN_ID if self.search_fn is None else None,
        )

        return _SearchTool(
            toolset=self,
            tool_def=search_tool_def,
            max_retries=1,
            args_validator=args_ta.validator,  # pyright: ignore[reportArgumentType]
            search_index=search_index,
        )

    def _parse_discovered_tools(self, ctx: RunContext[AgentDepsT]) -> set[str]:
        """Scan message history for tool discovery metadata.

        Picks up metadata written by our local ``search_tools`` (on
        ``ToolReturnPart`` in a ``ModelRequest``) and by the provider's native tool
        search (on ``BuiltinToolReturnPart`` in a ``ModelResponse`` — each adapter
        writes the ``discovered_tools`` key when mapping the provider's result block).
        """
        discovered: set[str] = set()
        for msg in ctx.messages:
            if isinstance(msg, ModelRequest):
                for part in msg.parts:
                    if isinstance(part, ToolReturnPart) and part.tool_name == _SEARCH_TOOLS_NAME:
                        self._collect_discovered_from_metadata(part.metadata, discovered)
            else:  # ModelResponse — the only other variant of ModelMessage.
                for part in msg.parts:
                    if isinstance(part, BuiltinToolReturnPart) and part.tool_name == _TOOL_SEARCH_BUILTIN_ID:
                        self._collect_discovered_from_metadata(part.metadata, discovered)
        return discovered

    @staticmethod
    def _collect_discovered_from_metadata(metadata: Any, discovered: set[str]) -> None:
        if not isinstance(metadata, dict):
            return
        tool_names = metadata.get(DISCOVERED_TOOLS_METADATA_KEY)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        if not isinstance(tool_names, list):
            return
        discovered.update(item for item in tool_names if isinstance(item, str))  # pyright: ignore[reportUnknownVariableType]

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        if name == _SEARCH_TOOLS_NAME and isinstance(tool, _SearchTool):
            return await self._search_tools(tool_args, tool)
        return await self.wrapped.call_tool(name, tool_args, ctx, tool)

    @staticmethod
    def _search_terms(name: str, description: str | None) -> set[str]:
        search_terms = set(_SEARCH_TOKEN_RE.findall(name.lower()))
        if description:
            search_terms.update(_SEARCH_TOKEN_RE.findall(description.lower()))
        return search_terms

    async def _search_tools(self, tool_args: dict[str, Any], search_tool: _SearchTool[AgentDepsT]) -> ToolReturn:
        """Run the configured search strategy over the deferred-but-not-yet-discovered tools."""
        keywords = tool_args['keywords']
        if not keywords:
            raise ModelRetry('Please provide search keywords.')

        if self.search_fn is not None:
            return self._run_search_fn(keywords, search_tool)
        return self._run_default_search(keywords, search_tool)

    def _run_default_search(self, keywords: str, search_tool: _SearchTool[AgentDepsT]) -> ToolReturn:
        """Score each tool by how many query tokens appear in its name/description.

        Tokenizes on alphanumeric runs for both the query and the indexed terms, so the
        top hit for "github profile" is ``github_get_me`` (two matches) without matching
        substrings inside longer words like ``comment`` for the query ``me``.
        """
        terms = self._search_terms(keywords, None)
        if not terms:
            raise ModelRetry('Please provide search keywords.')

        scored_matches: list[tuple[int, dict[str, str | None]]] = []
        for entry in search_tool.search_index:
            score = len(terms & entry.search_terms)
            if score == 0:
                continue
            scored_matches.append((score, {'name': entry.name, 'description': entry.description}))

        if not scored_matches:
            return ToolReturn(
                return_value='No matching tools found. The tools you need may not be available.',
                metadata={DISCOVERED_TOOLS_METADATA_KEY: []},
            )

        scored_matches.sort(key=lambda item: item[0], reverse=True)
        matches = [match for _, match in scored_matches[: self.max_results]]
        tool_names = [match['name'] for match in matches]

        return ToolReturn(
            return_value=matches,
            metadata={DISCOVERED_TOOLS_METADATA_KEY: tool_names},
        )

    def _run_search_fn(self, keywords: str, search_tool: _SearchTool[AgentDepsT]) -> ToolReturn:
        """Invoke a user-provided strategy and normalize its return to the metadata shape."""
        assert self.search_fn is not None

        tool_defs_by_name = {entry.name: entry for entry in search_tool.search_index}
        tool_defs: Sequence[ToolDefinition] = [
            ToolDefinition(name=entry.name, description=entry.description) for entry in search_tool.search_index
        ]

        matched = list(self.search_fn(keywords, tool_defs))[: self.max_results]

        matches: list[dict[str, str | None]] = []
        for name in matched:
            if entry := tool_defs_by_name.get(name):
                matches.append({'name': entry.name, 'description': entry.description})

        if not matches:
            return ToolReturn(
                return_value='No matching tools found. The tools you need may not be available.',
                metadata={DISCOVERED_TOOLS_METADATA_KEY: []},
            )

        return ToolReturn(
            return_value=matches,
            metadata={DISCOVERED_TOOLS_METADATA_KEY: [m['name'] for m in matches]},
        )
