"""Tool search toolset.

`ToolSearchToolset` wraps another toolset to support discovery of tools marked with
`defer_loading=True`. Depending on whether the running model supports native provider-side
tool search (via the [`ToolSearchTool`][pydantic_ai.builtin_tools.ToolSearchTool] builtin),
the wrapped tools are routed through one of two paths:

* **Native path**: each deferred tool is exposed under a managed key
  ``{name}~managed:tool_search`` with ``ToolDefinition.managed_by_builtin='tool_search'``.
  The model adapter keeps these tools in the request (setting ``defer_loading=True`` on the
  wire) and drops the local ``search_tools`` function via its ``prefer_builtin``.
* **Local path**: the model adapter drops the managed entries via ``managed_by_builtin``,
  keeps the local ``search_tools`` function tool, and only exposes already-discovered
  deferred tools until more are discovered via search.

Since toolsets return a flat ``dict[str, ToolsetTool]``, both representations are emitted
with different keys but dispatch to the same underlying tool on call.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import cache
from typing import Annotated, Any

from pydantic import Field, TypeAdapter
from typing_extensions import TypedDict

from .._run_context import AgentDepsT, RunContext
from ..builtin_tools import ToolSearchFunc
from ..exceptions import ModelRetry, UserError
from ..messages import BuiltinToolReturnPart, ModelRequest, ToolReturn, ToolReturnPart
from ..tools import ToolDefinition
from .abstract import ToolsetTool
from .wrapper import WrapperToolset

_SEARCH_TOOLS_NAME = 'search_tools'
_TOOL_SEARCH_BUILTIN_ID = 'tool_search'
_MANAGED_KEY_SUFFIX = f'~managed:{_TOOL_SEARCH_BUILTIN_ID}'

_DISCOVERED_TOOLS_METADATA_KEY = 'discovered_tools'

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

        # Native path: expose every deferred tool under a managed key so the model adapter
        # can emit it with provider-specific defer_loading on the wire. `managed_by_builtin`
        # causes the non-native adapter to drop these — the local path uses the regular
        # entries below.
        for name, tool in deferred.items():
            managed_def = replace(tool.tool_def, managed_by_builtin=_TOOL_SEARCH_BUILTIN_ID)
            result[f'{name}{_MANAGED_KEY_SUFFIX}'] = replace(tool, tool_def=managed_def)

        # Local path: only already-discovered deferred tools are visible under their real name
        # until the model calls `search_tools` to discover more. Preserve definition order.
        for name, tool in deferred.items():
            if name in discovered:
                result[name] = tool

        # If every deferred tool is already discovered, no need for the search function.
        if not discovered.issuperset(deferred):
            search_tool = self._build_search_tool(deferred, discovered)
            result[_SEARCH_TOOLS_NAME] = search_tool

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

        search_tool_def = ToolDefinition(
            name=_SEARCH_TOOLS_NAME,
            description=self.tool_description or _DEFAULT_TOOL_DESCRIPTION,
            parameters_json_schema=schema,
            prefer_builtin=_TOOL_SEARCH_BUILTIN_ID,
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
        tool_names = metadata.get(_DISCOVERED_TOOLS_METADATA_KEY)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        if not isinstance(tool_names, list):
            return
        discovered.update(item for item in tool_names if isinstance(item, str))  # pyright: ignore[reportUnknownVariableType]

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        if name == _SEARCH_TOOLS_NAME and isinstance(tool, _SearchTool):
            return await self._search_tools(tool_args, tool)
        # Strip the `~managed:tool_search` suffix so the wrapped toolset receives the real name.
        original_name = name.removesuffix(_MANAGED_KEY_SUFFIX)
        return await self.wrapped.call_tool(original_name, tool_args, ctx, tool)

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
                metadata={_DISCOVERED_TOOLS_METADATA_KEY: []},
            )

        scored_matches.sort(key=lambda item: item[0], reverse=True)
        matches = [match for _, match in scored_matches[: self.max_results]]
        tool_names = [match['name'] for match in matches]

        return ToolReturn(
            return_value=matches,
            metadata={_DISCOVERED_TOOLS_METADATA_KEY: tool_names},
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
                metadata={_DISCOVERED_TOOLS_METADATA_KEY: []},
            )

        return ToolReturn(
            return_value=matches,
            metadata={_DISCOVERED_TOOLS_METADATA_KEY: [m['name'] for m in matches]},
        )
