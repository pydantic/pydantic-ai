"""Internals for the tool-search builtin.

Tool search lets the model discover tools marked with `defer_loading=True` rather
than carrying every deferred tool's full schema in the prompt. The
[`ToolSearch`][pydantic_ai.capabilities.ToolSearch] capability picks between three
modes per provider:

* **Native server-side**: the provider's own tool-search tool (Anthropic
  `bm25`/`regex`, OpenAI server-executed `tool_search`).
* **Native client-executed (custom callable)**: the provider invokes our local
  callable through its native client-execution mode (Anthropic regular function tool
  + `tool_reference` result blocks, OpenAI `ToolSearchToolParam(execution='client')`).
* **Local fallback**: a regular `search_tools` function tool the model can call.
  Used on providers that don't expose any native tool-search surface.

The end-user surface is the [`ToolSearch`][pydantic_ai.capabilities.ToolSearch]
capability; the types in this module are kept here as implementation detail.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Union

from typing_extensions import TypedDict

from .._run_context import AgentDepsT, RunContext
from . import AbstractBuiltinTool

if TYPE_CHECKING:
    from ..tools import ToolDefinition


class ToolSearchMatch(TypedDict):
    """A single match in a tool search result."""

    name: str
    """Name of the discovered tool, as the model will call it."""

    description: str | None
    """Human-readable description, if the tool provided one."""


class ToolSearchReturn(TypedDict):
    """Typed return value of [`ToolSearchTool`][pydantic_ai.builtin_tools.tool_search.ToolSearchTool].

    Carried on `ToolReturnPart.content` when the local `search_tools` function runs
    (custom-callable native paths) — model adapters read this typed value via
    [`extract_tool_search_return`][pydantic_ai.builtin_tools.tool_search.extract_tool_search_return]
    to shape the provider-specific round-trip format. Future provider-native builtins
    (bash, text editor, apply patch, etc.) are expected to follow the same per-tool
    TypedDict pattern, with the `BuiltinToolReturnPart.content` discrimination
    converging on [issue #3561](https://github.com/pydantic/pydantic-ai/issues/3561).
    """

    tools: list[ToolSearchMatch]
    """Matches ordered by relevance. An empty list means "search ran, nothing matched"
    and adapters fall through to the default text-formatting path (Anthropic rejects an
    empty `tool_result` content list)."""


def extract_tool_search_return(content: Any) -> ToolSearchReturn | None:
    """Read a typed [`ToolSearchReturn`][pydantic_ai.builtin_tools.tool_search.ToolSearchReturn] off of a return part's `content` value.

    Returns `None` when `content` doesn't carry the expected shape — not a dict, or
    `tools` is missing / not a list of `{name, description}` entries. Callers should
    treat `None` as "not a tool-search return"; `ToolSearchReturn(tools=[])` means
    "search ran, nothing matched".
    """
    if not isinstance(content, dict):
        return None
    tools = content.get('tools')  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    if not isinstance(tools, list):
        return None
    matches: list[ToolSearchMatch] = []
    for entry in tools:  # pyright: ignore[reportUnknownVariableType]
        if not isinstance(entry, dict):
            continue
        name = entry.get('name')  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        if not isinstance(name, str):
            continue
        description = entry.get('description')  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        matches.append({'name': name, 'description': description if isinstance(description, str) else None})
    return {'tools': matches}


ToolSearchNativeStrategy = Literal['bm25', 'regex']
"""Named provider-native tool search strategy carried on
[`ToolSearchTool`][pydantic_ai.builtin_tools.tool_search.ToolSearchTool].

`'bm25'` and `'regex'` correspond to Anthropic's server-side tool search variants.
OpenAI's Responses API does not expose distinct named native strategies, so these values
are rejected by the OpenAI adapter.
"""

ToolSearchLocalStrategy = Literal['keywords']
"""Named local tool search strategy.

`'keywords'` opts into the built-in keyword-overlap algorithm explicitly — use this
to lock in the current local algorithm rather than the `None` default (which lets
Pydantic AI pick the best algorithm per provider and may change over time). Matches
the parameter name (`keywords:`) on the local `search_tools` function tool.
"""

ToolSearchFunc = Callable[
    [RunContext[AgentDepsT], str, Sequence['ToolDefinition']],
    Sequence[str] | Awaitable[Sequence[str]],
]
"""Custom search function for
[`ToolSearch`][pydantic_ai.capabilities.ToolSearch]'s `strategy` field.

Takes the run context, the natural-language query, and the deferred tool definitions,
and returns the matching tool names ordered by relevance. Both sync and async
implementations are accepted.

Usage `ToolSearchFunc[AgentDepsT]`.
"""

ToolSearchStrategy = Union[ToolSearchFunc[AgentDepsT], ToolSearchLocalStrategy, ToolSearchNativeStrategy]  # noqa: UP007
"""Strategy value accepted by [`ToolSearch.strategy`][pydantic_ai.capabilities.ToolSearch.strategy].

* `'keywords'`: force the local keyword-overlap algorithm regardless of provider.
* `'bm25'` / `'regex'`: force a specific provider-native strategy (Anthropic). The
  request fails on providers that can't honor the choice.
* Callable `(ctx, query, tools) -> names`: custom search function. Used locally, and also
  by the native "client-executed" surface on providers that support it (Anthropic custom
  tool-reference blocks, OpenAI `ToolSearchToolParam(execution='client')`).

`None` is not part of the union — it's accepted as the default on the
[`ToolSearch.strategy`][pydantic_ai.capabilities.ToolSearch.strategy] field and means
"let Pydantic AI pick"; see that field's docstring for details.
"""


TOOL_SEARCH_FUNCTION_TOOL_NAME = 'search_tools'
"""Name of the local function tool that backs [`ToolSearch`][pydantic_ai.capabilities.ToolSearch]
for keyword-based discovery when native tool search isn't available, and that model adapters
route to for provider-side "client-executed" custom callable modes (Anthropic tool-reference
blocks; OpenAI `execution='client'`)."""


_ToolSearchToolStrategy = Literal['bm25', 'regex', 'custom']
"""Internal strategy literal carried on [`ToolSearchTool.strategy`][pydantic_ai.builtin_tools.tool_search.ToolSearchTool.strategy].

Extends [`ToolSearchNativeStrategy`][pydantic_ai.builtin_tools.tool_search.ToolSearchNativeStrategy]
with `'custom'`, which marks an instance whose discovery is performed by a callable on our
side. The user-facing `ToolSearchStrategy` union (in the
[`ToolSearch`][pydantic_ai.capabilities.ToolSearch] capability) does not include
`'custom'` — users pass the callable directly and the capability sets
`strategy='custom'` on the builtin internally.
"""


@dataclass(kw_only=True)
class ToolSearchTool(AbstractBuiltinTool):
    """A builtin tool that enables native provider tool search.

    Tools marked as part of the search corpus (via `with_builtin='tool_search'`
    on their [`ToolDefinition`][pydantic_ai.tools.ToolDefinition]) are sent to supporting
    providers with `defer_loading` on the wire; the provider manages their visibility
    and only exposes them once they've been discovered.

    The mode of discovery depends on `strategy`:

    * A named native strategy (or `None` for the provider default): the provider runs
      the search server-side using its own indexing (Anthropic `bm25`/`regex`, OpenAI
      server-executed `tool_search`).
    * `'custom'`: the provider invokes our local search function to answer each search
      request. On Anthropic this goes via a regular function tool whose return value the
      adapter re-formats as `tool_reference` blocks; on OpenAI it goes via
      `ToolSearchToolParam(execution='client')` with our callable's parameter schema.

    When the model doesn't support native tool search at all, the
    [`ToolSearch`][pydantic_ai.capabilities.ToolSearch] capability's local
    implementation handles discovery via its own `search_tools` function tool.

    Supported by:

    * Anthropic (bm25, regex, custom callable) — Sonnet 4.5+, Opus 4.5+, Haiku 4.5+
    * OpenAI Responses (server default, custom callable via `execution='client'`) — GPT-5.4+
      (named strategies `'bm25'`/`'regex'` are not supported).
    """

    strategy: _ToolSearchToolStrategy | None = None
    """The search strategy to use.

    * `None` (default): use the provider's default native search. On Anthropic this is
      `bm25`; on OpenAI it is the server-executed `tool_search` tool.
    * `'bm25'` / `'regex'`: force a specific Anthropic native strategy. Adapters on
      providers that can't honor the choice raise `UserError`.
    * `'custom'`: discovery is performed by a callable on our side; provider adapters
      that support a "client-executed" native surface wire that surface up so the model
      sees a tool search call rather than a regular function tool. Set automatically by
      [`ToolSearch`][pydantic_ai.capabilities.ToolSearch] when its `strategy` is a
      callable; users don't pass `'custom'` directly.
    """

    optional: bool = False
    """Whether this instance is a best-effort upgrade rather than a hard requirement.

    When `True`, the instance is silently dropped from the request on a model that doesn't
    support native tool search, instead of raising when no local fallback is provided.

    Defaults to `False` (the user explicitly asked for native tool search; fail loudly if
    we can't honor it). [`ToolSearch`][pydantic_ai.capabilities.ToolSearch] sets it to
    `True` for the auto-injected default and callable strategies so models that don't
    support native tool search transparently fall back to the local `search_tools`
    function. The named native strategies `'bm25'` / `'regex'` keep `optional=False` —
    silently substituting a different algorithm would ignore the user's choice.
    """

    kind: str = 'tool_search'
    """The kind of tool."""
