from __future__ import annotations as _annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .builtin_tools import AbstractBuiltinTool
from abc import ABC
from dataclasses import dataclass
from typing import Literal

from typing_extensions import TypedDict

__all__ = ('AbstractBuiltinTool', 'WebSearchTool', 'WebSearchUserLocation', 'CodeExecutionTool', 'UrlContextTool')


@dataclass(kw_only=True)
class AbstractBuiltinTool(ABC):
    """A builtin tool that can be used by an agent.

    This class is abstract and cannot be instantiated directly.

    The builtin tools are passed to the model as part of the `ModelRequestParameters`.
    """

    kind: str = 'unknown_builtin_tool'
    """Built-in tool identifier, this should be available on all built-in tools as a discriminator."""


@dataclass(kw_only=True)
class WebSearchTool(AbstractBuiltinTool):
    """A builtin tool that allows your agent to search the web for information.

    The parameters that PydanticAI passes depend on the model, as some parameters may not be supported by certain models.

    Supported by:

    * Anthropic
    * OpenAI Responses
    * Groq
    * Google
    """

    search_context_size: Literal['low', 'medium', 'high'] = 'medium'
    """The `search_context_size` parameter controls how much context is retrieved from the web to help the tool formulate a response.

    Supported by:

    * OpenAI Responses
    """

    user_location: WebSearchUserLocation | None = None
    """The `user_location` parameter allows you to localize search results based on a user's location.

    Supported by:

    * Anthropic
    * OpenAI Responses
    """

    blocked_domains: list[str] | None = None
    """If provided, these domains will never appear in results.

    With Anthropic, you can only use one of `blocked_domains` or `allowed_domains`, not both.

    Supported by:

    * Anthropic, see <https://docs.anthropic.com/en/docs/build-with-claude/tool-use/web-search-tool#domain-filtering>
    * Groq, see <https://console.groq.com/docs/agentic-tooling#search-settings>
    """

    allowed_domains: list[str] | None = None
    """If provided, only these domains will be included in results.

    With Anthropic, you can only use one of `blocked_domains` or `allowed_domains`, not both.

    Supported by:

    * Anthropic, see <https://docs.anthropic.com/en/docs/build-with-claude/tool-use/web-search-tool#domain-filtering>
    * Groq, see <https://console.groq.com/docs/agentic-tooling#search-settings>
    """

    max_uses: int | None = None
    """If provided, the tool will stop searching the web after the given number of uses.

    Supported by:

    * Anthropic
    """

    kind: str = 'web_search'
    """The kind of tool."""


class WebSearchUserLocation(TypedDict, total=False):
    """Allows you to localize search results based on a user's location.

    Supported by:

    * Anthropic
    * OpenAI Responses
    """

    city: str
    """The city where the user is located."""

    country: str
    """The country where the user is located. For OpenAI, this must be a 2-letter country code (e.g., 'US', 'GB')."""

    region: str
    """The region or state where the user is located."""

    timezone: str
    """The timezone of the user's location."""


class CodeExecutionTool(AbstractBuiltinTool):
    """A builtin tool that allows your agent to execute code.

    Supported by:

    * Anthropic
    * OpenAI Responses
    * Google
    """

    kind: str = 'code_execution'
    """The kind of tool."""


class UrlContextTool(AbstractBuiltinTool):
    """Allows your agent to access contents from URLs.

    Supported by:

    * Google
    """

    kind: str = 'url_context'
    """The kind of tool."""


def merge_builtin_tools(
    base: Sequence[AbstractBuiltinTool] | None, runtime: Sequence[AbstractBuiltinTool] | None
) -> list[AbstractBuiltinTool]:
    """Merge two sets of builtin tools, with runtime tools having priority over base tools.

    Args:
        base: The base builtin tools (e.g., from agent initialization)
        runtime: The runtime builtin tools (e.g., from agent.run())

    Returns:
        A merged list of builtin tools with duplicates removed by type.
        Runtime tools take priority over base tools when both have the same type.
    """
    if not base and not runtime:
        return []

    if not base:
        return list(runtime) if runtime else []

    if not runtime:
        return list(base)

    # Create a mapping of tool types to tools, with runtime tools taking priority
    tool_map: dict[type[AbstractBuiltinTool], AbstractBuiltinTool] = {}

    # Add base tools first
    for tool in base:
        tool_map[type(tool)] = tool

    # Add runtime tools, which will override base tools of the same type
    for tool in runtime:
        tool_map[type(tool)] = tool

    return list(tool_map.values())
