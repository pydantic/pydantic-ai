from __future__ import annotations as _annotations

from abc import ABC
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from typing_extensions import TypedDict

if TYPE_CHECKING:
    from .builtin_tools import AbstractBuiltinTool

__all__ = (
    'AbstractBuiltinTool',
    'WebSearchTool',
    'WebSearchUserLocation',
    'CodeExecutionTool',
    'UrlContextTool',
    'ImageGenerationTool',
    'MemoryTool',
    'MCPServerTool',
)


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


@dataclass(kw_only=True)
class ImageGenerationTool(AbstractBuiltinTool):
    """A builtin tool that allows your agent to generate images.

    Supported by:

    * OpenAI Responses
    * Google
    """

    background: Literal['transparent', 'opaque', 'auto'] = 'auto'
    """Background type for the generated image.

    Supported by:

    * OpenAI Responses. 'transparent' is only supported for 'png' and 'webp' output formats.
    """

    input_fidelity: Literal['high', 'low'] | None = None
    """
    Control how much effort the model will exert to match the style and features,
    especially facial features, of input images.

    Supported by:

    * OpenAI Responses. Default: 'low'.
    """

    moderation: Literal['auto', 'low'] = 'auto'
    """Moderation level for the generated image.

    Supported by:

    * OpenAI Responses
    """

    output_compression: int = 100
    """Compression level for the output image.

    Supported by:

    * OpenAI Responses. Only supported for 'png' and 'webp' output formats.
    """

    output_format: Literal['png', 'webp', 'jpeg'] | None = None
    """The output format of the generated image.

    Supported by:

    * OpenAI Responses. Default: 'png'.
    """

    partial_images: int = 0
    """
    Number of partial images to generate in streaming mode.

    Supported by:

    * OpenAI Responses. Supports 0 to 3.
    """

    quality: Literal['low', 'medium', 'high', 'auto'] = 'auto'
    """The quality of the generated image.

    Supported by:

    * OpenAI Responses
    """

    size: Literal['1024x1024', '1024x1536', '1536x1024', 'auto'] = 'auto'
    """The size of the generated image.

    Supported by:

    * OpenAI Responses
    """

    kind: str = 'image_generation'
    """The kind of tool."""


class MemoryTool(AbstractBuiltinTool):
    """A builtin tool that allows your agent to use memory.

    Supported by:

    * Anthropic
    """

    kind: str = 'memory'
    """The kind of tool."""


@dataclass(kw_only=True)
class MCPServerTool(AbstractBuiltinTool):
    """A builtin tool that allows your agent to use MCP servers.

    Supported by:

    * OpenAI Responses
    * Anthropic
    """

    id: str
    """The ID of the MCP server."""

    url: str
    """The URL of the MCP server to use.

    For OpenAI Responses, it is possible to use `connector_id` by providing it as `x-openai-connector:<connector_id>`.
    """

    authorization_token: str | None = None
    """Authorization header to use when making requests to the MCP server.

    Supported by:

    * OpenAI Responses
    * Anthropic
    """

    description: str | None = None
    """A description of the MCP server.

    Supported by:

    * OpenAI Responses
    """

    allowed_tools: list[str] | None = None
    """A list of tools that the MCP server can use.

    Supported by:

    * OpenAI Responses
    * Anthropic
    """

    headers: dict[str, str] | None = None
    """Optional HTTP headers to send to the MCP server.

    Use for authentication or other purposes.

    Supported by:

    * OpenAI Responses
    """

    kind: str = 'mcp_server'

    LIST_TOOLS_KIND: Literal['mcp_server:mcp_list_tools'] = 'mcp_server:mcp_list_tools'
    CALL_TOOL_KIND: Literal['mcp_server:mcp_call_tool'] = 'mcp_server:mcp_call_tool'
