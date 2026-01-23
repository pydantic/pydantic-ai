from __future__ import annotations as _annotations

from abc import ABC
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Union

import pydantic
from pydantic_core import core_schema
from typing_extensions import TypedDict, deprecated

__all__ = (
    'AbstractBuiltinTool',
    'WebSearchTool',
    'WebSearchUserLocation',
    'CodeExecutionTool',
    'WebFetchTool',
    'UrlContextTool',
    'ImageGenerationTool',
    'MemoryTool',
    'MCPServerTool',
    'FileSearchTool',
    'BUILTIN_TOOL_TYPES',
    'DEPRECATED_BUILTIN_TOOLS',
    'SUPPORTED_BUILTIN_TOOLS',
)

BUILTIN_TOOL_TYPES: dict[str, type[AbstractBuiltinTool]] = {}
"""Registry of all builtin tool types, keyed by their kind string.

This dict is populated automatically via `__init_subclass__` when tool classes are defined.
"""

ImageAspectRatio = Literal['21:9', '16:9', '4:3', '3:2', '1:1', '9:16', '3:4', '2:3', '5:4', '4:5']
"""Supported aspect ratios for image generation tools."""


@dataclass(kw_only=True)
class AbstractBuiltinTool(ABC):
    """A builtin tool that can be used by an agent.

    This class is abstract and cannot be instantiated directly.

    The builtin tools are passed to the model as part of the `ModelRequestParameters`.
    """

    kind: str = 'unknown_builtin_tool'
    """Built-in tool identifier, this should be available on all built-in tools as a discriminator."""

    @property
    def unique_id(self) -> str:
        """A unique identifier for the builtin tool.

        If multiple instances of the same builtin tool can be passed to the model, subclasses should override this property to allow them to be distinguished.
        """
        return self.kind

    @property
    def label(self) -> str:
        """Human-readable label for UI display.

        Subclasses should override this to provide a meaningful label.
        """
        return self.kind.replace('_', ' ').title()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        BUILTIN_TOOL_TYPES[cls.kind] = cls

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, handler: pydantic.GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        if cls is not AbstractBuiltinTool:
            return handler(cls)

        tools = BUILTIN_TOOL_TYPES.values()
        if len(tools) == 1:  # pragma: no cover
            tools_type = next(iter(tools))
        else:
            tools_annotated = [Annotated[tool, pydantic.Tag(tool.kind)] for tool in tools]
            tools_type = Annotated[Union[tuple(tools_annotated)], pydantic.Discriminator(_tool_discriminator)]  # noqa: UP007

        return handler(tools_type)


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


@dataclass(kw_only=True)
class CodeExecutionTool(AbstractBuiltinTool):
    """A builtin tool that allows your agent to execute code.

    Supported by:

    * Anthropic
    * OpenAI Responses
    * Google
    * Bedrock (Nova2.0)
    """

    kind: str = 'code_execution'
    """The kind of tool."""


@dataclass(kw_only=True)
class WebFetchTool(AbstractBuiltinTool):
    """Allows your agent to access contents from URLs.

    The parameters that PydanticAI passes depend on the model, as some parameters may not be supported by certain models.

    Supported by:

    * Anthropic
    * Google
    """

    max_uses: int | None = None
    """If provided, the tool will stop fetching URLs after the given number of uses.

    Supported by:

    * Anthropic
    """

    allowed_domains: list[str] | None = None
    """If provided, only these domains will be fetched.

    With Anthropic, you can only use one of `blocked_domains` or `allowed_domains`, not both.

    Supported by:

    * Anthropic, see <https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/web-fetch-tool#domain-filtering>
    """

    blocked_domains: list[str] | None = None
    """If provided, these domains will never be fetched.

    With Anthropic, you can only use one of `blocked_domains` or `allowed_domains`, not both.

    Supported by:

    * Anthropic, see <https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/web-fetch-tool#domain-filtering>
    """

    enable_citations: bool = False
    """If True, enables citations for fetched content.

    Supported by:

    * Anthropic
    """

    max_content_tokens: int | None = None
    """Maximum content length in tokens for fetched content.

    Supported by:

    * Anthropic
    """

    kind: str = 'web_fetch'
    """The kind of tool."""


@deprecated('Use `WebFetchTool` instead.')
@dataclass(kw_only=True)
class UrlContextTool(WebFetchTool):
    """Deprecated alias for WebFetchTool. Use WebFetchTool instead.

    Overrides kind to 'url_context' so old serialized payloads with {"kind": "url_context", ...}
    can be deserialized to UrlContextTool for backward compatibility.
    """

    kind: str = 'url_context'
    """The kind of tool (deprecated value for backward compatibility)."""


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

    output_compression: int | None = None
    """Compression level for the output image.

    Supported by:

    * OpenAI Responses. Only supported for 'jpeg' and 'webp' output formats. Default: 100.
    * Google (Vertex AI only). Only supported for 'jpeg' output format. Default: 75.
      Setting this will default `output_format` to 'jpeg' if not specified.
    """

    output_format: Literal['png', 'webp', 'jpeg'] | None = None
    """The output format of the generated image.

    Supported by:

    * OpenAI Responses. Default: 'png'.
    * Google (Vertex AI only). Default: 'png', or 'jpeg' if `output_compression` is set.
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

    size: Literal['auto', '1024x1024', '1024x1536', '1536x1024', '1K', '2K', '4K'] | None = None
    """The size of the generated image.

    * OpenAI Responses: 'auto' (default: model selects the size based on the prompt), '1024x1024', '1024x1536', '1536x1024'
    * Google (Gemini 3 Pro Image and later): '1K' (default), '2K', '4K'
    """

    aspect_ratio: ImageAspectRatio | None = None
    """The aspect ratio to use for generated images.

    Supported by:

    * Google image-generation models (Gemini)
    * OpenAI Responses (maps '1:1', '2:3', and '3:2' to supported sizes)
    """

    kind: str = 'image_generation'
    """The kind of tool."""


@dataclass(kw_only=True)
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
    """A unique identifier for the MCP server."""

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

    @property
    def unique_id(self) -> str:
        return ':'.join([self.kind, self.id])

    @property
    def label(self) -> str:
        return f'MCP: {self.id}'


@dataclass(kw_only=True)
class FileSearchTool(AbstractBuiltinTool):
    """A builtin tool that allows your agent to search through uploaded files using vector search.

    This tool provides a fully managed Retrieval-Augmented Generation (RAG) system that handles
    file storage, chunking, embedding generation, and context injection into prompts.

    Supported by:

    * OpenAI Responses
    * Google (Gemini)
    """

    file_store_ids: Sequence[str]
    """The file store IDs to search through.

    For OpenAI, these are the IDs of vector stores created via the OpenAI API.
    For Google, these are file search store names that have been uploaded and processed via the Gemini Files API.
    """

    kind: str = 'file_search'
    """The kind of tool."""


def _tool_discriminator(tool_data: dict[str, Any] | AbstractBuiltinTool) -> str:
    if isinstance(tool_data, dict):
        return tool_data.get('kind', AbstractBuiltinTool.kind)
    else:
        return tool_data.kind


DEPRECATED_BUILTIN_TOOLS: frozenset[type[AbstractBuiltinTool]] = frozenset({UrlContextTool})  # pyright: ignore[reportDeprecated]
"""Set of deprecated builtin tool IDs that should not be offered in new UIs."""

SUPPORTED_BUILTIN_TOOLS = frozenset(cls for cls in BUILTIN_TOOL_TYPES.values() if cls not in DEPRECATED_BUILTIN_TOOLS)
"""Get the set of all builtin tool types (excluding deprecated tools)."""

BUILTIN_TOOLS_REQUIRING_CONFIG: frozenset[type[AbstractBuiltinTool]] = frozenset({MCPServerTool, MemoryTool})
