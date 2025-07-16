from __future__ import annotations as _annotations

from abc import ABC
from dataclasses import dataclass
from typing import Literal

from typing_extensions import TypedDict

__all__ = ('AbstractBuiltinTool', 'WebSearchTool', 'UserLocation')


@dataclass
class AbstractBuiltinTool(ABC):
    """A builtin tool that can be used by an agent.

    This class is abstract and cannot be instantiated directly.

    The builtin tools are passed to the model as part of the `ModelRequestParameters`.
    """

    # TODO(Marcelo): We need to handle custom tool definitions per model base.
    # def handle_custom_tool_definition(self, model: str) -> Any: ...


@dataclass
class WebSearchTool(AbstractBuiltinTool):
    """A builtin tool that allows your agent to search the web for information.

    The parameters that PydanticAI passes depend on the model, as some parameters may not be supported by certain models.

    Supported by:
    * OpenAI
    * Anthropic
    * Groq
    """

    search_context_size: Literal['low', 'medium', 'high'] = 'medium'
    """The `search_context_size` parameter controls how much context is retrieved from the web to help the tool formulate a response.

    Supported by:
    * OpenAI
    """

    user_location: UserLocation | None = None
    """The `user_location` parameter allows you to localize search results based on a user's location.

    Supported by:
    * Anthropic
    * OpenAI
    """

    blocked_domains: list[str] | None = None
    """If provided, these domains will never appear in results.

    With Anthropic, you can only use one of `blocked_domains` or `allowed_domains`, not both.

    Supported by:
    * Anthropic (https://docs.anthropic.com/en/docs/build-with-claude/tool-use/web-search-tool#domain-filtering)
    * Groq (https://console.groq.com/docs/agentic-tooling#search-settings)
    * MistralAI
    """

    allowed_domains: list[str] | None = None
    """If provided, only these domains will be included in results.

    With Anthropic, you can only use one of `blocked_domains` or `allowed_domains`, not both.

    Supported by:
    * Anthropic (https://docs.anthropic.com/en/docs/build-with-claude/tool-use/web-search-tool#domain-filtering)
    * Groq (https://console.groq.com/docs/agentic-tooling#search-settings)
    """

    max_uses: int | None = None
    """If provided, the tool will stop searching the web after the given number of uses.

    Supported by:
    * Anthropic
    """


class UserLocation(TypedDict, total=False):
    """Allows you to localize search results based on a user's location.

    Supported by:
    * Anthropic
    * OpenAI
    """

    city: str
    country: str
    region: str
    timezone: str


class CodeExecutionTool(AbstractBuiltinTool):
    """A builtin tool that allows your agent to execute code.

    Supported by:
    * Anthropic
    * Google
    """
