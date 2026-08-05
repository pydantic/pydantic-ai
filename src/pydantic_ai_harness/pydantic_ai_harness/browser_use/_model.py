"""A browser-use chat model backed by a Pydantic AI model.

Lets the browser-use sub-agent run on the same model configuration as the host
agent: one provider setup, Pydantic AI's structured-output handling (tool
calling with validation retries, instead of browser-use's provider-sensitive
`response_format`), and Logfire tracing when `logfire.instrument_pydantic_ai()`
is active.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias, TypeVar, overload

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import (
    BinaryContent,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserContent,
    UserPromptPart,
)
from pydantic_ai.models import KnownModelName, Model, infer_model
from pydantic_ai.usage import RunUsage

try:
    from browser_use.llm.base import BaseChatModel
    from browser_use.llm.messages import (
        AssistantMessage,
        BaseMessage,
        ContentPartImageParam,
        ContentPartTextParam,
        ImageURL,
        SystemMessage,
        UserMessage,
    )
    from browser_use.llm.views import ChatInvokeCompletion, ChatInvokeUsage
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'browser-use is required for BrowserUse. Install it with: pip install "pydantic-ai-harness[browser-use]"'
    ) from _import_error

T = TypeVar('T', bound=BaseModel)

ChatModelInput: TypeAlias = 'BaseChatModel | Model | KnownModelName | str'
"""What the capability accepts as a chat model: a browser-use `BaseChatModel`,
or anything Pydantic AI's `infer_model` takes (a `Model` instance or a model
name string), which is wrapped in `PydanticAIChatModel`."""


def _image_content(image_url: ImageURL) -> ImageUrl | BinaryContent:
    """A browser-use image part as Pydantic AI user content.

    Data URIs are decoded to `BinaryContent` (providers do not accept multi-MB
    data URLs as remote URLs); anything else stays a URL reference.
    """
    if image_url.url.startswith('data:'):
        content = BinaryContent.from_data_uri(image_url.url)
        if not content.media_type:
            content = BinaryContent(data=content.data, media_type=image_url.media_type)
        return content
    return ImageUrl(url=image_url.url)


def _user_content(content: str | list[ContentPartTextParam | ContentPartImageParam]) -> str | Sequence[UserContent]:
    if isinstance(content, str):
        return content
    parts: list[UserContent] = []
    for part in content:
        if isinstance(part, ContentPartTextParam):
            parts.append(part.text)
        else:
            parts.append(_image_content(part.image_url))
    return parts


def _assistant_text(message: AssistantMessage) -> str:
    content = message.content
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    return '\n'.join(part.text if isinstance(part, ContentPartTextParam) else part.refusal for part in content)


def _system_text(content: str | list[ContentPartTextParam]) -> str:
    if isinstance(content, str):
        return content
    return '\n'.join(part.text for part in content)


def _map_messages(messages: list[BaseMessage]) -> list[ModelMessage]:
    """Map a browser-use conversation onto Pydantic AI's message structure.

    Consecutive system/user messages collapse into one `ModelRequest`;
    assistant messages become `ModelResponse`s. The conversation must end with
    a request, since the mapped history is sent for a fresh model turn.
    """
    mapped: list[ModelMessage] = []
    pending: list[ModelRequestPart] = []
    for message in messages:
        if isinstance(message, SystemMessage):
            pending.append(SystemPromptPart(content=_system_text(message.content)))
        elif isinstance(message, UserMessage):
            pending.append(UserPromptPart(content=_user_content(message.content)))
        else:
            text = _assistant_text(message)
            if not text:
                continue
            if pending:
                mapped.append(ModelRequest(parts=pending))
                pending = []
            mapped.append(ModelResponse(parts=[TextPart(content=text)]))
    if not pending:
        raise ValueError('The browser-use conversation must end with a system or user message.')
    mapped.append(ModelRequest(parts=pending))
    return mapped


def _map_usage(usage: RunUsage) -> ChatInvokeUsage:
    return ChatInvokeUsage(
        prompt_tokens=usage.input_tokens,
        prompt_cached_tokens=usage.cache_read_tokens or None,
        prompt_cache_creation_tokens=usage.cache_write_tokens or None,
        prompt_image_tokens=None,
        completion_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
    )


class PydanticAIChatModel(BaseChatModel):
    """Implements browser-use's `BaseChatModel` protocol on top of a Pydantic AI model.

    Each `ainvoke` maps the browser-use conversation onto Pydantic AI messages
    and runs one model turn through an internal `Agent`, passing the step's
    output type per call. Structured output uses Pydantic AI's output handling
    (tool calling by default, with validation retries), which works uniformly
    across providers -- including ones that reject browser-use's
    `response_format` JSON schema.
    """

    # browser-use verifies provider API keys before the first step by reading its own
    # environment variables; a Pydantic AI model already carries its provider configuration,
    # so declare the keys verified and let the model raise if it is misconfigured.
    _verified_api_keys: bool = True

    def __init__(self, model: Model | KnownModelName | str) -> None:
        self._model = infer_model(model)
        self.model: str = self._model.model_name
        self._agent = Agent(self._model)

    @property
    def provider(self) -> str:
        """The wrapped model's provider identifier."""
        return self._model.system

    @property
    def name(self) -> str:
        """The wrapped model's name."""
        return self._model.model_name

    @overload
    async def ainvoke(  # pragma: no cover - overload is enforced by static type checking
        self, messages: list[BaseMessage], output_format: None = None, **kwargs: object
    ) -> ChatInvokeCompletion[str]: ...

    @overload
    async def ainvoke(  # pragma: no cover - overload is enforced by static type checking
        self, messages: list[BaseMessage], output_format: type[T], **kwargs: object
    ) -> ChatInvokeCompletion[T]: ...

    async def ainvoke(
        self, messages: list[BaseMessage], output_format: type[T] | None = None, **kwargs: object
    ) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
        """Run one model turn over the mapped conversation, optionally with structured output."""
        history = _map_messages(messages)
        if output_format is None:
            text_result = await self._agent.run(None, message_history=history)
            return ChatInvokeCompletion(completion=text_result.output, usage=_map_usage(text_result.usage))
        result = await self._agent.run(None, output_type=output_format, message_history=history)
        return ChatInvokeCompletion(completion=result.output, usage=_map_usage(result.usage))


def resolve_chat_model(llm: BaseChatModel | Model | KnownModelName | str | None) -> BaseChatModel | None:
    """A browser-use chat model from whatever the user configured.

    A browser-use `BaseChatModel` passes through; a Pydantic AI `Model` or a
    model name string is wrapped in `PydanticAIChatModel`; `None` stays `None`
    (browser-use's own default model selection).
    """
    if llm is None or isinstance(llm, BaseChatModel):
        return llm
    return PydanticAIChatModel(llm)
