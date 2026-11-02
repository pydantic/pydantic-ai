"""A browser-use chat model backed by a Pydantic AI model.

Lets the browser-use sub-agent run on the same model configuration as the host
agent: one provider setup, Pydantic AI's structured-output handling (tool
calling with validation retries, instead of browser-use's provider-sensitive
`response_format`), and Logfire tracing when `logfire.instrument_pydantic_ai()`
is active.
"""

from __future__ import annotations

from collections.abc import Generator, Sequence
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, TypeAlias, TypeVar, overload

from pydantic import BaseModel
from typing_extensions import TypeIs

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability, Instrumentation, WrapperCapability
from pydantic_ai.exceptions import UsageLimitExceeded, UserError
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
from pydantic_ai.models import AbstractModel, KnownModelName, Model, infer_model
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.usage import RequestUsage, RunUsage, UsageLimits

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
except ImportError as _import_error:
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


@dataclass
class _HostBudget:
    """The host run's usage and limits, shared by every browser model call of one `browse_web` call."""

    usage: RunUsage
    limits: UsageLimits | None
    exceeded: UsageLimitExceeded | None = None
    """The first limit error, kept because browser-use swallows errors from its steps, judge, and extraction."""

    @contextmanager
    def spend(self) -> Generator[None]:
        """Refuse a call once the budget is exhausted, and remember the error that exhausts it."""
        if self.exceeded is not None:
            # A fresh error each time, so repeated refusals do not pile frames onto one traceback.
            raise UsageLimitExceeded(str(self.exceeded)) from self.exceeded
        try:
            yield
        except UsageLimitExceeded as error:
            self.exceeded = error
            raise


_host_budget: ContextVar[_HostBudget | None] = ContextVar('browser_use_host_budget', default=None)


@contextmanager
def host_budget(ctx: RunContext[Any]) -> Generator[_HostBudget]:
    """Charge the `PydanticAIChatModel` calls made inside this block to the host run.

    A context variable rather than a per-call model, so it reaches every Pydantic AI backed model
    browser-use calls -- the main, extraction, judge, and fallback models, including ones a custom
    `BrowserAgentFactory` built -- without rebuilding them per run.
    """
    budget = _HostBudget(usage=ctx.usage, limits=ctx.usage_limits)
    token = _host_budget.set(budget)
    try:
        yield budget
    finally:
        _host_budget.reset(token)


def host_budget_exhausted() -> bool:
    """Whether a browser model call in this `browse_web` call has hit a host usage limit."""
    budget = _host_budget.get()
    return budget is not None and budget.exceeded is not None


def _call_usage(result: AgentRunResult[Any]) -> RequestUsage:
    """The tokens of this call alone; the run's own `usage` is the host total when the budget is shared."""
    usage = RequestUsage()
    for message in result.new_messages():
        if isinstance(message, ModelResponse):
            usage += message.usage
    return usage


def _map_usage(usage: RequestUsage) -> ChatInvokeUsage:
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
        self._agent = Agent(self._model, name='browser_use')

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
        """Run one model turn over the mapped conversation, optionally with structured output.

        Inside `browse_web`, the turn spends the host run's usage and is bounded by its limits.
        """
        history = _map_messages(messages)
        budget = _host_budget.get()
        usage = budget.usage if budget is not None else None
        usage_limits = budget.limits if budget is not None else None
        with budget.spend() if budget is not None else nullcontext():
            if output_format is None:
                text_result = await self._agent.run(
                    None, message_history=history, usage=usage, usage_limits=usage_limits
                )
                return ChatInvokeCompletion(completion=text_result.output, usage=_map_usage(_call_usage(text_result)))
            result = await self._agent.run(
                None, output_type=output_format, message_history=history, usage=usage, usage_limits=usage_limits
            )
            return ChatInvokeCompletion(completion=result.output, usage=_map_usage(_call_usage(result)))


def _is_request_response_model(model: AbstractModel) -> TypeIs[Model]:
    """Narrow a run's model without losing its provider client type."""
    return isinstance(model, Model)


class RunChatModel(PydanticAIChatModel):
    """Adapt the host run's model and instrumentation for a capability without an `llm`."""

    def __init__(self, ctx: RunContext[AgentDepsT]) -> None:
        model = ctx.model
        if not _is_request_response_model(model):
            raise UserError(
                "BrowserUse without an `llm` runs the browser agent on the host run's model, but "
                f'{model.model_id!r} is not a request-response model. Pass `llm` to BrowserUse.'
            )
        super().__init__(model)
        # Without a host `Instrumentation`, stay uninstrumented rather than fall back to global defaults.
        self._agent.instrument = False

        def select(capability: AbstractCapability[AgentDepsT]) -> None:
            # The last `Instrumentation` in application order instruments the host run; a resolved
            # capability function's one is visited inside its wrapper.
            while isinstance(capability, WrapperCapability):
                capability = capability.wrapped
            if isinstance(capability, Instrumentation):
                self._agent.instrument = capability.settings

        if ctx.root_capability is not None:
            ctx.root_capability.apply(select)


def resolve_chat_model(llm: BaseChatModel | Model | KnownModelName | str | None) -> BaseChatModel | None:
    """A browser-use chat model from whatever the user configured.

    A browser-use `BaseChatModel` passes through; a Pydantic AI `Model` or a
    model name string is wrapped in `PydanticAIChatModel`; `None` stays `None`
    (browser-use's own default model selection).
    """
    if llm is None or isinstance(llm, BaseChatModel):
        return llm
    return PydanticAIChatModel(llm)
