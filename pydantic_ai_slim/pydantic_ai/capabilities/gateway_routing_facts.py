from __future__ import annotations

import re
from collections.abc import Collection, Sequence
from dataclasses import KW_ONLY, dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

from pydantic_ai.messages import ModelMessage, ModelRequestPart, ModelResponse, RetryFeedbackPart, ToolReturnPart
from pydantic_ai.models import Model
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext

from .abstract import AbstractCapability, CapabilityOrdering

if TYPE_CHECKING:
    from pydantic_ai.models import ModelRequestContext

__all__ = ('GatewayRoutingFacts',)

_RequestKind: TypeAlias = Literal['user_prompt', 'tool_results', 'tool_retry', 'output_retry']

_MAX_STEP = 10_000
_MAX_RETRY = 100

# Printable ASCII only: httpx refuses a non-ASCII header value and h11 one with a line break, so such a value would
# fail the request rather than only leave its header out.
_HEADER_VALUE = re.compile(r'[\x20-\x7e]{1,128}')


@dataclass
class GatewayRoutingFacts(AbstractCapability[Any]):
    """Sends facts about the agent with each model request that goes through the [Pydantic AI Gateway](../gateway.md).

    The Gateway's Smart routing rules can match on these facts, and the conversation ID lets the Gateway keep a
    conversation on one provider. Every agent has this capability: a model request that doesn't go through the
    Gateway is sent unchanged. Each request carries these headers:

    - `x-pydantic-agent-name`: the agent's [`name`][pydantic_ai.agent.Agent.name], or the name inferred from the
      variable the agent is assigned to, without surrounding whitespace. Left out when the agent has no name, or
      when the name is longer than 128 characters or has a character outside printable ASCII.
    - `x-pydantic-agent-step`: the run's [`run_step`][pydantic_ai.tools.RunContext.run_step], from 1 to 10000.
    - `x-pydantic-agent-request-kind`: what the request answers. `output_retry` when it asks the model to fix its
      output, `tool_retry` when it asks the model to call a tool again, `tool_results` when it returns tool
      results, and `user_prompt` otherwise.
    - `x-pydantic-agent-retry`: for `output_retry` and `tool_retry`, which retry this is, from the
      [`attempt`][pydantic_ai.messages.RetryInfo.attempt] of the retry parts, up to 100; otherwise `0`.
    - `x-routing-conversation-id`: the run's [`conversation_id`][pydantic_ai.tools.RunContext.conversation_id].
      Left out when it is longer than 128 characters or has a character outside printable ASCII.

    The headers go into the request's [`extra_headers`][pydantic_ai.settings.ModelSettings.extra_headers], next to
    the ones already set. A header you set yourself in `extra_headers` is sent as you set it.

    With [`ModelFallback`][pydantic_ai.capabilities.ModelFallback], only the attempts at a model behind the Gateway
    carry the headers. A [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] sends one set of settings
    to every model it tries, so it carries them only when all of its models are behind the Gateway and set the
    same `extra_headers` of their own.

    To stop sending the headers, pass `GatewayRoutingFacts(enabled=False)` in `capabilities`, to the agent or to
    one run. See [Routing facts](../gateway.md#routing-facts).
    """

    _: KW_ONLY

    enabled: bool = True
    """Whether to send the headers. Pass `False` to send model requests through the Gateway without them."""

    id: str | None = 'gateway_routing_facts'
    """One-off: an agent either sends the facts or doesn't, so the id is fixed by default.

    Two of them resolve to one via [`combine`][pydantic_ai.capabilities.AbstractCapability.combine], which keeps
    the last. So `GatewayRoutingFacts(enabled=False)` passed to a run replaces the agent's for that run.
    """

    def get_ordering(self) -> CapabilityOrdering:
        # Innermost, so the hook runs after any capability that moves the attempt to another model.
        return CapabilityOrdering(position='innermost')

    @property
    def _has_prepare_model_request(self) -> bool:
        # The agent reads this to decide whether each attempt copies the tool definitions, in case a hook edits them
        # in place. This hook never touches them, and every agent has it, so it mustn't cost every agent those copies.
        return type(self).prepare_model_request is not GatewayRoutingFacts.prepare_model_request

    def _realtime_uncalled_hooks(self) -> tuple[str, ...]:
        # A realtime session sends no model requests through the Gateway's HTTP routes, and every agent has this
        # capability, so it mustn't make every session warn. Measured against this class, so a subclass that adds
        # its own hooks still warns.
        return self._realtime_uncalled_hooks_overriding(GatewayRoutingFacts)

    async def prepare_model_request(
        self,
        ctx: RunContext[Any],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Add the routing headers to this attempt's `extra_headers`, when its model is behind the Gateway."""
        if not self.enabled:
            return request_context
        models = _serving_models(request_context.model)
        providers = [provider for model in models if (provider := model.provider) is not None]
        if len(providers) < len(models):
            return request_context
        # Imported here because every agent has this capability, and a model with no provider, such as `TestModel`,
        # never needs the Gateway provider module.
        from pydantic_ai.providers.gateway import is_gateway_provider

        if not all(is_gateway_provider(provider) for provider in providers):
            return request_context
        headers = _current_headers(request_context.model_settings, models)
        if headers is None:
            return request_context

        # HTTP field names are case-insensitive, so a case variant of a header counts as set.
        supplied = {name.lower() for name in headers}
        output_tool_names = {tool.name for tool in request_context.model_request_parameters.output_tools}
        for name, value in _routing_headers(ctx, request_context.messages, output_tool_names).items():
            if name not in supplied:
                headers[name] = value

        model_settings = (request_context.model_settings or ModelSettings()).copy()
        model_settings['extra_headers'] = headers
        return replace(request_context, model_settings=model_settings)


def _serving_models(model: Model) -> list[Model]:
    """The models an attempt at `model` can be sent to: each model a `FallbackModel` tries, or `model` itself."""
    while isinstance(model, WrapperModel):
        model = model.wrapped
    if isinstance(model, FallbackModel):
        return [serving for fallback in model.models for serving in _serving_models(fallback)]
    return [model]


def _current_headers(model_settings: ModelSettings | None, models: list[Model]) -> dict[str, str] | None:
    """A copy of the `extra_headers` the attempt would send, or `None` when its models wouldn't all send the same ones.

    The request's settings are shallowly merged over each model's own `settings`, so `extra_headers` set on the
    request replaces any a model sets for itself. When the request sets none, the headers start from the models'
    own, which a `FallbackModel` can only send to all of its models when they agree.
    """
    if model_settings is not None and 'extra_headers' in model_settings:
        return dict(model_settings['extra_headers'])
    own = [(model.settings or ModelSettings()).get('extra_headers') for model in models]
    if any(headers != own[0] for headers in own):
        return None
    return dict(own[0] or {})


def _routing_headers(
    ctx: RunContext[Any], messages: Sequence[ModelMessage], output_tool_names: Collection[str]
) -> dict[str, str]:
    kind, retry = _request_kind(messages, output_tool_names)
    headers: dict[str, str] = {}
    if ctx.agent is not None and ctx.agent.name is not None and (name := _header_value(ctx.agent.name.strip())):
        headers['x-pydantic-agent-name'] = name
    headers['x-pydantic-agent-step'] = str(min(max(ctx.run_step, 1), _MAX_STEP))
    headers['x-pydantic-agent-request-kind'] = kind
    headers['x-pydantic-agent-retry'] = str(retry)
    if ctx.conversation_id is not None and (conversation_id := _header_value(ctx.conversation_id)):
        headers['x-routing-conversation-id'] = conversation_id
    return headers


def _header_value(value: str) -> str | None:
    return value if _HEADER_VALUE.fullmatch(value) else None


def _request_kind(messages: Sequence[ModelMessage], output_tool_names: Collection[str]) -> tuple[_RequestKind, int]:
    """What the request about to be sent answers, and which retry it is.

    The request is every `ModelRequest` after the last `ModelResponse`, as the agent merges consecutive requests
    into one before it sends them. In order of precedence: a `RetryFeedbackPart`, or a retried return of an output
    tool, makes it `output_retry`. A retried return of another tool makes it `tool_retry`. Any other tool return
    makes it `tool_results`, and anything else `user_prompt`. The retry is the highest `RetryInfo.attempt` among the
    parts that decided a retry kind, up to 100, and `0` for the other kinds.
    """
    output_retries: list[int | None] = []
    tool_retries: list[int | None] = []
    tool_results = False
    for part in _request_parts(messages):
        if isinstance(part, RetryFeedbackPart):
            output_retries.append(part.retry.attempt)
        elif isinstance(part, ToolReturnPart) and part.outcome == 'retried':
            attempt = part.retry.attempt if part.retry is not None else None
            (output_retries if part.tool_name in output_tool_names else tool_retries).append(attempt)
        elif isinstance(part, ToolReturnPart):
            tool_results = True
    if output_retries:
        return 'output_retry', _highest_attempt(output_retries)
    if tool_retries:
        return 'tool_retry', _highest_attempt(tool_retries)
    return ('tool_results' if tool_results else 'user_prompt'), 0


def _request_parts(messages: Sequence[ModelMessage]) -> list[ModelRequestPart]:
    """The parts of every `ModelRequest` after the last `ModelResponse`, in no particular order."""
    parts: list[ModelRequestPart] = []
    for message in reversed(messages):
        if isinstance(message, ModelResponse):
            break
        parts.extend(message.parts)
    return parts


def _highest_attempt(attempts: list[int | None]) -> int:
    return min(max(attempt or 0 for attempt in attempts), _MAX_RETRY)
