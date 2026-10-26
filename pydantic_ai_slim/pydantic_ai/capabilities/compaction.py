"""The types the [`compact`][pydantic_ai.capabilities.AbstractCapability.compact] capability hook takes and returns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic_ai.messages import CompactionPart, CompactionReason, ModelMessage
from pydantic_ai.usage import RequestUsage

if TYPE_CHECKING:
    from pydantic_ai.models import Model, ModelRequestParameters
    from pydantic_ai.settings import ModelSettings

__all__ = ('CompactionRequest', 'CompactionResult')


@dataclass(kw_only=True)
class CompactionRequest:
    """What a [`compact`][pydantic_ai.capabilities.AbstractCapability.compact] hook is asked to compact, and for which model."""

    reason: CompactionReason
    """Why compaction was requested, as passed to [`RunContext.request_compaction`][pydantic_ai.tools.RunContext.request_compaction]."""

    model: Model
    """The model that will serve the request the compacted history is sent with.

    Under [`ModelFallback`][pydantic_ai.capabilities.ModelFallback] this is the candidate the attempt is for,
    not the step's model in [`RunContext.model`][pydantic_ai.tools.RunContext.model]. Under durable execution
    it's the model durability wrapped, so a request made through it runs as a durable model operation, like
    [`compact_messages`][pydantic_ai.models.Model.compact_messages].
    """

    messages: list[ModelMessage]
    """The history the request would be sent with, ending with the request being answered.

    It's the attempt's history, as the `prepare_model_request` hooks left it.
    """

    focus: str | None = None
    """What the compaction should keep, if the request said, like the argument to a `/compact` command."""

    model_settings: ModelSettings | None = None
    """The settings the request will be sent with.

    A compaction request made with the same settings and parameters as the conversation's own requests can read
    the provider's prompt cache.
    """

    model_request_parameters: ModelRequestParameters | None = None
    """The tools, output schema and other parameters the request will be sent with."""


@dataclass(kw_only=True)
class CompactionResult:
    """The history a [`compact`][pydantic_ai.capabilities.AbstractCapability.compact] hook compacted."""

    messages: list[ModelMessage]
    """The history that replaces [`CompactionRequest.messages`][pydantic_ai.capabilities.CompactionRequest.messages].

    It has to contain `part` and end with the request being answered, and it can keep any other messages, like
    the most recent turns or, for OpenAI, the user messages the provider kept next to its compaction item.
    """

    part: CompactionPart
    """The part in `messages` that marks where the compacted history starts.

    A provider's compaction part goes in the [`ModelResponse`][pydantic_ai.messages.ModelResponse] it arrived
    in, and a summary you wrote in a [`ModelRequest`][pydantic_ai.messages.ModelRequest], without `provider_name`.
    It needs a summary in `content`, or a provider's encrypted content, so the history it replaces isn't lost.
    """

    handler: str
    """A stable identifier for what compacted, like `'openai'` or `'summarizing'`.

    It's reported on [`CompactionEndEvent.handler`][pydantic_ai.messages.CompactionEndEvent.handler]. Pick an
    identifier that is safe to persist and branch on, never a Python class name.
    """

    usage: RequestUsage = field(default_factory=RequestUsage)
    """What the compaction cost, for compaction that made model requests.

    It's counted in the run's usage, and checked against its usage limits. Without a `cost`, it's priced as a
    request to [`CompactionRequest.model`][pydantic_ai.capabilities.CompactionRequest.model].
    """
