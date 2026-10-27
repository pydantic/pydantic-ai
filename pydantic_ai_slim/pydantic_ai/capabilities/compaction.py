"""The types the [`compact_history`][pydantic_ai.capabilities.AbstractCapability.compact_history] capability hook takes and returns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic_ai.messages import CompactionPart, CompactionReason, ModelMessage
from pydantic_ai.usage import RequestUsage

if TYPE_CHECKING:
    from pydantic_ai.models import Model, ModelRequestParameters
    from pydantic_ai.settings import ModelSettings

__all__ = ('CompactionRequest', 'CompactionResult')


@dataclass(kw_only=True)
class CompactionRequest:
    """What a [`compact_history`][pydantic_ai.capabilities.AbstractCapability.compact_history] hook is asked to compact, and for which model."""

    reason: CompactionReason
    """Why compaction was requested.

    The reason passed to [`RunContext.request_compaction`][pydantic_ai.tools.RunContext.request_compaction], or
    `'context_overflow'` when the model rejected the attempt for exceeding its context window.
    """

    model: Model
    """The model that will serve the request the compacted history is sent with.

    Under [`ModelFallback`][pydantic_ai.capabilities.ModelFallback] this is the candidate the attempt is for,
    not the step's model in [`RunContext.model`][pydantic_ai.tools.RunContext.model]. Under durable execution
    it's the model durability wrapped, so a request made through it runs as a durable model operation, like
    [`compact_messages`][pydantic_ai.models.Model.compact_messages].
    """

    messages: list[ModelMessage]
    """The run's history, ending with the request being answered, as this request shows it.

    The model request hooks' substitutions for this request are applied, like a file described for a model that can't
    read it, and what they only add for this request, like a reminder, is left out: see
    [request-only changes and compaction](../capabilities/custom.md#request-only-changes-and-compaction). The history
    gets the persistent originals of the messages you keep, and the request sent after the compaction gets this
    request's additions again. A [request-scoped][pydantic_ai.messages.SystemPromptPart.scope] prompt is here with its
    `scope`: keep it out of a summary as standing content, since it only applies to its own request.
    """

    model_settings: ModelSettings | None
    """The settings the request will be sent with.

    A compaction request made with the same settings and parameters as the conversation's own requests can read
    the provider's prompt cache.
    """

    model_request_parameters: ModelRequestParameters
    """The tools, output schema and other parameters the request will be sent with."""

    focus: str | None = None
    """What the compaction should keep, if the request said, like the argument to a `/compact` command."""


@dataclass(kw_only=True)
class CompactionResult:
    """The history a [`compact_history`][pydantic_ai.capabilities.AbstractCapability.compact_history] hook compacted."""

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

    kept_from: int | None = None
    """The index in `messages` where the messages kept as they were start.

    The messages from this index on have to be the last messages of
    [`CompactionRequest.messages`][pydantic_ai.capabilities.CompactionRequest.messages], unchanged; everything
    before it, `part` included, stands in for the rest of that history. A store that keeps the full conversation
    uses it to tell which of the original messages were summarized; see
    [`CompactionRecord`][pydantic_ai.messages.CompactionRecord].

    Set it to where your handler's kept tail starts: after the summary when you keep the latest turns as they
    were, or at the request being answered, the last message, when you keep nothing else. When it's `None`, core
    uses the longest tail after the message holding `part` that matches the end of the history it was given.
    """

    metadata: dict[str, Any] | None = None
    """Additional data about the compaction, reported on [`CompactionRecord.metadata`][pydantic_ai.messages.CompactionRecord.metadata].

    For example what a later request needs to know about the compaction, like the provider-side state it relies on.
    It's serialized with the run result, so keep it to JSON-compatible values.
    """
