"""Vercel AI request types (UI messages).

Converted to Python from:
https://github.com/vercel/ai/blob/ai%405.0.34/packages/ai/src/ui/ui-messages.ts
"""

from typing import Annotated, Any, Literal

from pydantic import Discriminator, TypeAdapter

from ._utils import CamelBaseModel, ProviderMetadata

__all__ = [
    'TextUIPart',
    'ToolOutputAvailablePart',
    'UIPart',
    'UIMessage',
    'SubmitMessage',
    'RequestData',
    'request_data_ta',
]


class TextUIPart(CamelBaseModel):
    """A text part of a message."""

    type: Literal['text'] = 'text'
    text: str
    state: Literal['streaming', 'done'] | None = None
    provider_metadata: ProviderMetadata | None = None


class ToolOutputAvailablePart(CamelBaseModel):
    """Tool output available part."""

    type: str  # f"tool-{tool_name}"
    tool_call_id: str
    state: Literal['output-available'] = 'output-available'
    input: Any
    output: Any
    provider_executed: bool | None = None
    call_provider_metadata: ProviderMetadata | None = None
    preliminary: bool | None = None


# Since ToolOutputAvailablePart has a dynamic type field, we can't use Discriminator
UIPart = TextUIPart | ToolOutputAvailablePart


class UIMessage(CamelBaseModel):
    """A message in the UI protocol."""

    id: str
    role: Literal['user', 'assistant', 'system']
    metadata: dict[str, Any] | None = None
    parts: list[UIPart]


class SubmitMessage(CamelBaseModel, extra='allow'):
    """Submit message request."""

    trigger: Literal['submit-message'] = 'submit-message'
    id: str
    messages: list[UIMessage]


class RegenerateMessage(CamelBaseModel, extra='allow'):
    """Ask the agent to regenerate a message."""

    trigger: Literal['regenerate-message']
    id: str
    messages: list[UIMessage]
    message_id: str


RequestData = Annotated[SubmitMessage | RegenerateMessage, Discriminator('trigger')]

# Type adapter for parsing requests
request_data_ta: TypeAdapter[RequestData] = TypeAdapter(RequestData)
