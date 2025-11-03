"""Vercel AI request types (UI messages).

Converted to Python from:
https://github.com/vercel/ai/blob/ai%405.0.59/packages/ai/src/ui/ui-messages.ts
"""

from abc import ABC
from typing import Annotated, Any, Literal

from pydantic import Discriminator, Field

from ._utils import CamelBaseModel

# Technically this is recursive union of JSON types; for simplicity, we call it Any
JSONValue = Any
ProviderMetadata = dict[str, dict[str, JSONValue]]
"""Provider metadata."""


class BaseUIPart(CamelBaseModel, ABC):
    """Abstract base class for all UI parts."""


class TextUIPart(BaseUIPart):
    """A text part of a message."""

    type: Literal['text'] = 'text'

    text: str
    """The text content."""

    state: Literal['streaming', 'done'] | None = None
    """The state of the text part."""

    provider_metadata: ProviderMetadata | None = None
    """The provider metadata."""


class ReasoningUIPart(BaseUIPart):
    """A reasoning part of a message."""

    type: Literal['reasoning'] = 'reasoning'

    text: str
    """The reasoning text."""

    state: Literal['streaming', 'done'] | None = None
    """The state of the reasoning part."""

    provider_metadata: ProviderMetadata | None = None
    """The provider metadata."""


class SourceUrlUIPart(BaseUIPart):
    """A source part of a message."""

    type: Literal['source-url'] = 'source-url'
    source_id: str
    url: str
    title: str | None = None
    provider_metadata: ProviderMetadata | None = None


class SourceDocumentUIPart(BaseUIPart):
    """A document source part of a message."""

    type: Literal['source-document'] = 'source-document'
    source_id: str
    media_type: str
    title: str
    filename: str | None = None
    provider_metadata: ProviderMetadata | None = None


class FileUIPart(BaseUIPart):
    """A file part of a message."""

    type: Literal['file'] = 'file'

    media_type: str
    """
    IANA media type of the file.
    @see https://www.iana.org/assignments/media-types/media-types.xhtml
    """

    filename: str | None = None
    """Optional filename of the file."""

    url: str
    """
    The URL of the file.
    It can either be a URL to a hosted file or a [Data URL](https://developer.mozilla.org/en-US/docs/Web/HTTP/Basics_of_HTTP/Data_URLs).
    """

    provider_metadata: ProviderMetadata | None = None
    """The provider metadata."""


class StepStartUIPart(BaseUIPart):
    """A step boundary part of a message."""

    type: Literal['step-start'] = 'step-start'


class DataUIPart(BaseUIPart):
    """Data part with dynamic type based on data name."""

    type: Annotated[str, Field(pattern=r'^data-')]
    id: str | None = None
    data: Any


# Tool part states as separate models
class ToolInputStreamingPart(BaseUIPart):
    """Tool part in input-streaming state."""

    type: Annotated[str, Field(pattern=r'^tool-')]
    tool_call_id: str
    state: Literal['input-streaming'] = 'input-streaming'
    input: Any | None = None
    provider_executed: bool | None = None


class ToolInputAvailablePart(BaseUIPart):
    """Tool part in input-available state."""

    type: Annotated[str, Field(pattern=r'^tool-')]
    tool_call_id: str
    state: Literal['input-available'] = 'input-available'
    input: Any | None = None
    provider_executed: bool | None = None
    call_provider_metadata: ProviderMetadata | None = None


class ToolOutputAvailablePart(BaseUIPart):
    """Tool part in output-available state."""

    type: Annotated[str, Field(pattern=r'^tool-')]
    tool_call_id: str
    state: Literal['output-available'] = 'output-available'
    input: Any | None = None
    output: Any | None = None
    provider_executed: bool | None = None
    call_provider_metadata: ProviderMetadata | None = None
    preliminary: bool | None = None


class ToolOutputErrorPart(BaseUIPart):
    """Tool part in output-error state."""

    type: Annotated[str, Field(pattern=r'^tool-')]
    tool_call_id: str
    state: Literal['output-error'] = 'output-error'
    input: Any | None = None
    raw_input: Any | None = None
    error_text: str
    provider_executed: bool | None = None
    call_provider_metadata: ProviderMetadata | None = None


ToolUIPart = ToolInputStreamingPart | ToolInputAvailablePart | ToolOutputAvailablePart | ToolOutputErrorPart
"""Union of all tool part types."""


# Dynamic tool part states as separate models
class DynamicToolInputStreamingPart(BaseUIPart):
    """Dynamic tool part in input-streaming state."""

    type: Literal['dynamic-tool'] = 'dynamic-tool'
    tool_name: str
    tool_call_id: str
    state: Literal['input-streaming'] = 'input-streaming'
    input: Any | None = None


class DynamicToolInputAvailablePart(BaseUIPart):
    """Dynamic tool part in input-available state."""

    type: Literal['dynamic-tool'] = 'dynamic-tool'
    tool_name: str
    tool_call_id: str
    state: Literal['input-available'] = 'input-available'
    input: Any
    call_provider_metadata: ProviderMetadata | None = None


class DynamicToolOutputAvailablePart(BaseUIPart):
    """Dynamic tool part in output-available state."""

    type: Literal['dynamic-tool'] = 'dynamic-tool'
    tool_name: str
    tool_call_id: str
    state: Literal['output-available'] = 'output-available'
    input: Any
    output: Any
    call_provider_metadata: ProviderMetadata | None = None
    preliminary: bool | None = None


class DynamicToolOutputErrorPart(BaseUIPart):
    """Dynamic tool part in output-error state."""

    type: Literal['dynamic-tool'] = 'dynamic-tool'
    tool_name: str
    tool_call_id: str
    state: Literal['output-error'] = 'output-error'
    input: Any
    error_text: str
    call_provider_metadata: ProviderMetadata | None = None


DynamicToolUIPart = (
    DynamicToolInputStreamingPart
    | DynamicToolInputAvailablePart
    | DynamicToolOutputAvailablePart
    | DynamicToolOutputErrorPart
)
"""Union of all dynamic tool part types."""


UIMessagePart = (
    TextUIPart
    | ReasoningUIPart
    | ToolUIPart
    | DynamicToolUIPart
    | SourceUrlUIPart
    | SourceDocumentUIPart
    | FileUIPart
    | DataUIPart
    | StepStartUIPart
)
"""Union of all message part types."""


class UIMessage(CamelBaseModel):
    """A message as displayed in the UI by Vercel AI Elements."""

    id: str
    """A unique identifier for the message."""

    role: Literal['system', 'user', 'assistant']
    """The role of the message."""

    metadata: Any | None = None
    """The metadata of the message."""

    parts: list[UIMessagePart]
    """
    The parts of the message. Use this for rendering the message in the UI.
    System messages should be avoided (set the system prompt on the server instead).
    They can have text parts.
    User messages can have text parts and file parts.
    Assistant messages can have text, reasoning, tool invocation, and file parts.
    """


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
"""Union of all request data types."""
