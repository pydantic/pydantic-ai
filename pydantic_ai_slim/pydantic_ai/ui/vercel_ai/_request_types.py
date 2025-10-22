"""Vercel AI request types (UI messages).

Converted to Python from:
https://github.com/vercel/ai/blob/ai%405.0.59/packages/ai/src/ui/ui-messages.ts
"""

from typing import Annotated, Any, Literal

from pydantic import Discriminator, Field, TypeAdapter

from ._utils import CamelBaseModel, ProviderMetadata


class TextUIPart(CamelBaseModel):
    """A text part of a message."""

    type: Literal['text'] = 'text'

    text: str
    """The text content."""

    state: Literal['streaming', 'done'] | None = None
    """The state of the text part."""

    provider_metadata: ProviderMetadata | None = None
    """The provider metadata."""


class ReasoningUIPart(CamelBaseModel):
    """A reasoning part of a message."""

    type: Literal['reasoning'] = 'reasoning'

    text: str
    """The reasoning text."""

    state: Literal['streaming', 'done'] | None = None
    """The state of the reasoning part."""

    provider_metadata: ProviderMetadata | None = None
    """The provider metadata."""


class SourceUrlUIPart(CamelBaseModel):
    """A source part of a message."""

    type: Literal['source-url'] = 'source-url'
    source_id: str
    url: str
    title: str | None = None
    provider_metadata: ProviderMetadata | None = None


class SourceDocumentUIPart(CamelBaseModel):
    """A document source part of a message."""

    type: Literal['source-document'] = 'source-document'
    source_id: str
    media_type: str
    title: str
    filename: str | None = None
    provider_metadata: ProviderMetadata | None = None


class FileUIPart(CamelBaseModel):
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


class StepStartUIPart(CamelBaseModel):
    """A step boundary part of a message."""

    type: Literal['step-start'] = 'step-start'


class DataUIPart(CamelBaseModel):
    """Data part with dynamic type based on data name."""

    type: Annotated[str, Field(pattern=r'^data-')]
    id: str | None = None
    data: Any


# Tool part states as separate models
class ToolInputStreamingPart(CamelBaseModel):
    """Tool part in input-streaming state."""

    type: Annotated[str, Field(pattern=r'^tool-')]
    tool_call_id: str
    state: Literal['input-streaming'] = 'input-streaming'
    input: Any | None = None
    provider_executed: bool | None = None


class ToolInputAvailablePart(CamelBaseModel):
    """Tool part in input-available state."""

    type: Annotated[str, Field(pattern=r'^tool-')]
    tool_call_id: str
    state: Literal['input-available'] = 'input-available'
    input: Any | None = None
    provider_executed: bool | None = None
    call_provider_metadata: ProviderMetadata | None = None


class ToolOutputAvailablePart(CamelBaseModel):
    """Tool part in output-available state."""

    type: Annotated[str, Field(pattern=r'^tool-')]
    tool_call_id: str
    state: Literal['output-available'] = 'output-available'
    input: Any | None = None
    output: Any | None = None
    provider_executed: bool | None = None
    call_provider_metadata: ProviderMetadata | None = None
    preliminary: bool | None = None


class ToolOutputErrorPart(CamelBaseModel):
    """Tool part in output-error state."""

    type: Annotated[str, Field(pattern=r'^tool-')]
    tool_call_id: str
    state: Literal['output-error'] = 'output-error'
    input: Any | None = None
    raw_input: Any | None = None
    error_text: str
    provider_executed: bool | None = None
    call_provider_metadata: ProviderMetadata | None = None


# Union of all tool part states
ToolUIPart = ToolInputStreamingPart | ToolInputAvailablePart | ToolOutputAvailablePart | ToolOutputErrorPart


# Dynamic tool part states as separate models
class DynamicToolInputStreamingPart(CamelBaseModel):
    """Dynamic tool part in input-streaming state."""

    type: Literal['dynamic-tool'] = 'dynamic-tool'
    tool_name: str
    tool_call_id: str
    state: Literal['input-streaming'] = 'input-streaming'
    input: Any | None = None


class DynamicToolInputAvailablePart(CamelBaseModel):
    """Dynamic tool part in input-available state."""

    type: Literal['dynamic-tool'] = 'dynamic-tool'
    tool_name: str
    tool_call_id: str
    state: Literal['input-available'] = 'input-available'
    input: Any
    call_provider_metadata: ProviderMetadata | None = None


class DynamicToolOutputAvailablePart(CamelBaseModel):
    """Dynamic tool part in output-available state."""

    type: Literal['dynamic-tool'] = 'dynamic-tool'
    tool_name: str
    tool_call_id: str
    state: Literal['output-available'] = 'output-available'
    input: Any
    output: Any
    call_provider_metadata: ProviderMetadata | None = None
    preliminary: bool | None = None


class DynamicToolOutputErrorPart(CamelBaseModel):
    """Dynamic tool part in output-error state."""

    type: Literal['dynamic-tool'] = 'dynamic-tool'
    tool_name: str
    tool_call_id: str
    state: Literal['output-error'] = 'output-error'
    input: Any
    error_text: str
    call_provider_metadata: ProviderMetadata | None = None


# Union of all dynamic tool part states
DynamicToolUIPart = (
    DynamicToolInputStreamingPart
    | DynamicToolInputAvailablePart
    | DynamicToolOutputAvailablePart
    | DynamicToolOutputErrorPart
)


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

# Type adapter for parsing requests
request_data_ta: TypeAdapter[RequestData] = TypeAdapter(RequestData)
