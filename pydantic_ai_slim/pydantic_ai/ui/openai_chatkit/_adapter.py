"""OpenAI ChatKit adapter for handling requests.

This adapter integrates Pydantic AI agents with OpenAI's ChatKit UI framework.

1. Thread-based conversations: ChatKit works with persistent threads that contain message history.

2. Single endpoint: All communication happens through one POST endpoint that returns either JSON directly or streams SSE JSON events.

3. Rich UI components: Supports widgets, progress updates, and client-side tools beyond just text and tool calls.

4. Multiple formats: Supports both original ChatKit format and newer threads.create format.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any

from ...agent import AgentDepsT
from ...messages import (
    AudioUrl,
    DocumentUrl,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    UserPromptPart,
    VideoUrl,
)
from ...output import OutputDataT
from ..adapter import BaseAdapter
from ..event_stream import BaseEventStream
from ._event_stream import ChatKitEventStream

# Import ChatKit types from the proper modules
from ._request_types import (
    ChatKitReq,
    FileAttachment,
    ImageAttachment,
    ThreadItem,
    UserMessageTagContent,
    UserMessageTextContent,
    request_data_ta,
)
from ._response_types import (
    ThreadStreamEvent,
    UserMessageItem,
)

if TYPE_CHECKING:
    try:
        from starlette.requests import Request
    except ImportError:
        pass

__all__ = ['ChatKitAdapter']


@dataclass
class ChatKitAdapter(BaseAdapter[ChatKitReq, UserMessageItem, ThreadStreamEvent, AgentDepsT, OutputDataT]):
    """ChatKit adapter for integrating Pydantic AI agents with ChatKit UI.

    This adapter handles the translation between ChatKit's thread-based protocol
    and Pydantic AI's message-based system.
    """

    @classmethod
    async def validate_request(cls, request: Request) -> ChatKitReq:
        """Validate a ChatKit request, supporting multiple formats."""
        return request_data_ta.validate_json(await request.body())

    @classmethod
    def load_messages(cls, messages: Sequence[ThreadItem]) -> list[ModelMessage]:
        """Convert ChatKit UserMessageItem objects to Pydantic AI ModelMessage format."""
        result: list[ModelMessage] = []
        request_parts: list[ModelRequestPart] | None = None

        for item in messages:
            # User Messages
            if hasattr(item, 'type') and item.type == 'user_message':
                if request_parts is None:
                    request_parts = []
                    result.append(ModelRequest(parts=request_parts))

                # Process content parts
                for content in item.content:
                    if isinstance(content, UserMessageTextContent):
                        request_parts.append(UserPromptPart(content=content.text))
                    elif isinstance(content, UserMessageTagContent):
                        # For tag content, we'll treat it as text with the tag text
                        request_parts.append(UserPromptPart(content=content.text))

                # Process attachments
                for attachment in item.attachments:
                    if isinstance(attachment, FileAttachment):
                        if attachment.upload_url:
                            # Determine content type based on mime_type
                            media_type_prefix = attachment.mime_type.split('/', 1)[0]
                            match media_type_prefix:
                                case 'image':
                                    file = ImageUrl(url=str(attachment.upload_url), media_type=attachment.mime_type)
                                case 'video':
                                    file = VideoUrl(url=str(attachment.upload_url), media_type=attachment.mime_type)
                                case 'audio':
                                    file = AudioUrl(url=str(attachment.upload_url), media_type=attachment.mime_type)
                                case _:
                                    file = DocumentUrl(url=str(attachment.upload_url), media_type=attachment.mime_type)
                            request_parts.append(UserPromptPart(content=[file]))
                    elif isinstance(attachment, ImageAttachment):
                        if attachment.upload_url:
                            # Use the upload URL for the image
                            file = ImageUrl(url=str(attachment.upload_url), media_type=attachment.mime_type)
                            request_parts.append(UserPromptPart(content=[file]))

        return result

    def dump_messages(self, messages: Sequence[ModelMessage]) -> list[UserMessageItem]:
        """Convert Pydantic AI ModelMessage objects to ChatKit UserMessageItem format."""
        raise NotImplementedError

    @property
    def event_stream(self) -> BaseEventStream[ChatKitReq, ThreadStreamEvent, AgentDepsT, OutputDataT]:
        """Create the event stream handler for this adapter."""
        return ChatKitEventStream(self.request)

    @cached_property
    def messages(self) -> list[ModelMessage]:
        """Convert the current request's user message to Pydantic AI format."""
        raise NotImplementedError

    @cached_property
    def state(self) -> dict[str, Any] | None:
        """Extract state from the ChatKit thread metadata."""
        pass

    @property
    def response_headers(self) -> Mapping[str, str] | None:
        """Get HTTP response headers for ChatKit compatibility."""
        pass
