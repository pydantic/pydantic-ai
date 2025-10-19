"""OpenAI ChatKit integration for Pydantic AI.

This module provides integration between Pydantic AI agents and OpenAI's ChatKit UI framework.
ChatKit is a framework for building conversational AI applications with rich UI components.

Key components:
- ChatKitAdapter: Main adapter for handling ChatKit requests
- ChatKitEventStream: Event stream for converting Pydantic AI events to ChatKit format
- Request/Response types: Official ChatKit types imported from the `chatkit` library

For more details, see the ChatKit documentation:
https://platform.openai.com/docs/guides/chatkit
"""

from ._adapter import ChatKitAdapter
from ._event_stream import ChatKitEventStream
from ._request_types import (
    Attachment,
    ChatKitReq,
    NonStreamingReq,
    StreamingReq,
    ThreadsAddUserMessageReq,
    ThreadsCreateReq,
    UserMessageInput,
    is_streaming_req,
)
from ._response_types import (
    AssistantMessageContent,
    AssistantMessageItem,
    ErrorEvent,
    ProgressUpdateEvent,
    Thread,
    ThreadCreatedEvent,
    ThreadItem,
    ThreadItemAddedEvent,
    ThreadItemDoneEvent,
    ThreadMetadata,
    ThreadStreamEvent,
    ThreadUpdatedEvent,
    UserMessageItem,
)

__all__ = [
    # Main classes
    'ChatKitAdapter',
    'ChatKitEventStream',
    # Request types
    'ChatKitReq',
    'StreamingReq',
    'NonStreamingReq',
    'ThreadsCreateReq',
    'ThreadsAddUserMessageReq',
    'UserMessageInput',
    'Attachment',
    'is_streaming_req',
    # Response types
    'ThreadStreamEvent',
    'ThreadCreatedEvent',
    'ThreadUpdatedEvent',
    'ThreadItemAddedEvent',
    'ThreadItemDoneEvent',
    'ProgressUpdateEvent',
    'ErrorEvent',
    'Thread',
    'ThreadMetadata',
    'ThreadItem',
    'AssistantMessageItem',
    'UserMessageItem',
    'AssistantMessageContent',
]
