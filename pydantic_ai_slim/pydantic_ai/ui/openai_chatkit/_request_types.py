"""OpenAI ChatKit request types.

This module provides ChatKit-compatible request types. It attempts to import from the
official `chatkit` library.

For the complete type definitions, see:
https://github.com/openai/chatkit-python/blob/main/chatkit/types.py
"""

# Try to import from official library first
from chatkit.types import (
    Attachment,
    AttachmentBase,
    AttachmentCreateParams,
    AttachmentDeleteParams,
    AttachmentsCreateReq,
    AttachmentsDeleteReq,
    BaseReq,
    ChatKitReq,
    FeedbackKind,
    FileAttachment,
    ImageAttachment,
    InferenceOptions,
    ItemFeedbackParams,
    ItemsFeedbackReq,
    ItemsListParams,
    ItemsListReq,
    NonStreamingReq,
    StreamingReq,
    ThreadAddClientToolOutputParams,
    ThreadAddUserMessageParams,
    ThreadCreateParams,
    ThreadCustomActionParams,
    ThreadDeleteParams,
    ThreadGetByIdParams,
    ThreadItem,
    ThreadListParams,
    ThreadRetryAfterItemParams,
    ThreadsAddClientToolOutputReq,
    ThreadsAddUserMessageReq,
    ThreadsCreateReq,
    ThreadsCustomActionReq,
    ThreadsDeleteReq,
    ThreadsGetByIdReq,
    ThreadsListReq,
    ThreadsRetryAfterItemReq,
    ThreadsUpdateReq,
    ThreadUpdateParams,
    ToolChoice,
    UserMessageContent,
    UserMessageInput,
    UserMessageTagContent,
    UserMessageTextContent,
    is_streaming_req,
)
from pydantic import TypeAdapter

request_data_ta: TypeAdapter[ChatKitReq] = TypeAdapter(ChatKitReq)

__all__ = [
    # Base request types
    'BaseReq',
    'ChatKitReq',
    'StreamingReq',
    'NonStreamingReq',
    'is_streaming_req',
    # Specific request types
    'ThreadItem',
    'ThreadsCreateReq',
    'ThreadsGetByIdReq',
    'ThreadsListReq',
    'ThreadsAddUserMessageReq',
    'ThreadsAddClientToolOutputReq',
    'ThreadsCustomActionReq',
    'ThreadsRetryAfterItemReq',
    'ItemsFeedbackReq',
    'AttachmentsCreateReq',
    'AttachmentsDeleteReq',
    'ItemsListReq',
    'ThreadsUpdateReq',
    'ThreadsDeleteReq',
    # Parameter types
    'ThreadCreateParams',
    'ThreadGetByIdParams',
    'ThreadListParams',
    'ThreadAddUserMessageParams',
    'ThreadAddClientToolOutputParams',
    'ThreadCustomActionParams',
    'ThreadRetryAfterItemParams',
    'ItemFeedbackParams',
    'AttachmentCreateParams',
    'AttachmentDeleteParams',
    'ItemsListParams',
    'ThreadUpdateParams',
    'ThreadDeleteParams',
    # User message types
    'UserMessageInput',
    'UserMessageContent',
    'UserMessageTextContent',
    'UserMessageTagContent',
    # Tool and inference types
    'InferenceOptions',
    'ToolChoice',
    # Attachment types
    'Attachment',
    'FileAttachment',
    'ImageAttachment',
    'AttachmentBase',
    # Misc types
    'FeedbackKind',
    'request_data_ta',
]
