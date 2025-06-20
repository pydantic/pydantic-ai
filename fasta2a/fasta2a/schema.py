"""This module contains the schema for the agent card."""

from __future__ import annotations as _annotations

from typing import Union

from a2a.types import (
    A2ARequest as _A2ARequest,
    A2AResponse as _A2AResponse,
    AgentCard,
    AgentProvider as Provider,
    AgentSkill as Skill,
    Artifact,
    AuthenticationInfo as Authentication,
    CancelTaskRequest,
    CancelTaskResponse,
    Capabilities,
    ContentTypeNotSupportedError,
    GetTaskPushNotificationConfigRequest as GetTaskPushNotificationRequest,
    GetTaskPushNotificationConfigResponse as GetTaskPushNotificationResponse,
    GetTaskRequest,
    GetTaskResponse,
    InternalError,
    JSONRPCError,
    Message,
    MessageSendParams as TaskSendParams,
    Part,
    PushNotificationConfig,
    PushNotificationNotSupportedError,
    SendMessageRequest as SendTaskRequest,
    SendMessageResponse as SendTaskResponse,
    SendStreamingMessageRequest as SendTaskStreamingRequest,
    SendStreamingMessageResponse as SendTaskStreamingResponse,
    SetTaskPushNotificationConfigRequest as SetTaskPushNotificationRequest,
    SetTaskPushNotificationConfigResponse as SetTaskPushNotificationResponse,
    Task,
    TaskArtifactUpdateEvent,
    TaskIdParams,
    TaskNotCancelableError,
    TaskNotFoundError,
    TaskPushNotificationConfig,
    TaskQueryParams,
    TaskResubscriptionRequest as ResubscribeTaskRequest,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
    UnsupportedOperationError,
)
from pydantic import TypeAdapter

__all__ = [
    "AgentCard",
    "Provider",
    "Skill",
    "Artifact",
    "Authentication",
    "Capabilities",
    "Message",
    "TaskSendParams",
    "Part",
    "PushNotificationConfig",
    "Task",
    "TaskIdParams",
    "TaskQueryParams",
    "TextPart",
    "CancelTaskRequest",
    "CancelTaskResponse",
    "GetTaskPushNotificationRequest",
    "GetTaskPushNotificationResponse",
    "GetTaskRequest",
    "GetTaskResponse",
    "SendTaskStreamingRequest",
    "SendTaskStreamingResponse",
    "SendTaskRequest",
    "SendTaskResponse",
    "SetTaskPushNotificationRequest",
    "SetTaskPushNotificationResponse",
    "ResubscribeTaskRequest",
    "TaskState",
    "TaskStatus",
    "TaskArtifactUpdateEvent",
    "TaskPushNotificationConfig",
    "TaskStatusUpdateEvent",
    "JSONRPCError",
    "TaskNotFoundError",
    "TaskNotCancelableError",
    "PushNotificationNotSupportedError",
    "UnsupportedOperationError",
    "ContentTypeNotSupportedError",
    "InternalError",
    "a2a_request_ta",
    "a2a_response_ta",
    "A2ARequest",
    "A2AResponse",
]


A2ARequest = _A2ARequest
"""A JSON RPC request to the A2A server."""

A2AResponse = Union[
    SendTaskResponse,
    GetTaskResponse,
    CancelTaskResponse,
    SetTaskPushNotificationResponse,
    GetTaskPushNotificationResponse,
]
"""A JSON RPC response from the A2A server."""

a2a_request_ta: TypeAdapter[A2ARequest] = TypeAdapter(A2ARequest)
a2a_response_ta: TypeAdapter[A2AResponse] = TypeAdapter(A2AResponse)
