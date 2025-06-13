"""
This module re-exports the core A2A schema types from the `google-a2a` SDK.
The types are based on Pydantic `BaseModel` and align with the official A2A JSON specification.
"""

from a2a.types import (
    AgentCard,
    AgentProvider,
    AgentSkill as Skill,
    Artifact,
    CancelTaskRequest,
    CancelTaskResponse,
    DataPart,
    FilePart,
    GetTaskRequest,
    GetTaskResponse,
    Message,
    MessageSendParams,
    Part,
    PushNotificationConfig,
    Role,
    SendMessageRequest,
    SendMessageResponse,
    Task,
    TaskState,
    TextPart,
)

__all__ = [
    # Core Models
    "AgentCard",
    "AgentProvider",
    "Artifact",
    "Message",
    "Part",
    "Skill",
    "Task",
    # Enums
    "Role",
    "TaskState",
    # Part Types
    "TextPart",
    "FilePart",
    "DataPart",
    # Request/Response models
    "SendMessageRequest",
    "SendMessageResponse",
    "GetTaskRequest",
    "GetTaskResponse",
    "CancelTaskRequest",
    "CancelTaskResponse",
    # Parameter Models
    "MessageSendParams",
    "PushNotificationConfig",
]
