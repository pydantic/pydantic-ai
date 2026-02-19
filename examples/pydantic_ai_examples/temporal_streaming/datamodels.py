"""Data models for the temporal streaming example."""

from enum import Enum

from pydantic import BaseModel


class AgentDependencies(BaseModel):
    """Dependencies passed to the agent containing workflow identification."""

    workflow_id: str
    run_id: str


class EventKind(str, Enum):
    """Types of events that can be streamed."""

    CONTINUE_CHAT = 'continue_chat'
    EVENT = 'event'
    RESULT = 'result'


class EventStream(BaseModel):
    """Event stream data model for streaming agent events."""

    kind: EventKind
    content: str
