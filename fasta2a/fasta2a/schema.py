"""This module contains the schema for the agent card."""

from __future__ import annotations as _annotations

from typing import Annotated, Any, Generic, Literal, TypeVar, Union

import pydantic
from pydantic import Discriminator, TypeAdapter
from pydantic.alias_generators import to_camel
from typing_extensions import NotRequired, TypeAlias, TypedDict


@pydantic.with_config(config={'alias_generator': to_camel})
class AgentCard(TypedDict):
    """The card that describes an agent."""

    name: str
    """Human readable name of the agent e.g. "Recipe Agent"."""

    description: NotRequired[str]
    """A human-readable description of the agent.

    Used to assist users and other agents in understanding what the agent can do.
    (e.g. "Agent that helps users with recipes and cooking.")
    """

    # TODO(Marcelo): The spec makes url required.
    url: NotRequired[str]
    """A URL to the address the agent is hosted at."""

    provider: NotRequired[Provider]
    """The service provider of the agent."""

    # TODO(Marcelo): The spec makes version required.
    version: NotRequired[str]
    """The version of the agent - format is up to the provider. (e.g. "1.0.0")"""

    documentation_url: NotRequired[str]
    """A URL to documentation for the agent."""

    capabilities: Capabilities
    """The capabilities of the agent."""

    authentication: Authentication
    """The authentication schemes supported by the agent.

    Intended to match OpenAPI authentication structure.
    """

    default_input_modes: list[str]
    """Supported mime types for input data."""

    default_output_modes: list[str]
    """Supported mime types for output data."""

    skills: list[Skill]


agent_card_ta = pydantic.TypeAdapter(AgentCard)


class Provider(TypedDict):
    """The service provider of the agent."""

    organization: str
    url: str


@pydantic.with_config(config={'alias_generator': to_camel})
class Capabilities(TypedDict):
    """The capabilities of the agent."""

    streaming: NotRequired[bool]
    """Whether the agent supports streaming."""

    push_notifications: NotRequired[bool]
    """Whether the agent can notify updates to client."""

    state_transition_history: NotRequired[bool]
    """Whether the agent exposes status change history for tasks."""


@pydantic.with_config(config={'alias_generator': to_camel})
class Authentication(TypedDict):
    """The authentication schemes supported by the agent."""

    schemes: list[str]
    """The authentication schemes supported by the agent. (e.g. "Basic", "Bearer")"""

    credentials: NotRequired[str]
    """The credentials a client should use for private cards."""


@pydantic.with_config(config={'alias_generator': to_camel})
class Skill(TypedDict):
    """Skills are a unit of capability that an agent can perform."""

    id: str
    """A unique identifier for the skill."""

    name: str
    """Human readable name of the skill."""

    description: str
    """A human-readable description of the skill.

    It will be used by the client or a human as a hint to understand the skill.
    """

    tags: list[str]
    """Set of tag-words describing classes of capabilities for this specific skill.

    Examples: "cooking", "customer support", "billing".
    """

    examples: NotRequired[list[str]]
    """The set of example scenarios that the skill can perform.

    Will be used by the client as a hint to understand how the skill can be used. (e.g. "I need a recipe for bread")
    """

    input_modes: list[str]
    """Supported mime types for input data."""

    output_modes: list[str]
    """Supported mime types for output data."""


@pydantic.with_config(config={'alias_generator': to_camel})
class Artifact(TypedDict):
    """Agents generate Artifacts as an end result of a Task.

    Artifacts are immutable, can be named, and can have multiple parts. A streaming response can append parts to
    existing Artifacts.

    A single Task can generate many Artifacts. For example, "create a webpage" could create separate HTML and image
    Artifacts.
    """

    name: NotRequired[str]
    """The name of the artifact."""

    description: NotRequired[str]
    """A description of the artifact."""

    parts: list[Part]
    """The parts that make up the artifact."""

    metadata: NotRequired[dict[str, Any]]
    """Metadata about the artifact."""

    index: int
    """The index of the artifact."""

    append: NotRequired[bool]
    """Whether to append this artifact to an existing one."""

    last_chunk: NotRequired[bool]
    """Whether this is the last chunk of the artifact."""


@pydantic.with_config(config={'alias_generator': to_camel})
class PushNotificationConfig(TypedDict):
    """Configuration for push notifications.

    A2A supports a secure notification mechanism whereby an agent can notify a client of an update
    outside of a connected session via a PushNotificationService. Within and across enterprises,
    it is critical that the agent verifies the identity of the notification service, authenticates
    itself with the service, and presents an identifier that ties the notification to the executing
    Task.

    The target server of the PushNotificationService should be considered a separate service, and
    is not guaranteed (or even expected) to be the client directly. This PushNotificationService is
    responsible for authenticating and authorizing the agent and for proxying the verified notification
    to the appropriate endpoint (which could be anything from a pub/sub queue, to an email inbox or
    other service, etc).

    For contrived scenarios with isolated client-agent pairs (e.g. local service mesh in a contained
    VPC, etc.) or isolated environments without enterprise security concerns, the client may choose to
    simply open a port and act as its own PushNotificationService. Any enterprise implementation will
    likely have a centralized service that authenticates the remote agents with trusted notification
    credentials and can handle online/offline scenarios. (This should be thought of similarly to a
    mobile Push Notification Service).
    """

    url: str
    """The URL to send push notifications to."""

    token: NotRequired[str]
    """Token unique to this task/session."""

    authentication: NotRequired[Authentication]
    """Authentication details for push notifications."""


@pydantic.with_config(config={'alias_generator': to_camel})
class TaskPushNotificationConfig(TypedDict):
    """Configuration for task push notifications."""

    id: str
    """The task id."""

    push_notification_config: PushNotificationConfig
    """The push notification configuration."""


class Message(TypedDict):
    """A Message contains any content that is not an Artifact.

    This can include things like agent thoughts, user context, instructions, errors, status, or metadata.

    All content from a client comes in the form of a Message. Agents send Messages to communicate status or to provide
    instructions (whereas generated results are sent as Artifacts).

    A Message can have multiple parts to denote different pieces of content. For example, a user request could include
    a textual description from a user and then multiple files used as context from the client.
    """

    role: Literal['user', 'agent']
    """The role of the message."""

    parts: list[Part]
    """The parts of the message."""

    metadata: NotRequired[dict[str, Any]]
    """Metadata about the message."""


class _BasePart(TypedDict):
    """A base class for all parts."""

    metadata: NotRequired[dict[str, Any]]


class TextPart(_BasePart):
    """A part that contains text."""

    type: Literal['text']
    """The type of the part."""

    text: str
    """The text of the part."""


@pydantic.with_config(config={'alias_generator': to_camel})
class FilePart(_BasePart):
    """A part that contains a file."""

    type: Literal['file']
    """The type of the part."""

    file: File
    """The file of the part."""


@pydantic.with_config(config={'alias_generator': to_camel})
class _BaseFile(_BasePart):
    """A base class for all file types."""

    name: NotRequired[str]
    """The name of the file."""

    mime_type: str
    """The mime type of the file."""


@pydantic.with_config(config={'alias_generator': to_camel})
class _BinaryFile(_BaseFile):
    """A binary file."""

    data: str
    """The base64 encoded bytes of the file."""


@pydantic.with_config(config={'alias_generator': to_camel})
class _URLFile(_BaseFile):
    """A file that is hosted on a remote URL."""

    url: str
    """The URL of the file."""


File: TypeAlias = Union[_BinaryFile, _URLFile]
"""A file is a binary file or a URL file."""


@pydantic.with_config(config={'alias_generator': to_camel})
class DataPart(_BasePart):
    """A part that contains data."""

    type: Literal['data']
    """The type of the part."""

    data: dict[str, Any]
    """The data of the part."""


Part = Annotated[Union[TextPart, FilePart, DataPart], pydantic.Field(discriminator='type')]
"""A fully formed piece of content exchanged between a client and a remote agent as part of a Message or an Artifact.

Each Part has its own content type and metadata.
"""

TaskState: TypeAlias = Literal['submitted', 'working', 'input-required', 'completed', 'canceled', 'failed', 'unknown']
"""The possible states of a task."""


@pydantic.with_config(config={'alias_generator': to_camel})
class TaskStatus(TypedDict):
    """Status and accompanying message for a task."""

    state: TaskState
    """The current state of the task."""

    message: NotRequired[Message]
    """Additional status updates for client."""

    timestamp: NotRequired[str]
    """ISO datetime value of when the status was updated."""


@pydantic.with_config(config={'alias_generator': to_camel})
class Task(TypedDict):
    """A Task is a stateful entity that allows Clients and Remote Agents to achieve a specific outcome.

    Clients and Remote Agents exchange Messages within a Task. Remote Agents generate results as Artifacts.
    A Task is always created by a Client and the status is always determined by the Remote Agent.
    """

    id: str
    """Unique identifier for the task."""

    session_id: NotRequired[str]
    """Client-generated id for the session holding the task."""

    status: TaskStatus
    """Current status of the task."""

    history: NotRequired[list[Message]]
    """Optional history of messages."""

    artifacts: NotRequired[list[Artifact]]
    """Collection of artifacts created by the agent."""

    metadata: NotRequired[dict[str, Any]]
    """Extension metadata."""


@pydantic.with_config(config={'alias_generator': to_camel})
class TaskStatusUpdateEvent(TypedDict):
    """Sent by server during sendSubscribe or subscribe requests."""

    id: str
    """The id of the task."""

    status: TaskStatus
    """The status of the task."""

    final: bool
    """Indicates the end of the event stream."""

    metadata: NotRequired[dict[str, Any]]
    """Extension metadata."""


@pydantic.with_config(config={'alias_generator': to_camel})
class TaskArtifactUpdateEvent(TypedDict):
    """Sent by server during sendSubscribe or subscribe requests."""

    id: str
    """The id of the task."""

    artifact: Artifact
    """The artifact that was updated."""

    metadata: NotRequired[dict[str, Any]]
    """Extension metadata."""


@pydantic.with_config(config={'alias_generator': to_camel})
class TaskIdParams(TypedDict):
    """Parameters for a task id."""

    id: str
    metadata: NotRequired[dict[str, Any]]


@pydantic.with_config(config={'alias_generator': to_camel})
class TaskQueryParams(TaskIdParams):
    """Query parameters for a task."""

    history_length: NotRequired[int]
    """Number of recent messages to be retrieved."""


@pydantic.with_config(config={'alias_generator': to_camel})
class TaskSendParams(TypedDict):
    """Sent by the client to the agent to create, continue, or restart a task."""

    id: str
    """The id of the task."""

    session_id: NotRequired[str]
    """The server creates a new sessionId for new tasks if not set."""

    message: Message
    """The message to send to the agent."""

    history_length: NotRequired[int]
    """Number of recent messages to be retrieved."""

    push_notification: NotRequired[PushNotificationConfig]
    """Where the server should send notifications when disconnected."""

    metadata: NotRequired[dict[str, Any]]
    """Extension metadata."""


class JSONRPCMessage(TypedDict):
    """A JSON RPC message."""

    jsonrpc: Literal['2.0']
    """The JSON RPC version."""

    id: int | str | None
    """The request id."""


Method = TypeVar('Method')
Params = TypeVar('Params')


class JSONRPCRequest(JSONRPCMessage, Generic[Method, Params]):
    """A JSON RPC request."""

    method: Method
    """The method to call."""

    params: Params
    """The parameters to pass to the method."""


###############################################################################################
#######################################   Error codes   #######################################
###############################################################################################

CodeT = TypeVar('CodeT', bound=int)
MessageT = TypeVar('MessageT', bound=str)


class JSONRPCError(TypedDict, Generic[CodeT, MessageT]):
    """A JSON RPC error."""

    code: CodeT
    message: MessageT
    data: NotRequired[Any]


ResultT = TypeVar('ResultT')
ErrorT = TypeVar('ErrorT', bound=JSONRPCError[Any, Any])


class JSONRPCResponse(JSONRPCMessage, Generic[ResultT, ErrorT]):
    """A JSON RPC response."""

    result: NotRequired[ResultT]
    error: NotRequired[ErrorT]


JSONParseError = JSONRPCError[Literal[-32700], Literal['Invalid JSON payload']]
"""A JSON RPC error for a parse error."""

InvalidRequestError = JSONRPCError[Literal[-32600], Literal['Request payload validation error']]
"""A JSON RPC error for an invalid request."""

MethodNotFoundError = JSONRPCError[Literal[-32601], Literal['Method not found']]
"""A JSON RPC error for a method not found."""

InvalidParamsError = JSONRPCError[Literal[-32602], Literal['Invalid parameters']]
"""A JSON RPC error for invalid parameters."""

InternalError = JSONRPCError[Literal[-32603], Literal['Internal error']]
"""A JSON RPC error for an internal error."""

TaskNotFoundError = JSONRPCError[Literal[-32001], Literal['Task not found']]
"""A JSON RPC error for a task not found."""

TaskNotCancelableError = JSONRPCError[Literal[-32002], Literal['Task not cancelable']]
"""A JSON RPC error for a task not cancelable."""

PushNotificationNotSupportedError = JSONRPCError[Literal[-32003], Literal['Push notification not supported']]
"""A JSON RPC error for a push notification not supported."""

UnsupportedOperationError = JSONRPCError[Literal[-32004], Literal['This operation is not supported']]
"""A JSON RPC error for an unsupported operation."""

ContentTypeNotSupportedError = JSONRPCError[Literal[-32005], Literal['Incompatible content types']]
"""A JSON RPC error for incompatible content types."""

###############################################################################################
#######################################   Requests and responses   ############################
###############################################################################################

SendTaskRequest = JSONRPCRequest[Literal['tasks/send'], TaskSendParams]
"""A JSON RPC request to send a task."""

SendTaskResponse = JSONRPCResponse[Task, JSONRPCError[Any, Any]]
"""A JSON RPC response to send a task."""

SendTaskStreamingRequest = JSONRPCRequest[Literal['tasks/sendSubscribe'], TaskSendParams]
"""A JSON RPC request to send a task and receive updates."""

SendTaskStreamingResponse = JSONRPCResponse[Union[TaskStatusUpdateEvent, TaskArtifactUpdateEvent], InternalError]
"""A JSON RPC response to send a task and receive updates."""

GetTaskRequest = JSONRPCRequest[Literal['tasks/get'], TaskQueryParams]
"""A JSON RPC request to get a task."""

GetTaskResponse = JSONRPCResponse[Task, TaskNotFoundError]
"""A JSON RPC response to get a task."""

CancelTaskRequest = JSONRPCRequest[Literal['tasks/cancel'], TaskIdParams]
"""A JSON RPC request to cancel a task."""

CancelTaskResponse = JSONRPCResponse[Task, Union[TaskNotCancelableError, TaskNotFoundError]]
"""A JSON RPC response to cancel a task."""

SetTaskPushNotificationRequest = JSONRPCRequest[Literal['tasks/pushNotification/set'], TaskPushNotificationConfig]
"""A JSON RPC request to set a task push notification."""

SetTaskPushNotificationResponse = JSONRPCResponse[TaskPushNotificationConfig, PushNotificationNotSupportedError]
"""A JSON RPC response to set a task push notification."""

GetTaskPushNotificationRequest = JSONRPCRequest[Literal['tasks/pushNotification/get'], TaskIdParams]
"""A JSON RPC request to get a task push notification."""

GetTaskPushNotificationResponse = JSONRPCResponse[TaskPushNotificationConfig, PushNotificationNotSupportedError]
"""A JSON RPC response to get a task push notification."""

ResubscribeTaskRequest = JSONRPCRequest[Literal['tasks/resubscribe'], TaskIdParams]
"""A JSON RPC request to resubscribe to a task."""

A2ARequest = Annotated[
    Union[
        SendTaskRequest,
        GetTaskRequest,
        CancelTaskRequest,
        SetTaskPushNotificationRequest,
        GetTaskPushNotificationRequest,
        ResubscribeTaskRequest,
    ],
    Discriminator('method'),
]
"""A JSON RPC request to the A2A server."""

A2AResponse: TypeAlias = Union[
    SendTaskResponse,
    GetTaskResponse,
    CancelTaskResponse,
    SetTaskPushNotificationResponse,
    GetTaskPushNotificationResponse,
]
"""A JSON RPC response from the A2A server."""


a2a_request_ta: TypeAdapter[A2ARequest] = TypeAdapter(A2ARequest)
a2a_response_ta: TypeAdapter[A2AResponse] = TypeAdapter(A2AResponse)
