"""This module contains the schema for the agent card."""

from __future__ import annotations as _annotations

from typing import Annotated, Any, Generic, Literal, TypeVar, Union

import pydantic
from pydantic import Discriminator, TypeAdapter
from pydantic.alias_generators import to_camel
from typing_extensions import NotRequired, TypeAlias, TypedDict


@pydantic.with_config({'alias_generator': to_camel})
class AgentCard(TypedDict):
    """The card that describes an agent."""

    name: str
    """Human readable name of the agent e.g. "Recipe Agent"."""

    description: str
    """A human-readable description of the agent.

    Used to assist users and other agents in understanding what the agent can do.
    (e.g. "Agent that helps users with recipes and cooking.")
    """

    url: str
    """A URL to the address the agent is hosted at."""

    version: str
    """The version of the agent - format is up to the provider. (e.g. "1.0.0")"""

    protocol_version: str
    """The version of the A2A protocol this agent supports."""

    provider: NotRequired[AgentProvider]
    """The service provider of the agent."""

    documentation_url: NotRequired[str]
    """A URL to documentation for the agent."""

    icon_url: NotRequired[str]
    """A URL to an icon for the agent."""

    preferred_transport: NotRequired[str]
    """The transport of the preferred endpoint. If empty, defaults to JSONRPC."""

    additional_interfaces: NotRequired[list[AgentInterface]]
    """Announcement of additional supported transports."""

    capabilities: AgentCapabilities
    """The capabilities of the agent."""

    security: NotRequired[list[dict[str, list[str]]]]
    """Security requirements for contacting the agent."""

    security_schemes: NotRequired[dict[str, SecurityScheme]]
    """Security scheme definitions."""

    default_input_modes: list[str]
    """Supported mime types for input data."""

    default_output_modes: list[str]
    """Supported mime types for output data."""

    skills: list[Skill]


agent_card_ta = pydantic.TypeAdapter(AgentCard)


class AgentProvider(TypedDict):
    """The service provider of the agent."""

    organization: str
    url: str


@pydantic.with_config({'alias_generator': to_camel})
class AgentCapabilities(TypedDict):
    """The capabilities of the agent."""

    streaming: NotRequired[bool]
    """Whether the agent supports streaming."""

    push_notifications: NotRequired[bool]
    """Whether the agent can notify updates to client."""

    state_transition_history: NotRequired[bool]
    """Whether the agent exposes status change history for tasks."""


@pydantic.with_config({'alias_generator': to_camel})
class HttpSecurityScheme(TypedDict):
    """HTTP security scheme."""

    type: Literal['http']
    scheme: str
    """The name of the HTTP Authorization scheme."""
    bearer_format: NotRequired[str]
    """A hint to the client to identify how the bearer token is formatted."""
    description: NotRequired[str]
    """Description of this security scheme."""


@pydantic.with_config({'alias_generator': to_camel})
class ApiKeySecurityScheme(TypedDict):
    """API Key security scheme."""

    type: Literal['apiKey']
    name: str
    """The name of the header, query or cookie parameter to be used."""
    in_: Literal['query', 'header', 'cookie']
    """The location of the API key."""
    description: NotRequired[str]
    """Description of this security scheme."""


@pydantic.with_config({'alias_generator': to_camel})
class OAuth2SecurityScheme(TypedDict):
    """OAuth2 security scheme."""

    type: Literal['oauth2']
    flows: dict[str, Any]
    """An object containing configuration information for the flow types supported."""
    description: NotRequired[str]
    """Description of this security scheme."""


@pydantic.with_config({'alias_generator': to_camel})
class OpenIdConnectSecurityScheme(TypedDict):
    """OpenID Connect security scheme."""

    type: Literal['openIdConnect']
    open_id_connect_url: str
    """OpenId Connect URL to discover OAuth2 configuration values."""
    description: NotRequired[str]
    """Description of this security scheme."""


SecurityScheme = Annotated[
    Union[HttpSecurityScheme, ApiKeySecurityScheme, OAuth2SecurityScheme, OpenIdConnectSecurityScheme],
    pydantic.Field(discriminator='type'),
]
"""A security scheme for authentication."""


@pydantic.with_config({'alias_generator': to_camel})
class AgentInterface(TypedDict):
    """An interface that the agent supports."""

    transport: str
    """The transport protocol (e.g., 'jsonrpc', 'websocket')."""

    url: str
    """The URL endpoint for this transport."""

    description: NotRequired[str]
    """Description of this interface."""


@pydantic.with_config({'alias_generator': to_camel})
class AgentExtension(TypedDict):
    """A declaration of an extension supported by an Agent."""

    uri: str
    """The URI of the extension."""

    description: NotRequired[str]
    """A description of how this agent uses this extension."""

    required: NotRequired[bool]
    """Whether the client must follow specific requirements of the extension."""

    params: NotRequired[dict[str, Any]]
    """Optional configuration for the extension."""


@pydantic.with_config({'alias_generator': to_camel})
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


@pydantic.with_config({'alias_generator': to_camel})
class Artifact(TypedDict):
    """Agents generate Artifacts as an end result of a Task.

    Artifacts are immutable, can be named, and can have multiple parts. A streaming response can append parts to
    existing Artifacts.

    A single Task can generate many Artifacts. For example, "create a webpage" could create separate HTML and image
    Artifacts.
    """

    artifact_id: str
    """Unique identifier for the artifact."""

    name: NotRequired[str]
    """The name of the artifact."""

    description: NotRequired[str]
    """A description of the artifact."""

    parts: list[Part]
    """The parts that make up the artifact."""

    metadata: NotRequired[dict[str, Any]]
    """Metadata about the artifact."""

    extensions: NotRequired[list[str]]
    """Array of extensions."""

    append: NotRequired[bool]
    """Whether to append this artifact to an existing one."""

    last_chunk: NotRequired[bool]
    """Whether this is the last chunk of the artifact."""


@pydantic.with_config({'alias_generator': to_camel})
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

    id: NotRequired[str]
    """Server-assigned identifier."""

    url: str
    """The URL to send push notifications to."""

    token: NotRequired[str]
    """Token unique to this task/session."""

    authentication: NotRequired[SecurityScheme]
    """Authentication details for push notifications."""


@pydantic.with_config({'alias_generator': to_camel})
class TaskPushNotificationConfig(TypedDict):
    """Configuration for task push notifications."""

    id: str
    """The task id."""

    push_notification_config: PushNotificationConfig
    """The push notification configuration."""


@pydantic.with_config({'alias_generator': to_camel})
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

    kind: Literal['message']
    """Event type."""

    metadata: NotRequired[dict[str, Any]]
    """Metadata about the message."""

    # Additional fields
    message_id: str
    """Identifier created by the message creator."""

    context_id: NotRequired[str]
    """The context the message is associated with."""

    task_id: NotRequired[str]
    """Identifier of task the message is related to."""

    reference_task_ids: NotRequired[list[str]]
    """Array of task IDs this message references."""

    extensions: NotRequired[list[str]]
    """Array of extensions."""


class _BasePart(TypedDict):
    """A base class for all parts."""

    metadata: NotRequired[dict[str, Any]]


@pydantic.with_config({'alias_generator': to_camel})
class TextPart(_BasePart):
    """A part that contains text."""

    kind: Literal['text']
    """The kind of the part."""

    text: str
    """The text of the part."""


@pydantic.with_config({'alias_generator': to_camel})
class FileWithBytes(TypedDict):
    """File with base64 encoded data."""

    bytes: str
    """The base64 encoded content of the file."""

    mime_type: NotRequired[str]
    """Optional mime type for the file."""


@pydantic.with_config({'alias_generator': to_camel})
class FileWithUri(TypedDict):
    """File with URI reference."""

    uri: str
    """The URI of the file."""

    mime_type: NotRequired[str]
    """The mime type of the file."""


@pydantic.with_config({'alias_generator': to_camel})
class FilePart(_BasePart):
    """A part that contains a file."""

    kind: Literal['file']
    """The kind of the part."""

    file: FileWithBytes | FileWithUri
    """The file content - either bytes or URI."""


@pydantic.with_config({'alias_generator': to_camel})
class DataPart(_BasePart):
    """A part that contains structured data."""

    kind: Literal['data']
    """The kind of the part."""

    data: dict[str, Any]
    """The data of the part."""


Part = Annotated[Union[TextPart, FilePart, DataPart], pydantic.Field(discriminator='kind')]
"""A fully formed piece of content exchanged between a client and a remote agent as part of a Message or an Artifact.

Each Part has its own content type and metadata.
"""

TaskState: TypeAlias = Literal[
    'submitted', 'working', 'input-required', 'completed', 'canceled', 'failed', 'rejected', 'auth-required', 'unknown'
]
"""The possible states of a task."""


@pydantic.with_config({'alias_generator': to_camel})
class TaskStatus(TypedDict):
    """Status and accompanying message for a task."""

    state: TaskState
    """The current state of the task."""

    message: NotRequired[Message]
    """Additional status updates for client."""

    timestamp: NotRequired[str]
    """ISO datetime value of when the status was updated."""


@pydantic.with_config({'alias_generator': to_camel})
class Task(TypedDict):
    """A Task is a stateful entity that allows Clients and Remote Agents to achieve a specific outcome.

    Clients and Remote Agents exchange Messages within a Task. Remote Agents generate results as Artifacts.
    A Task is always created by a Client and the status is always determined by the Remote Agent.
    """

    id: str
    """Unique identifier for the task."""

    context_id: str
    """The context the task is associated with."""

    kind: Literal['task']
    """Event type."""

    status: TaskStatus
    """Current status of the task."""

    history: NotRequired[list[Message]]
    """Optional history of messages."""

    artifacts: NotRequired[list[Artifact]]
    """Collection of artifacts created by the agent."""

    metadata: NotRequired[dict[str, Any]]
    """Extension metadata."""


@pydantic.with_config({'alias_generator': to_camel})
class TaskStatusUpdateEvent(TypedDict):
    """Sent by server during message/stream requests."""

    task_id: str
    """The id of the task."""

    context_id: str
    """The context the task is associated with."""

    kind: Literal['status-update']
    """Event type."""

    status: TaskStatus
    """The status of the task."""

    final: bool
    """Indicates the end of the event stream."""

    metadata: NotRequired[dict[str, Any]]
    """Extension metadata."""


@pydantic.with_config({'alias_generator': to_camel})
class TaskArtifactUpdateEvent(TypedDict):
    """Sent by server during message/stream requests."""

    task_id: str
    """The id of the task."""

    context_id: str
    """The context the task is associated with."""

    kind: Literal['artifact-update']
    """Event type identification."""

    artifact: Artifact
    """The artifact that was updated."""

    append: NotRequired[bool]
    """Whether to append to existing artifact (true) or replace (false)."""

    last_chunk: NotRequired[bool]
    """Indicates this is the final chunk of the artifact."""

    metadata: NotRequired[dict[str, Any]]
    """Extension metadata."""


@pydantic.with_config({'alias_generator': to_camel})
class TaskIdParams(TypedDict):
    """Parameters for a task id."""

    id: str
    metadata: NotRequired[dict[str, Any]]


@pydantic.with_config({'alias_generator': to_camel})
class TaskQueryParams(TaskIdParams):
    """Query parameters for a task."""

    history_length: NotRequired[int]
    """Number of recent messages to be retrieved."""


@pydantic.with_config({'alias_generator': to_camel})
class MessageSendConfiguration(TypedDict):
    """Configuration for the send message request."""

    accepted_output_modes: list[str]
    """Accepted output modalities by the client."""

    blocking: NotRequired[bool]
    """If the server should treat the client as a blocking request."""

    history_length: NotRequired[int]
    """Number of recent messages to be retrieved."""

    push_notification_config: NotRequired[PushNotificationConfig]
    """Where the server should send notifications when disconnected."""


@pydantic.with_config({'alias_generator': to_camel})
class MessageSendParams(TypedDict):
    """Parameters for message/send method."""

    configuration: NotRequired[MessageSendConfiguration]
    """Send message configuration."""

    message: Message
    """The message being sent to the server."""

    metadata: NotRequired[dict[str, Any]]
    """Extension metadata."""


@pydantic.with_config({'alias_generator': to_camel})
class TaskSendParams(TypedDict):
    """Internal parameters for task execution within the framework.

    Note: This is not part of the A2A protocol - it's used internally
    for broker/worker communication.
    """

    id: str
    """The id of the task."""

    context_id: str
    """The context id for the task."""

    message: Message
    """The message to process."""

    history_length: NotRequired[int]
    """Number of recent messages to be retrieved."""

    metadata: NotRequired[dict[str, Any]]
    """Extension metadata."""


@pydantic.with_config({'alias_generator': to_camel})
class ListTaskPushNotificationConfigParams(TypedDict):
    """Parameters for getting list of pushNotificationConfigurations associated with a Task."""

    id: str
    """Task id."""

    metadata: NotRequired[dict[str, Any]]
    """Extension metadata."""


@pydantic.with_config({'alias_generator': to_camel})
class DeleteTaskPushNotificationConfigParams(TypedDict):
    """Parameters for removing pushNotificationConfiguration associated with a Task."""

    id: str
    """Task id."""

    push_notification_config_id: str
    """The push notification config id to delete."""

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

InvalidAgentResponseError = JSONRPCError[Literal[-32006], Literal['Invalid agent response']]
"""A JSON RPC error for invalid agent response."""

###############################################################################################
#######################################   Requests and responses   ############################
###############################################################################################

SendMessageRequest = JSONRPCRequest[Literal['message/send'], MessageSendParams]
"""A JSON RPC request to send a message."""

SendMessageResponse = JSONRPCResponse[Union[Task, Message], JSONRPCError[Any, Any]]
"""A JSON RPC response to send a message."""

StreamMessageRequest = JSONRPCRequest[Literal['message/stream'], MessageSendParams]
"""A JSON RPC request to stream a message."""

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

ListTaskPushNotificationConfigRequest = JSONRPCRequest[
    Literal['tasks/pushNotificationConfig/list'], ListTaskPushNotificationConfigParams
]
"""A JSON RPC request to list task push notification configs."""

DeleteTaskPushNotificationConfigRequest = JSONRPCRequest[
    Literal['tasks/pushNotificationConfig/delete'], DeleteTaskPushNotificationConfigParams
]
"""A JSON RPC request to delete a task push notification config."""

A2ARequest = Annotated[
    Union[
        SendMessageRequest,
        StreamMessageRequest,
        GetTaskRequest,
        CancelTaskRequest,
        SetTaskPushNotificationRequest,
        GetTaskPushNotificationRequest,
        ResubscribeTaskRequest,
        ListTaskPushNotificationConfigRequest,
        DeleteTaskPushNotificationConfigRequest,
    ],
    Discriminator('method'),
]
"""A JSON RPC request to the A2A server."""

A2AResponse: TypeAlias = Union[
    SendMessageResponse,
    GetTaskResponse,
    CancelTaskResponse,
    SetTaskPushNotificationResponse,
    GetTaskPushNotificationResponse,
]
"""A JSON RPC response from the A2A server."""


a2a_request_ta: TypeAdapter[A2ARequest] = TypeAdapter(A2ARequest)
a2a_response_ta: TypeAdapter[A2AResponse] = TypeAdapter(A2AResponse)
send_message_request_ta: TypeAdapter[SendMessageRequest] = TypeAdapter(SendMessageRequest)
send_message_response_ta: TypeAdapter[SendMessageResponse] = TypeAdapter(SendMessageResponse)
stream_message_request_ta: TypeAdapter[StreamMessageRequest] = TypeAdapter(StreamMessageRequest)
