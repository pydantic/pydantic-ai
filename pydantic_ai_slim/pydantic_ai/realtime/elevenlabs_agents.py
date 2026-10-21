"""ElevenLabs Agents provider for realtime voice sessions.

ElevenLabs Agents runs a cascaded pipeline (realtime ASR + a configurable LLM + ElevenLabs TTS + a
server-side turn-taking model) behind a pre-provisioned agent, reached over the agent WebSocket
(`wss://api.elevenlabs.io/v1/convai/conversation?agent_id=...`). So unlike the other realtime
providers, the model here wraps a *hosted agent* addressed by its `agent_id`, and Pydantic AI takes
ownership of as much of the conversation as ElevenLabs allows per-conversation:

- Instructions, LLM choice, first message, language, voice, TTS knobs, and text-only mode are pushed
  through `conversation_initiation_client_data.conversation_config_override`. Every overridable field
  is gated by a per-field toggle under Security > Overrides in the agent's settings
  (`platform_settings.overrides`); the connect-time preflight fetches the agent and raises a
  [`UserError`][pydantic_ai.exceptions.UserError] naming the exact toggle when an override the session needs
  is not permitted. (The raw
  `elevenlabs_agents_config_override` escape hatch skips this check by design and is judged by the server
  after the handshake instead.) When the Pydantic AI
  agent defines no instructions, the ElevenLabs-side prompt is inherited silently.
- Tool declarations cannot be sent inline per-conversation: client tools are workspace entities
  referenced from the agent. By default the preflight *errors on mismatch* between the session's
  [`ToolDefinition`][pydantic_ai.tools.ToolDefinition]s and the agent's client tools; the opt-in
  `elevenlabs_agents_tool_sync='sync'` setting instead creates/updates the workspace tools and re-points the
  agent's `tool_ids` over REST before dialing.
- Turn-taking, VAD, ASR configuration, audio formats, and platform settings (auth, privacy,
  guardrails, knowledge bases) stay owned by the ElevenLabs agent configuration.

Requires the `websockets` package, available via the `elevenlabs-agents-realtime` optional group (REST
preflight uses Pydantic AI's own HTTP client, so no ElevenLabs SDK is needed):

    pip install "pydantic-ai-slim[elevenlabs-agents-realtime]"
"""

from __future__ import annotations as _annotations

import asyncio
import base64
import json
import time
import warnings
from collections.abc import AsyncGenerator, AsyncIterator, Generator, Sequence
from contextlib import asynccontextmanager, contextmanager
from dataclasses import KW_ONLY, dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal, assert_never, cast

import httpx2
from pydantic import BaseModel, ConfigDict
from pydantic_core import to_json
from typing_extensions import TypedDict

try:
    import websockets
    from websockets.asyncio.client import ClientConnection
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `websockets` package to use the ElevenLabs realtime model, '
        'you can use the `elevenlabs-agents-realtime` optional group - '
        '`pip install "pydantic-ai-slim[elevenlabs-agents-realtime]"`'
    ) from _import_error

from .. import _utils
from .._http import legacy_httpx
from .._instrumentation import get_instructions
from .._json_schema import JsonSchema, JsonSchemaTransformer
from ..exceptions import ModelHTTPError, UserError
from ..messages import (
    AudioUrl,
    BinaryAudio,
    BinaryContent,
    CachePoint,
    DocumentUrl,
    ImageUrl,
    ModelMessage,
    NativeToolCallPart,
    NativeToolReturnPart,
    PartEndEvent,
    PartStartEvent,
    RealtimeResponseInterruptedEvent,
    RealtimeSessionErrorEvent,
    TextContent,
    UploadedFile,
    VideoUrl,
)
from ..models import ModelRequestParameters, get_user_agent
from ..native_tools import MCPServerTool
from ..profiles.elevenlabs import elevenlabs_agents_realtime_model_profile
from ..providers import Provider, infer_provider
from ..tools import ToolDefinition
from ..usage import RequestUsage
from ._elevenlabs_audio_tags import AudioTagStripper as _AudioTagStripper, strip_audio_tags as _strip_audio_tags
from ._inferred_lifecycle import InferredLifecycle
from ._lifecycle import LifecycleEvent, OutputItemCut, RealtimeFrame, ResponseEnded, StaleEvent
from ._utils import inject_trace_context, require_pcm_audio, resolve_advertised_tools
from .codec import (
    AudioDelta,
    InputTranscript,
    OutputTranscript,
    RealtimeCodecEvent,
    RealtimeConnection,
    RealtimeInput,
    ResponseDone,
    SessionUsage,
    TextContext,
    ToolCall,
    ToolResult,
)
from .model import RealtimeError, RealtimeModel
from .profiles import RealtimeModelProfile, RealtimeModelProfileSpec, merge_realtime_profile
from .settings import RealtimeModelSettings

if TYPE_CHECKING:
    import httpx

    from .._http import AsyncHTTPClient
    from ..providers.elevenlabs import ElevenLabsProvider

__all__ = (
    'ElevenLabsAgentsRealtimeModel',
    'ElevenLabsAgentsRealtimeModelSettings',
    'ElevenLabsAgentsRealtimeConnection',
    'ElevenLabsAgentsTTSSettings',
)

_PROVIDER_LABEL = 'ElevenLabs Agents'

_AUDIO_TAIL_GRACE = 1.0
"""How long a reply whose text is complete can go without audio before it counts as spoken in full.

`agent_response` arrives when the LLM finishes writing, while TTS can still be voicing it for seconds (verified live,
with tool rounds and long replies alike). The reply ends with its last audio, which only an agent sending
`context_usage` marks (that report arrives with it): otherwise a gap this long, where TTS streams its frames
a tenth of a second apart, is taken to be the end.
"""

_TOOL_CALL_BATCH_WINDOW = 0.05
"""How long a reply's tool calls wait for a sibling before their response ends, in seconds.

Verified live: a reply's parallel calls come as separate `client_tool_call` frames, back to back, and nothing else
follows until their results are in, so there is no frame to end their response at. The calls go to the session as
they come (their tools run at once, and their results aren't held back); only their response ends after this quiet
stretch, or at the next frame of any other kind (verified live: each result the agent gets is acknowledged with an
`agent_tool_response`, so the next reply's calls never follow these directly).
"""

_CONTEXT_USAGE_MISSING = (
    "The ElevenLabs agent {agent_id!r} doesn't send `context_usage` reports, so the session has no context usage, "
    'and a spoken reply is taken to end after a second without audio rather than with its last audio. Add '
    "`context_usage` to the agent's `conversation.client_events` (Advanced > Client events in the agent's settings "
    'in the ElevenLabs dashboard) for both.'
)

_REALTIME_MODE_MANAGED_OVERRIDES = {('agent', 'prompt', 'llm'), ('tts', 'model_id')}
"""The overrides Pydantic AI sends whose settings ElevenLabs manages itself for an agent in Realtime voice mode.

Verified live: the server accepts them, then answers with the agent's Realtime model and TTS model all the same."""

_MCP_APPROVAL_DENIED = (
    'The ElevenLabs agent asked to approve a call to one of its MCP server tools, which a Pydantic AI session cannot '
    'do, so the call was denied. Give the agent the tools through a Pydantic AI MCP toolset instead, or turn approval '
    'off for the tool on the ElevenLabs side.'
)

_TAIL_ENDING_FRAMES = frozenset(
    {
        'context_usage',
        'user_transcript',
        'client_tool_call',
        'agent_response',
        'agent_chat_response_part',
        'mcp_tool_call',
    }
)
"""Frames that say a reply whose text is complete has no more audio coming: the next turn has begun, or (`context_usage`)
the reply was voiced in full."""


@dataclass
class _AudioTail:
    """A reply whose whole text arrived, while its audio may still be streaming."""

    event_id: int | None
    response_id: str | None
    interrupted: bool = False
    provider_details: dict[str, Any] | None = None


class ElevenLabsAgentsTTSSettings(TypedDict, total=False):
    """Per-conversation TTS overrides, each gated by its own override toggle on the agent."""

    model_id: str
    """The ElevenLabs TTS model to synthesize with, e.g. `eleven_v3_conversational`.

    Raises for an agent in Realtime voice mode, whose TTS model ElevenLabs picks."""
    stability: float
    """Voice stability (0-1); lower is more expressive, higher more consistent."""
    speed: float
    """Playback speed multiplier."""
    similarity_boost: float
    """Voice similarity boost (0-1)."""


class ElevenLabsAgentsRealtimeModelSettings(RealtimeModelSettings, total=False):
    """Settings specific to ElevenLabs Agents realtime sessions.

    Every `elevenlabs_agents_*` field below that maps to a `conversation_config_override` entry requires the
    matching override toggle to be enabled under Security > Overrides in the agent's settings; the connect-time
    preflight raises [`UserError`][pydantic_ai.exceptions.UserError] naming the toggle otherwise.

    Of the shared settings, `output_modality='text'` maps to the `conversation.text_only` override, and
    a `tool_choice` of `'none'` or an allow-list to the `agent.prompt.tool_ids` override, which narrows
    the agent's attached tools for the conversation (a choice that forces a tool call raises). `max_tokens`,
    `parallel_tool_calls`, `async_tool_calls`, and `thinking` have no per-conversation surface on the
    agent WebSocket and are silently ignored, per the shared settings contract. `turn_detection` cannot be configured
    (ElevenLabs' server-side turn model is always on), `input_transcription_model=None` cannot be
    honored (ASR cannot be disabled per-conversation), and `reconnect` is unsupported (ElevenLabs'
    experimental, text-only resume isn't used), so each of those raises rather than silently under-delivering.
    """

    elevenlabs_agents_voice_id: str
    """Voice used for audio output (the `tts.voice_id` override)."""
    elevenlabs_agents_language: str
    """Conversation language code, e.g. `de` (the `agent.language` override)."""
    elevenlabs_agents_first_message: str
    """The greeting the agent opens with (the `agent.first_message` override)."""
    elevenlabs_agents_llm: str
    """The LLM the ElevenLabs agent pipeline runs, e.g. `gpt-5.2` (the `agent.prompt.llm` override).

    Raises for an agent in Realtime voice mode, whose LLM ElevenLabs picks."""
    elevenlabs_agents_tts: ElevenLabsAgentsTTSSettings
    """TTS model and voice-delivery knobs (the `tts.*` overrides)."""
    elevenlabs_agents_dynamic_variables: dict[str, str | int | float | bool]
    """Values substituted into `{{placeholders}}` in the agent's prompts and messages."""
    elevenlabs_agents_custom_llm_extra_body: dict[str, Any]
    """Extra body merged into the agent's custom-LLM request. Gated by the agent's
    `custom_llm_extra_body` security toggle (a sibling of the override toggles)."""
    elevenlabs_agents_user_id: str
    """Opaque end-user identifier attached to the conversation for ElevenLabs-side analytics."""
    elevenlabs_agents_tool_sync: Literal['error', 'sync', 'off']
    """How to reconcile the session's tools with the agent's client tools at connect time.

    - `'error'` (default): raise [`UserError`][pydantic_ai.exceptions.UserError] describing every
      difference between the session's [`ToolDefinition`][pydantic_ai.tools.ToolDefinition]s and the
      agent's client tools, so the two can never silently disagree.
    - `'sync'`: make the workspace match Pydantic AI over REST before dialing: create missing client
      tools, update differing ones, and re-point the agent's `tool_ids` (dropping client tools the
      session doesn't define; ElevenLabs-side webhook/MCP/system tools are left untouched). This
      mutates workspace state shared by every conversation with the agent, which is why it is opt-in.
    - `'off'`: skip the check and trust the agent's own tool definitions, e.g. for client tools configured on the
      ElevenLabs side with settings Pydantic AI doesn't sync. A call to a tool the session does not define reaches
      the tool manager like any other call and is answered with the standard unknown-tool error, which lands in
      the session's history.
    """
    elevenlabs_agents_strip_audio_tags: bool
    """Whether to remove expressive audio tags such as `[laughs]` from the agent's transcripts. Defaults to `True`.

    With an expressive TTS model such as `eleven_v3_conversational`, the agent's LLM writes tags into its reply to
    direct the delivery, and they are part of the transcript ElevenLabs sends, though they aren't spoken. They are
    removed from the transcript, streamed and in history alike, and the tags a response's transcript held are kept,
    in order and without brackets, in its `provider_details['audio_tags']`. Only lowercase words in square brackets
    count as tags, so citations (`[1]`), Markdown links, and placeholders like `[Your Name]` are kept. Set this to
    `False` to keep the transcript exactly as ElevenLabs sends it, tags included.
    """
    elevenlabs_agents_config_override: dict[str, Any]
    """Raw values deep-merged last into `conversation_config_override`, the escape hatch mirroring
    `google_config_overrides`. Not preflight-checked against the override toggles; the server still
    rejects non-permitted fields."""


# --- REST payload models -------------------------------------------------------------------------
# The GET-agent response mirrored just far enough for the preflight; everything else is tolerated
# and ignored (`extra='allow'` keeps unknown platform fields from failing validation).


class _AgentToolConfig(BaseModel):
    """One resolved tool on the agent (`conversation_config.agent.prompt.tools`).

    Verified live: resolved tools do *not* carry their workspace tool id inline, so `'sync'` mode
    resolves ids through the workspace tool listing instead.
    """

    model_config = ConfigDict(extra='allow')

    type: str | None = None
    name: str | None = None
    description: str | None = None
    parameters: dict[str, Any] | None = None
    expects_response: bool | None = None
    response_timeout_secs: int | None = None


class _AgentPromptConfig(BaseModel):
    model_config = ConfigDict(extra='allow')

    prompt: str | None = None
    llm: str | None = None
    tool_ids: list[str] | None = None
    tools: list[_AgentToolConfig] | None = None


class _AgentSectionConfig(BaseModel):
    model_config = ConfigDict(extra='allow')

    first_message: str | None = None
    language: str | None = None
    prompt: _AgentPromptConfig | None = None


class _ConversationSettings(BaseModel):
    model_config = ConfigDict(extra='allow')

    client_events: list[str] | None = None


class _ConversationConfig(BaseModel):
    model_config = ConfigDict(extra='allow')

    agent: _AgentSectionConfig | None = None
    conversation: _ConversationSettings | None = None
    realtime_model: str | None = None
    """The Eleven Realtime model of an agent in Realtime voice mode, e.g. `eleven_realtime_v1_mini`; `None` in Custom mode."""


class _AgentAuthSettings(BaseModel):
    model_config = ConfigDict(extra='allow')

    enable_auth: bool = False


class _PlatformSettings(BaseModel):
    model_config = ConfigDict(extra='allow')

    # The per-field override-permission tree mirrors the `conversation_config` shape with booleans at
    # the leaves, so it stays a plain mapping walked by `_override_allowed` rather than a typed model.
    overrides: dict[str, Any] | None = None
    auth: _AgentAuthSettings | None = None


class _AgentPreflight(BaseModel):
    """The subset of `GET /v1/convai/agents/{agent_id}` the connect-time preflight reads."""

    model_config = ConfigDict(extra='allow')

    agent_id: str | None = None
    conversation_config: _ConversationConfig | None = None
    platform_settings: _PlatformSettings | None = None

    @property
    def override_permissions(self) -> dict[str, Any]:
        if self.platform_settings is None or self.platform_settings.overrides is None:
            return {}
        permissions = self.platform_settings.overrides.get('conversation_config_override')
        return cast('dict[str, Any]', permissions) if isinstance(permissions, dict) else {}

    @property
    def sends_context_usage(self) -> bool:
        """Whether the agent sends `context_usage` reports, which aren't among its default `client_events`."""
        conversation = self.conversation_config.conversation if self.conversation_config is not None else None
        return conversation is not None and 'context_usage' in (conversation.client_events or ())

    @property
    def custom_llm_extra_body_allowed(self) -> bool:
        if self.platform_settings is None or self.platform_settings.overrides is None:
            return False
        return self.platform_settings.overrides.get('custom_llm_extra_body') is True

    @property
    def client_tools(self) -> list[_AgentToolConfig]:
        if self.conversation_config is None or self.conversation_config.agent is None:
            return []
        prompt = self.conversation_config.agent.prompt
        if prompt is None or prompt.tools is None:
            return []
        return [tool for tool in prompt.tools if tool.type == 'client']

    @property
    def tool_ids(self) -> list[str]:
        """The workspace tool ids attached to the agent, in their configured order."""
        if self.conversation_config is None or self.conversation_config.agent is None:
            return []
        prompt = self.conversation_config.agent.prompt
        if prompt is None or prompt.tool_ids is None:
            return []
        return prompt.tool_ids


class _SignedUrlResponse(BaseModel):
    model_config = ConfigDict(extra='allow')

    signed_url: str


class _CreatedToolResponse(BaseModel):
    model_config = ConfigDict(extra='allow')

    id: str


class _WorkspaceTool(BaseModel):
    """One workspace tool, as listed by `GET /v1/convai/tools` or fetched by `GET /v1/convai/tools/{id}`.

    The listing is the only place workspace tool ids are reported; the per-id fetch is how attached
    tools are resolved once the agent response no longer carries them inline.
    """

    model_config = ConfigDict(extra='allow')

    id: str
    tool_config: _AgentToolConfig


class _WorkspaceToolsResponse(BaseModel):
    """One page of `GET /v1/convai/tools`, which is cursor-paginated."""

    model_config = ConfigDict(extra='allow')

    tools: list[_WorkspaceTool] = []
    has_more: bool = False
    next_cursor: str | None = None


class _DependentAgent(BaseModel):
    """One agent referencing a tool: an `available` one the key can see, or an `unknown` one it can't."""

    model_config = ConfigDict(extra='allow')

    id: str | None = None


class _DependentAgentsResponse(BaseModel):
    """One page of `GET /v1/convai/tools/{tool_id}/dependent-agents` (verified live).

    Only who is listed is read: any agent but this one, or a further page, means the tool is shared.
    """

    model_config = ConfigDict(extra='allow')

    agents: list[_DependentAgent] = []
    has_more: bool = False


# --- Wire event models ---------------------------------------------------------------------------
# ElevenLabs wraps most server events as `{"type": <x>, "<x>_event": {...}}`; the models below type
# the payloads the adapter reads. There is no SDK to borrow event types from.


class _ConversationInitiationMetadata(BaseModel):
    model_config = ConfigDict(extra='allow')

    conversation_id: str | None = None
    agent_output_audio_format: str | None = None
    user_input_audio_format: str | None = None


class _ConversationInitiationMetadataEvent(BaseModel):
    conversation_initiation_metadata_event: _ConversationInitiationMetadata


class _AudioPayload(BaseModel):
    model_config = ConfigDict(extra='allow')

    audio_base_64: str
    event_id: int | None = None
    """The agent turn the audio belongs to: the `event_id` of the `agent_response` with its text."""


class _AudioEvent(BaseModel):
    audio_event: _AudioPayload


class _AgentResponsePayload(BaseModel):
    model_config = ConfigDict(extra='allow')

    agent_response: str
    response_id: str | None = None
    event_id: int | None = None


class _AgentResponseEvent(BaseModel):
    agent_response_event: _AgentResponsePayload


class _TextResponsePart(BaseModel):
    """One `agent_chat_response_part` chunk (wrapper key `text_response_part`, verified live).

    In text-only conversations the response text streams as `start`/`delta`/`stop` parts ahead of
    the whole-text `agent_response`; these arrive regardless of the agent's `client_events` set.
    """

    model_config = ConfigDict(extra='allow')

    text: str = ''
    type: str | None = None
    """`start`, `delta`, or `stop`."""
    response_id: str | None = None
    """The agent message the part belongs to, the `response_id` its `agent_response` will carry."""
    event_id: int | None = None
    """The agent turn the message belongs to."""


class _AgentChatResponsePartEvent(BaseModel):
    text_response_part: _TextResponsePart


class _AgentResponseCorrectionPayload(BaseModel):
    model_config = ConfigDict(extra='allow')

    original_agent_response: str | None = None
    corrected_agent_response: str
    response_id: str | None = None
    """The agent message corrected. A turn can hold several (text said before a tool call, then the answer to its
    result), and one barge-in corrects each the user didn't hear in full, with a correction of its own (verified
    live): the message cut off is cut down to what was heard, and those after it, never heard, to `''`."""


class _AgentResponseCorrectionEvent(BaseModel):
    agent_response_correction_event: _AgentResponseCorrectionPayload


class _AgentResponseCompletePayload(BaseModel):
    model_config = ConfigDict(extra='allow')

    event_id: int | None = None
    """The agent turn whose audio has played out in full."""


class _AgentResponseCompleteEvent(BaseModel):
    agent_response_complete_event: _AgentResponseCompletePayload


class _UserTranscriptionPayload(BaseModel):
    model_config = ConfigDict(extra='allow')

    user_transcript: str


class _UserTranscriptEvent(BaseModel):
    user_transcription_event: _UserTranscriptionPayload


class _PingPayload(BaseModel):
    model_config = ConfigDict(extra='allow')

    event_id: int | str | None = None


class _PingEvent(BaseModel):
    ping_event: _PingPayload


class _ClientToolCallPayload(BaseModel):
    model_config = ConfigDict(extra='allow')

    tool_name: str
    tool_call_id: str
    # Verified live: `parameters` arrives as a parsed object; a JSON string is tolerated defensively.
    parameters: dict[str, Any] | str | None = None
    expects_response: bool = True


class _ClientToolCallEvent(BaseModel):
    client_tool_call: _ClientToolCallPayload


class _QueueStatusPayload(BaseModel):
    model_config = ConfigDict(extra='allow')

    status: str
    """`waiting` (hold audio follows as regular `audio` frames), `admitted`, or `timed_out` (a 4300 close follows)."""


class _QueueStatusEvent(BaseModel):
    queue_status_event: _QueueStatusPayload


class _McpToolCallPayload(BaseModel):
    """One step of a call to a tool on an MCP server the agent is connected to, which ElevenLabs runs itself."""

    model_config = ConfigDict(extra='allow')

    service_id: str | None = None
    tool_call_id: str
    tool_name: str
    parameters: dict[str, Any] | None = None
    state: str
    """`loading`, `awaiting_approval` (the agent waits for an `mcp_tool_approval_result`), `success`, or `failure`."""
    approval_timeout_secs: int | None = None
    result: list[Any] | None = None
    error_message: str | None = None


class _McpToolCallEvent(BaseModel):
    mcp_tool_call: _McpToolCallPayload


class _ContextUsagePayload(BaseModel):
    model_config = ConfigDict(extra='allow')

    # The payload also names the pipeline LLM (`model`); it is deliberately not read, see
    # `ElevenLabsAgentsRealtimeConnection.model_name`.
    context_tokens: int | None = None
    context_limit_tokens: int | None = None


class _ContextUsageEvent(BaseModel):
    model_config = ConfigDict(extra='allow')

    # Verified live: the payload arrives under `context_usage_event`.
    context_usage_event: _ContextUsagePayload


class _ClientErrorPayload(BaseModel):
    model_config = ConfigDict(extra='allow')

    code: int | str | None = None
    error_name: str | None = None
    message: str | None = None


class _ClientErrorEvent(BaseModel):
    model_config = ConfigDict(extra='allow')

    # The AsyncAPI docs wrap the payload as `{"type": "client_error", "error_event": {...}}`; never
    # observed live (non-permitted overrides reject with a post-handshake 1008 close instead).
    error_event: _ClientErrorPayload


# --- Preflight helpers ---------------------------------------------------------------------------


def _loads_obj(raw: str) -> dict[str, Any]:
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(f'expected a JSON object frame, got {type(data).__name__}')
    return cast('dict[str, Any]', data)


def _override_allowed(permissions: dict[str, Any], path: tuple[str, ...]) -> bool:
    """Whether the agent's override-permission tree enables the override at `path`."""
    node: Any = permissions
    for key in path:
        if not isinstance(node, dict):
            return False
        node = cast('dict[str, Any]', node).get(key)
    return node is True


def _set_override(overrides: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    node = overrides
    for key in path[:-1]:
        node = cast('dict[str, Any]', node.setdefault(key, {}))
    node[path[-1]] = value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> None:
    for key, value in override.items():
        existing = base.get(key)
        if isinstance(value, dict) and isinstance(existing, dict):
            _deep_merge(cast('dict[str, Any]', existing), cast('dict[str, Any]', value))
        else:
            base[key] = value


# The REST preflight's transport failures, mapped to `RealtimeError` so a preflight that cannot
# reach the API surfaces like any other failed connect rather than leaking an HTTP-client exception.
_REST_TRANSPORT_ERRORS: tuple[type[Exception], ...] = (
    (httpx2.HTTPError,) if legacy_httpx is None else (httpx2.HTTPError, legacy_httpx.HTTPError)
)


@contextmanager
def _map_rest_errors(model_name: str) -> Generator[None]:
    """Map REST preflight transport failures to [`RealtimeError`][pydantic_ai.realtime.RealtimeError]."""
    try:
        yield
    except _REST_TRANSPORT_ERRORS as e:
        raise RealtimeError(model_name=model_name, message=f'Could not reach the ElevenLabs API: {e}') from e


def _raise_for_status(response: httpx.Response | httpx2.Response, model_name: str) -> None:
    """Raise a [`ModelHTTPError`][pydantic_ai.exceptions.ModelHTTPError] on a non-2xx preflight response."""
    if not response.is_success:
        raise ModelHTTPError(
            status_code=response.status_code,
            model_name=model_name,
            body=response.text.strip() or None,
            headers=response.headers,
        )


class _ElevenLabsSchemaTransformer(JsonSchemaTransformer):
    """Normalize a local JSON schema into plain nodes the ElevenLabs dialect can be projected from.

    Pydantic-generated schemas routinely carry `$defs`/`$ref` (nested models) and
    `anyOf: [X, null]` (optional parameters), neither of which the dialect can express directly:
    inlining the definitions and collapsing nullable unions onto the non-null member leaves every
    node with a plain `type`, so both the sync-mode conversion and the `'error'`-mode comparison see
    the schema the way the server would store it. (Optionality is carried by `required` alone; the
    `nullable` marker the collapse leaves behind is not part of the dialect and is dropped by the
    projection.)
    """

    def __init__(self, schema: JsonSchema):
        super().__init__(schema, prefer_inlined_defs=True, simplify_nullable_unions=True)

    def transform(self, schema: JsonSchema) -> JsonSchema:
        return schema


def _prepared_parameters(schema: dict[str, Any]) -> dict[str, Any]:
    """A tool's parameters schema with `$defs` inlined and nullable unions collapsed."""
    return _ElevenLabsSchemaTransformer(schema).walk()


def _elevenlabs_parameters(tool_name: str, schema: dict[str, Any]) -> dict[str, Any]:
    """Convert a prepared JSON schema to the ElevenLabs client-tool parameters dialect (verified live).

    The dialect accepts only `type`/`description`/`enum`/`items`/`properties`/`required`:
    `additionalProperties` is rejected outright (HTTP 422 `extra_forbidden`) and other JSON Schema
    keywords (numeric bounds etc.) are silently dropped by the server, so only the supported keys are
    sent. The server also requires a non-empty description on every property, which Pydantic AI
    cannot invent, so a schema without one fails loudly here; so does a node the dialect cannot
    express at all (a union that `_prepared_parameters` could not collapse, or a recursive `$ref`),
    which has no `type` left to send. Every offending parameter in the schema is reported in one
    error rather than the first found, so a tool with several problems costs one connect, not one
    per parameter.
    """
    problems: list[str] = []
    result = _convert_parameters(schema, path='', root=True, problems=problems)
    if problems:
        raise UserError(
            f'Cannot sync tool {tool_name!r}: '
            + '; '.join(problems)
            + '. Add parameter descriptions (e.g. in the function docstring) or simplify the parameter '
            "types, or keep the tool configured in the ElevenLabs dashboard with `elevenlabs_agents_tool_sync='off'`."
        )
    return result


def _convert_parameters(schema: dict[str, Any], *, path: str, root: bool, problems: list[str]) -> dict[str, Any]:
    """One node of `_elevenlabs_parameters`, recording problems instead of raising on the first."""
    result: dict[str, Any] = {}
    for key in ('type', 'description', 'enum'):
        if (value := schema.get(key)) is not None:
            result[key] = value
    if 'type' not in result:
        subject = f'parameter {path!r}' if path else 'the parameters schema'
        problems.append(
            f'{subject} uses a JSON Schema feature the ElevenLabs tool dialect cannot express '
            '(a union of types, or a recursive reference)'
        )
    if not root and not result.get('description'):
        problems.append(f'ElevenLabs requires a description for every tool parameter, and {path!r} has none')
    if isinstance(items := schema.get('items'), dict):
        items = cast('dict[str, Any]', items)
        result['items'] = _convert_parameters(items, path=f'{path}[]', root=False, problems=problems)
    if isinstance(properties := schema.get('properties'), dict):
        properties = cast('dict[str, Any]', properties)
        result['properties'] = {
            name: _convert_parameters(
                cast('dict[str, Any]', prop), path=f'{path}.{name}'.lstrip('.'), root=False, problems=problems
            )
            for name, prop in properties.items()
            if isinstance(prop, dict)
        }
    if (required := schema.get('required')) is not None:
        result['required'] = required
    return result


def _tool_config(tool: ToolDefinition, remote: _AgentToolConfig | None = None) -> dict[str, Any]:
    """Render a [`ToolDefinition`][pydantic_ai.tools.ToolDefinition] as an ElevenLabs client-tool config.

    An update replaces the whole config, which resets a timeout raised on the ElevenLabs side to the default
    (verified live), so the `remote` tool being updated keeps its own.
    """
    config: dict[str, Any] = {
        'type': 'client',
        'name': tool.name,
        'description': tool.description or '',
        # Pydantic AI's session always settles a call and sends its result back, so every synced tool
        # expects a response; the server's default `response_timeout_secs` (20s) is left in place.
        'expects_response': True,
        'parameters': _elevenlabs_parameters(tool.name, _prepared_parameters(tool.parameters_json_schema)),
    }
    if remote is not None and remote.response_timeout_secs is not None:
        config['response_timeout_secs'] = remote.response_timeout_secs
    return config


def _schemas_match(local: dict[str, Any], remote: dict[str, Any]) -> bool:
    """Whether a prepared local JSON schema (see `_prepared_parameters`) and a server-normalized stored schema agree.

    Verified live: the server stores schemas in its own dialect, injecting bookkeeping fields
    (`is_omitted`, `dynamic_variable`, `enum: null`, empty descriptions, ...) and dropping keywords
    it cannot express (`additionalProperties`, numeric bounds), so strict equality is impossible.
    Compared instead: `type`, `enum` (a remote `null` counts as absent), the `required` set, the
    property name set (recursively), `items`, and `description` only where the local schema declares
    one (the server *requires* property descriptions, so a description-less local schema can never
    match verbatim).
    """
    if local.get('type') != remote.get('type'):
        return False
    if (local.get('enum') or None) != (remote.get('enum') or None):
        return False
    if (description := local.get('description')) and description != remote.get('description'):
        return False
    if set(cast('list[str]', local.get('required') or [])) != set(cast('list[str]', remote.get('required') or [])):
        return False
    local_properties = cast('dict[str, Any]', local.get('properties') or {})
    remote_properties = cast('dict[str, Any]', remote.get('properties') or {})
    if set(local_properties) != set(remote_properties):
        return False
    for name, local_property in local_properties.items():
        if not _schemas_match(cast('dict[str, Any]', local_property), cast('dict[str, Any]', remote_properties[name])):
            return False
    local_items = local.get('items')
    remote_items = remote.get('items')
    if isinstance(local_items, dict):
        if not isinstance(remote_items, dict):
            return False
        return _schemas_match(cast('dict[str, Any]', local_items), cast('dict[str, Any]', remote_items))
    # A local array without `items` accepts anything; a remote that constrains its items is still a
    # different contract and must count as a mismatch, exactly like the inverse case above.
    return not isinstance(remote_items, dict)


def _tool_mismatches(local: Sequence[ToolDefinition], remote: Sequence[_AgentToolConfig]) -> list[str]:
    """Describe every difference between the session's tools and the agent's client tools.

    Names must match exactly (they are case-sensitive on the wire), and matched tools must agree on
    description, parameters schema (per `_schemas_match`), and `expects_response=True`. Only client
    tools participate: server-side webhook/MCP/system tools are ElevenLabs-owned and not managed
    from here.
    """
    mismatches: list[str] = []
    remote_by_name = {tool.name: tool for tool in remote if tool.name}
    local_by_name = {tool.name: tool for tool in local}
    for name in sorted(set(local_by_name) - set(remote_by_name)):
        mismatches.append(f'tool {name!r} is not configured on the agent')
    for name in sorted(set(remote_by_name) - set(local_by_name)):
        mismatches.append(f'agent client tool {name!r} is not defined by this agent run')
    for name in sorted(set(local_by_name) & set(remote_by_name)):
        local_tool, remote_tool = local_by_name[name], remote_by_name[name]
        if (local_tool.description or '') != (remote_tool.description or ''):
            mismatches.append(f'tool {name!r} differs in description')
        if not _schemas_match(_prepared_parameters(local_tool.parameters_json_schema), remote_tool.parameters or {}):
            mismatches.append(f'tool {name!r} differs in parameters schema')
        if remote_tool.expects_response is False:
            mismatches.append(
                f'agent client tool {name!r} has `expects_response` disabled, '
                'but Pydantic AI tools always return a result'
            )
    return mismatches


class _HandshakeError(Exception):
    """A failure while establishing the ElevenLabs conversation (bad frame, error frame, or timeout)."""


@contextmanager
def _map_connect_errors(model_name: str) -> Generator[None]:
    """Map dial/handshake failures to the typed exceptions the regular models raise.

    Mirrors the OpenAI-protocol `map_connect_errors` (not imported: that module requires the `openai`
    package, which this provider deliberately doesn't depend on).
    """
    try:
        yield
    except _HandshakeError as e:
        raise RealtimeError(model_name=model_name, message=str(e)) from e
    except websockets.InvalidStatus as e:
        # The WebSocket upgrade was rejected (expired signed URL, unknown agent); it carries a real
        # HTTP status, so it maps to `ModelHTTPError` exactly like a regular request.
        response = e.response
        body = response.body.decode(errors='replace') if response.body else response.reason_phrase
        raise ModelHTTPError(
            status_code=response.status_code,
            model_name=model_name,
            body=body,
            # Keep the rejected upgrade's headers, exactly like the REST path: `retry_after` on a
            # 429 reads them.
            headers=dict(response.headers),
        ) from e
    except websockets.WebSocketException as e:
        raise RealtimeError(model_name=model_name, message=f'WebSocket error during realtime handshake: {e}') from e
    except OSError as e:
        # DNS failure, refused, reset, or a dial timeout (`TimeoutError` is an `OSError`).
        raise RealtimeError(model_name=model_name, message=f'Could not reach the realtime API: {e}') from e


async def _expect_handshake(ws: ClientConnection, *, timeout: float) -> _ConversationInitiationMetadata:
    """Read frames until `conversation_initiation_metadata` arrives, answering pings meanwhile."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=max(0.0, deadline - time.monotonic()))
        except TimeoutError:
            raise _HandshakeError("timed out waiting for a 'conversation_initiation_metadata' event") from None
        if not isinstance(raw, str):
            raise _HandshakeError(f'expected a text frame, got {type(raw).__name__}')
        try:
            data = _loads_obj(raw)
        except ValueError as e:
            raise _HandshakeError(f'received a malformed frame: {e}') from e
        event_type = data.get('type')
        # Pydantic validation failures (`ValueError`s) are wrapped like the JSON-level ones above:
        # a known frame type with a payload the models reject is still a malformed handshake frame,
        # and must surface as a typed connect error rather than a raw `ValidationError`.
        try:
            if event_type == 'conversation_initiation_metadata':
                return _ConversationInitiationMetadataEvent.model_validate(data).conversation_initiation_metadata_event
            if event_type == 'ping':
                await ws.send(_pong_frame(data))
        except ValueError as e:
            raise _HandshakeError(f'received a malformed frame: {e}') from e
        # Verified live: a non-permitted override is rejected *after* a successful handshake, by a
        # WebSocket close with code 1008 and the reason naming the field (e.g. "Override for field
        # 'max_duration_seconds' is not allowed by config."), which surfaces through the receive
        # loop as a non-recoverable session error. An error-looking frame during the handshake is
        # still treated as a handshake failure defensively.
        if event_type in ('client_error', 'error'):
            raise _HandshakeError(f'the server rejected the conversation: {data}')


def _pong_frame(ping_frame: dict[str, Any]) -> str:
    """Render the `pong` answer to a `ping` frame, raising `ValueError` for a malformed ping."""
    event = _PingEvent.model_validate(ping_frame)
    return to_json({'type': 'pong', 'event_id': event.ping_event.event_id}).decode()


def _validate_audio_format(wire_format: str | None, *, expected_rate: int, direction: str, model_name: str) -> None:
    """Check a handshake-reported audio format against the profile's PCM sample rate."""
    if wire_format is None:
        # Both formats were always present in live handshakes; an absent one skips validation
        # rather than failing the connect.
        return
    prefix, _, rate = wire_format.rpartition('_')
    # `str.isdigit` accepts non-ASCII digits (e.g. superscripts) that `int()` rejects, which would
    # leak a raw `ValueError` out of the handshake instead of the typed error below.
    if prefix != 'pcm' or not rate.isascii() or not rate.isdigit() or int(rate) != expected_rate:
        raise RealtimeError(
            model_name=model_name,
            message=(
                f'The agent is configured for {direction} audio format {wire_format!r}, but the model profile '
                f"expects mono PCM16 at {expected_rate} Hz. Pass `profile={{...}}` with the agent's actual "
                '`audio_input_sample_rate`/`audio_output_sample_rate` (PCM formats only; `ulaw_8000` telephony '
                'agents are not supported).'
            ),
        )


@dataclass(init=False)
class ElevenLabsAgentsRealtimeModel(RealtimeModel):
    """ElevenLabs Agents realtime model, wrapping a hosted agent addressed by its `agent_id`.

    Args:
        agent_id: The ElevenLabs agent to converse with, e.g. `agent_0101k2...`. The agent must exist;
            Pydantic AI overrides its per-conversation configuration where the agent's override
            toggles allow, and fails loudly where they don't.
        provider: The provider to use for authentication and API access. Defaults to `'elevenlabs'`
            (reads `ELEVENLABS_API_KEY`); pass an
            [`ElevenLabsProvider`][pydantic_ai.providers.elevenlabs.ElevenLabsProvider] for a custom
            key, HTTP client, or regional (data-residency) base URL.
        settings: [Model settings][pydantic_ai.realtime.elevenlabs_agents.ElevenLabsAgentsRealtimeModelSettings]
            used as defaults for realtime sessions.
        profile: Optional override for the [realtime model profile][pydantic_ai.realtime.RealtimeModelProfile],
            merged over the provider's. For ElevenLabs this is how per-agent audio formats are
            declared when the agent isn't on the default 16 kHz PCM formats.
    """

    agent_id: str
    _: KW_ONLY
    settings: RealtimeModelSettings | None = None
    _provider: ElevenLabsProvider = field(init=False, repr=False)
    _warned_context_usage: bool = field(default=False, init=False, repr=False)
    """Whether a session has warned that the agent sends no `context_usage` reports (once per model)."""

    # Written out rather than generated because `profile` has to be an init argument handed to the
    # base constructor, while `RealtimeModel.profile` stays the *resolved* profile, exactly as on a
    # standard `Model`.
    def __init__(
        self,
        agent_id: str,
        *,
        provider: Provider[AsyncHTTPClient] | str = 'elevenlabs',
        settings: RealtimeModelSettings | None = None,
        profile: RealtimeModelProfileSpec | None = None,
    ) -> None:
        super().__init__(settings=settings, profile=profile)
        self.agent_id = agent_id
        self._warned_context_usage = False
        from ..providers.elevenlabs import ElevenLabsProvider

        if isinstance(provider, str):
            provider = infer_provider(provider)
        if not isinstance(provider, ElevenLabsProvider):
            # The REST preflight and signed-URL auth below read the ElevenLabs-specific `api_key` off
            # the provider; a foreign one would fail with an `AttributeError` instead of the real mistake.
            raise UserError(
                f"`ElevenLabsAgentsRealtimeModel` requires an `ElevenLabsProvider` or `provider='elevenlabs'`; "
                f'got {provider.name!r}.'
            )
        self._provider = provider

    @property
    def model_name(self) -> str:
        """The agent id; ElevenLabs has no realtime model id apart from the agent."""
        return self.agent_id

    @property
    def system(self) -> str:
        return 'elevenlabs'

    def _adjust_provider_profile(self, profile: RealtimeModelProfile) -> RealtimeModelProfile:
        # An agent id names no model, so the provider has nothing to say about one: what holds for every agent comes
        # from here, ahead of the `profile=` argument.
        return merge_realtime_profile(profile, elevenlabs_agents_realtime_model_profile(self.model_name))

    # --- REST preflight ---------------------------------------------------------------------------

    @property
    def _http_client(self) -> AsyncHTTPClient:
        return self._provider.client

    @property
    def _rest_headers(self) -> dict[str, str]:
        return {'xi-api-key': self._provider.api_key}

    async def _fetch_agent(self) -> _AgentPreflight:
        with _map_rest_errors(self.agent_id):
            response = await self._http_client.get(
                f'{self._provider.base_url}/v1/convai/agents/{self.agent_id}', headers=self._rest_headers
            )
        _raise_for_status(response, self.agent_id)
        agent = _AgentPreflight.model_validate_json(response.content)
        await self._resolve_attached_tools(agent)
        return agent

    async def _resolve_attached_tools(self, agent: _AgentPreflight) -> None:
        """Fill in `prompt.tools` from the workspace when the agent response no longer resolves them inline.

        ElevenLabs deprecated inline `prompt.tools` in favor of `tool_ids`, and its deprecation
        notice says `GET` responses stopped returning the resolved `tools` array on 2025-07-15. Live
        responses still carried it on 2026-08-28 (every recorded cassette relies on that), so the
        inline array stays the primary source. Tolerated defensively: an agent reporting attached
        `tool_ids` with no inline `tools` has each id fetched through `GET /v1/convai/tools/{id}`, so
        the preflight comparison and `'sync'` mode keep working the day the documented behavior
        lands, instead of reporting every tool as missing. A dangling id (404) is skipped: the agent
        cannot call a tool that no longer exists, and workspace hygiene should not block a connect.
        """
        if agent.conversation_config is None or agent.conversation_config.agent is None:
            return
        prompt = agent.conversation_config.agent.prompt
        if prompt is None or prompt.tools or not prompt.tool_ids:
            return
        resolved: list[_AgentToolConfig] = []
        for tool_id in prompt.tool_ids:
            with _map_rest_errors(self.agent_id):
                response = await self._http_client.get(
                    f'{self._provider.base_url}/v1/convai/tools/{tool_id}', headers=self._rest_headers
                )
            if response.status_code == 404:
                continue
            _raise_for_status(response, self.agent_id)
            resolved.append(_WorkspaceTool.model_validate_json(response.content).tool_config)
        prompt.tools = resolved

    async def _signed_url(self) -> str:
        """Mint a short-lived signed WebSocket URL for the conversation.

        Used unconditionally: it is the only server-side auth flow the WebSocket accepts (verified
        live: with agent auth enabled, dialing with the `xi-api-key` header is rejected by a close
        with code 3000 telling you to generate a signed link), it works for public agents too, and
        it keeps the API key out of the WebSocket URL.
        """
        with _map_rest_errors(self.agent_id):
            response = await self._http_client.get(
                f'{self._provider.base_url}/v1/convai/conversation/get-signed-url',
                params={'agent_id': self.agent_id},
                headers=self._rest_headers,
            )
        _raise_for_status(response, self.agent_id)
        return _SignedUrlResponse.model_validate_json(response.content).signed_url

    async def _workspace_tool_listing(self) -> dict[str, _WorkspaceTool]:
        """Fetch the full workspace tool listing by id, following the cursor pagination.

        Every page is fetched so attached ids are never misclassified (and duplicated as fresh
        creates) just because a workspace holds more tools than one page carries.
        """
        workspace_tools: dict[str, _WorkspaceTool] = {}
        cursor: str | None = None
        seen_cursors: set[str] = set()
        while True:
            params: dict[str, Any] = {'page_size': 100}
            if cursor is not None:
                params['cursor'] = cursor
            with _map_rest_errors(self.agent_id):
                response = await self._http_client.get(
                    f'{self._provider.base_url}/v1/convai/tools', params=params, headers=self._rest_headers
                )
            _raise_for_status(response, self.agent_id)
            page = _WorkspaceToolsResponse.model_validate_json(response.content)
            workspace_tools.update({tool.id: tool for tool in page.tools})
            # A repeated cursor (a server bug) would otherwise refetch the same page forever; ending
            # the listing there is safe because attached ids it leaves unresolved are preserved in
            # `tool_ids`, never dropped.
            if not page.has_more or page.next_cursor is None or page.next_cursor in seen_cursors:
                break
            seen_cursors.add(page.next_cursor)
            cursor = page.next_cursor
        return workspace_tools

    async def _tool_used_elsewhere(self, tool_id: str) -> bool:
        """Whether an agent other than this one references the tool.

        Workspace tools are shared: changing one changes every agent it is attached to. So an attached
        tool is only updated in place, and an unattached one only adopted, when no other agent uses it.
        A `has_more` page counts as shared, erring toward a fresh create over a shared mutation.
        """
        with _map_rest_errors(self.agent_id):
            response = await self._http_client.get(
                f'{self._provider.base_url}/v1/convai/tools/{tool_id}/dependent-agents',
                headers=self._rest_headers,
            )
        _raise_for_status(response, self.agent_id)
        page = _DependentAgentsResponse.model_validate_json(response.content)
        return page.has_more or any(agent.id != self.agent_id for agent in page.agents)

    async def _create_tool(self, config: dict[str, Any]) -> str:
        """Create a workspace client tool, returning its id."""
        with _map_rest_errors(self.agent_id):
            response = await self._http_client.post(
                f'{self._provider.base_url}/v1/convai/tools',
                headers={**self._rest_headers, 'Content-Type': 'application/json'},
                content=to_json({'tool_config': config}).decode(),
            )
        _raise_for_status(response, self.agent_id)
        return _CreatedToolResponse.model_validate_json(response.content).id

    async def _sync_tools(self, agent: _AgentPreflight, tools: Sequence[ToolDefinition]) -> list[str]:
        """Make the agent's client tools match the session's tools over REST (`elevenlabs_agents_tool_sync='sync'`).

        Creates missing workspace client tools, updates differing ones (`PATCH /v1/convai/tools/{id}`
        takes the same `tool_config` body as create, verified live), and re-points the agent's
        `tool_ids` at the resulting set plus the agent's untouched server-side tools (agent PATCH is
        a partial update: only the provided leaves change, verified live). This mutates workspace
        state shared by every conversation with the agent. The PATCH channel is the only one: the
        toggle-gated `tool_ids` *override* at initiation cannot add tools, it only restricts to ids
        already attached to the agent (foreign ids close the socket with 1008, verified live).

        When nothing differs, no extra request is made. Otherwise the workspace tool listing is
        fetched once to resolve tool ids, which the agent's resolved tool configs do not carry.

        Returns the workspace tool ids attached to the agent once the sync is done.
        """
        remote_by_name = {tool.name: tool for tool in agent.client_tools if tool.name}
        if not _tool_mismatches(tools, agent.client_tools):
            return agent.tool_ids

        # Convert every local tool into the ElevenLabs tool config dialect before the first REST
        # call: a validation failure must cost zero workspace mutations. Validating inside the create
        # loop would orphan the tools already created (created but never attached, since the
        # re-point at the end never runs) and surface offending tools one connect at a time.
        problems: list[str] = []
        configs: dict[str, dict[str, Any]] = {}
        for tool in tools:
            try:
                configs[tool.name] = _tool_config(tool, remote_by_name.get(tool.name))
            except UserError as error:
                problems.append(str(error))
        if problems:
            raise UserError('\n'.join(problems)) from None

        workspace_tools = await self._workspace_tool_listing()

        attached_ids = agent.tool_ids
        client_id_by_name: dict[str, str] = {}
        # Sort the agent's attached ids into two buckets. Client tools are the run's responsibility:
        # their name-to-id mapping is kept so a differing one can be updated in place (the agent's
        # inline tool configs carry no ids). Everything else is preserved as-is: server-side
        # (webhook/MCP/system) tools are owned by ElevenLabs and have no Pydantic AI counterpart, and
        # an attached id the listing doesn't report can't be classified at all, so it is kept rather
        # than risk dropping a server-side tool. `preserved_ids` seeds `target_tool_ids`, so these
        # survive the re-point; a client tool the run doesn't define lands in neither bucket and is
        # detached by it.
        preserved_ids: list[str] = []
        for tool_id in attached_ids:
            workspace_tool = workspace_tools.get(tool_id)
            if workspace_tool is None or workspace_tool.tool_config.type != 'client':
                preserved_ids.append(tool_id)
            elif workspace_tool.tool_config.name:  # pragma: no branch
                client_id_by_name[workspace_tool.tool_config.name] = tool_id

        # A sync that failed between a create and the final re-point leaves its created tools in the
        # workspace but unattached; recreating them on the retry would accumulate duplicates. An
        # unattached client tool whose config already matches exactly AND which no agent in the
        # workspace depends on is adopted instead. The dependents check matters because "not attached
        # to this agent" is not "orphaned": adopting another agent's matching tool would let a later
        # sync's PATCH rewrite it under that agent. A tool that differs, or that anything depends on,
        # is left alone and a fresh one is created.
        attached_id_set = set(attached_ids)
        adoptable_by_name: dict[str, list[_WorkspaceTool]] = {}
        for workspace_tool in workspace_tools.values():
            name = workspace_tool.tool_config.name
            if workspace_tool.id not in attached_id_set and workspace_tool.tool_config.type == 'client' and name:
                adoptable_by_name.setdefault(name, []).append(workspace_tool)

        # The ids the agent is re-pointed at: preserved server-side ids first, then one id per local
        # tool (adopted, created, or existing) in the run's order.
        target_tool_ids = list(preserved_ids)
        for tool in tools:
            config = configs[tool.name]
            remote = remote_by_name.get(tool.name)
            existing_id = client_id_by_name.get(tool.name)
            if remote is None or existing_id is None:
                target_tool_ids.append(await self._adopt_or_create_tool(tool, config, adoptable_by_name))
                continue
            if _tool_mismatches([tool], [remote]):
                if await self._tool_used_elsewhere(existing_id):
                    # Updating it would change the other agents' tool too: this agent gets one of its own.
                    target_tool_ids.append(await self._create_tool(config))
                    continue
                with _map_rest_errors(self.agent_id):
                    response = await self._http_client.patch(
                        f'{self._provider.base_url}/v1/convai/tools/{existing_id}',
                        headers={**self._rest_headers, 'Content-Type': 'application/json'},
                        content=to_json({'tool_config': config}).decode(),
                    )
                _raise_for_status(response, self.agent_id)
            target_tool_ids.append(existing_id)

        if target_tool_ids != attached_ids:
            with _map_rest_errors(self.agent_id):
                response = await self._http_client.patch(
                    f'{self._provider.base_url}/v1/convai/agents/{self.agent_id}',
                    headers={**self._rest_headers, 'Content-Type': 'application/json'},
                    content=to_json(
                        {'conversation_config': {'agent': {'prompt': {'tool_ids': target_tool_ids}}}}
                    ).decode(),
                )
            _raise_for_status(response, self.agent_id)
        return target_tool_ids

    async def _adopt_or_create_tool(
        self, tool: ToolDefinition, config: dict[str, Any], adoptable_by_name: dict[str, list[_WorkspaceTool]]
    ) -> str:
        """The id of an unattached matching tool no other agent uses (left by a failed sync), or of a new one."""
        for candidate in adoptable_by_name.get(tool.name, []):
            if not _tool_mismatches([tool], [candidate.tool_config]) and not await self._tool_used_elsewhere(
                candidate.id
            ):
                return candidate.id
        return await self._create_tool(config)

    async def _restricted_tool_ids(self, attached_ids: list[str], allowed: set[str]) -> list[str]:
        """The attached tool ids a conversation restricted to the `allowed` function tools may use.

        Client tools outside `allowed` are left out; server-side (webhook/MCP/system) tools, and ids
        the workspace listing doesn't report, are kept, as they are ElevenLabs' rather than the run's.
        """
        workspace_tools = await self._workspace_tool_listing()
        restricted: list[str] = []
        for tool_id in attached_ids:
            workspace_tool = workspace_tools.get(tool_id)
            if (
                workspace_tool is None
                or workspace_tool.tool_config.type != 'client'
                or workspace_tool.tool_config.name in allowed
            ):
                restricted.append(tool_id)
        return restricted

    # --- Configuration assembly -------------------------------------------------------------------

    def _check_settings(self, settings: ElevenLabsAgentsRealtimeModelSettings) -> None:
        """Reject shared settings whose intent the agent WebSocket cannot honor."""
        if settings.get('reconnect') is not None:
            raise UserError(
                'ElevenLabs Agents conversations are not resumed after a drop, so a `reconnect` policy '
                'would silently reconnect into a conversation that remembers nothing. Remove the `reconnect` '
                'policy and open a new session instead.'
            )
        if 'turn_detection' in settings and settings['turn_detection'] is not True:
            raise UserError(
                "ElevenLabs Agents own turn-taking server-side (the agent's turn model is always on), so "
                '`turn_detection` cannot be disabled or configured per conversation. Remove the setting, or '
                "configure turn-taking on the agent's `conversation_config.turn` instead."
            )
        if 'input_transcription_model' in settings and settings['input_transcription_model'] is None:
            raise UserError(
                'ElevenLabs Agents cannot disable input transcription per conversation (ASR drives the '
                'agent pipeline), so `input_transcription_model=None` cannot be honored. Remove the setting.'
            )

    @staticmethod
    def _requested_overrides(
        instructions: str, settings: ElevenLabsAgentsRealtimeModelSettings, tool_ids: list[str] | None
    ) -> list[tuple[tuple[str, ...], Any, str]]:
        """The `conversation_config_override` entries the session asks for, each with its source."""
        requested: list[tuple[tuple[str, ...], Any, str]] = []
        if tool_ids is not None:
            requested.append((('agent', 'prompt', 'tool_ids'), tool_ids, 'the `tool_choice` setting'))
        if instructions:
            # When the Pydantic AI agent defines instructions, the prompt override MUST
            # be permitted; without instructions the ElevenLabs-side prompt is inherited silently.
            requested.append((('agent', 'prompt', 'prompt'), instructions, 'the agent instructions'))
        if (llm := settings.get('elevenlabs_agents_llm')) is not None:
            requested.append((('agent', 'prompt', 'llm'), llm, 'the `elevenlabs_agents_llm` setting'))
        if (first_message := settings.get('elevenlabs_agents_first_message')) is not None:
            requested.append(
                (('agent', 'first_message'), first_message, 'the `elevenlabs_agents_first_message` setting')
            )
        if (language := settings.get('elevenlabs_agents_language')) is not None:
            requested.append((('agent', 'language'), language, 'the `elevenlabs_agents_language` setting'))
        if (voice_id := settings.get('elevenlabs_agents_voice_id')) is not None:
            requested.append((('tts', 'voice_id'), voice_id, 'the `elevenlabs_agents_voice_id` setting'))
        for key, value in settings.get('elevenlabs_agents_tts', {}).items():
            requested.append((('tts', key), value, f'the `elevenlabs_agents_tts` {key!r} setting'))
        if settings.get('output_modality', 'audio') == 'text':
            requested.append((('conversation', 'text_only'), True, "`output_modality='text'`"))
        return requested

    def _check_override_permissions(
        self,
        agent: _AgentPreflight,
        settings: ElevenLabsAgentsRealtimeModelSettings,
        requested: Sequence[tuple[tuple[str, ...], Any, str]],
    ) -> None:
        """Raise a `UserError` naming the toggle for a requested override the agent doesn't permit.

        Also for one the agent would ignore: in Realtime voice mode, ElevenLabs picks the LLM and TTS model itself.
        """
        realtime_model = agent.conversation_config.realtime_model if agent.conversation_config is not None else None
        for path, _, source in requested:
            if realtime_model is not None and path in _REALTIME_MODE_MANAGED_OVERRIDES:
                raise UserError(
                    f'The ElevenLabs agent {self.agent_id!r} is in Realtime voice mode ({realtime_model!r}), where '
                    f'ElevenLabs manages the `{".".join(path)}` setting and ignores the override required by {source}. '
                    'Remove the setting, or switch the agent to Custom voice mode in the ElevenLabs dashboard.'
                )
            if not _override_allowed(agent.override_permissions, path):
                dotted = '.'.join(path)
                raise UserError(
                    f'The ElevenLabs agent {self.agent_id!r} does not permit the `{dotted}` conversation '
                    f'override required by {source}. Enable the override toggle at '
                    f'`platform_settings.overrides.conversation_config_override.{dotted}` '
                    "(Security > Overrides in the agent's settings in the ElevenLabs dashboard), or remove the "
                    'conflicting input.'
                )
        if (
            settings.get('elevenlabs_agents_custom_llm_extra_body') is not None
            and not agent.custom_llm_extra_body_allowed
        ):
            raise UserError(
                f'The ElevenLabs agent {self.agent_id!r} does not permit `custom_llm_extra_body`. '
                'Enable the toggle at `platform_settings.overrides.custom_llm_extra_body` '
                "(Security > Overrides in the agent's settings), or remove the "
                '`elevenlabs_agents_custom_llm_extra_body` setting.'
            )

    def _initiation_payload(
        self,
        *,
        instructions: str,
        agent: _AgentPreflight,
        settings: ElevenLabsAgentsRealtimeModelSettings,
        tool_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Build `conversation_initiation_client_data`, preflight-checking every override's toggle."""
        requested = self._requested_overrides(instructions, settings, tool_ids)
        self._check_override_permissions(agent, settings, requested)
        overrides: dict[str, Any] = {}
        for path, value, _ in requested:
            _set_override(overrides, path, value)
        if config_override := settings.get('elevenlabs_agents_config_override'):
            _deep_merge(overrides, config_override)

        payload: dict[str, Any] = {
            'type': 'conversation_initiation_client_data',
            # ElevenLabs' `source` enum names its own SDKs and integrations only.
            'source_info': {'source': 'unknown', 'version': get_user_agent()},
        }
        if overrides:
            payload['conversation_config_override'] = overrides
        if (extra_body := settings.get('elevenlabs_agents_custom_llm_extra_body')) is not None:
            payload['custom_llm_extra_body'] = extra_body
        if (dynamic_variables := settings.get('elevenlabs_agents_dynamic_variables')) is not None:
            payload['dynamic_variables'] = dynamic_variables
        if (user_id := settings.get('elevenlabs_agents_user_id')) is not None:
            payload['user_id'] = user_id
        return payload

    @asynccontextmanager
    async def connect(
        self,
        *,
        messages: Sequence[ModelMessage],
        model_settings: RealtimeModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncGenerator[ElevenLabsAgentsRealtimeConnection]:
        settings = cast('ElevenLabsAgentsRealtimeModelSettings', self._merge_model_settings(model_settings) or {})
        self._check_settings(settings)
        instructions = get_instructions(messages, model_request_parameters) or ''
        tools = model_request_parameters.function_tools or []
        advertised_tools, tool_choice = resolve_advertised_tools(tools, settings.get('tool_choice'))
        if (tool_choice[0] if isinstance(tool_choice, tuple) else tool_choice) == 'required':
            raise UserError(
                "A realtime session can't force a tool call: ElevenLabs Agents have no per-response tool "
                "choice, so `tool_choice='required'` or a list of tool names would never let the agent "
                'answer. To restrict which tools the agent can use, pass `ToolOrOutput(function_tools=[...])` '
                'instead.'
            )

        # One REST round-trip yields the override-permission allowlist, the client tools, and the
        # agent's auth mode, so every fail-loudly check happens before the socket is dialed.
        agent = await self._fetch_agent()
        if not agent.sends_context_usage and not self._warned_context_usage:
            self._warned_context_usage = True
            warnings.warn(_CONTEXT_USAGE_MISSING.format(agent_id=self.agent_id), UserWarning, stacklevel=3)
        # The agent's client tools are reconciled with every tool of the run, as they are workspace
        # state shared with every conversation; a `tool_choice` restriction is this conversation's, and
        # narrows the attached tools through the `tool_ids` override below.
        tool_sync = settings.get('elevenlabs_agents_tool_sync', 'error')
        attached_ids = agent.tool_ids
        if tool_sync == 'error':
            if mismatches := _tool_mismatches(tools, agent.client_tools):
                details = '\n'.join(f'- {mismatch}' for mismatch in mismatches)
                raise UserError(
                    f'The tools of the ElevenLabs agent {self.agent_id!r} do not match this agent run:\n'
                    f'{details}\n'
                    "Update the agent's client tools to match, opt into `elevenlabs_agents_tool_sync='sync'` to "
                    "sync them from Pydantic AI, or set `elevenlabs_agents_tool_sync='off'` to trust the agent."
                )
        elif tool_sync == 'sync':
            # The sync changes workspace state, so everything else that can refuse the conversation is checked
            # first. Synced, the agent's client tools are the run's, so a `tool_choice` restriction needs the
            # `tool_ids` override exactly when it leaves out one of them (whatever ids it then names).
            restricts = (tool_choice == 'none' or isinstance(tool_choice, tuple)) and bool(
                {tool.name for tool in tools} - {tool.name for tool in advertised_tools}
            )
            self._check_override_permissions(
                agent, settings, self._requested_overrides(instructions, settings, [] if restricts else None)
            )
            attached_ids = await self._sync_tools(agent, tools)
        elif tool_sync == 'off':
            pass
        else:
            assert_never(tool_sync)

        restricted_tool_ids: list[str] | None = None
        if tool_choice == 'none' or isinstance(tool_choice, tuple):
            # Judged against the agent's attached tools rather than the run's: with `'off'`, the agent can
            # have client tools the run doesn't define, which the choice leaves out all the same.
            allowed = await self._restricted_tool_ids(attached_ids, {tool.name for tool in advertised_tools})
            if allowed != attached_ids:
                restricted_tool_ids = allowed
        payload = self._initiation_payload(
            instructions=instructions, agent=agent, settings=settings, tool_ids=restricted_tool_ids
        )
        handshake_timeout = settings.get('handshake_timeout', 30.0)
        url = await self._signed_url()
        headers: dict[str, str] = {}
        # Propagate trace context over the handshake (see the OpenAI provider for the rationale).
        inject_trace_context(headers)

        ws: ClientConnection | None = None
        # Dialed through the context manager protocol (not by awaiting `connect(...)` directly) so
        # tests can substitute a plain async context manager for the transport, matching the sibling
        # providers.
        opening = websockets.connect(url, additional_headers=headers)
        try:
            with _map_connect_errors(self.agent_id):
                ws = await opening.__aenter__()
                await ws.send(to_json(payload).decode())
                metadata = await _expect_handshake(ws, timeout=handshake_timeout)
            profile = self.profile
            _validate_audio_format(
                metadata.user_input_audio_format,
                expected_rate=profile.get('audio_input_sample_rate', 16000),
                direction='input',
                model_name=self.agent_id,
            )
            _validate_audio_format(
                metadata.agent_output_audio_format,
                expected_rate=profile.get('audio_output_sample_rate', 16000),
                direction='output',
                model_name=self.agent_id,
            )
            yield ElevenLabsAgentsRealtimeConnection(
                ws,
                conversation_id=metadata.conversation_id,
                text_output=settings.get('output_modality', 'audio') == 'text',
                strip_audio_tags=settings.get('elevenlabs_agents_strip_audio_tags', True),
            )
        finally:
            if ws is not None:
                await opening.__aexit__(None, None, None)


class _CascadeLifecycle(InferredLifecycle):
    """The inferred lifecycle of a cascaded pipeline, which replies to speech only once it has transcribed it.

    ElevenLabs' speech recognition hands the agent's LLM the user's words, and sends their transcript, before
    the LLM answers them. So audio with no transcript is never what a reply answers: microphone audio streamed
    before the agent's greeting (a reply nobody asked for) makes no spoken turn ahead of it.
    """

    transcripts_lag_replies = False
    untranscribed_audio_is_a_turn = False


class ElevenLabsAgentsRealtimeConnection(RealtimeConnection):
    """A live WebSocket connection to an ElevenLabs Agents conversation."""

    # `OSError` covers the socket-level failures (reset, broken pipe) that `websockets` lets through.
    transport_errors = (websockets.WebSocketException, OSError)

    def __init__(
        self,
        ws: ClientConnection,
        *,
        conversation_id: str | None = None,
        text_output: bool = False,
        strip_audio_tags: bool = True,
    ) -> None:
        self._ws = ws
        self._conversation_id = conversation_id
        self._context_limit_tokens: int | None = None
        self._text_output = text_output
        # The streamed text of the response under way, with its audio tags removed (see `_AudioTagStripper`).
        self._tag_stripper = _AudioTagStripper() if strip_audio_tags else None
        # Calls the agent fired without expecting a result (`expects_response=false`): the session
        # still settles them locally, but their `ToolResult` must never go back on the wire.
        self._fire_and_forget_tool_call_ids: set[str] = set()
        # Whether the model has streamed response output since the last turn boundary, so a boundary
        # event never finalizes a turn that produced nothing.
        self._response_open = False
        # The agent message the open response streams, once its text parts have named it, and the turn it is in.
        self._open_response_id: str | None = None
        self._open_event_id: int | None = None
        # What ElevenLabs said the user heard of the agent messages, by `agent_response_correction`, for the frame
        # under way to record on them, whether their responses are under way or over.
        self._cuts: list[OutputItemCut] = []
        self._turn_interrupted = False
        self._tail: _AudioTail | None = None
        # The MCP calls ElevenLabs runs itself, reported as native tool parts: those whose call part is out, and
        # those denied whose outcome ElevenLabs hasn't reported, which the response they belong to records.
        self._mcp_calls: set[str] = set()
        self._warned_mcp_approval = False
        self._calls_open = False
        """Whether the response under way made tool calls, and ends once no more come."""
        self._mcp_denied: dict[str, NativeToolReturnPart] = {}
        self._native_part_index = 0
        # Whether the conversation is waiting in the workspace's concurrency queue, where the server
        # plays hold audio over regular `audio` frames: that audio is no response of the agent's.
        self._queued = False
        # The responses, spoken turns, and inputs are worked out from the order of what is sent and
        # received. The protocol does carry ids (`event_id` per turn, `response_id` on the whole-text
        # `agent_response`, `in_response_to_ids`), but each arrives only with a response's *end*, while
        # a response has to be announced with its first audio, and a tool round shares one `event_id`
        # between the response that calls the tool and the one that answers its result.
        # The user's words are transcribed (and sent) before the agent's LLM answers them, never after.
        self._lifecycle = _CascadeLifecycle()
        # A barge-in on the response under way, which only its (truncated) terminal is yet to follow.
        self._deferred_cut: RealtimeResponseInterruptedEvent | None = None
        # Whether the lifecycle saw that barge-in before the cut-off response's terminal came.
        self._cut_shown = False

    @property
    def model_name(self) -> str | None:
        """Always `None`: ElevenLabs reports no realtime model id, so responses carry the agent id.

        The LLM behind the agent only surfaces in the `context_usage` report, which an agent sends only with it in
        its `client_events`, so it would name the model of some responses and not others, and the session
        falls back to [`ElevenLabsAgentsRealtimeModel.model_name`][pydantic_ai.realtime.elevenlabs_agents.ElevenLabsAgentsRealtimeModel.model_name].
        """
        return None

    @property
    def conversation_id(self) -> str | None:
        """The server-assigned conversation id, e.g. for post-hoc cost lookup via the conversations API."""
        return self._conversation_id

    @property
    def context_limit_tokens(self) -> int | None:
        """The context window of the LLM behind the agent, as last reported by a `context_usage` event.

        `None` until the agent reports one, which requires `context_usage` in its
        `conversation.client_events` list. It is a property of the conversation, not of a response,
        so it is kept here rather than summed into the session's usage, and the session's
        [`context_window_used`][pydantic_ai.realtime.RealtimeSession.context_window_used] is only known once such a
        report arrives.
        """
        return self._context_limit_tokens

    async def send(self, content: RealtimeInput) -> Sequence[LifecycleEvent]:
        """Send content to the ElevenLabs Agents conversation.

        Accepts `BinaryAudio` (raw PCM16 mono at the agent's input rate), a `str` text turn,
        `TextContext` (a `contextual_update`: context the agent reads from its next turn on, without
        interrupting or answering), and `ToolResult`. Manual turn-taking verbs, response
        cancellation, truncation, and images are not supported by the conversation WebSocket.
        """
        with self._lifecycle.sending(content) as input_id:
            sent = await self._send(content)
        if not sent:
            # A fire-and-forget call's result: the agent asked for none, so none will be answered.
            self._lifecycle.input_unanswerable(input_id)
        # ElevenLabs acknowledges nothing: where the input joined the conversation is known now.
        return self._lifecycle.take_pending()

    async def _send(self, content: RealtimeInput) -> bool:
        """Send `content` on the wire, returning whether anything was sent."""
        if isinstance(content, BinaryAudio):
            require_pcm_audio(content, provider_name=_PROVIDER_LABEL)
            # The one client message without a `type` discriminator.
            await self._send_event({'user_audio_chunk': base64.b64encode(content.data).decode('ascii')})
        elif isinstance(content, str):
            await self._send_event({'type': 'user_message', 'text': content})
        elif isinstance(content, TextContext):
            # Injected into the LLM's view of the conversation without a user turn, so the agent
            # neither stops speaking nor replies; the optional `context_id` (which lets a later update
            # supersede an earlier one) has no counterpart in `TextContext` and is not sent.
            await self._send_event({'type': 'contextual_update', 'text': content.text})
        elif isinstance(content, ToolResult):
            if content.tool_call_id in self._fire_and_forget_tool_call_ids:
                # The agent asked for fire-and-forget; the server tolerates a result sent anyway
                # (verified live), but the protocol contract is that none is expected, so it is
                # swallowed here.
                self._fire_and_forget_tool_call_ids.discard(content.tool_call_id)
                return False
            # Rendered before anything goes out, so a result that can't be delivered raises a
            # `UserError` rather than a transport error.
            result = self._render_tool_output(content)
            await self._send_event(
                {
                    'type': 'client_tool_result',
                    'tool_call_id': content.tool_call_id,
                    'result': result,
                    # A failed call, or one the model is asked to retry, as a standard run flags it on
                    # Anthropic. (`error_type` is left to ElevenLabs' default: its values classify where an
                    # integration's failure came from, which a Pydantic AI tool failure doesn't say.)
                    'is_error': content.failed,
                }
            )
        else:
            raise UserError(f'{_PROVIDER_LABEL} does not support {type(content).__name__} input.')
        return True

    def _render_tool_output(self, result: ToolResult) -> str:
        """Fold a tool result's text attachments into its string output; media cannot be delivered."""
        output = result.output
        if not result.content:
            return output
        text_content: list[str] = []
        for item in result.content:
            if isinstance(item, str):
                text_content.append(item)
            elif isinstance(item, TextContent):
                text_content.append(item.content)
            elif isinstance(item, CachePoint):
                continue
            elif isinstance(item, (ImageUrl, AudioUrl, DocumentUrl, VideoUrl, BinaryContent, UploadedFile)):
                raise UserError(
                    f'{_PROVIDER_LABEL} tool results are text-only, so `{type(item).__name__}` content '
                    'attached to a tool return cannot be delivered. Return text instead, or use a realtime '
                    'provider that supports tool-result media.'
                )
            else:
                assert_never(item)
        return '\n\n'.join(part for part in (output, *text_content) if part)

    async def _send_event(self, event: dict[str, Any]) -> None:
        await self._ws.send(to_json(event).decode())

    async def frames(self) -> AsyncIterator[RealtimeFrame]:
        read: asyncio.Task[str | bytes] | None = None
        try:
            while True:
                if read is None:
                    read = asyncio.create_task(self._ws.recv())
                # A reply whose text is complete waits only so long for more of its audio (see `_AUDIO_TAIL_GRACE`).
                # The read stays in flight across the wait: a cancelled read could drop the frame it holds.
                if self._calls_open:
                    timeout = _TOOL_CALL_BATCH_WINDOW
                elif self._tail is not None:
                    timeout = _AUDIO_TAIL_GRACE
                else:
                    timeout = None
                done, _ = await asyncio.wait({read}, timeout=timeout)
                if not done:
                    yield self._frame(self._end_calls() if self._calls_open else self._end_tail())
                    continue
                finished, read = read, None
                try:
                    raw = finished.result()
                except self.transport_errors as e:
                    yield self._connection_lost(e)
                    return
                if not isinstance(raw, str):
                    continue
                try:
                    # `ValueError` covers bad JSON, pydantic validation of a known frame's payload, and
                    # a bad base64 audio payload: a malformed frame shouldn't tear down the session
                    # (mirroring the OpenAI provider), so it surfaces as a recoverable error instead.
                    events = await self._map_event(_loads_obj(raw))
                except ValueError as e:
                    error = RealtimeSessionErrorEvent(
                        message=f'Failed to parse {_PROVIDER_LABEL} event: {e}', recoverable=True
                    )
                    yield self._lifecycle.frame_for([error])
                    continue
                except self.transport_errors as e:
                    # Answering a ping can hit an already-dropped socket; that is the same fatal
                    # condition as a failed receive.
                    yield self._connection_lost(e)
                    return
                if events or self._cuts:
                    yield self._frame(events)
        finally:
            if read is not None:
                await _utils.cancel_and_drain(read)

    def _frame(self, events: list[RealtimeCodecEvent]) -> RealtimeFrame:
        """The frame for the codec events one server message makes, with their lifecycle events."""
        frame = self._name_conversation(self._lifecycle_frame(events)) if events else []
        cuts, self._cuts = self._cuts, []
        # Last, so that a response the frame ends is recorded by the time its cut applies (one still under way gets
        # it when it is).
        return [*frame, *cuts]

    def _name_conversation(self, frame: RealtimeFrame) -> RealtimeFrame:
        """Name the conversation on every response the frame ends, as `_finalize_response` does on its terminal.

        A response ends without a terminal when it is cut off or the connection drops, and it is the conversation id
        that a consumer needs to look its cost up afterwards.
        """
        if self._conversation_id is None:
            return frame
        return [
            replace(
                event, provider_details={'conversation_id': self._conversation_id, **(event.provider_details or {})}
            )
            if isinstance(event, ResponseEnded)
            else event
            for event in frame
        ]

    def _lifecycle_frame(self, events: list[RealtimeCodecEvent]) -> RealtimeFrame:
        deferred, self._deferred_cut = self._deferred_cut, None
        ends_response = any(isinstance(event, ResponseDone) for event in events)
        if self._response_open and len(events) == 1 and isinstance(events[0], RealtimeResponseInterruptedEvent):
            # The user barged in on the response under way. The server answers with that response's
            # `agent_response`, cut down to what was heard (recorded live, right after the
            # interruption), which is still the cut-off response's: shown to the lifecycle now, the cut
            # would make that transcript a response of its own. So the application gets the barge-in
            # now, and the lifecycle once something other than that terminal follows.
            self._deferred_cut = events[0]
            return [*self._lifecycle.take_pending(), events[0]]
        if deferred is not None and not ends_response:
            # Something else came first (the user's words, say): the lifecycle needs the cut now, to tell
            # the speech that cut in from the turn being answered.
            self._cut_shown = True
            frame = self._lifecycle.frame_for([deferred, *events])
            return [event for event in frame if event is not deferred]
        if self._cut_shown and ends_response:
            # The cut-off response's terminal, after the lifecycle saw the cut: what it says was heard comes
            # after the response is over, so it ends the response, and its transcript trails it.
            self._cut_shown = False
            late = [event for event in events if isinstance(event, OutputTranscript)]
            frame = self._lifecycle.frame_for([event for event in events if not isinstance(event, OutputTranscript)])
            return [*frame, *(StaleEvent(event) for event in late)]
        return self._lifecycle.frame_for(events)

    def _connection_lost(self, error: Exception) -> RealtimeFrame:
        """The frame ending the conversation: no reconnect support, so a dropped conversation is fatal.

        It is surfaced as a non-recoverable error (mirroring the Gemini provider), so callers don't treat
        a truncated turn as complete, and everything under way on it is lost.
        """
        if isinstance(error, websockets.ConnectionClosed) and error.rcvd is not None and error.rcvd.code == 4300:
            message = (
                f"{_PROVIDER_LABEL} closed the conversation: it timed out in the workspace's concurrency queue "
                'before the agent was free. Retry later, or raise the workspace concurrency limit.'
            )
        else:
            message = f'{_PROVIDER_LABEL} connection closed: {error}'
        self._tail = None
        # The calls' response was over when their last one came; it isn't lost with the connection.
        ended = self._end_calls() if self._calls_open else []
        # A call denied for want of an approval was answered on the wire, so its return is in the response all the same.
        denied = [event for part in self._mcp_denied.values() for event in self._native_part(part)]
        self._mcp_denied.clear()
        return self._name_conversation(
            [
                *(self._lifecycle.frame_for(ended) if ended else []),
                *(self._lifecycle.frame_for(denied) if denied else []),
                *self._lifecycle.connection_lost(),
                RealtimeSessionErrorEvent(message=message, recoverable=False),
            ]
        )

    async def _map_event(self, data: dict[str, Any]) -> list[RealtimeCodecEvent]:
        """Map one server frame to codec events; pings are answered here and yield nothing."""
        event_type = data.get('type')
        if event_type == 'ping':
            await self._ws.send(_pong_frame(data))
            return []
        ended = self._end_calls() if self._calls_open and event_type != 'client_tool_call' else []
        if self._tail is not None and event_type in _TAIL_ENDING_FRAMES:
            ended += self._end_tail()
        return [*ended, *await self._map_frame(event_type, data)]

    def _end_calls(self) -> list[RealtimeCodecEvent]:
        """End the response whose tool calls are all in: the agent goes on once their results are."""
        self._calls_open = False
        # (ElevenLabs reports no usage per response: this one only ends it.)
        return [SessionUsage(usage=RequestUsage(), response_scoped=True)]

    async def _map_frame(self, event_type: object, data: dict[str, Any]) -> list[RealtimeCodecEvent]:
        if event_type == 'queue_status':
            status = _QueueStatusEvent.model_validate(data).queue_status_event.status
            # A `timed_out` conversation is closed with code 4300 next, which ends the stream.
            self._queued = status == 'waiting'
            return []
        if event_type == 'audio':
            # Parse (and decode) before marking the response open, so a malformed frame that is
            # reported as a recoverable error leaves the turn state untouched.
            payload = _AudioEvent.model_validate(data).audio_event
            # `validate=True` so corrupted base64 raises (a `ValueError` the receive loop reports as
            # a recoverable error) instead of silently decoding to garbage or empty bytes.
            audio = base64.b64decode(payload.audio_base_64, validate=True)
            if self._queued:
                # Hold audio the server plays while the conversation waits for a free slot.
                return []
            # Audio of another turn than the reply whose text is complete: that reply was voiced in full.
            ended = self._end_tail() if self._tail is not None and payload.event_id != self._tail.event_id else []
            self._response_open = True
            self._open_event_id = payload.event_id
            return [*ended, AudioDelta(data=audio)]
        if event_type == 'client_tool_call':
            call = _ClientToolCallEvent.model_validate(data).client_tool_call
            # Every call goes through, including one naming a tool this run does not define (possible
            # with `elevenlabs_agents_tool_sync='off'`, or after a dashboard edit that races the preflight):
            # the session's tool manager answers it with the standard unknown-tool error, so the call
            # and its outcome land in history and reach consumers like any other.
            # The calls end the response that makes them (the agent goes on in a response of its own once their
            # results are in, which the turn's `agent_response` then ends), so a barge-in from here on cuts nothing.
            self._response_open = False
            self._turn_interrupted = False
            if not call.expects_response:
                self._fire_and_forget_tool_call_ids.add(call.tool_call_id)
            if isinstance(call.parameters, str):
                args = call.parameters
            else:
                args = to_json(call.parameters or {}).decode()
            # Its response ends with the reply's last call (see `_TOOL_CALL_BATCH_WINDOW`), so that parallel calls are
            # one response, as the agent answers them.
            self._calls_open = True
            return [
                ToolCall(
                    tool_call_id=call.tool_call_id, tool_name=call.tool_name, args=args, response_usage_follows=True
                )
            ]
        if event_type == 'mcp_tool_call':
            return await self._map_mcp_tool_call(_McpToolCallEvent.model_validate(data).mcp_tool_call)
        if event_type == 'context_usage':
            usage = _ContextUsageEvent.model_validate(data).context_usage_event
            if usage.context_limit_tokens is not None:
                # The limit is a property of the conversation, not a per-response count: the
                # session sums every numeric `details` entry across turns, so it is exposed on the
                # connection instead of riding on the usage.
                self._context_limit_tokens = usage.context_limit_tokens
            # ElevenLabs reports LLM context consumption only: no output tokens and no credits reach
            # the WebSocket. Conversation cost appears post-hoc on `GET /v1/convai/conversations/{id}`.
            # Gated behind `client_events` (off by default); verified cadence: once per reply, with its
            # last audio, so it ends the reply (see `_TAIL_ENDING_FRAMES`) before it is counted: it isn't
            # attributed to a specific model response and accumulates into the run total only. The same report is the context
            # window fraction: `context_tokens` is the whole context the LLM holds, against the limit
            # (this frame's, or the last one reported when a frame omits it).
            context_tokens = usage.context_tokens or 0
            context_window_used = context_tokens / self._context_limit_tokens if self._context_limit_tokens else None
            return [
                SessionUsage(
                    usage=RequestUsage(input_tokens=context_tokens),
                    response_scoped=False,
                    context_window_used=context_window_used,
                )
            ]
        if event_type == 'client_error':
            error = _ClientErrorEvent.model_validate(data).error_event
            message = error.message or error.error_name or 'unknown error'
            return [RealtimeSessionErrorEvent(message=f'{_PROVIDER_LABEL} error: {message}', recoverable=True)]
        return self._map_turn_event(event_type, data)

    async def _map_mcp_tool_call(self, call: _McpToolCallPayload) -> list[RealtimeCodecEvent]:
        """A step of an MCP call ElevenLabs runs itself: its call part, a request for approval, or its outcome."""
        events: list[RealtimeCodecEvent] = []
        # Named as every model names a provider-run MCP call, so a consumer tells it from other native tools.
        tool_name = ':'.join([MCPServerTool.kind, call.service_id]) if call.service_id else MCPServerTool.kind
        if call.tool_call_id not in self._mcp_calls:
            self._mcp_calls.add(call.tool_call_id)
            events += self._native_part(
                NativeToolCallPart(
                    tool_name=tool_name,
                    args={'action': 'call_tool', 'tool_name': call.tool_name, 'tool_args': call.parameters or {}},
                    tool_call_id=call.tool_call_id,
                    provider_name='elevenlabs',
                )
            )
        if call.state == 'awaiting_approval':
            # Nothing in a Pydantic AI session can approve a call ElevenLabs runs itself, and the agent waits for an
            # answer, so it is denied rather than left to stall the conversation until the approval times out.
            if not self._warned_mcp_approval:
                self._warned_mcp_approval = True
                # (Raised from the connection's receive loop, so there is no caller to point at.)
                warnings.warn(_MCP_APPROVAL_DENIED, UserWarning, stacklevel=1)
            await self._send_event(
                {'type': 'mcp_tool_approval_result', 'tool_call_id': call.tool_call_id, 'is_approved': False}
            )
            self._mcp_denied[call.tool_call_id] = NativeToolReturnPart(
                tool_name=tool_name,
                content=_MCP_APPROVAL_DENIED,
                tool_call_id=call.tool_call_id,
                provider_name='elevenlabs',
                outcome='denied',
            )
        elif call.state in ('success', 'failure'):
            denied = self._mcp_denied.pop(call.tool_call_id, None)
            if call.state == 'success':
                outcome, content = 'success', call.result
            else:
                # A call denied for want of an approval is reported as failed: it stays the denial it was.
                outcome, content = ('denied', denied.content) if denied is not None else ('failed', call.error_message)
            events += self._native_part(
                NativeToolReturnPart(
                    tool_name=tool_name,
                    content=content,
                    tool_call_id=call.tool_call_id,
                    provider_name='elevenlabs',
                    outcome=outcome,
                )
            )
        # (`loading` says nothing more than the call part.)
        return events

    def _native_part(self, part: NativeToolCallPart | NativeToolReturnPart) -> list[RealtimeCodecEvent]:
        self._response_open = True
        index = self._native_part_index
        self._native_part_index += 1
        return [PartStartEvent(index=index, part=part), PartEndEvent(index=index, part=part)]

    def _map_turn_event(self, event_type: object, data: dict[str, Any]) -> list[RealtimeCodecEvent]:
        """Map the frames making up the agent's turn: transcripts, the boundary, and interruptions."""
        if event_type == 'agent_chat_response_part':
            # Text-only conversations stream the response as start/delta/stop parts ahead of the
            # whole-text `agent_response` (verified live; these arrive regardless of the agent's
            # `client_events` set). Deltas map incrementally; the empty start/stop markers carry
            # nothing (a tool-call-only model response is exactly one empty start/stop pair).
            part = _AgentChatResponsePartEvent.model_validate(data).text_response_part
            if part.type == 'start':
                # (A message's `stop` part can trail its `agent_response`, so only its `start` names the open one.)
                self._open_response_id = part.response_id
                self._open_event_id = part.event_id
            if part.type == 'delta' and part.text:
                self._response_open = True
                text = part.text if self._tag_stripper is None else self._tag_stripper.feed(part.text)
                if text:
                    # Named by its message, which a correction names to say what of it the user heard.
                    return [
                        OutputTranscript(
                            text=text, is_final=False, output_text=self._text_output, item_id=part.response_id
                        )
                    ]
            return []
        if event_type == 'agent_response':
            # The whole response text, once per reply, after any tool loop. In text mode it follows the streamed
            # parts and ends the reply; the session reconciles the full text against them. In audio mode it arrives
            # when the LLM finishes writing, while TTS can still be voicing the reply (verified live), so the reply
            # ends with its last audio instead (see `_AudioTail`). A reply already cut off has no more audio coming:
            # this is its text, cut down to what was heard. (Audio's `is_final` marks the end of one TTS generation,
            # which can come mid-reply, so it isn't read.)
            payload = _AgentResponseEvent.model_validate(data).agent_response_event
            self._response_open = True
            self._open_event_id = payload.event_id
            text, details = payload.agent_response, None
            if self._tag_stripper is not None:
                # The whole text, stripped at once, which is what the stripped pieces streamed before it add up to.
                self._tag_stripper = _AudioTagStripper()
                text, tags = _strip_audio_tags(text)
                details = {'audio_tags': tags} if tags else None
            transcript = OutputTranscript(
                text=text, is_final=True, output_text=self._text_output, item_id=payload.response_id
            )
            if self._text_output or self._turn_interrupted:
                return [
                    transcript,
                    *self._finalize_response(
                        interrupted=self._turn_interrupted,
                        provider_response_id=payload.response_id,
                        provider_details=details,
                    ),
                ]
            self._tail = _AudioTail(
                event_id=payload.event_id, response_id=payload.response_id, provider_details=details
            )
            return [transcript]
        if event_type == 'agent_response_correction':
            return self._map_correction(data)
        if event_type == 'agent_response_complete':
            return self._map_complete(data)
        if event_type == 'user_transcript':
            payload = _UserTranscriptEvent.model_validate(data).user_transcription_event
            # One final transcript per utterance; no tentative variants were observed live even with
            # `tentative_user_transcript` enabled in `client_events`.
            return [InputTranscript(text=payload.user_transcript, is_final=True)]
        if event_type == 'interruption':
            # The user barged in and the server stops streaming audio. Recorded live: while the reply
            # is still being generated, the `agent_response` that follows is already cut down to what
            # was heard (no correction frame); during playback of a finished reply, the server sends
            # an `agent_response_correction` instead. Only an interruption that lands while the
            # response is still open marks the turn interrupted: during playback the reply has already
            # ended with its last audio, and the flag must not leak into the next turn.
            if self._response_open:
                self._turn_interrupted = True
            if (tail := self._tail) is not None:
                # Cut off while its audio was still coming: the correction that follows ends it.
                tail.interrupted = True
            return [RealtimeResponseInterruptedEvent()]
        # Server-side tool/MCP progress, VAD scores, guardrail notices, response metadata, tentative
        # responses, and anything newer than this adapter are informational; ignore them rather than
        # failing the stream.
        return []

    def _map_complete(self, data: dict[str, Any]) -> list[RealtimeCodecEvent]:
        """End the reply of the turn `agent_response_complete` names, if it is still under way."""
        # Gated behind `client_events` (off by default). Verified live: in text mode it arrives right after the
        # `agent_response` that ended the turn's reply, and in audio mode once the turn's audio has played out,
        # long after its reply ended with its last audio; never for a turn cut off. So it is normally a dropped
        # no-op, kept as a boundary for agents whose `client_events` omit `agent_response`. It names its turn, and
        # ends the reply only if that is the reply's turn: a later turn's reply can be under way by then.
        complete = _AgentResponseCompleteEvent.model_validate(data).agent_response_complete_event
        if (tail := self._tail) is not None and complete.event_id in (None, tail.event_id):
            return self._end_tail()
        if tail is None and complete.event_id in (None, self._open_event_id):
            return self._finalize_response(interrupted=self._turn_interrupted)
        return []

    def _map_correction(self, data: dict[str, Any]) -> list[RealtimeCodecEvent]:
        """End the reply a barge-in cut off with what the user heard of it, as `agent_response_correction` says."""
        # After an interruption the server cuts each message of the turn the user didn't hear in full down to
        # what was heard: the one cut off ends in `...`, later ones are `''`. The streamed transcript can't be
        # shrunk through the codec (a final `OutputTranscript` only extends), so the message's part in history is
        # cut by an `OutputItemCut` instead, which applies whether its response is under way or, as usual in
        # audio mode (synthesis outruns playback), already over. A response still under way also ends here, with
        # the corrected text in its `provider_details` if the correction names it: one naming an earlier message
        # of the turn says nothing about the response under way.
        correction = _AgentResponseCorrectionEvent.model_validate(data).agent_response_correction_event
        corrected = correction.corrected_agent_response
        if self._tag_stripper is not None:
            corrected, _ = _strip_audio_tags(corrected)
        details = {'corrected_agent_response': corrected}
        if correction.response_id is not None:
            # History records what the user heard of the message the correction names, its response over or not.
            self._cuts.append(OutputItemCut(item_id=correction.response_id, heard_transcript=corrected))
        if (tail := self._tail) is not None:
            if None not in (correction.response_id, tail.response_id) and correction.response_id != tail.response_id:
                # An earlier message's: the reply being voiced gets a correction of its own in the same batch, or
                # ends with its audio as the interruption left it.
                return []
            # The reply was still being voiced when the user cut in: the correction ends it, with what was heard.
            tail.interrupted = True
            tail.provider_details = {**(tail.provider_details or {}), **details}
            return self._end_tail()
        # The turn was cut off, so the response under way ends; it keeps the correction only if it is the one named.
        names_open = correction.response_id is None or correction.response_id == self._open_response_id
        return self._finalize_response(interrupted=True, provider_details=details if names_open else None)

    def _end_tail(self) -> list[RealtimeCodecEvent]:
        """End the reply whose text is complete: its audio, if any more was coming, won't."""
        tail, self._tail = self._tail, None
        assert tail is not None
        return self._finalize_response(
            interrupted=tail.interrupted, provider_response_id=tail.response_id, provider_details=tail.provider_details
        )

    def _finalize_response(
        self,
        *,
        interrupted: bool,
        provider_response_id: str | None = None,
        provider_details: dict[str, Any] | None = None,
    ) -> list[RealtimeCodecEvent]:
        """Close the open turn with a `ResponseDone`; a boundary with nothing streamed is dropped."""
        self._turn_interrupted = False
        if not self._response_open:
            return []
        # A denied call whose outcome ElevenLabs never reported still gets its return, in the response it was in.
        denied = [event for part in self._mcp_denied.values() for event in self._native_part(part)]
        self._mcp_denied.clear()
        self._native_part_index = 0
        self._response_open = False
        self._open_response_id = None
        self._open_event_id = None
        # The conversation id rides on every response's `provider_details`: nothing billable reaches
        # the socket, so it is the key a consumer needs for the post-hoc cost lookup on the
        # conversations API, and the response is where it survives into persisted history. A
        # consumer holding the live connection can also read it from `conversation_id` directly.
        details = dict(provider_details or {})
        if self._conversation_id is not None:
            details.setdefault('conversation_id', self._conversation_id)
        return [
            *denied,
            ResponseDone(
                interrupted=interrupted, provider_response_id=provider_response_id, provider_details=details or None
            ),
        ]
