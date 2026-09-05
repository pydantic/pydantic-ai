"""ElevenLabs Agents provider for realtime speech-to-speech sessions.

ElevenLabs offers no direct realtime conversational model: its Agents platform runs a cascaded
pipeline (realtime ASR + a configurable LLM + ElevenLabs TTS + a server-side turn-taking model)
behind a pre-provisioned agent, and the only conversational API is the agent WebSocket
(`wss://api.elevenlabs.io/v1/convai/conversation?agent_id=...`). So unlike the other realtime
providers, the model here wraps a *hosted agent* addressed by its `agent_id`, and Pydantic AI takes
ownership of as much of the conversation as ElevenLabs allows per-conversation:

- Instructions, LLM choice, first message, language, voice, TTS knobs, and text-only mode are pushed
  through `conversation_initiation_client_data.conversation_config_override`. Every overridable field
  is gated by a per-field toggle in the agent's Security tab (`platform_settings.overrides`); the
  connect-time preflight fetches the agent and raises a [`UserError`][pydantic_ai.exceptions.UserError]
  naming the exact toggle when an override the session needs is not permitted. When the Pydantic AI
  agent defines no instructions, the ElevenLabs-side prompt is inherited silently.
- Tool declarations cannot be sent inline per-conversation: client tools are workspace entities
  referenced from the agent. By default the preflight *errors on mismatch* between the session's
  [`ToolDefinition`][pydantic_ai.tools.ToolDefinition]s and the agent's client tools; the opt-in
  `elevenlabs_tool_sync='sync'` setting instead creates/updates the workspace tools and re-points the
  agent's `tool_ids` over REST before dialing.
- Turn-taking, VAD, ASR configuration, audio formats, and platform settings (auth, privacy,
  guardrails, knowledge bases) stay owned by the ElevenLabs agent configuration.

Requires the `websockets` package, available via the `elevenlabs-realtime` optional group (REST
preflight uses Pydantic AI's own HTTP client, so no ElevenLabs SDK is needed):

    pip install "pydantic-ai-slim[elevenlabs-realtime]"
"""

from __future__ import annotations as _annotations

import asyncio
import base64
import json
import time
from collections.abc import AsyncGenerator, AsyncIterator, Generator, Sequence
from contextlib import asynccontextmanager, contextmanager
from dataclasses import KW_ONLY, dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

import httpx2
from pydantic import BaseModel, ConfigDict
from pydantic_core import to_json
from typing_extensions import TypedDict, assert_never

try:
    import websockets
    from websockets.asyncio.client import ClientConnection
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `websockets` package to use the ElevenLabs realtime model, '
        'you can use the `elevenlabs-realtime` optional group - '
        '`pip install "pydantic-ai-slim[elevenlabs-realtime]"`'
    ) from _import_error

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
    RealtimeResponseInterruptedEvent,
    RealtimeSessionErrorEvent,
    TextContent,
    UploadedFile,
    VideoUrl,
)
from ..models import ModelRequestParameters
from ..providers import infer_provider
from ..tools import ToolDefinition
from ..usage import RequestUsage
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
    ToolCall,
    ToolResult,
)
from .model import RealtimeError, RealtimeModel
from .profiles import RealtimeModelProfileSpec
from .settings import RealtimeModelSettings

if TYPE_CHECKING:
    import httpx

    from .._http import AsyncHTTPClient
    from ..providers.elevenlabs import ElevenLabsProvider

__all__ = (
    'ElevenLabsRealtimeModel',
    'ElevenLabsRealtimeModelSettings',
    'ElevenLabsRealtimeConnection',
    'ElevenLabsTTSSettings',
)

_PROVIDER_LABEL = 'ElevenLabs Agents'


class ElevenLabsTTSSettings(TypedDict, total=False):
    """Per-conversation TTS overrides, each gated by its own override toggle on the agent."""

    model_id: str
    """The ElevenLabs TTS model to synthesize with, e.g. `eleven_v3_conversational`."""
    stability: float
    """Voice stability (0-1); lower is more expressive, higher more consistent."""
    speed: float
    """Playback speed multiplier."""
    similarity_boost: float
    """Voice similarity boost (0-1)."""


class ElevenLabsRealtimeModelSettings(RealtimeModelSettings, total=False):
    """Settings specific to ElevenLabs Agents realtime sessions.

    Every `elevenlabs_*` field below that maps to a `conversation_config_override` entry requires the
    matching override toggle to be enabled in the agent's Security tab; the connect-time preflight
    raises [`UserError`][pydantic_ai.exceptions.UserError] naming the toggle otherwise.

    Of the shared settings, `output_modality='text'` maps to the `conversation.text_only` override,
    and `tool_choice` allow-lists restrict the tool set the preflight checks (or syncs). `max_tokens`,
    `parallel_tool_calls`, and `thinking` have no per-conversation surface on the agent WebSocket and
    are silently ignored, per the shared settings contract. `turn_detection` cannot be configured
    (ElevenLabs' server-side turn model is always on), `input_transcription_model=None` cannot be
    honored (ASR cannot be disabled per-conversation), and `reconnect` is unsupported (the WebSocket
    has no session resumption), so each of those raises rather than silently under-delivering.
    """

    elevenlabs_voice_id: str
    """Voice used for audio output (the `tts.voice_id` override)."""
    elevenlabs_language: str
    """Conversation language code, e.g. `de` (the `agent.language` override)."""
    elevenlabs_first_message: str
    """The greeting the agent opens with (the `agent.first_message` override)."""
    elevenlabs_llm: str
    """The LLM the ElevenLabs agent pipeline runs, e.g. `gpt-5.2` (the `agent.prompt.llm` override)."""
    elevenlabs_tts: ElevenLabsTTSSettings
    """TTS model and voice-delivery knobs (the `tts.*` overrides)."""
    elevenlabs_dynamic_variables: dict[str, str | int | float | bool]
    """Values substituted into `{{placeholders}}` in the agent's prompts and messages."""
    elevenlabs_custom_llm_extra_body: dict[str, Any]
    """Extra body merged into the agent's custom-LLM request. Gated by the agent's
    `custom_llm_extra_body` security toggle (a sibling of the override toggles)."""
    elevenlabs_user_id: str
    """Opaque end-user identifier attached to the conversation for ElevenLabs-side analytics."""
    elevenlabs_tool_sync: Literal['error', 'sync', 'off']
    """How to reconcile the session's tools with the agent's client tools at connect time.

    - `'error'` (default): raise [`UserError`][pydantic_ai.exceptions.UserError] describing every
      difference between the session's [`ToolDefinition`][pydantic_ai.tools.ToolDefinition]s and the
      agent's client tools, so the two can never silently disagree.
    - `'sync'`: make the workspace match Pydantic AI over REST before dialing: create missing client
      tools, update differing ones, and re-point the agent's `tool_ids` (dropping client tools the
      session doesn't define; ElevenLabs-side webhook/MCP/system tools are left untouched). This
      mutates workspace state shared by every conversation with the agent, which is why it is opt-in.
    - `'off'`: skip the check and trust the agent's configuration (e.g. for read-scoped API keys).
    """
    elevenlabs_config_override: dict[str, Any]
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


class _ConversationConfig(BaseModel):
    model_config = ConfigDict(extra='allow')

    agent: _AgentSectionConfig | None = None


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


class _AudioEvent(BaseModel):
    audio_event: _AudioPayload


class _AgentResponsePayload(BaseModel):
    model_config = ConfigDict(extra='allow')

    agent_response: str
    response_id: str | None = None


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


class _AgentChatResponsePartEvent(BaseModel):
    text_response_part: _TextResponsePart


class _AgentResponseCorrectionPayload(BaseModel):
    model_config = ConfigDict(extra='allow')

    original_agent_response: str | None = None
    corrected_agent_response: str


class _AgentResponseCorrectionEvent(BaseModel):
    agent_response_correction_event: _AgentResponseCorrectionPayload


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


class _ContextUsagePayload(BaseModel):
    model_config = ConfigDict(extra='allow')

    model: str | None = None
    context_tokens: int | None = None
    context_limit_tokens: int | None = None


class _ContextUsageEvent(BaseModel):
    model_config = ConfigDict(extra='allow')

    # Verified live: the payload arrives under `context_usage_event`; the top-level form is
    # tolerated defensively.
    context_usage_event: _ContextUsagePayload | None = None


class _ClientErrorPayload(BaseModel):
    model_config = ConfigDict(extra='allow')

    code: int | str | None = None
    error_name: str | None = None
    message: str | None = None


class _ClientErrorEvent(BaseModel):
    model_config = ConfigDict(extra='allow')

    # The AsyncAPI docs wrap the payload as `{"type": "client_error", "error_event": {...}}`; never
    # observed live (non-permitted overrides reject with a post-handshake 1008 close instead), so
    # a bare top-level payload is tolerated as a fallback for whatever the server does send.
    error_event: _ClientErrorPayload | None = None


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
            "types, or keep the tool configured in the ElevenLabs dashboard with `elevenlabs_tool_sync='off'`."
        )
    return result


def _convert_parameters(schema: dict[str, Any], *, path: str, root: bool, problems: list[str]) -> dict[str, Any]:
    """One node of `_elevenlabs_parameters`, recording problems instead of raising on the first."""
    result: dict[str, Any] = {}
    for key in ('type', 'description', 'enum'):
        if (value := schema.get(key)) is not None:
            result[key] = value
    if 'type' not in result:
        subject = f'parameter `{path}`' if path else 'the parameters schema'
        problems.append(
            f'{subject} uses a JSON Schema feature the ElevenLabs tool dialect cannot express '
            '(a union of types, or a recursive reference)'
        )
    if not root and not result.get('description'):
        problems.append(f'ElevenLabs requires a description for every tool parameter, and `{path}` has none')
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


def _tool_config(tool: ToolDefinition) -> dict[str, Any]:
    """Render a [`ToolDefinition`][pydantic_ai.tools.ToolDefinition] as an ElevenLabs client-tool config."""
    return {
        'type': 'client',
        'name': tool.name,
        'description': tool.description or '',
        # Pydantic AI's session always settles a call and sends its result back, so every synced tool
        # expects a response; the server's default `response_timeout_secs` (20s) is left in place.
        'expects_response': True,
        'parameters': _elevenlabs_parameters(tool.name, _prepared_parameters(tool.parameters_json_schema)),
    }


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
        except asyncio.TimeoutError:
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


async def _send_pong(ws: ClientConnection, ping_frame: dict[str, Any]) -> None:
    await ws.send(_pong_frame(ping_frame))


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
class ElevenLabsRealtimeModel(RealtimeModel):
    """ElevenLabs Agents realtime model, wrapping a hosted agent addressed by its `agent_id`.

    Args:
        agent_id: The ElevenLabs agent to converse with, e.g. `agent_0101k2...`. The agent must exist;
            Pydantic AI overrides its per-conversation configuration where the agent's Security-tab
            toggles allow, and fails loudly where they don't.
        provider: The provider to use for authentication and API access. Defaults to `'elevenlabs'`
            (reads `ELEVENLABS_API_KEY`); pass an
            [`ElevenLabsProvider`][pydantic_ai.providers.elevenlabs.ElevenLabsProvider] for a custom
            key, HTTP client, or regional (data-residency) base URL.
        settings: [Model settings][pydantic_ai.realtime.elevenlabs.ElevenLabsRealtimeModelSettings]
            used as defaults for realtime sessions.
        profile: Optional override for the [realtime model profile][pydantic_ai.realtime.RealtimeModelProfile],
            merged over the provider's. For ElevenLabs this is how per-agent audio formats are
            declared when the agent isn't on the default 16 kHz PCM formats.
    """

    agent_id: str
    _: KW_ONLY
    settings: RealtimeModelSettings | None = None
    _provider: ElevenLabsProvider = field(init=False, repr=False)

    # Written out rather than generated because `profile` has to be an init argument handed to the
    # base constructor, while `RealtimeModel.profile` stays the *resolved* profile, exactly as on a
    # standard `Model`.
    def __init__(
        self,
        agent_id: str,
        *,
        provider: ElevenLabsProvider | Literal['elevenlabs'] = 'elevenlabs',
        settings: RealtimeModelSettings | None = None,
        profile: RealtimeModelProfileSpec | None = None,
    ) -> None:
        super().__init__(settings=settings, profile=profile)
        self.agent_id = agent_id
        if isinstance(provider, str):
            provider = cast('ElevenLabsProvider', infer_provider(provider))
        if provider.name != 'elevenlabs':
            raise UserError(
                f"`ElevenLabsRealtimeModel` requires an `ElevenLabsProvider` or `provider='elevenlabs'`; "
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

    async def _sync_tools(self, agent: _AgentPreflight, tools: Sequence[ToolDefinition]) -> None:
        """Make the agent's client tools match the session's tools over REST (`elevenlabs_tool_sync='sync'`).

        Creates missing workspace client tools, updates differing ones (`PATCH /v1/convai/tools/{id}`
        takes the same `tool_config` body as create, verified live), and re-points the agent's
        `tool_ids` at the resulting set plus the agent's untouched server-side tools (agent PATCH is
        a partial update: only the provided leaves change, verified live). This mutates workspace
        state shared by every conversation with the agent. The PATCH channel is the only one: the
        toggle-gated `tool_ids` *override* at initiation cannot add tools, it only restricts to ids
        already attached to the agent (foreign ids close the socket with 1008, verified live).

        When nothing differs, no extra request is made. Otherwise the workspace tool listing is
        fetched once to resolve tool ids, which the agent's resolved tool configs do not carry.
        """
        remote_by_name = {tool.name: tool for tool in agent.client_tools if tool.name}
        if not _tool_mismatches(tools, agent.client_tools):
            return

        # Every config is built and validated before the first REST call: a validation failure must
        # cost zero workspace mutations. Validating inside the create loop orphaned the tools already
        # created (created but never attached, since the re-point at the end never ran) and surfaced
        # offending tools one connect at a time; a production deployment syncing seven tools hit both.
        problems: list[str] = []
        configs: dict[str, dict[str, Any]] = {}
        for tool in tools:
            try:
                configs[tool.name] = _tool_config(tool)
            except UserError as error:
                problems.append(str(error))
        if problems:
            raise UserError('\n'.join(problems)) from None

        # The listing is cursor-paginated; every page is fetched so attached ids are never
        # misclassified (and duplicated as fresh creates) just because a workspace holds more tools
        # than one page carries.
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

        attached_ids = agent.tool_ids
        client_id_by_name: dict[str, str] = {}
        # Server-side (webhook/MCP/system) tools stay attached untouched; so does any attached id the
        # listing doesn't report (never drop what can't be classified).
        preserved_ids: list[str] = []
        for tool_id in attached_ids:
            workspace_tool = workspace_tools.get(tool_id)
            if workspace_tool is None or workspace_tool.tool_config.type != 'client':
                preserved_ids.append(tool_id)
            elif workspace_tool.tool_config.name:  # pragma: no branch
                client_id_by_name[workspace_tool.tool_config.name] = tool_id

        tool_ids = preserved_ids
        for tool in tools:
            config = configs[tool.name]
            remote = remote_by_name.get(tool.name)
            existing_id = client_id_by_name.get(tool.name)
            if remote is None or existing_id is None:
                with _map_rest_errors(self.agent_id):
                    response = await self._http_client.post(
                        f'{self._provider.base_url}/v1/convai/tools',
                        headers={**self._rest_headers, 'Content-Type': 'application/json'},
                        content=to_json({'tool_config': config}).decode(),
                    )
                _raise_for_status(response, self.agent_id)
                tool_ids.append(_CreatedToolResponse.model_validate_json(response.content).id)
                continue
            if _tool_mismatches([tool], [remote]):
                with _map_rest_errors(self.agent_id):
                    response = await self._http_client.patch(
                        f'{self._provider.base_url}/v1/convai/tools/{existing_id}',
                        headers={**self._rest_headers, 'Content-Type': 'application/json'},
                        content=to_json({'tool_config': config}).decode(),
                    )
                _raise_for_status(response, self.agent_id)
            tool_ids.append(existing_id)

        if tool_ids != attached_ids:
            with _map_rest_errors(self.agent_id):
                response = await self._http_client.patch(
                    f'{self._provider.base_url}/v1/convai/agents/{self.agent_id}',
                    headers={**self._rest_headers, 'Content-Type': 'application/json'},
                    content=to_json({'conversation_config': {'agent': {'prompt': {'tool_ids': tool_ids}}}}).decode(),
                )
            _raise_for_status(response, self.agent_id)

    # --- Configuration assembly -------------------------------------------------------------------

    def _check_settings(self, settings: ElevenLabsRealtimeModelSettings) -> None:
        """Reject shared settings whose intent the agent WebSocket cannot honor."""
        if settings.get('reconnect') is not None:
            raise UserError(
                'ElevenLabs Agents conversations cannot be resumed after a drop, so a `reconnect` policy '
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
        instructions: str, settings: ElevenLabsRealtimeModelSettings
    ) -> list[tuple[tuple[str, ...], Any, str]]:
        """The `conversation_config_override` entries the session asks for, each with its source."""
        requested: list[tuple[tuple[str, ...], Any, str]] = []
        if instructions:
            # When the Pydantic AI agent defines instructions, the prompt override MUST
            # be permitted; without instructions the ElevenLabs-side prompt is inherited silently.
            requested.append((('agent', 'prompt', 'prompt'), instructions, 'the agent instructions'))
        if (llm := settings.get('elevenlabs_llm')) is not None:
            requested.append((('agent', 'prompt', 'llm'), llm, 'the `elevenlabs_llm` setting'))
        if (first_message := settings.get('elevenlabs_first_message')) is not None:
            requested.append((('agent', 'first_message'), first_message, 'the `elevenlabs_first_message` setting'))
        if (language := settings.get('elevenlabs_language')) is not None:
            requested.append((('agent', 'language'), language, 'the `elevenlabs_language` setting'))
        if (voice_id := settings.get('elevenlabs_voice_id')) is not None:
            requested.append((('tts', 'voice_id'), voice_id, 'the `elevenlabs_voice_id` setting'))
        for key, value in settings.get('elevenlabs_tts', {}).items():
            requested.append((('tts', key), value, f'the `elevenlabs_tts` {key!r} setting'))
        if settings.get('output_modality', 'audio') == 'text':
            requested.append((('conversation', 'text_only'), True, "`output_modality='text'`"))
        return requested

    def _initiation_payload(
        self,
        *,
        instructions: str,
        agent: _AgentPreflight,
        settings: ElevenLabsRealtimeModelSettings,
    ) -> dict[str, Any]:
        """Build `conversation_initiation_client_data`, preflight-checking every override's toggle."""
        permissions = agent.override_permissions
        requested = self._requested_overrides(instructions, settings)
        overrides: dict[str, Any] = {}
        for path, value, source in requested:
            if not _override_allowed(permissions, path):
                dotted = '.'.join(path)
                raise UserError(
                    f'The ElevenLabs agent {self.agent_id!r} does not permit the `{dotted}` conversation '
                    f'override required by {source}. Enable the override toggle at '
                    f'`platform_settings.overrides.conversation_config_override.{dotted}` '
                    "(the agent's Security tab in the ElevenLabs dashboard), or remove the conflicting input."
                )
            _set_override(overrides, path, value)
        if config_override := settings.get('elevenlabs_config_override'):
            _deep_merge(overrides, config_override)

        payload: dict[str, Any] = {'type': 'conversation_initiation_client_data'}
        if overrides:
            payload['conversation_config_override'] = overrides
        if (extra_body := settings.get('elevenlabs_custom_llm_extra_body')) is not None:
            if not agent.custom_llm_extra_body_allowed:
                raise UserError(
                    f'The ElevenLabs agent {self.agent_id!r} does not permit `custom_llm_extra_body`. '
                    'Enable the toggle at `platform_settings.overrides.custom_llm_extra_body` '
                    "(the agent's Security tab), or remove the `elevenlabs_custom_llm_extra_body` setting."
                )
            payload['custom_llm_extra_body'] = extra_body
        if (dynamic_variables := settings.get('elevenlabs_dynamic_variables')) is not None:
            payload['dynamic_variables'] = dynamic_variables
        if (user_id := settings.get('elevenlabs_user_id')) is not None:
            payload['user_id'] = user_id
        return payload

    @asynccontextmanager
    async def connect(
        self,
        *,
        messages: Sequence[ModelMessage],
        model_settings: RealtimeModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncGenerator[ElevenLabsRealtimeConnection]:
        settings = cast('ElevenLabsRealtimeModelSettings', self._merge_model_settings(model_settings) or {})
        self._check_settings(settings)
        instructions = get_instructions(messages, model_request_parameters) or ''
        # `'none'` and allow-list tool choices restrict the advertised set (like Gemini); declarative
        # `'auto'`/`'required'` have no per-conversation surface and are dropped by the resolution.
        advertised_tools, _ = resolve_advertised_tools(
            model_request_parameters.function_tools, settings.get('tool_choice')
        )

        # One REST round-trip yields the override-permission allowlist, the client tools, and the
        # agent's auth mode, so every fail-loudly check happens before the socket is dialed.
        agent = await self._fetch_agent()
        tool_sync = settings.get('elevenlabs_tool_sync', 'error')
        if tool_sync == 'error':
            if mismatches := _tool_mismatches(advertised_tools, agent.client_tools):
                details = '\n'.join(f'- {mismatch}' for mismatch in mismatches)
                raise UserError(
                    f'The tools of the ElevenLabs agent {self.agent_id!r} do not match this agent run:\n'
                    f'{details}\n'
                    "Update the agent's client tools to match, opt into `elevenlabs_tool_sync='sync'` to "
                    "sync them from Pydantic AI, or set `elevenlabs_tool_sync='off'` to trust the agent."
                )
        elif tool_sync == 'sync':
            await self._sync_tools(agent, advertised_tools)
        elif tool_sync == 'off':
            pass
        else:
            assert_never(tool_sync)

        payload = self._initiation_payload(instructions=instructions, agent=agent, settings=settings)
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
            yield ElevenLabsRealtimeConnection(
                ws,
                conversation_id=metadata.conversation_id,
                text_output=settings.get('output_modality', 'audio') == 'text',
            )
        finally:
            if ws is not None:
                await opening.__aexit__(None, None, None)


class ElevenLabsRealtimeConnection(RealtimeConnection):
    """A live WebSocket connection to an ElevenLabs Agents conversation."""

    # `OSError` covers the socket-level failures (reset, broken pipe) that `websockets` lets through.
    transport_errors = (websockets.WebSocketException, OSError)

    def __init__(
        self,
        ws: ClientConnection,
        *,
        conversation_id: str | None = None,
        text_output: bool = False,
    ) -> None:
        self._ws = ws
        self._conversation_id = conversation_id
        self._text_output = text_output
        # The LLM the agent pipeline reported serving this conversation (from `context_usage`);
        # the closest thing ElevenLabs has to a server-reported model id.
        self._llm_model_name: str | None = None
        # Calls the agent fired without expecting a result (`expects_response=false`): the session
        # still settles them locally, but their `ToolResult` must never go back on the wire.
        self._fire_and_forget_tool_call_ids: set[str] = set()
        # Whether the model has streamed response output since the last turn boundary, so a boundary
        # event never finalizes a turn that produced nothing.
        self._response_open = False
        self._turn_interrupted = False
        self._correction: _AgentResponseCorrectionPayload | None = None

    @property
    def model_name(self) -> str | None:
        return self._llm_model_name

    @property
    def conversation_id(self) -> str | None:
        """The server-assigned conversation id, e.g. for post-hoc cost lookup via the conversations API."""
        return self._conversation_id

    async def send(self, content: RealtimeInput) -> None:
        """Send content to the ElevenLabs Agents conversation.

        Accepts `BinaryAudio` (raw PCM16 mono at the agent's input rate), a `str` text turn, and
        `ToolResult`. Manual turn-taking verbs, response cancellation, truncation, and images are not
        supported by the conversation WebSocket.
        """
        if isinstance(content, BinaryAudio):
            require_pcm_audio(content, provider_name=_PROVIDER_LABEL)
            # The one client message without a `type` discriminator.
            await self._send_event({'user_audio_chunk': base64.b64encode(content.data).decode('ascii')})
        elif isinstance(content, str):
            await self._send_event({'type': 'user_message', 'text': content})
        elif isinstance(content, ToolResult):
            if content.tool_call_id in self._fire_and_forget_tool_call_ids:
                # The agent asked for fire-and-forget; the server tolerates a result sent anyway
                # (verified live), but the protocol contract is that none is expected, so it is
                # swallowed here.
                self._fire_and_forget_tool_call_ids.discard(content.tool_call_id)
                return
            await self._send_event(
                {
                    'type': 'client_tool_result',
                    'tool_call_id': content.tool_call_id,
                    'result': self._render_tool_output(content),
                    # The codec's `ToolResult` is the already-rendered success/retry text, with no
                    # error flag left to map, so results always report success here; the model reads
                    # the retry semantics from the rendered text itself, as on Gemini.
                    'is_error': False,
                }
            )
        else:
            raise UserError(f'{_PROVIDER_LABEL} does not support {type(content).__name__} input.')

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

    async def __aiter__(self) -> AsyncIterator[RealtimeCodecEvent]:
        while True:
            try:
                raw = await self._ws.recv()
            except self.transport_errors as e:
                # No reconnect support: a dropped conversation is fatal. Surface it as a
                # non-recoverable error and end the stream cleanly (mirroring the Gemini provider),
                # so callers don't treat a truncated turn as complete.
                yield RealtimeSessionErrorEvent(message=f'{_PROVIDER_LABEL} connection closed: {e}', recoverable=False)
                return
            if not isinstance(raw, str):
                continue
            try:
                # `ValueError` covers bad JSON, pydantic validation of a known frame's payload, and
                # a bad base64 audio payload: a malformed frame shouldn't tear down the session
                # (mirroring the OpenAI provider), so it surfaces as a recoverable error instead.
                events = await self._map_event(_loads_obj(raw))
            except ValueError as e:
                yield RealtimeSessionErrorEvent(
                    message=f'Failed to parse {_PROVIDER_LABEL} event: {e}', recoverable=True
                )
                continue
            except self.transport_errors as e:
                # Answering a ping can hit an already-dropped socket; that is the same fatal
                # condition as a failed receive.
                yield RealtimeSessionErrorEvent(message=f'{_PROVIDER_LABEL} connection closed: {e}', recoverable=False)
                return
            for event in events:
                yield event

    async def _map_event(self, data: dict[str, Any]) -> list[RealtimeCodecEvent]:
        """Map one server frame to codec events; pings are answered here and yield nothing."""
        event_type = data.get('type')
        if event_type == 'ping':
            await _send_pong(self._ws, data)
            return []
        if event_type == 'audio':
            # Parse (and decode) before marking the response open, so a malformed frame that is
            # reported as a recoverable error leaves the turn state untouched.
            payload = _AudioEvent.model_validate(data).audio_event
            # `validate=True` so corrupted base64 raises (a `ValueError` the receive loop reports as
            # a recoverable error) instead of silently decoding to garbage or empty bytes.
            delta = AudioDelta(data=base64.b64decode(payload.audio_base_64, validate=True))
            self._response_open = True
            return [delta]
        if event_type == 'client_tool_call':
            call = _ClientToolCallEvent.model_validate(data).client_tool_call
            self._response_open = True
            if not call.expects_response:
                self._fire_and_forget_tool_call_ids.add(call.tool_call_id)
            if isinstance(call.parameters, str):
                args = call.parameters
            else:
                args = to_json(call.parameters or {}).decode()
            return [ToolCall(tool_call_id=call.tool_call_id, tool_name=call.tool_name, args=args)]
        if event_type == 'context_usage':
            event = _ContextUsageEvent.model_validate(data)
            usage = event.context_usage_event or _ContextUsagePayload.model_validate(data)
            if usage.model:
                self._llm_model_name = usage.model
            details = {'context_limit_tokens': usage.context_limit_tokens} if usage.context_limit_tokens else {}
            # ElevenLabs reports LLM context consumption only: no output tokens and no credits reach
            # the WebSocket. Conversation cost appears post-hoc on `GET /v1/convai/conversations/{id}`.
            # Gated behind `client_events` (off by default); verified cadence: once per user turn,
            # *after* the `agent_response` turn boundary, so it cannot be attributed to a specific
            # model response and accumulates into the run total only.
            return [
                SessionUsage(
                    usage=RequestUsage(input_tokens=usage.context_tokens or 0, details=details),
                    response_scoped=False,
                )
            ]
        if event_type == 'client_error':
            error = _ClientErrorEvent.model_validate(data).error_event or _ClientErrorPayload.model_validate(data)
            message = error.message or error.error_name or 'unknown error'
            return [RealtimeSessionErrorEvent(message=f'{_PROVIDER_LABEL} error: {message}', recoverable=True)]
        return self._map_turn_event(event_type, data)

    def _map_turn_event(self, event_type: Any, data: dict[str, Any]) -> list[RealtimeCodecEvent]:
        """Map the frames making up the agent's turn: transcripts, the boundary, and interruptions."""
        if event_type == 'agent_chat_response_part':
            # Text-only conversations stream the response as start/delta/stop parts ahead of the
            # whole-text `agent_response` (verified live; these arrive regardless of the agent's
            # `client_events` set). Deltas map incrementally; the empty start/stop markers carry
            # nothing (a tool-call-only model response is exactly one empty start/stop pair).
            part = _AgentChatResponsePartEvent.model_validate(data).text_response_part
            if part.type == 'delta' and part.text:
                self._response_open = True
                return [OutputTranscript(text=part.text, is_final=False, output_text=self._text_output)]
            return []
        if event_type == 'agent_response':
            # The whole response text, and the reliable end of the agent's turn (verified live):
            # in audio mode it arrives after all `audio` frames, in text mode after the streamed
            # parts, and always once per user turn, after any tool loop. (`agent_response_complete`
            # exists but is off by default and never arrives in audio mode, so it cannot be the
            # boundary.) The session reconciles the final full text against the streamed deltas.
            payload = _AgentResponseEvent.model_validate(data).agent_response_event
            self._response_open = True
            return [
                OutputTranscript(text=payload.agent_response, is_final=True, output_text=self._text_output),
                *self._finalize_response(interrupted=self._turn_interrupted, provider_response_id=payload.response_id),
            ]
        if event_type == 'agent_response_correction':
            # After an interruption the server truncates the stored transcript to what was actually
            # spoken. The already-emitted transcript cannot be shrunk through the codec (a final
            # `OutputTranscript` only extends), so the corrected text is retained on the finalized
            # response's `provider_details` and the interruption closes the turn here. When the user
            # barged in during *playback*, after `agent_response` already closed the turn (the common
            # audio-mode case: synthesis outruns playback), there is nothing left to finalize and the
            # correction is dropped.
            correction = _AgentResponseCorrectionEvent.model_validate(data).agent_response_correction_event
            return self._finalize_response(
                interrupted=True,
                provider_details={'corrected_agent_response': correction.corrected_agent_response},
            )
        if event_type == 'agent_response_complete':
            # Gated behind `client_events` (off by default) and text-mode only, arriving right after
            # `agent_response` has already closed the turn (verified live), so this is normally a
            # dropped no-op; kept as a defensive boundary for agents whose `client_events` omit
            # `agent_response` itself.
            return self._finalize_response(interrupted=self._turn_interrupted)
        if event_type == 'user_transcript':
            payload = _UserTranscriptEvent.model_validate(data).user_transcription_event
            # One final transcript per utterance; no tentative variants were observed live even with
            # `tentative_user_transcript` enabled in `client_events`.
            return [InputTranscript(text=payload.user_transcript, is_final=True)]
        if event_type == 'interruption':
            # The user barged in; the server stops streaming audio and follows up immediately with an
            # `agent_response_correction` (verified live). Only an interruption that lands while the
            # response is still open marks the turn interrupted: during playback the turn has already
            # been finalized by `agent_response`, and the flag must not leak into the next turn.
            if self._response_open:
                self._turn_interrupted = True
            return [RealtimeResponseInterruptedEvent()]
        # Server-side tool/MCP progress, VAD scores, guardrail notices, response metadata, tentative
        # responses, and anything newer than this adapter are informational; ignore them rather than
        # failing the stream.
        return []

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
        self._response_open = False
        # The conversation id rides on every response's `provider_details`: nothing billable reaches
        # the socket, so it is the key a consumer needs for the post-hoc cost lookup on the
        # conversations API, and the response is where it survives into persisted history. A
        # consumer holding the live connection can also read it from `conversation_id` directly.
        details = dict(provider_details or {})
        if self._conversation_id is not None:
            details.setdefault('conversation_id', self._conversation_id)
        return [
            ResponseDone(
                interrupted=interrupted, provider_response_id=provider_response_id, provider_details=details or None
            )
        ]
