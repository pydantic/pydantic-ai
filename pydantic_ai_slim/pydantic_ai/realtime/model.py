"""The realtime model abstraction and inference, mirroring [`pydantic_ai.models`][pydantic_ai.models]."""

from __future__ import annotations as _annotations

from abc import abstractmethod
from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import KW_ONLY, dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, NoReturn, Protocol

from typing_extensions import TypeAliasType

from .._genai_prices import lookup_context_window, preload_pricing_data
from ..exceptions import ModelAPIError, UserError
from ..messages import ModelMessage
from ..models import ModelRequestParameters
from ..models._abstract import AbstractModel
from ..native_tools import AbstractNativeTool
from ..tools import ToolDefinition
from .codec import RealtimeConnection
from .profiles import (
    DEFAULT_AUDIO_SAMPLE_RATE,
    DEFAULT_REALTIME_PROFILE,
    RealtimeModelProfile,
    RealtimeModelProfileSpec,
    merge_realtime_profile,
)
from .settings import RealtimeModelSettings

if TYPE_CHECKING:
    from ..providers import Provider


class RealtimeError(ModelAPIError):
    """A realtime connection or protocol failure: the session could not be opened, or is over.

    Raised when the handshake fails, the provider closes the session, a send fails, or
    [reconnecting][pydantic_ai.realtime.ReconnectPolicy] gives up. A rejected WebSocket upgrade is the
    exception: it carries an HTTP status, so it raises
    [`ModelHTTPError`][pydantic_ai.exceptions.ModelHTTPError] like a regular request.

    A subclass of [`ModelAPIError`][pydantic_ai.exceptions.ModelAPIError], since losing the connection
    to a realtime provider is the same kind of failure as a request-response call that couldn't reach
    the API. Catch it specifically to separate the session's own failures from those of any text agent
    the session [delegates to](../realtime/tools.md#delegating-work-during-a-call).
    """


# WebRTC / browser-media artifacts.
#
# For browser voice agents, the recommended topology is WebRTC: audio flows browser <-> provider
# directly (lowest latency), while the server attaches a control-plane connection to the same call to
# run the agent loop (instructions, tools, history). These small, provider-neutral dataclasses are the
# handoff artifacts between the server's signaling endpoint and the sideband session.


@dataclass(frozen=True)
class RealtimeClientSecret:
    """An ephemeral client secret (short-lived token) that a browser can use to talk to a provider directly.

    Minted server-side by [`create_client_secret`][pydantic_ai.realtime.RealtimeModel.create_client_secret]
    so a long-lived API key never reaches the browser. The token is bound to a session configuration
    (instructions, tools, voice, VAD) and expires quickly (OpenAI: about a minute).
    """

    value: str = field(repr=False)
    """The ephemeral secret to hand to the browser client.

    Kept out of the `repr` so logging or inspecting the object doesn't leak the live token into logs.
    """
    _: KW_ONLY
    expires_at: datetime
    """When the secret expires (timezone-aware UTC)."""
    provider_details: dict[str, Any] | None = field(default=None, repr=False)
    """Raw provider fields returned alongside the secret (e.g. the resolved `session` object).

    Kept out of the `repr` alongside `value`: the resolved `session` carries the instructions and tool
    definitions, so a logged `repr` would otherwise expose them. Access the field explicitly to read it.
    """


class RealtimeProviderSession(Protocol):
    """A handle to a provider-side realtime session that a server sideband connection can attach to.

    The transport-neutral contract that [`Agent.realtime`][pydantic_ai.agent.Agent.realtime] accepts as
    `provider_session`: it needs only the owning provider (to check the attaching model matches) and an
    opaque session identifier (to address the control-plane connection). Different transports satisfy it
    with their own handle types — a WebRTC HTTP relay yields a [`WebRTCSession`][pydantic_ai.realtime.WebRTCSession];
    a provider that negotiates over a WebSocket can supply its own.
    """

    @property
    def provider_name(self) -> str:
        """The provider that owns the session (e.g. `'openai'` or `'azure'`); must match the model attaching to it."""
        ...

    @property
    def session_id(self) -> str:
        """The provider-assigned identifier used to address the session's control-plane connection."""
        ...


@dataclass(frozen=True)
class WebRTCSession:
    """A [`RealtimeProviderSession`][pydantic_ai.realtime.RealtimeProviderSession] for a WebRTC call.

    Produced by [`answer_webrtc_offer`][pydantic_ai.realtime.RealtimeModel.answer_webrtc_offer] and
    passed as `provider_session` to [`Agent.realtime`][pydantic_ai.agent.Agent.realtime] to run the
    agent loop over the call's control plane while the browser owns the audio.
    """

    provider_name: str
    """The provider that owns the call (e.g. `'openai'` or `'azure'`); must match the model attaching to it."""
    _: KW_ONLY
    session_id: str
    """The provider-assigned call identifier (OpenAI/Azure return it as the `call_id` in the `Location` header)."""
    provider_details: dict[str, Any] | None = None
    """Raw provider details about the call (e.g. the original `Location` header)."""

    @property
    def call_id(self) -> str:
        """Alias for [`session_id`][pydantic_ai.realtime.WebRTCSession.session_id] under the OpenAI/Azure wire name."""
        return self.session_id


@dataclass(frozen=True)
class WebRTCAnswer:
    """The provider's WebRTC SDP answer plus the [`WebRTCSession`][pydantic_ai.realtime.WebRTCSession] to attach to.

    Return `sdp` to the browser to complete the WebRTC handshake, then pass `session` as `provider_session`
    to [`Agent.realtime`][pydantic_ai.agent.Agent.realtime] to run the sideband session.
    """

    sdp: str
    """The provider's SDP answer, to send back to the browser as the remote description."""
    _: KW_ONLY
    session: WebRTCSession
    """The call handle the server sideband session attaches to."""


class RealtimeModel(AbstractModel):
    """Abstract base class for realtime model providers.

    [`RealtimeModel`][pydantic_ai.realtime.RealtimeModel] and the request-response
    [`Model`][pydantic_ai.models.Model] share [`AbstractModel`][pydantic_ai.models.AbstractModel].
    A realtime model opens a persistent bidirectional connection for streaming content in and out.

    Like [`Model`][pydantic_ai.models.Model], the `settings` attribute and the `model_settings`
    passed to `connect` are typed as the shared [`RealtimeModelSettings`][pydantic_ai.realtime.RealtimeModelSettings];
    each provider narrows to its own `TypedDict` subclass internally with a `cast` (as the
    request-response models do for `ModelSettings`), rather than the base class being generic over the
    settings type.
    """

    settings: RealtimeModelSettings | None = None
    """Model settings used as defaults for realtime sessions."""

    _profile: RealtimeModelProfileSpec | None = None
    """The user's `profile=` override, applied as the last layer of [`profile`][pydantic_ai.realtime.RealtimeModel.profile].

    Concrete models take it as a keyword-only `profile` init argument and pass it to the base
    constructor, mirroring how [`Model`][pydantic_ai.models.Model] stores its own `profile=`.
    """

    def __init__(
        self,
        *,
        settings: RealtimeModelSettings | None = None,
        profile: RealtimeModelProfileSpec | None = None,
    ) -> None:
        """Store the model-level `settings` and `profile=` layers, as [`Model`][pydantic_ai.models.Model]'s constructor does.

        Also loads genai-prices' data snapshot, which [`profile`][pydantic_ai.realtime.RealtimeModel.profile]
        consults for `context_window`, so that one-time cost stays off the event loop.
        """
        self.settings = settings
        self._profile = profile
        preload_pricing_data()

    @classmethod
    def supported_native_tools(cls) -> frozenset[type[AbstractNativeTool]]:
        """Return the native tool types implemented by this realtime model class."""
        return frozenset()

    def _merge_model_settings(self, model_settings: RealtimeModelSettings | None) -> RealtimeModelSettings | None:
        """Merge model-level defaults with connection-level overrides."""
        settings = self.settings.copy() if self.settings else None
        if model_settings:
            if settings is None:
                settings = model_settings.copy()
            else:
                settings.update(model_settings)
        return settings

    @abstractmethod
    def connect(
        self,
        *,
        messages: Sequence[ModelMessage],
        model_settings: RealtimeModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AbstractAsyncContextManager[RealtimeConnection]:
        """Open a connection to the realtime model.

        Args:
            messages: Prior conversation and the current request carrying session instructions,
                projected to the provider's initial conversation items. Replayable text, transcripts,
                thinking, tool rounds, images, and retained user audio are seeded according to the
                model profile; content the provider cannot represent raises `UserError`.
            model_settings: Optional provider-specific settings.
            model_request_parameters: Function and native tools available to the session.

        Returns:
            An async context manager yielding a [`RealtimeConnection`][pydantic_ai.realtime.codec.RealtimeConnection].
        """
        raise NotImplementedError

    def connect_webrtc(
        self,
        session: RealtimeProviderSession,
        *,
        messages: Sequence[ModelMessage],
        model_settings: RealtimeModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AbstractAsyncContextManager[RealtimeConnection]:
        """Attach a control-plane (sideband) connection to an existing provider-side `session`.

        The returned connection runs the agent loop over the session's control channel while the browser
        exchanges audio with the provider directly, so the sideband doesn't own the audio transport. Only
        realtime models whose provider supports WebRTC server-side controls (OpenAI and Azure OpenAI)
        implement this; the default raises [`UserError`][pydantic_ai.exceptions.UserError] and points
        callers to the WebSocket transport.
        """
        self._raise_unsupported_webrtc('connect_webrtc')

    async def create_client_secret(
        self,
        *,
        instructions: str | None = None,
        tools: Sequence[ToolDefinition] | None = None,
        model_settings: RealtimeModelSettings | None = None,
        expires_after_seconds: int | None = None,
    ) -> RealtimeClientSecret:
        """Mint an ephemeral [`RealtimeClientSecret`][pydantic_ai.realtime.RealtimeClientSecret] for a browser client.

        Binds the token to the given session configuration so a browser can open a realtime connection
        directly without ever holding a long-lived API key. Only implemented by providers that support
        ephemeral tokens (OpenAI and Azure OpenAI); the default raises
        [`UserError`][pydantic_ai.exceptions.UserError] and points callers to the WebSocket transport.
        """
        self._raise_unsupported_webrtc('create_client_secret')

    async def answer_webrtc_offer(
        self,
        sdp_offer: str,
        *,
        instructions: str | None = None,
        tools: Sequence[ToolDefinition] | None = None,
        model_settings: RealtimeModelSettings | None = None,
    ) -> WebRTCAnswer:
        """Relay a browser's WebRTC SDP offer to the provider and return the SDP answer plus a [`WebRTCSession`][pydantic_ai.realtime.WebRTCSession].

        This is the secure signaling path: the server (holding the API key) negotiates the WebRTC call
        on the browser's behalf, so the browser never sees a token. Return
        [`WebRTCAnswer.sdp`][pydantic_ai.realtime.WebRTCAnswer.sdp] to the browser, then pass
        [`WebRTCAnswer.session`][pydantic_ai.realtime.WebRTCAnswer.session] as `provider_session` to
        [`Agent.realtime`][pydantic_ai.agent.Agent.realtime]. Only implemented by
        providers that support WebRTC (OpenAI and Azure OpenAI); the default raises
        [`UserError`][pydantic_ai.exceptions.UserError] and points callers to the WebSocket transport.
        """
        self._raise_unsupported_webrtc('answer_webrtc_offer')

    def _raise_unsupported_webrtc(self, method: str) -> NoReturn:
        raise UserError(
            f'Realtime model {self.model_name!r} does not support WebRTC, so `{method}()` is unavailable. '
            "Branch on `model.profile['supports_webrtc']` up front, or connect over WebSockets instead."
        )

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The model name, e.g. `gpt-realtime`."""
        raise NotImplementedError

    @property
    def base_url(self) -> str | None:
        """The provider API base URL, when this model is backed by a provider."""
        provider: Provider[object] | None = getattr(self, '_provider', None)
        return provider.base_url if provider is not None else None

    @property
    def profile(self) -> RealtimeModelProfile:
        """The realtime model profile.

        Resolution order mirrors [`Model.profile`][pydantic_ai.models.Model.profile] (later layers
        override earlier ones):

          1. [`DEFAULT_REALTIME_PROFILE`][pydantic_ai.realtime.codec.DEFAULT_REALTIME_PROFILE] — base
             values for every key.
          2. The provider's `realtime_model_profile(model_name)` result — provider-specific defaults.
          3. A best-effort `context_window` value from
             [genai-prices](https://github.com/pydantic/genai-prices), unless the provider or a
             partial user profile explicitly set the field (including to `None`).
          4. The user's `profile=` argument — a partial dict merged on top, OR a callable
             `(resolved) -> profile` for full control.

        Then `supported_native_tools` is intersected with what this model class actually implements, so
        the resolved profile is the single source of truth for what is usable.
        """
        provider: Provider[object] | None = getattr(self, '_provider', None)
        provider_profile = provider.realtime_model_profile(self.model_name) if provider is not None else None
        resolved = merge_realtime_profile(DEFAULT_REALTIME_PROFILE, provider_profile)
        user = self._profile
        context_window_set = 'context_window' in (provider_profile or {}) or (
            user is not None and not callable(user) and 'context_window' in user
        )
        if not context_window_set:
            context_window = lookup_context_window(self)
            if context_window is not None:
                resolved = merge_realtime_profile(resolved, RealtimeModelProfile(context_window=context_window))
        if user is not None:
            # The callable form replaces the resolved profile wholesale rather than merging, so a caller
            # can drop a claim the provider made and not just add to it.
            resolved = user(resolved) if callable(user) else merge_realtime_profile(resolved, user)
        profile_supported = resolved.get('supported_native_tools', frozenset())
        effective_tools = profile_supported & self.__class__.supported_native_tools()
        if effective_tools != profile_supported:
            resolved = merge_realtime_profile(resolved, RealtimeModelProfile(supported_native_tools=effective_tools))
        return resolved

    @property
    def context_window(self) -> int | None:
        """The resolved profile's [`context_window`][pydantic_ai.realtime.RealtimeModelProfile.context_window]."""
        return self.profile.get('context_window')

    @property
    def audio_input_sample_rate(self) -> int:
        """The sample rate, in Hz, expected for raw PCM audio input.

        Also available on the session as
        [`RealtimeSession.audio_input_sample_rate`][pydantic_ai.realtime.RealtimeSession.audio_input_sample_rate];
        read it here when audio capture must be configured before a session exists.
        """
        return self.profile.get('audio_input_sample_rate', DEFAULT_AUDIO_SAMPLE_RATE)

    @property
    def audio_output_sample_rate(self) -> int:
        """The sample rate, in Hz, of the raw PCM audio the model produces.

        Also available on the session as
        [`RealtimeSession.audio_output_sample_rate`][pydantic_ai.realtime.RealtimeSession.audio_output_sample_rate];
        read it here when audio playback must be configured before a session exists.
        """
        return self.profile.get('audio_output_sample_rate', DEFAULT_AUDIO_SAMPLE_RATE)


KnownRealtimeModelName = TypeAliasType(
    'KnownRealtimeModelName',
    Literal[
        'openai:gpt-realtime',
        'openai:gpt-realtime-2.1',
        'openai:gpt-realtime-2.1-mini',
        'azure:gpt-realtime',
        'xai:grok-voice-latest',
        'xai:grok-voice-think-fast-2.0',
        'google:gemini-2.5-flash-native-audio-latest',
        'google:gemini-3.1-flash-live-preview',
    ],
)
"""Known realtime model identifiers, surfaced for autocomplete and pinned to provider aliases by a sync test."""


def infer_realtime_model(model: KnownRealtimeModelName | str) -> RealtimeModel:
    """Infer a realtime model from a `provider:model` identifier.

    The provider is one of `openai`, `azure`, `xai`, `google` (the Gemini Developer API), or
    `google-cloud` (Vertex AI) — e.g. `openai:gpt-realtime` — or a
    [Pydantic AI Gateway](../gateway.md) route (`gateway/openai:gpt-realtime`,
    `gateway/google:gemini-live-2.5-flash`), which connects through the gateway's built-in provider —
    the provider string is passed to the realtime model as its `provider`, so authentication and the
    base URL come from [`gateway_provider`][pydantic_ai.providers.gateway.gateway_provider].
    """
    provider, separator, model_name = model.partition(':')
    if not separator or not model_name:
        raise UserError(
            f'Realtime model identifiers use the `provider:model` format (e.g. `openai:gpt-realtime`); got {model!r}.'
        )
    model_kind = provider
    if model_kind.startswith('gateway/'):
        from ..providers.gateway import normalize_gateway_provider

        # Same alias resolution as `infer_model`: the gateway's Google upstream is the Vertex route,
        # so `gateway/google` collapses onto `google-cloud`. The un-normalized string stays the
        # model's `provider`, whose handshake reads the gateway base URL and bearer key from
        # `gateway_provider` (the OpenAI protocol already carries the same trace context the
        # gateway's HTTP request hook would add).
        model_kind = normalize_gateway_provider(model_kind)
        if model_kind not in ('openai', 'google-cloud'):
            raise UserError(
                f'Realtime model provider {provider!r} cannot be routed through the Pydantic AI Gateway. '
                'Supported gateway routes are `gateway/openai` and `gateway/google`.'
            )

    if model_kind == 'openai':
        from .openai import OpenAIRealtimeModel

        return OpenAIRealtimeModel(model_name, provider=provider)
    if model_kind == 'azure':
        from .azure import AzureRealtimeModel

        return AzureRealtimeModel(model_name)
    if model_kind == 'xai':
        from .xai import XaiRealtimeModel

        return XaiRealtimeModel(model_name)
    # `google` is the Gemini Developer API and `google-cloud` is Vertex AI, exactly as in `infer_model`.
    if model_kind in ('google', 'google-cloud'):
        from .google import GoogleRealtimeModel

        return GoogleRealtimeModel(model_name, provider='gateway' if provider.startswith('gateway/') else model_kind)
    raise UserError(
        f'Unknown realtime model provider {provider!r}. Supported providers are `openai`, `azure`, '
        '`xai`, `google`, and `google-cloud`, or `gateway/openai` / `gateway/google` to route OpenAI '
        'or Gemini Live realtime through the Pydantic AI Gateway.'
    )
