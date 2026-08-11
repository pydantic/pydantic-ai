"""The realtime model abstraction and inference, mirroring [`pydantic_ai.models`][pydantic_ai.models]."""

from __future__ import annotations as _annotations

from abc import abstractmethod
from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, Literal

from typing_extensions import TypeAliasType

from ..exceptions import ModelAPIError, UserError
from ..messages import ModelMessage
from ..models import ModelRequestParameters
from ..models._abstract import AbstractModel
from ..native_tools import AbstractNativeTool
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

    Concrete models take it as a keyword-only `profile` init argument and assign it here, mirroring how
    [`Model`][pydantic_ai.models.Model] stores its own `profile=`.
    """

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
          3. The user's `profile=` argument — a partial dict merged on top, OR a callable
             `(resolved) -> profile` for full control.

        Then `supported_native_tools` is intersected with what this model class actually implements, so
        the resolved profile is the single source of truth for what is usable.
        """
        provider: Provider[object] | None = getattr(self, '_provider', None)
        provider_profile = provider.realtime_model_profile(self.model_name) if provider is not None else None
        resolved = merge_realtime_profile(DEFAULT_REALTIME_PROFILE, provider_profile)
        if (user := self._profile) is not None:
            # The callable form replaces the resolved profile wholesale rather than merging, so a caller
            # can drop a claim the provider made and not just add to it.
            resolved = user(resolved) if callable(user) else merge_realtime_profile(resolved, user)
        profile_supported = resolved.get('supported_native_tools', frozenset())
        effective_tools = profile_supported & self.__class__.supported_native_tools()
        if effective_tools != profile_supported:
            resolved = merge_realtime_profile(resolved, RealtimeModelProfile(supported_native_tools=effective_tools))
        return resolved

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
"""Known realtime model identifiers, surfaced for autocomplete."""


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

        return GoogleRealtimeModel(model_name, provider=provider)
    raise UserError(
        f'Unknown realtime model provider {provider!r}. Supported providers are `openai`, `azure`, '
        '`xai`, `google`, and `google-cloud`, or `gateway/openai` / `gateway/google` to route OpenAI '
        'or Gemini Live realtime through the Pydantic AI Gateway.'
    )
