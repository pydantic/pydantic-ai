from __future__ import annotations as _annotations

import os
from datetime import timedelta
from typing import TYPE_CHECKING, overload

from pydantic_ai import ModelProfile
from pydantic_ai._file_input import KNOWN_IMAGE_MEDIA_TYPES
from pydantic_ai.native_tools import SUPPORTED_NATIVE_TOOLS, ComputerTool
from pydantic_ai.profiles import merge_profile
from pydantic_ai.profiles._special_tokens import special_token_route_profile
from pydantic_ai.profiles.openai import (
    _GPT_6_2_MODEL_PREFIXES,  # pyright: ignore[reportPrivateUsage]
    _GPT_6_MODEL_PREFIXES,  # pyright: ignore[reportPrivateUsage]
    OpenAIJsonSchemaTransformer,
    OpenAIModelProfile,
    # Keep the provider-specific specialization private; the provider only selects it.
    _OpenAIJsonSchemaTransformer,  # pyright: ignore[reportPrivateUsage]
    is_openai_computer_use_model,
    openai_model_profile,
    openai_realtime_model_profile,
)
from pydantic_ai.providers import missing_api_key_error
from pydantic_ai.providers._model_deprecations import OPENAI_MODEL_DEPRECATIONS

if TYPE_CHECKING:
    from pydantic_ai.realtime import RealtimeModelProfile

try:
    from openai import AsyncAzureOpenAI, AsyncOpenAI
except ImportError as _import_error:
    raise ImportError(
        'Please install the `openai` package to use the OpenAI provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error

from ._openai_compatible import (
    AsyncHTTPClient as _OpenAIHTTPClient,
    OpenAIAPIKey as OpenAIAPIKey,
    _OpenAISDKProvider,  # pyright: ignore[reportPrivateUsage]
)


class OpenAIProvider(_OpenAISDKProvider):
    """Provider for the OpenAI API, or a transparent proxy to it.

    Changing `base_url` does not change the OpenAI model profile selection, nor the
    [`canonical_system`][pydantic_ai.providers.Provider.canonical_system]: encrypted reasoning and
    compaction items from OpenAI are sent to it, and the ones it returns are sent to OpenAI. For another
    service, use its dedicated provider when available, or
    [`OpenAICompatibleProvider`][pydantic_ai.providers.openai_compatible.OpenAICompatibleProvider].
    See [OpenAI-compatible models](https://pydantic.dev/docs/ai/models/openai/#openai-compatible-models).
    """

    @property
    def name(self) -> str:
        return 'openai'

    _canonical_system_follows_client = True

    @property
    def canonical_system(self) -> str:
        """`'azure'` with an `AsyncAzureOpenAI` client, as Azure only accepts its own encrypted data, otherwise `name`.

        `name` rather than `'openai'`, so a subclass that renames itself to talk to another service doesn't send
        that service's encrypted data to OpenAI, or OpenAI's to it.
        """
        return 'azure' if isinstance(self._client, AsyncAzureOpenAI) else self.name

    @property
    def base_url(self) -> str:
        return str(self.client.base_url)

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        # No per-model gate on `additional_tools`. OpenAI documents a model restriction for the sibling
        # feature ("Only `gpt-5.4` and later models support `tool_search`") and states none for the item,
        # and measurement agrees: 13 models from `gpt-4o-mini` through `gpt-5.6` each called a tool that
        # only an `additional_tools` item declared, 3/3, against 0/3 for a control with the item removed.
        # An earlier list here recorded `gpt-5.4` and `gpt-5` as silently ignoring it; that came from a
        # probe whose prompt named the tool and told the model to call it, which a model that never saw
        # the declaration can satisfy from the prompt text alone.
        #
        # The flag stays here rather than moving into `openai_model_profile`, which is shared with
        # OpenAI-compatible endpoints (Azure, OpenRouter, vLLM, ...) that speak the Responses API without
        # necessarily implementing this item — the same reasoning `openai_supports_phase` documents.
        profile = openai_model_profile(model_name)
        supported_native_tools = profile.get('supported_native_tools', SUPPORTED_NATIVE_TOOLS)
        if is_openai_computer_use_model(model_name):
            # Added here, not in `openai_model_profile`, because the native computer tool is unverified on the other
            # OpenAI-compatible servers that use that profile. `OpenAIChatModel` never sends it.
            # https://developers.openai.com/api/docs/guides/tools-computer-use
            supported_native_tools = supported_native_tools | {ComputerTool}
        return merge_profile(
            profile,
            OpenAIModelProfile(
                # The OpenAI API answers HEIC, AVIF, BMP and TIFF images with a 400 that lists JPEG, PNG, GIF and
                # WebP (checked live with `gpt-5-mini` on Chat Completions and Responses, 2026-11-04). Set here
                # rather than in `openai_model_profile`, which OpenAI-compatible servers that read more types share.
                supported_image_media_types=KNOWN_IMAGE_MEDIA_TYPES,
                # A `system` message after a tool result is accepted and followed on Chat Completions and Responses
                # (checked live with `gpt-4.1-mini` and `gpt-6.1-terra`, 2026-11-23). Set here rather than in
                # `openai_model_profile`, which OpenAI-compatible endpoints share, and strict ones reject a
                # mid-conversation `system` message with a 400 (https://github.com/pydantic/pydantic-ai/issues/14135).
                supports_inline_system_prompts=True,
                # Fine-tuned models keep the conservative string-bound behavior.
                json_schema_transformer=(
                    OpenAIJsonSchemaTransformer if model_name.startswith('ft:') else _OpenAIJsonSchemaTransformer
                ),
                tool_addition_mode='with_definitions',
                tool_deferral_mode='with_tool_search',
                # GPT-5.6 and later document a model-determined floor: a cached prefix stays eligible for
                # reuse for at least 30 minutes after its last write or reuse. These are the models with
                # explicit cache breakpoints. Earlier models have no honest boundary to record, because
                # their retention policy defaults to `24h` for organizations without zero data retention and
                # to `in_memory` (5-10 minutes of inactivity) for organizations with it — an org setting that
                # isn't knowable from the model. Those stay unset (`'unknown'`) rather than guess in either
                # direction: too low a boundary would declare a live cache cold and throw away a real hit, too
                # high a one would report expiry as an unexpected collapse. Explicitly requested retention,
                # such as `openai_prompt_cache_retention='24h'`, is resolved by the model's
                # `resolve_cache_retention`.
                # https://developers.openai.com/api/docs/guides/prompt-caching
                default_cache_retention=timedelta(minutes=30)
                if profile.get('openai_supports_prompt_cache_breakpoints')
                else None,
                # Models before GPT-5.6 cache prompts implicitly with no request-side configuration, so the unified
                # `cache` setting has nothing to turn on there. GPT-5.6 and later take `prompt_cache_options` and
                # explicit breakpoints, which it maps onto. Set here rather than in `openai_model_profile`, which is
                # shared with OpenAI-compatible endpoints that don't necessarily accept them.
                supports_cache=bool(profile.get('openai_supports_prompt_cache_breakpoints')),
                # OpenAI ignores `CachePoint.ttl`. GPT-5.6 and later accept a request-wide
                # `prompt_cache_options.ttl` of `'30m'` (the default) or, on the Responses API only, `'12h'`
                # (verified live on every GPT-5.6, GPT-6, GPT-6.1 and GPT-6.2 model; `OpenAIChatModel` leaves `'12h'`
                # out, as Chat Completions rejects it). `AzureProvider` follows Microsoft's documentation, which
                # lists only `'30m'`, so this is set here rather than in `openai_model_profile`. Earlier models
                # have no retention tier to request (`prompt_cache_retention` is a maximum-retention policy, not
                # a tier).
                # https://developers.openai.com/api/docs/guides/prompt-caching
                supported_cache_retentions=('30m', '12h')
                if profile.get('openai_supports_prompt_cache_breakpoints')
                else (),
                # Prompt cache diagnostics are documented for GPT-5.6 and later, the models with explicit cache
                # breakpoints. Earlier models accept the field but always answer `unavailable`. Set here rather than
                # in `openai_model_profile` because OpenAI-compatible Responses endpoints don't all accept it:
                # OpenRouter rejects it with a 400.
                # https://developers.openai.com/api/docs/guides/prompt-caching/diagnostics
                openai_responses_supports_prompt_cache_diagnostics=bool(
                    profile.get('openai_supports_prompt_cache_breakpoints')
                ),
                # Only OpenAI's audio models (`gpt-audio*`, `gpt-4o-audio-preview`, `gpt-4o-mini-audio-preview`) take
                # `input_audio` on Chat Completions; the others reject it with a 400, so a tool-returned audio file
                # would end the run instead of becoming a note. Responses takes no audio input on any model, which
                # `OpenAIResponsesModel` already declares. Set here rather than in `openai_model_profile`, which is
                # the fallback for unknown models on OpenAI-compatible endpoints, some of which serve audio models.
                # https://developers.openai.com/api/docs/guides/audio
                supports_audio_file_input='audio' in model_name,
                # Measured live: every model from `gpt-4o-mini` through `gpt-5.5` rejects a `tool_choice` naming a tool
                # that only an `additional_tools` item declares, and GPT-5.6 and GPT-6 accept it. Those are the models
                # with explicit cache breakpoints. Set here with `tool_addition_mode`, since the item is first-party.
                openai_responses_tool_choice_accepts_additional_tools=bool(
                    profile.get('openai_supports_prompt_cache_breakpoints')
                ),
                # `gpt-6.2-sol` and `gpt-6.2-astra` put the top-level `instructions` after `input` and bill them at
                # the uncached rate, and OpenAI recommends leading input messages for reusable instructions
                # (measured live: https://github.com/pydantic/pydantic-ai/pull/14095). Set here rather than in
                # `openai_model_profile` because OpenAI-compatible endpoints that share it may render instructions
                # differently.
                # https://developers.openai.com/api/docs/guides/latest-model
                openai_responses_stable_instructions_as_input=model_name.startswith(_GPT_6_2_MODEL_PREFIXES),
                # GPT-5.6 and later, the models with explicit cache breakpoints, accept `reasoning.effort='max'`
                # (verified live). Set here rather than in `openai_model_profile` because `max` is unverified on
                # the OpenAI-compatible endpoints that share it.
                openai_responses_supports_max_reasoning_effort=bool(
                    profile.get('openai_supports_prompt_cache_breakpoints')
                ),
                # GPT-5.1 through GPT-5.5 reject `reasoning_effort='minimal'` on every OpenAI API that serves them
                # (verified live, 2026-11). Set here rather than in `openai_model_profile` because OpenRouter accepts
                # it for these models.
                openai_supports_minimal_reasoning_effort=profile.get('openai_supports_minimal_reasoning_effort', True)
                and not model_name.startswith(('gpt-5.1', 'gpt-5.2', 'gpt-5.3', 'gpt-5.4', 'gpt-5.5')),
                supported_native_tools=supported_native_tools,
                # Set here rather than in `openai_model_profile` because OpenAI-compatible Chat Completions APIs
                # don't all honor it: OpenRouter and xAI silently ignore it, and OpenRouter rejects it for Claude.
                # A server reached through a custom `base_url` that doesn't honor it opts out in the model's profile.
                # https://platform.openai.com/docs/api-reference/chat/create
                openai_chat_supports_allowed_tools=True,
                # The `configuration_update` input item that changes reasoning effort mid-conversation is
                # documented for the GPT-6 family. Set here rather than in `openai_model_profile` because the
                # OpenAI-compatible Responses endpoints that share it don't necessarily implement the item.
                # https://developers.openai.com/api/docs/guides/reasoning
                supports_mid_conversation_effort=model_name.startswith(_GPT_6_MODEL_PREFIXES),
                # GPT-6 rejects function tools on Chat Completions while reasoning is on, asking for
                # `/v1/responses` or `reasoning_effort='none'` (verified live on gpt-6-luna, gpt-6-sol,
                # gpt-6.1-terra, gpt-6-astra, gpt-6.1-sol, gpt-6.2-sol and gpt-6.2-astra, 2026-11). `OpenAIChatModel` raises a
                # `ModelRequestRejectedError` before sending. Set here rather than in `openai_model_profile` because the
                # restriction is unverified on the OpenAI-compatible endpoints that share it.
                openai_chat_function_tools_require_reasoning_off=model_name.startswith(_GPT_6_MODEL_PREFIXES),
                # Gemini's OpenAI-compatible endpoint repeats the running usage across streamed chunks: see
                # `test_gemini_stream_usage_replaces_running_totals` and the live check across Gemini model families
                # on https://github.com/pydantic/pydantic-ai/pull/14285. Set here rather than in
                # `openai_model_profile`, which OpenAI-compatible hosts share without this being verified on them.
                openai_chat_streaming_repeats_usage_totals=model_name.startswith('gemini'),
                # Gemini's OpenAI-compatible endpoint counts thinking tokens in `total_tokens` but not in
                # `completion_tokens`: see `test_gemini_usage_counts_thinking_tokens_left_out_of_completion_tokens` and
                # the live probe on https://github.com/pydantic/pydantic-ai/issues/14291. Set here rather than in
                # `openai_model_profile` for the same reason as the key above.
                openai_chat_completion_tokens_omit_reasoning=model_name.startswith('gemini'),
                # The Batch API is OpenAI's own: set here rather than in `openai_model_profile`, which OpenAI-compatible
                # providers share without necessarily serving `/v1/batches`. `OpenAIResponsesModel` turns it off for an
                # Azure OpenAI client.
                openai_supports_batch_api=True,
                # Measured live on Chat Completions and Responses: a tool call with no result, or a result naming no
                # call, fails the request with a 400 (https://github.com/pydantic/pydantic-ai/issues/13404). Set here
                # rather than in `openai_model_profile`, which OpenAI-compatible endpoints share without being
                # verified to reject it.
                rejects_unpaired_tool_calls=True,
            ),
            special_token_route_profile('openai', model_name),
            # `gpt-5-search-api` matches the shared `'gpt-5'` reasoning row, but OpenAI's API rejects any
            # `reasoning_effort` on it with a 400, so unified `thinking` sends none; sampling parameters stay dropped,
            # as it rejects `temperature` too (verified live, 2026-11). Set here rather than in `openai_model_profile`
            # because the restriction is unverified on the OpenAI-compatible endpoints that share it.
            # `OpenAIProvider(base_url=...)` keeps it, as a proxy to OpenAI should.
            OpenAIModelProfile(openai_reasoning_efforts=frozenset())
            if model_name.startswith('gpt-5-search-api')
            else OpenAIModelProfile(),
            # The OpenAI models drop it for an Azure OpenAI client, as Azure retires models on its own schedule.
            OpenAIModelProfile()
            if (deprecation := OPENAI_MODEL_DEPRECATIONS.get(model_name)) is None
            else OpenAIModelProfile(deprecation=deprecation),
        )

    @staticmethod
    def realtime_model_profile(model_name: str) -> RealtimeModelProfile:
        return openai_realtime_model_profile(model_name)

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI) -> None: ...

    @overload
    def __init__(
        self,
        base_url: str | None = None,
        api_key: OpenAIAPIKey | None = None,
        openai_client: None = None,
        http_client: _OpenAIHTTPClient | None = None,
    ) -> None: ...

    def __init__(
        self,
        base_url: str | None = None,
        api_key: OpenAIAPIKey | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: _OpenAIHTTPClient | None = None,
    ) -> None:
        """Create a new OpenAI provider.

        Args:
            base_url: The base url for the OpenAI requests. If not provided, the `OPENAI_BASE_URL` environment variable
                will be used if available. Otherwise, defaults to OpenAI's base url.
            api_key: The API key to use for authentication, if not provided, the `OPENAI_API_KEY` environment variable
                will be used if available. For short-lived or rotating credentials, pass an async callable that
                returns the key, which the OpenAI SDK calls before each request.
            openai_client: An existing
                [`AsyncOpenAI`](https://github.com/openai/openai-python?tab=readme-ov-file#async-usage)
                client to use. If provided, `base_url`, `api_key`, and `http_client` must be `None`.
            http_client: An existing `httpx2.AsyncClient` or legacy `httpx.AsyncClient` to use for making HTTP requests.
        """
        if api_key is None and 'OPENAI_API_KEY' not in os.environ and openai_client is None:
            if base_url is None and 'OPENAI_BASE_URL' not in os.environ:
                # When talking to OpenAI directly, a missing key would otherwise surface as a raw
                # `openai.OpenAIError`; raise our own `UserError` instead so the message is consistent with
                # other providers and points newcomers to the keyless test model.
                raise missing_api_key_error(
                    'Set the `OPENAI_API_KEY` environment variable or pass it via '
                    f'`{type(self).__name__}(api_key=...)` to use the OpenAI provider.'
                )
            else:
                # This is a workaround for the OpenAI client requiring an API key, whilst locally served,
                # openai compatible models do not always need an API key, but a placeholder (non-empty) key is required.
                api_key = 'api-key-not-set'

        if openai_client is not None:
            assert base_url is None, 'Cannot provide both `openai_client` and `base_url`'
            assert http_client is None, 'Cannot provide both `openai_client` and `http_client`'
            assert api_key is None, 'Cannot provide both `openai_client` and `api_key`'
            self._client = openai_client
        else:
            self._client = self._create_openai_client(base_url=base_url, api_key=api_key, http_client=http_client)
