from __future__ import annotations as _annotations

import os
from collections.abc import Callable
from types import MappingProxyType
from typing import overload

from pydantic_ai import ModelProfile
from pydantic_ai.profiles import merge_profile
from pydantic_ai.profiles.anthropic import anthropic_model_profile
from pydantic_ai.profiles.google import google_model_profile
from pydantic_ai.profiles.grok import grok_model_profile
from pydantic_ai.profiles.moonshotai import moonshotai_model_profile
from pydantic_ai.profiles.openai import (
    SAMPLING_PARAMS,
    OpenAIJsonSchemaTransformer,
    OpenAIModelProfile,
    openai_model_profile,
)
from pydantic_ai.providers import missing_api_key_error

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:
    raise ImportError(
        'Please install the `openai` package to use the GitHub Copilot provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error
else:
    from ._openai_compatible import (
        AsyncHTTPClient as _OpenAIHTTPClient,
        OpenAICompatibleProvider as _OpenAICompatibleProvider,
    )

_COPILOT_PLUGIN_VERSION = '0.26.7'
"""Version reported to Copilot as the editor plugin build; see `_COPILOT_CLIENT_HEADERS`."""

_COPILOT_CLIENT_HEADERS = MappingProxyType(
    {
        'editor-version': 'vscode/1.95.0',
        'copilot-integration-id': 'vscode-chat',
        'editor-plugin-version': f'copilot-chat/{_COPILOT_PLUGIN_VERSION}',
        'openai-intent': 'conversation-panel',
        'x-github-api-version': '2025-04-01',
    }
)
"""Headers every Copilot client sends, for parity with them.

None of these is documented as required, and none could be *made* required against
`api.githubcopilot.com`: omitting `editor-version` still returns 200, as does an unknown
`copilot-integration-id`, on a plain completion, a tool-call round trip, and an image request. They
are sent as cheap insurance for the enterprise and GHE hosts we cannot reach, which are reported to
enforce `editor-version`. Do not describe them as API requirements.

The `User-Agent` is deliberately not among them: Copilot accepts `pydantic-ai/<version>`, which is
what `OpenAIChatModel` sends, and posing as a Copilot chat client would buy nothing.
"""


class GitHubCopilotModelProfile(OpenAIModelProfile, total=False):
    """Profile for models used with `GitHubCopilotModel`.

    ALL FIELDS MUST BE `github_copilot_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.
    """

    github_copilot_supports_reasoning_effort: bool
    """Whether Copilot's Chat Completions endpoint accepts `reasoning_effort` for this model. Default: `True`.

    Copilot rejects the parameter outright for Anthropic models — including `reasoning_effort='none'` —
    even though those models support thinking through `/v1/messages`.
    `GET https://api.githubcopilot.com/models` reports the accepted values per model, and omits the
    key entirely for Anthropic ids.
    """


def _github_copilot_overlay(model_name: str, family_profile: ModelProfile | None) -> GitHubCopilotModelProfile:
    """Facts about Copilot's own gateway, which the upstream family profile cannot know.

    `model_name` is already lowercased and stripped of a leading `copilot/`.
    """
    overlay = GitHubCopilotModelProfile(
        # Copilot rejects `max_tokens` with an explicit "use `max_completion_tokens` instead" 400, so
        # the gateway pins the field regardless of what a family profile asks for.
        openai_chat_supports_max_completion_tokens=True,
        github_copilot_supports_reasoning_effort=not model_name.startswith('claude-'),
    )

    if family_profile and family_profile.get('anthropic_disallows_sampling_settings'):
        # Anthropic's own transport drops these; the OpenAI-shaped one has to be told to.
        overlay['openai_unsupported_model_settings'] = SAMPLING_PARAMS

    if model_name.startswith('gemini-'):
        # Copilot speaks OpenAI tools and `response_format`, not Gemini `generateContent`, so the
        # `GoogleJsonSchemaTransformer` that `google_model_profile` installs would be wrong here.
        overlay['json_schema_transformer'] = OpenAIJsonSchemaTransformer

    return overlay


class GitHubCopilotProvider(_OpenAICompatibleProvider):
    """Provider for [GitHub Copilot](https://docs.github.com/en/copilot).

    Routes requests through Copilot's OpenAI-compatible Chat Completions API at
    `https://api.githubcopilot.com/chat/completions`, which serves Anthropic, OpenAI, Google, xAI and
    MoonshotAI models under a Copilot subscription. Which model ids you can reach depends on your
    plan; list yours with `GET https://api.githubcopilot.com/models`.

    This is not [`GitHubProvider`][pydantic_ai.providers.github.GitHubProvider], which served the
    retired GitHub Models API.
    """

    @property
    def name(self) -> str:
        return 'github-copilot'

    @property
    def base_url(self) -> str:
        return str(self.client.base_url)

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        # Copilot serves bare model ids with no provider-prefix delimiter, so match each to its family
        # by prefix. A leading `copilot/` is tolerated here because other Copilot clients use it as an
        # id namespace; it is never sent on the wire, where the id goes out exactly as given.
        model_name = model_name.removeprefix('copilot/').casefold()

        prefix_to_profile: dict[str, Callable[[str], ModelProfile | None]] = {
            # The dot-to-hyphen rewrite is Anthropic-only, and that is load-bearing:
            # `anthropic_model_profile` matches hyphenated ids (`claude-haiku-4-5`) while Copilot
            # lists dotted ones, but `grok_model_profile` and `moonshotai_model_profile` match
            # *dotted* ids (`grok-4.5`, `kimi-k3`), so normalizing globally would blank those two.
            'claude-': lambda name: anthropic_model_profile(name.replace('.', '-')),
            'gpt-': openai_model_profile,
            'o1': openai_model_profile,
            'o3': openai_model_profile,
            'o4': openai_model_profile,
            'gemini-': google_model_profile,
            'grok-': grok_model_profile,
            'kimi-': moonshotai_model_profile,
            # GitHub's own models, served on the same OpenAI-shaped surface.
            'mai-': openai_model_profile,
            'oswe': openai_model_profile,
            'raptor': openai_model_profile,
            'exec-agent-': openai_model_profile,
        }

        family_profile: ModelProfile | None = None
        for prefix, profile_func in prefix_to_profile.items():
            if model_name.startswith(prefix):
                family_profile = profile_func(model_name)
                break

        # As `GitHubCopilotProvider` is always used with `GitHubCopilotModel`, which is based on
        # `OpenAIChatModel`, we maintain the base `OpenAIJsonSchemaTransformer` unless the family
        # profile sets one explicitly. An id from no known family gets that fallback alone: it is
        # deliberately not told it can think, since we'd have no evidence that it can.
        return merge_profile(
            OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer),
            family_profile,
            _github_copilot_overlay(model_name, family_profile),
        )

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI) -> None: ...

    @overload
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        openai_client: None = None,
        http_client: _OpenAIHTTPClient | None = None,
    ) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: _OpenAIHTTPClient | None = None,
    ) -> None:
        """Create a new GitHub Copilot provider.

        Args:
            api_key: The Copilot token to authenticate with. Defaults to the `GITHUB_COPILOT_API_KEY`
                environment variable, then to `GITHUB_COPILOT_API_TOKEN` and `COPILOT_GITHUB_TOKEN`,
                the names GitHub's own tooling uses. The general-purpose `GITHUB_TOKEN`, `GH_TOKEN`
                and `GITHUB_API_KEY` variables are deliberately not read, so a token meant for the
                GitHub API is never sent to Copilot.
            base_url: The base URL of the Copilot inference API, e.g. for an enterprise host or a
                local proxy. Defaults to the `GITHUB_COPILOT_BASE_URL`, `COPILOT_API_URL` or
                `GITHUB_COPILOT_API_BASE` environment variable, then to `https://api.githubcopilot.com`.
            openai_client: An existing `AsyncOpenAI` client to use. Its `base_url` must already point
                at the Copilot inference API, and it is used as-is, without the Copilot client
                headers. If provided, `api_key`, `base_url` and `http_client` must be `None`.
            http_client: An existing `httpx2.AsyncClient` or legacy `httpx.AsyncClient` to use for making HTTP requests.
        """
        if openai_client is not None:
            assert api_key is None, 'Cannot provide both `openai_client` and `api_key`'
            assert base_url is None, 'Cannot provide both `openai_client` and `base_url`'
            assert http_client is None, 'Cannot provide both `openai_client` and `http_client`'
            self._client = openai_client
            return

        api_key = (
            api_key
            or os.getenv('GITHUB_COPILOT_API_KEY')
            or os.getenv('GITHUB_COPILOT_API_TOKEN')
            or os.getenv('COPILOT_GITHUB_TOKEN')
        )
        if not api_key:
            raise missing_api_key_error(
                'Set the `GITHUB_COPILOT_API_KEY` environment variable or pass it via'
                ' `GitHubCopilotProvider(api_key=...)` to use the GitHub Copilot provider.'
            )

        # No `/v1`: Copilot serves `/chat/completions` off the root, and `AsyncOpenAI` appends that
        # path itself.
        base_url = (
            base_url
            or os.getenv('GITHUB_COPILOT_BASE_URL')
            or os.getenv('COPILOT_API_URL')
            or os.getenv('GITHUB_COPILOT_API_BASE')
            or 'https://api.githubcopilot.com'
        )

        self._client = self._create_openai_client(
            base_url=base_url, api_key=api_key, http_client=http_client, default_headers=_COPILOT_CLIENT_HEADERS
        )
