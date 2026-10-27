"""Which system can verify the stateless signed or encrypted data in a response.

Stateless round-trip data is data the API verifies on replay by its own key and format, not by looking
up stored state: OpenAI encrypted reasoning and compaction items, Anthropic thinking signatures and
redacted thinking, Gemini thought signatures, xAI encrypted reasoning and tool results, and OpenRouter
or Snowflake reasoning details. The system that verifies it is the producing provider's
[`canonical_system`][pydantic_ai.providers.Provider.canonical_system], which the response records in
[`ModelResponse.canonical_system`][pydantic_ai.messages.ModelResponse.canonical_system] when it differs
from `provider_name`, and always when the provider's canonical system depends on its client (Google
providers and `OpenAIProvider`).

Stateful data (response, item, file, container and conversation ids) keeps matching `provider_name`
exactly, and doesn't go through this module.
"""

from __future__ import annotations as _annotations

import re
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from .messages import ModelResponse

__all__ = ('accepts_stateless_artifact', 'stateless_artifact_system')

_LEGACY_CANONICAL_SYSTEMS: dict[str, str] = {
    # Pre-v2 names, still found in stored histories.
    'google-gla': 'google',
    'google-vertex': 'google-cloud',
    'grok': 'xai',
    # A built-in provider whose canonical system always differs from its name, for histories recorded
    # before responses carried `canonical_system`.
    'openai-codex': 'openai',
}

_GOOGLE_PROVIDER_NAMES = frozenset({'google', 'google-cloud'})
# The global endpoint, a regional one (`us-central1-aiplatform.googleapis.com`), or a data residency one
# (`aiplatform.eu.rep.googleapis.com`).
_GOOGLE_CLOUD_HOST = re.compile(
    r'(?:[a-z0-9-]+-)?aiplatform\.googleapis\.com|aiplatform\.[a-z0-9-]+\.rep\.googleapis\.com'
)


def _google_system_from_url(url: str | None) -> str | None:
    """The Google transport a first-party Gemini API or Google Cloud base URL belongs to, if it is one.

    Only for histories without `canonical_system`: `google` and `google-cloud` each name both
    transports, depending on the client. Any other URL, such as a proxy's, is no evidence either way.
    """
    if not url:
        return None
    host = urlsplit(url).hostname or ''
    if host == 'generativelanguage.googleapis.com':
        return 'google'
    if _GOOGLE_CLOUD_HOST.fullmatch(host):
        return 'google-cloud'
    return None


def stateless_artifact_system(response: ModelResponse, provider_name: str) -> tuple[str, bool]:
    """The canonical system of stateless data a part from `provider_name` carries in `response`.

    Returns the system and whether it was derived from the name alone, without the response's
    `canonical_system` or, for an older Google history, its `provider_url`.
    """
    if provider_name == response.provider_name:
        if response.canonical_system is not None:
            return response.canonical_system, False
        if provider_name in _GOOGLE_PROVIDER_NAMES and (system := _google_system_from_url(response.provider_url)):
            return system, False
    return _LEGACY_CANONICAL_SYSTEMS.get(provider_name, provider_name), True


def accepts_stateless_artifact(
    response: ModelResponse,
    provider_name: str | None,
    *,
    system: str,
    canonical_system: str,
    compatible_systems: frozenset[str],
) -> bool:
    """Whether a receiver can verify stateless data a part from `provider_name` carries in `response`.

    The data is accepted when its canonical system is the receiver's `canonical_system` or one of its
    `compatible_systems`. Data whose system could only be derived from its name is also accepted when
    that name is the receiver's own `system`, as it always was: a provider can produce either of two
    canonical systems under one name, such as `OpenAIProvider` with an Azure client, and a history
    recorded before responses carried `canonical_system` doesn't say which.
    """
    if provider_name is None:
        return False
    artifact_system, from_name = stateless_artifact_system(response, provider_name)
    if artifact_system == canonical_system or artifact_system in compatible_systems:
        return True
    return from_name and provider_name == system
