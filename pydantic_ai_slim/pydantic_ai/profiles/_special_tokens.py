"""Which routes may encode chat-template special tokens in message content as real control tokens.

The host decides this, not the model family: the same weights are text on one host and control tokens on the next,
and a routing service can pick either. This table is the one place that decides which routes get
[`may_tokenize_special_tokens_in_content`][pydantic_ai.profiles.ModelProfile.may_tokenize_special_tokens_in_content].
Each provider asks it from its `model_profile`, under the provider's own name, so narrowing a route is one line here.
"""

from __future__ import annotations as _annotations

from typing import Final, Literal

from . import ModelProfile

_SpecialTokenRoute = Literal[
    'azure',
    'cerebras',
    'crusoe',
    'groq',
    'heroku',
    'huggingface',
    'nebius',
    'ollama',
    'openai',
    'openai-compatible',
    'openrouter',
    'ovhcloud',
    'sambanova',
    'together',
    'vercel',
    'vllm',
    'zai',
]

_EVERY_MODEL: Final[frozenset[_SpecialTokenRoute]] = frozenset(
    {
        # Self-hosted: a server renders the model's own template, and tokenizes the result unless told otherwise.
        'huggingface',
        'ollama',
        'vllm',
        # Hosts of open-weight models.
        'cerebras',
        'crusoe',
        'groq',
        'nebius',
        'ovhcloud',
        'sambanova',
        'together',
        'zai',
    }
)
"""Routes that only serve open-weight models. Every other route gets the flag for names `_is_closed_model` doesn't match."""

_CLOSED_MODEL_PREFIXES: Final = (
    # Served by the lab's own API, whichever route reaches it, which doesn't render a public chat template.
    'claude',
    'command',
    'gemini',
    'grok',
    'muse-',
    'nova-',
    'sonar',
    # OpenAI's own names. `gpt-oss` is the exception below.
    'chatgpt',
    'codex',
    'computer-use',
    'dall-e',
    'ft:',
    'gpt-',
    'o1',
    'o3',
    'o4',
    'omni-',
    'sora',
    'text-',
    'tts-',
    'whisper',
)


def _is_closed_model(model_name: str) -> bool:
    bare_name = model_name.lower().removeprefix('~').rpartition('/')[2]
    return bare_name.startswith(_CLOSED_MODEL_PREFIXES) and not bare_name.startswith('gpt-oss')


def special_token_route_profile(route: _SpecialTokenRoute, model_name: str) -> ModelProfile:
    """The `may_tokenize_special_tokens_in_content` setting for `model_name` on `route`."""
    return ModelProfile(
        may_tokenize_special_tokens_in_content=route in _EVERY_MODEL or not _is_closed_model(model_name)
    )
