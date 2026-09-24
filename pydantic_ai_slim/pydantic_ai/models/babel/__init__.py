"""Drop-in models whose provider wire mapping is done by [babel](https://github.com/pydantic/babel).

Each model here subclasses the corresponding Pydantic AI model and replaces only its wire mapping,
request assembly, response parsing and streaming, with babel's `llm_transform` codecs: one
transform spec per provider, compiled to Python, Rust and TypeScript and verified byte-for-byte
against a shared corpus. Authentication, HTTP clients, model settings, profiles and the agent loop
stay exactly as in the parent model.

The models live in per-provider modules so importing one never requires another provider's SDK:

- [`BabelOpenAIChatModel`][pydantic_ai.models.babel.openai.BabelOpenAIChatModel]
- [`BabelAnthropicModel`][pydantic_ai.models.babel.anthropic.BabelAnthropicModel]
- [`BabelGoogleModel`][pydantic_ai.models.babel.google.BabelGoogleModel]
- [`BabelBedrockConverseModel`][pydantic_ai.models.babel.bedrock.BabelBedrockConverseModel]

The boundary between Pydantic AI's messages and babel's IR is exposed for anyone building a
babel-backed model for another provider.
"""

import importlib

try:
    importlib.import_module('llm_transform')
except ImportError as _import_error:
    raise ImportError(
        'Please install `llm-babel` to use the babel models, '
        'you can use the `babel` optional group — `pip install "pydantic-ai-slim[babel]"`'
    ) from _import_error

from ._adapters import fold_stream_emits, ir_to_model_response, messages_to_ir

__all__ = ('fold_stream_emits', 'ir_to_model_response', 'messages_to_ir')
