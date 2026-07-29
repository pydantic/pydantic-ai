from __future__ import annotations as _annotations

import re

from . import ModelProfile

# Kimi reasoning models (kimi-k2.5/k2.6/k2.7-code, kimi-k2-thinking, kimi-k3, …) accept
# reasoning_effort and emit reasoning_content; the moonshot-v1 and non-thinking k2 models don't.
#
# Gateways disagree on how to punctuate the minor version: MoonshotAI, OpenRouter and Bedrock serve
# `kimi-k2.5`, while Heroku serves `kimi-k2-5` (see `heroku:kimi-k2-5` in `_known_model_names.py`),
# so accept either separator. `kimi-k2-thinking` is matched separately: it is a distinct reasoning
# model rather than a minor version, and it must not widen the match to plain `kimi-k2` /
# `kimi-k2-0905`, which do not support reasoning.
_KIMI_REASONING_RE = re.compile(r'^kimi-(k2[.\-](5|6|7)|k2-thinking|k3|thinking)')


def moonshotai_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a MoonshotAI model."""
    # `thinking_always_enabled` is left to the direct provider, since the `reasoning_effort='none'`
    # quirk is specific to the `api.moonshot.ai` endpoint and this profile is also routed through
    # OpenRouter, Heroku and Bedrock.
    is_reasoning = bool(_KIMI_REASONING_RE.match(model_name.lower()))
    return ModelProfile(
        ignore_streamed_leading_whitespace=True,
        supports_thinking=is_reasoning,
    )
