from __future__ import annotations as _annotations

import re

from . import ModelProfile

# Kimi reasoning models (kimi-k2.5/k2.6/k2.7-code, kimi-k3, kimi-thinking, …) accept
# reasoning_effort and emit reasoning_content; the moonshot-v1/instruct models don't. The
# minor-version separator is spelled inconsistently across providers (`kimi-k2.5` on
# Moonshot/Bedrock, `kimi-k2-5` on Heroku) and there is a `kimi-k2-thinking` alias, so a
# `str.startswith` tuple with dotted prefixes only misses the hyphenated and `k2-thinking`
# spellings — silently dropping the `thinking` setting. The anchored regex accepts both
# separators (and `k2-thinking`) while keeping bare `kimi-k2`/`kimi-k2-0905` excluded, since
# those genuinely lack reasoning. `thinking_always_enabled` is left to the direct provider,
# since the `reasoning_effort='none'` quirk is specific to the `api.moonshot.ai` endpoint and
# this profile is also routed through OpenRouter and Heroku.
_KIMI_REASONING_RE = re.compile(r'^kimi-(k2[.\-](5|6|7)|k2-thinking|k3|thinking)')


def moonshotai_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a MoonshotAI model."""
    is_reasoning = bool(_KIMI_REASONING_RE.match(model_name.lower()))
    return ModelProfile(
        ignore_streamed_leading_whitespace=True,
        supports_thinking=is_reasoning,
    )
