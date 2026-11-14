from __future__ import annotations as _annotations

from . import InlineDefsJsonSchemaTransformer, ModelProfile

_THINKING_MODEL_PREFIXES = ('minimax-m2', 'minimax-m3')
"""Prefixes of the published MiniMax reasoning models.

The `-highspeed` variants (`minimax-m2.7-highspeed`, `minimax-m2.5-highspeed`, `minimax-m2.1-highspeed`)
are first-party only and weren't live-run; per https://github.com/pydantic/pydantic-ai/issues/12077 they
take their base model's flags.
"""

_ALWAYS_THINKING_MODEL_PREFIXES = ('minimax-m2', 'minimax-m3.1')
"""Published MiniMax models that always reason.

The M2 family (M2, M2.1, M2.5, M2.7) rejects disabling thinking, and first-party M3.1-Flash rejects
`reasoning_effort: 'none'` ('requires adaptive thinking'), per the probes recorded in
https://github.com/pydantic/pydantic-ai/issues/12077. `minimax-m3` is hybrid instead: it thinks by default
and can be turned off; #12065 keeps its host-side on/off switch out of this profile.
"""


def minimax_model_profile(model_name: str) -> ModelProfile | None:
    """The model profile for MiniMax models, matched by MiniMax's published `minimax-*` ids.

    MiniMax endpoints stringify nested tool arguments whose schema is a `$ref` into `$defs` (recorded on
    the M2 family and on M3/M3.1-Flash in https://github.com/pydantic/pydantic-ai/issues/12077, first
    reported in https://github.com/pydantic/pydantic-ai/issues/7728), so schemas of matched models are
    inlined via `InlineDefsJsonSchemaTransformer`.

    The function takes the model id lowercased, as `split_bedrock_model_id` and the gateways produce it;
    dotted ids (`minimax-m3.1-flash`) are matched as published — aliasing/normalization of other spellings
    (e.g. `minimax-m3p1-flash`) is handled upstream, see https://github.com/pydantic/pydantic-ai/issues/12102.

    `minimax-m2-her` (no reasoning recorded in the probes) and legacy `minimax-01` are not matched and
    return `None`. `minimax-m1`/`minimax-m1-80k` are matched for schema inlining only, without thinking
    flags.
    """
    model_lower = model_name.lower()
    if model_lower.startswith('minimax-m2-her') or model_lower.startswith('minimax-01'):
        return None
    if not model_lower.startswith((*_THINKING_MODEL_PREFIXES, 'minimax-m1')):
        return None
    return ModelProfile(
        json_schema_transformer=InlineDefsJsonSchemaTransformer,
        supports_thinking=model_lower.startswith(_THINKING_MODEL_PREFIXES),
        thinking_always_enabled=model_lower.startswith(_ALWAYS_THINKING_MODEL_PREFIXES),
        thinking_enabled_by_default=(
            model_lower.startswith('minimax-m3') and not model_lower.startswith('minimax-m3.1')
        ),
    )
