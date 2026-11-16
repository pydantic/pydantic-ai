from __future__ import annotations as _annotations

from . import InlineDefsJsonSchemaTransformer, ModelProfile

_ALWAYS_THINKING_MODELS = frozenset(
    {
        'minimax-m3.1-flash',
        'minimax-m2.7',
        'minimax-m2.7-highspeed',
        'minimax-m2.5',
        'minimax-m2.5-highspeed',
        'minimax-m2.1',
        'minimax-m2.1-highspeed',
        'minimax-m2',
    }
)
"""Published MiniMax models that always reason.

The M2 family rejects disabling thinking, and first-party M3.1-Flash rejects `reasoning_effort: 'none'` ('requires
adaptive thinking'), per the probes recorded in https://github.com/pydantic/pydantic-ai/issues/12077. The `-highspeed`
variants are first-party only and weren't probed; they take their base model's flags.
"""

_NON_THINKING_MODELS = frozenset({'minimax-m1', 'minimax-m1-80k'})


def minimax_model_profile(model_name: str) -> ModelProfile | None:
    """The model profile for a MiniMax model, matched by its published `minimax-*` id.

    MiniMax models stringify nested tool arguments whose schema is a `$ref` into `$defs` (recorded in
    https://github.com/pydantic/pydantic-ai/issues/12077, first reported in
    https://github.com/pydantic/pydantic-ai/issues/7728), so their schemas are inlined.

    The function takes the dotted id MiniMax publishes (`minimax-m3.1-flash`). Unpublished ids, `minimax-m2-her`
    (which doesn't reason) and legacy `minimax-01` return `None`.
    """
    model_name = model_name.lower()
    if model_name == 'minimax-m3':
        # Hybrid: thinks by default and can be turned off. The gateways' switch for it is
        # https://github.com/pydantic/pydantic-ai/issues/12065.
        return ModelProfile(
            json_schema_transformer=InlineDefsJsonSchemaTransformer,
            supports_thinking=True,
            thinking_enabled_by_default=True,
        )
    if model_name in _ALWAYS_THINKING_MODELS:
        return ModelProfile(
            json_schema_transformer=InlineDefsJsonSchemaTransformer,
            supports_thinking=True,
            thinking_always_enabled=True,
        )
    if model_name in _NON_THINKING_MODELS:
        return ModelProfile(json_schema_transformer=InlineDefsJsonSchemaTransformer)
    return None
