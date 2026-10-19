from __future__ import annotations

from . import ModelProfile, merge_profile
from .decision import DecisionModelProfile, decision_model_profile


def cloudflare_model_profile(model_name: str) -> ModelProfile | None:
    """Get the profile for a Cloudflare Clef or Clairic model."""
    if model_name == '@cf/cloudflare/clairic':
        return ModelProfile(
            context_window=65_536,
            supports_tools=False,
            supports_text_output=False,
            supports_json_schema_output=True,
            supports_json_object_output=False,
            default_structured_output_mode='native',
        )
    elif model_name == '@cf/cloudflare/clef':
        context_window = 65_536
    elif model_name == '@cf/cloudflare/clef-flash':
        context_window = 24_576
    else:
        return None
    return merge_profile(
        decision_model_profile(model_name),
        DecisionModelProfile(
            decision_max_choice_options=255,
            decision_max_score_levels=10,
            context_window=context_window,
        ),
    )
