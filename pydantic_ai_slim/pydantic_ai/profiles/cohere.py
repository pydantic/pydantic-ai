from __future__ import annotations as _annotations

from . import ModelProfile


class CohereModelProfile(ModelProfile, total=False):
    """Profile for Cohere models.

    ALL FIELDS MUST BE `cohere_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.
    """

    cohere_supports_image_content: bool
    """Whether the model accepts `image_url` content blocks in user messages. Default: `False`.

    Only Cohere's vision models take images; the others reject the block.
    See [Cohere's image inputs docs](https://docs.cohere.com/docs/image-inputs).
    """


def cohere_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Cohere model."""
    profile = CohereModelProfile()
    if 'reasoning' in model_name:
        profile['supports_thinking'] = True
        profile['thinking_always_enabled'] = True
    if 'vision' in model_name:
        # Cohere's vision models carry `vision` in the name, e.g. `command-a-vision-07-2025`.
        profile['cohere_supports_image_content'] = True
    # Only the keys a model actually needs, so a plain `command` resolves to the same profile as before.
    return profile or None
