from __future__ import annotations as _annotations

from . import ModelProfile


class HuggingFaceModelProfile(ModelProfile, total=False):
    """Profile for models used with HuggingFaceModel.

    ALL FIELDS MUST BE `huggingface_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.
    """

    huggingface_send_back_thinking_parts: bool
    """Whether to re-render prior ThinkingParts as thinking tags in history. Default: `False`.

    When False (default), ThinkingParts are silently dropped from the assistant turn rather than
    being wrapped in `<think>…</think>` tags — which the model would otherwise reproduce verbatim
    in subsequent turns (reasoning-content leak).

    Set to True only if you deliberately want prior thinking content re-sent to the HuggingFace model.
    """
