from __future__ import annotations as _annotations

from typing import Literal

from . import ModelProfile


class HuggingFaceModelProfile(ModelProfile, total=False):
    """Profile for models used with HuggingFaceModel.

    ALL FIELDS MUST BE `huggingface_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.
    """

    huggingface_send_back_thinking_parts: Literal['auto', 'tags']
    """How thinking parts in history are sent back to the model. Default: `'auto'`.

    HuggingFace has no native thinking round-trip, so this governs all `ThinkingPart`s:

    * `'auto'` (default): they are dropped. The model does not re-absorb `<think>` tags from history, so
    re-rendering them as text teaches the model to mimic the format in its user-visible answers, leaking
    reasoning to end users.
    * `'tags'`: they are re-rendered as text wrapped in the `thinking_tags`, the behavior from before this
    setting existed.

    See `AnthropicModelProfile.anthropic_send_back_thinking_parts` for more context."""
