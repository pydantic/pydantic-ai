"""Reasoning effort, sampling parameters while reasoning, and reasoning items."""

from __future__ import annotations as _annotations

import warnings
from typing import Any, Literal, get_args

from openai.types import responses
from openai.types.shared import ReasoningEffort
from openai.types.shared_params import Reasoning

from ...messages import ThinkingPart
from ...profiles.openai import OPENAI_REASONING_EFFORT_MAP, SAMPLING_PARAMS, OpenAIModelProfile, OpenAIReasoningEffort
from ...settings import ThinkingLevel
from .. import ModelRequestParameters
from ._settings import OpenAIChatModelSettings, OpenAIResponsesModelSettings

_REASONING_EFFORT_ORDER: tuple[OpenAIReasoningEffort, ...] = get_args(OpenAIReasoningEffort)
"""Every `reasoning_effort` value, from least to most reasoning, in the order `OpenAIReasoningEffort` lists them."""


def resolve_openai_thinking_effort(
    thinking: ThinkingLevel, profile: OpenAIModelProfile, *, supports_max: bool
) -> ReasoningEffort:
    """Map unified thinking to the closest reasoning effort the model accepts, or `None` to send none.

    `thinking=False` on a model that always reasons resolves to the least reasoning it accepts.
    `supports_max` says whether the API the request goes to accepts `'max'` for the model, which differs
    between OpenAI's Chat Completions and Responses APIs.
    """
    supported = profile.get('openai_reasoning_efforts')
    if supported is None:
        if thinking is False and profile.get('thinking_always_enabled', False):
            # Without the model's accepted efforts there's no lowest one to send, so it keeps its default.
            return None
        if thinking == 'minimal' and not profile.get('openai_supports_minimal_reasoning_effort', True):
            return 'low'
        if thinking == 'max' and supports_max:
            return 'max'
        return OPENAI_REASONING_EFFORT_MAP[thinking]
    return closest_accepted_reasoning_effort(thinking, supported, supports_max=supports_max)


def closest_accepted_reasoning_effort(
    thinking: ThinkingLevel, supported: frozenset[OpenAIReasoningEffort], *, supports_max: bool
) -> ReasoningEffort:
    """Map unified thinking to the closest of the efforts the model accepts, or `None` when it accepts none."""
    accepted: list[OpenAIReasoningEffort] = sorted(
        (effort for effort in supported if effort != 'max' or supports_max), key=_REASONING_EFFORT_ORDER.index
    )
    if thinking is False:
        # `'none'` where the model accepts it, else its lowest effort.
        return next(iter(accepted), None)
    requested_effort = 'max' if thinking == 'max' and supports_max else OPENAI_REASONING_EFFORT_MAP[thinking]
    requested = _REASONING_EFFORT_ORDER.index(requested_effort)
    active: list[OpenAIReasoningEffort] = [effort for effort in accepted if effort != 'none']
    # The lowest accepted effort at or above the requested one, else the highest.
    for effort in active:
        if _REASONING_EFFORT_ORDER.index(effort) >= requested:
            return effort
    return active[-1] if active else None


def supports_reasoning_effort_none(profile: OpenAIModelProfile) -> bool:
    """Whether the model accepts `reasoning_effort='none'`, read from its accepted efforts where they're pinned."""
    if (supported := profile.get('openai_reasoning_efforts')) is not None:
        return 'none' in supported
    return profile.get('openai_supports_reasoning_effort_none', False)


def reasoning_active(
    profile: OpenAIModelProfile,
    model_settings: OpenAIChatModelSettings | OpenAIResponsesModelSettings,
    model_request_parameters: ModelRequestParameters,
) -> bool:
    """Whether reasoning is effectively on for this request.

    Explicit effort wins; then the unified `thinking` setting; otherwise the model's default.
    """
    reasoning_effort = model_settings.get('openai_reasoning_effort')
    if reasoning_effort is not None:
        return reasoning_effort != 'none'
    thinking = model_request_parameters.thinking
    if thinking is not None:
        # `thinking=False` on a model that always reasons sends its lowest effort, so reasoning stays on.
        return thinking is not False or profile.get('thinking_always_enabled', False)
    return profile.get('thinking_enabled_by_default', False)


def drop_sampling_params_for_reasoning(
    profile: OpenAIModelProfile,
    model_settings: OpenAIChatModelSettings,
    model_request_parameters: ModelRequestParameters,
) -> None:
    """Drop sampling params when reasoning is enabled on models that support it.

    Mutates `model_settings`.

    Reasoning models don't support sampling parameters while reasoning is active. For models that
    can turn reasoning off (`openai_supports_reasoning_effort_none`), sampling params are allowed
    when reasoning is off. Whether reasoning is on when no effort is set depends on the model's
    default (`thinking_enabled_by_default`): the GPT-5.1..5.4 mainline models default to
    off, while the o-series, the original GPT-5, and GPT-5.5+ default to on.
    """
    if not profile.get('openai_supports_reasoning', False):
        return

    # When the model can turn reasoning off, sampling params are allowed while reasoning is inactive.
    if supports_reasoning_effort_none(profile) and not reasoning_active(
        profile, model_settings, model_request_parameters
    ):
        return

    if dropped := [k for k in SAMPLING_PARAMS if k in model_settings]:
        warnings.warn(
            f'Sampling parameters {dropped} are not supported when reasoning is enabled. '
            'These settings will be ignored.',
            UserWarning,
        )

    for k in SAMPLING_PARAMS:
        model_settings.pop(k, None)


def set_reasoning_context(
    reasoning: Reasoning,
    reasoning_context: Literal['auto', 'current_turn', 'all_turns'] | None,
    profile: OpenAIModelProfile,
) -> None:
    """Set the `context` of a Responses request's reasoning, where the model accepts it.

    Mutates `reasoning`.
    """
    if reasoning_context is None:
        # Default to `all_turns` on models that support it, so earlier-turn reasoning stays
        # available without opting in — consistent with sending prior thinking (including
        # "foreign" thinking parts from another model) back to every other model. Models
        # without `all_turns` support get no `context` at all, never `auto`.
        if profile.get('openai_responses_supports_reasoning_context', False):
            reasoning['context'] = 'all_turns'
    else:
        # Support is per-value, not per-field: every reasoning model accepts `auto` and
        # `current_turn`, while `all_turns` is limited to the families that persist reasoning
        # across turns. Gating the whole field on the narrower fact would silently discard a
        # value the user explicitly set.
        if reasoning_context == 'all_turns':
            supports_reasoning_context = profile.get('openai_responses_supports_reasoning_context', False)
        else:
            supports_reasoning_context = profile.get('openai_supports_reasoning', False)
        if supports_reasoning_context:
            reasoning['context'] = reasoning_context


def map_reasoning_item(item: responses.ResponseReasoningItem, provider_name: str) -> list[ThinkingPart]:
    """Map a Responses reasoning item to the `ThinkingPart`s that replay it."""
    signature = item.encrypted_content
    # Handle raw CoT content from gpt-oss models
    provider_details: dict[str, Any] = {}
    raw_content: list[str] | None = [c.text for c in item.content] if item.content else None
    if raw_content:
        provider_details['raw_content'] = raw_content

    parts: list[ThinkingPart] = []
    if item.summary:
        for summary in item.summary:
            # We use the same id for all summaries so that we can merge them on the round trip.
            parts.append(
                ThinkingPart(
                    content=summary.text,
                    id=item.id,
                    signature=signature,
                    provider_name=provider_name,
                    provider_details=provider_details or None,
                )
            )
            # We only need to store the signature and raw_content once.
            signature = None
            provider_details = {}
    elif signature or provider_details:
        parts.append(
            ThinkingPart(
                content='',
                id=item.id,
                signature=signature,
                provider_name=provider_name,
                provider_details=provider_details or None,
            )
        )
    return parts
