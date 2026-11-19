"""Whether a request thinks, and how, and whether it may then force tool use."""

from __future__ import annotations as _annotations

from typing import TYPE_CHECKING, Literal, cast

from anthropic.types.beta import BetaThinkingConfigParam

from ...profiles.anthropic import AnthropicModelProfile
from ...settings import ModelSettings, ThinkingLevel
from .. import ModelRequestParameters
from .._tool_choice import support_tool_forcing, tool_forcing_unavailable_reason

if TYPE_CHECKING:
    from . import AnthropicModelSettings


def effective_thinking_type(
    anthropic_thinking: BetaThinkingConfigParam | None,
    unified_thinking: ThinkingLevel | None,
    profile: AnthropicModelProfile,
) -> Literal['enabled', 'adaptive'] | None:
    """Resolve whether a request will think, and how, for the output-mode and tool-forcing decisions.

    Extended thinking (`{'type': 'enabled'}`) rejects a forced `tool_choice`; adaptive thinking accepts it, but
    the model then answers without thinking. Unified thinking maps to `adaptive` when the profile advertises it
    and to `enabled` otherwise, the same mapping `_translate_thinking` uses to build the wire payload. With no
    thinking setting, models that think by default resolve to `adaptive`, as does `thinking=False` on a model
    that can't turn thinking off, which then thinks at its lowest effort. Returns `None` when the request won't
    think, including Claude Sonnet 5.5's `between_tools`, which doesn't think before responding.
    """
    if anthropic_thinking:
        thinking_type = anthropic_thinking.get('type')
        return thinking_type if thinking_type in ('enabled', 'adaptive') else None
    if unified_thinking:
        return 'adaptive' if profile.get('anthropic_supports_adaptive_thinking', False) else 'enabled'
    if unified_thinking is False and not profile.get('thinking_always_enabled', False):
        return None
    return 'adaptive' if profile.get('thinking_enabled_by_default', False) else None


def request_thinking_type(
    profile: AnthropicModelProfile,
    model_settings: ModelSettings | None,
    model_request_parameters: ModelRequestParameters,
) -> Literal['enabled', 'adaptive'] | None:
    """`effective_thinking_type` for a request, before or after `Model.prepare_request` runs.

    `params.thinking` is checked first since `Model.prepare_request` moves unified `thinking` from `model_settings`
    into it, but `AnthropicModel.prepare_request` also asks before that happens.
    """
    anthropic_settings = cast('AnthropicModelSettings', model_settings or {})
    unified_thinking = model_request_parameters.thinking
    if unified_thinking is None:
        unified_thinking = anthropic_settings.get('thinking')
    return effective_thinking_type(anthropic_settings.get('anthropic_thinking'), unified_thinking, profile)


def request_allows_tool_forcing(
    model_name: str,
    profile: AnthropicModelProfile,
    model_settings: AnthropicModelSettings,
    model_request_parameters: ModelRequestParameters,
) -> bool:
    """Whether to send a forced `tool_choice` ('any'/specific tool), raising if explicitly requested but unavailable.

    On top of the profile's forcing flags, extended thinking rejects forcing, and adaptive thinking accepts it
    but answers without thinking, so only an explicit forcing `tool_choice` is sent then.
    Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/implement-tool-use#forcing-tool-use
    """
    thinking_type = request_thinking_type(profile, model_settings, model_request_parameters)
    unavailable_reason = tool_forcing_unavailable_reason(
        profile,
        thinking=thinking_type is not None,
        thinking_remedy="Disable thinking with `thinking=False` or `anthropic_thinking={'type': 'disabled'}`",
    )
    if unavailable_reason is None and thinking_type == 'enabled':
        unavailable_reason = (
            "Extended thinking doesn't support forcing tool use. Disable thinking or use `tool_choice='auto'`."
        )
        if profile.get('anthropic_supports_adaptive_thinking', False):
            unavailable_reason += " Alternatively, `anthropic_thinking={'type': 'adaptive'}` supports forcing."
    return support_tool_forcing(
        model_name,
        model_settings,
        unavailable_reason,
        disables_thinking=thinking_type == 'adaptive' and profile.get('forced_tool_choice_disables_thinking', False),
    )
