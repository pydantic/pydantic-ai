from __future__ import annotations as _annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from ..settings import ThinkingLevel
from . import ModelProfile

AnthropicCodeExecutionToolVersion: TypeAlias = Literal['20250825', '20260120']
"""Concrete Anthropic code execution tool version to send for `CodeExecutionTool`."""

_ANTHROPIC_CODE_EXECUTION_20260120_MODEL_PREFIXES = (
    'claude-opus-4-5',
    'claude-opus-4-6',
    'claude-opus-4-7',
    'claude-sonnet-4-5',
    'claude-sonnet-4-6',
)


@dataclass(kw_only=True)
class AnthropicModelProfile(ModelProfile):
    """Profile for models used with `AnthropicModel`.

    ALL FIELDS MUST BE `anthropic_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.
    """

    anthropic_supports_fast_speed: bool = False
    """Whether the model supports fast inference speed (`anthropic_speed='fast'`).

    Currently only Claude Opus 4.6 supports fast mode. See the Anthropic docs for the latest list.
    """

    anthropic_supports_adaptive_thinking: bool = False
    """Whether the model supports adaptive thinking (Sonnet 4.6+, Opus 4.6+).

    When True, unified `thinking` translates to `{'type': 'adaptive'}`.
    When False, it translates to `{'type': 'enabled', 'budget_tokens': N}`.
    """

    anthropic_supports_effort: bool = False
    """Whether the model supports the `effort` parameter in `output_config` (Opus 4.5+, Sonnet 4.6+).

    When True and the unified thinking level is a string (e.g. 'high'), it is also
    mapped to `output_config.effort`.
    """

    anthropic_supports_xhigh_effort: bool = False
    """Whether the model supports the `xhigh` effort value in `output_config`.

    Claude Opus 4.7 adds `xhigh`; older Anthropic models should use `max` instead.
    """

    anthropic_disallows_budget_thinking: bool = False
    """Whether the model rejects budget-based thinking settings.

    Claude Opus 4.7+ requires adaptive thinking and returns a 400 for
    `{'type': 'enabled', 'budget_tokens': ...}`.
    """

    anthropic_disallows_sampling_settings: bool = False
    """Whether the model rejects sampling settings like `temperature` and `top_p`.

    Claude Opus 4.7+ requires these settings to be omitted from request payloads.
    """

    anthropic_default_code_execution_tool_version: AnthropicCodeExecutionToolVersion = '20250825'
    """The Anthropic code execution tool version used when `anthropic_code_execution_tool_version='auto'`."""

    anthropic_supported_code_execution_tool_versions: tuple[AnthropicCodeExecutionToolVersion, ...] = ('20250825',)
    """The Anthropic code execution tool versions supported by the model."""

    anthropic_supports_task_budgets: bool = False
    """Whether the model supports `output_config.task_budget`.

    Anthropic currently documents task budgets as a Claude Opus 4.7 beta feature.
    """


ANTHROPIC_THINKING_BUDGET_MAP: dict[ThinkingLevel, int] = {
    True: 10000,
    'minimal': 1024,
    'low': 2048,
    'medium': 10000,
    'high': 16384,
    'xhigh': 32768,
}
"""Maps unified thinking values to Anthropic budget_tokens for non-adaptive models."""


def anthropic_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for an Anthropic model."""
    models_that_support_json_schema_output = (
        'claude-haiku-4-5',
        'claude-sonnet-4-5',
        'claude-sonnet-4-6',
        'claude-opus-4-1',
        'claude-opus-4-5',
        'claude-opus-4-6',
        'claude-opus-4-7',
    )
    """These models support both structured outputs and strict tool calling."""
    # TODO update when new models are released that support structured outputs
    # https://docs.claude.com/en/docs/build-with-claude/structured-outputs#example-usage

    supports_json_schema_output = model_name.startswith(models_that_support_json_schema_output)
    anthropic_supports_fast_speed = model_name.startswith('claude-opus-4-6')

    # Sonnet 4.6+ and Opus 4.6+ support adaptive thinking; older models use budget-based
    supports_adaptive = model_name.startswith(('claude-sonnet-4-6', 'claude-opus-4-6', 'claude-opus-4-7'))

    # Opus 4.5+ and Sonnet 4.6+ support the effort parameter in output_config
    supports_effort = model_name.startswith(
        ('claude-opus-4-5', 'claude-opus-4-6', 'claude-opus-4-7', 'claude-sonnet-4-6')
    )
    supports_xhigh_effort = model_name.startswith('claude-opus-4-7')
    disallows_budget_thinking = model_name.startswith('claude-opus-4-7')
    disallows_sampling_settings = model_name.startswith('claude-opus-4-7')
    default_code_execution_tool_version, supported_code_execution_tool_versions = _code_execution_tool_versions(
        model_name
    )
    supports_task_budgets = model_name.startswith('claude-opus-4-7')

    return AnthropicModelProfile(
        thinking_tags=('<thinking>', '</thinking>'),
        supports_json_schema_output=supports_json_schema_output,
        anthropic_supports_fast_speed=anthropic_supports_fast_speed,
        supports_thinking=True,
        anthropic_supports_adaptive_thinking=supports_adaptive,
        anthropic_supports_effort=supports_effort,
        anthropic_supports_xhigh_effort=supports_xhigh_effort,
        anthropic_disallows_budget_thinking=disallows_budget_thinking,
        anthropic_disallows_sampling_settings=disallows_sampling_settings,
        anthropic_default_code_execution_tool_version=default_code_execution_tool_version,
        anthropic_supported_code_execution_tool_versions=supported_code_execution_tool_versions,
        anthropic_supports_task_budgets=supports_task_budgets,
    )


def _code_execution_tool_versions(
    model_name: str,
) -> tuple[AnthropicCodeExecutionToolVersion, tuple[AnthropicCodeExecutionToolVersion, ...]]:
    versions: tuple[AnthropicCodeExecutionToolVersion, ...] = ('20250825',)
    default_version: AnthropicCodeExecutionToolVersion = '20250825'
    if model_name.startswith(_ANTHROPIC_CODE_EXECUTION_20260120_MODEL_PREFIXES):
        default_version = '20260120'
        versions = (*versions, default_version)
    return default_version, versions
