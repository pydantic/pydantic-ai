from __future__ import annotations as _annotations

from typing import Literal, TypeAlias

from ..native_tools import (
    AdvisorTool,
    CodeExecutionTool,
    MCPServerTool,
    MemoryTool,
    WebFetchTool,
    WebSearchTool,
)
from ..native_tools._tool_search import ToolSearchTool
from ..settings import ThinkingEffort, ThinkingLevel
from . import ModelProfile

_ANTHROPIC_BASE_BUILTINS = frozenset({WebSearchTool, CodeExecutionTool, WebFetchTool, MemoryTool, MCPServerTool})
"""Native tool types Anthropic generally supports across the model line. Mirrors
`AnthropicModel.supported_native_tools()` minus `ToolSearchTool`, which is gated
per-model in the profile below."""

AnthropicCodeExecutionToolVersion: TypeAlias = Literal['20250825', '20260120']
"""Concrete Anthropic code execution tool version to send for `CodeExecutionTool`."""

_ANTHROPIC_CODE_EXECUTION_20260120_MODEL_PREFIXES = (
    'claude-fable-5',
    'claude-mythos-5',
    'claude-opus-4-5',
    'claude-opus-4-6',
    'claude-opus-4-7',
    'claude-opus-4-8',
    'claude-sonnet-4-5',
    'claude-sonnet-4-6',
    'claude-sonnet-5',
)


class AnthropicModelProfile(ModelProfile, total=False):
    """Profile for models used with `AnthropicModel`.

    ALL FIELDS MUST BE `anthropic_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.
    """

    anthropic_supports_fast_speed: bool
    """Whether the model supports fast inference speed (`anthropic_speed='fast'`). Default: `False`.

    Currently Claude Opus 4.6, 4.7, and 4.8 support fast mode. See the Anthropic docs for the latest list.
    """

    anthropic_supports_adaptive_thinking: bool
    """Whether the model supports adaptive thinking (Sonnet 4.6+, Opus 4.6+). Default: `False`.

    When True, unified `thinking` translates to `{'type': 'adaptive'}`.
    When False, it translates to `{'type': 'enabled', 'budget_tokens': N}`.
    """

    anthropic_supports_effort: bool
    """Whether the model supports the `effort` parameter in `output_config` (Opus 4.5+, Sonnet 4.6+). Default: `False`.

    When True and the unified thinking level is a string (e.g. 'high'), it is also
    mapped to `output_config.effort`.
    """

    anthropic_supports_dynamic_filtering: bool
    """Whether the model supports Anthropic-managed dynamic filtering for web search/fetch. Default: `False`.

    When enabled, Pydantic AI selects the `web_search_20260209` / `web_fetch_20260209` tool versions,
    which let Claude filter web results via code execution before they enter context.
    """

    anthropic_supports_xhigh_effort: bool
    """Whether the model supports the `xhigh` effort value in `output_config`. Default: `False`.

    Claude Opus 4.7 and 4.8 accept `xhigh`; older Anthropic models should use `max` instead.
    """

    anthropic_disallows_budget_thinking: bool
    """Whether the model rejects budget-based thinking settings. Default: `False`.

    Claude Opus 4.7 and 4.8 require adaptive thinking and return a 400 for
    `{'type': 'enabled', 'budget_tokens': ...}`.
    """

    anthropic_disallows_sampling_settings: bool
    """Whether the model rejects sampling settings like `temperature` and `top_p`. Default: `False`.

    Claude Opus 4.7 and 4.8 require these settings to be omitted from request payloads.
    """

    anthropic_default_code_execution_tool_version: AnthropicCodeExecutionToolVersion
    """The Anthropic code execution tool version used when `anthropic_code_execution_tool_version='auto'`. Default: `'20250825'`."""

    anthropic_supported_code_execution_tool_versions: tuple[AnthropicCodeExecutionToolVersion, ...]
    """The Anthropic code execution tool versions supported by the model. Default: `('20250825',)`."""

    anthropic_supports_task_budgets: bool
    """Whether the model supports `output_config.task_budget`. Default: `False`.

    Anthropic currently documents task budgets as a Claude Opus 4.7 / 4.8 beta feature.
    """

    anthropic_supports_forced_tool_choice: bool
    """Whether the model accepts a forced `tool_choice` (`{'type': 'any'}` or `{'type': 'tool'}`).

    Most Anthropic models only reject forcing alongside thinking mode; Claude Fable 5 and Claude
    Mythos Preview reject it unconditionally with a 400. When False, a resolved `required` tool choice
    falls back to `auto` (filtering tools to the requested set), and an explicit `tool_choice='required'`
    (or an explicit list of tools) raises a `UserError`.
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


AnthropicEffort: TypeAlias = Literal['low', 'medium', 'high', 'xhigh', 'max']
"""Effort values Anthropic accepts at `output_config.effort`."""


ANTHROPIC_THINKING_EFFORT_MAP: dict[ThinkingEffort, AnthropicEffort] = {
    'minimal': 'low',
    'low': 'low',
    'medium': 'medium',
    'high': 'high',
    'xhigh': 'max',
}
"""Maps unified thinking effort levels to Anthropic `output_config.effort`.

`xhigh` maps to `'max'` by default; callers that target a model with
`anthropic_supports_xhigh_effort` should pass `supports_xhigh=True` to
[`resolve_anthropic_effort`][pydantic_ai.profiles.anthropic.resolve_anthropic_effort]
to preserve `xhigh` instead of downshifting.
"""


def resolve_anthropic_effort(level: ThinkingEffort, *, supports_xhigh: bool) -> AnthropicEffort:
    """Resolve a unified thinking effort level to the Anthropic `output_config.effort` value.

    Shared between the direct Anthropic path and any provider that translates to the
    Anthropic `output_config` wire shape (e.g. Bedrock Converse for Anthropic models).
    Keeps `ANTHROPIC_THINKING_EFFORT_MAP` as the single source of truth for the
    base mapping, while letting the `xhigh` passthrough decision live in one place.
    """
    if level == 'xhigh' and supports_xhigh:
        return 'xhigh'
    return ANTHROPIC_THINKING_EFFORT_MAP[level]


def anthropic_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for an Anthropic model."""
    models_that_support_json_schema_output = (
        'claude-fable-5',
        'claude-mythos-5',
        'claude-haiku-4-5',
        'claude-sonnet-4-5',
        'claude-sonnet-4-6',
        'claude-opus-4-1',
        'claude-opus-4-5',
        'claude-opus-4-6',
        'claude-opus-4-7',
        'claude-opus-4-8',
        'claude-sonnet-5',
    )
    """These models support both structured outputs and strict tool calling."""
    # TODO update when new models are released that support structured outputs
    # https://docs.claude.com/en/docs/build-with-claude/structured-outputs#example-usage

    supports_json_schema_output = model_name.startswith(models_that_support_json_schema_output)
    anthropic_supports_fast_speed = model_name.startswith(('claude-opus-4-6', 'claude-opus-4-7', 'claude-opus-4-8'))

    # Sonnet 4.6+ and Opus 4.6+ support adaptive thinking; older models use budget-based
    supports_adaptive = model_name.startswith(
        (
            'claude-fable-5',
            'claude-mythos-5',
            'claude-sonnet-4-6',
            'claude-sonnet-5',
            'claude-opus-4-6',
            'claude-opus-4-7',
            'claude-opus-4-8',
        )
    )

    # Opus 4.5+ and Sonnet 4.6+ support the effort parameter in output_config
    supports_effort = model_name.startswith(
        (
            'claude-fable-5',
            'claude-mythos-5',
            'claude-opus-4-5',
            'claude-opus-4-6',
            'claude-opus-4-7',
            'claude-opus-4-8',
            'claude-sonnet-4-6',
            'claude-sonnet-5',
        )
    )
    supports_xhigh_effort = model_name.startswith(
        ('claude-fable-5', 'claude-mythos-5', 'claude-opus-4-7', 'claude-opus-4-8', 'claude-sonnet-5')
    )
    disallows_budget_thinking = model_name.startswith(
        ('claude-fable-5', 'claude-mythos-5', 'claude-opus-4-7', 'claude-opus-4-8', 'claude-sonnet-5')
    )
    disallows_sampling_settings = model_name.startswith(
        ('claude-fable-5', 'claude-mythos-5', 'claude-opus-4-7', 'claude-opus-4-8', 'claude-sonnet-5')
    )
    default_code_execution_tool_version, supported_code_execution_tool_versions = _code_execution_tool_versions(
        model_name
    )
    supports_task_budgets = model_name.startswith(
        ('claude-fable-5', 'claude-mythos-5', 'claude-opus-4-7', 'claude-opus-4-8', 'claude-sonnet-5')
    )

    # Claude Fable 5, Claude Mythos 5, and Claude Mythos Preview reject a forced `tool_choice`
    # (`any`/`tool`) outright, unlike other Anthropic models which only reject forcing while thinking
    # is enabled. The forcing-tool-use docs name Mythos Preview explicitly; Mythos 5 is its successor
    # and the safety-classifier-free twin of Fable 5, both of which reject forcing.
    supports_forced_tool_choice = not model_name.startswith(
        ('claude-fable-5', 'claude-mythos-5', 'claude-mythos-preview')
    )

    supports_dynamic_filtering = model_name.startswith(
        (
            'claude-fable-5',
            'claude-mythos-5',
            'claude-mythos-preview',
            'claude-sonnet-4-6',
            'claude-sonnet-5',
            'claude-opus-4-6',
            'claude-opus-4-7',
            'claude-opus-4-8',
        )
    )

    # Native tool search requires the `tool_search_tool_bm25_20251119` /
    # `tool_search_tool_regex_20251119` API types, which post-date Claude 4.0. In
    # practice, Anthropic enables it for Sonnet 4.5+, Opus 4.5+, and Haiku 4.5+.
    supports_tool_search = model_name.startswith(
        (
            'claude-fable-5',
            'claude-mythos-5',
            'claude-sonnet-4-5',
            'claude-sonnet-4-6',
            'claude-sonnet-5',
            'claude-opus-4-5',
            'claude-opus-4-6',
            'claude-opus-4-7',
            'claude-opus-4-8',
            'claude-haiku-4-5',
        )
    )
    # The advisor tool's valid *executors* (the model driving generation) are Fable/Mythos 5, Opus
    # 4.6-4.8, Sonnet 4.6/5, and Haiku 4.5. The executor/advisor pairing itself is validated API-side.
    supports_advisor = model_name.startswith(
        (
            'claude-fable-5',
            'claude-mythos-5',
            'claude-opus-4-6',
            'claude-opus-4-7',
            'claude-opus-4-8',
            'claude-sonnet-4-6',
            'claude-sonnet-5',
            'claude-haiku-4-5',
        )
    )
    supported_native_tools = _ANTHROPIC_BASE_BUILTINS
    if supports_tool_search:
        supported_native_tools = supported_native_tools | {ToolSearchTool}
    if supports_advisor:
        supported_native_tools = supported_native_tools | {AdvisorTool}

    return AnthropicModelProfile(
        thinking_tags=('<thinking>', '</thinking>'),
        supports_json_schema_output=supports_json_schema_output,
        anthropic_supports_fast_speed=anthropic_supports_fast_speed,
        supports_thinking=True,
        anthropic_supports_adaptive_thinking=supports_adaptive,
        anthropic_supports_effort=supports_effort,
        anthropic_supports_dynamic_filtering=supports_dynamic_filtering,
        anthropic_supports_xhigh_effort=supports_xhigh_effort,
        anthropic_disallows_budget_thinking=disallows_budget_thinking,
        anthropic_disallows_sampling_settings=disallows_sampling_settings,
        anthropic_default_code_execution_tool_version=default_code_execution_tool_version,
        anthropic_supported_code_execution_tool_versions=supported_code_execution_tool_versions,
        anthropic_supports_task_budgets=supports_task_budgets,
        anthropic_supports_forced_tool_choice=supports_forced_tool_choice,
        supported_native_tools=supported_native_tools,
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
