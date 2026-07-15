from __future__ import annotations as _annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from .._json_schema import JsonSchema, JsonSchemaTransformer
from ..exceptions import UserError
from ..native_tools import (
    CodeExecutionTool,
    FileSearchTool,
    ImageGenerationTool,
    MCPServerTool,
    WebSearchTool,
)
from ..native_tools._tool_search import ToolSearchTool
from ..settings import ThinkingLevel
from . import ModelProfile

_OPENAI_BASE_BUILTINS = frozenset(
    {WebSearchTool, CodeExecutionTool, FileSearchTool, MCPServerTool, ImageGenerationTool}
)
"""Builtin tool types OpenAI supports — the union of what `OpenAIChatModel` and
`OpenAIResponsesModel` can handle. `ToolSearchTool` is gated per-model in the
profile below."""

OPENAI_REASONING_EFFORT_MAP: dict[ThinkingLevel, str] = {
    True: 'medium',
    False: 'none',
    'minimal': 'minimal',
    'low': 'low',
    'medium': 'medium',
    'high': 'high',
    'xhigh': 'xhigh',
}
"""Maps unified thinking values to OpenAI reasoning_effort strings."""

SAMPLING_PARAMS = (
    'temperature',
    'top_p',
    'presence_penalty',
    'frequency_penalty',
    'logit_bias',
    'openai_logprobs',
    'openai_top_logprobs',
)
"""Sampling parameter names that are incompatible with reasoning.

These parameters are not supported when reasoning is enabled (reasoning_effort != 'none').
See https://platform.openai.com/docs/guides/reasoning for details.
"""

OpenAISystemPromptRole = Literal['system', 'developer', 'user']


@dataclass(frozen=True)
class _ReasoningSupport:
    """How an OpenAI model family supports reasoning, as three orthogonal facts."""

    enabled_by_default: bool
    """Reasoning is on when `reasoning_effort` is omitted (the model's default effort is active, e.g. `'medium'`)."""

    can_be_disabled: bool
    """The model accepts `reasoning_effort='none'`, which turns reasoning off and allows sampling parameters."""

    supports_mode: bool
    """The Responses API accepts `reasoning.mode` (`'standard' | 'pro'`) for the model."""

    @property
    def supported(self) -> bool:
        """Whether the model reasons at all."""
        return self.enabled_by_default or self.can_be_disabled

    @property
    def always_enabled(self) -> bool:
        """Whether the model reasons and can't be turned off."""
        return self.enabled_by_default and not self.can_be_disabled


_NO_REASONING = _ReasoningSupport(enabled_by_default=False, can_be_disabled=False, supports_mode=False)
"""The model doesn't reason at all."""

_OPT_IN_REASONING = _ReasoningSupport(enabled_by_default=False, can_be_disabled=True, supports_mode=False)
"""The model defaults to `reasoning_effort='none'` (reasoning off, sampling parameters allowed) and reasons on request."""

_ALWAYS_ON_REASONING = _ReasoningSupport(enabled_by_default=True, can_be_disabled=False, supports_mode=False)
"""The model always reasons; it doesn't accept `reasoning_effort='none'`."""

_REASONING_SUPPORT_BY_PREFIX: dict[str, _ReasoningSupport] = {
    # GPT-5.6 (sol/terra/luna) reasons by default (at 'medium'), accepts `effort='none'` to turn
    # reasoning off, and is the only family that supports `reasoning.mode`.
    'gpt-5.6': _ReasoningSupport(enabled_by_default=True, can_be_disabled=True, supports_mode=True),
    # gpt-5.5 reasons by default like gpt-5.6, but has no `reasoning.mode`.
    'gpt-5.5-pro': _ALWAYS_ON_REASONING,
    'gpt-5.5': _ReasoningSupport(enabled_by_default=True, can_be_disabled=True, supports_mode=False),
    'gpt-5.4-pro': _ALWAYS_ON_REASONING,
    'gpt-5.4': _OPT_IN_REASONING,
    # The GPT-5.1+ chat variants always reason at a fixed 'medium' effort: they reject sampling
    # parameters and any other `reasoning.effort` value, including 'none'.
    'gpt-5.3-chat': _ALWAYS_ON_REASONING,
    'gpt-5.3': _OPT_IN_REASONING,
    'gpt-5.2-pro': _ALWAYS_ON_REASONING,
    'gpt-5.2-chat': _ALWAYS_ON_REASONING,
    'gpt-5.2': _OPT_IN_REASONING,
    # Covers gpt-5.1-codex and gpt-5.1-codex-max; the gpt-5.3+ codex variants match their
    # mainline prefix instead (gpt-5.3-codex is opt-in like gpt-5.4).
    'gpt-5.1-codex': _ALWAYS_ON_REASONING,
    'gpt-5.1-chat': _ALWAYS_ON_REASONING,
    'gpt-5.1': _OPT_IN_REASONING,
    # gpt-5-chat-latest doesn't reason at all.
    'gpt-5-chat': _NO_REASONING,
    # The original GPT-5 family (incl. -mini/-pro/-codex) reasons at 'medium' by default.
    # See https://platform.openai.com/docs/guides/reasoning
    'gpt-5': _ALWAYS_ON_REASONING,
    # The o-series.
    'o': _ALWAYS_ON_REASONING,
}
"""Reasoning support per model-name prefix; the first matching prefix wins, so a more specific
prefix (e.g. `'gpt-5.3-chat'`) must be listed before the broader one it would otherwise match
(e.g. `'gpt-5.3'`), and every newer `gpt-5.x` family before the plain `'gpt-5'` catch-all.
Models that don't match any prefix don't reason. Every cell was verified against the live
Responses API (2026-07): a model reasons by default exactly when it rejects sampling parameters
with no `reasoning.effort` set, and can be disabled exactly when it accepts `effort='none'`.
The full resolved matrix is pinned in `tests/profiles/test_openai.py`."""


def _reasoning_support(model_name: str) -> _ReasoningSupport:
    return next(
        (support for prefix, support in _REASONING_SUPPORT_BY_PREFIX.items() if model_name.startswith(prefix)),
        _NO_REASONING,
    )


class OpenAIModelProfile(ModelProfile, total=False):
    """Profile for models used with `OpenAIChatModel`.

    ALL FIELDS MUST BE `openai_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.
    """

    openai_chat_thinking_field: str | None
    """Non-standard field name used by some providers for model thinking content in Chat Completions API responses. Default: `None`.

    Plenty of providers use custom field names for thinking content. Ollama and newer versions of vLLM use `reasoning`,
    while DeepSeek, older vLLM and some others use `reasoning_content`.

    Notice that the thinking field configured here is currently limited to `str` type content.

    If `openai_chat_send_back_thinking_parts` is set to `'field'`, this field must be set to a non-None value."""

    openai_chat_send_back_thinking_parts: Literal['auto', 'tags', 'field', False]
    """Whether the model includes thinking content in requests. Default: `'auto'`.

    This can be:
    * `'auto'` (default): Automatically detects how to send thinking content. If thinking was received in a custom field
    (tracked via `ThinkingPart.id` and `ThinkingPart.provider_name`), it's sent back in that same field. Otherwise,
    it's sent using tags. Only the `reasoning` and `reasoning_content` fields are checked by
    default when receiving responses. If your provider uses a different field name, you must explicitly set
    `openai_chat_thinking_field` to that field name.
    * `'tags'`: The thinking content is included in the main `content` field, enclosed within thinking tags as
    specified in `thinking_tags` profile option.
    * `'field'`: The thinking content is included in a separate field specified by `openai_chat_thinking_field`.
    * `False`: No thinking content is sent in the request.

    Defaults to `'auto'` to ensure thinking is sent back in the format expected by the model/provider."""

    openai_supports_strict_tool_definition: bool
    """This can be set by a provider or user if the OpenAI-"compatible" API doesn't support strict tool definitions. Default: `True`."""

    openai_unsupported_model_settings: Sequence[str]
    """A list of model settings that are not supported by this model. Default: `()`."""

    # Some OpenAI-compatible providers (e.g. MoonshotAI) currently do **not** accept
    # `tool_choice="required"`.  This flag lets the calling model know whether it's
    # safe to pass that value along.  Default is `True` to preserve existing
    # behaviour for OpenAI itself and most providers.
    openai_supports_tool_choice_required: bool
    """Whether the provider accepts the value `tool_choice='required'` in the request payload. Default: `True`."""

    openai_system_prompt_role: OpenAISystemPromptRole | None
    """The role to use for the system prompt message. If not provided, defaults to `'system'`."""

    openai_chat_supports_multiple_system_messages: bool
    """Whether the Chat Completions API accepts more than one system-role message at the start of the conversation. Default: `True`.

    OpenAI itself and most compatible providers accept multiple system messages, so this defaults to `True`.
    Set to `False` for strict OpenAI-compatible backends (e.g. some LiteLLM/vLLM deployments) that require
    exactly one initial system message; consecutive system messages at the start will be merged into one
    (joined with two newlines) before being sent."""

    openai_chat_supports_web_search: bool
    """Whether the model supports web search in Chat Completions API. Default: `False`."""

    openai_chat_audio_input_encoding: Literal['base64', 'uri']
    """The encoding to use for audio input in Chat Completions requests. Default: `'base64'`.

    - `'base64'`: Raw base64 encoded string. (Default, used by OpenAI)
    - `'uri'`: Data URI (e.g. `data:audio/wav;base64,...`).
    """

    openai_chat_supports_file_urls: bool
    """Whether the Chat API supports file URLs directly in the `file_data` field. Default: `False`.

    OpenAI's native Chat API only supports base64-encoded data, but some providers
    like OpenRouter support passing URLs directly.
    """

    openai_supports_encrypted_reasoning_content: bool
    """Whether the model supports including encrypted reasoning content in the response. Default: `False`."""

    openai_supports_reasoning: bool
    """Whether the model supports reasoning (o-series, GPT-5+). Default: `False`.

    When True, sampling parameters may need to be dropped depending on reasoning_effort setting."""

    openai_reasoning_enabled_by_default: bool
    """Whether the model reasons by default when `reasoning_effort` is omitted. Default: `False`.

    True for models whose default effort is active (e.g. 'medium'), such as the o-series, the original GPT-5,
    and GPT-5.5+, and False for the GPT-5.1..5.4 mainline models which default to `reasoning_effort='none'`.
    This decides whether sampling parameters must be dropped when no effort is set, and is independent of
    whether reasoning can be turned off (`openai_supports_reasoning_effort_none`)."""

    openai_supports_reasoning_effort_none: bool
    """Whether the model accepts `reasoning_effort='none'` and allows sampling parameters (temperature, top_p, etc.)
    while reasoning is off. Default: `False`.

    The GPT-5.1+ mainline models support turning reasoning off via `effort='none'`, and sampling params are
    accepted in that mode. When reasoning is enabled (low/medium/high/xhigh), sampling params are not supported.
    Whether the model reasons by default is tracked separately by `openai_reasoning_enabled_by_default`."""

    openai_responses_supports_reasoning_mode: bool
    """Whether the Responses API supports `reasoning.mode` (`'standard' | 'pro'`) for this model. Default: `False`.

    Currently only supported by the GPT-5.6 family."""

    openai_responses_requires_function_call_status_none: bool
    """Whether the Responses API requires the `status` field on function tool calls to be `None`. Default: `False`.

    This is required by vLLM Responses API versions before https://github.com/vllm-project/vllm/pull/26706.
    See https://github.com/pydantic/pydantic-ai/issues/3245 for more details.
    """

    openai_supports_phase: bool
    """Whether the Responses API supports the `phase` field on assistant messages. Default: `False`.

    `phase` labels an assistant message as intermediate `commentary` or the `final_answer`. When the model
    supports it, OpenAI recommends preserving and sending it back unchanged on every assistant message in
    follow-up requests; dropping it can cause preambles to be interpreted as final answers and degrade
    behavior in long-running or tool-heavy flows.

    Supported by `gpt-5.3-codex`, `gpt-5.4` and later mainline models. The official OpenAI Responses API
    silently ignores the field on older models, but defaults to `False` so we don't risk sending an
    unrecognized field to OpenAI-compatible APIs (vLLM, Bifrost, ...) that haven't been verified to accept it.
    """

    openai_chat_supports_document_input: bool
    """Whether the Chat Completions API supports document content parts (`type='file'`). Default: `True`.

    Some OpenAI-compatible providers (e.g. Azure) do not support document input via the Chat Completions API.
    """

    openai_chat_supports_max_completion_tokens: bool
    """Whether the Chat Completions API accepts the `max_completion_tokens` field for the `max_tokens` setting. Default: `True`.

    OpenAI itself (including the o-series reasoning models) uses `max_completion_tokens`, the field that caps
    visible output plus reasoning tokens, so this defaults to `True`. Many OpenAI-compatible providers (e.g.
    OpenRouter) only accept the older `max_tokens` field; set this to `False` for those so the `max_tokens`
    setting is sent as `max_tokens` instead.
    """


def validate_openai_profile(profile: ModelProfile) -> None:
    """Validate an OpenAI-compatible profile after resolution. Called from `OpenAIChatModel.__init__`."""
    if profile.get('openai_chat_send_back_thinking_parts') == 'field' and not profile.get('openai_chat_thinking_field'):
        raise UserError(
            'If `openai_chat_send_back_thinking_parts` is "field", '
            '`openai_chat_thinking_field` must be set to a non-None value.'
        )


def openai_model_profile(model_name: str) -> ModelProfile:
    """Get the model profile for an OpenAI model."""
    reasoning = _reasoning_support(model_name)

    # `phase` is supported by gpt-5.3-codex, gpt-5.4 and later mainline models, including gpt-5.6
    # (its responses label messages with `phase`, as recorded in the reasoning-mode cassette).
    # See https://developers.openai.com/api/docs/guides/prompt-guidance.
    supports_phase = model_name.startswith(('gpt-5.3-codex', 'gpt-5.4', 'gpt-5.5', 'gpt-5.6'))

    # The o1-mini model doesn't support the `system` role, so we default to `user`.
    # See https://github.com/pydantic/pydantic-ai/issues/974 for more details.
    openai_system_prompt_role = 'user' if model_name.startswith('o1-mini') else None

    # Check if the model supports web search (only specific search-preview models)
    supports_web_search = '-search-preview' in model_name
    supports_image_output = (
        model_name.startswith('gpt-5') or 'o3' in model_name or '4.1' in model_name or '4o' in model_name
    )

    # OpenAI's native `tool_search` tool with `defer_loading` is available on gpt-5.4 and later
    # mainline families (https://developers.openai.com/api/docs/guides/tools-tool-search; GPT-5.6
    # verified live). Like the other gates in this function, this enumerates known versions rather
    # than matching open-endedly, so a new family must be added here explicitly once confirmed;
    # until then it falls back to local search.
    supports_tool_search = model_name.startswith(('gpt-5.4', 'gpt-5.5', 'gpt-5.6'))
    supported_native_tools = _OPENAI_BASE_BUILTINS | {ToolSearchTool} if supports_tool_search else _OPENAI_BASE_BUILTINS

    # Structured Outputs (output mode 'native') is only supported with the gpt-4o-mini, gpt-4o-mini-2024-07-18,
    # and gpt-4o-2024-08-06 model snapshots and later. We leave it in here for all models because the
    # `default_structured_output_mode` is `'tool'`, so `native` is only used when the user specifically uses
    # the `NativeOutput` marker, so an error from the API is acceptable.
    return OpenAIModelProfile(
        json_schema_transformer=OpenAIJsonSchemaTransformer,
        supports_json_schema_output=True,
        supports_json_object_output=True,
        supports_image_output=supports_image_output,
        supports_inline_system_prompts=True,
        supports_thinking=reasoning.supported,
        thinking_always_enabled=reasoning.always_enabled,
        openai_system_prompt_role=openai_system_prompt_role,
        openai_chat_supports_web_search=supports_web_search,
        openai_supports_encrypted_reasoning_content=reasoning.supported,
        openai_supports_reasoning=reasoning.supported,
        openai_reasoning_enabled_by_default=reasoning.enabled_by_default,
        openai_supports_reasoning_effort_none=reasoning.can_be_disabled,
        openai_responses_supports_reasoning_mode=reasoning.supports_mode,
        openai_supports_phase=supports_phase,
        supported_native_tools=supported_native_tools,
    )


_STRICT_INCOMPATIBLE_KEYS = [
    'minLength',
    'maxLength',
    'patternProperties',
    'unevaluatedProperties',
    'propertyNames',
    'minProperties',
    'maxProperties',
    'unevaluatedItems',
    'contains',
    'minContains',
    'maxContains',
    'uniqueItems',
]

_STRICT_COMPATIBLE_STRING_FORMATS = [
    'date-time',
    'time',
    'date',
    'duration',
    'email',
    'hostname',
    'ipv4',
    'ipv6',
    'uuid',
]

_REGEX_LOOKAROUND_TOKENS = ('(?=', '(?!', '(?<=', '(?<!')

_sentinel = object()


def _regex_contains_lookaround(pattern: str) -> bool:
    escaped = False
    for i, char in enumerate(pattern):
        if escaped:
            escaped = False
            continue
        if char == '\\':
            escaped = True
            continue
        if pattern.startswith(_REGEX_LOOKAROUND_TOKENS, i):
            return True
    return False


@dataclass(init=False)
class OpenAIJsonSchemaTransformer(JsonSchemaTransformer):
    """Recursively handle the schema to make it compatible with OpenAI strict mode.

    See https://platform.openai.com/docs/guides/function-calling?api-mode=responses#strict-mode for more details,
    but this basically just requires:
    * `additionalProperties` must be set to false for each object in the parameters
    * all fields in properties must be marked as required
    """

    def __init__(self, schema: JsonSchema, *, strict: bool | None = None):
        super().__init__(schema, strict=strict)
        self.root_ref = schema.get('$ref')

    def walk(self) -> JsonSchema:
        # Note: OpenAI does not support anyOf at the root in strict mode
        # However, we don't need to check for it here because we ensure in pydantic_ai._utils.check_object_json_schema
        # that the root schema either has type 'object' or is recursive.
        result = super().walk()

        # For recursive models, we need to tweak the schema to make it compatible with strict mode.
        # Because the following should never change the semantics of the schema we apply it unconditionally.
        if self.root_ref is not None:
            result.pop('$ref', None)  # We replace references to the self.root_ref with just '#' in the transform method
            root_key = re.sub(r'^#/\$defs/', '', self.root_ref)
            result.update(self.defs.get(root_key) or {})

        return result

    def transform(self, schema: JsonSchema) -> JsonSchema:  # noqa: C901
        # Remove unnecessary keys
        schema.pop('title', None)
        schema.pop('$schema', None)
        schema.pop('discriminator', None)

        default = schema.get('default', _sentinel)
        if default is not _sentinel:
            # the "default" keyword is not allowed in strict mode, but including it makes some Ollama models behave
            # better, so we keep it around when not strict
            if self.strict is True:
                schema.pop('default', None)
            elif self.strict is None:  # pragma: no branch
                self.is_strict_compatible = False

        if schema_ref := schema.get('$ref'):
            if schema_ref == self.root_ref:
                schema['$ref'] = '#'
            if len(schema) > 1:
                # OpenAI Strict mode doesn't support siblings to "$ref", but _does_ allow siblings to "anyOf".
                # So if there is a "description" field or any other extra info, we move the "$ref" into an "anyOf":
                schema['anyOf'] = [{'$ref': schema.pop('$ref')}]

        # Track strict-incompatible keys
        incompatible_values: dict[str, Any] = {}
        for key in _STRICT_INCOMPATIBLE_KEYS:
            value = schema.get(key, _sentinel)
            if value is not _sentinel:
                incompatible_values[key] = value
        if format := schema.get('format'):
            if format not in _STRICT_COMPATIBLE_STRING_FORMATS:
                incompatible_values['format'] = format
        pattern = schema.get('pattern')
        if isinstance(pattern, str) and _regex_contains_lookaround(pattern):
            incompatible_values['pattern'] = pattern
        description = schema.get('description')
        if incompatible_values:
            if self.strict is True:
                notes: list[str] = []
                for key, value in incompatible_values.items():
                    schema.pop(key)
                    notes.append(f'{key}={value}')
                notes_string = ', '.join(notes)
                schema['description'] = notes_string if not description else f'{description} ({notes_string})'
            elif self.strict is None:  # pragma: no branch
                self.is_strict_compatible = False

        schema_type = schema.get('type')
        if 'oneOf' in schema:
            # OpenAI does not support oneOf in strict mode
            if self.strict is True:
                schema['anyOf'] = schema.pop('oneOf')
            else:
                self.is_strict_compatible = False

        if schema_type == 'object':
            # Always ensure 'properties' key exists - OpenAI drops objects without it
            if 'properties' not in schema:
                schema['properties'] = dict[str, Any]()

            if self.strict is True:
                # additional properties are disallowed
                schema['additionalProperties'] = False

                # all properties are required
                schema['required'] = list(schema['properties'].keys())

            elif self.strict is None:
                if schema.get('additionalProperties', None) not in (None, False):
                    self.is_strict_compatible = False
                else:
                    # additional properties are disallowed by default
                    schema['additionalProperties'] = False

                if 'properties' not in schema or 'required' not in schema:
                    self.is_strict_compatible = False
                else:
                    required = schema['required']
                    for k in schema['properties'].keys():
                        if k not in required:
                            self.is_strict_compatible = False
        return schema
