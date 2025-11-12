from __future__ import annotations as _annotations

import re
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from .._json_schema import JsonSchema, JsonSchemaTransformer
from . import ModelProfile

OpenAISystemPromptRole = Literal['system', 'developer', 'user']


@dataclass(kw_only=True)
class OpenAIModelProfile(ModelProfile):
    """Profile for models used with `OpenAIChatModel`.

    ALL FIELDS MUST BE `openai_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.
    """

    openai_supports_strict_tool_definition: bool = True
    """This can be set by a provider or user if the OpenAI-"compatible" API doesn't support strict tool definitions."""

    openai_supports_sampling_settings: bool = True
    """Turn off to don't send sampling settings like `temperature` and `top_p` to models that don't support them, like OpenAI's o-series reasoning models."""

    openai_unsupported_model_settings: Sequence[str] = ()
    """A list of model settings that are not supported by this model."""

    # Some OpenAI-compatible providers (e.g. MoonshotAI) currently do **not** accept
    # `tool_choice="required"`.  This flag lets the calling model know whether it's
    # safe to pass that value along.  Default is `True` to preserve existing
    # behaviour for OpenAI itself and most providers.
    openai_supports_tool_choice_required: bool = True
    """Whether the provider accepts the value ``tool_choice='required'`` in the request payload."""

    openai_system_prompt_role: OpenAISystemPromptRole | None = None
    """The role to use for the system prompt message. If not provided, defaults to `'system'`."""

    openai_chat_supports_web_search: bool = False
    """Whether the model supports web search in Chat Completions API."""

    openai_supports_encrypted_reasoning_content: bool = False
    """Whether the model supports including encrypted reasoning content in the response."""

    openai_responses_requires_function_call_status_none: bool = False
    """Whether the Responses API requires the `status` field on function tool calls to be `None`.

    This is required by vLLM Responses API versions before https://github.com/vllm-project/vllm/pull/26706.
    See https://github.com/pydantic/pydantic-ai/issues/3245 for more details.
    """

    def __post_init__(self):  # pragma: no cover
        if not self.openai_supports_sampling_settings:
            warnings.warn(
                'The `openai_supports_sampling_settings` has no effect, and it will be removed in future versions. '
                'Use `openai_unsupported_model_settings` instead.',
                DeprecationWarning,
            )


def openai_model_profile(model_name: str) -> ModelProfile:
    """Get the model profile for an OpenAI model."""
    is_gpt_5 = model_name.startswith('gpt-5')
    is_o_series = model_name.startswith('o')
    is_reasoning_model = is_o_series or (is_gpt_5 and 'gpt-5-chat' not in model_name)

    # Check if the model supports web search (only specific search-preview models)
    supports_web_search = '-search-preview' in model_name

    # Structured Outputs (output mode 'native') is only supported with the gpt-4o-mini, gpt-4o-mini-2024-07-18, and gpt-4o-2024-08-06 model snapshots and later.
    # We leave it in here for all models because the `default_structured_output_mode` is `'tool'`, so `native` is only used
    # when the user specifically uses the `NativeOutput` marker, so an error from the API is acceptable.

    if is_reasoning_model:
        openai_unsupported_model_settings = (
            'temperature',
            'top_p',
            'presence_penalty',
            'frequency_penalty',
            'logit_bias',
            'logprobs',
            'top_logprobs',
        )
    else:
        openai_unsupported_model_settings = ()

    # The o1-mini model doesn't support the `system` role, so we default to `user`.
    # See https://github.com/pydantic/pydantic-ai/issues/974 for more details.
    openai_system_prompt_role = 'user' if model_name.startswith('o1-mini') else None

    return OpenAIModelProfile(
        json_schema_transformer=OpenAIJsonSchemaTransformer,
        supports_json_schema_output=True,
        supports_json_object_output=True,
        supports_image_output=is_gpt_5 or 'o3' in model_name or '4.1' in model_name or '4o' in model_name,
        openai_unsupported_model_settings=openai_unsupported_model_settings,
        openai_system_prompt_role=openai_system_prompt_role,
        openai_chat_supports_web_search=supports_web_search,
        openai_supports_encrypted_reasoning_content=is_reasoning_model,
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

_sentinel = object()


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

    def transform(self, schema: JsonSchema) -> JsonSchema:  # noqa C901
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
            if self.strict is True:
                # additional properties are disallowed
                schema['additionalProperties'] = False

                # all properties are required
                if 'properties' not in schema:
                    schema['properties'] = dict[str, Any]()
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
