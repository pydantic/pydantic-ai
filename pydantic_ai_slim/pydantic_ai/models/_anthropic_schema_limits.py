"""Anthropic's explicit complexity limits on the strict schemas in one request.

Anthropic compiles every strict schema in a request (each tool sent with `strict: true`, and the JSON schema output
format) into a grammar, and rejects a request whose strict schemas together exceed these limits with a 400:
https://platform.claude.com/docs/en/build-with-claude/structured-outputs#schema-complexity-limits

The counts mirror what the API was observed to count, which is less than a literal reading of its docs: only the
parameters reached without following a `$ref`, so the parameters of a model under `$defs` don't count, and only
parameters whose own schema has `anyOf` or a list of types count as union-typed. Anything uncertain is left uncounted,
so a request the API would accept is never refused here.

`AnthropicModel` and `BedrockConverseModel` (for Claude) share this, so it lives outside `models/anthropic/`, which
`models/bedrock.py` can't import from without the `anthropic` SDK.
"""

from __future__ import annotations as _annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import replace
from typing import cast

from .._json_schema import DEFINITION_KEYS, JsonSchema, inline_objects, split_definitions
from .._utils import is_str_dict
from ..exceptions import ModelRequestRejectedError
from ..output import StructuredOutputMode
from ..profiles.anthropic import _COUNTING_STRICT_SCHEMAS  # pyright: ignore[reportPrivateUsage]
from ..settings import ModelSettings
from ..tools import ToolDefinition
from . import Model, ModelRequestParameters

MAX_STRICT_TOOLS = 20
MAX_OPTIONAL_PARAMETERS = 24
MAX_UNION_PARAMETERS = 16

_DOCS_URL = 'https://pydantic.dev/docs/ai/models/anthropic/#schema-complexity-limits'

SCHEMA_TOO_COMPLEX_HINT = (
    "Anthropic couldn't compile the request's strict schemas into a grammar. Simplify them, set `strict=True` only "
    'on the tools that need it, or use `ToolOutput` without `strict=True` or `PromptedOutput` for a large output '
    f'type. See {_DOCS_URL}'
)
"""Appended to the 400 Anthropic sends when strict schemas exceed its internal grammar-size limits."""

_SCHEMA_TOO_COMPLEX_MESSAGES = ('schema is too complex', 'compiled grammar is too large')


def is_schema_too_complex_error(message: object) -> bool:
    """Whether an error message is Anthropic's for strict schemas that are too complex to compile."""
    return isinstance(message, str) and any(text in message.lower() for text in _SCHEMA_TOO_COMPLEX_MESSAGES)


def exceeded_limit(strict_tool_schemas: Sequence[JsonSchema], output_schema: JsonSchema | None) -> str | None:
    """Describe the first limit that these strict schemas, sent in one request, exceed, or `None` if they fit."""
    if (tools := len(strict_tool_schemas)) > MAX_STRICT_TOOLS:
        return f'{tools} strict tools, over the limit of {MAX_STRICT_TOOLS}'
    schemas = [*strict_tool_schemas, *([output_schema] if output_schema is not None else [])]
    if any(_is_recursive(schema) for schema in schemas):
        return 'a recursive schema, which strict mode does not support'
    optional = union = 0
    for schema in schemas:
        # Only what's reached without following a `$ref` counts, as the API was observed to count.
        for node, _ in inline_objects(schema):
            required = node.get('required', [])
            for name, prop in node['properties'].items():
                if name not in required:
                    optional += 1
                if is_str_dict(prop) and ('anyOf' in prop or isinstance(prop.get('type'), list)):
                    union += 1
    if optional > MAX_OPTIONAL_PARAMETERS:
        return f'{optional} optional parameters, over the limit of {MAX_OPTIONAL_PARAMETERS}'
    if union > MAX_UNION_PARAMETERS:
        return f'{union} parameters with union types, over the limit of {MAX_UNION_PARAMETERS}'
    return None


def request_exceeded_limit(params: ModelRequestParameters, sent_strict_tools: Iterable[ToolDefinition]) -> str | None:
    """`exceeded_limit` for a request: the tools it sends with `strict`, and its output schema in native mode."""
    output_object = params.output_object if params.output_mode == 'native' else None
    output_schema = output_object.json_schema if output_object is not None else None
    return exceeded_limit([t.parameters_json_schema for t in sent_strict_tools], output_schema)


SentTools = Callable[[ModelSettings | None, ModelRequestParameters], Iterable[ToolDefinition]]
"""The tool definitions an adapter sends for prepared request parameters, after any `tool_choice` restriction."""


def model_exceeded_limit(
    model: Model,
    model_settings: ModelSettings | None,
    params: ModelRequestParameters,
    *,
    supports_strict_tools: Callable[[], bool],
    sent_tools: SentTools,
) -> str | None:
    """The limit a model's request goes over, or `None` if it fits or the model's API doesn't enforce the limits.

    `supports_strict_tools` says whether the adapter sends `strict` on a tool at all, and is only asked when a tool is
    strict. Anthropic counts every strict tool in the request whatever the `tool_choice`, so the count covers exactly
    the tools `sent_tools` returns.
    """
    if not model.profile.get('anthropic_enforces_strict_schema_limits', False):
        return None
    strict_tools: list[ToolDefinition] = []
    if any(tool.strict for tool in params.declared_tool_defs.values()) and supports_strict_tools():
        strict_tools = [tool for tool in sent_tools(model_settings, params) if tool.strict]
    return request_exceeded_limit(params, strict_tools)


def default_output_mode_within_limits(
    model: Model,
    mode: StructuredOutputMode,
    model_settings: ModelSettings | None,
    params: ModelRequestParameters,
    exceeded: Callable[[ModelSettings | None, ModelRequestParameters], str | None],
) -> StructuredOutputMode:
    """`mode`, or Tool Output where `mode` is a default Native Output whose schemas would go over Anthropic's limits.

    Non-strict tools don't count toward the limits, and where forcing the output tool would stop the request from
    thinking, the adapter offers it without forcing, as on any request that thinks. Native Output is only ever the
    default where output tools exist, so Prompted Output is never needed as a last resort.

    The schemas are transformed here as `Model.prepare_request` would for Native Output, only to count what Anthropic
    would compile, so the transformer doesn't warn about a strict schema that may not be sent.
    """
    if (
        mode != 'native'
        or params.output_mode != 'auto'
        or not model.profile.get('anthropic_enforces_strict_schema_limits', False)
    ):
        return mode
    assert params.output_object is not None
    candidate = params.with_default_output_mode('native')
    candidate = replace(candidate, output_object=replace(params.output_object, strict=True))
    token = _COUNTING_STRICT_SCHEMAS.set(True)
    try:
        candidate = model.customize_request_parameters(candidate)
    finally:
        _COUNTING_STRICT_SCHEMAS.reset(token)
    return 'tool' if exceeded(model_settings, candidate) is not None else mode


def limit_exceeded_error(exceeded: str, model_name: str) -> ModelRequestRejectedError:
    """The error for a request that explicitly asks for strict schemas over Anthropic's limits.

    It's raised before the request is sent, as the same category as the 400 Anthropic would send, so
    `FallbackModel` and `except ModelAPIError` handle it as they handle that 400.
    """
    return ModelRequestRejectedError(
        model_name,
        f"The strict schemas in this request exceed Anthropic's schema complexity limits, which count all strict "
        f'schemas in a request together: they have {exceeded}. Strict schemas are those of tools with `strict=True` '
        'and of `NativeOutput`. Set `strict=True` only on the tools that need it, simplify the schemas, or use '
        f'`ToolOutput` without `strict=True` or `PromptedOutput` for the output type. See {_DOCS_URL}',
    )


def _is_recursive(schema: JsonSchema) -> bool:
    """Whether a definition reachable from `schema`, or `schema` itself, references itself through `$ref`s."""
    definitions: dict[str, object] = {'#': split_definitions(schema)[0]}
    for key in DEFINITION_KEYS:
        if is_str_dict(group := schema.get(key)):
            definitions.update({f'#/{key}/{name}': value for name, value in group.items()})
    visiting: set[str] = set()
    done: set[str] = set()

    def has_cycle(ref: str) -> bool:
        if ref in visiting:
            return True
        if ref in done or ref not in definitions:
            return False
        visiting.add(ref)
        found = any(has_cycle(child) for child in _refs(definitions[ref]))
        visiting.discard(ref)
        done.add(ref)
        return found

    return has_cycle('#')


def _refs(node: object) -> Iterator[str]:
    """Every `$ref` in `node`, at any depth."""
    if is_str_dict(node):
        for key, value in node.items():
            if key == '$ref' and isinstance(value, str):
                yield value
            else:
                yield from _refs(value)
    elif isinstance(node, list):
        for value in cast(list[object], node):
            yield from _refs(value)
