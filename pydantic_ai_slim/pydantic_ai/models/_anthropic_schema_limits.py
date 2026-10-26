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

from collections.abc import Iterator, Sequence
from typing import Any, cast

from .._json_schema import JsonSchema
from .._utils import is_str_dict
from ..exceptions import UserError
from ..settings import ModelSettings
from . import ModelRequestParameters
from ._tool_choice import resolve_tool_choice

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

_DEFINITIONS = ('$defs', 'definitions')


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
        for properties, required in _inline_objects(schema):
            for name, prop in properties.items():
                if name not in required:
                    optional += 1
                if is_str_dict(prop) and ('anyOf' in prop or isinstance(prop.get('type'), list)):
                    union += 1
    if optional > MAX_OPTIONAL_PARAMETERS:
        return f'{optional} optional parameters, over the limit of {MAX_OPTIONAL_PARAMETERS}'
    if union > MAX_UNION_PARAMETERS:
        return f'{union} parameters with union types, over the limit of {MAX_UNION_PARAMETERS}'
    return None


def request_exceeded_limit(
    model_settings: ModelSettings | None,
    params: ModelRequestParameters,
    *,
    supports_strict_tools: bool,
) -> str | None:
    """`exceeded_limit` for the strict schemas a request sends: its strict tools, and the output schema in native mode.

    `supports_strict_tools` says whether the adapter sends `strict` on a tool at all. A `tool_choice` that keeps tools
    out of the request is only resolved once the request is over a limit, and only ever leaves tools uncounted.
    """
    strict_tools = [t for t in params.declared_tool_defs.values() if t.strict] if supports_strict_tools else []
    output_object = params.output_object if params.output_mode == 'native' else None
    output_schema = output_object.json_schema if output_object is not None else None
    exceeded = exceeded_limit([t.parameters_json_schema for t in strict_tools], output_schema)
    if exceeded is None or not strict_tools:
        return exceeded
    tool_choice = resolve_tool_choice(model_settings, params)
    if tool_choice == 'none':
        strict_tools = []
    elif isinstance(tool_choice, tuple):
        strict_tools = [t for t in strict_tools if t.name in tool_choice[1]]
    return exceeded_limit([t.parameters_json_schema for t in strict_tools], output_schema)


def limit_exceeded_error(exceeded: str) -> UserError:
    """The error for a request that explicitly asks for strict schemas over Anthropic's limits."""
    return UserError(
        f"The strict schemas in this request exceed Anthropic's schema complexity limits, which count all strict "
        f'schemas in a request together: they have {exceeded}. Strict schemas are those of tools with `strict=True` '
        'and of `NativeOutput`. Set `strict=True` only on the tools that need it, simplify the schemas, or use '
        f'`ToolOutput` without `strict=True` or `PromptedOutput` for the output type. See {_DOCS_URL}'
    )


def _inline_objects(schema: object) -> Iterator[tuple[dict[str, Any], list[Any]]]:
    """The `properties` and `required` of each object reached from `schema` through properties, array items and `anyOf`.

    `$ref`s aren't followed and `$defs` aren't entered, as the API doesn't count them.
    """
    if not is_str_dict(schema):
        return
    children: list[object] = [schema.get('items'), *schema.get('anyOf', [])]
    if is_str_dict(properties := schema.get('properties')):
        yield properties, schema.get('required', [])
        children.extend(properties.values())
    for child in children:
        yield from _inline_objects(child)


def _is_recursive(schema: JsonSchema) -> bool:
    """Whether a definition reachable from `schema`, or `schema` itself, references itself through `$ref`s."""
    definitions: dict[str, object] = {'#': {key: value for key, value in schema.items() if key not in _DEFINITIONS}}
    for key in _DEFINITIONS:
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
