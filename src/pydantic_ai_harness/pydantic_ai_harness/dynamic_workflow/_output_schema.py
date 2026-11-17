"""Per-call output schemas a workflow script passes to a sub-agent, and their validation."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Literal, TypeGuard

from pydantic import ConfigDict, JsonValue, TypeAdapter, ValidationError, with_config
from typing_extensions import TypedDict

from pydantic_ai import ModelRetry, StructuredDict

_MAX_ERRORS = 10
_MAX_SCHEMA_VALUES = 10_000

JsonObject = dict[str, JsonValue]
_JsonType = Literal['object', 'array', 'string', 'integer', 'number', 'boolean', 'null']


# Only keywords `schema_errors` enforces, so none is left unenforced. References are left out: they
# would be inlined on the host, where they can expand a small schema without bound.
@with_config(ConfigDict(extra='forbid'))
class OutputSchema(TypedDict, total=False):
    """The JSON schema subset a workflow script can pass as `output_schema`."""

    type: _JsonType | list[_JsonType]
    properties: dict[str, OutputSchema]
    required: list[str]
    additionalProperties: bool | OutputSchema
    items: OutputSchema
    enum: list[JsonValue]
    const: JsonValue
    anyOf: list[OutputSchema]
    title: str
    description: str
    default: JsonValue
    examples: list[JsonValue]


_OUTPUT_SCHEMA_ADAPTER = TypeAdapter(OutputSchema)


def _is_dict(value: object) -> TypeGuard[dict[object, object]]:
    return isinstance(value, dict)


def _is_list(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


def _check_size(schema: object) -> None:
    """Raise `ValueError` if `schema` holds more than `_MAX_SCHEMA_VALUES` values, counting each reuse.

    A script can reuse one dict at many places, so a schema that is small in the sandbox can expand into
    exponentially many nodes once copied or validated against on the host.
    """
    stack = [schema]
    count = 0
    while stack:
        count += 1
        if count > _MAX_SCHEMA_VALUES:
            raise ValueError(f'it has more than {_MAX_SCHEMA_VALUES} values')
        node = stack.pop()
        if _is_dict(node):
            stack.extend(node.values())
        elif _is_list(node):
            stack.extend(node)


def parse_output_schema(schema: object) -> OutputSchema:
    """Validate `schema` as an object `OutputSchema`, raising `ValueError` with the reasons it is not one."""
    _check_size(schema)
    try:
        parsed = _OUTPUT_SCHEMA_ADAPTER.validate_python(schema, strict=True)
    except ValidationError as exc:
        reasons = '; '.join(
            f'{".".join(str(part) for part in error["loc"]) or "schema"}: {error["msg"]}'
            for error in exc.errors()[:_MAX_ERRORS]
        )
        raise ValueError(reasons) from exc
    if parsed.get('type') != 'object':
        raise ValueError('type: must be "object"')
    return parsed


_TYPE_CHECKS: dict[_JsonType, Callable[[JsonValue], bool]] = {
    'object': lambda value: isinstance(value, dict),
    'array': lambda value: isinstance(value, list),
    'string': lambda value: isinstance(value, str),
    'integer': lambda value: (
        (isinstance(value, int) and not isinstance(value, bool)) or (isinstance(value, float) and value.is_integer())
    ),
    'number': lambda value: isinstance(value, int | float) and not isinstance(value, bool),
    'boolean': lambda value: isinstance(value, bool),
    'null': lambda value: value is None,
}


def _json_equal(a: JsonValue, b: JsonValue) -> bool:
    """JSON Schema equality: unlike Python's, it keeps `true` and `1` apart, and like it, `1` and `1.0` equal."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_json_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_json_equal(value, b[key]) for key, value in a.items())
    return a == b


def _value_error(value: JsonValue, schema: OutputSchema) -> str | None:
    """Why `value` itself does not match `schema`, ignoring what it contains."""
    type_ = schema.get('type')
    types: list[_JsonType] = type_ if isinstance(type_, list) else [type_] if type_ is not None else []
    if types and not any(_TYPE_CHECKS[t](value) for t in types):
        return f'expected {" or ".join(types)}'
    if 'enum' in schema and not any(_json_equal(value, option) for option in schema['enum']):
        return f'must be one of {json.dumps(schema["enum"])}'
    if 'const' in schema and not _json_equal(value, schema['const']):
        return f'must be {json.dumps(schema["const"])}'
    if 'anyOf' in schema and all(schema_errors(value, option) for option in schema['anyOf']):
        return 'matches none of the anyOf options'
    return None


def schema_errors(value: JsonValue, schema: OutputSchema, path: str = '$') -> list[str]:
    """Where `value` does not match `schema`."""
    if (error := _value_error(value, schema)) is not None:
        return [f'{path}: {error}']
    if isinstance(value, dict):
        return _object_errors(value, schema, path)
    if isinstance(value, list) and 'items' in schema:
        items = schema['items']
        return [error for index, item in enumerate(value) for error in schema_errors(item, items, f'{path}[{index}]')]
    return []


def _object_errors(value: JsonObject, schema: OutputSchema, path: str) -> list[str]:
    errors = [f'{path}.{name}: required' for name in schema.get('required', []) if name not in value]
    properties = schema.get('properties', {})
    additional = schema.get('additionalProperties', True)
    for name, item in value.items():
        subschema = properties.get(name, additional)
        if subschema is False:
            errors.append(f'{path}.{name}: additional property not allowed')
        elif subschema is not True:
            errors.extend(schema_errors(item, subschema, f'{path}.{name}'))
    return errors


def schema_output(schema: OutputSchema) -> Callable[[JsonObject], JsonObject]:
    """An output function whose tool takes `schema` as is, and asks the model to retry an output that does not match it."""

    def final_result(output: JsonObject) -> JsonObject:
        errors = schema_errors(output, schema)
        if errors:
            shown = '\n'.join(errors[:_MAX_ERRORS])
            raise ModelRetry(f'Output does not match the required schema:\n{shown}')
        return output

    # Annotated with the schema's own type, the parameter is unwrapped to the tool's top level.
    final_result.__annotations__ = {'output': StructuredDict(dict(schema)), 'return': JsonObject}
    return final_result
