"""Per-call output schemas a workflow script passes to a sub-agent, and their validation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from pydantic import ConfigDict, JsonValue, TypeAdapter, ValidationError, with_config
from typing_extensions import TypedDict

from pydantic_ai import ModelRetry, StructuredDict

_MAX_ERRORS = 10

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


def parse_output_schema(schema: object) -> OutputSchema:
    """Validate `schema` as an object `OutputSchema`, raising `ValueError` with the reasons it is not one."""
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


def _matches_type(value: JsonValue, type_: _JsonType) -> bool:
    match type_:
        case 'object':
            return isinstance(value, dict)
        case 'array':
            return isinstance(value, list)
        case 'string':
            return isinstance(value, str)
        case 'integer':
            return isinstance(value, int) and not isinstance(value, bool)
        case 'number':
            return isinstance(value, int | float) and not isinstance(value, bool)
        case 'boolean':
            return isinstance(value, bool)
    return value is None


def _value_error(value: JsonValue, schema: OutputSchema) -> str | None:
    """Why `value` itself does not match `schema`, ignoring what it contains."""
    type_ = schema.get('type')
    types: list[_JsonType] = type_ if isinstance(type_, list) else [type_] if type_ is not None else []
    if types and not any(_matches_type(value, t) for t in types):
        return f'expected {" or ".join(types)}'
    if 'enum' in schema and value not in schema['enum']:
        return f'must be one of {schema["enum"]}'
    if 'const' in schema and value != schema['const']:
        return f'must be {schema["const"]!r}'
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
