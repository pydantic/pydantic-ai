"""Per-call output schemas a workflow script passes to a sub-agent."""

from __future__ import annotations

from typing import Literal, TypeGuard

from pydantic import ConfigDict, JsonValue, TypeAdapter, ValidationError, with_config
from pydantic.json_schema import JsonSchemaValue
from typing_extensions import TypedDict

from pydantic_ai import StructuredDict

_MAX_ERRORS = 10
_MAX_SCHEMA_VALUES = 10_000

_JsonType = Literal['object', 'array', 'string', 'integer', 'number', 'boolean', 'null']


# The schema is model output that is processed and validated against on the host, so references, which
# would be inlined there, and regular expressions, which can backtrack, are left out.
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
    allOf: list[OutputSchema]
    oneOf: list[OutputSchema]
    minimum: float
    maximum: float
    exclusiveMinimum: float
    exclusiveMaximum: float
    multipleOf: float
    minLength: int
    maxLength: int
    minItems: int
    maxItems: int
    uniqueItems: bool
    minProperties: int
    maxProperties: int
    format: str
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
    exponentially many values once copied or validated against on the host.
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


def output_schema_type(schema: object) -> type[JsonSchemaValue]:
    """The output type for a validated `output_schema`, raising `ValueError` with the reasons `schema` is not one."""
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
    return StructuredDict(_OUTPUT_SCHEMA_ADAPTER.dump_python(parsed), validate=True)
