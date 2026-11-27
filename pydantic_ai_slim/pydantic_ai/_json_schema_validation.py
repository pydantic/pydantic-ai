"""Validating output against a JSON schema for `StructuredDict(validate=True)`, using the `jsonschema` package.

Patterns are matched with pydantic-core's linear-time regex engine rather than Python's backtracking `re`, so crafted
model output can't stall the event loop.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Iterator
from contextvars import ContextVar
from typing import Any, cast

from pydantic.json_schema import JsonSchemaValue
from pydantic_core import PydanticCustomError, SchemaError, SchemaValidator, core_schema

from . import exceptions

__all__ = ('VALIDATING_PARTIAL_OUTPUT', 'json_schema_validator')


def json_schema_validator(json_schema: JsonSchemaValue) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Build a function that checks a value against `json_schema` and raises a validation error listing what's wrong."""
    try:
        from jsonschema.exceptions import SchemaError as SchemaValidationError
        from jsonschema.validators import (
            extend,  # pyright: ignore[reportUnknownVariableType]  # `extend` has no annotations
            validator_for,
        )
        from referencing import Registry
        from referencing.exceptions import Unresolvable
    except ImportError as e:
        raise exceptions.UserError(
            '`StructuredDict(validate=True)` requires the `jsonschema` package. '
            'Install it with `pip install "pydantic-ai-slim[jsonschema]"` (or `"pydantic-ai[jsonschema]"`).'
        ) from e

    base_validator: Any = validator_for(json_schema)
    try:
        base_validator.check_schema(json_schema)
    except SchemaValidationError as e:
        raise exceptions.UserError(f'`StructuredDict(validate=True)` got an invalid JSON schema: {e.message}') from e
    matchers = _compile_schema_patterns(json_schema)
    # `pattern` checks use pydantic-core's linear-time regex engine rather than Python's backtracking `re`, so
    # crafted model output can't stall the event loop on a pattern like `^(a+)+$`.
    validator_class: Any = cast(Any, extend)(
        base_validator,
        {
            'pattern': _pattern_keyword(matchers),
            'patternProperties': _pattern_properties_keyword(matchers),
            'additionalProperties': _additional_properties_keyword(matchers),
        },
    )
    # An empty registry resolves references inside the schema and never fetches a remote one.
    validator: Any = validator_class(json_schema, registry=Registry[Any]())

    def validate(value: dict[str, Any]) -> dict[str, Any]:
        if VALIDATING_PARTIAL_OUTPUT.get():
            # A partial object rarely meets `required` and the like yet, so only the complete output is checked.
            return value
        # JSON has no NaN or infinity, and they pass `minimum`/`maximum` checks, so they're refused first. Some
        # keywords, like `multipleOf`, would raise on NaN instead of reporting an error.
        problems = [f'at `{path}`: {number!r} is not a finite number' for path, number in _non_finite_numbers(value)]
        if not problems:
            problems = _schema_problems(validator, value, Unresolvable)
        if problems:
            raise PydanticCustomError(
                'json_schema_mismatch',
                'Output does not match the JSON schema: {problems}',
                {'problems': '; '.join(problems)},
            )
        return value

    return validate


def _schema_problems(validator: Any, value: dict[str, Any], unresolvable: type[Exception]) -> list[str]:
    """What's wrong with `value` according to `validator`, one line per problem."""
    try:
        return [_error_message(error) for error in sorted(validator.iter_errors(value), key=_error_path)]
    except unresolvable as e:
        raise exceptions.UserError(
            f'`StructuredDict(validate=True)` can only resolve references inside the schema, not {getattr(e, "ref", e)!r}.'
        ) from e
    except OverflowError:
        # `multipleOf` divides by a float, which overflows for an integer beyond the float range.
        return ['a number is too large to check against the schema']


def _error_path(error: Any) -> list[str]:
    return [str(part) for part in error.absolute_path]


def _error_message(error: Any) -> str:
    """A `jsonschema` validation error as one line, prefixed with where it is unless it's at the top level.

    For `anyOf`/`oneOf`, it adds the closest branch's error, so the model has something concrete to fix.
    """
    path = '.'.join(_error_path(error))
    message = f'at `{path}`: {error.message}' if path else error.message
    if error.context:
        from jsonschema.exceptions import best_match  # pyright: ignore[reportUnknownVariableType]

        closest: Any = cast(Any, best_match)(error.context)  # `jsonschema` has no type information
        message += f' (closest: {_error_message(closest)})'
    return message


VALIDATING_PARTIAL_OUTPUT: ContextVar[bool] = ContextVar('pydantic_ai.output.validating_partial', default=False)
"""Whether output is being validated as a partial, streamed result, which `StructuredDict(validate=True)` skips."""


_DATA_KEYWORDS = frozenset({'enum', 'const', 'default', 'examples', 'dependentRequired'})
"""Keywords whose values are data rather than subschemas, so a `pattern` key inside them isn't a pattern."""

_SCHEMA_MAP_KEYWORDS = frozenset(
    {'properties', 'patternProperties', '$defs', 'definitions', 'dependentSchemas', 'dependencies'}
)
"""Keywords whose values map names to subschemas, so their keys are names, not keywords."""


_PatternMatchers = dict[str, Callable[[str], bool]]
"""Matchers for every pattern in a schema, compiled with pydantic-core's linear-time regex engine."""


def _compile_schema_patterns(json_schema: JsonSchemaValue) -> _PatternMatchers:
    """Compile every `pattern` and `patternProperties` key in `json_schema` with pydantic-core's Rust regex engine.

    Raises a `UserError` naming the pattern when the engine can't compile it, such as a lookaround or a backreference,
    and when `unevaluatedProperties` is used with `patternProperties`, which `jsonschema` matches with Python's `re`.
    """
    matchers: _PatternMatchers = {pattern: _compile_pattern(pattern) for pattern in _schema_patterns(json_schema)}
    if matchers and _uses_keyword(json_schema, 'unevaluatedProperties'):
        raise exceptions.UserError(
            '`StructuredDict(validate=True)` does not support `unevaluatedProperties` together with `pattern` or '
            '`patternProperties`, because it could not check their property names in linear time.'
        )
    if any(_uses_keyword(subschema, '$schema') for subschema in _subschemas(json_schema)):
        # A nested `$schema` makes `jsonschema` switch to its stock validator, which checks patterns with Python's `re`.
        raise exceptions.UserError(
            '`StructuredDict(validate=True)` supports `$schema` only at the top level of the schema, not in a subschema.'
        )
    return matchers


def _compile_pattern(pattern: str) -> Callable[[str], bool]:
    try:
        validator = SchemaValidator(core_schema.str_schema(pattern=pattern, regex_engine='rust-regex'))
    except SchemaError as e:
        raise exceptions.UserError(
            f'`StructuredDict(validate=True)` checks `pattern` with a linear-time regex engine, which does not '
            f'support the pattern {pattern!r} (lookarounds and backreferences are not supported): {e}'
        ) from e
    return validator.isinstance_python


def _subschemas(schema: dict[str, Any]) -> Iterator[Any]:
    """The subschemas directly inside `schema`, skipping keywords whose values are data."""
    for key, value in schema.items():
        if key in _DATA_KEYWORDS:
            continue
        if key in _SCHEMA_MAP_KEYWORDS and isinstance(value, dict):
            yield from cast(dict[str, Any], value).values()
        elif isinstance(value, list):
            yield from cast(list[Any], value)
        elif isinstance(value, dict):
            yield value


def _schema_patterns(schema: Any) -> Iterator[str]:
    """Every `pattern` and `patternProperties` key in a JSON schema."""
    if not isinstance(schema, dict):
        return
    schema = cast(dict[str, Any], schema)
    if isinstance(pattern := schema.get('pattern'), str):
        yield pattern
    if isinstance(pattern_properties := schema.get('patternProperties'), dict):
        yield from cast(dict[str, Any], pattern_properties)
    for subschema in _subschemas(schema):
        yield from _schema_patterns(subschema)


def _uses_keyword(schema: Any, keyword: str) -> bool:
    """Whether `keyword` appears as a keyword anywhere in a JSON schema."""
    if not isinstance(schema, dict):
        return False
    schema = cast(dict[str, Any], schema)
    return keyword in schema or any(_uses_keyword(subschema, keyword) for subschema in _subschemas(schema))


def _pattern_keyword(matchers: _PatternMatchers) -> Callable[..., Iterator[Any]]:
    from jsonschema.exceptions import ValidationError

    def pattern(validator: Any, pattern: str, instance: Any, schema: Any) -> Iterator[Any]:
        if validator.is_type(instance, 'string') and not matchers[pattern](instance):
            yield ValidationError(f'{instance!r} does not match {pattern!r}')

    return pattern


def _pattern_properties_keyword(matchers: _PatternMatchers) -> Callable[..., Iterator[Any]]:
    def pattern_properties(
        validator: Any, pattern_properties: dict[str, Any], instance: Any, schema: Any
    ) -> Iterator[Any]:
        if not validator.is_type(instance, 'object'):
            return
        for pattern, subschema in pattern_properties.items():
            for key, value in instance.items():
                if matchers[pattern](key):
                    yield from validator.descend(value, subschema, path=key, schema_path=pattern)

    return pattern_properties


def _additional_properties_keyword(matchers: _PatternMatchers) -> Callable[..., Iterator[Any]]:
    """`additionalProperties`, matching `patternProperties` keys with `matchers` instead of `re.search`."""
    from jsonschema.exceptions import ValidationError

    def additional_properties(validator: Any, additional: Any, instance: Any, schema: Any) -> Iterator[Any]:
        if not validator.is_type(instance, 'object'):
            return
        extras = _additional_keys(instance, schema, matchers)
        if validator.is_type(additional, 'object'):
            for extra in extras:
                yield from validator.descend(instance[extra], additional, path=extra)
        elif additional is False and extras:
            names = ', '.join(repr(extra) for extra in sorted(extras, key=str))
            verb = 'was' if len(extras) == 1 else 'were'
            yield ValidationError(f'Additional properties are not allowed ({names} {verb} unexpected)')

    return additional_properties


def _additional_keys(instance: dict[str, Any], schema: dict[str, Any], matchers: _PatternMatchers) -> list[str]:
    """The keys of `instance` that neither `properties` nor a `patternProperties` pattern in `schema` covers."""
    properties = schema.get('properties', {})
    patterns = [matchers[pattern] for pattern in schema.get('patternProperties', {})]
    return [key for key in instance if key not in properties and not any(match(key) for match in patterns)]


def _non_finite_numbers(value: Any, path: str = '') -> list[tuple[str, float]]:
    """The NaN and infinite floats anywhere in `value`, with their dotted paths."""
    if isinstance(value, float):
        return [] if math.isfinite(value) else [(path, value)]
    if isinstance(value, dict):
        items: Iterable[tuple[Any, Any]] = cast(dict[Any, Any], value).items()
    elif isinstance(value, list):
        items = enumerate(cast(list[Any], value))
    else:
        return []
    return [found for key, item in items for found in _non_finite_numbers(item, f'{path}.{key}' if path else str(key))]
