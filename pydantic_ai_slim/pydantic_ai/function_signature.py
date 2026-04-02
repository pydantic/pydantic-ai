"""Generate function signatures from functions and JSON schemas.

This module provides utilities to represent tool definitions as human-readable
function signatures, which LLMs can understand more easily than raw
JSON schemas. Used by code mode to present tools as callable functions.
"""

from __future__ import annotations

import inspect
import json
import re
import types
from collections.abc import Callable
from dataclasses import dataclass, field
from inspect import Parameter, Signature as InspectSignature, signature
from typing import Any, Literal, TypeAlias, Union, cast, get_args, get_origin

from pydantic import BaseModel, TypeAdapter
from typing_extensions import get_type_hints

from ._run_context import RunContext
from ._utils import is_model_like


def _get_schema_from_type(t: Any, *, mode: Literal['validation', 'serialization'] = 'validation') -> dict[str, Any]:
    """Extract JSON schema from a `BaseModel`, dataclass, or `TypedDict`."""
    if isinstance(t, type) and issubclass(t, BaseModel):
        return t.model_json_schema(mode=mode)
    return TypeAdapter(t).json_schema(mode=mode)  # pyright: ignore[reportUnknownArgumentType]


# =============================================================================
# Type expression tree
# =============================================================================


@dataclass
class SimpleTypeExpr:
    """A simple named type like `str`, `int`, `Any`, `None`."""

    name: str

    def __str__(self) -> str:
        return self.name


@dataclass
class LiteralTypeExpr:
    """A Literal type expression like `Literal['a', 'b']` or `Literal[42]`."""

    values: list[Any]

    def __str__(self) -> str:
        return f'Literal[{", ".join(repr(v) for v in self.values)}]'


@dataclass
class GenericTypeExpr:
    """A generic type expression like `list[User]`, `dict[str, User]`, `tuple[int, str]`."""

    base: str
    args: list[TypeExpr]

    def __str__(self) -> str:
        return f'{self.base}[{", ".join(str(a) for a in self.args)}]'


@dataclass
class UnionTypeExpr:
    """A union type expression like `User | None`, `str | int`."""

    members: list[TypeExpr]

    def __str__(self) -> str:
        return ' | '.join(str(m) for m in self.members)


TypeExpr: TypeAlias = 'TypeSignature | SimpleTypeExpr | LiteralTypeExpr | GenericTypeExpr | UnionTypeExpr'
"""A type expression node in the signature's type tree."""


# =============================================================================
# Signature dataclasses
# =============================================================================


def _render_description(text: str, indent: str = '') -> list[str]:
    """Render a description as a list of indented docstring lines."""
    text = text.strip()
    if '\n' in text:
        lines = [f'{indent}"""']
        for line in text.split('\n'):
            lines.append(f'{indent}{line}' if line.strip() else '')
        lines.append(f'{indent}"""')
        return lines
    return [f'{indent}"""{text}"""']


@dataclass
class TypeFieldSignature:
    """A single field in a TypedDict-style type definition."""

    name: str
    type: TypeExpr
    required: bool
    description: str | None

    def __str__(self) -> str:
        """Render this field as a line in a TypedDict class body."""
        type_str = str(self.type)
        if not self.required:
            type_str = f'NotRequired[{type_str}]'
        lines: list[str] = [f'    {self.name}: {type_str}']
        if self.description:
            lines.extend(_render_description(self.description, indent='    '))
        return '\n'.join(lines)


@dataclass
class TypeSignature:
    """A TypedDict-style class definition with named fields."""

    name: str

    description: str | None = None

    fields: dict[str, TypeFieldSignature] = field(default_factory=dict[str, TypeFieldSignature])

    def __str__(self) -> str:
        """Return the type name (for use in type expressions like `def foo(x: User)`)."""
        return self.name

    def render_definition(self) -> str:
        """Render the full TypedDict class definition."""
        lines = [f'class {self.name}(TypedDict):']
        if self.description:
            lines.extend(_render_description(self.description, indent='    '))
        if not self.fields:
            if not self.description:
                lines.append('    pass')
        else:
            for f in self.fields.values():
                lines.append(str(f))
        return '\n'.join(lines)

    def structurally_equal(self, other: TypeSignature) -> bool:
        """Compare two TypeSignatures structurally, ignoring descriptions."""
        if set(self.fields.keys()) != set(other.fields.keys()):
            return False
        for name, f in self.fields.items():
            other_f = other.fields[name]
            if f.required != other_f.required:
                return False
            if str(f.type) != str(other_f.type):
                return False
        return True


@dataclass
class FunctionParam:
    """A single parameter in a function signature."""

    name: str
    type: TypeExpr
    default: str | None

    def __str__(self) -> str:
        """Render this parameter as a function parameter string."""
        type_str = str(self.type)
        if self.default is not None:
            return f'{self.name}: {type_str} = {self.default}'
        return f'{self.name}: {type_str}'


@dataclass
class FunctionSignature:
    """Function signature with referenced type definitions.

    This class holds all the data needed to render a function signature.
    Use `str(sig)` for the default rendering, or `render()` for customization.
    """

    name: str
    """The function name."""

    params: dict[str, FunctionParam]
    """Function parameters."""

    return_type: TypeExpr
    """The return type expression."""

    description: str | None = None
    """Optional description for the function."""

    referenced_types: list[TypeSignature] = field(default_factory=list[TypeSignature])
    """TypedDict class definitions needed by the signature."""

    is_async: bool = False
    """Whether the underlying function is async."""

    def __str__(self) -> str:
        """Render with `...` body."""
        return self.render('...')

    def render(self, body: str, *, is_async: bool | None = None) -> str:
        """Render the signature with a specific body.

        Args:
            body: The function body (e.g. `'...'` or `'return await tool()'`).
            is_async: Override async rendering. If `None`, uses `self.is_async`.
        """
        async_flag = is_async if is_async is not None else self.is_async
        prefix = 'async def' if async_flag else 'def'
        params_str = ', '.join(str(p) for p in self.params.values())

        return_str = str(self.return_type)
        if params_str:
            # Force keyword-only params so LLMs always use named arguments
            parts = [f'{prefix} {self.name}(*, {params_str}) -> {return_str}:']
        else:
            parts = [f'{prefix} {self.name}() -> {return_str}:']

        if self.description:
            parts.extend(_render_description(self.description, indent='    '))

        parts.append(f'    {body}')

        return '\n'.join(parts)

    @classmethod
    def from_function(
        cls,
        func: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> FunctionSignature:
        """Build a FunctionSignature from a Python function using inspect.

        Uses `inspect.signature()` and `get_type_hints()` for rich type information
        (e.g. TypedDict fields, union types, BaseModel references).
        """
        name = name or func.__name__
        sig = signature(func)

        try:
            type_hints = get_type_hints(func)
        except (NameError, TypeError, AttributeError):
            type_hints = {}

        referenced_types: dict[str, TypeSignature] = {}
        params = _build_function_params(sig, type_hints, referenced_types, name)

        return_annotation = type_hints.get('return')
        if return_annotation is not None:
            _collect_referenced_types(return_annotation, referenced_types, name, 'Return', mode='serialization')

        return_type: TypeExpr = (
            _annotation_to_type_expr(return_annotation, referenced_types) if return_annotation else _ANY
        )

        return cls(
            name=name,
            params=params,
            return_type=return_type,
            description=description,
            referenced_types=list(referenced_types.values()),
            is_async=inspect.iscoroutinefunction(func),
        )

    @classmethod
    def from_schema(
        cls,
        *,
        name: str,
        parameters_schema: dict[str, Any],
        description: str | None = None,
        return_schema: dict[str, Any] | None = None,
    ) -> FunctionSignature:
        """Build a FunctionSignature from a JSON schema.

        Parameter and return schemas are processed independently — each resolves
        `$ref`s against its own `$defs`. Name collisions between parameter and return
        types (e.g. both define a `User` `$def` with different structures) are handled
        by `dedup_referenced_types` at a later stage.
        """
        # Process parameter schema with its own $defs
        param_defs = parameters_schema.get('$defs', {})
        param_referenced: dict[str, TypeSignature] = {}
        _process_schema_defs(param_defs, param_referenced, name)
        params = _build_params_from_schema(parameters_schema, param_defs, param_referenced, name)

        # Process return schema independently (its own $defs)
        resolved_return_type: TypeExpr = _ANY
        return_referenced: dict[str, TypeSignature] = {}
        if return_schema is not None:
            return_defs = return_schema.get('$defs', {})
            _process_schema_defs(return_defs, return_referenced, name)
            resolved_return_type = _schema_to_type_expr(return_schema, return_defs, return_referenced, name, 'Return')

        # If return schema couldn't be resolved to a concrete type, append it as description
        final_description = description
        if (
            return_schema is not None
            and isinstance(resolved_return_type, SimpleTypeExpr)
            and resolved_return_type.name == 'Any'
        ):
            return_schema_blob = json.dumps(return_schema, indent=2)
            return_schema_note = f'\n\nReturn schema:\n{return_schema_blob}'
            final_description = (description or '') + return_schema_note
            final_description = final_description.strip()

        # Merge referenced types — dedup_referenced_types handles collisions later
        all_referenced = list(param_referenced.values()) + list(return_referenced.values())

        return cls(
            name=name,
            params=params,
            return_type=resolved_return_type,
            description=final_description if final_description else None,
            referenced_types=all_referenced,
        )

    @staticmethod
    def dedup_referenced_types(signatures: list[FunctionSignature]) -> None:
        """Resolve TypedDict name conflicts across multiple tool signatures in place.

        Each signature keeps all its referenced types (so it remains self-contained),
        but identical types (same name and structure) are unified to the same object
        instance, and conflicting types (same name, different structure) are renamed
        by prefixing the tool name to disambiguate.

        Use `collect_unique_referenced_types()` when rendering to emit each definition once.
        """
        seen: dict[str, TypeSignature] = {}

        for sig in signatures:
            deduped: list[TypeSignature] = []
            for type_sig in sig.referenced_types:
                name = type_sig.name
                if name not in seen:
                    seen[name] = type_sig
                    deduped.append(type_sig)
                elif seen[name].structurally_equal(type_sig):
                    canonical = seen[name]
                    _replace_type_refs(sig, type_sig, canonical)
                    deduped.append(canonical)
                else:
                    new_name = f'{sig.name}_{name}'
                    type_sig.name = new_name
                    seen[new_name] = type_sig
                    deduped.append(type_sig)
            sig.referenced_types = deduped

    @staticmethod
    def collect_unique_referenced_types(signatures: list[FunctionSignature]) -> list[TypeSignature]:
        """Collect unique TypeSignature objects from signatures, deduplicating by identity."""
        seen_ids: set[int] = set()
        result: list[TypeSignature] = []
        for sig in signatures:
            for type_sig in sig.referenced_types:
                if id(type_sig) not in seen_ids:
                    seen_ids.add(id(type_sig))
                    result.append(type_sig)
        return result


# Shared singleton for 'Any' type expression
_ANY = SimpleTypeExpr('Any')
_NONE = SimpleTypeExpr('None')


# =============================================================================
# Type annotation to TypeExpr conversion (Python annotations)
# =============================================================================


def _get_type_name(t: Any) -> SimpleTypeExpr:
    """Get a SimpleTypeExpr for a type."""
    if isinstance(t, type):
        return SimpleTypeExpr(t.__name__)
    s = repr(t)
    return SimpleTypeExpr(s.replace('typing.', '').replace('typing_extensions.', ''))


def _annotation_to_type_expr(
    annotation: Any,
    referenced_types: dict[str, TypeSignature],
) -> TypeExpr:
    """Convert a Python type annotation to a TypeExpr."""
    if annotation is None or annotation is type(None):
        return _NONE

    # Named types (BaseModel/TypedDict/dataclass) → look up in referenced_types
    if is_model_like(annotation):
        type_name = annotation.__name__
        if type_name in referenced_types:
            return referenced_types[type_name]
        return SimpleTypeExpr(type_name)

    # Handle Python 3.10+ union syntax (X | Y creates types.UnionType)
    if isinstance(annotation, types.UnionType):
        members = [_annotation_to_type_expr(arg, referenced_types) for arg in get_args(annotation)]
        return UnionTypeExpr(members=members)

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is not None:
        if args:
            if origin is Union:
                members = [_annotation_to_type_expr(arg, referenced_types) for arg in args]
                return UnionTypeExpr(members=members)
            base = _get_type_name(origin)
            type_args = [_annotation_to_type_expr(arg, referenced_types) for arg in args]
            return GenericTypeExpr(base=base.name, args=type_args)
        return _get_type_name(origin)

    return _get_type_name(annotation)


# =============================================================================
# Function signature builder (from Python functions)
# =============================================================================


def _collect_referenced_types(
    annotation: Any,
    referenced_types: dict[str, TypeSignature],
    tool_name: str,
    path: str = '',
    *,
    mode: Literal['validation', 'serialization'] = 'validation',
) -> None:
    """Walk a type annotation tree, adding TypeSignature entries for any structured types found.

    Structured types (BaseModel, TypedDict, dataclass) are converted to TypeSignature
    objects and added to `referenced_types`. This is called before `_annotation_to_type_expr`
    so that type references can be resolved by name.
    """
    if annotation is None or annotation is type(None):
        return

    if is_model_like(annotation):
        type_name = annotation.__name__
        if type_name not in referenced_types:
            schema = _get_schema_from_type(annotation, mode=mode)
            schema_defs = schema.get('$defs', {})

            # Process any $defs first (nested models referenced by the main schema)
            for def_name, def_schema in schema_defs.items():
                if (
                    def_name not in referenced_types
                    and def_schema.get('type') == 'object'
                    and 'properties' in def_schema
                ):
                    _build_and_register_type(def_name, def_schema, schema_defs, referenced_types, tool_name, path)

            # Then process the main schema
            if schema.get('type') == 'object' and 'properties' in schema:
                _build_and_register_type(type_name, schema, schema_defs, referenced_types, tool_name, path)
        return

    # Handle Python 3.10+ union syntax (X | Y creates types.UnionType)
    if isinstance(annotation, types.UnionType):
        for arg in get_args(annotation):
            _collect_referenced_types(arg, referenced_types, tool_name, path, mode=mode)
        return

    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is not None and args:
        for arg in args:
            _collect_referenced_types(arg, referenced_types, tool_name, path, mode=mode)


def _build_function_params(
    sig: InspectSignature,
    type_hints: dict[str, Any],
    referenced_types: dict[str, TypeSignature],
    tool_name: str,
) -> dict[str, FunctionParam]:
    """Build FunctionParam objects from a function's signature and type hints."""
    params: dict[str, FunctionParam] = {}
    for i, (param_name, param) in enumerate(sig.parameters.items()):
        annotation = type_hints.get(param_name)

        if i == 0 and annotation is not None and (annotation is RunContext or get_origin(annotation) is RunContext):
            continue

        if annotation is not None:
            _collect_referenced_types(annotation, referenced_types, tool_name, param_name)
            type_expr = _annotation_to_type_expr(annotation, referenced_types)
        else:
            type_expr = _ANY

        if param.default is Parameter.empty:
            params[param_name] = FunctionParam(name=param_name, type=type_expr, default=None)
        else:
            default_str = repr(param.default)
            params[param_name] = FunctionParam(name=param_name, type=type_expr, default=default_str)
    return params


# Keep module-level aliases for backward compatibility and use by ToolDefinition
def function_to_signature(
    func: Callable[..., Any],
    *,
    name: str | None = None,
    description: str | None = None,
) -> FunctionSignature:
    """Build a FunctionSignature from a Python function. Alias for `FunctionSignature.from_function`."""
    return FunctionSignature.from_function(func, name=name, description=description)


def schema_to_signature(
    *,
    name: str,
    parameters_schema: dict[str, Any],
    description: str | None = None,
    return_schema: dict[str, Any] | None = None,
) -> FunctionSignature:
    """Build a FunctionSignature from a JSON schema. Alias for `FunctionSignature.from_schema`."""
    return FunctionSignature.from_schema(
        name=name,
        parameters_schema=parameters_schema,
        description=description,
        return_schema=return_schema,
    )


# =============================================================================
# JSON schema to signature conversion
# =============================================================================


_JSON_TYPE_TO_PYTHON: dict[str, str] = {
    'string': 'str',
    'integer': 'int',
    'number': 'float',
    'boolean': 'bool',
    'null': 'None',
    'array': 'list',
    'object': 'dict',
}


def _json_type_to_python(json_type: str) -> SimpleTypeExpr:
    """Convert a JSON type string to a SimpleTypeExpr."""
    return SimpleTypeExpr(_JSON_TYPE_TO_PYTHON.get(json_type, 'Any'))


_NON_ALNUM_RE = re.compile(r'[^a-zA-Z0-9]')


def _to_pascal_case(s: str) -> str:
    """Convert a string to PascalCase."""
    s = _NON_ALNUM_RE.sub('_', s)
    parts = s.split('_')
    result = ''.join(part.capitalize() for part in parts if part)
    if result and result[0].isdigit():
        result = '_' + result
    return result


def _path_to_typename(tool_name: str, path: str) -> str:
    """Convert a traversal path to a unique TypedDict name.

    Examples:
        _path_to_typename('get_user', '') -> 'GetUser'
        _path_to_typename('get_user', 'address') -> 'GetUserAddress'
        _path_to_typename('get_user', 'home.address') -> 'GetUserHomeAddress'
    """
    parts = [tool_name] + [p for p in path.split('.') if p]
    return ''.join(_to_pascal_case(p) for p in parts)


def _process_schema_defs(
    defs: dict[str, dict[str, Any]],
    referenced_types: dict[str, TypeSignature],
    tool_name: str,
) -> None:
    """Process $defs from a JSON schema, adding TypeSignatures for object-type definitions."""
    for def_name, def_schema in defs.items():
        if def_schema.get('type') == 'object' and 'properties' in def_schema:
            if def_name not in referenced_types:
                _build_and_register_type(def_name, def_schema, defs, referenced_types, tool_name, def_name)


def _build_params_from_schema(
    schema: dict[str, Any],
    defs: dict[str, dict[str, Any]],
    referenced_types: dict[str, TypeSignature],
    tool_name: str,
) -> dict[str, FunctionParam]:
    """Convert a JSON schema to a dict of FunctionParam objects."""
    properties = schema.get('properties', {})
    required = set(schema.get('required', []))

    required_params: dict[str, FunctionParam] = {}
    optional_params: dict[str, FunctionParam] = {}

    for prop_name, prop_schema in properties.items():
        type_expr = _schema_to_type_expr(prop_schema, defs, referenced_types, tool_name, prop_name)

        if 'default' in prop_schema:
            default_str = repr(prop_schema['default'])
            optional_params[prop_name] = FunctionParam(name=prop_name, type=type_expr, default=default_str)
        elif prop_name in required:
            required_params[prop_name] = FunctionParam(name=prop_name, type=type_expr, default=None)
        else:
            # Optional without default — add | None
            if _schema_allows_null(prop_schema):
                optional_params[prop_name] = FunctionParam(name=prop_name, type=type_expr, default='None')
            else:
                nullable_expr = UnionTypeExpr(members=[type_expr, _NONE])
                optional_params[prop_name] = FunctionParam(name=prop_name, type=nullable_expr, default='None')

    return {**required_params, **optional_params}


def _schema_allows_null(schema: dict[str, Any]) -> bool:
    """Check if a schema already allows null values."""
    schema_type = schema.get('type')
    if isinstance(schema_type, list) and 'null' in schema_type:
        return True
    if 'anyOf' in schema:
        return any(s.get('type') == 'null' for s in schema['anyOf'])
    if 'oneOf' in schema:
        return any(s.get('type') == 'null' for s in schema['oneOf'])
    return False


def _schema_to_type_expr(
    schema: dict[str, Any],
    defs: dict[str, dict[str, Any]],
    referenced_types: dict[str, TypeSignature],
    tool_name: str,
    path: str,
) -> TypeExpr:
    """Convert a JSON schema to a TypeExpr."""
    # Handle $ref
    if '$ref' in schema:
        ref = schema['$ref']
        ref_name = ref.split('/')[-1]
        # Ensure referenced def generates TypeSignature if needed
        if ref_name in defs and ref_name not in referenced_types:
            ref_schema = defs[ref_name]
            if ref_schema.get('type') == 'object' and 'properties' in ref_schema:
                _build_and_register_type(ref_name, ref_schema, defs, referenced_types, tool_name, path)
        # Return the TypeSignature object if available, otherwise the name
        if ref_name in referenced_types:
            return referenced_types[ref_name]
        return SimpleTypeExpr(ref_name)

    # Handle anyOf/oneOf (union types)
    if 'anyOf' in schema:
        return _handle_union_schema(schema['anyOf'], defs, referenced_types, tool_name, path)
    if 'oneOf' in schema:
        return _handle_union_schema(schema['oneOf'], defs, referenced_types, tool_name, path)

    # Handle allOf
    if 'allOf' in schema:
        if len(schema['allOf']) == 1:
            return _schema_to_type_expr(schema['allOf'][0], defs, referenced_types, tool_name, path)
        return _ANY

    # Handle const
    if 'const' in schema:
        return LiteralTypeExpr([schema['const']])

    # Handle enum
    if 'enum' in schema:
        return LiteralTypeExpr(schema['enum'])

    # Handle by type
    schema_type = schema.get('type')
    return _type_to_expr(schema_type, schema, defs, referenced_types, tool_name, path)


def _type_to_expr(
    schema_type: str | list[str] | None,
    schema: dict[str, Any],
    defs: dict[str, dict[str, Any]],
    referenced_types: dict[str, TypeSignature],
    tool_name: str,
    path: str,
) -> TypeExpr:
    """Convert a schema type to a TypeExpr."""
    # Simple types — use shared mapping, skip compound types handled below
    if isinstance(schema_type, str) and schema_type in _JSON_TYPE_TO_PYTHON and schema_type not in ('array', 'object'):
        return SimpleTypeExpr(_JSON_TYPE_TO_PYTHON[schema_type])

    # Array type
    if schema_type == 'array':
        items = schema.get('items', {})
        if items:
            # Handle tuple schemas (items as list)
            if isinstance(items, list):
                items_list = cast(list[dict[str, Any]], items)
                item_exprs = [
                    _schema_to_type_expr(item, defs, referenced_types, tool_name, f'{path}.{i}')
                    for i, item in enumerate(items_list)
                ]
                return GenericTypeExpr(base='tuple', args=item_exprs)
            item_expr = _schema_to_type_expr(
                cast(dict[str, Any], items), defs, referenced_types, tool_name, f'{path}Item'
            )
            return GenericTypeExpr(base='list', args=[item_expr])
        return GenericTypeExpr(base='list', args=[_ANY])

    # Object type
    if schema_type == 'object':
        if 'properties' in schema:
            # Generate TypeSignature with path-based unique name
            td_name = _path_to_typename(tool_name, path)
            if td_name not in referenced_types:
                referenced_types[td_name] = _build_type_signature(
                    td_name, schema, defs, referenced_types, tool_name, path
                )
            return referenced_types[td_name]
        if 'additionalProperties' in schema:
            additional = schema['additionalProperties']
            if additional is True:
                return GenericTypeExpr(base='dict', args=[SimpleTypeExpr('str'), _ANY])
            if isinstance(additional, dict):
                additional_schema = cast(dict[str, Any], additional)
                value_expr = _schema_to_type_expr(additional_schema, defs, referenced_types, tool_name, f'{path}Value')
                return GenericTypeExpr(base='dict', args=[SimpleTypeExpr('str'), value_expr])
        return GenericTypeExpr(base='dict', args=[SimpleTypeExpr('str'), _ANY])

    # Type list (e.g., ['string', 'null'])
    if isinstance(schema_type, list):
        return _type_list_to_expr(schema_type, schema, defs, referenced_types, tool_name, path)

    return _ANY


def _type_list_to_expr(
    schema_type: list[str],
    schema: dict[str, Any],
    defs: dict[str, dict[str, Any]],
    referenced_types: dict[str, TypeSignature],
    tool_name: str,
    path: str,
) -> TypeExpr:
    """Handle type lists like ['string', 'null']."""
    # Check if this is object with properties + null
    if 'object' in schema_type and 'properties' in schema:
        base_expr = _type_to_expr('object', schema, defs, referenced_types, tool_name, path)
        if 'null' in schema_type:
            return UnionTypeExpr(members=[base_expr, _NONE])
        return base_expr

    type_exprs: list[TypeExpr] = [_json_type_to_python(t) for t in schema_type]
    type_exprs = [t for t in type_exprs if str(t)]
    if len(type_exprs) == 2 and any(str(t) == 'None' for t in type_exprs):
        non_none = [t for t in type_exprs if str(t) != 'None'][0]
        return UnionTypeExpr(members=[non_none, _NONE])
    if type_exprs:
        return UnionTypeExpr(members=type_exprs) if len(type_exprs) > 1 else type_exprs[0]
    return _ANY


def _handle_union_schema(
    schemas: list[dict[str, Any]],
    defs: dict[str, dict[str, Any]],
    referenced_types: dict[str, TypeSignature],
    tool_name: str,
    path: str,
) -> TypeExpr:
    """Handle anyOf/oneOf schemas, returning a TypeExpr."""
    type_exprs: list[TypeExpr] = []
    has_null = False

    for s in schemas:
        if s.get('type') == 'null':
            has_null = True
        else:
            type_exprs.append(_schema_to_type_expr(s, defs, referenced_types, tool_name, path))

    # Deduplicate while preserving order (compare rendered strings)
    seen: set[str] = set()
    unique_exprs: list[TypeExpr] = []
    for expr in type_exprs:
        rendered = str(expr)
        if rendered not in seen:
            seen.add(rendered)
            unique_exprs.append(expr)

    if has_null:
        unique_exprs.append(_NONE)

    if len(unique_exprs) == 1:
        return unique_exprs[0]
    return UnionTypeExpr(members=unique_exprs)


def _build_and_register_type(
    name: str,
    schema: dict[str, Any],
    defs: dict[str, dict[str, Any]],
    referenced_types: dict[str, TypeSignature],
    tool_name: str,
    path: str,
) -> TypeSignature:
    """Build a TypeSignature, registering a placeholder first to prevent infinite recursion.

    Self-referential schemas (e.g. recursive models) would otherwise cause infinite recursion
    when `_build_type_signature` processes properties that `$ref` back to the same type.

    Returns the completed TypeSignature.
    """
    placeholder = TypeSignature(name=name)
    referenced_types[name] = placeholder
    built = _build_type_signature(name, schema, defs, referenced_types, tool_name, path)
    placeholder.fields = built.fields
    placeholder.description = built.description
    return placeholder


def _build_type_signature(
    name: str,
    schema: dict[str, Any],
    defs: dict[str, dict[str, Any]],
    referenced_types: dict[str, TypeSignature],
    tool_name: str,
    path: str,
) -> TypeSignature:
    """Build a TypeSignature from an object schema."""
    properties = schema.get('properties', {})
    required = set(schema.get('required', []))

    fields: dict[str, TypeFieldSignature] = {}

    for prop_name, prop_schema in properties.items():
        prop_path = f'{path}.{prop_name}' if path else prop_name
        type_expr = _schema_to_type_expr(prop_schema, defs, referenced_types, tool_name, prop_path)
        is_required = prop_name in required
        desc = prop_schema.get('description', '') or None

        fields[prop_name] = TypeFieldSignature(
            name=prop_name,
            type=type_expr,
            required=is_required,
            description=desc,
        )

    description = schema.get('description') or None
    return TypeSignature(name=name, description=description, fields=fields)


# =============================================================================
# Deduplication helpers
# =============================================================================


def _replace_type_refs(sig: FunctionSignature, old_ref: TypeSignature, canonical: TypeSignature) -> None:
    """Replace all references to old_ref with canonical in a signature's TypeExpr trees."""

    def _replace_in_expr(expr: TypeExpr) -> TypeExpr:
        if expr is old_ref:
            return canonical
        if isinstance(expr, GenericTypeExpr):
            new_args = [_replace_in_expr(a) for a in expr.args]
            if any(new is not orig for new, orig in zip(new_args, expr.args)):
                expr.args = new_args
        elif isinstance(expr, UnionTypeExpr):
            new_members = [_replace_in_expr(m) for m in expr.members]
            if any(new is not orig for new, orig in zip(new_members, expr.members)):
                expr.members = new_members
        return expr

    # Replace in params
    for param in sig.params.values():
        param.type = _replace_in_expr(param.type)

    # Replace in return type
    sig.return_type = _replace_in_expr(sig.return_type)

    # Replace in field types of referenced types
    for type_sig in sig.referenced_types:
        for f in type_sig.fields.values():
            f.type = _replace_in_expr(f.type)


# Keep top-level aliases for backward compatibility
dedup_referenced_types = FunctionSignature.dedup_referenced_types
collect_unique_referenced_types = FunctionSignature.collect_unique_referenced_types
