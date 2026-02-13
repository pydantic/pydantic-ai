"""Generate Python function signatures from functions and JSON schemas.

This module provides utilities to represent tool definitions as human-readable
Python function signatures, which LLMs can understand more easily than raw
JSON schemas. Used by code mode to present tools as callable Python functions.
"""

from __future__ import annotations

import dataclasses
import json
import re
import types
from collections.abc import Callable
from dataclasses import dataclass, field
from inspect import Parameter, Signature as InspectSignature, signature
from typing import Any, Literal, TypeAlias, Union, cast, get_origin

from pydantic import BaseModel, TypeAdapter
from typing_extensions import get_type_hints, is_typeddict

from ._run_context import RunContext


def _is_named_type(t: Any) -> bool:
    """Check if a type is a `BaseModel`, dataclass, or `TypedDict` that needs a definition."""
    if not isinstance(t, type):
        return False
    return issubclass(t, BaseModel) or dataclasses.is_dataclass(t) or is_typeddict(t)  # pyright: ignore[reportUnknownArgumentType]


def _get_schema_from_type(t: Any, *, mode: Literal['validation', 'serialization'] = 'validation') -> dict[str, Any]:
    """Extract JSON schema from a `BaseModel`, dataclass, or `TypedDict`."""
    if isinstance(t, type) and issubclass(t, BaseModel):
        return t.model_json_schema(mode=mode)
    return TypeAdapter(t).json_schema(mode=mode)  # pyright: ignore[reportUnknownArgumentType]


# =============================================================================
# Type expression tree
# =============================================================================


@dataclass
class GenericTypeExpr:
    """A generic type expression like `list[User]`, `dict[str, User]`, `tuple[int, str]`."""

    base: str
    args: list[TypeExpr]

    def render(self) -> str:
        return f'{self.base}[{", ".join(render_type_expr(a) for a in self.args)}]'


@dataclass
class UnionTypeExpr:
    """A union type expression like `User | None`, `str | int`."""

    members: list[TypeExpr]

    def render(self) -> str:
        return ' | '.join(render_type_expr(m) for m in self.members)


# =============================================================================
# Signature dataclasses
# =============================================================================


def _render_docstring(text: str, indent: str = '') -> list[str]:
    """Render a docstring as a list of indented lines."""
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
    name: str
    type: TypeExpr
    required: bool
    description: str | None

    def render(self) -> str:
        """Render this field as a line in a TypedDict class body."""
        type_str = render_type_expr(self.type)
        if not self.required:
            type_str = f'NotRequired[{type_str}]'
        lines: list[str] = [f'    {self.name}: {type_str}']
        if self.description:
            lines.extend(_render_docstring(self.description, indent='    '))
        return '\n'.join(lines)


@dataclass
class TypeSignature:
    name: str

    docstring: str | None = None

    fields: dict[str, TypeFieldSignature] = field(default_factory=dict[str, TypeFieldSignature])

    def render(self) -> str:
        """Render this type as a TypedDict class definition."""
        lines = [f'class {self.name}(TypedDict):']
        if self.docstring:
            lines.extend(_render_docstring(self.docstring, indent='    '))
        if not self.fields:
            if not self.docstring:
                lines.append('    pass')
        else:
            for f in self.fields.values():
                lines.append(f.render())
        return '\n'.join(lines)

    def structurally_equal(self, other: TypeSignature) -> bool:
        """Compare two TypeSignatures structurally, ignoring descriptions and docstrings."""
        if set(self.fields.keys()) != set(other.fields.keys()):
            return False
        for name, f in self.fields.items():
            other_f = other.fields[name]
            if f.required != other_f.required:
                return False
            if render_type_expr(f.type) != render_type_expr(other_f.type):
                return False
        return True


TypeExpr: TypeAlias = 'TypeSignature | str | GenericTypeExpr | UnionTypeExpr'
"""A type expression that can reference `TypeSignature` objects, enabling automatic name propagation."""


def render_type_expr(expr: TypeExpr) -> str:
    """Render a type expression to a string."""
    if isinstance(expr, TypeSignature):
        return expr.name
    if isinstance(expr, str):
        return expr
    return expr.render()


@dataclass
class FunctionParam:
    name: str
    type: TypeExpr
    default: str | None

    def render(self) -> str:
        """Render this parameter as a function parameter string."""
        type_str = render_type_expr(self.type)
        if self.default is not None:
            return f'{self.name}: {type_str} = {self.default}'
        return f'{self.name}: {type_str}'


@dataclass
class FunctionSignature:
    """Python function signature with TypedDict definitions.

    This class holds all the data needed to render a function signature as Python code.
    Use `str(sig)` for the default rendering, or call specific methods for variants.
    """

    name: str
    """The function name."""

    params: dict[str, FunctionParam]
    """Function parameters."""

    return_type: TypeExpr
    """The return type expression."""

    docstring: str | None = None
    """Optional docstring for the function."""

    referenced_types: list[TypeSignature] = field(default_factory=list[TypeSignature])
    """TypedDict class definitions needed by the signature."""

    is_async: bool = True
    """Whether to generate 'async def' (True) or 'def' (False)."""

    def __str__(self) -> str:
        """Render with `...` body."""
        return self.render('...')

    def render(self, body: str) -> str:
        """Render the signature with a specific body."""
        prefix = 'async def' if self.is_async else 'def'
        params_str = ', '.join(p.render() for p in self.params.values())

        return_str = render_type_expr(self.return_type)
        if params_str:
            parts = [f'{prefix} {self.name}(*, {params_str}) -> {return_str}:']
        else:
            parts = [f'{prefix} {self.name}() -> {return_str}:']

        if self.docstring:
            parts.extend(_render_docstring(self.docstring, indent='    '))

        parts.append(f'    {body}')

        return '\n'.join(parts)


# =============================================================================
# Type annotation to TypeExpr conversion (Python annotations)
# =============================================================================


def _get_type_name(t: Any) -> str:
    """Get the name of a type."""
    if t is type(None):
        return 'None'
    if hasattr(t, '__name__'):
        return t.__name__
    s = repr(t)
    return s.replace('typing.', '').replace('typing_extensions.', '')


def _annotation_to_type_expr(
    annotation: Any,
    referenced_types: dict[str, TypeSignature],
) -> TypeExpr:
    """Convert a Python type annotation to a TypeExpr."""
    if annotation is None or annotation is type(None):
        return 'None'

    # Named types (BaseModel/TypedDict/dataclass) → look up in referenced_types
    if _is_named_type(annotation):
        type_name = annotation.__name__
        if type_name in referenced_types:
            return referenced_types[type_name]
        return type_name

    # Handle Python 3.10+ union syntax (X | Y creates types.UnionType)
    if isinstance(annotation, types.UnionType):
        args = getattr(annotation, '__args__', ())
        members = [_annotation_to_type_expr(arg, referenced_types) for arg in args]
        return UnionTypeExpr(members=members)

    origin = getattr(annotation, '__origin__', None)
    args = getattr(annotation, '__args__', None)

    if origin is not None:
        if args:
            if origin is Union:
                members = [_annotation_to_type_expr(arg, referenced_types) for arg in args]
                return UnionTypeExpr(members=members)
            base = _get_type_name(origin)
            type_args = [_annotation_to_type_expr(arg, referenced_types) for arg in args]
            return GenericTypeExpr(base=base, args=type_args)
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
    """Recursively collect TypeSignature definitions from type annotations."""
    if annotation is None or annotation is type(None):
        return

    if _is_named_type(annotation):
        type_name = annotation.__name__
        if type_name not in referenced_types:
            schema = _get_schema_from_type(annotation, mode=mode)
            schema_defs = schema.get('$defs', {})

            # Process any $defs first
            for def_name, def_schema in schema_defs.items():
                if (
                    def_name not in referenced_types
                    and def_schema.get('type') == 'object'
                    and 'properties' in def_schema
                ):
                    referenced_types[def_name] = _build_type_signature(
                        def_name, def_schema, schema_defs, referenced_types, tool_name, path
                    )

            # Then process the main schema
            if schema.get('type') == 'object' and 'properties' in schema:
                referenced_types[type_name] = _build_type_signature(
                    type_name, schema, schema_defs, referenced_types, tool_name, path
                )
            elif '$ref' in schema:
                ref_name = (
                    schema['$ref'][8:] if schema['$ref'].startswith('#/$defs/') else schema['$ref'].split('/')[-1]
                )
                if ref_name in schema_defs and ref_name not in referenced_types:
                    ref_schema = schema_defs[ref_name]
                    if ref_schema.get('type') == 'object' and 'properties' in ref_schema:
                        referenced_types[ref_name] = _build_type_signature(
                            ref_name, ref_schema, schema_defs, referenced_types, tool_name, path
                        )
        return

    origin = get_origin(annotation)
    args = getattr(annotation, '__args__', None)
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

        if annotation:
            type_expr = _annotation_to_type_expr(annotation, referenced_types)
        else:
            type_expr = 'Any'

        if param.default is Parameter.empty:
            params[param_name] = FunctionParam(name=param_name, type=type_expr, default=None)
        else:
            default_str = repr(param.default)
            params[param_name] = FunctionParam(name=param_name, type=type_expr, default=default_str)
    return params


def function_to_signature(
    func: Callable[..., Any],
    name: str | None = None,
    description: str | None = None,
) -> FunctionSignature:
    """Build Signature from a Python function using inspect."""
    name = name or func.__name__
    sig = signature(func)

    try:
        type_hints = get_type_hints(func, include_extras=True)
    except Exception:
        type_hints = {}

    referenced_types: dict[str, TypeSignature] = {}
    params = _build_function_params(sig, type_hints, referenced_types, name)

    return_annotation = type_hints.get('return')
    if return_annotation is not None:
        _collect_referenced_types(return_annotation, referenced_types, name, 'Return', mode='serialization')

    return_type: TypeExpr = (
        _annotation_to_type_expr(return_annotation, referenced_types) if return_annotation else 'Any'
    )

    return FunctionSignature(
        name=name,
        params=params,
        return_type=return_type,
        docstring=description,
        referenced_types=list(referenced_types.values()),
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


def _json_type_to_python(json_type: str) -> str:
    """Convert a JSON type string to Python type."""
    return _JSON_TYPE_TO_PYTHON.get(json_type, 'Any')


def _to_pascal_case(s: str) -> str:
    """Convert a string to PascalCase."""
    s = re.sub(r'[^a-zA-Z0-9]', '_', s)
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
) -> Callable[[dict[str, Any], str], TypeExpr]:
    """Set up $defs processing for a schema and return a to_type converter."""

    def to_type(schema: dict[str, Any], path: str) -> TypeExpr:
        return schema_to_type_expr(schema, defs, referenced_types, tool_name, path)

    for def_name, def_schema in defs.items():
        if def_schema.get('type') == 'object' and 'properties' in def_schema:
            if def_name not in referenced_types:
                referenced_types[def_name] = _build_type_signature(
                    def_name, def_schema, defs, referenced_types, tool_name, def_name
                )

    return to_type


def schema_to_signature(
    name: str,
    parameters_schema: dict[str, Any],
    description: str | None = None,
    return_type: str = 'Any',
    return_schema: dict[str, Any] | None = None,
) -> FunctionSignature:
    """Convert JSON schema to a FunctionSignature.

    Parameter and return schemas are processed independently — each resolves
    $refs against its own $defs. Name collisions between parameter and return
    types (e.g. both define a 'User' $def with different structures) are handled
    by `dedup_referenced_types` at a later stage.
    """
    # Process parameter schema with its own $defs
    param_defs = parameters_schema.get('$defs', {})
    param_referenced: dict[str, TypeSignature] = {}
    param_to_type = _process_schema_defs(param_defs, param_referenced, name)
    params = _build_params_from_schema(parameters_schema, param_to_type)

    # Process return schema independently (its own $defs)
    resolved_return_type: TypeExpr = return_type
    return_referenced: dict[str, TypeSignature] = {}
    if return_schema is not None and return_type == 'Any':
        return_defs = return_schema.get('$defs', {})
        return_to_type = _process_schema_defs(return_defs, return_referenced, name)
        resolved_return_type = return_to_type(return_schema, 'Return')

    # Handle case where return type couldn't be resolved
    final_description = description
    if return_schema is not None and resolved_return_type == 'Any':
        return_schema_blob = json.dumps(return_schema, indent=2)
        return_schema_note = f'\n\nReturn schema:\n{return_schema_blob}'
        final_description = (description or '') + return_schema_note
        final_description = final_description.strip()

    # Merge referenced types — dedup_referenced_types handles collisions later
    all_referenced = list(param_referenced.values())
    seen_ids = {id(ts) for ts in all_referenced}
    for ts in return_referenced.values():
        if id(ts) not in seen_ids:
            all_referenced.append(ts)
            seen_ids.add(id(ts))

    return FunctionSignature(
        name=name,
        params=params,
        return_type=resolved_return_type,
        docstring=final_description if final_description else None,
        referenced_types=all_referenced,
    )


def _build_params_from_schema(
    schema: dict[str, Any],
    to_type: Callable[[dict[str, Any], str], TypeExpr],
) -> dict[str, FunctionParam]:
    """Convert a JSON schema to a dict of FunctionParam objects."""
    properties = schema.get('properties', {})
    required = set(schema.get('required', []))

    required_params: dict[str, FunctionParam] = {}
    optional_params: dict[str, FunctionParam] = {}

    for prop_name, prop_schema in properties.items():
        type_expr = to_type(prop_schema, prop_name)

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
                nullable_expr = UnionTypeExpr(members=[type_expr, 'None'])
                optional_params[prop_name] = FunctionParam(name=prop_name, type=nullable_expr, default='None')

    return {**required_params, **optional_params}


def _schema_allows_null(schema: dict[str, Any]) -> bool:
    """Check if a schema already allows null values."""
    schema_type = schema.get('type')
    if isinstance(schema_type, list) and 'null' in schema_type:
        return True
    if 'anyOf' in schema or 'oneOf' in schema:
        union = schema.get('anyOf') or schema.get('oneOf', [])
        return any(s.get('type') == 'null' for s in union)
    return False


def schema_to_type_expr(
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
        ref_name = ref[8:] if ref.startswith('#/$defs/') else ref.split('/')[-1]
        # Ensure referenced def generates TypeSignature if needed
        if ref_name in defs and ref_name not in referenced_types:
            ref_schema = defs[ref_name]
            if ref_schema.get('type') == 'object' and 'properties' in ref_schema:
                referenced_types[ref_name] = _build_type_signature(
                    ref_name, ref_schema, defs, referenced_types, tool_name, path
                )
        # Return the TypeSignature object if available, otherwise the name string
        if ref_name in referenced_types:
            return referenced_types[ref_name]
        return ref_name

    # Handle anyOf/oneOf (union types)
    if 'anyOf' in schema:
        return _handle_union_schema(schema['anyOf'], defs, referenced_types, tool_name, path)
    if 'oneOf' in schema:
        return _handle_union_schema(schema['oneOf'], defs, referenced_types, tool_name, path)

    # Handle allOf
    if 'allOf' in schema:
        if len(schema['allOf']) == 1:
            return schema_to_type_expr(schema['allOf'][0], defs, referenced_types, tool_name, path)
        return 'Any'

    # Handle const
    if 'const' in schema:
        return f'Literal[{repr(schema["const"])}]'

    # Handle enum
    if 'enum' in schema:
        enum_values = ', '.join(repr(v) for v in schema['enum'])
        return f'Literal[{enum_values}]'

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
    if schema_type in _JSON_TYPE_TO_PYTHON and schema_type not in ('array', 'object'):
        return _JSON_TYPE_TO_PYTHON[schema_type]

    # Array type
    if schema_type == 'array':
        items = schema.get('items', {})
        if items:
            # Handle tuple schemas (items as list)
            if isinstance(items, list):
                items_list = cast(list[dict[str, Any]], items)
                item_exprs = [
                    schema_to_type_expr(item, defs, referenced_types, tool_name, f'{path}.{i}')
                    for i, item in enumerate(items_list)
                ]
                return GenericTypeExpr(base='tuple', args=item_exprs)
            item_expr = schema_to_type_expr(
                cast(dict[str, Any], items), defs, referenced_types, tool_name, f'{path}Item'
            )
            return GenericTypeExpr(base='list', args=[item_expr])
        return 'list[Any]'

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
                return 'dict[str, Any]'
            if isinstance(additional, dict):
                additional_schema = cast(dict[str, Any], additional)
                value_expr = schema_to_type_expr(additional_schema, defs, referenced_types, tool_name, f'{path}Value')
                return GenericTypeExpr(base='dict', args=['str', value_expr])
        return 'dict[str, Any]'

    # Type list (e.g., ['string', 'null'])
    if isinstance(schema_type, list):
        return _type_list_to_expr(schema_type, schema, defs, referenced_types, tool_name, path)

    return 'Any'


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
            return UnionTypeExpr(members=[base_expr, 'None'])
        return base_expr

    type_exprs: list[TypeExpr] = [_json_type_to_python(t) for t in schema_type]
    type_exprs = [t for t in type_exprs if t]
    if len(type_exprs) == 2 and 'None' in type_exprs:
        non_none = [t for t in type_exprs if t != 'None'][0]
        return UnionTypeExpr(members=[non_none, 'None'])
    if type_exprs:
        return UnionTypeExpr(members=type_exprs) if len(type_exprs) > 1 else type_exprs[0]
    return 'Any'


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
            type_exprs.append(schema_to_type_expr(s, defs, referenced_types, tool_name, path))

    # Deduplicate while preserving order (compare rendered strings)
    seen: set[str] = set()
    unique_exprs: list[TypeExpr] = []
    for expr in type_exprs:
        rendered = render_type_expr(expr)
        if rendered not in seen:
            seen.add(rendered)
            unique_exprs.append(expr)

    if has_null:
        unique_exprs.append('None')

    if len(unique_exprs) == 1:
        return unique_exprs[0]
    return UnionTypeExpr(members=unique_exprs)


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
        type_expr = schema_to_type_expr(prop_schema, defs, referenced_types, tool_name, prop_path)
        is_required = prop_name in required
        desc = prop_schema.get('description', '') or None

        fields[prop_name] = TypeFieldSignature(
            name=prop_name,
            type=type_expr,
            required=is_required,
            description=desc,
        )

    return TypeSignature(name=name, fields=fields)


# =============================================================================
# Deduplication
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
                # First occurrence — keep it
                seen[name] = type_sig
                deduped.append(type_sig)
            elif seen[name].structurally_equal(type_sig):
                # Same name, same definition — unify to canonical instance
                canonical = seen[name]
                _replace_type_refs(sig, type_sig, canonical)
                deduped.append(canonical)
            else:
                # Same name, different definition — rename to avoid conflict
                new_name = f'{sig.name}_{name}'
                type_sig.name = new_name  # Mutate in place → propagates everywhere via TypeExpr refs
                seen[new_name] = type_sig
                deduped.append(type_sig)
        sig.referenced_types = deduped


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
