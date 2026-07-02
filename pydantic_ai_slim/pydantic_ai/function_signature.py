"""Generate function signatures from functions and JSON schemas.

This module provides utilities to represent tool definitions as human-readable
function signatures, which LLMs can understand more easily than raw
JSON schemas. Used by code mode to present tools as callable functions.
"""

from __future__ import annotations

__all__ = (
    'FunctionSignature',
    'FunctionParam',
    'TypeSignature',
    'TypeFieldSignature',
    'TypeExpr',
    'SimpleTypeName',
    'SimpleTypeExpr',
    'LiteralTypeExpr',
    'GenericTypeExpr',
    'UnionTypeExpr',
)

import re
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias, cast

# Set during rendering to map original type names to prefixed names for
# dedup conflict resolution (e.g. {'User': 'tool_a_User'}).
# Populated by FunctionSignature.render(), consulted by TypeSignature.display_name.
_type_name_overrides: ContextVar[dict[str, str]] = ContextVar('_type_name_overrides', default={})

# =============================================================================
# Type expression tree
# =============================================================================


SimpleTypeName = Literal['str', 'int', 'float', 'bool', 'Any', 'None']


@dataclass
class SimpleTypeExpr:
    """A simple named type like `str`, `int`, `Any`, `None`."""

    name: SimpleTypeName
    kind: Literal['simple'] = 'simple'

    def __str__(self) -> str:
        return self.name


@dataclass
class LiteralTypeExpr:
    """A Literal type expression like `Literal['a', 'b']` or `Literal[42]`."""

    values: list[Any]
    kind: Literal['literal'] = 'literal'

    def __str__(self) -> str:
        return f'Literal[{", ".join(repr(v) for v in self.values)}]'


@dataclass
class GenericTypeExpr:
    """A generic type expression like `list[User]`, `dict[str, User]`, `tuple[int, str]`."""

    base: str
    args: list[TypeExpr]
    kind: Literal['generic'] = 'generic'

    def __str__(self) -> str:
        return f'{self.base}[{", ".join(str(a) for a in self.args)}]'


@dataclass
class UnionTypeExpr:
    """A union type expression like `User | None`, `str | int`."""

    members: list[TypeExpr]
    kind: Literal['union'] = 'union'

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


@dataclass(kw_only=True)
class TypeFieldSignature:
    """A single field in a TypedDict-style type definition."""

    name: str
    type: TypeExpr
    required: bool = False
    description: str | None = None
    kind: Literal['field'] = 'field'

    def __str__(self) -> str:
        """Render this field as a line in a TypedDict class body."""
        type_str = str(self.type)
        if not self.required:
            type_str = f'NotRequired[{type_str}]'
        lines: list[str] = [f'    {self.name}: {type_str}']
        if self.description:
            lines.extend(_render_description(self.description, indent='    '))
        return '\n'.join(lines)


@dataclass(kw_only=True)
class TypeSignature:
    """A TypedDict-style class definition with named fields."""

    name: str

    description: str | None = None

    fields: dict[str, TypeFieldSignature] = field(default_factory=dict[str, TypeFieldSignature])
    kind: Literal['type'] = 'type'

    @property
    def display_name(self) -> str:
        """The type name, with tool-name prefix applied if rendering context is set."""
        return _type_name_overrides.get().get(self.name, self.name)

    def __str__(self) -> str:
        """Return the type name (for use in type expressions like `def foo(x: User)`)."""
        return self.display_name

    def render_definition(
        self, *, owner_name: str | None = None, conflicting_type_names: frozenset[str] = frozenset()
    ) -> str:
        """Render the full TypedDict class definition.

        Args:
            owner_name: The owning tool name, used to build prefixed type names
                for conflicting types (e.g. `get_user_Address`).
            conflicting_type_names: Set of type names that need tool-name prefixes
                (from `get_conflicting_type_names`). Only effective when `owner_name`
                is also provided.
        """
        if owner_name and conflicting_type_names:
            overrides = {n: f'{owner_name}_{n}' for n in conflicting_type_names}
            token = _type_name_overrides.set(overrides)
            try:
                return self._render_definition()
            finally:
                _type_name_overrides.reset(token)
        return self._render_definition()

    def _render_definition(self) -> str:
        """Render the full TypedDict class definition (internal, assumes overrides are set)."""
        lines = [f'class {self.display_name}(TypedDict):']
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


@dataclass(kw_only=True)
class FunctionParam:
    """A single parameter in a function signature."""

    name: str
    type: TypeExpr
    default: str | None = None
    kind: Literal['param'] = 'param'

    def __str__(self) -> str:
        """Render this parameter as a function parameter string."""
        type_str = str(self.type)
        if self.default is not None:
            return f'{self.name}: {type_str} = {self.default}'
        return f'{self.name}: {type_str}'


@dataclass(kw_only=True)
class FunctionSignature:
    """Function signature shape with referenced type definitions.

    This class holds the structural data (params, return type, referenced types)
    needed to render a function signature. Name and description can be overridden
    at render time (e.g. from a `ToolDefinition`).
    """

    name: str
    description: str | None = None

    params: dict[str, FunctionParam] = field(default_factory=dict[str, FunctionParam])
    """Function parameters, all rendered as keyword-only (JSON schema doesn't distinguish positional/keyword)."""

    return_type: TypeExpr
    """The return type expression."""

    referenced_types: list[TypeSignature] = field(default_factory=list[TypeSignature])
    """TypedDict class definitions needed by the signature."""

    is_async: bool = False
    """Whether the underlying function is async."""

    kind: Literal['function'] = 'function'

    def render(
        self,
        body: str,
        *,
        name: str | None = None,
        description: str | None = None,
        is_async: bool | None = None,
        conflicting_type_names: frozenset[str] = frozenset(),
    ) -> str:
        """Render the signature with a specific body.

        Sets `_type_name_overrides` so that dedup-prefixed types resolve
        correctly during rendering.

        Args:
            body: The function body (e.g. `'...'` or `'return await tool()'`).
            name: The function name (also used for dedup prefix resolution). Falls back to `self.name`.
            description: Optional docstring to include. Falls back to `self.description`.
            is_async: Override async rendering. If `None`, uses `self.is_async`.
            conflicting_type_names: Set of type names that need tool-name prefixes (from `get_conflicting_type_names`).
        """
        render_name = name or self.name
        description = description if description is not None else self.description
        overrides = {n: f'{render_name}_{n}' for n in conflicting_type_names}
        token = _type_name_overrides.set(overrides)
        try:
            return self._render(body, name=render_name, description=description, is_async=is_async)
        finally:
            _type_name_overrides.reset(token)

    def _render(
        self,
        body: str,
        *,
        name: str,
        description: str | None = None,
        is_async: bool | None = None,
    ) -> str:
        async_flag = is_async if is_async is not None else self.is_async
        prefix = 'async def' if async_flag else 'def'
        params_str = ', '.join(str(p) for p in self.params.values())

        return_str = str(self.return_type)
        if params_str:
            # Force keyword-only params so LLMs always use named arguments
            parts = [f'{prefix} {name}(*, {params_str}) -> {return_str}:']
        else:
            parts = [f'{prefix} {name}() -> {return_str}:']

        if description:
            parts.extend(_render_description(description, indent='    '))

        parts.append(f'    {body}')

        return '\n'.join(parts)

    @classmethod
    def from_schema(
        cls,
        *,
        name: str,
        parameters_schema: dict[str, Any],
        return_schema: dict[str, Any] | None = None,
    ) -> FunctionSignature:
        """Build a FunctionSignature from JSON schemas.

        `name` is stored on the resulting signature and also used for generating
        fallback type names (e.g. `GetUserAddress`) when the schema has no `title`.

        Parameter and return schemas are processed independently — each resolves
        `$ref`s against its own `$defs`. Name collisions between parameter and return
        types (e.g. both define a `User` `$def` with different structures) are handled
        by `get_conflicting_type_names` at a later stage.
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
            resolved_return_type = _schema_to_type_expr(
                return_schema, return_defs, return_referenced, name, path='Return'
            )

        # Merge referenced types, deduplicating structurally identical types within this signature.
        # Cross-signature collisions are handled later by get_conflicting_type_names.
        all_referenced = list(param_referenced.values())
        for ret_type in return_referenced.values():
            existing = param_referenced.get(ret_type.name)
            if existing is not None and existing.structurally_equal(ret_type):
                continue  # already present from param schema
            all_referenced.append(ret_type)

        return cls(
            name=name,
            params=params,
            return_type=resolved_return_type,
            referenced_types=all_referenced,
        )

    @staticmethod
    def get_conflicting_type_names(signatures: list[FunctionSignature]) -> frozenset[str]:
        """Identify TypedDict name conflicts across multiple tool signatures.

        Each signature keeps all its referenced types (so it remains self-contained),
        but identical types (same name and structure) are unified to the same object
        instance.

        Returns the set of type names that have conflicts (same name, different
        structure) and need tool-name prefixes at render time. Pass this set to
        `FunctionSignature.render(conflicting_type_names=...)`.

        Use `collect_unique_referenced_types()` when rendering to emit each
        definition once.
        """
        seen: dict[str, TypeSignature] = {}
        prefixed: set[str] = set()

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
                    prefixed.add(name)
                    deduped.append(type_sig)
            sig.referenced_types = deduped

        return frozenset(prefixed)

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

    @staticmethod
    def render_type_definitions(
        signatures: list[FunctionSignature],
        conflicting_type_names: frozenset[str],
    ) -> list[str]:
        """Render unique TypedDict definitions for a set of function signatures.

        For types whose names conflict across signatures (as identified by
        `get_conflicting_type_names`), each definition is rendered with a
        tool-name prefix (e.g. `get_user_Address`).

        Args:
            signatures: The function signatures (after `get_conflicting_type_names`).
            conflicting_type_names: The set returned by `get_conflicting_type_names`.

        Returns:
            A list of rendered TypedDict class definitions as strings.
        """
        unique_types = FunctionSignature.collect_unique_referenced_types(signatures)
        if not unique_types:
            return []

        owner_for: dict[int, str] = {}
        for sig in signatures:
            for tsig in sig.referenced_types:
                if tsig.name in conflicting_type_names and id(tsig) not in owner_for:
                    owner_for[id(tsig)] = sig.name

        rendered: list[str] = []
        for tsig in unique_types:
            owner = owner_for.get(id(tsig))
            rendered.append(tsig.render_definition(owner_name=owner, conflicting_type_names=conflicting_type_names))
        return rendered


# Shared singletons
_ANY = SimpleTypeExpr('Any')
_NONE = SimpleTypeExpr('None')


# =============================================================================
# JSON schema to signature conversion
# =============================================================================


_JSON_SIMPLE_TYPE_TO_PYTHON: dict[str, SimpleTypeName] = {
    'string': 'str',
    'integer': 'int',
    'number': 'float',
    'boolean': 'bool',
    'null': 'None',
}

_JSON_TYPE_TO_PYTHON: dict[str, str] = {
    **_JSON_SIMPLE_TYPE_TO_PYTHON,
    'array': 'list',
    'object': 'dict',
}


def _json_type_to_python(json_type: str) -> SimpleTypeExpr:
    """Convert a JSON type string to a SimpleTypeExpr."""
    return SimpleTypeExpr(_JSON_SIMPLE_TYPE_TO_PYTHON.get(json_type, 'Any'))


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
        return TypeSignature(name=ref_name)

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
    if isinstance(schema_type, str) and schema_type in _JSON_SIMPLE_TYPE_TO_PYTHON:
        return SimpleTypeExpr(_JSON_SIMPLE_TYPE_TO_PYTHON[schema_type])

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
            # Use `title` from the schema if available and is a valid Python identifier
            # (preserves real class names like `User`), otherwise fall back to a path-based name
            title = schema.get('title')
            td_name = title if title and title.isidentifier() else _path_to_typename(tool_name, path)
            if td_name not in referenced_types:
                _build_and_register_type(td_name, schema, defs, referenced_types, tool_name, path)
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
