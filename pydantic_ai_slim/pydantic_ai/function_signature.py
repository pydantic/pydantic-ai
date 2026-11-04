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
from collections import Counter
from collections.abc import Iterable, Iterator
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, TypeAlias, assert_never, cast
from urllib.parse import unquote, urldefrag, urljoin

# Set during rendering and consulted by TypeSignature.display_name. Not being empty means some names had to change,
# so a type uses the name `get_conflicting_type_names` stored on it. The mapping (e.g. {'User': 'tool_a_User'}) is only
# the fallback for a set of names that call didn't resolve.
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


# One level of nesting inside a rendered docstring (Google style: four spaces).
_DOC_INDENT = '    '

# Keeps arbitrary description text from breaking the generated `"""..."""` literal:
# backslashes would start escape sequences, null bytes are rejected by compile(),
# and quotes could terminate the delimiter.
_DESCRIPTION_ESCAPES = str.maketrans({'\\': '\\\\', '\0': '\\x00', '"': '\\"'})


def _render_description(text: str, indent: str = '') -> list[str]:
    """Render a description as a list of indented docstring lines."""
    text = text.strip().translate(_DESCRIPTION_ESCAPES)
    description_lines = text.splitlines()
    if len(description_lines) > 1:
        lines = [f'{indent}"""']
        for line in description_lines:
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

    def __post_init__(self) -> None:
        # Set by `FunctionSignature.get_conflicting_type_names` when this type's name has to change.
        # Not a field, as it comes from the signatures being rendered together, not from the type.
        self._resolved_name: str | None = None

    @property
    def display_name(self) -> str:
        """The type name, or the one rendering context calls for when some names have to change."""
        overrides = _type_name_overrides.get()
        if overrides and self._resolved_name:
            return self._resolved_name
        return overrides.get(self.name, self.name)

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
        """Compare two TypeSignatures structurally, all the way down, ignoring descriptions.

        The two types' own names don't count, but the names of the types nested inside them do,
        including a type nested inside itself.
        """
        classes = _structure_classes(_with_nested_types([self, other]))
        return _fields_key(self, classes) == _fields_key(other, classes)


@dataclass(kw_only=True)
class FunctionParam:
    """A single parameter in a function signature."""

    name: str
    type: TypeExpr
    default: str | None = None
    description: str | None = None
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

    # Names the rendered text relies on, so a type definition can't take them.
    _RENDERED_TEXT_NAMES: ClassVar[frozenset[str]] = frozenset(
        {
            'asyncio',
            'Any',
            'Literal',
            'NotRequired',
            'TypedDict',
            'bool',
            'dict',
            'float',
            'int',
            'list',
            'str',
            'tuple',
        }
    )

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
            conflicting_type_names: The set returned by a `get_conflicting_type_names` call made just before rendering.
                Names in a set from elsewhere take a tool-name prefix.
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

        description_sections = [description] if description else []

        args_lines: list[str] = []
        for param in self.params.values():
            description_lines = (param.description or '').strip().splitlines()
            if not description_lines:
                continue
            first_line, *continuation = description_lines
            args_lines.append(f'{_DOC_INDENT}{param.name}: {first_line}')
            # Continuation lines nest one level past the parameter name
            args_lines.extend(f'{_DOC_INDENT * 2}{line}' for line in continuation)
        if args_lines:
            description_sections.append('\n'.join(['Args:', *args_lines]))
        description = '\n\n'.join(description_sections)

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

        Parameter and return schemas are separate documents, and each resolves `$ref`s against
        itself. Each schema object becomes its own `TypeSignature`, and identical ones are then
        unified. Types left with one name and different structures (e.g. a parameter and a return
        `User`) are told apart by `get_conflicting_type_names` at a later stage.
        """
        _ref_expansions.set(0)
        param_referenced: dict[int, TypeSignature] = {}
        _process_schema_defs(parameters_schema, param_referenced, name)
        params = _build_params_from_schema(parameters_schema, param_referenced, name)

        resolved_return_type: TypeExpr = _ANY
        return_referenced: dict[int, TypeSignature] = {}
        if return_schema is not None:
            _process_schema_defs(return_schema, return_referenced, name)
            resolved_return_type = _schema_to_type_expr(return_schema, return_schema, return_referenced, name, 'Return')

        signature = cls(
            name=name,
            params=params,
            return_type=resolved_return_type,
            referenced_types=[*param_referenced.values(), *return_referenced.values()],
        )
        FunctionSignature._unify_identical_types([signature])
        return signature

    @staticmethod
    def get_conflicting_type_names(signatures: list[FunctionSignature]) -> frozenset[str]:
        """Give every type across multiple tool signatures one name and one definition.

        Each signature keeps all its referenced types (so it remains self-contained),
        but identical types (same name and structure all the way down) are unified to
        the same object instance.

        A type keeps its name unless another type or a tool has it, or the rendered text
        relies on it. Then it takes the name of the first tool using it as a prefix
        (e.g. `get_user_Address`), and a number if that is taken too.

        The new names are stored on the types, and apply when this set is passed on to
        `FunctionSignature.render(conflicting_type_names=...)`. Render straight after this
        call, as resolving a signature again, among other signatures, replaces them.

        Returns the set of type names that had to change.

        Use `collect_unique_referenced_types()` when rendering to emit each
        definition once.
        """
        FunctionSignature._unify_identical_types(signatures)
        types = FunctionSignature.collect_unique_referenced_types(signatures)
        owners = FunctionSignature._type_owners(signatures)
        reserved = FunctionSignature._RENDERED_TEXT_NAMES | {sig.name for sig in signatures}
        counts = Counter(t.name for t in types)
        conflicting = {name for name, count in counts.items() if count > 1 or name in reserved}

        # Names that stay are taken before any is generated, so a generated name can't take a real one.
        used = {*reserved, *(counts.keys() - conflicting)}
        for type_sig in types:
            resolved = None
            if type_sig.name in conflicting:
                base = resolved = f'{owners[id(type_sig)]}_{type_sig.name}'
                suffix = 2
                while resolved in used:
                    resolved, suffix = f'{base}_{suffix}', suffix + 1
                used.add(resolved)
            # The only place that sets this attribute: resolving names is this method's job.
            type_sig._resolved_name = resolved  # pyright: ignore[reportPrivateUsage]
        return frozenset(conflicting)

    @staticmethod
    def _unify_identical_types(signatures: list[FunctionSignature]) -> None:
        """Make types with one name and one structure a single object everywhere, keeping the first seen."""
        # A hand-built signature needn't list every type its expressions name, so start from those as well.
        exprs = [expr for sig in signatures for expr in (*(p.type for p in sig.params.values()), sig.return_type)]
        listed = (t for sig in signatures for t in sig.referenced_types)
        types = _with_nested_types([*listed, *(t for expr in exprs for t in _types_in(expr))])

        classes = _structure_classes(types)
        first_of_class: dict[object, TypeSignature] = {}
        canonical = {id(t): first_of_class.setdefault(classes[id(t)], t) for t in types}

        for sig in signatures:
            listed_once = {id(canonical[id(t)]): canonical[id(t)] for t in sig.referenced_types}
            sig.referenced_types = list(listed_once.values())
            for param in sig.params.values():
                param.type = FunctionSignature._replace_types(param.type, canonical)
            sig.return_type = FunctionSignature._replace_types(sig.return_type, canonical)
        for type_sig in types:
            for f in type_sig.fields.values():
                f.type = FunctionSignature._replace_types(f.type, canonical)

    @staticmethod
    def _type_owners(signatures: list[FunctionSignature]) -> dict[int, str]:
        """The name of the first signature listing each type, by type identity."""
        owners: dict[int, str] = {}
        for sig in signatures:
            for type_sig in sig.referenced_types:
                owners.setdefault(id(type_sig), sig.name)
        return owners

    @staticmethod
    def _replace_types(expr: TypeExpr, canonical: dict[int, TypeSignature]) -> TypeExpr:
        """Point an expression at canonical types."""
        if isinstance(expr, TypeSignature):
            return canonical[id(expr)]
        if isinstance(expr, GenericTypeExpr):
            expr.args = [FunctionSignature._replace_types(arg, canonical) for arg in expr.args]
        elif isinstance(expr, UnionTypeExpr):
            # Types that have just become one are one member.
            return _union_of([FunctionSignature._replace_types(member, canonical) for member in expr.members])
        return expr

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

        A type whose name had to change uses the one `get_conflicting_type_names` stored on it:
        a tool-name prefix (e.g. `get_user_Address`), and a number if that is taken too.

        Args:
            signatures: The function signatures, straight after `get_conflicting_type_names`.
            conflicting_type_names: The set returned by `get_conflicting_type_names`.

        Returns:
            A list of rendered TypedDict class definitions as strings.
        """
        unique_types = FunctionSignature.collect_unique_referenced_types(signatures)
        if not unique_types:
            return []

        owner_for = FunctionSignature._type_owners(signatures)

        return [
            tsig.render_definition(owner_name=owner_for[id(tsig)], conflicting_type_names=conflicting_type_names)
            for tsig in unique_types
        ]


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


# The non-object defs currently being resolved inline, by schema identity. A non-object def has no
# TypedDict to point at, so a self-referential one (`type Json = str | list[Json]`) would
# recurse forever; `_build_and_register_type` solves the same problem for object defs by
# registering a placeholder before it walks the fields.
_resolving_refs: ContextVar[frozenset[int]] = ContextVar('_resolving_refs', default=frozenset())

# How many `$ref`s to non-object targets one `from_schema` call expands, as a small schema can expand exponentially.
_MAX_REF_EXPANSIONS = 1_000
_ref_expansions: ContextVar[int] = ContextVar('_ref_expansions', default=0)


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


def _normalize_schema_node(node: dict[str, Any] | bool) -> dict[str, Any]:
    """Normalize a JSON Schema node to a dict.

    Per the JSON Schema spec, `True` and `False` are valid schemas anywhere a schema may
    appear (most commonly as `additionalProperties`, but also as a property value or an
    `allOf`/`anyOf`/`oneOf`/`items`/`$defs` member). `True` (equivalent to `{}`) permits any
    value; `False` permits none and is *not* equivalent to `{}`. Neither the "anything" nor
    the "nothing" case maps to a precise Python type here, so we deliberately collapse both to
    an empty schema, which the walker renders as `Any` — the practical fallback for the model.
    Non-bool nodes are returned unchanged.

    Called at every point where a nested node may be a raw schema, so no walker has to
    special-case the boolean form.
    """
    return {} if isinstance(node, bool) else node


def _process_schema_defs(
    root: dict[str, Any],
    referenced_types: dict[int, TypeSignature],
    tool_name: str,
) -> None:
    """Add TypeSignatures for the object-type `$defs` of a schema document. A `$ref` builds any other target."""
    for def_name, def_schema_raw in root.get('$defs', {}).items():
        def_schema = _normalize_schema_node(def_schema_raw)
        if _is_typed_dict_schema(def_schema):
            _build_and_register_type(def_name, def_schema, root, referenced_types, tool_name, def_name)


# Keywords whose value maps names to schemas, so a pointer ending on one has landed on the map and not on a schema.
_MAPS_OF_SCHEMAS = frozenset({'$defs', 'definitions', 'dependentSchemas', 'patternProperties', 'properties'})


def _is_typed_dict_schema(schema: dict[str, Any]) -> bool:
    """Whether a schema is an object with named fields, which can be nullable too."""
    schema_type = schema.get('type')
    is_object = schema_type == 'object' or (isinstance(schema_type, list) and 'object' in schema_type)
    return is_object and 'properties' in schema


def _ref_pointer(root: dict[str, Any], ref: Any) -> list[str] | None:
    """The JSON pointer tokens of a `$ref` into its own document, or `None` if it doesn't point into that document.

    `$ref` is a URI reference resolved against the document's `$id`, so a URI naming the document itself is local
    (https://json-schema.org/draft/2020-12/json-schema-core#section-8.2.3.1). Anything else is another document or an
    anchor, neither of which is followed. A fragment is percent-decoded, then split, then has `~1` and `~0` decoded in
    that order (RFC 6901, sections 4 and 6).
    """
    if not isinstance(ref, str):
        return None
    try:
        uri, fragment = urldefrag(ref)
        if uri:
            base = root.get('$id')
            if not isinstance(base, str) or urldefrag(urljoin(base, uri)).url != urldefrag(base).url:
                return None
    except ValueError:  # Not a URI, so it has no target.
        return None
    pointer = unquote(fragment)
    if pointer and not pointer.startswith('/'):
        return None
    return [token.replace('~1', '/').replace('~0', '~') for token in pointer.split('/')[1:]]


def _follow_pointer(root: dict[str, Any], tokens: list[str]) -> dict[str, Any] | None:
    """The schema at a JSON pointer, or `None` if it doesn't land on one."""
    node: Any = root
    at_map_of_schemas = False
    for token in tokens:
        if isinstance(node, dict):
            node = cast(dict[str, Any], node).get(token)
        elif isinstance(node, list):
            # By spelling, as RFC 6901 section 4 has one way to write an index: `01` isn't one.
            node = {str(index): item for index, item in enumerate(cast(list[Any], node))}.get(token)
        else:
            return None
        at_map_of_schemas = not at_map_of_schemas and token in _MAPS_OF_SCHEMAS
    return cast(dict[str, Any], node) if isinstance(node, dict) and not at_map_of_schemas else None


def _ref_to_type_expr(
    ref: Any,
    root: dict[str, Any],
    referenced_types: dict[int, TypeSignature],
    tool_name: str,
    path: str,
) -> TypeExpr:
    """The TypeExpr of a `$ref`'s target, or `Any`, as JSON Schema allows anything if there is no target to check."""
    tokens = _ref_pointer(root, ref)
    target = None if tokens is None else _follow_pointer(root, tokens)
    if tokens is None or target is None:
        return _ANY
    if _is_typed_dict_schema(target):
        if id(target) not in referenced_types:
            name = _ref_target_name(target, tokens, tool_name)
            _build_and_register_type(name, target, root, referenced_types, tool_name, path)
        target_type = referenced_types[id(target)]
        return UnionTypeExpr(members=[target_type, _NONE]) if 'null' in target['type'] else target_type
    # Only object defs become TypedDicts. Pydantic emits enums, constrained aliases, list aliases
    # and unions as non-object defs, so resolve them inline: `Literal[...]`, `int`, `list[str]`.
    resolving = _resolving_refs.get()
    expansions = _ref_expansions.get()
    if id(target) in resolving or expansions >= _MAX_REF_EXPANSIONS:
        # A self-referential non-object def has no finite inline expression and no TypedDict
        # to name, and `Any` never rejects a correct call, so it also stands in past the limit.
        return _ANY
    _ref_expansions.set(expansions + 1)
    token = _resolving_refs.set(resolving | {id(target)})
    try:
        return _schema_to_type_expr(target, root, referenced_types, tool_name, path)
    finally:
        _resolving_refs.reset(token)


def _ref_target_name(target: dict[str, Any], tokens: list[str], tool_name: str) -> str:
    """Named from the pointer, not the reference, so a name doesn't depend on which of the two comes first."""
    last_token = tokens[-1] if tokens else ''
    is_definition = tokens[-2:-1] in (['$defs'], ['definitions'])
    return last_token if is_definition else _object_type_name(target, tool_name, last_token)


def _object_type_name(schema: dict[str, Any], tool_name: str, path: str) -> str:
    """The `title` of an object schema if it can name a type (preserving real class names), or a path-based name."""
    title = schema.get('title')
    return title if isinstance(title, str) and title.isidentifier() else _path_to_typename(tool_name, path)


def _build_params_from_schema(
    root: dict[str, Any],
    referenced_types: dict[int, TypeSignature],
    tool_name: str,
) -> dict[str, FunctionParam]:
    """Convert a JSON schema to a dict of FunctionParam objects."""
    # A recursive model's parameters schema is a root `$ref` to its own definition, which can be a `$ref` in turn.
    schema = root
    followed = {id(root)}
    while 'properties' not in schema and (tokens := _ref_pointer(root, schema.get('$ref'))) is not None:
        target = _follow_pointer(root, tokens)
        if target is None or id(target) in followed:
            break
        schema = target
        followed.add(id(schema))
    properties = schema.get('properties', {})
    required = set(schema.get('required', []))

    required_params: dict[str, FunctionParam] = {}
    optional_params: dict[str, FunctionParam] = {}

    for prop_name, prop_schema_raw in properties.items():
        prop_schema = _normalize_schema_node(prop_schema_raw)
        type_expr = _schema_to_type_expr(prop_schema, root, referenced_types, tool_name, prop_name)
        description = prop_schema.get('description', '') or None

        if 'default' in prop_schema:
            default_str = repr(prop_schema['default'])
            optional_params[prop_name] = FunctionParam(
                name=prop_name, type=type_expr, default=default_str, description=description
            )
        elif prop_name in required:
            required_params[prop_name] = FunctionParam(
                name=prop_name, type=type_expr, default=None, description=description
            )
        else:
            # Optional without default — add | None
            if _schema_allows_null(prop_schema):
                optional_params[prop_name] = FunctionParam(
                    name=prop_name, type=type_expr, default='None', description=description
                )
            else:
                nullable_expr = UnionTypeExpr(members=[type_expr, _NONE])
                optional_params[prop_name] = FunctionParam(
                    name=prop_name, type=nullable_expr, default='None', description=description
                )

    return {**required_params, **optional_params}


def _schema_allows_null(schema: dict[str, Any]) -> bool:
    """Check if a schema already allows null values."""
    schema_type = schema.get('type')
    if isinstance(schema_type, list) and 'null' in schema_type:
        return True
    if 'anyOf' in schema:
        return any(_normalize_schema_node(s).get('type') == 'null' for s in schema['anyOf'])
    if 'oneOf' in schema:
        return any(_normalize_schema_node(s).get('type') == 'null' for s in schema['oneOf'])
    return False


def _schema_to_type_expr(
    schema: dict[str, Any] | bool,
    root: dict[str, Any],
    referenced_types: dict[int, TypeSignature],
    tool_name: str,
    path: str,
) -> TypeExpr:
    """Convert a JSON schema to a TypeExpr."""
    schema = _normalize_schema_node(schema)

    if '$ref' in schema:
        return _ref_to_type_expr(schema['$ref'], root, referenced_types, tool_name, path)

    # Handle anyOf/oneOf (union types)
    if 'anyOf' in schema:
        return _handle_union_schema(schema['anyOf'], root, referenced_types, tool_name, path)
    if 'oneOf' in schema:
        return _handle_union_schema(schema['oneOf'], root, referenced_types, tool_name, path)

    # Handle allOf
    if 'allOf' in schema:
        if len(schema['allOf']) == 1:
            return _schema_to_type_expr(schema['allOf'][0], root, referenced_types, tool_name, path)
        return _ANY

    # Handle const
    if 'const' in schema:
        return LiteralTypeExpr([schema['const']])

    # Handle enum
    if 'enum' in schema:
        return LiteralTypeExpr(schema['enum'])

    # Handle by type
    schema_type = schema.get('type')
    return _type_to_expr(schema_type, schema, root, referenced_types, tool_name, path)


def _type_to_expr(
    schema_type: str | list[str] | None,
    schema: dict[str, Any],
    root: dict[str, Any],
    referenced_types: dict[int, TypeSignature],
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
                    _schema_to_type_expr(item, root, referenced_types, tool_name, f'{path}.{i}')
                    for i, item in enumerate(items_list)
                ]
                return GenericTypeExpr(base='tuple', args=item_exprs)
            item_expr = _schema_to_type_expr(
                cast(dict[str, Any], items), root, referenced_types, tool_name, f'{path}Item'
            )
            return GenericTypeExpr(base='list', args=[item_expr])
        return GenericTypeExpr(base='list', args=[_ANY])

    # Object type
    if schema_type == 'object':
        if 'properties' in schema:
            if id(schema) in referenced_types:
                return referenced_types[id(schema)]
            name = _object_type_name(schema, tool_name, path)
            return _build_and_register_type(name, schema, root, referenced_types, tool_name, path)
        if 'additionalProperties' in schema:
            additional = schema['additionalProperties']
            if additional is True:
                return GenericTypeExpr(base='dict', args=[SimpleTypeExpr('str'), _ANY])
            if isinstance(additional, dict):
                additional_schema = cast(dict[str, Any], additional)
                value_expr = _schema_to_type_expr(additional_schema, root, referenced_types, tool_name, f'{path}Value')
                return GenericTypeExpr(base='dict', args=[SimpleTypeExpr('str'), value_expr])
        return GenericTypeExpr(base='dict', args=[SimpleTypeExpr('str'), _ANY])

    # Type list (e.g., ['string', 'null'])
    if isinstance(schema_type, list):
        return _type_list_to_expr(schema_type, schema, root, referenced_types, tool_name, path)

    return _ANY


def _type_list_to_expr(
    schema_type: list[str],
    schema: dict[str, Any],
    root: dict[str, Any],
    referenced_types: dict[int, TypeSignature],
    tool_name: str,
    path: str,
) -> TypeExpr:
    """Handle type lists like ['string', 'null']."""
    # Check if this is object with properties + null
    if 'object' in schema_type and 'properties' in schema:
        base_expr = _type_to_expr('object', schema, root, referenced_types, tool_name, path)
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
    schemas: list[dict[str, Any] | bool],
    root: dict[str, Any],
    referenced_types: dict[int, TypeSignature],
    tool_name: str,
    path: str,
) -> TypeExpr:
    """Handle anyOf/oneOf schemas, returning a TypeExpr."""
    type_exprs: list[TypeExpr] = []
    has_null = False

    for s_raw in schemas:
        s = _normalize_schema_node(s_raw)
        if s.get('type') == 'null':
            has_null = True
        else:
            type_exprs.append(_schema_to_type_expr(s, root, referenced_types, tool_name, path))

    if has_null:
        type_exprs.append(_NONE)

    return _union_of(type_exprs)


def _build_and_register_type(
    name: str,
    schema: dict[str, Any],
    root: dict[str, Any],
    referenced_types: dict[int, TypeSignature],
    tool_name: str,
    path: str,
) -> TypeSignature:
    """Build a TypeSignature, registering a placeholder first to prevent infinite recursion.

    Self-referential schemas (e.g. recursive models) would otherwise cause infinite recursion
    when `_build_type_signature` processes properties that `$ref` back to the same type.

    Returns the completed TypeSignature.
    """
    placeholder = TypeSignature(name=name)
    referenced_types[id(schema)] = placeholder
    built = _build_type_signature(name, schema, root, referenced_types, tool_name, path)
    placeholder.fields = built.fields
    placeholder.description = built.description
    return placeholder


def _build_type_signature(
    name: str,
    schema: dict[str, Any],
    root: dict[str, Any],
    referenced_types: dict[int, TypeSignature],
    tool_name: str,
    path: str,
) -> TypeSignature:
    """Build a TypeSignature from an object schema."""
    properties = schema.get('properties', {})
    required = set(schema.get('required', []))

    fields: dict[str, TypeFieldSignature] = {}

    for prop_name, prop_schema_raw in properties.items():
        prop_schema = _normalize_schema_node(prop_schema_raw)
        prop_path = f'{path}.{prop_name}' if path else prop_name
        type_expr = _schema_to_type_expr(prop_schema, root, referenced_types, tool_name, prop_path)
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


def _types_in(expr: TypeExpr) -> Iterator[TypeSignature]:
    """The types an expression names, without looking inside them."""
    if isinstance(expr, TypeSignature):
        yield expr
    elif isinstance(expr, GenericTypeExpr | UnionTypeExpr):
        for child in expr.args if isinstance(expr, GenericTypeExpr) else expr.members:
            yield from _types_in(child)


def _with_nested_types(types: Iterable[TypeSignature]) -> list[TypeSignature]:
    """The given types and every type nested inside them, each once, in the order found."""
    pending = list(types)
    found: dict[int, TypeSignature] = {}
    for type_sig in pending:
        if id(type_sig) not in found:
            found[id(type_sig)] = type_sig
            for f in type_sig.fields.values():
                pending.extend(_types_in(f.type))
    return list(found.values())


def _union_of(members: list[TypeExpr]) -> TypeExpr:
    """A union without repeats, or its one member if that's all that's left. A type only repeats as the same object."""
    classes: dict[int, object] = {id(t): id(t) for member in members for t in _types_in(member)}
    unique = list({_expr_key(member, classes): member for member in members}.values())
    return unique[0] if len(unique) == 1 else UnionTypeExpr(members=unique)


def _expr_key(expr: TypeExpr, classes: dict[int, object]) -> object:
    """A comparable form of an expression, with each type in it standing as its structure class."""
    if isinstance(expr, TypeSignature):
        return TypeSignature, classes[id(expr)]
    if isinstance(expr, GenericTypeExpr):
        return (expr.base, *(_expr_key(arg, classes) for arg in expr.args))
    if isinstance(expr, UnionTypeExpr):
        keys = list(dict.fromkeys(_expr_key(member, classes) for member in expr.members))
        return keys[0] if len(keys) == 1 else ('|', *keys)
    if isinstance(expr, SimpleTypeExpr | LiteralTypeExpr):
        return str(expr)
    assert_never(expr)


def _fields_key(type_sig: TypeSignature, classes: dict[int, object]) -> object:
    return frozenset((name, f.required, _expr_key(f.type, classes)) for name, f in type_sig.fields.items())


def _structure_classes(types: list[TypeSignature]) -> dict[int, object]:
    """Number the types so that two share a number exactly when they have one name and one structure all the way down.

    Starts by grouping types by name, then splits any group whose members' fields differ, counting a nested type by
    its group, until no group splits. Types in a cycle are only ever split by a real difference.
    """
    classes: dict[int, object] = {id(t): t.name for t in types}
    count = len(set(classes.values()))
    while True:
        numbers: dict[object, int] = {}
        classes = {id(t): numbers.setdefault((classes[id(t)], _fields_key(t, classes)), len(numbers)) for t in types}
        if len(numbers) == count:
            return classes
        count = len(numbers)
