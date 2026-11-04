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

import keyword
import re
import unicodedata
from collections.abc import Callable, Iterable, Iterator
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias, assert_never, cast
from urllib.parse import unquote

# Set during rendering, consulted by TypeSignature.display_name. `None` when no type name had to change.
_type_name_overrides: ContextVar[dict[str, str] | None] = ContextVar('_type_name_overrides', default=None)


def _overrides_for(conflicting_type_names: frozenset[str], owner_name: str | None) -> dict[str, str] | None:
    if not conflicting_type_names:
        return None
    return {n: f'{owner_name}_{n}' for n in conflicting_type_names} if owner_name else {}


def _reads_back_exactly(name: str) -> bool:
    """Whether Python source can use `name` as exactly this name: not a keyword, and not changed by normalization."""
    return (
        name.isidentifier()
        and not keyword.iskeyword(name)
        and name != '__debug__'
        and unicodedata.normalize('NFKC', name) == name
    )


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


def _render_comment(text: str | None, *, indent: str) -> list[str]:
    """Render a description as comment lines, for the places a docstring can't go."""
    if not text:
        return []
    return [f'{indent}# {line}'.rstrip() for line in text.strip().replace('\0', '\\x00').splitlines()]


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
        lines: list[str] = [f'    {self.name}: {_annotation(self)}']
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
        # Set by `FunctionSignature.get_conflicting_type_names`. Not a field, as it isn't part of what the type is.
        self._resolved_name: str | None = None

    @property
    def display_name(self) -> str:
        """The name this type renders under: its own unless the rendering context says some names had to change."""
        overrides = _type_name_overrides.get()
        if overrides is None:
            return self.name
        return self._resolved_name or overrides.get(self.name, self.name)

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
        token = _type_name_overrides.set(_overrides_for(conflicting_type_names, owner_name))
        try:
            return self._render_definition()
        finally:
            _type_name_overrides.reset(token)

    def _render_definition(self) -> str:
        """Render the full TypedDict definition (internal, assumes render names are set)."""
        # Beyond names Python can't read back, a double underscore prefix is mangled inside a class, and a field
        # named like something its class's annotations use makes a type checker read those as the field.
        used = {name for f in self.fields.values() for name in _names_in(f)}
        if any(
            not _reads_back_exactly(name) or (name.startswith('__') and not name.endswith('__')) or name in used
            for name in self.fields
        ):
            return self._render_functional_definition()
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

    def _render_functional_definition(self) -> str:
        """Render the `Name = TypedDict('Name', {...})` form, with descriptions as comments."""
        name = self.display_name
        lines = [*_render_comment(self.description, indent=''), f'{name} = TypedDict({name!r}, {{']
        for f in self.fields.values():
            lines.extend(_render_comment(f.description, indent=_DOC_INDENT))
            lines.append(f'{_DOC_INDENT}{f.name!r}: {_annotation(f)},')
        lines.append('})')
        return '\n'.join(lines)

    def structurally_equal(self, other: TypeSignature) -> bool:
        """Compare two TypeSignatures structurally, all the way down.

        The two types' own names and every description are ignored. The names of types nested inside them
        count, as a field typed `User` is not the same as one typed `Person`. Recursive types are safe.
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

    def __post_init__(self) -> None:
        # The type describing `**kwargs`, when a parameter's name can't be written in a parameter list. It follows from
        # `params` at construction, so `dataclasses.replace` rebuilds it, and it isn't a field. The parameter stays
        # `Any` rather than `Unpack[...]`: a type checker tests a `**{...}` literal's value type against every field,
        # so `Unpack` rejects a correct call whenever the fields have different types.
        self._kwargs_type: TypeSignature | None = None
        if not all(_reads_back_exactly(param.name) for param in self.params.values()):
            self._kwargs_type = TypeSignature(
                name=f'{_to_pascal_case(self.name)}Args',
                fields={
                    param.name: TypeFieldSignature(name=param.name, type=param.type, required=param.default is None)
                    for param in self.params.values()
                },
            )

    def _listed_types(self) -> list[TypeSignature]:
        return self.referenced_types if self._kwargs_type is None else [*self.referenced_types, self._kwargs_type]

    def _reachable_types(self) -> list[TypeSignature]:
        """Every type this signature uses, whether or not it is listed, listed ones first."""
        exprs = [*(param.type for param in self.params.values()), self.return_type]
        return _with_nested_types([*self._listed_types(), *(t for expr in exprs for t in _types_in(expr))])

    @staticmethod
    def _unify_identical_types(signatures: list[FunctionSignature]) -> None:
        """Make types with one name and one structure a single object everywhere, keeping the first seen.

        Each signature ends up listing all the types it uses, so it remains self-contained.
        """
        reachable = {id(signature): signature._reachable_types() for signature in signatures}
        types = _with_nested_types(t for signature_types in reachable.values() for t in signature_types)
        classes = _structure_classes(types)
        first_of_class: dict[object, TypeSignature] = {}
        canonical = {id(t): first_of_class.setdefault(classes[id(t)], t) for t in types}
        for signature in signatures:
            kwargs_type = signature._kwargs_type
            listed = (canonical[id(t)] for t in reachable[id(signature)] if t is not kwargs_type)
            signature.referenced_types = list({id(t): t for t in listed}.values())
            for param in signature.params.values():
                param.type = _replace_types(param.type, canonical)
            signature.return_type = _replace_types(signature.return_type, canonical)
            if kwargs_type is not None:
                signature._kwargs_type = canonical[id(kwargs_type)]
        for type_sig in types:
            for f in type_sig.fields.values():
                f.type = _replace_types(f.type, canonical)

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

        Args:
            body: The function body (e.g. `'...'` or `'return await tool()'`).
            name: The function name. Falls back to `self.name`.
            description: Optional docstring to include. Falls back to `self.description`.
            is_async: Override async rendering. If `None`, uses `self.is_async`.
            conflicting_type_names: Set of type names that need tool-name prefixes, for types
                `get_conflicting_type_names` has not named.
        """
        render_name = name or self.name
        description = description if description is not None else self.description
        token = _type_name_overrides.set(_overrides_for(conflicting_type_names, render_name))
        try:
            return self._render(body, name=render_name, description=description, is_async=is_async)
        finally:
            _type_name_overrides.reset(token)

    def _render_args_lines(self) -> list[str]:
        lines: list[str] = []
        for param in self.params.values():
            # Without a parameter list to show them in, defaults go here.
            default = (
                f'Defaults to `{param.default}`.' if self._kwargs_type is not None and param.default is not None else ''
            )
            description_lines = f'{(param.description or "").strip()} {default}'.strip().splitlines()
            if not description_lines:
                continue
            first_line, *continuation = description_lines
            lines.append(f'{_DOC_INDENT}{param.name}: {first_line}')
            # Continuation lines nest one level past the parameter name
            lines.extend(f'{_DOC_INDENT * 2}{line}' for line in continuation)
        return lines

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
        if self._kwargs_type is not None:
            parts = [f'{prefix} {name}(**kwargs: Any) -> {return_str}:']
        elif params_str:
            # Force keyword-only params so LLMs always use named arguments
            parts = [f'{prefix} {name}(*, {params_str}) -> {return_str}:']
        else:
            parts = [f'{prefix} {name}() -> {return_str}:']

        description_sections = [description] if description else []
        if self._kwargs_type is not None:
            description_sections.append(
                f'Some argument names are not valid Python keywords, so pass arguments as `{name}(**{{...}})`. '
                f'`{self._kwargs_type}` describes them.'
            )

        args_lines = self._render_args_lines()
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

        Parameter and return schemas are separate documents, and each resolves `$ref`s
        against itself. Each schema object becomes its own `TypeSignature`, whatever its
        name, and identical ones are then unified. Types left with one name and different
        structures (e.g. a parameter and a return `User`) are told apart by
        `get_conflicting_type_names` at a later stage.
        """
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
        """Resolve one name for every type across multiple tool signatures.

        Types with the same name and the same structure all the way down are unified to one object,
        and each signature keeps all the types it uses, so it remains self-contained.

        A type keeps its own name when that is the only type with that name and Python can use it there.
        Otherwise it takes the name of the first tool using it as a prefix (e.g. `get_user_Address`).

        The names are stored on the types, so render straight after this call: resolving a signature
        again, as part of another set of signatures, gives its types the names of that set.

        Returns the names that had to change.
        """
        FunctionSignature._unify_identical_types(signatures)
        owners: dict[int, str] = {}
        for signature in signatures:
            for type_sig in signature._listed_types():
                owners.setdefault(id(type_sig), signature.name)
        types = FunctionSignature.collect_unique_referenced_types(signatures)
        names = _assign_type_names(types, signatures, owners)
        for type_sig in types:
            # The only place that sets this attribute: resolving names is this method's job.
            type_sig._resolved_name = names[id(type_sig)]  # pyright: ignore[reportPrivateUsage]
        return frozenset(t.name for t in types if names[id(t)] != t.name)

    @staticmethod
    def collect_unique_referenced_types(signatures: list[FunctionSignature]) -> list[TypeSignature]:
        """Collect unique TypeSignature objects from signatures, deduplicating by identity."""
        seen_ids: set[int] = set()
        result: list[TypeSignature] = []
        for sig in signatures:
            for type_sig in sig._listed_types():
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

        Each definition takes the name `get_conflicting_type_names` resolved for it. The signatures
        can be a subset of the ones it was given, which keeps the names of the whole set.

        Args:
            signatures: The function signatures (after `get_conflicting_type_names`).
            conflicting_type_names: Set of type names that need tool-name prefixes, for types
                `get_conflicting_type_names` has not named.

        Returns:
            A list of rendered TypedDict definitions as strings.
        """
        unique_types = FunctionSignature.collect_unique_referenced_types(signatures)
        if not unique_types:
            return []

        owner_for: dict[int, str] = {}
        for sig in signatures:
            for tsig in sig._listed_types():
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


_JSON_POINTER_INDEX_RE = re.compile(r'0|[1-9][0-9]{0,8}')

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


# The non-object schemas currently being resolved inline, by identity. A non-object schema has
# no TypedDict to point at, so a self-referential one (`type Json = str | list[Json]`) would
# recurse forever; `_build_and_register_type` solves the same problem for object schemas by
# registering a placeholder before it walks the fields, and clears this set, as recursion
# through a placeholder ends.
_resolving_refs: ContextVar[frozenset[int]] = ContextVar('_resolving_refs', default=frozenset())


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
        if _is_typed_dict_schema(def_schema) and id(def_schema) not in referenced_types:
            _build_and_register_type(def_name, def_schema, root, referenced_types, tool_name, def_name)


def _is_typed_dict_schema(schema: dict[str, Any]) -> bool:
    return schema.get('type') == 'object' and 'properties' in schema


def _json_pointer_child(node: Any, token: str) -> Any:
    """The value a JSON pointer token names inside `node`, or `None` if there is none."""
    if isinstance(node, dict):
        return cast(dict[str, Any], node).get(token)
    if isinstance(node, list) and _JSON_POINTER_INDEX_RE.fullmatch(token) and int(token) < len(cast(list[Any], node)):
        return cast(list[Any], node)[int(token)]
    return None


def _resolve_json_pointer(root: dict[str, Any], ref: Any) -> tuple[dict[str, Any], str | None] | None:
    """Find the schema a `$ref` points at within its document.

    Returns the schema and, if it is a definition, its name. Returns `None` for anything this document
    can't answer: a remote reference, an anchor, or a pointer that does not land on a schema.
    """
    if not isinstance(ref, str) or not ref.startswith('#'):
        return None
    pointer = unquote(ref[1:])
    if pointer and not pointer.startswith('/'):
        return None
    tokens = [t.replace('~1', '/').replace('~0', '~') for t in pointer.split('/')[1:]]
    node: Any = root
    for token in tokens:
        node = _json_pointer_child(node, token)
    if isinstance(node, bool):
        return {}, None
    if not isinstance(node, dict):
        return None
    def_name = tokens[-1] if len(tokens) >= 2 and tokens[-2] in ('$defs', 'definitions') else None
    return cast(dict[str, Any], node), def_name


def _ref_to_type_expr(
    ref: Any,
    root: dict[str, Any],
    referenced_types: dict[int, TypeSignature],
    tool_name: str,
    path: str,
) -> TypeExpr:
    """Convert a `$ref` to the same TypeExpr its target gets, or `Any` if there is no target in this document."""
    resolved = _resolve_json_pointer(root, ref)
    if resolved is None:
        return _ANY
    target, def_name = resolved
    if id(target) in referenced_types:
        return referenced_types[id(target)]
    if _is_typed_dict_schema(target):
        name = def_name or _object_type_name(target, tool_name, '' if target is root else path)
        return _build_and_register_type(name, target, root, referenced_types, tool_name, path)
    # Only object defs become TypedDicts. Pydantic emits enums, constrained aliases, list aliases
    # and unions as non-object defs, so resolve them inline: `Literal[...]`, `int`, `list[str]`.
    resolving = _resolving_refs.get()
    if id(target) in resolving:
        # A self-referential non-object def has no finite inline expression and no TypedDict
        # to name, so `Any` keeps the rendered signature self-contained.
        return _ANY
    token = _resolving_refs.set(resolving | {id(target)})
    try:
        return _schema_to_type_expr(target, root, referenced_types, tool_name, path)
    finally:
        _resolving_refs.reset(token)


def _object_type_name(schema: dict[str, Any], tool_name: str, path: str) -> str:
    """The `title` of an object schema when it can name a type (preserving real class names), or a path-based name."""
    title = schema.get('title')
    return title if isinstance(title, str) and title.isidentifier() else _path_to_typename(tool_name, path)


def _build_params_from_schema(
    root: dict[str, Any],
    referenced_types: dict[int, TypeSignature],
    tool_name: str,
) -> dict[str, FunctionParam]:
    """Convert a JSON schema to a dict of FunctionParam objects."""
    schema = root
    followed = {id(root)}
    # A recursive model's parameters schema is a root `$ref` to its own definition.
    while (resolved := _resolve_json_pointer(root, schema.get('$ref'))) and id(resolved[0]) not in followed:
        schema = resolved[0]
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

    # Handle $ref
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
        # `Literal[]` is not valid Python.
        return LiteralTypeExpr(schema['enum']) if schema['enum'] else _ANY

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
            td_name = _object_type_name(schema, tool_name, path)
            return _build_and_register_type(td_name, schema, root, referenced_types, tool_name, path)
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

    unique_exprs = _unique_exprs(type_exprs)

    if has_null:
        unique_exprs.append(_NONE)

    if not unique_exprs:
        return _ANY
    if len(unique_exprs) == 1:
        return unique_exprs[0]
    return UnionTypeExpr(members=unique_exprs)


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
    token = _resolving_refs.set(frozenset())
    try:
        built = _build_type_signature(name, schema, root, referenced_types, tool_name, path)
    finally:
        _resolving_refs.reset(token)
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
# Resolving one catalog: which types exist, and what each is called
# =============================================================================


# =============================================================================
# Resolving one set of signatures
# =============================================================================


def _annotation(type_field: TypeFieldSignature) -> str:
    return str(type_field.type) if type_field.required else f'NotRequired[{type_field.type}]'


def _expr_nodes(expr: TypeExpr) -> Iterator[TypeExpr]:
    """An expression and every expression inside it, without looking inside the types it names."""
    yield expr
    children = (
        expr.args if isinstance(expr, GenericTypeExpr) else expr.members if isinstance(expr, UnionTypeExpr) else []
    )
    for child in children:
        yield from _expr_nodes(child)


def _types_in(expr: TypeExpr) -> Iterator[TypeSignature]:
    return (node for node in _expr_nodes(expr) if isinstance(node, TypeSignature))


def _names_in(type_field: TypeFieldSignature) -> Iterator[str]:
    """The names a field's annotation uses, leaving out what is only text, like the values of a `Literal`."""
    if not type_field.required:
        yield 'NotRequired'
    for node in _expr_nodes(type_field.type):
        if isinstance(node, GenericTypeExpr):
            yield node.base
        elif isinstance(node, LiteralTypeExpr):
            yield 'Literal'
        elif isinstance(node, SimpleTypeExpr | TypeSignature):
            yield str(node)
        elif not isinstance(node, UnionTypeExpr):
            assert_never(node)


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


def _expr_key(expr: TypeExpr, type_key: Callable[[TypeSignature], object] = id) -> object:
    """A key for an expression, with its types standing as `type_key` says, whatever they render as."""
    if isinstance(expr, TypeSignature):
        return TypeSignature, type_key(expr)
    if isinstance(expr, GenericTypeExpr):
        return (expr.base, *(_expr_key(arg, type_key) for arg in expr.args))
    if isinstance(expr, UnionTypeExpr):
        return ('|', *(_expr_key(member, type_key) for member in expr.members))
    if isinstance(expr, SimpleTypeExpr | LiteralTypeExpr):
        return str(expr)
    assert_never(expr)


def _unique_exprs(exprs: list[TypeExpr]) -> list[TypeExpr]:
    unique: dict[object, TypeExpr] = {}
    for expr in exprs:
        unique.setdefault(_expr_key(expr), expr)
    return list(unique.values())


def _fields_key(type_sig: TypeSignature, classes: dict[int, object]) -> object:
    return frozenset(
        (name, f.required, _expr_key(f.type, lambda nested: classes[id(nested)])) for name, f in type_sig.fields.items()
    )


def _structure_classes(types: list[TypeSignature]) -> dict[int, object]:
    """Number the types so that two get one number exactly when they are the same all the way down.

    Starts by grouping types by name, then splits any group whose members' fields differ, counting a nested
    type by its group, until no group splits. Types in a cycle are only ever split by a real difference.
    """
    classes: dict[int, object] = {id(t): t.name for t in types}
    count = len(set(classes.values()))
    while True:
        numbers: dict[object, int] = {}
        classes = {id(t): numbers.setdefault((classes[id(t)], _fields_key(t, classes)), len(numbers)) for t in types}
        if len(numbers) == count:
            return classes
        count = len(numbers)


def _replace_types(expr: TypeExpr, canonical: dict[int, TypeSignature]) -> TypeExpr:
    """Point an expression at canonical types, collapsing union members this makes identical."""
    if isinstance(expr, TypeSignature):
        return canonical[id(expr)]
    if isinstance(expr, GenericTypeExpr):
        expr.args = [_replace_types(arg, canonical) for arg in expr.args]
    elif isinstance(expr, UnionTypeExpr):
        expr.members = _unique_exprs([_replace_types(member, canonical) for member in expr.members])
    return expr


def _safe_identifier(name: str) -> str:
    """Rewrite a name into one Python reads back exactly, keeping what it can."""
    normalized = unicodedata.normalize('NFKC', name)
    cleaned = ''.join(ch if f'a{ch}'.isidentifier() else '_' for ch in normalized)
    return cleaned if cleaned.isidentifier() else f'_{cleaned}'


def _assign_type_names(
    types: list[TypeSignature], signatures: list[FunctionSignature], owners: dict[int, str]
) -> dict[int, str]:
    """Give every type a name no other definition, tool or helper has.

    A type that can keep its name does, before any name is generated, so a generated name can't take a real one.
    """
    # Names the rendered text itself relies on, so a definition can't take them.
    rendered_text_names = {
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
    reserved = rendered_text_names | {n for s in signatures for n in (s.name, _safe_identifier(s.name))}
    name_counts: dict[str, int] = {}
    for type_sig in types:
        name_counts[type_sig.name] = name_counts.get(type_sig.name, 0) + 1

    names: dict[int, str] = {}
    used = set(reserved)
    for type_sig in types:
        if name_counts[type_sig.name] == 1 and type_sig.name not in reserved and _reads_back_exactly(type_sig.name):
            names[id(type_sig)] = type_sig.name
            used.add(type_sig.name)
    for type_sig in types:
        if id(type_sig) not in names:
            base = _safe_identifier(f'{owners[id(type_sig)]}_{type_sig.name}')
            candidate, suffix = base, 2
            while candidate in used:
                candidate, suffix = f'{base}_{suffix}', suffix + 1
            used.add(candidate)
            names[id(type_sig)] = candidate
    return names
