from __future__ import annotations as _annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, NoReturn, TypeAlias, assert_never, cast

from pydantic import JsonValue

from ._json_schema_keywords import SUBSCHEMA_KEYWORDS, Subschema, iter_subschemas, map_subschemas
from ._utils import capture_class_source, contains_ref
from .exceptions import UserError

JsonSchema = dict[str, Any]
_JsonSchemaNode: TypeAlias = JsonSchema | bool

MAX_INLINED_SCHEMA_VALUES = 250_000
"""The most JSON values inlining a schema's `$ref`s may copy into it.

Each object, array, string, number, boolean and null in a copied definition counts as one value, as the definition
is written: what `transform()` adds to a copy is not counted.
"""


class InlinedSchemaTooLargeError(UserError):
    """A schema whose `$ref`s expand past `MAX_INLINED_SCHEMA_VALUES` once inlined, with `subject` naming it."""

    subject: str
    """What the schema belongs to, as the message names it."""
    remedy: str
    """What the message suggests doing about it."""

    def __init__(self, subject: str, remedy: str = 'Refer to large definitions from fewer places.'):
        self.subject = subject
        self.remedy = remedy
        super().__init__(
            f'{subject} is too large: its `$ref`s expand to more than {MAX_INLINED_SCHEMA_VALUES:,} JSON values. {remedy}'
        )

    def __reduce__(self) -> tuple[type[InlinedSchemaTooLargeError], tuple[str, str]]:
        return self.__class__, (self.subject, self.remedy)


@dataclass
class InlinedValues:
    """A count of the JSON values that inlining `$ref`s copies into a schema, which stops at `MAX_INLINED_SCHEMA_VALUES`.

    Every copy of a definition counts in full, so the count is the number of values inlining copies, and passing the
    limit stops the inlining before the rest is built. A string counts as one value, whatever its length.
    """

    subject: str
    """What the schema belongs to, as the error names it."""
    count: int = 0

    def add_copy(self, definition: JsonValue) -> None:
        """Count one copy of `definition`: every JSON value in it."""
        values = 0
        stack = [definition]
        while stack:
            value = stack.pop()
            values += 1
            if self.count + values > MAX_INLINED_SCHEMA_VALUES:
                # Stop as soon as the copy passes the limit: a definition can share one object at many places.
                raise InlinedSchemaTooLargeError(self.subject)
            if isinstance(value, dict):
                stack.extend(value.values())
            elif isinstance(value, list):
                stack.extend(value)
        self.add(values)

    def add(self, values: int) -> None:
        """Count `values` more JSON values, as another copy of a definition counted before adds."""
        self.count += values
        if self.count > MAX_INLINED_SCHEMA_VALUES:
            raise InlinedSchemaTooLargeError(self.subject)


def _is_type_value(value: Any) -> bool:
    return isinstance(value, str) or (
        isinstance(value, list | tuple) and all(isinstance(t, str) for t in cast(Sequence[Any], value))
    )


def _is_integer(value: Any) -> bool:
    # Since draft 6, a number with a zero fractional part is an integer, as JSON text like `2.0` parses to a float.
    return (isinstance(value, int) and not isinstance(value, bool)) or (isinstance(value, float) and value.is_integer())


def _is_sequence(value: Any) -> bool:
    # A tuple serializes as a JSON array, and every reader treats these as sequences.
    return isinstance(value, list | tuple)


# The keywords the walker and the provider `transform()`s read.
# Tool schemas that come from outside Pydantic AI (`Tool.from_schema`, `ExternalToolset`, MCP `inputSchema`,
# AG-UI frontend tools) are plain dicts, so a node that is not a schema can reach the walker. `_check_schema_shape`
# checks these keywords and nothing more: keywords not listed here are passed through to the provider untouched,
# so provider extensions are unaffected. If a transformer starts reading a new keyword, add it here.
_CHECKED_SUBSCHEMA_KEYWORDS = (
    'additionalProperties',
    'items',
    'propertyNames',
    'contains',
    'properties',
    'patternProperties',
    '$defs',
    'prefixItems',
    'allOf',
    'anyOf',
    'oneOf',
)
"""The `SUBSCHEMA_KEYWORDS` the transformers read, each checked for the shape the table gives it."""
_VALUE_KEYWORDS: dict[str, tuple[Callable[[Any], bool], str]] = {
    # keyword: (is the value well-formed, what a well-formed value is)
    '$ref': (lambda value: isinstance(value, str), 'must be a string'),
    'type': (_is_type_value, 'must be a string or a list of strings'),
    'required': (_is_sequence, 'must be a list'),
    'enum': (_is_sequence, 'must be a list'),
    'minItems': (_is_integer, 'must be an integer'),
}
"""Keywords whose value the transformers interpret."""


def _invalid_schema(path: str, what: str, value: Any) -> NoReturn:
    raise UserError(f'Invalid JSON Schema at `{path or "<root>"}`: {what}, got {value!r}')


def _check_schema_shape(node: Any, path: str = '') -> None:
    """Raise `UserError` naming the offending node if `node` is not shaped like the JSON Schema the transformers read."""
    # Draft 2020-12, section 4.3: "A JSON Schema MUST be an object or a boolean."
    if isinstance(node, bool):
        return
    if not isinstance(node, dict):
        _invalid_schema(path, 'a schema must be an object or a boolean', node)
    schema = cast(dict[str, Any], node)
    prefix = f'{path}.' if path else ''

    for key in _CHECKED_SUBSCHEMA_KEYWORDS:
        # `None` is skipped like an absent key, as the walker and OpenAI's `transform()` do.
        if (value := schema.get(key)) is None:
            continue
        match shape := SUBSCHEMA_KEYWORDS[key]:
            case 'schema' | 'schema_or_list':
                _check_subschema(value, f'{prefix}{key}', allow_list=shape == 'schema_or_list')
            case 'schema_map':
                _check_schema_container(key, dict, value, f'{prefix}{key}')
            case 'schema_list':
                _check_schema_container(key, list, value, f'{prefix}{key}')
            case _:
                assert_never(shape)
    for key, (is_well_formed, what) in _VALUE_KEYWORDS.items():
        if (value := schema.get(key)) is not None and not is_well_formed(value):
            _invalid_schema(f'{prefix}{key}', f'`{key}` {what}', value)


def _check_subschema(value: Any, path: str, *, allow_list: bool) -> None:
    """Check a keyword whose value is a schema; `items` may also be a list of schemas (draft 4 to 2019-09 tuples)."""
    if allow_list and isinstance(value, list):
        for index, member in enumerate(cast(list[Any], value)):
            _check_schema_shape(member, f'{path}.{index}')
    else:
        _check_schema_shape(value, path)


def _check_schema_container(key: str, container: type[dict[str, Any]] | type[list[Any]], value: Any, path: str) -> None:
    """Check a keyword whose value is an object or a list of schemas."""
    # A tuple of schemas worked before this check existed: the walker iterates it like a list.
    if not (isinstance(value, container) or (container is list and isinstance(value, tuple))):
        kind = 'an object' if container is dict else 'a list'
        _invalid_schema(path, f'`{key}` must be {kind} of schemas', value)
    members: Iterable[tuple[str | int, Any]] = (
        cast(dict[str, Any], value).items() if isinstance(value, dict) else enumerate(cast(Sequence[Any], value))
    )
    for name, member in members:
        _check_schema_shape(member, f'{path}.{name}')


def _json_pointer_path(ref: str) -> str:
    """The dotted path an error names for a local JSON pointer, with its `~1` and `~0` escapes decoded."""
    return '.'.join(token.replace('~1', '/').replace('~0', '~') for token in ref.split('/')[1:])


def resolve_json_pointer(schema: JsonSchema, ref: str) -> JsonSchema | None:
    """The schema object a local JSON-pointer `$ref` (RFC 6901) like `#/properties/from` points at, if any.

    Not every producer collects shared subschemas into `$defs`: `zod-to-json-schema`, which the MCP
    TypeScript SDK uses, points a reused subschema at its first occurrence instead.
    """
    if ref != '#' and not ref.startswith('#/'):
        return None
    node: JsonValue = schema
    for token in ref.split('/')[1:]:
        token = token.replace('~1', '/').replace('~0', '~')
        if isinstance(node, dict):
            node = node.get(token)
        elif isinstance(node, list) and token in map(str, range(len(node))):
            # An RFC 6901 array index is written in canonical decimal: ASCII digits, no leading zero.
            node = node[int(token)]
        else:
            node = None
    return node if isinstance(node, dict) else None


def definition_name(ref: str) -> str:
    """The `$defs` name a `$ref` like `#/$defs/Model` points at, with its `~1` and `~0` escapes decoded."""
    return ref.removeprefix('#/$defs/').replace('~1', '/').replace('~0', '~')


def definition_refs(schema: Subschema) -> Iterator[str]:
    """Each `$ref` into `$defs` in `schema` or a subschema under it, skipping a `$ref` in data like `examples`."""
    pending: list[Subschema] = [schema]
    while pending:
        node = pending.pop()
        if isinstance(node, dict):
            if isinstance(ref := node.get('$ref'), str) and ref.startswith('#/$defs/'):
                yield ref
            pending.extend(iter_subschemas(node))


def merge_definition(definition: _JsonSchemaNode, siblings: JsonSchema) -> JsonSchema:
    """A copy of `definition` with the other keywords of a `$ref` to it merged over it, as inlining the `$ref` does."""
    if isinstance(definition, bool):
        # A false definition must reject even when a sibling supplies its own `not`.
        schema = deepcopy(siblings)
        if not definition:
            schema['not'] = {}
        return schema
    return {**deepcopy(definition), **siblings}


@dataclass
class UnwalkedRefWriter:
    """Replaces each `$defs` `$ref` under a keyword the walk doesn't enter by the definition it points at, as written.

    `JsonSchemaTransformer` leaves `not`, `if`/`then`/`else`, `dependentSchemas` and every other keyword outside
    `WALKED_KEYWORDS` as written, so inlining drops `$defs` without reaching a `$ref` there. The `$ref`'s other keywords
    are merged over the definition, as inlining merges them, and `transform()` doesn't run on it, as it doesn't on
    anything else written there. A `$ref` that leads back into a definition it is already inside stays, as a recursive
    one does where the walk inlines, and `kept` names the definition.
    """

    WALKED_KEYWORDS: ClassVar[frozenset[str]] = frozenset(
        (
            'properties',
            'additionalProperties',
            'patternProperties',
            'propertyNames',
            'prefixItems',
            'items',
            'contains',
            'allOf',
            'anyOf',
            'oneOf',
        )
    )
    """The `SUBSCHEMA_KEYWORDS` the walk enters, on a schema of the type they apply to; the root `$defs` aside."""

    schema: JsonSchema
    """The schema as given, which a JSON pointer `$ref` into `$defs` resolves against."""
    defs: Mapping[str, _JsonSchemaNode]
    inlined: InlinedValues
    """The count each written-out definition adds to, like a definition the walk inlines."""
    kept: list[str] = field(default_factory=list[str])
    """The definitions a `$ref` that stays points at, in the order they're found."""
    wrote_out: bool = False

    def write_out(self, node: JsonSchema, *, unwalked: bool, stack: tuple[str, ...] = ()) -> JsonSchema:
        """`node` with each `$defs` `$ref` under a keyword the walk doesn't enter written out.

        `unwalked` says whether `node` itself is under one, and `stack` holds the definitions it is inside.
        """
        if unwalked and (written := self._write_out_ref(node, stack)) is not None:
            return written
        return map_subschemas(
            node,
            lambda keyword, member: (
                member
                if isinstance(member, bool)
                else self.write_out(member, unwalked=unwalked or keyword not in self.WALKED_KEYWORDS, stack=stack)
            ),
        )

    def _write_out_ref(self, node: JsonSchema, stack: tuple[str, ...]) -> JsonSchema | None:
        """`node` with its `$ref` replaced by what it points at in `$defs`, or `None` when the `$ref` stays."""
        ref = node.get('$ref')
        if not isinstance(ref, str) or not ref.startswith('#/$defs/'):
            return None
        name = definition_name(ref)
        if name in self.defs:
            key, target = name, self.defs[name]
        elif (pointed := resolve_json_pointer(self.schema, ref)) is not None:
            key, target = ref, pointed
        else:
            return None
        if key in stack:
            # A JSON pointer that closes a cycle stays pointing at nothing: only a whole definition can be kept.
            if key == name and name not in self.kept:
                self.kept.append(name)
            return None
        merged = merge_definition(target, {k: v for k, v in node.items() if k != '$ref'})
        self.inlined.add_copy(merged)
        self.wrote_out = True
        return self.write_out(merged, unwalked=True, stack=(*stack, key))


class UseEnumMemberDocstrings:
    """Mix into an `Enum` to describe each of its members by the docstring written under it.

    This is the enum counterpart to `model_config = ConfigDict(use_attribute_docstrings=True)` on a Pydantic
    model, which is how a model's fields get their docstrings as descriptions. With it, the enum is described
    to the model as `anyOf` of `const`s carrying those docstrings as descriptions, rather than as a bare list
    of values, so the model can tell similar options apart. Without it the docstrings are ignored and the
    schema is unchanged.

    The two are expressed differently because an `Enum` has no `model_config` to carry a flag, and cannot carry
    a plain class attribute either — annotated or not, any assigned value becomes a member. A base class is the
    only marker left, so mix it in ahead of `str`, `int` or `Enum`:
    `class Urgency(UseEnumMemberDocstrings, str, Enum)`. It changes nothing else about the enum: its members,
    their values, and their `str`/`int` behaviour are exactly what they would be without it.

    The docstrings are read from the enum's source. In a notebook or IPython session, where `inspect.getsource`
    cannot find a class, they are read from the cell the enum was defined in. An enum built by calling an opted-in
    enum class (`Priority('Urgency', {...})`) has no source, so it is described by its values alone, with an
    [`EnumDocstringsUnavailableWarning`][pydantic_ai.exceptions.EnumDocstringsUnavailableWarning].

    See [enum options](../tools.md#enum-options) for an example.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # The cell a notebook class was defined in can only be found while the `class` statement is running.
        capture_class_source(cls, inspect.currentframe())


@dataclass(init=False)
class JsonSchemaTransformer(ABC):
    """Walks a JSON schema, applying transformations to it at each level.

    The transformer is called during a model's prepare_request() step to build the JSON schema
    before it is sent to the model provider.

    Note: We may eventually want to rework tools to build the JSON schema from the type directly, using a subclass of
    pydantic.json_schema.GenerateJsonSchema, rather than making use of this machinery.
    """

    def __init__(
        self,
        schema: JsonSchema,
        *,
        strict: bool | None = None,
        prefer_inlined_defs: bool = False,
        simplify_nullable_unions: bool = False,
    ):
        _check_schema_shape(schema)
        self.schema = schema

        self.strict = strict
        """The `strict` parameter forces the conversion of the original JSON schema (`self.schema`) of a `ToolDefinition` or `OutputObjectDefinition` to a format supported by the model provider.

        The "strict mode" offered by model providers ensures that the model's output adheres closely to the defined schema. However, not all model providers offer it, and their support for various schema features may differ. For example, a model provider's required schema may not support certain validation constraints like `minLength` or `pattern`.
        """
        self.is_strict_compatible = True
        """Whether the schema is compatible with strict mode.

        This value is used to set `ToolDefinition.strict` or `OutputObjectDefinition.strict` when their values are `None`.
        """
        self.prefer_inlined_defs = prefer_inlined_defs
        self.simplify_nullable_unions = simplify_nullable_unions

        self.defs: dict[str, _JsonSchemaNode] = deepcopy(self.schema.get('$defs', {}))
        self.refs_stack: list[str] = []
        self.recursive_refs = set[str]()
        self._walked_defs: dict[str, tuple[JsonSchema, int]] = {}
        self._inlined = InlinedValues('JSON schema')
        self._unwalked_ref_writer: UnwalkedRefWriter | None = None
        """Set during `walk()`'s second pass, which writes out each definition the walk reaches as it reaches it."""

    @abstractmethod
    def transform(self, schema: JsonSchema) -> JsonSchema:
        """Make changes to the schema."""
        return schema

    def walk(self) -> JsonSchema:
        self._inlined.count = 0
        walked = self._walk_schema()
        if self.prefer_inlined_defs and any(
            resolve_json_pointer(walked, ref) is None for ref in definition_refs(walked)
        ):
            return self._walk_with_unwalked_refs_written_out(walked)
        return walked

    def _walk_schema(self) -> JsonSchema:
        """One pass of `walk()` over `self.schema`."""
        schema = deepcopy(self.schema)

        self.recursive_refs.clear()
        self._walked_defs.clear()

        # First, handle everything but $defs:
        schema.pop('$defs', None)
        handled = self._handle(schema)
        assert not isinstance(handled, bool)

        if not self.prefer_inlined_defs and self.defs:
            handled['$defs'] = {k: self._handle(v) for k, v in self.defs.items()}
            return handled

        defs = {key: self._copy_walked_def(key) for key in self.recursive_refs}
        if (writer := self._unwalked_ref_writer) is not None and writer.kept:
            # Each kept definition, written out the same way, keeping the `$ref`s that lead back into it.
            kept: dict[str, JsonSchema] = {}
            # `write_out` adds each definition it keeps to `writer.kept`, and the loop reaches those too.
            for name in writer.kept:
                definition = merge_definition(writer.defs[name], {})
                writer.inlined.add_copy(definition)
                kept[name] = writer.write_out(definition, unwalked=True, stack=(name,))
            self._add_referenced_definitions(defs, handled, kept)
        if not defs:
            return handled

        # If we are preferring inlined defs and there are recursive refs, we _have_ to use a $defs+$ref structure
        root_ref, root_key = self._root_ref_and_key(defs)
        defs[root_key] = handled
        return {'$defs': defs, '$ref': root_ref or f'#/$defs/{root_key}'}

    def _root_ref_and_key(self, defs: Mapping[str, JsonSchema]) -> tuple[str | None, str]:
        """The root `$ref` the output keeps, if any, and the `$defs` key the walked root goes under.

        We try to use whatever the original root key was, but if it is already in use, we modify it to avoid
        collisions.
        """
        root_ref = self.schema.get('$ref')
        root_key = None if root_ref is None else definition_name(root_ref)
        if root_key is not None and root_key in defs and root_key not in self.recursive_refs:
            # The root `$ref`s a definition kept as written, which a `$ref` under a keyword the walk doesn't enter
            # points at, so the walked root goes under a name of its own.
            root_ref = None
        if root_key is None or root_ref is None:
            root_key = self.schema.get('title', 'root') if root_key is None else root_key
            while root_key in defs:
                # Modify the root key until it is not already in use
                root_key = f'{root_key}_root'
        return root_ref, root_key

    def _walk_with_unwalked_refs_written_out(self, walked: JsonSchema) -> JsonSchema:
        """`walk()`'s second pass, for a schema whose first pass, `walked`, leaves a `$ref` into `$defs` dangling.

        Each `$defs` `$ref` under a keyword the walk doesn't enter is written out in place by `UnwalkedRefWriter`: in
        the schema before the walk, and in each definition as the walk reaches it. A recursive one stays, and the
        definition it points at goes under `$defs`. Writing out counts toward `MAX_INLINED_SCHEMA_VALUES` like
        inlining does. A schema whose writing out passes that limit, or raises `RecursionError` on a long chain of
        definitions, keeps `walked`, which the first pass produced without raising. Any other raise propagates.
        """
        schema = self.schema
        recursive_refs, is_strict_compatible = set(self.recursive_refs), self.is_strict_compatible
        self._unwalked_ref_writer = writer = UnwalkedRefWriter(schema, self.defs, self._inlined)
        self._inlined.count = 0
        try:
            self.schema = writer.write_out({k: v for k, v in schema.items() if k != '$defs'}, unwalked=False)
            rewalked = self._walk_schema()
        # The value limit's `UserError`, or a `RecursionError` on a long chain of definitions the first pass never
        # entered: the second pass must not fail a schema the first one walked.
        except (UserError, RecursionError):
            self.recursive_refs, self.is_strict_compatible = recursive_refs, is_strict_compatible
            self.refs_stack.clear()
            return walked
        finally:
            self.schema, self._unwalked_ref_writer = schema, None
        return rewalked if writer.wrote_out or writer.kept else walked

    def _add_referenced_definitions(
        self, defs: dict[str, JsonSchema], handled: JsonSchema, kept: Mapping[str, JsonSchema]
    ) -> None:
        """Add to `defs` each `kept` definition the walked schema `handled` or another added definition refers to."""
        pending: list[JsonSchema] = [handled, *defs.values()]
        while pending:
            for ref in definition_refs(pending.pop()):
                if (name := definition_name(ref)) in kept and name not in defs:
                    defs[name] = kept[name]
                    pending.append(kept[name])

    def _handle(self, schema: _JsonSchemaNode) -> _JsonSchemaNode:
        if isinstance(schema, bool):
            return schema

        if (
            self.prefer_inlined_defs
            and (ref := schema.get('$ref'))
            and (inlined := self._inlined_ref(schema, ref)) is not None
        ):
            return inlined

        # Handle the schema based on its type / structure. Object and array keywords are only walked when `type` is
        # exactly `'object'` or `'array'` respectively, not when it's absent or a list like `['object', 'null']`:
        # walking them there would reshape subtrees that otherwise pass through unchanged, collapsing their
        # single-member unions and running `transform()` on them, and an inlining transformer would raise `UserError`
        # on a `$ref` it can't resolve.
        type_ = schema.get('type')
        match type_:
            case 'object':
                schema = self._handle_object(schema)
            case 'array':
                schema = self._handle_array(schema)
            case None:
                schema = self._handle_union(schema, 'allOf')
                schema = self._handle_union(schema, 'anyOf')
                schema = self._handle_union(schema, 'oneOf')
            case _:
                pass

        if type_ is not None:
            for union_kind in ('allOf', 'anyOf', 'oneOf'):
                if members := schema.get(union_kind):
                    schema[union_kind] = [self._handle(member) for member in members]
        # Apply the base transform
        return self.transform(schema)

    def _inlined_ref(self, schema: JsonSchema, ref: str) -> JsonSchema | None:
        """The walked definition `ref` points at, with the other keywords of `schema` merged over it.

        `None` when the `$ref` stays, as a recursive one does.
        """
        # Decode only a definition name; other pointers are decoded by `resolve_json_pointer()`.
        key = definition_name(ref) if ref.startswith('#/$defs/') else ref
        if key in self.refs_stack:
            if key not in self.defs:
                # Only a `$defs` name can stay a `$ref` once the schema around it is inlined.
                raise UserError(f'Recursive JSON pointer `$ref` {key!r} is not supported when inlining definitions')
            # A recursive ref can't be unpacked; `walk()` emits the definition and the `$ref` stays put.
            self.recursive_refs.add(key)
        elif key not in self.recursive_refs:
            # Keywords sitting alongside the `$ref` (e.g. a field-level `description`
            # or `default`) are part of the field's own schema and must survive
            # inlining, so merge them over the referenced definition.
            if siblings := {k: v for k, v in schema.items() if k != '$ref'}:
                # `transform()` sees the merged schema, so the result is specific to this reference
                # site and can't come from (or go into) the shared walked-definition cache.
                return self._walk_def(key, siblings)
            return self._copy_walked_def(key)
        return None

    def _copy_walked_def(self, key: str) -> JsonSchema:
        """A copy of the definition `key` refers to, walked once per transformer and cached.

        Inlining a definition means walking its whole subtree at every reference site, and the result
        is the same at each of them, so the walk is done once and callers get a `deepcopy` of it. This
        also keeps the inlined copies independent of each other: the walk transforms schemas in place,
        so handing out the same object at multiple sites would let each site corrupt the next.

        The cache keeps how many JSON values the walk counted, so each later copy counts as that many again.
        """
        if (cached := self._walked_defs.get(key)) is None:
            start = self._inlined.count
            walked = self._walk_def(key, {})
            self._walked_defs[key] = (walked, self._inlined.count - start)
        else:
            walked, values = cached
            self._inlined.add(values)
        return deepcopy(walked)

    def _walk_def(self, key: str, siblings: JsonSchema) -> JsonSchema:
        """Walk the definition `key` refers to, with `$ref` sibling keywords merged over it.

        `key` is a `$defs` name, or else a local JSON pointer (RFC 6901) into the original schema. The merged
        definition counts toward `MAX_INLINED_SCHEMA_VALUES` before it is walked, and so does each definition the
        walk copies into it.
        """
        def_schema = self.defs.get(key)
        written_out = None
        if def_schema is None:
            def_schema = resolve_json_pointer(self.schema, key)
            # A pointer can reach nodes the up-front check doesn't visit, like draft-07 `definitions`.
            if def_schema is not None:
                _check_schema_shape(def_schema, _json_pointer_path(key))
        elif self._unwalked_ref_writer is not None and isinstance(def_schema, dict):
            written_out = self._unwalked_ref_writer.write_out(def_schema, unwalked=False, stack=(key,))
        if def_schema is None:
            raise UserError(f'Could not find $ref definition for {key}')

        self.refs_stack.append(key)
        schema = merge_definition(def_schema, siblings)
        # Counted as written: writing it out already counted each copy that added.
        self._inlined.add_copy(schema)
        if written_out is not None:
            schema = merge_definition(written_out, siblings)
        walked = self._handle(schema)
        self.refs_stack.pop()

        assert not isinstance(walked, bool)
        return walked

    def _handle_object(self, schema: JsonSchema) -> JsonSchema:
        if properties := schema.get('properties'):
            handled_properties = {}
            for key, value in properties.items():
                handled_properties[key] = self._handle(value)
            schema['properties'] = handled_properties

        if (additional_properties := schema.get('additionalProperties')) is not None:
            if isinstance(additional_properties, bool):
                schema['additionalProperties'] = additional_properties
            else:
                schema['additionalProperties'] = self._handle(additional_properties)

        if (pattern_properties := schema.get('patternProperties')) is not None:
            handled_pattern_properties = {}
            for key, value in pattern_properties.items():
                handled_pattern_properties[key] = self._handle(value)
            schema['patternProperties'] = handled_pattern_properties

        if (property_names := schema.get('propertyNames')) is not None and self._walks_keyword_subschema(
            property_names
        ):
            schema['propertyNames'] = self._handle(property_names)

        return schema

    def _handle_array(self, schema: JsonSchema) -> JsonSchema:
        if prefix_items := schema.get('prefixItems'):
            schema['prefixItems'] = [self._handle(item) for item in prefix_items]

        items: _JsonSchemaNode | list[_JsonSchemaNode] | None = schema.get('items')
        if isinstance(items, list):
            # Drafts before 2020-12 spell a tuple as an `items` list; 2020-12 replaced it with `prefixItems`.
            schema['items'] = [self._handle(item) for item in items]
        elif items:
            schema['items'] = self._handle(items)

        if (contains := schema.get('contains')) is not None and self._walks_keyword_subschema(contains):
            schema['contains'] = self._handle(contains)

        return schema

    def _walks_keyword_subschema(self, subschema: _JsonSchemaNode) -> bool:
        """Whether to walk a `propertyNames` or `contains` subschema.

        Inlining drops `$defs`, so a `$ref` left under either would point at nothing. A subschema without one
        is left as written, so `transform()` doesn't rewrite a schema that was already valid.
        """
        return self.prefer_inlined_defs and contains_ref(subschema)

    def _handle_union(self, schema: JsonSchema, union_kind: Literal['allOf', 'anyOf', 'oneOf']) -> JsonSchema:
        try:
            members = schema.pop(union_kind)
        except KeyError:
            return schema

        handled = [self._handle(member) for member in members]

        if self.simplify_nullable_unions:
            handled = self._simplify_nullable_union(handled)
        if len(handled) == 1:
            # Keep a rejecting member wrapped so a sibling `not` cannot overwrite its constraint.
            if isinstance(handled[0], dict) and handled[0].get('not') != {}:
                return handled[0] | schema

        # If we have keys besides the union kind (such as title or discriminator), keep them without modifications
        schema = schema.copy()
        schema[union_kind] = handled
        return schema

    @staticmethod
    def _simplify_nullable_union(cases: list[_JsonSchemaNode]) -> list[_JsonSchemaNode]:
        if len(cases) == 2 and {'type': 'null'} in cases:
            # Find the non-null schema
            non_null_schema = next(
                (item for item in cases if item != {'type': 'null'}),
                None,
            )
            if isinstance(non_null_schema, dict):
                # Create a new schema based on the non-null part, mark as nullable
                new_schema = deepcopy(non_null_schema)
                new_schema['nullable'] = True
                return [new_schema]
            if non_null_schema is not None:
                return cases
            else:  # pragma: no cover
                # they are both null, so just return one of them
                return [cases[0]]

        return cases


class JsonSchemaTransformerWithOutputOverride(JsonSchemaTransformer):
    """A tool-schema transformer that selects a different transformer for output objects."""

    output_object_transformer: ClassVar[type[JsonSchemaTransformer]]


class InlineDefsJsonSchemaTransformer(JsonSchemaTransformer):
    """Transforms the JSON Schema to inline `$defs`, and `$ref`s that point elsewhere in the schema like `#/properties/from`.

    Object keywords (`properties`, `additionalProperties`, `patternProperties`, `propertyNames`) are only walked
    when `type` is `'object'`, and array keywords (`items`, `prefixItems`, `contains`) only when it is `'array'`.
    On a schema with no `type`, or with a `type` list such as `['object', 'null']`, they are left as written,
    so a `$ref` inside them is not inlined and can point at a definition the output no longer contains.
    If the schema is only meant to accept objects (or arrays), set `type` to `'object'` (or `'array'`) to have
    it inlined; for a nullable one, put it in an `anyOf` with `{'type': 'null'}` instead of using a `type` list.

    Keywords that apply to every type, such as `not`, `if`/`then`/`else`, `dependentSchemas` and
    `unevaluatedProperties`, aren't walked, so `transform()` doesn't run under them. A `$ref` there is replaced by its
    definition as written, and one that leads back into its own definition stays, with the definition kept under
    `$defs`.
    """

    def __init__(self, schema: JsonSchema, *, strict: bool | None = None):
        super().__init__(schema, strict=strict, prefer_inlined_defs=True)

    def _walks_keyword_subschema(self, subschema: _JsonSchemaNode) -> bool:
        # Unlike the base class, walk one without a `$ref` too: `transform()` is the identity here, so the walk
        # only collapses single-member unions.
        return True

    def transform(self, schema: JsonSchema) -> JsonSchema:
        return schema


DEFINITION_KEYS = ('$defs', 'definitions')
"""The keys a schema keeps its definitions under: `$defs`, and draft-07 `definitions`."""


def split_definitions(schema: JsonSchema) -> tuple[JsonSchema, list[object]]:
    """A schema without its definitions, and the definitions under `DEFINITION_KEYS`."""
    root = {key: value for key, value in schema.items() if key not in DEFINITION_KEYS}
    definitions: list[object] = []
    for key in DEFINITION_KEYS:
        group = schema.get(key)
        if isinstance(group, dict):
            definitions.extend(cast(dict[str, object], group).values())
    return root, definitions


def inline_objects(schema: object, depth: int = 1) -> Iterator[tuple[JsonSchema, int]]:
    """Each object with `properties` reached from `schema` through properties, array items and `anyOf`, and its depth.

    The depth is the number of such objects along the path, the object included. `$ref`s aren't followed and
    definitions aren't entered: providers that limit strict schemas were observed to count this way.
    """
    if not isinstance(schema, dict):
        return
    node = cast(JsonSchema, schema)
    properties = node.get('properties')
    children: list[object] = [node.get('items'), *node.get('anyOf', [])]
    child_depth = depth
    if isinstance(properties, dict):
        yield node, depth
        children.extend(cast(dict[str, object], properties).values())
        child_depth = depth + 1
    for child in children:
        yield from inline_objects(child, child_depth)
