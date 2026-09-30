from __future__ import annotations as _annotations

import re
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

from pydantic import JsonValue

from .exceptions import UserError

JsonSchema = dict[str, Any]
_JsonSchemaNode: TypeAlias = JsonSchema | bool

_DEFS_REF_PREFIX = re.compile(r'^#/\$defs/')
_OBJECT_KEYWORDS = ('properties', 'additionalProperties', 'patternProperties')
_ARRAY_KEYWORDS = ('items', 'prefixItems')


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

    See [enum options](../tools.md#enum-options) for an example.
    """


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

        self.defs: dict[str, JsonSchema] = deepcopy(self.schema.get('$defs', {}))
        self.refs_stack: list[str] = []
        self.recursive_refs = set[str]()
        self._walked_defs: dict[str, JsonSchema] = {}

    @abstractmethod
    def transform(self, schema: JsonSchema) -> JsonSchema:
        """Make changes to the schema."""
        return schema

    def walk(self) -> JsonSchema:
        schema = deepcopy(self.schema)

        self.recursive_refs.clear()
        self._walked_defs.clear()

        # First, handle everything but $defs:
        schema.pop('$defs', None)
        handled = self._handle(schema)
        assert not isinstance(handled, bool)

        if not self.prefer_inlined_defs and self.defs:
            handled['$defs'] = {k: self._handle(v) for k, v in self.defs.items()}

        elif self.recursive_refs:
            # If we are preferring inlined defs and there are recursive refs, we _have_ to use a $defs+$ref structure
            # We try to use whatever the original root key was, but if it is already in use,
            # we modify it to avoid collisions.
            # Walking a definition can find more recursive refs, when a `$ref`'s sibling keywords first reached it.
            defs: dict[str, JsonSchema] = {}
            while pending := [key for key in self.recursive_refs if key not in defs]:
                for key in pending:
                    defs[key] = deepcopy(self._walked_def(key))
            root_ref = self.schema.get('$ref')
            root_key = None if root_ref is None else _DEFS_REF_PREFIX.sub('', root_ref)
            if root_key is None:
                root_key = self.schema.get('title', 'root')
                while root_key in defs:
                    # Modify the root key until it is not already in use
                    root_key = f'{root_key}_root'

            defs[root_key] = handled
            return {'$defs': defs, '$ref': f'#/$defs/{root_key}'}

        return handled

    def _handle(self, schema: _JsonSchemaNode) -> _JsonSchemaNode:
        if isinstance(schema, bool):
            return schema

        if self.prefer_inlined_defs and (ref := schema.get('$ref')):
            key = _DEFS_REF_PREFIX.sub('', ref)
            if key in self.refs_stack:
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
                return deepcopy(self._walked_def(key))

        # Handle the schema based on its type / structure
        type_ = schema.get('type')
        if type_ == 'object':
            schema = self._handle_object(schema)
        elif type_ == 'array':
            schema = self._handle_array(schema)
        elif self.prefer_inlined_defs and (type_ is None or isinstance(type_, list)):
            # `properties`, `items` etc. also apply when `type` is absent or a list admitting objects or arrays,
            # like `['object', 'null']`, and `walk()` drops `$defs` when inlining, so their `$ref`s must be walked
            # or they dangle.
            if (type_ is None or 'object' in type_) and self._should_walk_untyped_keywords(schema, _OBJECT_KEYWORDS):
                schema = self._handle_object(schema)
            if (type_ is None or 'array' in type_) and self._should_walk_untyped_keywords(schema, _ARRAY_KEYWORDS):
                schema = self._handle_array(schema)

        if type_ is None:
            schema = self._handle_union(schema, 'allOf')
            schema = self._handle_union(schema, 'anyOf')
            schema = self._handle_union(schema, 'oneOf')
        else:
            for union_kind in ('allOf', 'anyOf', 'oneOf'):
                if members := schema.get(union_kind):
                    schema[union_kind] = [self._handle(member) for member in members]
        # Apply the base transform
        return self.transform(schema)

    def _should_walk_untyped_keywords(self, schema: JsonSchema, keywords: tuple[str, ...]) -> bool:
        """Whether to walk `keywords`, which the node's `type` doesn't name.

        Only a `$ref` into `$defs`, which `walk()` drops, needs it. Otherwise they're left exactly as written,
        since walking reshapes a subtree (single-member unions collapse, `transform()` runs). So are they when
        a `$ref` under them, or in a definition one points at, can't be resolved, since walking raises on it.
        When they are walked, it's all of them, exactly as under a matching `type`.
        """
        pending: list[JsonValue] = [schema[keyword] for keyword in keywords if keyword in schema]
        seen: set[str] = set()
        while pending:
            node = pending.pop()
            if isinstance(node, list):
                pending.extend(node)
            elif isinstance(node, dict):
                if isinstance(ref := node.get('$ref'), str):
                    key = _DEFS_REF_PREFIX.sub('', ref)
                    # A boolean definition is valid JSON Schema, but `_walk_def` can only merge a dict.
                    if not isinstance(definition := self.defs.get(key), dict):
                        return False
                    if key not in seen:
                        seen.add(key)
                        pending.append(definition)
                pending.extend(node.values())
        return bool(seen)

    def _walked_def(self, key: str) -> JsonSchema:
        """The definition `key` refers to, walked once per transformer and cached.

        Inlining a definition means walking its whole subtree at every reference site, and the result
        is the same at each of them, so the walk is done once and callers get a `deepcopy` of it. This
        also keeps the inlined copies independent of each other: the walk transforms schemas in place,
        so handing out the same object at multiple sites would let each site corrupt the next.
        """
        if (walked := self._walked_defs.get(key)) is None:
            self._walked_defs[key] = walked = self._walk_def(key, {})
        return walked

    def _walk_def(self, key: str, siblings: JsonSchema) -> JsonSchema:
        """Walk the definition `key` refers to, with `$ref` sibling keywords merged over it."""
        def_schema = self.defs.get(key)
        if def_schema is None:  # pragma: no cover
            raise UserError(f'Could not find $ref definition for {key}')

        self.refs_stack.append(key)
        walked = self._handle({**deepcopy(def_schema), **siblings})
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

        return schema

    def _handle_array(self, schema: JsonSchema) -> JsonSchema:
        if prefix_items := schema.get('prefixItems'):
            schema['prefixItems'] = [self._handle(item) for item in prefix_items]

        items: _JsonSchemaNode | list[_JsonSchemaNode] | None = schema.get('items')
        if isinstance(items, list):
            # Draft 7 and earlier spell a tuple as an `items` list, which 2020-12 replaced with `prefixItems`.
            schema['items'] = [self._handle(item) for item in items]
        elif items:
            schema['items'] = self._handle(items)

        return schema

    def _handle_union(self, schema: JsonSchema, union_kind: Literal['allOf', 'anyOf', 'oneOf']) -> JsonSchema:
        try:
            members = schema.pop(union_kind)
        except KeyError:
            return schema

        handled = [self._handle(member) for member in members]

        if self.simplify_nullable_unions:
            handled = self._simplify_nullable_union(handled)
        if len(handled) == 1:
            # In this case, no need to retain the union
            if isinstance(handled[0], dict):
                return handled[0] | schema
            # Non-dict schema node (e.g. boolean): fall through to wrap in union key

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


class InlineDefsJsonSchemaTransformer(JsonSchemaTransformer):
    """Transforms the JSON Schema to inline $defs."""

    def __init__(self, schema: JsonSchema, *, strict: bool | None = None):
        super().__init__(schema, strict=strict, prefer_inlined_defs=True)

    def transform(self, schema: JsonSchema) -> JsonSchema:
        return schema
