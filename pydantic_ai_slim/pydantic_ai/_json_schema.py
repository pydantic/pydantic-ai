from __future__ import annotations as _annotations

import re
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, cast

from .exceptions import UserError

JsonSchema = dict[str, Any]

__all__ = ['JsonSchemaTransformer', 'InlineDefsJsonSchemaTransformer', 'flatten_allof']


@dataclass(init=False)
class JsonSchemaTransformer(ABC):
    """Walks a JSON schema, applying transformations to it at each level.

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
        flatten_allof: bool = False,
    ):
        self.schema = schema

        self.strict = strict
        # Can be set to False by subclasses to set `strict` on `ToolDefinition`
        # when not set explicitly by the user.
        self.is_strict_compatible = True

        self.prefer_inlined_defs = prefer_inlined_defs
        self.simplify_nullable_unions = simplify_nullable_unions
        self.flatten_allof = flatten_allof

        self.defs: dict[str, JsonSchema] = self.schema.get('$defs', {})
        self.refs_stack: list[str] = []
        self.recursive_refs = set[str]()

    @abstractmethod
    def transform(self, schema: JsonSchema) -> JsonSchema:
        """Make changes to the schema."""
        return schema

    def walk(self) -> JsonSchema:
        schema = deepcopy(self.schema)

        # First, handle everything but $defs:
        schema.pop('$defs', None)
        handled = self._handle(schema)

        if not self.prefer_inlined_defs and self.defs:
            handled['$defs'] = {k: self._handle(v) for k, v in self.defs.items()}

        elif self.recursive_refs:
            # If we are preferring inlined defs and there are recursive refs, we _have_ to use a $defs+$ref structure
            # We try to use whatever the original root key was, but if it is already in use,
            # we modify it to avoid collisions.
            defs = {key: self.defs[key] for key in self.recursive_refs}
            root_ref = self.schema.get('$ref')
            root_key = None if root_ref is None else re.sub(r'^#/\$defs/', '', root_ref)
            if root_key is None:  # pragma: no cover
                root_key = self.schema.get('title', 'root')
                while root_key in defs:
                    # Modify the root key until it is not already in use
                    root_key = f'{root_key}_root'

            defs[root_key] = handled
            return {'$defs': defs, '$ref': f'#/$defs/{root_key}'}

        return handled

    def _handle(self, schema: JsonSchema) -> JsonSchema:
        # Flatten allOf if requested, before processing the schema
        if self.flatten_allof:
            schema = flatten_allof(schema)

        nested_refs = 0
        if self.prefer_inlined_defs:
            while ref := schema.get('$ref'):
                key = re.sub(r'^#/\$defs/', '', ref)
                if key in self.recursive_refs:
                    break
                if key in self.refs_stack:
                    self.recursive_refs.add(key)
                    break  # recursive ref can't be unpacked
                self.refs_stack.append(key)
                nested_refs += 1

                def_schema = self.defs.get(key)
                if def_schema is None:  # pragma: no cover
                    raise UserError(f'Could not find $ref definition for {key}')
                schema = def_schema

        # Handle the schema based on its type / structure
        type_ = schema.get('type')
        if type_ == 'object':
            schema = self._handle_object(schema)
        elif type_ == 'array':
            schema = self._handle_array(schema)
        elif type_ is None:
            schema = self._handle_union(schema, 'anyOf')
            schema = self._handle_union(schema, 'oneOf')

        # Apply the base transform
        schema = self.transform(schema)

        if nested_refs > 0:
            self.refs_stack = self.refs_stack[:-nested_refs]

        return schema

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

        if items := schema.get('items'):
            schema['items'] = self._handle(items)

        return schema

    def _handle_union(self, schema: JsonSchema, union_kind: Literal['anyOf', 'oneOf']) -> JsonSchema:
        try:
            members = schema.pop(union_kind)
        except KeyError:
            return schema

        handled = [self._handle(member) for member in members]

        # convert nullable unions to nullable types
        if self.simplify_nullable_unions:
            handled = self._simplify_nullable_union(handled)

        if len(handled) == 1:
            # In this case, no need to retain the union
            return handled[0] | schema

        # If we have keys besides the union kind (such as title or discriminator), keep them without modifications
        schema = schema.copy()
        schema[union_kind] = handled
        return schema

    @staticmethod
    def _simplify_nullable_union(cases: list[JsonSchema]) -> list[JsonSchema]:
        # TODO: Should we move this to relevant subclasses? Or is it worth keeping here to make reuse easier?
        if len(cases) == 2 and {'type': 'null'} in cases:
            # Find the non-null schema
            non_null_schema = next(
                (item for item in cases if item != {'type': 'null'}),
                None,
            )
            if non_null_schema:
                # Create a new schema based on the non-null part, mark as nullable
                new_schema = deepcopy(non_null_schema)
                new_schema['nullable'] = True
                return [new_schema]
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


def _allof_is_object_like(member: JsonSchema) -> bool:
    member_type = member.get('type')
    if member_type is None:
        keys = ('properties', 'additionalProperties', 'patternProperties')
        return bool(any(k in member for k in keys))
    return member_type == 'object'


def _merge_additional_properties_values(values: list[Any]) -> bool | JsonSchema:
    if any(isinstance(v, dict) for v in values):
        return True
    return False if values and all(v is False for v in values) else True


def _flatten_current_level(s: JsonSchema) -> JsonSchema:
    raw_members = s.get('allOf')
    if not isinstance(raw_members, list) or not raw_members:
        return s

    members = cast(list[JsonSchema], raw_members)
    for raw in members:
        if not isinstance(raw, dict):
            return s
    if not all(_allof_is_object_like(member) for member in members):
        return s

    processed_members = [_recurse_flatten_allof(member) for member in members]
    merged: JsonSchema = {k: v for k, v in s.items() if k != 'allOf'}
    merged['type'] = 'object'

    properties: dict[str, JsonSchema] = {}
    if isinstance(merged.get('properties'), dict):
        properties.update(merged['properties'])

    required: set[str] = set(merged.get('required', []) or [])
    pattern_properties: dict[str, JsonSchema] = dict(merged.get('patternProperties', {}) or {})
    additional_values: list[Any] = []

    for m in processed_members:
        if isinstance(m.get('properties'), dict):
            properties.update(m['properties'])
        if isinstance(m.get('required'), list):
            required.update(m['required'])
        if isinstance(m.get('patternProperties'), dict):
            pattern_properties.update(m['patternProperties'])
        if 'additionalProperties' in m:
            additional_values.append(m['additionalProperties'])

    if properties:
        merged['properties'] = {k: _recurse_flatten_allof(v) for k, v in properties.items()}
    if required:
        merged['required'] = sorted(required)
    if pattern_properties:
        merged['patternProperties'] = {k: _recurse_flatten_allof(v) for k, v in pattern_properties.items()}

    if additional_values:
        merged['additionalProperties'] = _merge_additional_properties_values(additional_values)

    return merged


def _recurse_children(s: JsonSchema) -> JsonSchema:
    t = s.get('type')
    if t == 'object':
        if isinstance(s.get('properties'), dict):
            s['properties'] = {
                k: _recurse_flatten_allof(cast(JsonSchema, v))
                for k, v in s['properties'].items()
                if isinstance(v, dict)
            }
        ap = s.get('additionalProperties')
        if isinstance(ap, dict):
            ap_schema = cast(JsonSchema, ap)
            s['additionalProperties'] = _recurse_flatten_allof(ap_schema)
        if isinstance(s.get('patternProperties'), dict):
            s['patternProperties'] = {
                k: _recurse_flatten_allof(cast(JsonSchema, v))
                for k, v in s['patternProperties'].items()
                if isinstance(v, dict)
            }
    elif t == 'array':
        items = s.get('items')
        if isinstance(items, dict):
            s['items'] = _recurse_flatten_allof(cast(JsonSchema, items))
    return s


def _recurse_flatten_allof(schema: JsonSchema) -> JsonSchema:
    s = deepcopy(schema)
    s = _flatten_current_level(s)
    s = _recurse_children(s)
    return s


def flatten_allof(schema: JsonSchema) -> JsonSchema:
    """Flatten simple object-only allOf combinations by merging object members.

    - Merges properties and unions required lists.
    - Combines additionalProperties conservatively: only False if all are False; otherwise True.
    - Recurses into nested object/array members.
    - Leaves non-object allOfs untouched.
    """
    return _recurse_flatten_allof(schema)
