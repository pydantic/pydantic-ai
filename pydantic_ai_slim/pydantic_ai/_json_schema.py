from __future__ import annotations as _annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, cast

from .exceptions import UserError

JsonSchema = dict[str, Any]

__all__ = ['JsonSchemaTransformer', 'InlineDefsJsonSchemaTransformer']


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
        simplify_nullable_unions: bool = False,  # TODO (v2): Remove this, no longer used
        flatten_allof: bool = False,
    ):
        self.schema = schema

        self.strict = strict
        self.is_strict_compatible = True  # Can be set to False by subclasses to set `strict` on `ToolDefinition `when not set explicitly by the user.

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
            schema = _recurse_flatten_allof(schema)

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
        _process_object_nested_schemas(schema, self._handle)
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

        # TODO (v2): Remove this feature, no longer used
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
        # TODO (v2): Remove this method, no longer used
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


def _get_type_set(schema: JsonSchema) -> set[str] | None:
    """Extract type(s) from a schema as a set of strings."""
    schema_type = schema.get('type')
    if isinstance(schema_type, list):
        return {str(t) for t in cast(list[Any], schema_type)}
    if isinstance(schema_type, str):
        return {schema_type}
    return None


def _process_object_nested_schemas(schema: JsonSchema, process_fn: Callable[[JsonSchema], JsonSchema]) -> None:
    """Process nested schemas in an object schema (properties, additionalProperties, patternProperties).

    Args:
        schema: The object schema to process (modified in place)
        process_fn: Function to apply to each nested schema
    """
    if properties := schema.get('properties'):
        if isinstance(properties, dict):
            properties_dict = cast(dict[str, Any], properties)
            schema['properties'] = {
                k: process_fn(cast(JsonSchema, v)) if isinstance(v, dict) else v for k, v in properties_dict.items()
            }

    if (additional_properties := schema.get('additionalProperties')) is not None:
        if isinstance(additional_properties, dict):
            schema['additionalProperties'] = process_fn(cast(JsonSchema, additional_properties))
        # If it's a bool, leave it as is

    if pattern_properties := schema.get('patternProperties'):
        if isinstance(pattern_properties, dict):
            pattern_properties_dict = cast(dict[str, Any], pattern_properties)
            schema['patternProperties'] = {
                k: process_fn(cast(JsonSchema, v)) if isinstance(v, dict) else v
                for k, v in pattern_properties_dict.items()
            }


def _process_nested_schemas_without_allof(s: JsonSchema) -> JsonSchema:
    """Process nested schemas recursively when there is no allOf at the current level."""
    schema_type = s.get('type')
    if schema_type == 'object':
        _process_object_nested_schemas(s, _recurse_flatten_allof)
    elif schema_type == 'array':
        if isinstance(s.get('items'), dict):
            s['items'] = _recurse_flatten_allof(cast(JsonSchema, s['items']))
    return s


def _collect_base_schema_data(
    result: JsonSchema,
) -> tuple[dict[str, JsonSchema], set[str], dict[str, JsonSchema], list[Any], list[set[str]]]:
    """Collect data from base schema: properties, required, patternProperties, additionalProperties."""
    properties: dict[str, JsonSchema] = {}
    required: set[str] = set()
    pattern_properties: dict[str, JsonSchema] = {}
    additional_values: list[Any] = []
    restricted_property_sets: list[set[str]] = []

    base_properties = (
        cast(dict[str, JsonSchema], result.get('properties', {})) if isinstance(result.get('properties'), dict) else {}
    )
    base_additional = result.get('additionalProperties')

    if base_properties:
        properties.update(base_properties)
    if isinstance(result.get('required'), list):
        required.update(result['required'])
    if isinstance(result.get('patternProperties'), dict):
        pattern_properties.update(result['patternProperties'])
    if base_additional is False:
        additional_values.append(False)
        # Only restrict if base schema has properties; if base has no properties but additionalProperties: False,
        # it means no additional properties are allowed, but properties from allOf members are still valid
        if base_properties:
            restricted_property_sets.append(set(base_properties.keys()))

    return properties, required, pattern_properties, additional_values, restricted_property_sets


def _collect_member_data(
    processed_members: list[JsonSchema],
    properties: dict[str, JsonSchema],
    required: set[str],
    pattern_properties: dict[str, JsonSchema],
    additional_values: list[Any],
    restricted_property_sets: list[set[str]],
    members_properties: list[dict[str, JsonSchema]],
    members_additional_props: list[Any],
) -> None:
    """Collect data from allOf members and update the collections."""
    for m in processed_members:
        member_props = (
            cast(dict[str, JsonSchema], m.get('properties', {})) if isinstance(m.get('properties'), dict) else {}
        )
        members_properties.append(member_props)
        members_additional_props.append(m.get('additionalProperties'))

        if member_props:
            properties.update(member_props)
        if isinstance(m.get('required'), list):
            required.update(m['required'])
        if isinstance(m.get('patternProperties'), dict):
            pattern_properties.update(m['patternProperties'])
        if 'additionalProperties' in m:
            additional_values.append(m['additionalProperties'])
            if m['additionalProperties'] is False:
                restricted_property_sets.append(set(member_props.keys()))


def _filter_by_restricted_property_sets(
    properties: dict[str, JsonSchema], required: set[str], restricted_property_sets: list[set[str]]
) -> tuple[dict[str, JsonSchema], set[str]]:
    """Filter properties and required by restricted property sets (intersection when some/all have additionalProperties: False)."""
    if not restricted_property_sets:
        return properties, required

    # Intersection of allowed properties from all members with additionalProperties: False
    allowed_names = restricted_property_sets[0].copy()
    for prop_set in restricted_property_sets[1:]:
        allowed_names &= prop_set
    # Filter properties to only include allowed names
    if allowed_names:
        properties = {k: v for k, v in properties.items() if k in allowed_names}
        required = {r for r in required if r in allowed_names}
    else:
        # Empty intersection - remove all properties
        properties = {}
        required = set()

    return properties, required


def _filter_incompatible_properties(
    properties: dict[str, JsonSchema],
    required: set[str],
    members_properties: list[dict[str, JsonSchema]],
    members_additional_props: list[Any],
) -> tuple[dict[str, JsonSchema], set[str]]:
    """Filter incompatible properties based on additionalProperties constraints."""
    if not properties:
        return properties, required

    incompatible_props: set[str] = set()

    for prop_name, prop_schema in properties.items():
        prop_types = _get_type_set(prop_schema)

        # Check compatibility with each member (including base)
        for member_props, member_additional in zip(members_properties, members_additional_props):
            if prop_name in member_props:
                # Property explicitly defined - check type compatibility
                member_prop_types = _get_type_set(member_props[prop_name])
                if prop_types and member_prop_types and not prop_types & member_prop_types:
                    incompatible_props.add(prop_name)
                    break
                continue  # Compatible, check next member
            if isinstance(member_additional, dict):
                allowed_types = _get_type_set(cast(JsonSchema, member_additional))
                # Property type must be a subset of allowed types
                if prop_types and allowed_types and not (prop_types <= allowed_types):
                    incompatible_props.add(prop_name)
                    break

    if incompatible_props:
        allowed_names = {k for k in properties.keys() if k not in incompatible_props}
        properties = {k: v for k, v in properties.items() if k in allowed_names}
        required = {r for r in required if r in allowed_names}

    return properties, required


def _process_result_nested_schemas(result: JsonSchema) -> None:
    """Recursively process nested schemas in the result (additionalProperties, patternProperties, items)."""
    if isinstance(result.get('additionalProperties'), dict):
        result['additionalProperties'] = _recurse_flatten_allof(cast(JsonSchema, result['additionalProperties']))
    if isinstance(result.get('patternProperties'), dict):
        result['patternProperties'] = {
            k: _recurse_flatten_allof(cast(JsonSchema, v))
            for k, v in result['patternProperties'].items()
            if isinstance(v, dict)
        }
    if isinstance(result.get('items'), dict):
        result['items'] = _recurse_flatten_allof(cast(JsonSchema, result['items']))


def _recurse_flatten_allof(schema: JsonSchema) -> JsonSchema:
    """Recursively flatten allOf in a JSON schema.

    This function:
    1. Makes a deep copy of the schema
    2. Flattens allOf at the current level
    3. Recursively processes nested schemas (properties, items, etc.)
    """
    s = deepcopy(schema)

    # Case 1: No allOf - process nested schemas recursively and return
    allof = s.get('allOf')
    if not isinstance(allof, list) or not allof:
        return _process_nested_schemas_without_allof(s)

    # Check all members are dicts
    members = cast(list[JsonSchema], allof)
    if not all(isinstance(m, dict) for m in members):
        return s

    # Check all members are object-like (can be merged)
    def _is_object_like(member: JsonSchema) -> bool:
        member_type = member.get('type')
        if member_type is None:
            # No type but has object-like keys
            keys = ('properties', 'additionalProperties', 'patternProperties')
            return bool(any(k in member for k in keys))
        return isinstance(member_type, str) and member_type == 'object'

    if not all(_is_object_like(m) for m in members):
        return s

    # Recursively flatten each member first
    processed_members = [_recurse_flatten_allof(m) for m in members]
    result: JsonSchema = {k: v for k, v in s.items() if k != 'allOf'}
    result['type'] = 'object'

    # Collect data from base schema and members
    base_properties = (
        cast(dict[str, JsonSchema], result.get('properties', {})) if isinstance(result.get('properties'), dict) else {}
    )
    base_additional = result.get('additionalProperties')

    properties, required, pattern_properties, additional_values, restricted_property_sets = _collect_base_schema_data(
        result
    )

    # Then merge properties from all members
    members_properties: list[dict[str, JsonSchema]] = [base_properties]
    members_additional_props: list[Any] = [base_additional]

    _collect_member_data(
        processed_members,
        properties,
        required,
        pattern_properties,
        additional_values,
        restricted_property_sets,
        members_properties,
        members_additional_props,
    )

    # Filter by restricted property sets and incompatible properties
    properties, required = _filter_by_restricted_property_sets(properties, required, restricted_property_sets)
    properties, required = _filter_incompatible_properties(
        properties, required, members_properties, members_additional_props
    )

    # Apply filtered properties
    if properties:
        # Recursively flatten nested properties
        result['properties'] = {k: _recurse_flatten_allof(v) for k, v in properties.items()}
    if required:
        result['required'] = sorted(required)
    if pattern_properties:
        result['patternProperties'] = {k: _recurse_flatten_allof(v) for k, v in pattern_properties.items()}

    # Merge additionalProperties
    if additional_values:
        # If any is False, result is False (most restrictive)
        if any(v is False for v in additional_values):
            result['additionalProperties'] = False
        # If there's exactly one dict schema, preserve it
        elif len(additional_values) == 1 and isinstance(additional_values[0], dict):
            result['additionalProperties'] = additional_values[0]
        # If any is a dict schema (multiple), result is True (can't merge multiple schemas)
        elif any(isinstance(v, dict) for v in additional_values):
            result['additionalProperties'] = True
        # Otherwise, default to True
        else:
            result['additionalProperties'] = True

    # Recursively process nested schemas (additionalProperties, patternProperties)
    # Note: items is only valid for array types, not object types, so result.get('items') should never
    # be present when result['type'] == 'object'. However, we keep this check for robustness.
    _process_result_nested_schemas(result)

    return result
