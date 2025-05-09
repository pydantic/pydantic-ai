import re
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

from pydantic_ai.exceptions import UserError

JsonSchema = dict[str, Any]


@dataclass(init=False)
class WalkJsonSchema(ABC):
    """Walks a JSON schema, applying transformations to it at each level.

    Note: We may eventually want to rework tools to build the JSON schema from the type directly, using a subclass of
    pydantic.json_schema.GenerateJsonSchema, rather than making use of this machinery.
    """

    def __init__(
        self, schema: JsonSchema, *, prefer_inlined_defs: bool = False, simplify_nullable_unions: bool = False
    ):
        self.schema = schema
        self.prefer_inlined_defs = prefer_inlined_defs
        self.simplify_nullable_unions = simplify_nullable_unions

        self.defs: dict[str, JsonSchema] = self.schema.get('$defs', {})
        self.refs_stack = tuple[str, ...]()
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

        elif self.recursive_refs:  # pragma: no cover
            # If we are preferring inlined defs and there are recursive refs, we _have_ to use a $defs+$ref structure
            # We try to use whatever the original root key was, but if it is already in use,
            # we modify it to avoid collisions.
            defs = {key: self.defs[key] for key in self.recursive_refs}
            root_ref = self.schema.get('$ref')
            root_key = None if root_ref is None else re.sub(r'^#/\$defs/', '', root_ref)
            if root_key is None:
                root_key = self.schema.get('title', 'root')
                while root_key in defs:
                    # Modify the root key until it is not already in use
                    root_key = f'{root_key}_root'

            defs[root_key] = handled
            return {'$defs': defs, '$ref': f'#/$defs/{root_key}'}

        return handled

    def _handle(self, schema: JsonSchema) -> JsonSchema:
        if self.prefer_inlined_defs:
            while ref := schema.get('$ref'):
                key = re.sub(r'^#/\$defs/', '', ref)
                if key in self.refs_stack:
                    self.recursive_refs.add(key)
                    break  # recursive ref can't be unpacked
                self.refs_stack += (key,)
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
        members = schema.get(union_kind)
        if not members:
            return schema

        handled = [self._handle(member) for member in members]

        # convert nullable unions to nullable types
        if self.simplify_nullable_unions:
            handled = self._simplify_nullable_union(handled)

        if len(handled) == 1:
            # In this case, no need to retain the union
            return handled[0]

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

        return cases  # pragma: no cover
