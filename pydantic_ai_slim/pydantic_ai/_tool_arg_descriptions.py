from __future__ import annotations

import copy
from typing import Any

from .exceptions import UserError

# JSON Schema keys used for traversal
_PROPERTIES = 'properties'
_DEFS = '$defs'
_REF = '$ref'
_REF_PREFIX = '#/$defs/'
_DESCRIPTION = 'description'


class ToolArgDescriptions:
    """Utility class for handling tool argument descriptions in JSON schemas."""

    @staticmethod
    def from_json_schema(schema: dict[str, Any]) -> dict[str, str]:
        """Extract field descriptions from a JSON schema into dot notation format.

        Recursively traverses the schema's properties to build a flat dictionary mapping
        dot-notation paths to their descriptions. This is useful for prompt optimizers
        that need to modify tool argument descriptions.
        """
        properties = schema.get(_PROPERTIES, {})
        if not properties:
            return {}

        result: dict[str, str] = {}
        defs = schema.get(_DEFS, {})
        visited: set[str] = set()

        def extract_from_properties(path: str, props: dict[str, Any]) -> None:
            """Recursively extract descriptions from properties."""
            for key, value in props.items():
                full_path = f'{path}.{key}' if path else key

                if description := value.get(_DESCRIPTION):
                    result[full_path] = description

                if nested_props := value.get(_PROPERTIES):
                    # Handle nested properties directly (e.g. nested models inline)
                    extract_from_properties(full_path, nested_props)
                elif (ref := value.get(_REF)) and ref.startswith(_REF_PREFIX):
                    # Handle $ref (shared definitions / recursive models)
                    def_name = ref[len(_REF_PREFIX) :]
                    # Avoid infinite recursion for recursive models (e.g. User -> best_friend: User)
                    if def_name not in visited:
                        visited.add(def_name)
                        if nested_props := defs.get(def_name, {}).get(_PROPERTIES):
                            extract_from_properties(full_path, nested_props)
                        visited.remove(def_name)

        extract_from_properties('', properties)
        return result

    @staticmethod
    def update_in_json_schema(
        schema: dict[str, Any],
        arg_descriptions: dict[str, str],
        tool_name: str,
    ) -> dict[str, Any]:
        """Update descriptions for argument paths in the JSON schema.

        Returns a new schema with updated descriptions, leaving the original schema unchanged.
        """
        schema = copy.deepcopy(schema)
        defs = schema.get(_DEFS, {})

        for arg_path, description in arg_descriptions.items():
            current = schema
            parts = arg_path.split('.')

            for i, part in enumerate(parts):
                # 1. Resolve $ref if present.
                # We inline the definition to avoid modifying shared definitions in $defs
                # and to handle chained references (A -> B -> C).
                visited_refs: set[str] = set()
                while (ref := current.get(_REF)) and isinstance(ref, str) and ref.startswith(_REF_PREFIX):
                    if ref in visited_refs:
                        raise UserError(f"Circular reference detected in schema at '{arg_path}': {ref}")
                    visited_refs.add(ref)

                    def_name = ref[len(_REF_PREFIX) :]
                    if def_name not in defs:
                        raise UserError(f"Invalid path '{arg_path}' for tool '{tool_name}': undefined $ref '{ref}'.")

                    # Inline the definition: replace 'current' contents with the definition's contents.
                    # This ensures we don't mutate the shared definition in $defs.
                    #
                    # Example of why this is needed:
                    #   "sender": { "$ref": "#/$defs/User" }
                    #   "receiver": { "$ref": "#/$defs/User" }
                    #
                    # If we want to change the description of a field inside 'sender', we must
                    # resolve the reference and copy the 'User' definition into 'sender' first.
                    #
                    # Result after inlining 'sender':
                    #   "sender": {
                    #       # ... copy of User fields ...
                    #       "properties": {
                    #           "name": { "description": "I CHANGED THIS!", ... }
                    #       }
                    #   },
                    #   "receiver": { "$ref": "#/$defs/User" }  <-- Remains a reference!

                    target_def = defs[def_name]
                    # Detach from the shared definition to isolate changes to this specific path.
                    # e.g. If both 'sender' and 'receiver' ref 'User', modifying 'sender'
                    # should not affect 'receiver'.
                    #
                    # We pop '_REF' because a JSON object cannot simultaneously be a reference ('$ref')
                    # and have its own 'properties'. By removing '$ref', we convert this node from
                    # a pointer into a standard object that holds its own copy of the data.
                    #
                    # Before (current is a pointer):
                    #   { "$ref": "#/$defs/User" }
                    #
                    # After pop(_REF) (current is empty, ready for update):
                    #   { }
                    #
                    # After update(target_def) (current is now a standalone copy):
                    #   { "properties": { "name": ... }, "description": ... }
                    current.pop(_REF)

                    # Inline the schema definition so we can traverse and modify its nested fields.
                    current.update(copy.deepcopy(target_def))

                # 2. Check if the property exists in the current schema
                props = current.get(_PROPERTIES, {})
                if part not in props:
                    available = ', '.join(f"'{p}'" for p in props)
                    raise UserError(
                        f"Invalid path '{arg_path}' for tool '{tool_name}': "
                        f"'{part}' not found. Available properties: {available}"
                    )

                # 3. If it's the last part of the path, update the description
                if i == len(parts) - 1:
                    props[part][_DESCRIPTION] = description
                else:
                    # 4. Otherwise, continue traversing down the tree
                    current = props[part]
        return schema
