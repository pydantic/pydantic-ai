"""JSON schema to Python type conversion utilities.

This module converts JSON schemas into Python type strings and TypedDict definitions,
used by `_python_signature.py` to generate function signatures from tool definitions.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable
from typing import Any, cast

from ._python_signature import Signature

# =============================================================================
# Path-based naming for unique TypedDict names
# =============================================================================


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


# =============================================================================
# JSON type mapping
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


# =============================================================================
# Schema to signature conversion
# =============================================================================


def schema_to_signature(
    name: str,
    parameters_schema: dict[str, Any],
    description: str | None = None,
    return_type: str = 'Any',
    return_schema: dict[str, Any] | None = None,
    *,
    namespace_defs: bool = False,
) -> Signature:
    """Convert JSON schema to a Signature."""
    # Merge all $defs into a single dict
    defs: dict[str, dict[str, Any]] = {}
    typeddicts: dict[str, str] = {}  # name -> definition string

    # Process parameter schema $defs
    if '$defs' in parameters_schema:
        param_defs = parameters_schema['$defs']
        if namespace_defs:
            param_defs = {f'Param{k}': v for k, v in param_defs.items()}
            parameters_schema = _update_refs_with_prefix(parameters_schema, 'Param')
        defs.update(param_defs)

    # Process return schema $defs
    if return_schema is not None and '$defs' in return_schema:
        return_defs = return_schema['$defs']
        if namespace_defs:
            return_defs = {f'Return{k}': v for k, v in return_defs.items()}
            return_schema = _update_refs_with_prefix(return_schema, 'Return')
        defs.update(return_defs)

    def to_type(schema: dict[str, Any], path: str) -> str:
        """Convert schema to type string. Closure captures defs, typeddicts."""
        return schema_type_to_str(schema, defs, typeddicts, name, path)

    # Pre-process $defs to generate TypedDicts
    for def_name, def_schema in defs.items():
        if def_schema.get('type') == 'object' and 'properties' in def_schema:
            if def_name not in typeddicts:
                typeddicts[def_name] = generate_typeddict_str(def_name, def_schema, to_type, def_name)

    # Build parameters
    params = _build_params_from_schema(parameters_schema, to_type)

    # Resolve return type
    resolved_return_type = return_type
    if return_schema is not None and return_type == 'Any':
        resolved_return_type = to_type(return_schema, 'Return')

    # Handle case where return type couldn't be resolved
    final_description = description
    if return_schema is not None and resolved_return_type == 'Any':
        return_schema_blob = json.dumps(return_schema, indent=2)
        return_schema_note = f'\n\nReturn schema:\n{return_schema_blob}'
        final_description = (description or '') + return_schema_note
        final_description = final_description.strip()

    return Signature(
        name=name,
        params=params,
        return_type=resolved_return_type,
        docstring=final_description if final_description else None,
        typeddicts=list(typeddicts.values()),
    )


def _build_params_from_schema(
    schema: dict[str, Any],
    to_type: Callable[[dict[str, Any], str], str],
) -> list[str]:
    """Convert a JSON schema to a list of parameter strings."""
    properties = schema.get('properties', {})
    required = set(schema.get('required', []))

    required_params: list[str] = []
    optional_params: list[str] = []

    for prop_name, prop_schema in properties.items():
        type_str = to_type(prop_schema, prop_name)

        if 'default' in prop_schema:
            default_str = repr(prop_schema['default'])
            optional_params.append(f'{prop_name}: {type_str} = {default_str}')
        elif prop_name in required:
            required_params.append(f'{prop_name}: {type_str}')
        else:
            # Check if schema already allows null before adding | None
            if _schema_allows_null(prop_schema):
                optional_params.append(f'{prop_name}: {type_str} = None')
            else:
                optional_params.append(f'{prop_name}: {type_str} | None = None')

    return required_params + optional_params


def _schema_allows_null(schema: dict[str, Any]) -> bool:
    """Check if a schema already allows null values."""
    schema_type = schema.get('type')
    if isinstance(schema_type, list) and 'null' in schema_type:
        return True
    if 'anyOf' in schema or 'oneOf' in schema:
        union = schema.get('anyOf') or schema.get('oneOf', [])
        return any(s.get('type') == 'null' for s in union)
    return False


def schema_type_to_str(
    schema: dict[str, Any],
    defs: dict[str, dict[str, Any]],
    typeddicts: dict[str, str],
    tool_name: str,
    path: str,
) -> str:
    """Convert a JSON schema to a Python type string."""
    # Handle $ref
    if '$ref' in schema:
        ref = schema['$ref']
        ref_name = ref[8:] if ref.startswith('#/$defs/') else ref.split('/')[-1]
        # Ensure referenced def generates TypedDict if needed
        if ref_name in defs and ref_name not in typeddicts:
            ref_schema = defs[ref_name]
            if ref_schema.get('type') == 'object' and 'properties' in ref_schema:
                typeddicts[ref_name] = generate_typeddict_str(
                    ref_name, ref_schema, lambda s, p: schema_type_to_str(s, defs, typeddicts, tool_name, p), path
                )
        return ref_name

    # Handle anyOf/oneOf (union types)
    if 'anyOf' in schema:
        return _handle_union_schema(schema['anyOf'], defs, typeddicts, tool_name, path)
    if 'oneOf' in schema:
        return _handle_union_schema(schema['oneOf'], defs, typeddicts, tool_name, path)

    # Handle allOf
    if 'allOf' in schema:
        if len(schema['allOf']) == 1:
            return schema_type_to_str(schema['allOf'][0], defs, typeddicts, tool_name, path)
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
    return _type_to_str(schema_type, schema, defs, typeddicts, tool_name, path)


def _type_to_str(
    schema_type: str | list[str] | None,
    schema: dict[str, Any],
    defs: dict[str, dict[str, Any]],
    typeddicts: dict[str, str],
    tool_name: str,
    path: str,
) -> str:
    """Convert a schema type to Python type string."""
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
                item_types = [
                    schema_type_to_str(item, defs, typeddicts, tool_name, f'{path}.{i}')
                    for i, item in enumerate(items_list)
                ]
                return f'tuple[{", ".join(item_types)}]'
            item_type = schema_type_to_str(cast(dict[str, Any], items), defs, typeddicts, tool_name, f'{path}Item')
            return f'list[{item_type}]'
        return 'list[Any]'

    # Object type
    if schema_type == 'object':
        if 'properties' in schema:
            # Generate TypedDict with path-based unique name
            td_name = _path_to_typename(tool_name, path)
            if td_name not in typeddicts:
                typeddicts[td_name] = generate_typeddict_str(
                    td_name, schema, lambda s, p: schema_type_to_str(s, defs, typeddicts, tool_name, p), path
                )
            return td_name
        if 'additionalProperties' in schema:
            additional = schema['additionalProperties']
            if additional is True:
                return 'dict[str, Any]'
            if isinstance(additional, dict):
                additional_schema = cast(dict[str, Any], additional)
                value_type = schema_type_to_str(additional_schema, defs, typeddicts, tool_name, f'{path}Value')
                return f'dict[str, {value_type}]'
        return 'dict[str, Any]'

    # Type list (e.g., ['string', 'null'])
    if isinstance(schema_type, list):
        # Check if this is object with properties + null
        if 'object' in schema_type and 'properties' in schema:
            base_type = _type_to_str('object', schema, defs, typeddicts, tool_name, path)
            if 'null' in schema_type:
                return f'{base_type} | None'
            return base_type

        types = [_json_type_to_python(t) for t in schema_type]
        types = [t for t in types if t]
        if len(types) == 2 and 'None' in types:
            non_none = [t for t in types if t != 'None'][0]
            return f'{non_none} | None'
        return ' | '.join(types) if types else 'Any'

    return 'Any'


def _handle_union_schema(
    schemas: list[dict[str, Any]],
    defs: dict[str, dict[str, Any]],
    typeddicts: dict[str, str],
    tool_name: str,
    path: str,
) -> str:
    """Handle anyOf/oneOf schemas."""
    types: list[str] = []
    has_null = False

    for s in schemas:
        if s.get('type') == 'null':
            has_null = True
        else:
            types.append(schema_type_to_str(s, defs, typeddicts, tool_name, path))

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_types: list[str] = []
    for t in types:
        if t not in seen:
            seen.add(t)
            unique_types.append(t)

    if has_null and len(unique_types) == 1:
        return f'{unique_types[0]} | None'
    elif has_null:
        return ' | '.join(unique_types) + ' | None'
    elif len(unique_types) == 1:
        return unique_types[0]
    else:
        return ' | '.join(unique_types)


def generate_typeddict_str(
    name: str,
    schema: dict[str, Any],
    to_type: Callable[[dict[str, Any], str], str],
    path: str,
) -> str:
    """Generate a TypedDict definition string for an object schema."""
    properties = schema.get('properties', {})
    required = set(schema.get('required', []))

    lines = [f'class {name}(TypedDict):']

    if not properties:
        lines.append('    pass')
    else:
        for prop_name, prop_schema in properties.items():
            prop_path = f'{path}.{prop_name}' if path else prop_name
            type_str = to_type(prop_schema, prop_path)
            is_required = prop_name in required
            if not is_required:
                type_str = f'NotRequired[{type_str}]'
            desc = prop_schema.get('description', '')
            if desc:
                # Handle multiline descriptions by putting them as comments above the field
                desc_lines = desc.split('\n')
                for desc_line in desc_lines:
                    lines.append(f'    # {desc_line}')
            lines.append(f'    {prop_name}: {type_str}')

    return '\n'.join(lines)


def _update_refs_with_prefix(schema: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Return a copy of a schema with $ref names prefixed."""

    def update_refs(obj: Any) -> Any:
        if isinstance(obj, dict):
            obj_dict = cast(dict[str, Any], obj)
            result: dict[str, Any] = {}
            for key, value in obj_dict.items():
                if key == '$ref' and isinstance(value, str) and value.startswith('#/$defs/'):
                    ref_name = value[8:]
                    result[key] = f'#/$defs/{prefix}{ref_name}'
                elif key == '$defs':
                    # Skip $defs - they're handled separately
                    result[key] = value
                else:
                    result[key] = update_refs(value)
            return result
        elif isinstance(obj, list):
            obj_list = cast(list[Any], obj)
            return [update_refs(item) for item in obj_list]
        return obj

    return update_refs(copy.deepcopy(schema))
