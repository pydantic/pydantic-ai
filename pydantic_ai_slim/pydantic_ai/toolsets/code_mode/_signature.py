"""Convert JSON schemas to Python function signatures.

This module provides utilities to convert JSON schemas (like those used in tool definitions)
into human-readable Python function signature strings, which LLMs can understand more easily
than raw JSON schemas.
"""

from __future__ import annotations

import dataclasses
import re
import types
from collections.abc import Callable
from dataclasses import dataclass, field
from inspect import Parameter, signature
from typing import Any, Union, cast, get_origin

from pydantic import BaseModel, TypeAdapter
from pydantic._internal import _typing_extra

from ..._function_schema import _is_call_ctx  # pyright: ignore[reportPrivateUsage]


def _is_typeddict(t: Any) -> bool:
    """Check if a type is a `TypedDict` subclass."""
    return isinstance(t, type) and hasattr(t, '__annotations__') and hasattr(t, '__total__')


def _is_named_type(t: Any) -> bool:
    """Check if a type is a `BaseModel`, dataclass, or `TypedDict` that needs a definition."""
    if not isinstance(t, type):
        return False
    return issubclass(t, BaseModel) or dataclasses.is_dataclass(t) or _is_typeddict(t)  # pyright: ignore[reportUnknownArgumentType]


def _get_schema_from_type(t: Any) -> dict[str, Any]:
    """Extract JSON schema from a `BaseModel`, dataclass, or `TypedDict`."""
    if isinstance(t, type) and issubclass(t, BaseModel):
        return t.model_json_schema()
    return TypeAdapter(t).json_schema()  # pyright: ignore[reportUnknownArgumentType]


# =============================================================================
# Signature class - the main public API
# =============================================================================


@dataclass
class Signature:
    """Python function signature with TypedDict definitions.

    This class holds all the data needed to render a function signature as Python code.
    Use `str(sig)` for the default rendering, or call specific methods for variants.
    """

    name: str
    """The function name."""

    params: list[str]
    """Pre-formatted parameter strings, e.g. ["x: int", "y: str = 'default'"]."""

    return_type: str
    """The return type annotation string."""

    docstring: str | None = None
    """Optional docstring for the function."""

    typeddicts: list[str] = field(default_factory=lambda: [])
    """TypedDict class definitions needed by the signature."""

    is_async: bool = True
    """Whether to generate 'async def' (True) or 'def' (False)."""

    def __str__(self) -> str:
        """Render with `...` body."""
        return self._render('...')

    def with_ellipsis(self) -> str:
        """Render with `...` body (for LLM display)."""
        return self._render('...')

    def with_typeddicts(self, body: str = '...') -> str:
        """Render with TypedDict definitions prepended."""
        sig = self._render(body)
        if not self.typeddicts:
            return sig
        return '\n\n'.join(self.typeddicts + [sig])

    def _render(self, body: str) -> str:
        """Render the signature with a specific body."""
        prefix = 'async def' if self.is_async else 'def'
        params_str = ', '.join(self.params)
        sig_line = f'{prefix} {self.name}({params_str}) -> {self.return_type}'

        if self.docstring:
            docstring_str = _format_docstring(self.docstring)
            return f'{sig_line}:\n{docstring_str}\n    {body}'
        return f'{sig_line}:\n    {body}'


# =============================================================================
# Public API functions
# =============================================================================


def signature_from_function(
    func: Callable[..., Any],
    name: str | None = None,
    description: str | None = None,
    *,
    include_return_type: bool = True,
) -> Signature:
    """Generate a Signature from an actual function.

    Uses inspect.signature() and typing.get_type_hints() to reconstruct
    the function signature with type annotations.

    Args:
        func: The function to generate a signature for.
        name: Override the function name. If None, uses func.__name__.
        description: Optional description to include as a docstring.
        include_return_type: Whether to include the return type annotation.

    Returns:
        A Signature object that can be rendered to string with str() or methods.
    """
    return _function_to_signature(func, name, description, include_return_type=include_return_type)


def signature_from_schema(
    name: str,
    parameters_json_schema: dict[str, Any],
    description: str | None = None,
    return_type: str = 'Any',
    return_json_schema: dict[str, Any] | None = None,
    *,
    namespace_defs: bool = False,
) -> Signature:
    """Convert a JSON schema to a Signature.

    Args:
        name: The function name.
        parameters_json_schema: The JSON schema for the function's parameters.
        description: Optional function description to include as docstring.
        return_type: The return type annotation string. Defaults to 'Any'.
        return_json_schema: Optional JSON schema for the return value.
        namespace_defs: Whether to prefix $defs names to avoid param/return collisions.

    Returns:
        A Signature object that can be rendered to string with str() or methods.
    """
    return _schema_to_signature(
        name,
        parameters_json_schema,
        description,
        return_type,
        return_json_schema,
        namespace_defs=namespace_defs,
    )


# =============================================================================
# Path-based naming for unique TypedDict names
# =============================================================================


def _to_pascal_case(s: str) -> str:
    """Convert a string to PascalCase."""
    s = re.sub(r'[^a-zA-Z0-9]', '_', s)
    parts = s.split('_')
    return ''.join(part.capitalize() for part in parts if part)


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
# Formatting utilities
# =============================================================================


def _format_docstring(description: str, indent: str = '    ') -> str:
    """Format a description as a docstring."""
    lines = description.strip().split('\n')
    if len(lines) == 1:
        return f'{indent}"""{lines[0]}"""'
    else:
        result = [f'{indent}"""']
        for line in lines:
            result.append(f'{indent}{line}' if line.strip() else '')
        result.append(f'{indent}"""')
        return '\n'.join(result)


def _format_annotation(annotation: Any) -> str:
    """Format a type annotation as a string."""
    if annotation is None or annotation is type(None):
        return 'None'

    # Handle Python 3.10+ union syntax (X | Y creates types.UnionType)
    if isinstance(annotation, types.UnionType):
        args = getattr(annotation, '__args__', ())
        return ' | '.join(_format_annotation(arg) for arg in args)

    origin = getattr(annotation, '__origin__', None)
    args = getattr(annotation, '__args__', None)

    if origin is not None:
        origin_name = _get_type_name(origin)

        if args:
            formatted_args = ', '.join(_format_annotation(arg) for arg in args)
            if origin is Union:
                return ' | '.join(_format_annotation(arg) for arg in args)
            return f'{origin_name}[{formatted_args}]'
        return origin_name

    return _get_type_name(annotation)


def _get_type_name(t: Any) -> str:
    """Get the name of a type."""
    if t is type(None):
        return 'None'
    if hasattr(t, '__name__'):
        return t.__name__
    s = str(t)
    return s.replace('typing.', '').replace('typing_extensions.', '')


# =============================================================================
# Function signature builder (using closures)
# =============================================================================


def _function_to_signature(
    func: Callable[..., Any],
    name: str | None = None,
    description: str | None = None,
    *,
    include_return_type: bool = True,
) -> Signature:
    """Build Signature from a Python function using inspect."""
    name = name or func.__name__
    sig = signature(func)

    try:
        type_hints = _typing_extra.get_function_type_hints(func)
    except Exception:
        type_hints = {}

    # Closure state for collecting TypedDicts
    typeddicts: dict[str, str] = {}

    def collect_typeddicts_from_annotation(annotation: Any, path: str = '') -> None:
        """Recursively collect TypedDicts from complex type annotations."""
        if annotation is None or annotation is type(None):
            return

        if _is_named_type(annotation):
            type_name = annotation.__name__
            if type_name not in typeddicts:
                schema = _get_schema_from_type(annotation)
                # Process any $defs first
                if '$defs' in schema:
                    for def_name, def_schema in schema['$defs'].items():
                        if (
                            def_name not in typeddicts
                            and def_schema.get('type') == 'object'
                            and 'properties' in def_schema
                        ):
                            typeddicts[def_name] = _generate_typeddict_str(
                                def_name,
                                def_schema,
                                lambda s, p: _schema_type_to_str(s, schema.get('$defs', {}), typeddicts, name or '', p),
                                path,
                            )
                # Then process the main schema
                if schema.get('type') == 'object' and 'properties' in schema:
                    typeddicts[type_name] = _generate_typeddict_str(
                        type_name,
                        schema,
                        lambda s, p: _schema_type_to_str(s, schema.get('$defs', {}), typeddicts, name or '', p),
                        path,
                    )
                elif '$ref' in schema:
                    # Handle $ref at top level - ensure referenced def is generated
                    ref_name = (
                        schema['$ref'][8:] if schema['$ref'].startswith('#/$defs/') else schema['$ref'].split('/')[-1]
                    )
                    if ref_name in schema.get('$defs', {}) and ref_name not in typeddicts:
                        ref_schema = schema['$defs'][ref_name]
                        if ref_schema.get('type') == 'object' and 'properties' in ref_schema:
                            typeddicts[ref_name] = _generate_typeddict_str(
                                ref_name,
                                ref_schema,
                                lambda s, p: _schema_type_to_str(s, schema.get('$defs', {}), typeddicts, name or '', p),
                                path,
                            )
            return

        origin = get_origin(annotation)
        args = getattr(annotation, '__args__', None)

        if origin is not None and args:
            for arg in args:
                collect_typeddicts_from_annotation(arg, path)

    params: list[str] = []

    for i, (param_name, param) in enumerate(sig.parameters.items()):
        annotation = type_hints.get(param_name)

        # Skip RunContext parameter (first param only)
        if i == 0 and annotation is not None and _is_call_ctx(annotation):
            continue

        if annotation is not None:
            collect_typeddicts_from_annotation(annotation, param_name)

        annotation_str = _format_annotation(annotation) if annotation else ''

        if param.default is Parameter.empty:
            if annotation_str:
                params.append(f'{param_name}: {annotation_str}')
            else:
                params.append(param_name)
        else:
            default_str = repr(param.default)
            if annotation_str:
                params.append(f'{param_name}: {annotation_str} = {default_str}')
            else:
                params.append(f'{param_name}={default_str}')

    # Handle return type
    return_annotation = type_hints.get('return')
    if return_annotation is not None:
        collect_typeddicts_from_annotation(return_annotation, 'Return')

    if include_return_type:
        return_type_str = _format_annotation(return_annotation) if return_annotation else 'Any'
    else:
        return_type_str = 'Any'

    return Signature(
        name=name,
        params=params,
        return_type=return_type_str,
        docstring=description,
        typeddicts=list(typeddicts.values()),
    )


# =============================================================================
# Schema signature builder (using closures)
# =============================================================================


def _schema_to_signature(
    name: str,
    parameters_schema: dict[str, Any],
    description: str | None = None,
    return_type: str = 'Any',
    return_schema: dict[str, Any] | None = None,
    *,
    namespace_defs: bool = False,
) -> Signature:
    """Convert JSON schema to Signature using closures for state management."""
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
        return _schema_type_to_str(schema, defs, typeddicts, name, path)

    # Pre-process $defs to generate TypedDicts
    for def_name, def_schema in defs.items():
        if def_schema.get('type') == 'object' and 'properties' in def_schema:
            if def_name not in typeddicts:
                typeddicts[def_name] = _generate_typeddict_str(def_name, def_schema, to_type, def_name)

    # Build parameters
    params = _build_params_from_schema(parameters_schema, to_type)

    # Resolve return type
    resolved_return_type = return_type
    if return_schema is not None and return_type == 'Any':
        resolved_return_type = to_type(return_schema, 'Return')

    # Handle case where return type couldn't be resolved
    final_description = description
    if return_schema is not None and resolved_return_type == 'Any':
        import json

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


def _schema_type_to_str(
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
                typeddicts[ref_name] = _generate_typeddict_str(
                    ref_name, ref_schema, lambda s, p: _schema_type_to_str(s, defs, typeddicts, tool_name, p), path
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
            return _schema_type_to_str(schema['allOf'][0], defs, typeddicts, tool_name, path)
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
    # Simple types
    type_mapping = {
        'string': 'str',
        'integer': 'int',
        'number': 'float',
        'boolean': 'bool',
        'null': 'None',
    }

    if schema_type in type_mapping:
        return type_mapping[schema_type]

    # Array type
    if schema_type == 'array':
        items = schema.get('items', {})
        if items:
            # Handle tuple schemas (items as list)
            if isinstance(items, list):
                items_list = cast(list[dict[str, Any]], items)
                item_types = [
                    _schema_type_to_str(item, defs, typeddicts, tool_name, f'{path}.{i}')
                    for i, item in enumerate(items_list)
                ]
                return f'tuple[{", ".join(item_types)}]'
            item_type = _schema_type_to_str(cast(dict[str, Any], items), defs, typeddicts, tool_name, f'{path}Item')
            return f'list[{item_type}]'
        return 'list[Any]'

    # Object type
    if schema_type == 'object':
        if 'properties' in schema:
            # Generate TypedDict with path-based unique name
            td_name = _path_to_typename(tool_name, path)
            if td_name not in typeddicts:
                typeddicts[td_name] = _generate_typeddict_str(
                    td_name, schema, lambda s, p: _schema_type_to_str(s, defs, typeddicts, tool_name, p), path
                )
            return td_name
        if 'additionalProperties' in schema:
            additional = schema['additionalProperties']
            if additional is True:
                return 'dict[str, Any]'
            if isinstance(additional, dict):
                additional_schema = cast(dict[str, Any], additional)
                value_type = _schema_type_to_str(additional_schema, defs, typeddicts, tool_name, f'{path}Value')
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
            types.append(_schema_type_to_str(s, defs, typeddicts, tool_name, path))

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


def _json_type_to_python(json_type: str) -> str:
    """Convert a JSON type string to Python type."""
    mapping = {
        'string': 'str',
        'integer': 'int',
        'number': 'float',
        'boolean': 'bool',
        'null': 'None',
        'array': 'list',
        'object': 'dict',
    }
    return mapping.get(json_type, 'Any')


def _generate_typeddict_str(
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
    import copy

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
