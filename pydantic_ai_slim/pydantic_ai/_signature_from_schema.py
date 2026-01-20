"""Convert JSON schemas to Python function signatures.

This module provides utilities to convert JSON schemas (like those used in tool definitions)
into human-readable Python function signature strings, which LLMs can understand more easily
than raw JSON schemas.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from inspect import Parameter, signature
from typing import Any, Union, cast

from pydantic._internal import _typing_extra

from ._run_context import RunContext


def _is_run_context(annotation: Any) -> bool:
    """Check if an annotation is RunContext or RunContext[T]."""
    if annotation is RunContext:
        return True
    origin = getattr(annotation, '__origin__', None)
    return origin is RunContext


@dataclass
class SignatureResult:
    """Result of signature generation."""

    signature: str
    """The Python signature string, e.g. 'def foo(x: int, y: str = "default") -> Any'."""
    typeddict_defs: list[str]
    """Any TypedDict class definitions needed by the signature."""
    return_type: str
    """The resolved return type annotation string."""


def signature_from_function(
    func: Callable[..., Any],
    name: str | None = None,
    description: str | None = None,
) -> str:
    """Generate a Python signature string from an actual function.

    Uses inspect.signature() and typing.get_type_hints() to reconstruct
    the function signature with type annotations.

    Args:
        func: The function to generate a signature for.
        name: Override the function name. If None, uses func.__name__.
        description: Optional description to include as a docstring.

    Returns:
        A Python signature string including optional docstring.
    """
    name = name or func.__name__
    sig = signature(func)

    try:
        type_hints = _typing_extra.get_function_type_hints(func)
    except Exception:
        type_hints = {}

    params: list[str] = []

    for i, (param_name, param) in enumerate(sig.parameters.items()):
        annotation = type_hints.get(param_name)

        if i == 0 and annotation is not None and _is_run_context(annotation):
            continue

        annotation_str = _format_annotation(annotation) if annotation else ''

        if param.default is Parameter.empty:
            if annotation_str:
                params.append(f'{param_name}: {annotation_str}')
            else:
                params.append(param_name)
        else:
            default_str = _format_default(param.default)
            if annotation_str:
                params.append(f'{param_name}: {annotation_str} = {default_str}')
            else:
                params.append(f'{param_name}={default_str}')

    return_annotation = type_hints.get('return')
    return_str = f' -> {_format_annotation(return_annotation)}' if return_annotation else ' -> Any'

    is_async = _is_async_function(func)
    prefix = 'async def' if is_async else 'def'

    signature_line = f'{prefix} {name}({", ".join(params)}){return_str}'

    if description:
        docstring = _format_docstring(description)
        return f'{signature_line}:\n{docstring}'
    else:
        return f'{signature_line}: ...'


def _is_async_function(func: Callable[..., Any]) -> bool:
    """Check if a function is async."""
    if asyncio.iscoroutinefunction(func):
        return True
    if inspect.ismethod(func):
        return asyncio.iscoroutinefunction(func.__func__)
    return False


def _format_annotation(annotation: Any) -> str:
    """Format a type annotation as a string."""
    if annotation is None or annotation is type(None):
        return 'None'

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


# TODO: refactor this and the rest of the unstaged code to look cleaner
# i.e. handle all cases where return str(value) in one branch ffs!
def _format_default(value: Any) -> str:
    """Format a default value as a string."""
    if isinstance(value, str):
        return repr(value)
    if value is None:
        return 'None'
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, dict)):
        return repr(value)  # pyright: ignore[reportUnknownArgumentType]
    return repr(value)


# TODO we need to scope how much of this should be customizable by the user
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


def signature_from_schema(
    name: str,
    parameters_json_schema: dict[str, Any],
    description: str | None = None,
    return_type: str = 'Any',
    return_json_schema: dict[str, Any] | None = None,
) -> SignatureResult:
    """Convert a JSON schema to a Python function signature string.

    Args:
        name: The function name.
        parameters_json_schema: The JSON schema for the function's parameters.
        description: Optional function description to include as docstring.
        return_type: The return type annotation string. Defaults to 'Any'.
        return_json_schema: Optional JSON schema for the return value.

    Returns:
        A SignatureResult containing the signature string and any TypedDict definitions.
    """
    context = _ConversionContext(name)
    params = _schema_to_params(parameters_json_schema, context)

    resolved_return_type = return_type
    if return_json_schema is not None and return_type == 'Any':
        if return_json_schema.get('$defs'):
            context.set_defs(return_json_schema['$defs'])
            _process_defs(return_json_schema['$defs'], context)
        resolved_return_type = _schema_to_type(return_json_schema, context, 'Return')

    typeddict_defs = context.get_typeddict_definitions()

    signature_line = f'def {name}({", ".join(params)}) -> {resolved_return_type}'

    if description:
        docstring = _format_docstring(description)
        signature_str = f'{signature_line}:\n{docstring}'
    else:
        signature_str = f'{signature_line}: ...'

    if return_json_schema is not None and resolved_return_type == 'Any':
        return_schema_blob = json.dumps(return_json_schema, indent=2)
        return_schema_note = f'\n\nReturn schema:\n{return_schema_blob}'
        if description:
            signature_str = f'{signature_line}:\n{_format_docstring(description + return_schema_note)}'
        else:
            signature_str = f'{signature_line}:\n{_format_docstring(return_schema_note.strip())}'

    return SignatureResult(signature=signature_str, typeddict_defs=typeddict_defs, return_type=resolved_return_type)


class _ConversionContext:
    """Context for tracking TypedDict definitions during conversion."""

    def __init__(self, tool_name: str):
        self.tool_name = tool_name
        self._typeddict_defs: dict[str, str] = {}
        self._defs: dict[str, dict[str, Any]] = {}
        self._counter = 0

    def set_defs(self, defs: dict[str, dict[str, Any]]) -> None:
        # TODO: Using update() can silently overwrite $defs if parameter schema and return schema
        # both define a $def with the same name but different definitions.
        # Example: params has `$defs: {User: {name: str}}`, return has `$defs: {User: {id: int}}`
        # The return schema's User will overwrite the params User, causing incorrect TypedDict generation.
        #
        # Expected correct behavior options:
        # 1. Raise an error if conflicting $defs are detected (same name, different schema)
        # 2. Namespace the defs by source (e.g., "ParamsUser" vs "ReturnUser")
        # 3. Check if definitions are identical - allow if same, error if different
        #
        # TODO: Consider stashing separate defs for params/returns to avoid silent clashes.
        # For now, we assume $defs names are unique across param and return schemas.
        self._defs.update(defs)

    def resolve_ref(self, ref: str) -> dict[str, Any]:
        """Resolve a $ref to its schema."""
        if ref.startswith('#/$defs/'):
            def_name = ref[len('#/$defs/') :]
            if def_name in self._defs:
                return self._defs[def_name]
        raise ValueError(f'Unable to resolve $ref: {ref}')

    def get_ref_name(self, ref: str) -> str:
        """Get the Python type name for a $ref."""
        if ref.startswith('#/$defs/'):
            return ref[len('#/$defs/') :]
        return ref.split('/')[-1]

    def add_typeddict(self, name: str, definition: str) -> None:
        """Add a TypedDict definition."""
        self._typeddict_defs[name] = definition

    def get_typeddict_definitions(self) -> list[str]:
        """Get all TypedDict definitions in order."""
        return list(self._typeddict_defs.values())

    def generate_name(self, base: str) -> str:
        """Generate a unique TypedDict name."""
        self._counter += 1
        return f'{_to_pascal_case(self.tool_name)}{_to_pascal_case(base)}'


def _to_pascal_case(s: str) -> str:
    """Convert a string to PascalCase."""
    s = re.sub(r'[^a-zA-Z0-9]', '_', s)
    parts = s.split('_')
    return ''.join(part.capitalize() for part in parts if part)


def _schema_to_params(schema: dict[str, Any], context: _ConversionContext) -> list[str]:
    """Convert a JSON schema to a list of parameter strings."""
    if '$defs' in schema:
        context.set_defs(schema['$defs'])
        _process_defs(schema['$defs'], context)

    properties = schema.get('properties', {})
    required = set(schema.get('required', []))

    params: list[str] = []
    required_params: list[str] = []
    optional_params: list[str] = []

    for prop_name, prop_schema in properties.items():
        type_str = _schema_to_type(prop_schema, context, prop_name)
        is_required = prop_name in required

        if 'default' in prop_schema:
            default_str = _format_schema_default(prop_schema['default'])
            optional_params.append(f'{prop_name}: {type_str} = {default_str}')
        elif is_required:
            required_params.append(f'{prop_name}: {type_str}')
        else:
            optional_params.append(f'{prop_name}: {type_str} | None = None')

    params.extend(required_params)
    params.extend(optional_params)
    return params


def _process_defs(defs: dict[str, dict[str, Any]], context: _ConversionContext) -> None:
    """Process $defs and generate TypedDict definitions."""
    for def_name, def_schema in defs.items():
        if def_schema.get('type') == 'object' and 'properties' in def_schema:
            _generate_typeddict(def_name, def_schema, context)


def _generate_typeddict(name: str, schema: dict[str, Any], context: _ConversionContext) -> str:
    """Generate a TypedDict definition for an object schema."""
    properties = schema.get('properties', {})
    required = set(schema.get('required', []))

    lines = [f'class {name}(TypedDict):']

    if not properties:
        lines.append('    pass')
    else:
        for prop_name, prop_schema in properties.items():
            type_str = _schema_to_type(prop_schema, context, prop_name)
            is_required = prop_name in required
            if not is_required:
                type_str = f'NotRequired[{type_str}]'
            desc = prop_schema.get('description', '')
            if desc:
                lines.append(f'    {prop_name}: {type_str}  # {desc}')
            else:
                lines.append(f'    {prop_name}: {type_str}')

    definition = '\n'.join(lines)
    context.add_typeddict(name, definition)
    return name


def _schema_to_type(schema: dict[str, Any], context: _ConversionContext, prop_name: str = '') -> str:
    """Convert a JSON schema to a Python type string."""
    if '$ref' in schema:
        return context.get_ref_name(schema['$ref'])

    if 'anyOf' in schema:
        return _handle_any_of(schema['anyOf'], context, prop_name)

    if 'oneOf' in schema:
        return _handle_any_of(schema['oneOf'], context, prop_name)

    if 'allOf' in schema:
        if len(schema['allOf']) == 1:
            return _schema_to_type(schema['allOf'][0], context, prop_name)
        return 'Any'

    if 'const' in schema:
        return f'Literal[{repr(schema["const"])}]'

    if 'enum' in schema:
        enum_values = ', '.join(repr(v) for v in schema['enum'])
        return f'Literal[{enum_values}]'

    schema_type = schema.get('type')
    return _type_from_schema_type(schema_type, schema, context, prop_name)


def _type_from_schema_type(
    schema_type: str | list[str] | None, schema: dict[str, Any], context: _ConversionContext, prop_name: str
) -> str:
    """Convert a schema type to Python type string."""
    if schema_type == 'string':
        return 'str'
    if schema_type == 'integer':
        return 'int'
    if schema_type == 'number':
        return 'float'
    if schema_type == 'boolean':
        return 'bool'
    if schema_type == 'null':
        return 'None'
    if schema_type == 'array':
        return _handle_array_type(schema, context, prop_name)
    if schema_type == 'object':
        return _handle_object_type(schema, context, prop_name)
    if isinstance(schema_type, list):
        return _handle_type_list(schema_type)
    return 'Any'


def _handle_array_type(schema: dict[str, Any], context: _ConversionContext, prop_name: str) -> str:
    """Handle array type schema."""
    items = schema.get('items', {})
    if items:
        item_type = _schema_to_type(items, context, f'{prop_name}Item')
        return f'list[{item_type}]'
    return 'list[Any]'


def _handle_object_type(schema: dict[str, Any], context: _ConversionContext, prop_name: str) -> str:
    """Handle object type schema."""
    if 'properties' in schema:
        td_name = context.generate_name(prop_name) if prop_name else f'{context.tool_name}Data'
        return _generate_typeddict(td_name, schema, context)
    if 'additionalProperties' in schema:
        additional = schema['additionalProperties']
        if additional is True:
            return 'dict[str, Any]'
        if isinstance(additional, dict):
            additional = cast(dict[str, Any], additional)
            value_type = _schema_to_type(additional, context, f'{prop_name}Value')
            return f'dict[str, {value_type}]'
    return 'dict[str, Any]'


def _handle_type_list(schema_type: list[str]) -> str:
    """Handle list of types (e.g., ['string', 'null'])."""
    types = [_json_type_to_python(str(t)) for t in schema_type]
    types = [t for t in types if t]
    if len(types) == 2 and 'None' in types:
        non_none = [t for t in types if t != 'None'][0]
        return f'{non_none} | None'
    return ' | '.join(types) if types else 'Any'


def _handle_any_of(schemas: list[dict[str, Any]], context: _ConversionContext, prop_name: str) -> str:
    """Handle anyOf/oneOf schemas."""
    types: list[str] = []
    has_null = False

    for s in schemas:
        if s.get('type') == 'null':
            has_null = True
        else:
            types.append(_schema_to_type(s, context, prop_name))

    if has_null and len(types) == 1:
        return f'{types[0]} | None'
    elif has_null:
        return ' | '.join(types) + ' | None'
    elif len(types) == 1:
        return types[0]
    else:
        return ' | '.join(types)


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


def _format_schema_default(value: Any) -> str:
    """Format a default value from a JSON schema."""
    if value is None:
        return 'None'
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return repr(value)  # pyright: ignore[reportUnknownArgumentType]
    if isinstance(value, dict):
        return repr(value)  # pyright: ignore[reportUnknownArgumentType]
    return repr(value)


def validate_signature(signature_str: str) -> bool:
    """Validate that a signature string is valid Python syntax.

    Args:
        signature_str: The signature string to validate.

    Returns:
        True if valid, raises SyntaxError if invalid.
    """
    ast.parse(signature_str)
    return True
