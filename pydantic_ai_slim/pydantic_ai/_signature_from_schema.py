"""Convert JSON schemas to Python function signatures.

This module provides utilities to convert JSON schemas (like those used in tool definitions)
into human-readable Python function signature strings, which LLMs can understand more easily
than raw JSON schemas.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import inspect
import json
import re
import types
from collections.abc import Callable
from dataclasses import dataclass
from inspect import Parameter, signature
from typing import Any, Union, cast, get_origin

from pydantic import BaseModel, TypeAdapter
from pydantic._internal import _typing_extra

from ._run_context import RunContext


def _is_run_context(annotation: Any) -> bool:
    """Check if an annotation is RunContext or RunContext[T]."""
    if annotation is RunContext:
        return True
    origin = getattr(annotation, '__origin__', None)
    return origin is RunContext


def _is_typeddict(t: Any) -> bool:
    """Check if a type is a `TypedDict` subclass."""
    return isinstance(t, type) and hasattr(t, '__annotations__') and hasattr(t, '__total__')


def _is_named_type(t: Any) -> bool:
    """Check if a type is a `BaseModel`, dataclass, or `TypedDict` that needs a definition."""
    if t is None or t is type(None) or not isinstance(t, type):
        return False
    else:
        return issubclass(t, BaseModel) or dataclasses.is_dataclass(t) or _is_typeddict(t)  # pyright: ignore[reportUnknownArgumentType]


def _get_schema_from_type(t: Any) -> dict[str, Any]:
    """Extract JSON schema from a `BaseModel`, dataclass, or `TypedDict`."""
    if isinstance(t, type) and issubclass(t, BaseModel):
        return t.model_json_schema()
    return TypeAdapter(t).json_schema()  # pyright: ignore[reportUnknownArgumentType]


def _collect_named_types_from_annotation(annotation: Any, collected: dict[str, Any]) -> None:
    """Recursively collect named types from an annotation."""
    if annotation is None or annotation is type(None):
        return

    if _is_named_type(annotation):
        name = annotation.__name__
        if name not in collected:
            collected[name] = annotation
            schema = _get_schema_from_type(annotation)
            if '$defs' in schema:
                for def_name, def_schema in schema['$defs'].items():
                    if def_name not in collected:
                        collected[def_name] = def_schema
        return

    origin = get_origin(annotation)
    args = getattr(annotation, '__args__', None)

    if origin is not None and args:
        for arg in args:
            _collect_named_types_from_annotation(arg, collected)


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
    *,
    include_return_type: bool = True,
) -> SignatureResult:
    """Generate a Python signature string from an actual function.

    Uses inspect.signature() and typing.get_type_hints() to reconstruct
    the function signature with type annotations.

    Args:
        func: The function to generate a signature for.
        name: Override the function name. If None, uses func.__name__.
        description: Optional description to include as a docstring.
        include_return_type: Whether to include the return type annotation.

    Returns:
        A SignatureResult containing the signature string and any TypedDict definitions.
    """
    name = name or func.__name__
    sig = signature(func)

    try:
        type_hints = _typing_extra.get_function_type_hints(func)
    except Exception:
        type_hints = {}

    collected_types: dict[str, Any] = {}

    params: list[str] = []

    for i, (param_name, param) in enumerate(sig.parameters.items()):
        annotation = type_hints.get(param_name)

        if i == 0 and annotation is not None and _is_run_context(annotation):
            continue

        if annotation is not None:
            _collect_named_types_from_annotation(annotation, collected_types)

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
    if return_annotation is not None:
        _collect_named_types_from_annotation(return_annotation, collected_types)

    if include_return_type:
        return_type_str = _format_annotation(return_annotation) if return_annotation else 'Any'
    else:
        return_type_str = 'Any'

    is_async = _is_async_function(func)
    prefix = 'async def' if is_async else 'def'

    signature_line = f'{prefix} {name}({", ".join(params)}) -> {return_type_str}'

    typeddict_defs = _generate_typeddict_defs_from_collected(collected_types, name)

    if description:
        docstring = _format_docstring(description)
        signature_str = f'{signature_line}:\n{docstring}\n    raise NotImplementedError()'
    else:
        signature_str = f'{signature_line}:\n    raise NotImplementedError()'

    return SignatureResult(signature=signature_str, typeddict_defs=typeddict_defs, return_type=return_type_str)


def _generate_typeddict_defs_from_collected(collected_types: dict[str, Any], tool_name: str) -> list[str]:
    """Generate `TypedDict` definitions from collected named types."""
    if not collected_types:
        return []

    context = _ConversionContext(tool_name)

    for type_name, type_or_schema in collected_types.items():
        if isinstance(type_or_schema, dict):
            type_or_schema = cast(dict[str, Any], type_or_schema)
            if type_or_schema.get('type') == 'object' and 'properties' in type_or_schema:
                context.set_defs({type_name: type_or_schema})
                _generate_typeddict(type_name, type_or_schema, context)
        elif _is_named_type(type_or_schema):
            schema = _get_schema_from_type(type_or_schema)
            if '$defs' in schema:
                context.set_defs(schema['$defs'])
                _process_defs(schema['$defs'], context)
            if '$ref' in schema:
                # TODO: Review by codex: For David's reference - a schema like {'$ref': '#/$defs/User'}
                # skips generation entirely, so no TypedDict is produced for User unless it appears elsewhere.
                pass
            elif schema.get('type') == 'object' and 'properties' in schema:
                _generate_typeddict(type_name, schema, context)

    return context.get_typeddict_definitions()


def _is_async_function(func: Callable[..., Any]) -> bool:
    """Check if a function is async."""
    if inspect.iscoroutinefunction(func):  # -> Changed from asyncio.iscoroutinefunction which got deprecated
        return True
    if inspect.ismethod(func):
        return inspect.iscoroutinefunction(func.__func__)
    return False


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


def _format_default(value: Any) -> str:
    if value is None:
        return 'None'
    elif isinstance(value, (bool, int, float)):
        return str(value)
    elif isinstance(value, (str, list, dict)):
        return repr(value)  # pyright: ignore[reportUnknownArgumentType]
    else:
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
    *,
    namespace_defs: bool = False,
) -> SignatureResult:
    """Convert a JSON schema to a Python function signature string.

    Args:
        name: The function name.
        parameters_json_schema: The JSON schema for the function's parameters.
        description: Optional function description to include as docstring.
        return_type: The return type annotation string. Defaults to 'Any'.
        return_json_schema: Optional JSON schema for the return value.
        namespace_defs: Whether to prefix $defs names to avoid param/return collisions.

    Returns:
        A SignatureResult containing the signature string and any TypedDict definitions.
    """
    context = _ConversionContext(name)
    params = _schema_to_params(parameters_json_schema, context, namespace_defs=namespace_defs)

    resolved_return_type = return_type
    if return_json_schema is not None and return_type == 'Any':
        if return_json_schema.get('$defs'):
            return_defs = return_json_schema['$defs']
            if namespace_defs:
                return_defs = _namespace_defs(return_defs, 'Return')
                return_json_schema = _namespace_schema_refs(return_json_schema, 'Return')
            context.set_defs(return_defs)
            _process_defs(return_defs, context)
        assert return_json_schema is not None
        resolved_return_type = _schema_to_type(return_json_schema, context, 'Return')

    typeddict_defs = context.get_typeddict_definitions()

    signature_line = f'def {name}({", ".join(params)}) -> {resolved_return_type}'

    if description:
        docstring = _format_docstring(description)
        signature_str = f'{signature_line}:\n{docstring}\n    raise NotImplementedError()'
    else:
        signature_str = f'{signature_line}:\n    raise NotImplementedError()'

    if return_json_schema is not None and resolved_return_type == 'Any':
        return_schema_blob = json.dumps(return_json_schema, indent=2)
        return_schema_note = f'\n\nReturn schema:\n{return_schema_blob}'
        if description:
            signature_str = f'{signature_line}:\n{_format_docstring(description + return_schema_note)}\n    raise NotImplementedError()'
        else:
            signature_str = (
                f'{signature_line}:\n{_format_docstring(return_schema_note.strip())}\n    raise NotImplementedError()'
            )

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
        # We can namespace $defs via signature_from_schema(..., namespace_defs=True) to avoid
        # collisions (e.g., ParamUser vs ReturnUser), but this changes the generated type names.
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
        # TODO: Review by codex: For David's reference - the counter is unused, so repeated
        # object properties like `address` in different branches can collide as the same name.
        return f'{_to_pascal_case(self.tool_name)}{_to_pascal_case(base)}'


def _to_pascal_case(s: str) -> str:
    """Convert a string to PascalCase."""
    s = re.sub(r'[^a-zA-Z0-9]', '_', s)
    parts = s.split('_')
    return ''.join(part.capitalize() for part in parts if part)


def _schema_to_params(
    schema: dict[str, Any],
    context: _ConversionContext,
    *,
    namespace_defs: bool = False,
) -> list[str]:
    """Convert a JSON schema to a list of parameter strings."""
    if '$defs' in schema:
        param_defs = schema['$defs']
        if namespace_defs:
            param_defs = _namespace_defs(param_defs, 'Param')
            schema = _namespace_schema_refs(schema, 'Param')
        context.set_defs(param_defs)
        _process_defs(param_defs, context)

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
            # TODO: Review by codex: For David's reference - this always appends `| None`,
            # even when schema already allows null (e.g., type: ["string", "null"]).
            optional_params.append(f'{prop_name}: {type_str} | None = None')

    params.extend(required_params)
    params.extend(optional_params)
    return params


def _namespace_schema_refs(schema: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Return a copy of a schema with $ref names prefixed."""

    def _rename(schema_part: dict[str, Any]) -> dict[str, Any]:
        updated: dict[str, Any] = {}
        for key, value in schema_part.items():
            if key == '$ref' and isinstance(value, str) and value.startswith('#/$defs/'):
                ref_name = value[len('#/$defs/') :]
                updated[key] = f'#/$defs/{prefix}{ref_name}'
            elif isinstance(value, dict):
                value = cast(dict[str, Any], value)
                updated[key] = _rename(value)
            elif isinstance(value, list):
                value = cast(list[dict[str, Any]], value)
                updated[key] = [_rename(item) if isinstance(item, dict) else item for item in value]
            else:
                updated[key] = value
        return updated

    return _rename(copy.deepcopy(schema))


def _namespace_defs(defs: dict[str, dict[str, Any]], prefix: str) -> dict[str, dict[str, Any]]:
    """Return a new $defs dict with prefixed names and updated $refs."""

    def _rename_ref(schema: dict[str, Any], old: str, new: str) -> dict[str, Any]:
        if isinstance(schema, dict):
            updated: dict[str, Any] = {}
            for key, value in schema.items():
                if key == '$ref' and value == f'#/$defs/{old}':
                    updated[key] = f'#/$defs/{new}'
                elif isinstance(value, dict):
                    value = cast(dict[str, Any], value)
                    updated[key] = _rename_ref(value, old, new)
                elif isinstance(value, list):
                    value = cast(list[dict[str, Any]], value)
                    updated[key] = [_rename_ref(item, old, new) if isinstance(item, dict) else item for item in value]
                else:
                    updated[key] = value
            return updated
        return schema

    renamed: dict[str, dict[str, Any]] = {}
    for name, schema in defs.items():
        new_name = f'{prefix}{name}'
        renamed[new_name] = copy.deepcopy(schema)

    rename_map = {name: f'{prefix}{name}' for name in defs}
    for old_name, new_name in rename_map.items():
        renamed = {k: _rename_ref(v, old_name, new_name) for k, v in renamed.items()}

    return renamed


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
        # TODO: Review by codex: For David's reference - allOf with multiple entries (e.g.,
        # [{"$ref": "#/$defs/User"}, {"type": "object", "properties": {"age": {"type": "integer"}}}])
        # loses information and returns Any instead of merging.
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
        # TODO: Review by codex: For David's reference - schema like {"type": ["object", "null"],
        # "properties": {"name": {"type": "string"}}} ignores properties and returns `dict | None`.
        return _handle_type_list(schema_type)
    return 'Any'


def _handle_array_type(schema: dict[str, Any], context: _ConversionContext, prop_name: str) -> str:
    """Handle array type schema."""
    items = schema.get('items', {})
    if items:
        # TODO: Review by codex: For David's reference - tuple schemas like
        # {"items": [{"type": "string"}, {"type": "integer"}]} are lists, not dicts, and
        # will error or mis-format because `_schema_to_type` expects a dict.
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
