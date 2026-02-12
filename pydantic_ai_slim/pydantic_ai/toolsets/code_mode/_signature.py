"""Convert JSON schemas to Python function signatures.

This module provides utilities to convert JSON schemas (like those used in tool definitions)
into human-readable Python function signature strings, which LLMs can understand more easily
than raw JSON schemas.
"""

from __future__ import annotations

import dataclasses
import types
from collections.abc import Callable
from dataclasses import dataclass, field
from inspect import Parameter, Signature as InspectSignature, signature
from typing import Any, Union, get_origin

from pydantic import BaseModel, TypeAdapter
from typing_extensions import get_type_hints, is_typeddict

from ..._run_context import RunContext


def _is_named_type(t: Any) -> bool:
    """Check if a type is a `BaseModel`, dataclass, or `TypedDict` that needs a definition."""
    if not isinstance(t, type):
        return False
    return issubclass(t, BaseModel) or dataclasses.is_dataclass(t) or is_typeddict(t)  # pyright: ignore[reportUnknownArgumentType]


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

    typeddicts: list[str] = field(default_factory=list)
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
    from ._schema_types import _schema_to_signature

    return _schema_to_signature(
        name,
        parameters_json_schema,
        description,
        return_type,
        return_json_schema,
        namespace_defs=namespace_defs,
    )


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
    s = repr(t)
    return s.replace('typing.', '').replace('typing_extensions.', '')


# =============================================================================
# Function signature builder
# =============================================================================


def _collect_typeddicts_from_annotation(
    annotation: Any,
    typeddicts: dict[str, str],
    tool_name: str,
    path: str = '',
) -> None:
    """Recursively collect TypedDict definitions from type annotations."""
    from ._schema_types import _generate_typeddict_str, _schema_type_to_str

    if annotation is None or annotation is type(None):
        return

    if _is_named_type(annotation):
        type_name = annotation.__name__
        if type_name not in typeddicts:
            schema = _get_schema_from_type(annotation)
            schema_defs = schema.get('$defs', {})

            def to_type(s: dict[str, Any], p: str) -> str:
                return _schema_type_to_str(s, schema_defs, typeddicts, tool_name, p)

            # Process any $defs first
            for def_name, def_schema in schema_defs.items():
                if def_name not in typeddicts and def_schema.get('type') == 'object' and 'properties' in def_schema:
                    typeddicts[def_name] = _generate_typeddict_str(def_name, def_schema, to_type, path)

            # Then process the main schema
            if schema.get('type') == 'object' and 'properties' in schema:
                typeddicts[type_name] = _generate_typeddict_str(type_name, schema, to_type, path)
            elif '$ref' in schema:
                ref_name = (
                    schema['$ref'][8:] if schema['$ref'].startswith('#/$defs/') else schema['$ref'].split('/')[-1]
                )
                if ref_name in schema_defs and ref_name not in typeddicts:
                    ref_schema = schema_defs[ref_name]
                    if ref_schema.get('type') == 'object' and 'properties' in ref_schema:
                        typeddicts[ref_name] = _generate_typeddict_str(ref_name, ref_schema, to_type, path)
        return

    origin = get_origin(annotation)
    args = getattr(annotation, '__args__', None)
    if origin is not None and args:
        for arg in args:
            _collect_typeddicts_from_annotation(arg, typeddicts, tool_name, path)


def _build_function_params(
    sig: InspectSignature,
    type_hints: dict[str, Any],
    typeddicts: dict[str, str],
    tool_name: str,
) -> list[str]:
    """Build parameter strings from a function's signature and type hints."""
    params: list[str] = []
    for i, (param_name, param) in enumerate(sig.parameters.items()):
        annotation = type_hints.get(param_name)

        if i == 0 and annotation is not None and (annotation is RunContext or get_origin(annotation) is RunContext):
            continue

        if annotation is not None:
            _collect_typeddicts_from_annotation(annotation, typeddicts, tool_name, param_name)

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
    return params


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
        type_hints = get_type_hints(func, include_extras=True)
    except Exception:
        type_hints = {}

    typeddicts: dict[str, str] = {}
    params = _build_function_params(sig, type_hints, typeddicts, name)

    return_annotation = type_hints.get('return')
    if return_annotation is not None:
        _collect_typeddicts_from_annotation(return_annotation, typeddicts, name, 'Return')

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
