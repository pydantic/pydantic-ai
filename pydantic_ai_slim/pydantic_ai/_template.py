"""Internal template helpers."""

from __future__ import annotations

import inspect
from typing import Any, get_args, get_origin

from pydantic import TypeAdapter

from pydantic_ai._utils import get_function_type_hints
from pydantic_ai.template import TemplateStr


def validate_from_spec_args(
    cls: type[Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    validation_context: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Validate `from_spec` arguments, resolving `TemplateStr` types via Pydantic.

    Inspects the `from_spec` method's type hints to find parameters that accept
    `TemplateStr`. For those parameters, values are validated through Pydantic's
    `TypeAdapter`, which invokes `TemplateStr.__get_pydantic_core_schema__`
    to automatically compile template strings (containing `{{`) into `TemplateStr`
    instances using the `deps_type`/`deps_schema` from the validation context.
    """
    try:
        hints = get_function_type_hints(cls.from_spec)
    except Exception:
        return args, kwargs

    hints.pop('return', None)
    if not any(_hint_contains_template_str(h) for h in hints.values()):
        return args, kwargs

    sig = inspect.signature(cls.from_spec)
    params = [p for p in sig.parameters.values() if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]

    new_args = list(args)
    new_kwargs = dict(kwargs)

    for i, param in enumerate(params):
        hint = hints.get(param.name)
        if hint is None or not _hint_contains_template_str(hint):
            continue

        ta = TypeAdapter(hint)
        if i < len(args):
            new_args[i] = ta.validate_python(args[i], context=validation_context)
        elif param.name in kwargs:
            new_kwargs[param.name] = ta.validate_python(kwargs[param.name], context=validation_context)

    return tuple(new_args), new_kwargs


def _hint_contains_template_str(hint: Any) -> bool:
    """Check if a type hint includes TemplateStr."""
    if hint is TemplateStr or get_origin(hint) is TemplateStr:
        return True
    args = get_args(hint)
    if args:
        return any(_hint_contains_template_str(a) for a in args)
    return False
