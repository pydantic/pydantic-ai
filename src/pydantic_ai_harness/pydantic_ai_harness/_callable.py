"""Detect a leading run context without resolving unrelated callback annotations."""

from __future__ import annotations

import functools
import inspect
import re
from collections.abc import Callable, Sequence
from types import UnionType
from typing import Annotated, Any, ForwardRef, Union, get_args, get_origin

from pydantic_ai.tools import RunContext

_RUN_CONTEXT_NAME = re.compile(r'(?<!\w)RunContext\b')


def takes_context(func: Callable[..., object], *, parameters: Sequence[inspect.Parameter] | None = None) -> bool | None:
    """Return whether a callback takes `(ctx, value)`, or `None` when either shape is ambiguous.

    Callers choose their policy for an unannotated or unresolvable optional-argument signature.
    Only the first annotation is resolved; other annotations may be imported only for typing.
    Callers that already inspected the signature can reuse its parameters.
    """
    if parameters is None:
        try:
            parameters = tuple(inspect.signature(func).parameters.values())
        except ValueError:
            return False
    positional = [
        parameter
        for parameter in parameters
        if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if len(positional) < 2:
        if not positional or not any(parameter.kind is inspect.Parameter.VAR_POSITIONAL for parameter in parameters):
            return False
    elif positional[1].default is inspect.Parameter.empty:
        return True
    annotation = _resolve(func, positional[0].annotation)
    while get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    either = annotation is inspect.Parameter.empty or annotation is Any or annotation is object
    unresolved = isinstance(annotation, str) or _has_forward_ref(annotation)
    if either or (unresolved and not is_run_context(annotation)):
        return None
    return is_run_context(annotation)


def _resolve(func: Callable[..., object], annotation: object) -> object:
    """Evaluate a string annotation in the callback's module, retaining unresolved strings."""
    if not isinstance(annotation, str):
        return annotation
    target = inspect.unwrap(func.func if isinstance(func, functools.partial) else func)
    # A callable instance's annotations come from its class's `__call__`.
    namespace = getattr(target, '__globals__', None) or getattr(type(target).__call__, '__globals__', {})
    try:
        return eval(annotation, namespace)  # same as `typing.get_type_hints`
    except Exception:
        return annotation


def is_run_context(annotation: object) -> bool:
    """Recognize `RunContext`, including unions, `Annotated` and unresolved references."""
    if isinstance(annotation, str):
        return _names_run_context(annotation)
    if isinstance(annotation, ForwardRef):
        return _names_run_context(annotation.__forward_arg__)
    if annotation is RunContext:
        return True
    origin = get_origin(annotation)
    if origin is RunContext:
        return True
    if origin is Annotated:
        return is_run_context(get_args(annotation)[0])
    if origin is Union or origin is UnionType:
        return any(is_run_context(arg) for arg in get_args(annotation) if arg is not type(None))
    return False


def _has_forward_ref(annotation: object) -> bool:
    """Whether an annotation holds a quoted name that was never resolved."""
    return isinstance(annotation, ForwardRef) or any(_has_forward_ref(arg) for arg in get_args(annotation))


def _names_run_context(annotation: str) -> bool:
    """Match `RunContext` in qualified or compound annotations, but not in names such as `NotRunContext`."""
    return _RUN_CONTEXT_NAME.search(annotation) is not None
