"""Encode a model error as JSON so it can be rebuilt with its original type on the other side of a durable boundary.

A durable unit that runs on another worker reports its failure to the durable container through the
engine, which carries an error message, not the exception. Workflow-side code that handles model
errors, like an `on_model_request_error` hook or the `ModelFallback` capability, needs the
[`ModelAPIError`][pydantic_ai.exceptions.ModelAPIError] the model raised, so engines encode it in
what they report and rebuild it in the container.

The error is encoded from its pickling protocol (`__reduce__`), which every `ModelAPIError` subclass
in Pydantic AI implements, so a subclass and its fields cross without a per-class table. Only Pydantic
AI's own classes are rebuilt; an application-defined subclass, or an error whose fields can't be
encoded, is reported as the engine's own error.
"""

from __future__ import annotations

from typing import Any, cast

from pydantic_core import to_jsonable_python

from ..exceptions import ModelAPIError, ModelHTTPError, _CategorizedHTTPError  # pyright: ignore[reportPrivateUsage]

__all__ = ('decode_model_error', 'encode_model_error')


def encode_model_error(error: ModelAPIError) -> dict[str, Any] | None:
    """The JSON-compatible form of `error`, or `None` if it can't be rebuilt where it's received."""
    error_type = type(error)
    if not error_type.__module__.startswith('pydantic_ai.'):
        # Only Pydantic AI's own error classes are rebuilt: each is one class wherever the error is received. An
        # application's classes can't be relied on to be, e.g. in a Temporal workflow sandbox, which re-imports
        # application modules.
        return None
    # An HTTP error with a category is named after its category, so it's marked to be rebuilt as the HTTP class with
    # that name rather than the category itself, from the arguments it takes as a `ModelHTTPError`.
    categorized_http = isinstance(error, _CategorizedHTTPError)
    reduced = ModelHTTPError.__reduce__(error) if categorized_http else error.__reduce__()
    if not isinstance(reduced, tuple) or len(reduced) < 2 or reduced[0] is not error_type:
        return None
    args = reduced[1]
    state = reduced[2] if len(reduced) > 2 else None
    if isinstance(state, dict):
        # What Pydantic AI attached to the error isn't JSON. An engine carries what has to cross, like the response
        # a stream produced before failing, next to the encoded error.
        state = {
            key: value for key, value in cast(dict[str, Any], state).items() if not key.startswith('_pydantic_ai_')
        }
    encoded: dict[str, Any] = {
        'module': error_type.__module__,
        'qualname': error_type.__qualname__,
        # A response body can hold anything the provider sent; whatever isn't JSON crosses as a string.
        'args': to_jsonable_python(list(args), fallback=str),
        'state': to_jsonable_python(state, fallback=str) if state is not None else None,
    }
    if categorized_http:
        encoded['categorized_http'] = True
    return encoded


def _model_error_class(module_name: object, qualname: object, *, categorized_http: bool) -> type[ModelAPIError] | None:
    """The loaded Pydantic AI model error class with this name, if any.

    Looked up among the subclasses already loaded rather than imported by name, so details that came
    off the wire never decide what gets imported. An HTTP error with a category has its category's name,
    so `categorized_http` says which of the two to find.
    """
    pending: list[type[ModelAPIError]] = [ModelAPIError]
    while pending:
        error_type = pending.pop()
        if (
            error_type.__module__ == module_name
            and error_type.__qualname__ == qualname
            and issubclass(error_type, _CategorizedHTTPError) == categorized_http
        ):
            return error_type
        pending.extend(error_type.__subclasses__())
    return None


def decode_model_error(encoded: dict[str, Any]) -> ModelAPIError | None:
    """Rebuild the model error `encode_model_error` encoded, or `None` if its class can't be found."""
    module_name = encoded.get('module')
    if not isinstance(module_name, str) or not module_name.startswith('pydantic_ai.'):
        return None
    categorized_http = encoded.get('categorized_http') is True
    if (
        error_type := _model_error_class(module_name, encoded.get('qualname'), categorized_http=categorized_http)
    ) is None:
        return None
    args: list[Any] = encoded.get('args') or []
    try:
        error = error_type(*args)
        if (state := encoded.get('state')) is not None:
            error.__setstate__(state)
    except Exception:
        # A subclass whose constructor doesn't take what its `__reduce__` returns can't be rebuilt.
        return None
    return error
