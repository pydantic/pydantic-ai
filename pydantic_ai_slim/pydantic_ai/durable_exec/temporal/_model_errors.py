"""Carry model errors across the activity boundary with their original type.

A model request runs in an activity, so an error it raises reaches the workflow as Temporal's
`ActivityError` wrapping an `ApplicationError`, rather than as the
[`ModelAPIError`][pydantic_ai.exceptions.ModelAPIError] the model raised. Workflow-side code that
handles model errors, like an `on_model_request_error` hook or the `Fallback` capability, would
never see one. So the activity raises the `ApplicationError` itself, with the model error encoded in
its details, and the workflow raises a rebuilt copy of the original error.

The error is encoded from its pickling protocol (`__reduce__`), which every `ModelAPIError` subclass
in Pydantic AI implements, so a subclass and its fields cross without a per-class table. Only Pydantic
AI's own classes are rebuilt; an application-defined subclass, or an error whose fields can't be
encoded, fails as Temporal's `ActivityError` as before.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, cast

from pydantic_core import to_jsonable_python
from temporalio.exceptions import ActivityError, ApplicationError

from ...exceptions import ModelAPIError

_DETAILS_KEY = 'pydantic_ai_model_error'
"""Marks the `ApplicationError` detail that holds an encoded model error."""


def _encode(error: ModelAPIError) -> dict[str, Any] | None:
    """The JSON-compatible form of `error`, or `None` if it can't be rebuilt in the workflow."""
    error_type = type(error)
    if not error_type.__module__.startswith('pydantic_ai.'):
        # Only Pydantic AI's own error classes are rebuilt. `pydantic_ai` is passed through the workflow
        # sandbox, so each of its classes is one class on both sides; an application module is re-imported
        # per sandbox, so its classes can't be relied on to resolve to the one the activity raised.
        return None
    reduced = error.__reduce__()
    if not isinstance(reduced, tuple) or len(reduced) < 2 or reduced[0] is not error_type:
        return None
    args = reduced[1]
    state = reduced[2] if len(reduced) > 2 else None
    return {
        'module': error_type.__module__,
        'qualname': error_type.__qualname__,
        # A response body can hold anything the provider sent; whatever isn't JSON crosses as a string.
        'args': to_jsonable_python(list(args), fallback=str),
        'state': to_jsonable_python(state, fallback=str) if state is not None else None,
    }


def _model_error_class(module_name: object, qualname: object) -> type[ModelAPIError] | None:
    """The loaded Pydantic AI model error class with this name, if any.

    Looked up among the subclasses already loaded rather than imported by name, so details that came
    off the wire never decide what gets imported.
    """
    pending: list[type[ModelAPIError]] = [ModelAPIError]
    while pending:
        error_type = pending.pop()
        if error_type.__module__ == module_name and error_type.__qualname__ == qualname:
            return error_type
        pending.extend(error_type.__subclasses__())
    return None


def _decode(encoded: dict[str, Any]) -> ModelAPIError | None:
    """Rebuild the model error `_encode` encoded, or `None` if its class can't be found."""
    module_name = encoded.get('module')
    if not isinstance(module_name, str) or not module_name.startswith('pydantic_ai.'):
        return None
    if (error_type := _model_error_class(module_name, encoded.get('qualname'))) is None:
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


@contextmanager
def model_errors_as_application_errors() -> Generator[None]:
    """In a model activity, raise a model error as an `ApplicationError` the workflow can rebuild it from.

    The `ApplicationError`'s `type` is the error's class name, which is what Temporal would have used
    for the raw exception, so the activity's retry policy treats it exactly as before.
    """
    try:
        yield
    except ModelAPIError as error:
        # Checked here, where the error's class is the same as in the workflow, so the workflow can
        # count on rebuilding whatever it receives. Anything that can't cross fails as it always did.
        try:
            encoded = _encode(error)
        except Exception:
            encoded = None
        if encoded is None or _decode(encoded) is None:
            raise
        raise ApplicationError(str(error), {_DETAILS_KEY: encoded}, type=type(error).__name__) from error


@contextmanager
def rebuilt_model_errors() -> Generator[None]:
    """In the workflow, re-raise a model error a model activity failed with as its original type.

    Any other activity failure, including one from an error that wasn't a model error, is raised
    unchanged.
    """
    try:
        yield
    except ActivityError as activity_error:
        cause = activity_error.cause
        if isinstance(cause, ApplicationError) and (rebuilt := _rebuild(cause)) is not None:
            raise rebuilt from activity_error
        raise


def _rebuild(application_error: ApplicationError) -> ModelAPIError | None:
    details = application_error.details
    if not details or not isinstance(details[0], dict):
        return None
    encoded = cast(dict[str, Any], details[0]).get(_DETAILS_KEY)
    if not isinstance(encoded, dict):
        return None
    return _decode(cast(dict[str, Any], encoded))
