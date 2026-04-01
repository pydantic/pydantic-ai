from __future__ import annotations


class PydanticEvalsDeprecationWarning(UserWarning):
    """Warning emitted when a deprecated Pydantic Evals API is used.

    Inherits from `UserWarning` instead of `DeprecationWarning` so that
    deprecations are visible by default at runtime, following the approach
    described in https://sethmlarson.dev/deprecations-via-warnings-dont-work-for-python-libraries.
    """
