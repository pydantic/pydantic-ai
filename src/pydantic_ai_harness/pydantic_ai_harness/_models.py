"""Private helpers for narrowing a run's model."""

from __future__ import annotations

from typing import Any

from typing_extensions import TypeIs

from pydantic_ai.models import Model


def is_request_response_model(model: object) -> TypeIs[Model[Any]]:
    """Whether `model` is a request-response `Model` a nested agent can run on, keeping its client type."""
    return isinstance(model, Model)
