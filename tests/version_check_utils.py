"""A stand-in for the version check's endpoint, shared by the tests that let the check run."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx2
import pytest

import pydantic_ai._version_check as _version_check


def install_transport(
    monkeypatch: pytest.MonkeyPatch, handler: Callable[[httpx2.Request], httpx2.Response]
) -> list[httpx2.Request]:
    """Answer the version check's request with `handler`, returning the list every request lands in."""
    requests: list[httpx2.Request] = []

    def record(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return handler(request)

    def get(url: str, **kwargs: Any) -> httpx2.Response:
        with httpx2.Client(transport=httpx2.MockTransport(record)) as client:
            return client.get(url, **kwargs)

    monkeypatch.setattr(_version_check.httpx2, 'get', get)
    return requests
