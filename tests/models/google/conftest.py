"""Shared fixtures for Google model tests."""

from __future__ import annotations as _annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import pytest

from ...conftest import try_import

if TYPE_CHECKING:
    from vcr import VCR

with try_import() as imports_successful:
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

    GoogleModelFactory = Callable[..., GoogleModel]


def _function_calling_mode(request: Any) -> str | None:
    """Extract `toolConfig.functionCallingConfig.mode` from a Google request body, or `None`.

    `request` is a VCR `Request` (untyped in practice), so it's typed loosely here.
    """
    body: Any = request.body
    data: dict[str, Any] = json.loads(body) if body else {}
    tool_config: Any = data.get('toolConfig') or {}
    fc_config: Any = tool_config.get('functionCallingConfig') or {}
    mode: Any = fc_config.get('mode')
    return mode if isinstance(mode, str) else None


def _match_function_calling_mode(request1: Any, request2: Any) -> None:
    assert _function_calling_mode(request1) == _function_calling_mode(request2)


def pytest_recording_configure(config: pytest.Config, vcr: VCR) -> None:
    """Register the `function_calling_mode` VCR request matcher.

    VCR's default matchers ignore the request body, so a test that asserts on the recorded body
    (e.g. via `get_first_post_body`) can't detect the live code drifting away from what it asserts —
    the stale cassette replays regardless. Opt a test in with
    `@pytest.mark.vcr(additional_matchers=['function_calling_mode'])` to make the recorded
    `functionCallingConfig.mode` part of the cassette match: on replay, a request whose mode no
    longer equals the recorded value won't match and the test fails, catching the regression. This
    is the reusable form of "any field a test explicitly asserts should also gate cassette matching".
    """
    vcr.register_matcher('function_calling_mode', _match_function_calling_mode)  # pyright: ignore[reportUnknownMemberType]


@pytest.fixture
def google_model(gemini_api_key: str) -> GoogleModelFactory:
    """Factory to create Google models. Used by VCR-recorded integration tests."""

    def _create_model(
        model_name: str,
        api_key: str | None = None,
    ) -> GoogleModel:
        return GoogleModel(
            model_name,
            provider=GoogleProvider(api_key=api_key or gemini_api_key),
        )

    return _create_model
