"""Shared fixtures for model-profile tests."""

from __future__ import annotations as _annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from vcr import VCR


def _probe_shape(request: Any) -> tuple[Any, Any, Any]:
    """Extract the fields that decide whether the Responses API accepts a probe.

    `request` is a VCR `Request` (untyped in practice), so it's typed loosely here.
    """
    data: dict[str, Any] = json.loads(request.body)
    return data.get('model'), data.get('reasoning'), data.get('temperature')


def _match_probe_shape(request1: Any, request2: Any) -> None:
    assert _probe_shape(request1) == _probe_shape(request2)


def pytest_recording_configure(config: pytest.Config, vcr: VCR) -> None:
    """Register a VCR matcher tying a recorded verdict to the request shape that produced it.

    Every probe in `test_openai_reasoning_ground_truth.py` is a POST to the same Responses URL, so
    VCR's default matchers (method + path) can only tell them apart by recorded order — while the
    thing those tests assert, the API's accept/reject, is decided entirely by the `reasoning` and
    `temperature` fields in the body. Several probes on one model also share a verdict, so without
    this matcher a reordered or reshaped probe replays a stale answer and still passes green.
    Opt in with `@pytest.mark.vcr(additional_matchers=['probe_shape'])`.
    """
    vcr.register_matcher('probe_shape', _match_probe_shape)  # pyright: ignore[reportUnknownMemberType]
