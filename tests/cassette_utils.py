"""Cassette verification utilities for VCR and XAI proto cassettes.

This module provides a unified interface for verifying cassette contents across
different cassette formats (VCR HTTP cassettes and XAI protobuf cassettes).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vcr.cassette import Cassette


def get_cassette_request_bodies(cassette: Cassette) -> list[str]:
    """Get all request bodies from a VCR cassette as strings."""
    bodies: list[str] = []
    for request in cassette.requests:  # pyright: ignore[reportUnknownVariableType,reportUnknownMemberType]
        raw_body = request.body  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        if raw_body:
            body = raw_body.decode('utf-8', errors='ignore') if isinstance(raw_body, bytes) else raw_body  # pyright: ignore[reportUnknownVariableType]
            bodies.append(body)  # pyright: ignore[reportUnknownArgumentType]
        elif getattr(request, 'parsed_body', None):  # pyright: ignore[reportUnknownArgumentType]  # pragma: no cover
            bodies.append(json.dumps(request.parsed_body))  # pyright: ignore[reportUnknownMemberType]
    return bodies


def get_xai_cassette_request_bodies(cassette_path: Path) -> list[str]:  # pragma: no cover
    """Get all request and response bodies from an XAI cassette as strings."""
    from tests.models.xai_proto_cassettes import (
        SampleInteraction,
        StreamInteraction,
        XaiProtoCassette,
        xai_sdk_available,
    )

    if not xai_sdk_available():
        return []

    bodies: list[str] = []
    cassette = XaiProtoCassette.load(cassette_path)

    for interaction in cassette.interactions:
        if interaction.request_json:
            bodies.append(json.dumps(interaction.request_json))

        if isinstance(interaction, SampleInteraction) and interaction.response_json:
            bodies.append(json.dumps(interaction.response_json))
        elif isinstance(interaction, StreamInteraction) and interaction.chunks_json:
            for chunk in interaction.chunks_json:
                bodies.append(json.dumps(chunk))

    return bodies


def sanitize_cassette_filename(name: str, max_length: int = 240) -> str:
    """Sanitize filename to be filesystem-safe."""
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', name)
    return sanitized[:max_length]


def get_xai_cassette_path(test_name: str, test_module: str) -> Path:
    """Construct XAI cassette path following the same pattern as conftest.py.

    Args:
        test_name: The test function name with parameters.
        test_module: The test module name (without .py extension).
    """
    cassette_name = sanitize_cassette_filename(test_name, 240)
    return Path(__file__).parent / 'cassettes' / test_module / f'{cassette_name}.xai.yaml'


def pattern_in_bodies(pattern: str, bodies: list[str]) -> bool:
    """Check if pattern exists in any of the request bodies."""
    return any(pattern in body for body in bodies)


@dataclass
class CassetteContext:
    """Unified cassette verification context for VCR and XAI cassettes.

    Encapsulates provider-specific cassette handling (VCR vs XAI proto format)
    and provides a uniform verification interface.
    """

    provider: str
    vcr: Cassette | None
    test_name: str
    test_module: str

    def _get_bodies(self) -> list[str]:
        """Get request/response bodies from the appropriate cassette format."""
        if self.provider == 'xai':
            cassette_path = get_xai_cassette_path(self.test_name, self.test_module)
            if cassette_path.exists():  # pragma: no cover
                return get_xai_cassette_request_bodies(cassette_path)
            return []
        if self.vcr is not None:
            return get_cassette_request_bodies(self.vcr)
        return []  # pragma: no cover

    def verify_contains(self, *patterns: str | tuple[str, ...]) -> None:
        """Verify that all patterns appear in cassette request/response bodies.

        Args:
            patterns: Patterns to search for. Each pattern can be a string or a tuple
                (where any one of the tuple elements matching is sufficient).

        Raises:
            AssertionError: If a pattern is not found.
        """
        bodies = self._get_bodies()
        if not bodies:
            # Skip verification if no bodies found (e.g., cassette doesn't exist yet during recording)
            return

        for pattern in patterns:
            if isinstance(pattern, tuple):
                assert any(pattern_in_bodies(p, bodies) for p in pattern), (
                    f'Expected one of {pattern} in cassette but none found'
                )
            else:
                assert pattern_in_bodies(pattern, bodies), f'Expected "{pattern}" in cassette but not found'

    def verify_ordering(self, *patterns: str | tuple[str, ...]) -> None:
        """Verify that patterns appear in cassette bodies in the given order.

        Args:
            patterns: Patterns that must appear in order. Each pattern can be a string
                or a tuple (where any one of the tuple elements is used for position checking).

        Raises:
            AssertionError: If ordering is violated or a pattern is not found.
        """
        bodies = self._get_bodies()
        if not bodies:
            return

        content = ''.join(bodies)
        last_index = -1

        for pattern in patterns:
            if isinstance(pattern, tuple):
                indices = [content.find(p) for p in pattern]
                valid_indices = [i for i in indices if i != -1]
                assert valid_indices, f'Expected one of {pattern} in cassette but none found'
                current_index = min(valid_indices)
            else:
                current_index = content.find(pattern)
                assert current_index != -1, f'Expected "{pattern}" in cassette but not found'

            assert current_index > last_index, (
                f'Pattern "{pattern}" found at index {current_index}, '
                f'but expected after index {last_index} (ordering violation)'
            )
            last_index = current_index
