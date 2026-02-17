"""Cassette verification utilities for VCR and XAI proto cassettes.

This module provides a unified interface for verifying cassette contents across
different cassette formats (VCR HTTP cassettes and XAI protobuf cassettes).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vcr.cassette import Cassette


def _get_cassette_request_bodies(cassette: Cassette) -> list[str]:
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


def _get_cassette_bodies_from_yaml(path: Path) -> list[str]:
    """Read request bodies from a VCR cassette YAML file on disk.

    Used as fallback when the VCR cassette object is not available (e.g. CI playback).
    """
    import yaml

    data: dict[str, Any] = yaml.safe_load(path.read_text(encoding='utf-8'))
    bodies: list[str] = []
    for interaction in data.get('interactions', []):
        request = interaction.get('request', {})
        parsed_body = request.get('parsed_body') or request.get('body')
        if parsed_body is None:
            continue
        if isinstance(parsed_body, dict | list):
            bodies.append(json.dumps(parsed_body))
        elif isinstance(parsed_body, str) and parsed_body:
            bodies.append(parsed_body)
    return bodies


def _get_xai_cassette_request_bodies(cassette_path: Path) -> list[str]:  # pragma: no cover
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


def _sanitize_cassette_filename(name: str, max_length: int = 240) -> str:
    """Sanitize filename to be filesystem-safe."""
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', name)
    return sanitized[:max_length]


def _pattern_in_bodies(pattern: str, bodies: list[str]) -> bool:
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
    test_dir: Path

    def _vcr_cassette_path(self) -> Path:
        return self.test_dir / 'cassettes' / self.test_module / f'{_sanitize_cassette_filename(self.test_name)}.yaml'

    def _xai_cassette_path(self) -> Path:
        return (
            self.test_dir / 'cassettes' / self.test_module / f'{_sanitize_cassette_filename(self.test_name)}.xai.yaml'
        )

    def _get_bodies(self) -> list[str]:
        """Get request/response bodies from the appropriate cassette format."""
        if self.provider == 'xai':
            path = self._xai_cassette_path()
            if path.exists():  # pragma: no cover
                return _get_xai_cassette_request_bodies(path)
            return []
        if self.vcr is not None:
            bodies = _get_cassette_request_bodies(self.vcr)
            if bodies:
                return bodies
        path = self._vcr_cassette_path()
        if path.exists():
            return _get_cassette_bodies_from_yaml(path)
        return []

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
            return

        for pattern in patterns:
            if isinstance(pattern, tuple):
                assert any(_pattern_in_bodies(p, bodies) for p in pattern), (
                    f'Expected one of {pattern} in cassette but none found'
                )
            else:
                assert _pattern_in_bodies(pattern, bodies), f'Expected "{pattern}" in cassette but not found'

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
