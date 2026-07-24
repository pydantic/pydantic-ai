"""Cassette verification utilities for VCR and XAI proto cassettes.

This module provides a unified interface for verifying cassette contents across
different cassette formats (VCR HTTP cassettes and XAI protobuf cassettes).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeGuard
from urllib.parse import urlparse

import pytest
import yaml

from pydantic_ai._utils import is_str_dict

try:
    from yaml import CSafeLoader as SafeLoader
except ImportError:  # pragma: no cover
    from yaml import SafeLoader

if TYPE_CHECKING:
    from vcr.cassette import Cassette

PrefixBlock = tuple[str, str]


def _is_list(value: Any) -> TypeGuard[list[Any]]:
    return isinstance(value, list)


@dataclass(frozen=True)
class CassettePrefixViolation:
    """A consecutive request pair whose provider-cache wire prefix moved."""

    shape: str
    pair_index: int
    level: str
    block_index: int
    earlier_block: str
    later_block: str


def check_cache_prefix_stability(node: pytest.Item, cassette_path: Path) -> None:
    """Fail when a cassette moves its provider-cache wire prefix without an exemption."""
    if (marker := node.get_closest_marker('moves_cache_prefix')) is not None:
        if not marker.kwargs.get('reason'):
            pytest.fail(
                '@pytest.mark.moves_cache_prefix requires reason=... explaining why this test '
                'deliberately moves the cache prefix'
            )
        return

    violations = list(iter_cassette_prefix_violations(cassette_path))
    if violations:
        details = '\n'.join(
            f'{cassette_path} [{violation.shape}] pair {violation.pair_index}, {violation.level} block '
            f'{violation.block_index}:\n  earlier: {violation.earlier_block}\n  later:   {violation.later_block}'
            for violation in violations
        )
        pytest.fail(
            f"{details}\nA moving wire prefix busts the provider prompt cache on every turn; if this test's behavior is "
            'deliberately prefix-moving (compaction, dynamic tool disclosure, history rewriting), add '
            '@pytest.mark.moves_cache_prefix(reason=...) to the test'
        )


def canonical_prefix_blocks(body: dict[str, Any], url: str) -> tuple[str, list[PrefixBlock]] | None:
    """Flatten a provider request into cache-ordered JSON blocks.

    The supported shapes cover the multi-request endpoints in the 2026-07-15 cassette corpus.
    Blocks deliberately use insertion-order `json.dumps` output because wire order is the invariant.
    """
    parsed_url = urlparse(url)
    host, path = parsed_url.hostname or '', parsed_url.path
    blocks: list[PrefixBlock] = []

    def add(level: str, items: Any) -> None:
        if items is None:
            return
        # Some fields hold a single block rather than a list: a plain-string system prompt
        # (Anthropic/Bedrock), or Google's `systemInstruction`, which is one Content *dict* --
        # iterating it would silently reduce it to its keys and blind the check to content changes.
        block_items: list[Any] = items if _is_list(items) else [items]
        for item in block_items:
            blocks.append((level, json.dumps(item)))

    if path.endswith('/v1/messages') or (host == 'api.anthropic.com' and '/messages' in path):
        add('tools', body.get('tools'))
        add('system', body.get('system'))
        add('messages', body.get('messages'))
        return 'anthropic', blocks
    if path.endswith('/chat/completions'):
        add('tools', body.get('tools'))
        add('messages', body.get('messages'))
        return 'openai-chat', blocks
    if path.endswith('/responses') or '/responses' in path:
        add('system', body.get('instructions'))
        add('tools', body.get('tools'))
        input_ = body.get('input')
        add('messages', input_ if isinstance(input_, list) else [input_] if input_ is not None else None)
        return 'openai-responses', blocks
    if 'generativelanguage' in host or ':generateContent' in path or ':streamGenerateContent' in path:
        add('tools', body.get('tools'))
        add('system', body.get('systemInstruction') or body.get('system_instruction'))
        add('messages', body.get('contents'))
        return 'google', blocks
    if '/converse' in path:
        tool_config = body.get('toolConfig')
        add('tools', tool_config.get('tools') if is_str_dict(tool_config) else None)
        add('system', body.get('system'))
        add('messages', body.get('messages'))
        return 'bedrock', blocks
    return None


def is_new_user_turn(block: str) -> bool:
    """True when a `messages` block is a fresh user prompt rather than a tool/function result.

    A new user turn beyond the previous request's history marks a new conversation turn or run (a
    fresh `agent.run()`), where a legitimately different toolset -- including none -- is expected.
    Within a single run the agent loop only appends assistant and tool-result messages, never a new
    user prompt, so a genuine user turn is the reliable boundary signal. Tool and function results are
    carried on user-role messages by several providers (Anthropic/Bedrock `tool_result`/`toolResult`
    content, Google `functionResponse` parts, OpenAI Responses `function_call_output`); those are part
    of the same turn, so they are not counted as a new one.
    """
    try:
        message = json.loads(block)
    except json.JSONDecodeError:
        return False
    if not is_str_dict(message) or message.get('role') != 'user':
        return False
    content = message.get('content')
    parts = content if _is_list(content) else message.get('parts')
    if _is_list(parts):
        return not any(
            is_str_dict(part)
            and (
                part.get('type') in ('tool_result', 'function_call_output')
                or 'toolResult' in part
                or 'functionResponse' in part
            )
            for part in parts
        )
    return True


def classify_prefix_pair(a: list[PrefixBlock], b: list[PrefixBlock]) -> tuple[str, int]:
    """Classify how the cache-ordered blocks change between consecutive requests."""
    if a == b:
        return 'identical', -1
    shared_length = min(len(a), len(b))
    divergent_index = next((i for i in range(shared_length) if a[i] != b[i]), shared_length)
    if divergent_index == len(a) and len(b) > len(a):
        return 'extension', -1
    if divergent_index == len(b) and len(a) > len(b):
        return 'shrunk', divergent_index

    level = a[divergent_index][0] if divergent_index < len(a) else b[divergent_index][0]
    first_message_index = next((i for i, (block_level, _) in enumerate(a) if block_level == 'messages'), None)
    if level == 'messages' and divergent_index == first_message_index:
        return 'new-conversation', divergent_index

    def conversation_identity(blocks: list[PrefixBlock]) -> str | None:
        for block_level, block in blocks:
            if block_level != 'messages':
                continue
            try:
                role = json.loads(block).get('role')
            except (AttributeError, TypeError, json.JSONDecodeError):
                role = None
            if role not in ('system', 'developer'):
                return block
        return None

    a_identity = conversation_identity(a)
    b_identity = conversation_identity(b)
    if a_identity is not None and b_identity is not None and a_identity != b_identity:
        return 'different-conversation', divergent_index

    # A request that drops the entire toolset as a new user turn begins is a new conversation turn or
    # run (a fresh agent reusing an earlier run's history, e.g. a tool-using generator followed by a
    # tool-free probe), not a moved prefix. This is only a boundary when a genuine new user turn is
    # appended: within a single run the toolset is constant and only assistant/tool-result messages
    # are appended, so a tools-drop *without* a new user turn -- a tool-search or deferred-loading bug
    # wrongly clearing the tools mid-run -- still falls through to `tools-divergent` and is flagged.
    if level == 'tools' and not any(block_level == 'tools' for block_level, _ in b):
        a_messages = [block for block_level, block in a if block_level == 'messages']
        b_messages = [block for block_level, block in b if block_level == 'messages']
        if b_messages[: len(a_messages)] == a_messages and any(
            is_new_user_turn(block) for block in b_messages[len(a_messages) :]
        ):
            return 'different-conversation', -1

    return f'{level}-divergent', divergent_index


def iter_cassette_prefix_violations(cassette_path: Path) -> Iterator[CassettePrefixViolation]:
    """Yield prompt-cache prefix violations from one VCR cassette.

    Across 1,177 cassettes on 2026-07-15, this found 15 deliberately prefix-moving pairs in ten
    cassettes. Requests are grouped by host and provider shape so unrelated endpoints are not paired.
    """
    cassette = yaml.load(cassette_path.read_text(encoding='utf-8'), Loader=SafeLoader)
    if not is_str_dict(cassette):
        return
    requests_by_endpoint: dict[tuple[str, str], list[list[PrefixBlock]]] = defaultdict(list)

    raw_interactions = cassette.get('interactions')
    if not _is_list(raw_interactions):
        return
    interactions = raw_interactions
    for interaction in interactions:
        if not is_str_dict(interaction) or not is_str_dict(request := interaction.get('request')):
            continue
        method = request.get('method')
        if not isinstance(method, str) or method.upper() != 'POST':
            continue
        body = request.get('parsed_body')
        if not is_str_dict(body):
            continue
        uri = request.get('uri')
        if not isinstance(uri, str):
            continue
        canonical = canonical_prefix_blocks(body, uri)
        if canonical is None:
            continue
        shape, blocks = canonical
        requests_by_endpoint[(urlparse(uri).hostname or '', shape)].append(blocks)

    for (_, shape), requests in requests_by_endpoint.items():
        for pair_index, (earlier, later) in enumerate(zip(requests, requests[1:])):
            classification, block_index = classify_prefix_pair(earlier, later)
            if classification != 'shrunk' and not classification.endswith('-divergent'):
                continue
            level = classification.removesuffix('-divergent')
            earlier_block = earlier[block_index][1] if block_index < len(earlier) else '<missing>'
            later_block = later[block_index][1] if block_index < len(later) else '<missing>'
            yield CassettePrefixViolation(
                shape=shape,
                pair_index=pair_index,
                level=level,
                block_index=block_index,
                earlier_block=earlier_block[:200],
                later_block=later_block[:200],
            )


def get_first_post_body(cassette: Cassette) -> dict[str, Any]:
    """Return the first POST request body in a VCR cassette, parsed as JSON.

    Some VCR serializers (e.g. the project's custom JSON body serializer used for
    huggingface cassettes) deserialize `request.body` to a dict ahead of time;
    others leave it as raw bytes/str. Handle both shapes.
    """
    for request in cassette.requests:  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        if request.method != 'POST':  # pyright: ignore[reportUnknownMemberType]
            continue
        body = request.body  # pyright: ignore[reportUnknownVariableType,reportUnknownMemberType]
        if not body:
            continue  # pragma: no cover
        if isinstance(body, dict):
            return body  # pyright: ignore[reportUnknownVariableType]
        parsed: dict[str, Any] = json.loads(body)  # pyright: ignore[reportUnknownArgumentType]
        return parsed
    return {}  # pragma: no cover


def single_request_body(cassette: Cassette) -> dict[str, Any]:
    """Decode the JSON body of the single recorded request in `cassette`.

    Use this for cassette-backed tests that send exactly one request and want to
    assert directly on the wire body (e.g. that a specific field survived
    translation). Asserts the single-request invariant — tests with intentional
    multi-request cassettes should access `cassette.requests` directly.
    """
    requests = cassette.requests  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
    assert len(requests) == 1, f'Expected 1 request, got {len(requests)}'  # pyright: ignore[reportUnknownArgumentType]
    return json.loads(requests[0].body)  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]


# Provider-specific cassette extractors — group new ones under this header so the module
# doesn't grow into a flat bag of `get_<provider>_*` helpers.


def get_bedrock_tool_config_from_cassette(cassette: Cassette) -> dict[str, Any]:
    """Return the `toolConfig` from the first POST request body in a Bedrock VCR cassette."""
    return get_first_post_body(cassette).get('toolConfig', {})


def get_bedrock_tool_names_from_cassette(cassette: Cassette) -> list[str]:
    """Extract Bedrock tool definition names from the first recorded POST request body."""
    tools: list[dict[str, Any]] = get_bedrock_tool_config_from_cassette(cassette).get('tools', [])
    return [tool['toolSpec']['name'] for tool in tools if 'toolSpec' in tool]


def get_cohere_tool_names_from_cassette(cassette: Cassette) -> list[str]:
    """Extract Cohere tool definition names from the first recorded POST request body."""
    tools: list[dict[str, Any]] = get_first_post_body(cassette).get('tools', [])
    return [tool['function']['name'] for tool in tools if 'function' in tool]


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


def _get_xai_cassette_request_bodies(cassette_path: Path) -> list[str]:
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
            if path.exists():
                return _get_xai_cassette_request_bodies(path)
            return []
        if self.vcr is not None:
            bodies = _get_cassette_request_bodies(self.vcr)
            if bodies:  # pragma: no branch
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
