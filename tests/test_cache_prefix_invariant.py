from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import yaml
from vcr.record_mode import RecordMode

from .cassette_utils import (
    canonical_prefix_blocks,
    check_cache_prefix_stability,
    classify_prefix_pair,
    iter_cassette_prefix_violations,
)
from .conftest import fail_cache_prefix_violations


@pytest.fixture
def prefix_moving_cassette(tmp_path: Path) -> Path:
    cassette_path = tmp_path / 'prefix-moving.yaml'
    cassette = {
        'interactions': [
            {
                'request': {
                    'method': 'POST',
                    'uri': 'https://api.openai.com/v1/chat/completions',
                    'parsed_body': {'tools': [{'type': 'function', 'function': {'name': 'first'}}], 'messages': []},
                }
            },
            {
                'request': {
                    'method': 'POST',
                    'uri': 'https://api.openai.com/v1/chat/completions',
                    'parsed_body': {'tools': [{'type': 'function', 'function': {'name': 'changed'}}], 'messages': []},
                }
            },
        ]
    }
    cassette_path.write_text(yaml.safe_dump(cassette), encoding='utf-8')
    return cassette_path


def test_synthetic_cassette_detects_prefix_violation(prefix_moving_cassette: Path) -> None:
    """Exercise cassette parsing because VCR matching does not protect request-body prefix shape."""
    violations = list(iter_cassette_prefix_violations(prefix_moving_cassette))

    assert classify_prefix_pair([('messages', 'one')], [('messages', 'one'), ('messages', 'two')]) == (
        'extension',
        -1,
    )
    assert len(violations) == 1
    assert violations[0].level == 'tools'
    assert violations[0].block_index == 0


def test_check_cache_prefix_stability_fails_unmarked(
    request: pytest.FixtureRequest, prefix_moving_cassette: Path
) -> None:
    node = cast(pytest.Item, request.node)  # pyright: ignore[reportUnknownMemberType]
    with pytest.raises(pytest.fail.Exception, match='@pytest.mark.moves_cache_prefix'):
        check_cache_prefix_stability(node, prefix_moving_cassette)


@pytest.mark.moves_cache_prefix(reason='unit test covers the deliberate exemption')
def test_check_cache_prefix_stability_allows_marked(
    request: pytest.FixtureRequest, prefix_moving_cassette: Path
) -> None:
    node = cast(pytest.Item, request.node)  # pyright: ignore[reportUnknownMemberType]
    check_cache_prefix_stability(node, prefix_moving_cassette)


def test_check_cache_prefix_stability_allows_clean(request: pytest.FixtureRequest, tmp_path: Path) -> None:
    cassette_path = tmp_path / 'clean.yaml'
    cassette_path.write_text('interactions: []\n', encoding='utf-8')
    node = cast(pytest.Item, request.node)  # pyright: ignore[reportUnknownMemberType]
    check_cache_prefix_stability(node, cassette_path)


@pytest.mark.parametrize(
    'call_report,cassette_path',
    [
        (SimpleNamespace(skipped=False, failed=True), '/unused/after-failure.yaml'),
        (SimpleNamespace(skipped=False, failed=False), '/missing/cassette.yaml'),
    ],
)
def test_cache_prefix_fixture_skips_uncheckable_cassettes(call_report: Any, cassette_path: str) -> None:
    """Failed tests and missing cassette files must not produce a second teardown failure."""
    node = SimpleNamespace(rep_setup=SimpleNamespace(skipped=False, failed=False), rep_call=call_report)
    request = SimpleNamespace(node=node)
    vcr = SimpleNamespace(record_mode=RecordMode.NONE, _path=cassette_path)
    fixture = cast(Callable[[Any, Any], Iterator[None]], getattr(fail_cache_prefix_violations, '__wrapped__'))
    iterator = fixture(cast(Any, request), cast(Any, vcr))

    next(iterator)
    with pytest.raises(StopIteration):
        next(iterator)


def test_canonical_prefix_blocks_google_system_instruction_dict() -> None:
    """Google's `systemInstruction` is a single Content dict; it must serialize as one block, not its keys."""
    shape_and_blocks = canonical_prefix_blocks(
        {
            'systemInstruction': {'parts': [{'text': 'Be helpful.'}], 'role': 'user'},
            'contents': [{'role': 'user', 'parts': [{'text': 'Hi'}]}],
        },
        'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent',
    )
    assert shape_and_blocks is not None
    shape, blocks = shape_and_blocks
    assert shape == 'google'
    assert blocks[0] == ('system', '{"parts": [{"text": "Be helpful."}], "role": "user"}')

    changed = canonical_prefix_blocks(
        {
            'systemInstruction': {'parts': [{'text': 'Be terse.'}], 'role': 'user'},
            'contents': [{'role': 'user', 'parts': [{'text': 'Hi'}]}],
        },
        'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent',
    )
    assert changed is not None
    assert classify_prefix_pair(blocks, changed[1]) == ('system-divergent', 0)


@pytest.mark.moves_cache_prefix
def test_check_cache_prefix_stability_requires_reason(
    request: pytest.FixtureRequest, prefix_moving_cassette: Path
) -> None:
    """A bare marker without `reason=` must not silently exempt the test."""
    node = cast(pytest.Item, request.node)  # pyright: ignore[reportUnknownMemberType]
    with pytest.raises(pytest.fail.Exception, match='requires reason'):
        check_cache_prefix_stability(node, prefix_moving_cassette)


def test_canonical_prefix_blocks_bedrock() -> None:
    """The corpus has no Converse cassettes with parsed bodies, so exercise the shape directly."""
    shape_and_blocks = canonical_prefix_blocks(
        {
            'toolConfig': {'tools': [{'toolSpec': {'name': 'tool'}}]},
            'system': [{'text': 'Be helpful.'}],
            'messages': [{'role': 'user', 'content': [{'text': 'Hi'}]}],
        },
        'https://bedrock-runtime.us-east-1.amazonaws.com/model/us.anthropic.claude-sonnet-4-5-v1:0/converse',
    )
    assert shape_and_blocks is not None
    shape, blocks = shape_and_blocks
    assert shape == 'bedrock'
    assert [level for level, _ in blocks] == ['tools', 'system', 'messages']

    shape_and_blocks = canonical_prefix_blocks(
        {'messages': []},
        'https://bedrock-runtime.us-east-1.amazonaws.com/model/amazon.nova-pro-v1:0/converse',
    )
    assert shape_and_blocks is not None
    assert shape_and_blocks[0] == 'bedrock'


def test_classify_prefix_pair_non_object_message_blocks() -> None:
    """Message blocks that aren't JSON objects (e.g. plain strings) fall back to no conversation identity."""
    a = [('messages', '"one"'), ('messages', '"two"')]
    b = [('messages', '"one"'), ('messages', '"different"')]
    assert classify_prefix_pair(a, b) == ('messages-divergent', 1)


def test_iter_cassette_prefix_violations_skips_malformed_cassettes(tmp_path: Path) -> None:
    non_dict = tmp_path / 'non-dict.yaml'
    non_dict.write_text('- just\n- a\n- list\n', encoding='utf-8')
    assert list(iter_cassette_prefix_violations(non_dict)) == []

    non_list_interactions = tmp_path / 'non-list.yaml'
    non_list_interactions.write_text('interactions: not-a-list\n', encoding='utf-8')
    assert list(iter_cassette_prefix_violations(non_list_interactions)) == []

    skipped_requests = tmp_path / 'skipped-requests.yaml'
    skipped_requests.write_text(
        yaml.safe_dump(
            {
                'interactions': [
                    'not-a-dict',
                    {'request': 'not-a-dict'},
                    {'request': {'method': 'GET', 'uri': 'https://api.openai.com/v1/chat/completions'}},
                    {'request': {'method': 'POST', 'uri': 'https://api.openai.com/v1/chat/completions'}},
                    {'request': {'method': 'POST', 'parsed_body': {'messages': []}}},
                    {'request': {'method': 'POST', 'uri': 'https://unknown.example.com/x', 'parsed_body': {}}},
                ]
            }
        ),
        encoding='utf-8',
    )
    assert list(iter_cassette_prefix_violations(skipped_requests)) == []
