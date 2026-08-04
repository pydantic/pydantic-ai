from __future__ import annotations

import json
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
    is_new_user_turn,
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
    # 'shrunk' is only produced by deliberately-compacting tests, which the marker exempts before
    # classification runs, so exercise it directly.
    assert classify_prefix_pair([('messages', 'one'), ('messages', 'two')], [('messages', 'one')]) == ('shrunk', 1)
    assert len(violations) == 1
    assert violations[0].level == 'tools'
    assert violations[0].block_index == 0


def test_check_cache_prefix_stability_fails_unmarked(
    request: pytest.FixtureRequest, prefix_moving_cassette: Path
) -> None:
    node = cast(pytest.Item, request.node)  # pyright: ignore[reportUnknownMemberType]
    with pytest.raises(pytest.fail.Exception, match='moves_cache_prefix'):
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


@pytest.mark.moves_cache_prefix(reason=True)
def test_check_cache_prefix_stability_rejects_non_string_reason(
    request: pytest.FixtureRequest, prefix_moving_cassette: Path
) -> None:
    """`reason=` must be an explanatory string, not any truthy value."""
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


def test_canonical_prefix_blocks_anthropic_excludes_deferred_tools() -> None:
    """Deferred declarations are outside Anthropic's measured prompt-cache key."""
    before = canonical_prefix_blocks(
        {
            'tools': [{'name': 'visible'}, {'name': 'searchable', 'defer_loading': True}],
            'messages': [{'role': 'user', 'content': 'Hi'}],
        },
        'https://api.anthropic.com/v1/messages',
    )
    after = canonical_prefix_blocks(
        {
            'tools': [
                {'name': 'visible'},
                {'name': 'searchable', 'defer_loading': True},
                {'name': 'revealed_later', 'defer_loading': True},
            ],
            'messages': [{'role': 'user', 'content': 'Hi'}],
        },
        'https://api.anthropic.com/v1/messages',
    )
    assert before is not None and after is not None
    assert before == after


def test_canonical_prefix_blocks_anthropic_keeps_visible_tools() -> None:
    """A change to a visible Anthropic tool remains a cache-prefix violation."""
    before = canonical_prefix_blocks(
        {'tools': [{'name': 'visible'}], 'messages': [{'role': 'user', 'content': 'Hi'}]},
        'https://api.anthropic.com/v1/messages',
    )
    after = canonical_prefix_blocks(
        {'tools': [{'name': 'changed'}], 'messages': [{'role': 'user', 'content': 'Hi'}]},
        'https://api.anthropic.com/v1/messages',
    )
    assert before is not None and after is not None
    assert classify_prefix_pair(before[1], after[1]) == ('tools-divergent', 0)


def test_classify_prefix_pair_non_object_message_blocks() -> None:
    """Message blocks that aren't JSON objects (e.g. plain strings) fall back to no conversation identity."""
    a = [('messages', '"one"'), ('messages', '"two"')]
    b = [('messages', '"one"'), ('messages', '"different"')]
    assert classify_prefix_pair(a, b) == ('messages-divergent', 1)


def _openai_chat_blocks(*, tools: bool, messages: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Cache-ordered blocks for an `openai-chat` request, mirroring `canonical_prefix_blocks`."""
    blocks: list[tuple[str, str]] = []
    if tools:
        blocks.append(('tools', json.dumps({'function': {'name': 'get_weather'}})))
    blocks.extend(('messages', json.dumps(message)) for message in messages)
    return blocks


def test_classify_prefix_pair_inserted_tool_is_flagged() -> None:
    """A tools block inserted ahead of an unchanged history is a moved prefix, not a new conversation.

    The divergence lands on the message block the new tool shifted back, so the classifier must derive
    the level from the inserted tools block (earlier in cache order) and report `tools-divergent`
    rather than silently skipping it as `new-conversation`.
    """
    a = [('tools', '"t1"'), ('messages', '"m"')]
    b = [('tools', '"t1"'), ('tools', '"t2"'), ('messages', '"m"')]
    assert classify_prefix_pair(a, b) == ('tools-divergent', 1)


def test_classify_prefix_pair_toolset_dropped_at_new_turn_is_boundary() -> None:
    """A tool-using run followed by a tool-free run that appends a new user turn is a run boundary.

    This is the shape a tool-using `generator` agent followed by a tool-free `probe` agent produces
    in one cassette: the toolset drops to nothing, but a genuine new user turn marks a new run, so it
    must not be reported as a moved prefix.
    """
    turn = [
        {'role': 'user', 'content': 'What is the weather in Paris?'},
        {'role': 'assistant', 'tool_calls': [{'id': 'c1'}]},
        {'role': 'tool', 'tool_call_id': 'c1', 'content': 'sunny'},
    ]
    a = _openai_chat_blocks(tools=True, messages=turn)
    b = _openai_chat_blocks(
        tools=False,
        messages=[*turn, {'role': 'assistant', 'content': 'It is sunny.'}, {'role': 'user', 'content': 'Reply OK'}],
    )
    assert classify_prefix_pair(a, b) == ('different-conversation', -1)


def test_classify_prefix_pair_toolset_dropped_mid_run_is_flagged() -> None:
    """Clearing the toolset mid-run (no new user turn) stays a violation, not a benign boundary.

    A tool-search or deferred-loading bug that wrongly drops every tool between two requests of the
    same run appends only assistant/tool-result messages, so it must still surface as `tools-divergent`.
    """
    turn = [
        {'role': 'user', 'content': 'What is the weather in Paris?'},
        {'role': 'assistant', 'tool_calls': [{'id': 'c1'}]},
        {'role': 'tool', 'tool_call_id': 'c1', 'content': 'sunny'},
    ]
    a = _openai_chat_blocks(tools=True, messages=turn)
    b = _openai_chat_blocks(
        tools=False,
        messages=[*turn, {'role': 'assistant', 'tool_calls': [{'id': 'c2'}]}, {'role': 'tool', 'tool_call_id': 'c2'}],
    )
    assert classify_prefix_pair(a, b) == ('tools-divergent', 0)


def test_is_new_user_turn_ignores_tool_results() -> None:
    """Tool/function results ride on user-role messages for several providers; they aren't new turns."""
    assert is_new_user_turn(json.dumps({'role': 'user', 'content': 'Reply OK'})) is True
    assert is_new_user_turn(json.dumps({'role': 'assistant', 'content': 'done'})) is False
    assert is_new_user_turn(json.dumps({'role': 'user', 'content': [{'type': 'tool_result', 'content': 'x'}]})) is False
    assert is_new_user_turn(json.dumps({'role': 'user', 'content': [{'toolResult': {}}]})) is False
    assert is_new_user_turn(json.dumps({'role': 'user', 'parts': [{'functionResponse': {}}]})) is False
    assert is_new_user_turn('not-json') is False
    assert is_new_user_turn('["a", "bare", "list"]') is False


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


def _anthropic_cassette(tmp_path: Path, name: str, bodies: list[dict[str, Any]]) -> Path:
    cassette_path = tmp_path / name
    cassette = {
        'interactions': [
            {'request': {'method': 'POST', 'uri': 'https://api.anthropic.com/v1/messages', 'parsed_body': body}}
            for body in bodies
        ]
    }
    cassette_path.write_text(yaml.safe_dump(cassette), encoding='utf-8')
    return cassette_path


def test_anthropic_deferred_tail_must_be_append_only(tmp_path: Path) -> None:
    """Deferred entries escape the cache-key model, but their wire contract is still checked.

    Within a continuing conversation the `defer_loading: true` tail may only grow at the end, in
    first-reveal order; a reorder, edit, or removal is flagged as a `deferred-tools` violation
    even though Anthropic's cache key ignores those entries.
    """
    visible = {'name': 'visible'}
    searchable = {'name': 'searchable', 'defer_loading': True}
    revealed = {'name': 'revealed_later', 'defer_loading': True}
    base_messages = [{'role': 'user', 'content': 'Hi'}]
    grown = [*base_messages, {'role': 'assistant', 'content': 'ok'}, {'role': 'user', 'content': 'more'}]

    append = _anthropic_cassette(
        tmp_path,
        'append.yaml',
        [
            {'tools': [visible, searchable], 'messages': base_messages},
            {'tools': [visible, searchable, revealed], 'messages': grown},
        ],
    )
    assert list(iter_cassette_prefix_violations(append)) == []

    reorder = _anthropic_cassette(
        tmp_path,
        'reorder.yaml',
        [
            {'tools': [visible, searchable, revealed], 'messages': base_messages},
            {'tools': [visible, revealed, searchable], 'messages': grown},
        ],
    )
    [violation] = list(iter_cassette_prefix_violations(reorder))
    assert (violation.level, violation.block_index) == ('deferred-tools', 0)

    edited = _anthropic_cassette(
        tmp_path,
        'edited.yaml',
        [
            {'tools': [visible, searchable], 'messages': base_messages},
            {
                'tools': [visible, {'name': 'searchable', 'defer_loading': True, 'description': 'changed'}],
                'messages': grown,
            },
        ],
    )
    [violation] = list(iter_cassette_prefix_violations(edited))
    assert violation.level == 'deferred-tools'

    removed = _anthropic_cassette(
        tmp_path,
        'removed.yaml',
        [
            {'tools': [visible, searchable, revealed], 'messages': base_messages},
            {'tools': [visible, searchable], 'messages': grown},
        ],
    )
    [violation] = list(iter_cassette_prefix_violations(removed))
    assert (violation.level, violation.later_block) == ('deferred-tools', '<missing>')

    # A genuinely new conversation resets the tail together with everything else.
    new_conversation = _anthropic_cassette(
        tmp_path,
        'new-conversation.yaml',
        [
            {'tools': [visible, searchable, revealed], 'messages': base_messages},
            {'tools': [visible, searchable], 'messages': [{'role': 'user', 'content': 'Different'}]},
        ],
    )
    assert list(iter_cassette_prefix_violations(new_conversation)) == []
