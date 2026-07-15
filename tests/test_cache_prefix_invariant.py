from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from .cassette_utils import canonical_prefix_blocks, classify_prefix_pair, iter_cassette_prefix_violations

TESTS_ROOT = Path(__file__).parent
CASSETTES = sorted(TESTS_ROOT.glob('**/cassettes/**/*.yaml'))

KNOWN_PREFIX_MOVERS: dict[str, str] = {
    'cassettes/test_tool_search/test_tool_search_eval[anthropic].yaml': (
        'dynamic tool disclosure after ToolSearch discovery'
    ),
    'cassettes/test_tool_search/test_tool_search_eval[google].yaml': (
        'dynamic tool disclosure after ToolSearch discovery'
    ),
    'cassettes/test_tool_search/test_tool_search_eval[openai-chat].yaml': (
        'dynamic tool disclosure after ToolSearch discovery'
    ),
    'cassettes/test_tool_search/test_cross_provider_capability_replay[google-gemini-3-flash-preview-openai-responses-gpt-5.4].yaml': (
        'dynamic tool disclosure after ToolSearch discovery'
    ),
    'cassettes/test_tool_search/test_openai_deferred_capability_runs_on_model_without_native_tool_search.yaml': (
        'dynamic tool disclosure after ToolSearch discovery'
    ),
    'models/cassettes/test_deepseek/test_deepseek_deferred_capability_with_thinking.yaml': (
        'dynamic tool disclosure after ToolSearch discovery'
    ),
    'models/cassettes/test_openai_responses/test_openai_responses_compact_with_auto_previous_response_id_chain.yaml': (
        'deliberate history compaction'
    ),
    'models/cassettes/test_openai_responses/test_openai_responses_compact_with_auto_previous_response_id.yaml': (
        'deliberate history compaction'
    ),
    'models/cassettes/test_openai_responses/test_openai_responses_compact_with_instructions.yaml': (
        'deliberate history compaction'
    ),
    'models/cassettes/test_openai_responses/test_openai_responses_thinking_with_modified_history.yaml': (
        'test deliberately modifies history'
    ),
}


@pytest.mark.parametrize(
    'cassette_path',
    [pytest.param(path, id=path.relative_to(TESTS_ROOT).as_posix()) for path in CASSETTES],
)
def test_cassette_cache_prefix_invariant(cassette_path: Path) -> None:
    """Every recorded conversation preserves its provider-cache wire prefix as history grows."""
    relative_path = cassette_path.relative_to(TESTS_ROOT).as_posix()
    violations = list(iter_cassette_prefix_violations(cassette_path))

    if relative_path in KNOWN_PREFIX_MOVERS:
        assert violations, f'{relative_path}: allowlist entry is stale, remove it'
        return

    details = '\n'.join(
        f'{relative_path} [{violation.shape}] pair {violation.pair_index}, {violation.level} block '
        f'{violation.block_index}:\n  earlier: {violation.earlier_block}\n  later:   {violation.later_block}'
        for violation in violations
    )
    assert not violations, (
        f"{details}\nA moving wire prefix busts the provider prompt cache on every turn; if this test's behavior is "
        'deliberately prefix-moving (compaction, dynamic tool disclosure, history rewriting), add the cassette to '
        'KNOWN_PREFIX_MOVERS with a reason'
    )


def test_known_prefix_movers_exist() -> None:
    """Require deleted or renamed allowlist entries to be removed explicitly."""
    missing = [path for path in KNOWN_PREFIX_MOVERS if not (TESTS_ROOT / path).is_file()]
    assert not missing, f'{missing}: allowlist entry is stale, remove it'


def test_synthetic_cassette_detects_prefix_violation(tmp_path: Path) -> None:
    """Exercise cassette parsing because VCR matching does not protect request-body prefix shape."""
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

    violations = list(iter_cassette_prefix_violations(cassette_path))

    assert classify_prefix_pair([('messages', 'one')], [('messages', 'one'), ('messages', 'two')]) == (
        'extension',
        -1,
    )
    assert len(violations) == 1
    assert violations[0].level == 'tools'
    assert violations[0].block_index == 0


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
