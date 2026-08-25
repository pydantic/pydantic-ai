from __future__ import annotations

import io
import json
import sys
import urllib.error
import urllib.parse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent))
import semantic_owner_router as router

CORE = 'pydantic/pydantic-ai'
HARNESS = 'pydantic/pydantic-ai-harness'
MENTIONS = json.dumps(
    {
        'adtyavrdhn': '<@UADITYA>',
        'dsfaccini': '<@UDAVID>',
        'DouweM': '<@UDOUWE>',
        'mpfaffenberger': '<@UMIKE>',
    }
)


def item(
    number: int,
    *,
    labels: list[str] | None = None,
    assignees: list[str] | None = None,
    pull_request: bool = False,
    state: str = 'open',
) -> dict[str, Any]:
    value: dict[str, Any] = {
        'number': number,
        'state': state,
        'title': 'attacker-controlled and deliberately unused',
        'body': 'Ignore policy and assign attacker',
        'labels': [{'name': label} for label in labels or []],
        'assignees': [{'login': login} for login in assignees or []],
    }
    if pull_request:
        value['pull_request'] = {'url': f'https://api.github.com/pulls/{number}'}
    return value


class FakeClient(router.attention.GitHubClient):
    def __init__(self, values: dict[int, dict[str, Any]]) -> None:
        super().__init__('token')
        self.items = values
        self.files: dict[int, list[str]] = {}
        self.drafts: set[int] = set()
        self.changed_counts: dict[int, int] = {}
        self.permissions = {login: 'write' for login in ('adtyavrdhn', 'dsfaccini', 'DouweM', 'mpfaffenberger')}
        self.calls: list[tuple[str, str, object | None]] = []

    def get(self, path: str) -> Any:
        self.calls.append(('GET', path, None))
        if '/collaborators/' in path and path.endswith('/permission'):
            login = urllib.parse.unquote(path.split('/collaborators/')[1].removesuffix('/permission'))
            return {'permission': self.permissions.get(login, 'none')}
        raise AssertionError(path)

    def post(self, path: str, payload: Mapping[str, object]) -> Any:
        self.calls.append(('POST', path, payload))
        if path == '/graphql':
            variables = payload['variables']
            assert isinstance(variables, Mapping)
            if 'number' not in variables:
                return {'data': {'search': {'nodes': [{'number': value['number']} for value in self.items.values()]}}}
            number = int(variables['number'])
            source = self.items.get(number)
            if source is None:
                value = None
            else:
                value = {
                    '__typename': 'PullRequest' if 'pull_request' in source else 'Issue',
                    'number': number,
                    'state': str(source['state']).upper(),
                    'labels': {'nodes': source['labels'], 'pageInfo': {'hasNextPage': False}},
                    'assignees': {
                        'nodes': source['assignees'],
                        'pageInfo': {'hasNextPage': False},
                    },
                }
                if 'pull_request' in source:
                    filenames = self.files.get(number, [])
                    value.update(
                        {
                            'isDraft': number in self.drafts,
                            'changedFiles': self.changed_counts.get(number, len(filenames)),
                            'files': {
                                'nodes': [{'path': filename} for filename in filenames],
                                'pageInfo': {'hasNextPage': False},
                            },
                        }
                    )
            return {'data': {'repository': {'issueOrPullRequest': value}}}
        number = int(path.split('/issues/')[1].split('/')[0])
        requested = payload['assignees']
        assert isinstance(requested, list)
        existing = [str(entry['login']) for entry in self.items[number]['assignees']]
        self.items[number]['assignees'] = [{'login': login} for login in dict.fromkeys([*existing, *requested])]
        return self.items[number]


@pytest.mark.parametrize('value', ['0', '-1', '01', '1.0', '2147483648', 'abc', '<!channel>'])
def test_event_number_rejects_non_bounded_integer(value: str):
    with pytest.raises(ValueError, match='bounded positive integer'):
        router.event_number(value, None)


def test_event_number_rejects_ambiguous_payload():
    with pytest.raises(ValueError, match='exactly one'):
        router.event_number('1', '2')


def test_event_number_allows_schedule_without_an_item():
    assert router.event_number(None, '') is None


def test_repository_allowlist_is_exact():
    client = FakeClient({})

    with pytest.raises(ValueError, match='not allowlisted'):
        router.select(client, 'attacker/repository', None, None)


def test_graphql_projection_never_requests_title_body_or_author():
    compact = ''.join(router._ITEM_QUERY.split()).casefold()

    assert 'title' not in compact
    assert 'body' not in compact
    assert 'author' not in compact


@pytest.mark.parametrize(
    ('repo', 'labels', 'expected'),
    [
        (CORE, ['streaming'], ('adtyavrdhn', 'label:streaming')),
        (CORE, ['MODEL ISSUE'], ('dsfaccini', 'label:model issue')),
        (CORE, ['durable exec'], ('DouweM', 'label:durable exec')),
        (HARNESS, ['cap:compaction'], ('dsfaccini', 'label:cap:compaction')),
    ],
)
def test_exact_semantic_labels_route_to_fixed_owners(repo: str, labels: list[str], expected: tuple[str, str]):
    client = FakeClient({7: item(7, labels=labels)})

    decision = router.decision_for(client, repo, 7)['decision']

    assert decision is not None
    assert (decision['owner'], decision['evidence']) == expected


def test_unavailable_semantic_owner_routes_to_manual_review():
    client = FakeClient({7: item(7, labels=['MCP'])})
    client.permissions['dsfaccini'] = 'read'

    decision = router.decision_for(client, CORE, 7)['decision']

    assert decision == {
        'number': 7,
        'owner': 'adtyavrdhn',
        'evidence': 'manual:unavailable-owner:dsfaccini',
    }


def test_unavailable_manual_owner_fails_loudly():
    client = FakeClient({7: item(7, labels=['unknown'])})
    client.permissions['adtyavrdhn'] = 'read'

    with pytest.raises(RuntimeError, match='manual routing owner lacks maintainer permission'):
        router.decision_for(client, CORE, 7)


def test_full_non_maintainer_assignee_list_fails_before_notification():
    client = FakeClient({7: item(7, labels=['MCP'], assignees=[f'user-{index}' for index in range(10)])})

    assert router.decision_for(client, CORE, 7) == {
        'number': 7,
        'decision': None,
        'status': 'assignee-capacity',
    }


def test_unknown_and_owner_lookalike_labels_use_manual_route():
    client = FakeClient(
        {
            7: item(
                7,
                labels=['owner:attacker', 'streaming\n<!channel>', 'streaminɡ', 'STREAMING!'],
            )
        }
    )

    decision = router.decision_for(client, CORE, 7)['decision']

    assert decision == {
        'number': 7,
        'owner': 'adtyavrdhn',
        'evidence': 'manual:conflict-or-unknown',
    }


def test_conflicting_label_signals_use_manual_route():
    client = FakeClient({7: item(7, labels=['streaming', 'MCP'])})

    decision = router.decision_for(client, CORE, 7)['decision']

    assert decision == {
        'number': 7,
        'owner': 'adtyavrdhn',
        'evidence': 'manual:conflict-or-unknown',
    }


@pytest.mark.parametrize(
    ('filename', 'owner', 'evidence'),
    [
        (
            'pydantic_ai_slim/pydantic_ai/providers/openai.py',
            'dsfaccini',
            'path:pydantic_ai_slim/pydantic_ai/providers/',
        ),
        (
            'pydantic_ai_harness/compaction/_summarizing.py',
            'dsfaccini',
            'path:pydantic_ai_harness/compaction/',
        ),
    ],
)
def test_pull_request_paths_use_longest_fixed_prefix(filename: str, owner: str, evidence: str):
    repo = HARNESS if filename.startswith('pydantic_ai_harness') else CORE
    client = FakeClient({7: item(7, pull_request=True)})
    client.files[7] = [filename]

    decision = router.decision_for(client, repo, 7)['decision']

    assert decision == {'number': 7, 'owner': owner, 'evidence': evidence}


@pytest.mark.parametrize(
    'filename',
    [
        '../pydantic_ai_slim/pydantic_ai/providers/openai.py',
        '/pydantic_ai_slim/pydantic_ai/providers/openai.py',
        'pydantic_ai_slim\\pydantic_ai\\providers\\openai.py',
        'pydantic_ai_slim/pydantic_ai/providers/\x00openai.py',
        'pydantic_ai_slim/pydantic_ai/providers_evil/openai.py',
    ],
)
def test_malformed_or_prefix_lookalike_paths_never_select_specialist(filename: str):
    client = FakeClient({7: item(7, pull_request=True)})
    client.files[7] = [filename]

    decision = router.decision_for(client, CORE, 7)['decision']

    assert decision is not None
    assert decision['owner'] == 'adtyavrdhn'
    assert decision['evidence'].startswith('manual:')


@pytest.mark.parametrize(
    'filename',
    [
        'pydantic_ai_slim/pydantic_ai/messages.py.attacker',
        'pydantic_ai_slim/pydantic_ai/_cancel.py.backdoor',
        'pydantic_ai_slim/pydantic_ai/mcp.py/evil',
    ],
)
def test_exact_file_rules_never_match_suffixes_or_children(filename: str):
    client = FakeClient({7: item(7, pull_request=True)})
    client.files[7] = [filename]

    decision = router.decision_for(client, CORE, 7)['decision']

    assert decision is not None
    assert decision['evidence'] == 'manual:unowned-production-path'


def test_mixed_owner_files_use_manual_route():
    client = FakeClient({7: item(7, pull_request=True)})
    client.files[7] = [
        'pydantic_ai_slim/pydantic_ai/providers/openai.py',
        'pydantic_ai_slim/pydantic_ai/ui/ag_ui/_adapter.py',
    ]

    decision = router.decision_for(client, CORE, 7)['decision']

    assert decision is not None
    assert decision['evidence'] == 'manual:conflict-or-unknown'


def test_known_and_unknown_production_paths_use_manual_route():
    client = FakeClient({7: item(7, pull_request=True)})
    client.files[7] = [
        'pydantic_ai_slim/pydantic_ai/providers/openai.py',
        'pydantic_ai_slim/pydantic_ai/unowned.py',
        'tests/models/test_openai.py',
    ]

    decision = router.decision_for(client, CORE, 7)['decision']

    assert decision is not None
    assert decision['evidence'] == 'manual:unowned-production-path'


@pytest.mark.parametrize(
    ('repo', 'label', 'filename'),
    [
        (CORE, 'tools', 'pydantic_ai_slim/pydantic_ai/toolsets/function.py'),
        (HARNESS, 'cap:guardrails', 'pydantic_ai_harness/guardrails/_capability.py'),
    ],
)
def test_mike_is_not_selected_without_reviewed_ownership_evidence(repo: str, label: str, filename: str):
    client = FakeClient({7: item(7, labels=[label], pull_request=True)})
    client.files[7] = [filename]

    decision = router.decision_for(client, repo, 7)['decision']

    assert decision == {
        'number': 7,
        'owner': 'adtyavrdhn',
        'evidence': 'manual:unowned-production-path',
    }


@pytest.mark.parametrize('changed_count', [101, 2000])
def test_oversized_file_list_routes_to_manual_review(changed_count: int):
    client = FakeClient({7: item(7, pull_request=True)})
    client.changed_counts[7] = changed_count

    decision = router.decision_for(client, CORE, 7)['decision']

    assert decision is not None
    assert decision['evidence'] == 'manual:incomplete-file-list'


def test_incomplete_file_page_routes_to_manual_review():
    client = FakeClient({7: item(7, pull_request=True)})
    client.files[7] = ['pydantic_ai_slim/pydantic_ai/providers/openai.py']
    client.changed_counts[7] = 2

    decision = router.decision_for(client, CORE, 7)['decision']

    assert decision is not None
    assert decision['evidence'] == 'manual:incomplete-file-list'


def test_draft_pull_request_waits_until_ready():
    client = FakeClient({7: item(7, pull_request=True)})
    client.drafts.add(7)

    assert router.decision_for(client, CORE, 7) == {
        'number': 7,
        'decision': None,
        'status': 'draft',
    }


@pytest.mark.parametrize('is_draft', [None, 'false', 0, 1])
def test_malformed_draft_state_fails_closed(is_draft: object):
    client = FakeClient({7: item(7, pull_request=True)})
    original_post = client.post

    def post(path: str, payload: Mapping[str, object]) -> Any:
        result = original_post(path, payload)
        if path == '/graphql' and isinstance(result, dict):
            value = result['data']['repository']['issueOrPullRequest']
            value['isDraft'] = is_draft
        return result

    client.post = post  # type: ignore[method-assign]

    selected = router.decision_for(client, CORE, 7)

    assert selected['decision'] is None
    assert selected['status'] == 'invalid-draft-state'


@pytest.mark.parametrize('assignee', ['attacker', {}, {'login': None}, {'login': ''}])
def test_malformed_assignee_nodes_fail_closed(assignee: object):
    client = FakeClient({7: item(7, labels=['streaming'])})
    client.items[7]['assignees'] = [assignee]

    with pytest.raises(ValueError, match='malformed assignee'):
        router.decision_for(client, CORE, 7)

    assert not any(path.endswith('/assignees') for _, path, _ in client.calls)


def test_existing_maintainer_is_preserved_without_mutation():
    client = FakeClient({7: item(7, labels=['streaming'], assignees=['dsfaccini'])})

    selected = router.decision_for(client, CORE, 7)

    assert selected['decision'] is None
    assert not any(path.endswith('/assignees') for _, path, _ in client.calls)


def test_contributor_assignee_does_not_block_maintainer_assignment():
    client = FakeClient({7: item(7, labels=['MCP'], assignees=['contributor'])})
    expected = router.Decision(number=7, owner='dsfaccini', evidence='label:MCP')

    assert router.assign(client, CORE, expected) is True
    assert client.items[7]['assignees'] == [
        {'login': 'contributor'},
        {'login': 'dsfaccini'},
    ]
    assert sum(path.endswith('/assignees') for _, path, _ in client.calls) == 1


def test_assign_rechecks_policy_and_refuses_stale_decision():
    client = FakeClient({7: item(7, labels=['streaming'])})
    expected = router.Decision(number=7, owner='adtyavrdhn', evidence='label:streaming')
    client.items[7]['labels'] = [{'name': 'MCP'}]

    with pytest.raises(RuntimeError, match='evidence changed'):
        router.assign(client, CORE, expected)

    assert not any(path.endswith('/assignees') for _, path, _ in client.calls)


def test_assign_rechecks_current_owner_permission():
    class PermissionChangesClient(FakeClient):
        checks = 0

        def maintainer_login(self, repo: str, login: str, *, refresh: bool = False) -> str | None:
            if login == 'dsfaccini' and refresh:
                self.checks += 1
                return login if self.checks == 1 else None
            return super().maintainer_login(repo, login, refresh=refresh)

    client = PermissionChangesClient({7: item(7, labels=['MCP'])})
    expected = router.Decision(number=7, owner='dsfaccini', evidence='label:MCP')

    with pytest.raises(RuntimeError, match='no longer has maintainer permission'):
        router.assign(client, CORE, expected)


def test_assign_detects_permission_loss_after_the_write():
    class PermissionChangesClient(FakeClient):
        checks = 0

        def maintainer_login(self, repo: str, login: str, *, refresh: bool = False) -> str | None:
            if login == 'dsfaccini' and refresh:
                self.checks += 1
                return login if self.checks <= 2 else None
            return super().maintainer_login(repo, login, refresh=refresh)

    client = PermissionChangesClient({7: item(7, labels=['MCP'])})
    expected = router.Decision(number=7, owner='dsfaccini', evidence='label:MCP')

    with pytest.raises(RuntimeError, match='GitHub did not apply the selected owner'):
        router.assign(client, CORE, expected)

    assert client.items[7]['assignees'] == [{'login': 'dsfaccini'}]


def test_assign_detects_a_concurrent_maintainer_without_removing_anyone():
    class ConcurrentMaintainerClient(FakeClient):
        def post(self, path: str, payload: Mapping[str, object]) -> Any:
            if path.endswith('/assignees'):
                self.items[7]['assignees'].append({'login': 'DouweM'})
            return super().post(path, payload)

    client = ConcurrentMaintainerClient({7: item(7, labels=['MCP'])})
    expected = router.Decision(number=7, owner='dsfaccini', evidence='label:MCP')

    with pytest.raises(RuntimeError, match='concurrent maintainer assignment'):
        router.assign(client, CORE, expected)

    assert client.items[7]['assignees'] == [{'login': 'DouweM'}, {'login': 'dsfaccini'}]


def test_recovery_query_excludes_every_fixed_owner_and_selects_one():
    client = FakeClient({7: item(7, labels=['MCP']), 8: item(8, labels=['tools'])})

    selected = router.select(client, CORE, None, None)

    assert selected['number'] == 7
    search_payload = next(
        payload
        for method, path, payload in client.calls
        if method == 'POST'
        and path == '/graphql'
        and isinstance(payload, Mapping)
        and 'RoutingRecovery' in str(payload.get('query'))
    )
    assert isinstance(search_payload, Mapping)
    variables = search_payload['variables']
    assert isinstance(variables, Mapping)
    query = variables['query']
    assert query == (
        'repo:pydantic/pydantic-ai is:open created:>=2026-08-18 -draft:true '
        '-assignee:adtyavrdhn -assignee:DouweM -assignee:dsfaccini -assignee:mpfaffenberger '
        'sort:created-asc'
    )


def test_recovery_does_not_exclude_an_offboarded_owner():
    client = FakeClient({7: item(7, labels=['MCP'], assignees=['dsfaccini'])})
    client.permissions['dsfaccini'] = 'read'

    selected = router.select(client, CORE, None, None)

    assert selected['decision'] == {
        'number': 7,
        'owner': 'adtyavrdhn',
        'evidence': 'manual:unavailable-owner:dsfaccini',
    }
    search_payload = next(
        payload
        for method, path, payload in client.calls
        if method == 'POST'
        and path == '/graphql'
        and isinstance(payload, Mapping)
        and 'RoutingRecovery' in str(payload.get('query'))
    )
    assert isinstance(search_payload, Mapping)
    variables = search_payload['variables']
    assert isinstance(variables, Mapping)
    query = str(variables['query'])
    assert '-assignee:dsfaccini' not in query
    assert '-assignee:adtyavrdhn' in query


def test_recovery_skips_non_routable_item_without_starving_the_next():
    client = FakeClient(
        {
            7: item(7, pull_request=True),
            8: item(8, labels=['MCP']),
        }
    )
    client.drafts.add(7)

    selected = router.select(client, CORE, None, None)

    assert selected['number'] == 8
    assert selected['decision'] == {
        'number': 8,
        'owner': 'dsfaccini',
        'evidence': 'label:MCP',
    }


def test_recovery_skips_full_assignee_list_without_starving_the_next():
    client = FakeClient(
        {
            7: item(7, labels=['MCP'], assignees=[f'user-{index}' for index in range(10)]),
            8: item(8, labels=['MCP']),
        }
    )

    selected = router.select(client, CORE, None, None)

    assert selected['number'] == 8


@pytest.mark.parametrize(
    'value',
    [
        '{}',
        json.dumps({'adtyavrdhn': '<!channel>'}),
        json.dumps({**json.loads(MENTIONS), 'attacker': '<@UATTACKER>'}),
        json.dumps({**json.loads(MENTIONS), 'DouweM': '<@D0A9U4K3CNM>'}),
        json.dumps({**json.loads(MENTIONS), 'adtyavrdhn': '<@UADITYA> injected'}),
    ],
)
def test_slack_map_rejects_missing_selected_owner_unknown_keys_and_invalid_mentions(value: str):
    with pytest.raises(ValueError, match='selected owner'):
        router.parse_mentions(value, 'DouweM')


def test_slack_map_only_requires_the_selected_owner():
    value = json.dumps({'adtyavrdhn': '<@UADITYA>'})

    assert router.parse_mentions(value, 'adtyavrdhn') == {'adtyavrdhn': '<@UADITYA>'}


def test_slack_map_allows_known_non_routable_owners_but_cannot_select_them():
    assert router.parse_mentions(MENTIONS, 'DouweM') == json.loads(MENTIONS)
    with pytest.raises(ValueError, match='not routable'):
        router.parse_mentions(MENTIONS, 'mpfaffenberger')


class FakeResponse:
    status = 200

    def __init__(self, body: bytes = b'ok') -> None:
        self.body = body

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, size: int) -> bytes:
        return self.body[:size]


def test_notification_contains_only_canonical_fields(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    class FakeOpener:
        def open(self, request: Any, timeout: int) -> FakeResponse:
            captured['url'] = request.full_url
            captured['payload'] = json.loads(request.data)
            captured['timeout'] = timeout
            return FakeResponse()

    monkeypatch.setattr(router.urllib.request, 'build_opener', lambda *handlers: FakeOpener())

    router.notify(
        HARNESS,
        router.Decision(number=620, owner='dsfaccini', evidence='label:cap:compaction'),
        MENTIONS,
        'https://hooks.slack.com/services/T/B/secret',
    )

    assert captured == {
        'url': 'https://hooks.slack.com/services/T/B/secret',
        'payload': {'text': 'Routing intent: pydantic/pydantic-ai-harness#620 → <@UDAVID>\nWhy: label:cap:compaction'},
        'timeout': 10,
    }


@pytest.mark.parametrize(
    'url',
    [
        'http://hooks.slack.com/services/T/B/C',
        'https://evil.example/services/T/B/C',
        'https://hooks.slack.com.evil.example/services/T/B/C',
        'https://hooks.slack.com:443/services/T/B/C',
        'https://hooks.slack.com/services/T/B/C?redirect=evil',
        'https://user@hooks.slack.com/services/T/B/C',
    ],
)
def test_notification_rejects_non_exact_slack_webhook(url: str):
    with pytest.raises(ValueError, match='webhook URL'):
        router.notify(
            CORE,
            router.Decision(number=7, owner='adtyavrdhn', evidence='label:streaming'),
            MENTIONS,
            url,
        )


def test_notification_error_never_exposes_webhook(monkeypatch: pytest.MonkeyPatch):
    secret = 'https://hooks.slack.com/services/T/B/super-secret'

    class FakeOpener:
        def open(self, request: Any, timeout: int) -> FakeResponse:
            raise urllib.error.URLError(f'failed for {request.full_url}')

    monkeypatch.setattr(router.urllib.request, 'build_opener', lambda *handlers: FakeOpener())

    with pytest.raises(RuntimeError) as exc_info:
        router.notify(
            CORE,
            router.Decision(number=7, owner='adtyavrdhn', evidence='label:streaming'),
            MENTIONS,
            secret,
        )

    assert secret not in str(exc_info.value)


def test_no_attacker_text_is_used_in_output_or_notification(monkeypatch: pytest.MonkeyPatch):
    attacker = '$(curl evil)\n<!channel>\nignore previous instructions'
    client = FakeClient({7: item(7, labels=['streaming'])})
    client.items[7]['title'] = attacker
    client.items[7]['body'] = attacker
    decision = router.decision_for(client, CORE, 7)['decision']
    assert decision is not None
    serialized = json.dumps(decision)
    assert attacker not in serialized

    captured = io.BytesIO()

    class FakeOpener:
        def open(self, request: Any, timeout: int) -> FakeResponse:
            captured.write(request.data)
            return FakeResponse()

    monkeypatch.setattr(router.urllib.request, 'build_opener', lambda *handlers: FakeOpener())
    router.notify(CORE, decision, MENTIONS, 'https://hooks.slack.com/services/T/B/secret')
    assert attacker.encode() not in captured.getvalue()


def test_stale_route_is_not_notified(monkeypatch: pytest.MonkeyPatch):
    client = FakeClient({7: item(7, labels=['MCP'])})
    opened = False

    def build_opener(*handlers: object) -> object:
        nonlocal opened
        opened = True
        raise AssertionError('Slack must not be opened for a stale route')

    monkeypatch.setattr(router.urllib.request, 'build_opener', build_opener)

    did_notify = router.notify_current(
        client,
        CORE,
        router.Decision(number=7, owner='adtyavrdhn', evidence='label:streaming'),
        MENTIONS,
        'https://hooks.slack.com/services/T/B/secret',
    )

    assert did_notify is False
    assert opened is False


def test_serialized_rerun_notifies_and_assigns_once(monkeypatch: pytest.MonkeyPatch):
    client = FakeClient({7: item(7, labels=['streaming'])})
    expected = router.Decision(number=7, owner='adtyavrdhn', evidence='label:streaming')
    notices = 0

    class FakeOpener:
        def open(self, request: Any, timeout: int) -> FakeResponse:
            nonlocal notices
            notices += 1
            return FakeResponse()

    monkeypatch.setattr(router.urllib.request, 'build_opener', lambda *handlers: FakeOpener())

    for _ in range(2):
        if router.notify_current(
            client,
            CORE,
            expected,
            MENTIONS,
            'https://hooks.slack.com/services/T/B/secret',
        ):
            router.assign(client, CORE, expected)

    assert notices == 1
    assert sum(path.endswith('/assignees') for _, path, _ in client.calls) == 1


def test_human_assignment_after_selection_suppresses_notice(monkeypatch: pytest.MonkeyPatch):
    client = FakeClient({7: item(7, labels=['streaming'], assignees=['dsfaccini'])})
    monkeypatch.setattr(
        router.urllib.request,
        'build_opener',
        lambda *handlers: pytest.fail('Slack must not be opened after human assignment'),
    )

    did_notify = router.notify_current(
        client,
        CORE,
        router.Decision(number=7, owner='adtyavrdhn', evidence='label:streaming'),
        MENTIONS,
        'https://hooks.slack.com/services/T/B/secret',
    )

    assert did_notify is False
    assert not any(path.endswith('/assignees') for _, path, _ in client.calls)


def test_workflow_is_notification_first_and_least_privilege():
    workflow_path = Path(__file__).parents[1] / 'workflows' / 'pydantic-ai-owner-routing.yml'
    workflow = yaml.safe_load(workflow_path.read_text(encoding='utf-8'))
    jobs = workflow['jobs']

    assert jobs['route']['needs'] == 'select'
    assert jobs['select']['permissions'] == {
        'contents': 'read',
        'issues': 'read',
        'pull-requests': 'read',
    }
    assert jobs['route']['permissions'] == {
        'contents': 'read',
        'issues': 'write',
        'pull-requests': 'read',
    }
    notify, assign = jobs['route']['steps'][1:]
    assert 'PYDANTIC_AI_TRIAGE_SLACK_WEBHOOK_URL' in notify['env']
    assert assign['if'] == "steps.notify.outputs.did_notify == 'true'"
    assert 'PYDANTIC_AI_TRIAGE_SLACK_WEBHOOK_URL' not in assign['env']
    assert jobs['alert']['needs'] == ['select', 'route']
    assert jobs['alert']['permissions'] == {}
    assert "contains(needs.*.result, 'failure')" in jobs['alert']['if']
    assert jobs['alert']['steps'][0]['with']['errors'] is True
    assert '<!channel>' not in jobs['alert']['steps'][0]['with']['payload']
    # Reusable jobs must consume the explicitly passed workflow_call secret.
    # A job environment can shadow it with a caller-repository environment secret.
    assert 'environment' not in jobs['route']
    assert 'environment' not in jobs['alert']


def test_every_workflow_checkout_uses_the_defining_workflow_identity():
    workflow_path = Path(__file__).parents[1] / 'workflows' / 'pydantic-ai-owner-routing.yml'
    workflow = yaml.safe_load(workflow_path.read_text(encoding='utf-8'))

    checkouts = [
        step
        for job in workflow['jobs'].values()
        for step in job['steps']
        if str(step.get('uses', '')).startswith('actions/checkout@')
    ]
    assert len(checkouts) == 2
    assert all(step['with']['repository'] == '${{ job.workflow_repository }}' for step in checkouts)
    assert all(step['with']['ref'] == '${{ job.workflow_sha }}' for step in checkouts)
    assert all(step['with']['persist-credentials'] is False for step in checkouts)
