"""Hackathon fleet control: what a run reports as loaded, and per-session MCP policy records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import logfire
from logfire.variables import VariablesConfig

from pydantic_ai_harness.policy import PolicyDecision
from pydantic_clai2 import policy_state
from pydantic_clai2.builtin_plugins.fleet import Fleet


def _fleet(tmp_path: Path, agent: dict[str, Any], catalog: dict[str, Any]) -> Fleet:
    def var(name: str, value: dict[str, Any]) -> dict[str, Any]:
        label = {'version': 1, 'serialized_value': json.dumps(value)}
        return {
            'name': name,
            'labels': {'production': label},
            'rollout': {'labels': {'production': 1.0}},
            'overrides': [],
        }

    config = VariablesConfig.model_validate(
        {'variables': {'agent__clai2': var('agent__clai2', agent), 'catalog__clai2': var('catalog__clai2', catalog)}}
    )
    instance = logfire.configure(
        local=True, send_to_logfire=False, console=False, variables=logfire.LocalVariablesOptions(config=config)
    )
    return Fleet(instance=instance, name='clai2', state_file=tmp_path / 'state.json')


def test_an_item_that_fails_to_load_is_reported_not_announced(tmp_path: Path) -> None:
    skill = {'name': 'pr-shepherd', 'description': 'Babysit PRs', 'instructions': 'Watch CI.'}
    bad = {
        'kind': 'mcp_server',
        'name': 'bad',
        'default': 'on',
        'payload': {'url': 'https://example.com/mcp', 'headers': {'Authorization': '${env:AWS_SECRET_ACCESS_KEY}'}},
    }
    fleet = _fleet(tmp_path, {'skills': [skill]}, {'items': [bad]})

    build = fleet.prepare()
    assert [item.key for item in build.loaded] == ['skill:pr-shepherd']
    assert [(item.key, error) for item, error in build.failed] == [
        (
            'mcp_server:bad',
            "ValueError: Logfire config references $AWS_SECRET_ACCESS_KEY, which isn't allowed "
            '(fleet_env_allow / policy.env_allow)',
        )
    ]
    assert [change.describe() for change in fleet.changes(build)] == ['Added company skill from Logfire: pr-shepherd']
    # The run uses the build its turn announced, and the next run builds afresh.
    assert fleet.take() is build
    assert fleet.take() is not build
    # Failing again later is not reported as a removal.
    assert fleet.changes(fleet.build()) == []


def test_mcp_policy_records_once_per_session() -> None:
    from pydantic_ai_harness.policy import MCPPolicy, Policy

    recorded: list[PolicyDecision] = []
    policy = Policy(mcp=MCPPolicy(allow=['deepwiki'], mode='observe'))

    def start_session() -> None:
        policy_state.install(policy_state.PolicySource(policy=lambda: policy, record=recorded.append))

    try:
        start_session()
        assert policy_state.mcp_allowed('context7', 'https://mcp.context7.com/mcp', subject='context7')
        assert policy_state.mcp_allowed('context7', 'https://mcp.context7.com/mcp', subject='context7')
        assert policy_state.mcp_allowed('deepwiki', '', subject='deepwiki')
        start_session()
        assert policy_state.mcp_allowed('context7', 'https://mcp.context7.com/mcp', subject='context7')
    finally:
        policy_state.install(None)
    assert [(d.tool_name, d.outcome) for d in recorded] == [('mcp:context7', 'would_deny')] * 2


def test_mcp_allowlist_matches_a_stdio_command_and_names_the_server() -> None:
    from pydantic_ai_harness.policy import MCPPolicy, Policy

    recorded: list[PolicyDecision] = []
    policy = Policy(mcp=MCPPolicy(allow=['npx -y @acme/allowed*'], mode='enforce'))
    policy_state.install(policy_state.PolicySource(policy=lambda: policy, record=recorded.append))
    try:
        assert policy_state.mcp_allowed('acme', '', subject='npx -y @acme/allowed-mcp')
        assert not policy_state.mcp_allowed('other', '', subject='npx -y @other/mcp')
    finally:
        policy_state.install(None)
    assert [(d.server_name, d.subject, d.outcome) for d in recorded] == [('other', 'npx -y @other/mcp', 'denied')]


def test_pushed_servers_wait_for_consent_keyed_by_target_and_env(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv('DEEPWIKI_TOKEN', 'secret')
    server = {
        'name': 'deepwiki',
        'url': 'https://mcp.deepwiki.com/mcp',
        'headers': {'X-Token': '${env:DEEPWIKI_TOKEN}'},
    }
    fleet = _fleet(tmp_path, {'mcp_servers': [server]}, {'items': []})
    fleet.env_allow = ('DEEPWIKI_*',)

    build = fleet.build()
    [consent] = build.pending
    assert build.loaded == []
    assert consent.question() == (
        'Logfire wants to connect MCP server `deepwiki` at https://mcp.deepwiki.com/mcp and send $DEEPWIKI_TOKEN.'
    )
    fleet.decide(consent, allow=True)
    assert [item.key for item in fleet.build().loaded] == ['mcp_server:deepwiki']

    # A new target is a new question.
    moved = {**server, 'url': 'https://elsewhere.example/mcp'}
    fleet = _fleet(tmp_path, {'mcp_servers': [moved]}, {'items': []})
    fleet.env_allow = ('DEEPWIKI_*',)
    assert [c.target for c in fleet.build().pending] == ['https://elsewhere.example/mcp']


def test_an_allowed_but_unset_variable_is_a_load_failure(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.delenv('DEEPWIKI_TOKEN', raising=False)
    server = {
        'name': 'deepwiki',
        'url': 'https://mcp.deepwiki.com/mcp',
        'headers': {'X-Token': '${env:DEEPWIKI_TOKEN}'},
    }
    fleet = _fleet(tmp_path, {'mcp_servers': [server], 'policy': {'env_allow': ['DEEPWIKI_TOKEN']}}, {'items': []})
    [(item, error)] = fleet.build().failed
    assert error == 'ValueError: Logfire config references $DEEPWIKI_TOKEN, which is not set in your environment'


def test_notices_name_the_version_of_what_changed(tmp_path: Path) -> None:
    from io import StringIO

    from rich.console import Console

    from pydantic_clai2.builtin_plugins.fleet_ui import notice_panel

    skill = {'name': 'pr-shepherd', 'description': 'Babysit PRs', 'instructions': 'Watch CI.'}
    addon = {'kind': 'skill', 'name': 'iterate', 'default': 'on', 'payload': {'instructions': 'Iterate.'}}
    fleet = _fleet(tmp_path, {'skills': [skill]}, {'items': [addon]})
    build = fleet.build()
    changes = fleet.changes(build, mark_seen=False)

    def title(selected: list[Any]) -> str:
        console = Console(file=StringIO(), width=200)
        console.print(notice_panel(selected, snapshot=build.snapshot, link=None))
        return console.file.getvalue().splitlines()[0]  # pyright: ignore[reportAttributeAccessIssue]

    by_tier = {change.tier: change for change in changes}
    assert '(catalog v1)' in title([by_tier['catalog']])
    assert '(config v1)' in title([by_tier['company']])
    assert '(config v1 · catalog v1)' in title(changes)
    assert fleet.compliance(build.snapshot, build.loaded)['clai2.catalog.version'] == '1'
