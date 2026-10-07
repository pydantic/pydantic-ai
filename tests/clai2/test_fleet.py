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
            "ValueError: Logfire config references $AWS_SECRET_ACCESS_KEY, which isn't in this server's env_allow list",
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
        'env_allow': ['DEEPWIKI_TOKEN'],
        'headers': {'X-Token': '${env:DEEPWIKI_TOKEN}'},
    }
    fleet = _fleet(tmp_path, {'mcp_servers': [server]}, {'items': []})

    build = fleet.build()
    [consent] = build.pending
    assert build.loaded == []
    assert consent.question() == (
        'Logfire wants to connect MCP server `deepwiki` at https://mcp.deepwiki.com/mcp. Sends: $DEEPWIKI_TOKEN.'
    )
    fleet.decide(consent, allow=True)
    assert [item.key for item in fleet.build().loaded] == ['mcp_server:deepwiki']

    # A new target is a new question.
    moved = {**server, 'url': 'https://elsewhere.example/mcp'}
    fleet = _fleet(tmp_path, {'mcp_servers': [moved]}, {'items': []})
    assert [c.target for c in fleet.build().pending] == ['https://elsewhere.example/mcp']


def test_an_allowed_but_unset_variable_is_a_load_failure(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.delenv('DEEPWIKI_TOKEN', raising=False)
    server = {
        'name': 'deepwiki',
        'url': 'https://mcp.deepwiki.com/mcp',
        'env_allow': ['DEEPWIKI_TOKEN'],
        'headers': {'X-Token': '${env:DEEPWIKI_TOKEN}'},
    }
    fleet = _fleet(tmp_path, {'mcp_servers': [server]}, {'items': []})
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

    def render(selected: list[Any]) -> str:
        console = Console(file=StringIO(), width=200)
        console.print(notice_panel(selected, snapshot=build.snapshot, source='logfire/clai2', link=None))
        return console.file.getvalue()  # pyright: ignore[reportAttributeAccessIssue]

    by_tier = {change.tier: change for change in changes}
    assert 'catalog v1 · /catalog' in render([by_tier['catalog']])
    assert 'config v1 · /catalog' in render([by_tier['company']])
    both = render(changes)
    assert 'Updated from Logfire · logfire/clai2' in both
    assert 'Added: skills pr-shepherd · catalog iterate' in both
    assert 'config v1 · catalog v1 · /catalog' in both
    assert fleet.compliance(build.snapshot, build.loaded)['clai2.catalog.version'] == '1'


def test_items_apply_by_team_and_repo(tmp_path: Path) -> None:
    here = {'name': 'here', 'instructions': 'x', 'applies_to': {'teams': ['ai'], 'repos': ['pydantic/*']}}
    there = {'name': 'there', 'instructions': 'y', 'applies_to': {'repos': ['acme/*']}}
    anyone = {'name': 'anyone', 'instructions': 'z'}
    fleet = _fleet(tmp_path, {'skills': [here, there, anyone]}, {'items': []})
    fleet.scope = lambda: ('ai', 'pydantic/pydantic-ai')
    assert [item.name for item in fleet.build().loaded] == ['here', 'anyone']
    fleet.scope = lambda: ('platform', 'pydantic/pydantic-ai')
    assert [item.name for item in fleet.build().loaded] == ['anyone']
    rows = {row.name: row.elsewhere for row in fleet.rows(fleet.snapshot())}
    assert rows == {'here': True, 'there': True, 'anyone': False}


def test_the_launch_header_says_who_manages_clai2_and_links_to_logfire(tmp_path: Path) -> None:
    import io

    from rich.console import Console

    from pydantic_clai2.builtin_plugins.logfire import LogfirePlugin, LogfireSettings
    from pydantic_clai2.plugins import PluginHost

    def header(agent: dict[str, Any], settings: dict[str, str] | None = None, *, terminal: bool = False) -> str:
        output = io.StringIO()
        console = Console(
            file=output, width=200, force_terminal=terminal, color_system='standard' if terminal else None
        )
        plugin = object.__new__(LogfirePlugin)
        plugin.host = PluginHost[None](name='observability', console=console, settings={})
        plugin._settings = LogfireSettings.model_validate(settings or {})  # pyright: ignore[reportPrivateUsage]
        plugin.fleet = _fleet(tmp_path, agent, {'items': []})
        plugin._print_header()  # pyright: ignore[reportPrivateUsage]
        return output.getvalue()

    eu = {'base_url': 'https://logfire-eu.pydantic.info', 'project': 'logfire/clai2'}
    assert header({'display_name': 'Pydantic'}, eu) == (
        '◆ Managed by Pydantic through Logfire · logfire/clai2 · /catalog\n'
    )
    assert header({}, eu) == '◆ Managed by logfire/clai2 through Logfire · /catalog\n'
    assert header({}) == '◆ Managed by Logfire through Logfire · /catalog\n'
    # A terminal that understands links gets the agent's configuration page.
    linked = header({'display_name': 'Pydantic'}, eu, terminal=True)
    assert '\x1b]8;' in linked
    assert 'https://logfire-eu.pydantic.info/logfire/clai2/agents/clai2/configure/edit' in linked
