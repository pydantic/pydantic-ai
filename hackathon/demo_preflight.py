"""Demo preflight, and a targeted reset of the clai2 fleet variables on EU staging (hackathon).

    uv run --env-file .env python hackathon/demo_preflight.py                       # preflight, then what a reset would do
    uv run --env-file .env python hackathon/demo_preflight.py --reset --yes         # also write the reset
    uv run --env-file .env python hackathon/demo_preflight.py --suffix _test --reset --yes   # rehearse on `_test` vars

Reads `LOGFIRE_CLAI2_API_KEY` (read and write variables) and `PYDANTIC_AI_GATEWAY_API_KEY`. Nothing is written
without `--reset --yes`.

The reset puts the demo back to a clean start and keeps what is real:

- `agent__clai2`: keeps `model`, `display_name` and `policy.memory.shared`; removes instructions, skills,
  `mcp_servers` and policy rules (everything the demo adds through accepted proposals).
- `catalog__clai2`: no items.
- `memory__clai2`: no files, if the variable exists.
- `fleet_proposals__clai2`: accepted and dismissed proposals go back to pending, so they can be accepted again in
  the demo. Pending and stale ones are untouched. The miner never re-proposes an accepted or dismissed id and
  refreshes a pending one in place (`patterns.merge`), so a reset proposal is simply pending again: the next miner
  run refreshes its evidence, or marks it stale if it is no longer found.

Every write is checked: write the value, re-read it, then point `production` at `latest`. Logfire deduplicates
versions by content, so a value an older version already holds gets no new version; `production` is then pinned to
that version instead, and the script says so.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

BASE_URL = 'https://logfire-eu.pydantic.info'
PLATFORM_PR = ('pydantic/platform', 45493)
UVX = [
    'uvx',
    '--refresh-package',
    'pydantic-clai2',
    '--from',
    'git+https://github.com/pydantic/pydantic-ai@control-plane#subdirectory=src/pydantic_clai2',
    '--with',
    'logfire[variables] @ git+https://github.com/pydantic/logfire.git@ca08c9e15746e53a364fdf72d9156ce87492ad3a#subdirectory=logfire',
    '--with',
    'logfire-sdk @ git+https://github.com/pydantic/logfire.git@ca08c9e15746e53a364fdf72d9156ce87492ad3a#subdirectory=logfire-sdk',
    '--with',
    'pydantic-handlebars>=0.2.1',
    'clai2',
    '--help',
]
MINER_FRESH = timedelta(minutes=15)
REVIEWED = ('accepted', 'dismissed')
CLEARED_ON_RESET = ('accepted_at', 'accepted_by', 'accepted_tier', 'status_reason')


@dataclass
class Report:
    """PASS/WARN/FAIL lines, and how many failed or warned."""

    failed: int = 0
    warned: int = 0
    lines: list[str] = field(default_factory=list[str])

    def check(self, ok: bool | None, what: str, detail: str = '') -> None:
        """`True` PASS, `False` FAIL, `None` WARN."""
        tag = 'PASS' if ok else 'WARN' if ok is None else 'FAIL'
        self.failed += ok is False
        self.warned += ok is None
        print(f'{tag}  {what}' + (f': {detail}' if detail else ''), flush=True)


def _client() -> httpx.Client:
    return httpx.Client(
        base_url=BASE_URL,
        headers={'Authorization': f'bearer {os.environ["LOGFIRE_CLAI2_API_KEY"]}'},
        timeout=httpx.Timeout(30, read=60),
    )


def _variables(client: httpx.Client) -> dict[str, Any]:
    response = client.get('/v1/variables/')
    response.raise_for_status()
    return response.json()['variables']


def _latest(variable: dict[str, Any]) -> tuple[int | None, Any]:
    latest = variable.get('latest_version') or {}
    raw = latest.get('serialized_value')
    return latest.get('version'), (json.loads(raw) if raw else None)


def _production(variable: dict[str, Any]) -> str:
    label = (variable.get('labels') or {}).get('production') or {}
    if label.get('ref') == 'latest':
        return 'latest'
    return f'pinned to v{label.get("version")}' if label else 'no production label'


# Preflight


def preflight(report: Report, *, suffix: str, slow: bool = True) -> dict[str, Any]:
    """Run every check; returns the variables as read, for planning the reset."""
    names = [f'{base}{suffix}' for base in ('agent__clai2', 'catalog__clai2', 'fleet_proposals__clai2')]
    control = 'fleet_miner_control__clai2'  # The miner runs against the real variables only.
    variables: dict[str, Any] = {}
    try:
        with _client() as client:
            variables = _variables(client)
        report.check(True, 'staging API reachable', f'{len(variables)} variables in logfire/clai2')
    except (httpx.HTTPError, KeyError) as error:
        report.check(False, 'staging API reachable', f'{type(error).__name__}: {error}')
        return variables
    for name in [*names, control]:
        variable = variables.get(name)
        if variable is None:
            report.check(False, f'{name} exists', 'missing')
            continue
        version, _ = _latest(variable)
        production = _production(variable)
        report.check(production == 'latest', f'{name} production -> latest', f'v{version}, production {production}')
    memory = variables.get(f'memory__clai2{suffix}')
    report.check(
        True,
        f'memory__clai2{suffix}',
        'absent (fine: clai2 treats it as no repo notes)'
        if memory is None
        else f'v{_latest(memory)[0]}, {_production(memory)}',
    )
    if (miner := variables.get(control)) is not None:
        _, value = _latest(miner)
        stamps = [value.get(key) for key in ('finished_at', 'started_at')] if isinstance(value, dict) else []
        last = max((datetime.fromisoformat(stamp) for stamp in stamps if stamp), default=None)
        age = datetime.now(UTC) - last if last else None
        report.check(
            age is not None and age < MINER_FRESH,
            'miner --watch alive',
            f'last run {int(age.total_seconds() // 60)} min ago ({value.get("status")})' if age else 'no run recorded',
        )
    if slow:
        _gateway(report)
        _preview(report)
        _uvx(report)
    _test_runs(report)
    return variables


def _gateway(report: Report) -> None:
    if not os.getenv('PYDANTIC_AI_GATEWAY_API_KEY'):
        report.check(False, 'gateway call', 'PYDANTIC_AI_GATEWAY_API_KEY is not set')
        return
    os.environ.setdefault('PYDANTIC_AI_NO_BANNER', '1')
    from pydantic_ai import Agent

    try:
        result = Agent('gateway/anthropic:claude-sonnet-5-5').run_sync(
            'Reply with exactly: OK', model_settings={'max_tokens': 5}
        )
        report.check('OK' in result.output, 'gateway call (Sonnet 5.5)', repr(result.output[:20]))
    except Exception as error:  # any failure is the finding
        report.check(False, 'gateway call (Sonnet 5.5)', f'{type(error).__name__}: {str(error)[:160]}')


def _preview(report: Report) -> None:
    repo, number = PLATFORM_PR
    try:
        comments = subprocess.run(
            ['gh', 'pr', 'view', str(number), '--repo', repo, '--json', 'comments', '--jq', '.comments[].body'],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        report.check(False, f'UI preview for {repo}#{number}', f'gh failed: {type(error).__name__}')
        return
    urls = re.findall(r'Preview Deployment URL: (https://\S+)', comments)
    if not urls:
        report.check(False, f'UI preview for {repo}#{number}', 'no preview-deployment comment')
        return
    url = urls[-1]
    try:
        status = httpx.get(url, timeout=30, follow_redirects=True).status_code
    except httpx.HTTPError as error:
        report.check(False, 'UI preview responds', f'{url}: {type(error).__name__}')
        return
    report.check(status < 500, 'UI preview responds', f'{url} -> HTTP {status}')


def _uvx(report: Report) -> None:
    try:
        result = subprocess.run(UVX, capture_output=True, text=True, timeout=900, env={**os.environ, 'CLAI2_TEST': '1'})
    except (OSError, subprocess.SubprocessError) as error:
        report.check(False, 'colleague uvx command resolves', f'{type(error).__name__}')
        return
    ok = result.returncode == 0 and 'usage' in result.stdout.lower()
    tail = (result.stderr.strip().splitlines() or [''])[-1][:160]
    report.check(ok, 'colleague uvx command resolves (clai2 --help)', '' if ok else f'exit {result.returncode}: {tail}')


def _test_runs(report: Report) -> None:
    since = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    sql = (
        "SELECT count(*) AS runs FROM records WHERE span_name LIKE 'invoke_agent%' "
        "AND attributes->>'clai2.test' = 'true'"
    )
    try:
        with _client() as client:
            response = client.get('/v1/query', params={'sql': sql, 'min_timestamp': since, 'json_rows': 'true'})
    except httpx.HTTPError as error:
        report.check(None, 'no clai2.test runs in the last hour', f'query failed: {type(error).__name__}')
        return
    if response.status_code in (401, 403):
        report.check(None, 'no clai2.test runs in the last hour', 'this key cannot query (needs project:read_otlp)')
        return
    rows = response.json().get('rows') or [{}]
    runs = int(rows[0].get('runs') or 0)
    report.check(True if runs == 0 else None, 'no clai2.test runs in the last hour', f'{runs} test runs')


# Reset


def reset_agent(value: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Keep `model`, `display_name`, `policy.memory.shared`; everything else the demo adds goes."""
    kept: dict[str, Any] = {key: value[key] for key in ('model', 'display_name') if key in value}
    policy = value.get('policy') or {}
    shared = (policy.get('memory') or {}).get('shared')
    if shared is not None:
        kept['policy'] = {'memory': {'shared': shared}}
    removed: list[str] = []
    for block in value.get('instructions') or ():
        label = block.get('name') or block.get('id') if isinstance(block, dict) else None
        removed.append(f'instruction {label or str(block)[:60]!r}')
    removed += [f'skill {skill.get("name")}' for skill in value.get('skills') or ()]
    removed += [f'mcp_server {server.get("name")}' for server in value.get('mcp_servers') or ()]
    removed += [f'policy rule {rule.get("name")}' for rule in policy.get('rules') or ()]
    if policy.get('mcp'):
        removed.append('policy mcp allowlist')
    removed += [
        f'field {key}'
        for key in value
        if key not in ('model', 'display_name', 'policy', 'instructions', 'skills', 'mcp_servers')
    ]
    return kept, removed


def reset_proposals(value: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Accepted and dismissed proposals back to pending; the rest untouched."""
    proposals = []
    changed: list[str] = []
    for proposal in value.get('proposals') or ():
        if proposal.get('status') in REVIEWED:
            changed.append(f'{proposal.get("kind")} {proposal.get("id")}: {proposal["status"]} -> pending')
            proposal = {**proposal, 'status': 'pending', **dict.fromkeys(CLEARED_ON_RESET)}
        proposals.append(proposal)
    return {**value, 'proposals': proposals}, changed


def plan_reset(variables: dict[str, Any], *, suffix: str) -> dict[str, tuple[Any, list[str]]]:
    """What to write, per variable, with what changes; only variables that would change."""
    plans: dict[str, tuple[Any, list[str]]] = {}
    if (agent := variables.get(f'agent__clai2{suffix}')) is not None:
        _, value = _latest(agent)
        target, removed = reset_agent(value or {})
        if target != value:
            plans[f'agent__clai2{suffix}'] = (target, removed or ['(fields reordered)'])
    if (catalog := variables.get(f'catalog__clai2{suffix}')) is not None:
        _, value = _latest(catalog)
        items = (value or {}).get('items') or []
        if items:
            plans[f'catalog__clai2{suffix}'] = ({'items': []}, [f'{i.get("kind")} {i.get("name")}' for i in items])
    if (memory := variables.get(f'memory__clai2{suffix}')) is not None:
        _, value = _latest(memory)
        files = (value or {}).get('files') or []
        if files:
            plans[f'memory__clai2{suffix}'] = ({'files': []}, [f'repo note {f.get("path")}' for f in files])
    if (proposals := variables.get(f'fleet_proposals__clai2{suffix}')) is not None:
        _, value = _latest(proposals)
        target, changed = reset_proposals(value or {})
        if changed:
            plans[f'fleet_proposals__clai2{suffix}'] = (target, changed)
    return plans


def write(name: str, target: Any) -> str:
    """Write `target`, re-read, then point production at latest (or pin it to the deduplicated older version)."""
    import logfire
    from logfire.variables.config import LabeledValue, LabelRef

    instance = logfire.configure(
        local=True,
        send_to_logfire=False,
        console=False,
        api_key=os.environ['LOGFIRE_CLAI2_API_KEY'],
        advanced=logfire.AdvancedOptions(base_url=BASE_URL),
        variables=logfire.VariablesOptions(),
    )
    provider = instance.config.get_variable_provider()
    try:
        provider.refresh(force=True)
        config = provider.get_variable_config(name)
        assert config is not None, name
        latest = config.latest_version
        serialized = json.dumps(target)
        labels: dict[str, Any] = dict(config.labels)
        labels['production'] = LabeledValue(version=(latest.version if latest else 0) + 1, serialized_value=serialized)
        provider.update_variable(name, config.model_copy(update={'labels': labels, 'json_schema': None}))
        provider.refresh(force=True)
        config = provider.get_variable_config(name)
        assert config is not None and config.latest_version is not None, name
        if json.loads(config.latest_version.serialized_value) == target:
            labels = dict(config.labels)
            labels['production'] = LabelRef(ref='latest')
            provider.update_variable(name, config.model_copy(update={'labels': labels, 'json_schema': None}))
            provider.refresh(force=True)
            config = provider.get_variable_config(name)
            assert config is not None and config.latest_version is not None, name
            return f'v{config.latest_version.version}, production -> latest'
        production = config.labels.get('production')
        if isinstance(production, LabeledValue) and json.loads(production.serialized_value) == target:
            return f'an older version already held this value: production pinned to v{production.version}'
        raise RuntimeError(f'{name}: neither latest nor production holds the reset value after writing')
    finally:
        provider.shutdown()


def main() -> int:
    """Preflight, then the reset plan; writes only with `--reset --yes`."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--suffix', default='', help='Variable name suffix, such as `_test` to rehearse the reset.')
    parser.add_argument('--reset', action='store_true', help='Plan the reset (and with --yes, write it).')
    parser.add_argument('--yes', action='store_true', help='Actually write the reset.')
    parser.add_argument('--skip-slow', action='store_true', help='Skip the gateway, preview and uvx checks.')
    args = parser.parse_args()
    report = Report()
    variables = preflight(report, suffix=args.suffix, slow=not args.skip_slow)
    print(f'\npreflight: {report.failed} failed, {report.warned} warnings')
    plans = plan_reset(variables, suffix=args.suffix)
    print(f'\nreset{" (dry run)" if not (args.reset and args.yes) else ""}:')
    if not plans:
        print('  nothing to reset')
    for name, (_, changes) in plans.items():
        print(f'  {name}:')
        for change in changes:
            print(f'    - {change}')
    if args.reset and args.yes:
        for name, (target, _) in plans.items():
            print(f'  wrote {name}: {write(name, target)}')
    elif plans:
        print('  (run with --reset --yes to write)')
    return 1 if report.failed else 0


if __name__ == '__main__':
    sys.exit(main())
