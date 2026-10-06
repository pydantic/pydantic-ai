"""Restore a known-good demo state for the clai2 fleet variables on EU staging (hackathon).

    uv run --env-file .env python hackathon/demo_reset.py            # print the plan and the diff, change nothing
    uv run --env-file .env python hackathon/demo_reset.py --yes      # apply it

What it does, for `agent__clai2`, `catalog__clai2` and `fleet_proposals__clai2`:

- `agent__clai2` becomes the curated config in `AGENT` below, built from the current production value. It
  keeps the `policy.rules` that production has and resets every rule to `observe` except Monty rules, plus
  the named instruction `run-tests-first` and the `pr-shepherd` skill.
- `catalog__clai2` becomes `CATALOG` below, unless `--keep-catalog` is passed.
- `fleet_proposals__clai2` keeps every proposal, with `status` reset to `pending` and acceptance cleared.
- Each variable gets a new version holding that value. `production` is set to follow `latest`, and every
  targeting override is removed. Other labels, such as `test`, are pointed at `production`.

Needs `LOGFIRE_CLAI2_API_KEY` with `project:read_variables` and `project:write_variables`.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
from typing import Any

import logfire
from logfire.variables.config import LabeledValue, LabelRef, VariableConfig

BASE_URL = 'https://logfire-eu.pydantic.info'
AGENT, CATALOG, PROPOSALS = 'agent__clai2', 'catalog__clai2', 'fleet_proposals__clai2'

RUN_TESTS_FIRST = {
    'name': 'run-tests-first',
    'instructions': 'Company rule (from Logfire): run the relevant tests before opening a PR.',
}
PR_SHEPHERD = {
    'name': 'pr-shepherd',
    'description': 'After opening a PR, keep watching CI and agentic review comments and iterate until it is ready',
    'instructions': (
        'When you open a PR: watch CI with `gh pr checks --watch`, read every new review comment and review summary '
        '(including bots), fix and push, and repeat until CI is green and no unresolved comments remain. Report '
        'what you changed each round.'
    ),
    'source': 'fleet-miner',
    'why': '4 teammates kept asking their agent to babysit PRs',
}


def _value(config: VariableConfig, label: str = 'production') -> Any:
    """The JSON value a label serves, following refs to `latest` or other labels."""
    seen: set[str] = set()
    current: Any = config.labels.get(label)
    while isinstance(current, LabelRef) and current.ref not in seen:
        seen.add(current.ref)
        current = config.latest_version if current.ref == 'latest' else config.labels.get(current.ref)
    serialized = getattr(current, 'serialized_value', None)
    return json.loads(serialized) if serialized else None


def curated_agent(current: dict[str, Any] | None) -> dict[str, Any]:
    current = dict(current or {})
    policy = dict(current.get('policy') or {})
    rules = [{**rule, 'mode': 'enforce' if rule.get('monty') else 'observe'} for rule in policy.get('rules') or []]
    if rules or policy:
        policy['rules'] = rules
    value: dict[str, Any] = {'instructions': [RUN_TESTS_FIRST], 'skills': [PR_SHEPHERD]}
    if policy:
        value['policy'] = policy
    return value


def reset_proposals(current: dict[str, Any] | None) -> dict[str, Any]:
    value = dict(current or {'proposals': []})
    value['proposals'] = [
        {**proposal, 'status': 'pending', 'accepted_tier': None, 'accepted_at': None}
        for proposal in value.get('proposals') or []
    ]
    return value


def _diff(name: str, before: Any, after: Any) -> str:
    lines = difflib.unified_diff(
        json.dumps(before, indent=2, sort_keys=True).splitlines(),
        json.dumps(after, indent=2, sort_keys=True).splitlines(),
        f'{name} (production now)',
        f'{name} (after reset)',
        lineterm='',
    )
    return '\n'.join(lines) or f'{name}: value unchanged'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--yes', action='store_true', help='apply the reset; without it nothing is written')
    parser.add_argument('--keep-catalog', action='store_true', help='leave catalog__clai2 as it is')
    parser.add_argument('--catalog-file', help='JSON file with the catalog value to restore')
    args = parser.parse_args()

    instance = logfire.configure(
        local=True,
        send_to_logfire=False,
        console=False,
        api_key=os.environ['LOGFIRE_CLAI2_API_KEY'],
        advanced=logfire.AdvancedOptions(base_url=BASE_URL),
        variables=logfire.VariablesOptions(),
    )
    provider = instance.config.get_variable_provider()
    provider.refresh(force=True)
    configs = {name: provider.get_variable_config(name) for name in (AGENT, CATALOG, PROPOSALS)}
    missing = [name for name, config in configs.items() if config is None]
    if missing:
        raise SystemExit(f'Missing variables: {", ".join(missing)}')

    targets: dict[str, Any] = {
        AGENT: curated_agent(_value(configs[AGENT])),  # pyright: ignore[reportArgumentType]
        PROPOSALS: reset_proposals(_value(configs[PROPOSALS])),  # pyright: ignore[reportArgumentType]
    }
    if not args.keep_catalog:
        if args.catalog_file:
            with open(args.catalog_file) as file:
                targets[CATALOG] = json.load(file)
        else:
            targets[CATALOG] = _value(configs[CATALOG])  # pyright: ignore[reportArgumentType]

    for name, target in targets.items():
        config = configs[name]
        assert config is not None
        print(_diff(name, _value(config), target))
        extra = sorted(label for label in config.labels if label != 'production')
        print(
            f'{name}: labels {sorted(config.labels)}; overrides {len(config.overrides)}; other labels -> production: {extra}'
        )
        print()

    if not args.yes:
        print('Dry run: nothing written. Re-run with --yes to apply.')
        return

    for name, target in targets.items():
        config = configs[name]
        assert config is not None
        latest = getattr(config.latest_version, 'version', 0) or 0
        labels: dict[str, LabeledValue | LabelRef] = {
            label: LabelRef(ref='production') for label in config.labels if label != 'production'
        }
        labels['production'] = LabeledValue(version=latest + 1, serialized_value=json.dumps(target))
        provider.update_variable(
            name, config.model_copy(update={'labels': labels, 'overrides': [], 'json_schema': None})
        )
        provider.refresh(force=True)
        updated = provider.get_variable_config(name)
        assert updated is not None
        labels = dict(updated.labels)
        labels['production'] = LabelRef(ref='latest')
        provider.update_variable(name, updated.model_copy(update={'labels': labels, 'json_schema': None}))
        print(f'{name}: production now follows latest (v{latest + 1} written)')
    provider.shutdown()


if __name__ == '__main__':
    main()
