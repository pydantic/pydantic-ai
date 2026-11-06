"""Reject a weekly CE audit publication that differs from the reviewed proposal.

The agent's post-step validates coverage, the edit boundary and the independent
Task review against `.ce-audit/review/proposal.json`. This script runs in the
safe-outputs job, where the ingested items are available, and proves the one
`create_pull_request` item is exactly that proposal. It also rejects a `noop`
published alongside a pull request.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from pydantic import TypeAdapter

PROPOSAL_KEYS = ('title', 'body', 'branch')
MAPPING_ADAPTER = TypeAdapter(dict[str, object])
ITEMS_ADAPTER = TypeAdapter(list[dict[str, object]])


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError('Expected a JSON object')
    return MAPPING_ADAPTER.validate_python(value)


def _items(path: pathlib.Path) -> list[dict[str, object]]:
    """Read the ingested safe-output items, failing closed when the file is absent."""
    if not path.is_file():
        raise ValueError(f'ingested agent output is missing: {path}')
    payload: object = json.loads(path.read_text(encoding='utf-8'))
    items = _mapping(payload).get('items')
    if not isinstance(items, list):
        raise ValueError('ingested agent output has no items list')
    return ITEMS_ADAPTER.validate_python(items)


def _proposal(item: dict[str, object]) -> dict[str, object]:
    return {key: item[key] for key in PROPOSAL_KEYS}


def verify(safe_outputs: pathlib.Path, reviewed: pathlib.Path) -> None:
    """Reject a publication that no reviewed proposal authorizes."""
    items = _items(safe_outputs)
    proposals = [item for item in items if item.get('type') == 'create_pull_request']
    has_noop = any(item.get('type') == 'noop' for item in items)
    if has_noop and proposals:
        raise ValueError('A noop cannot publish a pull request')
    if not proposals:
        return
    if len(proposals) > 1:
        raise ValueError('Only one focused pull request is permitted')
    if not reviewed.is_file():
        raise ValueError('Published pull request has no independently reviewed proposal')
    expected = _mapping(json.loads(reviewed.read_text(encoding='utf-8')))
    published = _proposal(proposals[0])
    if any(expected.get(key) != published.get(key) for key in PROPOSAL_KEYS):
        raise ValueError('Published PR metadata differs from the reviewed proposal')


def main() -> int:
    """Run the verifier from the safe-outputs job."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agent-output', type=pathlib.Path, required=True)
    parser.add_argument('--reviewed', type=pathlib.Path, required=True)
    args = parser.parse_args()
    try:
        verify(args.agent_output, args.reviewed)
    except (OSError, ValueError) as error:
        print(f'weekly CE audit publication rejected: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
