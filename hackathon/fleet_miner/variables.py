"""Read and write the `fleet_proposals__clai2` managed variable through Logfire's public variables API."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx

from .models import ProposalsDoc

VARIABLE = 'fleet_proposals__clai2'
LABEL = 'production'


class VariablesClient:
    def __init__(self, api_key: str, *, base_url: str):
        self._client = httpx.AsyncClient(base_url=base_url, headers={'Authorization': f'bearer {api_key}'}, timeout=30)

    async def __aenter__(self) -> VariablesClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._client.aclose()

    async def _config(self) -> dict[str, Any] | None:
        response = await self._client.get('/v1/variables/')
        response.raise_for_status()
        return response.json().get('variables', {}).get(VARIABLE)

    async def read(self) -> ProposalsDoc | None:
        """The document the `production` label points at (the UI moves that label when it writes statuses)."""
        config = await self._config()
        value = _label_value(config, LABEL) if config else None
        return ProposalsDoc.model_validate_json(value) if value else None

    async def update(self, build: Callable[[ProposalsDoc | None], ProposalsDoc], *, attempts: int = 3) -> ProposalsDoc:
        """Read, merge, write, then re-read and verify, so a status the UI wrote meanwhile is not clobbered.

        `build` merges our proposals into whatever is current right now (the read happens just before the write,
        not when the miner started minutes earlier). If the re-read shows someone else's write landed after ours,
        merge again onto theirs. The API has no compare-and-swap, so a write landing in the milliseconds between our
        read and our write can't be prevented, only detected: the version then jumps by more than one.
        """
        for _ in range(attempts):
            before = await self._config()
            current_value = _label_value(before, LABEL) if before else None
            doc = build(ProposalsDoc.model_validate_json(current_value) if current_value else None)
            ours = doc.model_dump_json()
            await self.write(doc, exists=before is not None)
            after = await self._config()
            if after is not None and _label_value(after, LABEL) == ours:
                old_version = (before or {}).get('latest_version', {}).get('version', 0)
                if after['latest_version']['version'] > old_version + 1:
                    print('warning: another write landed between our read and write; re-check its statuses')
                return doc
            print('note: the variable changed right after our write; merging again onto the newer value')
        raise RuntimeError(f'could not write `{VARIABLE}` without racing another writer ({attempts} attempts)')

    async def write(self, doc: ProposalsDoc, *, exists: bool) -> None:
        label = {'target_type': 'version', 'serialized_value': doc.model_dump_json()}
        if exists:
            response = await self._client.put(f'/v1/variables/{VARIABLE}/', json={'labels': {LABEL: label}})
        else:
            response = await self._client.post(
                '/v1/variables/',
                json={
                    'name': VARIABLE,
                    'description': 'Fleet miner proposals for clai2 (hackathon): patterns many users typed, '
                    'drafted as skills or instructions to push to everyone.',
                    'json_schema': None,
                    'rollout': {'labels': {LABEL: 1.0}},
                    'overrides': [],
                    'labels': {LABEL: label},
                },
            )
        response.raise_for_status()


def _label_value(config: dict[str, Any], label: str, depth: int = 0) -> str | None:
    """A label holds a value, or points at `latest` or another label (the server stores a label on the newest version
    as a `latest` ref)."""
    target = (config.get('labels') or {}).get(label)
    if target is None or depth > 5:
        return None
    if 'serialized_value' in target:
        return target['serialized_value']
    ref = target.get('ref')
    if ref == 'latest':
        return (config.get('latest_version') or {}).get('serialized_value')
    return _label_value(config, ref, depth + 1) if ref else None
