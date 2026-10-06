"""Read and write the `fleet_proposals__clai2` managed variable through Logfire's public variables API."""

from __future__ import annotations

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

    async def read(self) -> ProposalsDoc | None:
        response = await self._client.get('/v1/variables/')
        response.raise_for_status()
        config = response.json().get('variables', {}).get(VARIABLE)
        if config is None:
            return None
        value = _label_value(config, LABEL)
        return ProposalsDoc.model_validate_json(value) if value else None

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
