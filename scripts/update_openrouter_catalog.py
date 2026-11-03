"""Generate `pydantic_ai_slim/pydantic_ai/providers/_openrouter_catalog.py` from the committed snapshot.

The snapshot (`scripts/openrouter_catalog_snapshot.json`) mirrors OpenRouter's public model list and
endpoint data for the fields `OpenRouterProvider.model_profile` uses. It is refreshed explicitly —
never at runtime or import time — by running:

    uv run python scripts/update_openrouter_catalog.py refresh

which fetches `https://openrouter.ai/api/v1/models` plus each route's endpoints, re-derives the
conservative route-level view, updates the snapshot, regenerates the module, and prints a semantic
change report for the refresh PR. The default subcommand regenerates the module from the committed
snapshot with no network access; `--check` fails instead of writing when the committed module is
stale, which is what keeps generation deterministic.

Route-level data is aggregated conservatively: a capability is only kept when every endpoint of the
route carries it, while hazards (mandatory reasoning, one-hour cache-write pricing) apply when any
endpoint has them. Fields these profiles don't rely on — `tool_choice` support, effort lists,
pricing and context limits — stay out of the catalog; pricing remains owned by `genai-prices`.
Routes that disappear upstream are dropped from the data but always reported, since accepting a
removal requires a re-probe first.
"""

from __future__ import annotations as _annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_PATH = REPO_ROOT / 'scripts' / 'openrouter_catalog_snapshot.json'
MODULE_PATH = REPO_ROOT / 'pydantic_ai_slim' / 'pydantic_ai' / 'providers' / '_openrouter_catalog.py'

MODELS_URL = 'https://openrouter.ai/api/v1/models'
ENDPOINTS_URL = 'https://openrouter.ai/api/v1/models/{route_id}/endpoints'

SNAPSHOT_SOURCES = ['https://openrouter.ai/api/v1/models', 'https://openrouter.ai/api/v1/models/{id}/endpoints']

# Fields where every endpoint must agree before the route-level view keeps them.
CAPABILITY_FIELDS = ('structured_outputs', 'tools')
# Fields that apply as soon as any endpoint of the route has them.
HAZARD_FIELDS = ('thinking_mandatory', 'cache_write_1h')


def _modality_flags(raw: dict[str, Any]) -> list[str] | None:
    """Input modalities declared by a model-list or endpoint entry, or `None` when unusable."""
    architecture = raw.get('architecture')
    modalities = architecture.get('input_modalities') if isinstance(architecture, dict) else None
    if isinstance(modalities, list) and all(isinstance(m, str) for m in modalities):
        return sorted(modalities)
    return None


def _parameter_flags(raw: dict[str, Any]) -> dict[str, Any]:
    """Capability flags from a `supported_parameters` list, omitting fields whose shape is unusable."""
    supported_parameters = raw.get('supported_parameters')
    if not isinstance(supported_parameters, list):
        return {}
    parameters = {parameter for parameter in supported_parameters if isinstance(parameter, str)}
    flags: dict[str, Any] = {}
    if 'structured_outputs' in parameters:
        flags['structured_outputs'] = True
    if 'tools' in parameters:
        flags['tools'] = True
    return flags


def _reasoning_flag(raw: dict[str, Any]) -> dict[str, Any]:
    """The `reasoning.mandatory` flag, omitted when the entry doesn't state it."""
    reasoning = raw.get('reasoning')
    if isinstance(reasoning, dict) and isinstance(reasoning.get('mandatory'), bool):
        return {'thinking_mandatory': reasoning['mandatory']}
    return {}


def _cache_write_1h_flag(raw: dict[str, Any]) -> dict[str, Any]:
    """Whether the entry's pricing exposes a one-hour cache-write price, omitted when pricing is unusable."""
    pricing = raw.get('pricing')
    if not isinstance(pricing, dict):
        return {}
    has_1h_cache_write = any(
        isinstance(key, str) and 'cache' in key and 'write' in key and '1h' in key for key in pricing
    )
    return {'cache_write_1h': has_1h_cache_write}


def _normalize(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize one upstream entry into the catalog shape, or `None` when its shape is unusable."""
    flags: dict[str, Any] = {}
    modalities = _modality_flags(raw)
    if modalities is not None:
        flags['input_modalities'] = modalities
    flags.update(_parameter_flags(raw))
    flags.update(_reasoning_flag(raw))
    flags.update(_cache_write_1h_flag(raw))
    return flags or None


def _aggregate(route_flags: dict[str, Any], endpoint_flags: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate route-level flags with per-endpoint flags.

    Capabilities survive only when every endpoint carries them, hazards when any endpoint does. When
    endpoint data is missing entirely, the model-list entry stands alone.
    """
    flags = dict(route_flags)
    modalities = flags.get('input_modalities')
    for endpoint in endpoint_flags:
        for field in CAPABILITY_FIELDS:
            if flags.get(field) is True and endpoint.get(field) is not True:
                flags.pop(field)
        if isinstance(modalities, list):
            endpoint_modalities = endpoint.get('input_modalities')
            modalities = [
                modality
                for modality in modalities
                if isinstance(endpoint_modalities, list) and modality in endpoint_modalities
            ]
        for field in HAZARD_FIELDS:
            if endpoint.get(field) is True:
                flags[field] = True
    if isinstance(modalities, list):
        if modalities:
            flags['input_modalities'] = sorted(modalities)
        else:
            flags.pop('input_modalities')
    return flags


def _build_entry(
    raw: dict[str, Any], endpoint_response: dict[str, Any] | None
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Build one catalog entry plus its endpoints view, both `None` when the upstream shape is unusable."""
    model_id = raw.get('id')
    if not isinstance(model_id, str) or '/' not in model_id:
        return None, None
    route_flags = _normalize(raw)
    endpoints: dict[str, dict[str, Any]] = {}
    endpoint_flags: list[dict[str, Any]] = []
    endpoint_data = endpoint_response.get('data') if isinstance(endpoint_response, dict) else None
    if isinstance(endpoint_data, list):
        for endpoint_raw in endpoint_data:
            if not isinstance(endpoint_raw, dict):
                continue
            endpoint_id = endpoint_raw.get('id')
            endpoint = _normalize(endpoint_raw)
            if isinstance(endpoint_id, str) and endpoint is not None:
                endpoints[endpoint_id] = endpoint
                endpoint_flags.append(endpoint)
    entry = _aggregate(route_flags, endpoint_flags) if route_flags is not None else None
    if entry is not None and endpoints:
        entry['endpoints'] = endpoints
    alias_target = raw.get('alias_target')
    return entry, alias_target if isinstance(alias_target, str) else None


def _fetch_snapshot() -> dict[str, Any]:
    """Fetch the public model list and endpoints, returning a fresh snapshot document."""
    import httpx

    models_response = httpx.get(MODELS_URL)
    models_response.raise_for_status()
    models_data = models_response.json().get('data')
    if not isinstance(models_data, list):
        raise ValueError(f'{MODELS_URL} returned no `data` list')

    models: dict[str, dict[str, Any]] = {}
    aliases: dict[str, str] = {}
    unusable = 0
    with httpx.Client() as client:
        for raw in models_data:
            if not isinstance(raw, dict):
                unusable += 1
                continue
            model_id = raw.get('id')
            if not isinstance(model_id, str) or '/' not in model_id:
                unusable += 1
                continue
            route_id = model_id.removeprefix('~')
            endpoint_response: dict[str, Any] | None = None
            try:
                endpoints_response = client.get(ENDPOINTS_URL.format(route_id=model_id))
                endpoints_response.raise_for_status()
                endpoint_response = endpoints_response.json()
            except (httpx.HTTPError, ValueError):
                # Endpoint data is optional: keep the model-list facts and let the report show the gap.
                endpoint_response = None
            entry, alias_target = _build_entry(raw, endpoint_response)
            if entry is None:
                unusable += 1
                continue
            if alias_target is not None:
                aliases[route_id] = alias_target.removeprefix('~')
            else:
                models[route_id] = entry

    if unusable:
        print(f'Skipped {unusable} model-list entries with an unusable shape.')
    today = date.today().isoformat()
    return {'aliases': aliases, 'generated_at': today, 'models': models, 'sources': SNAPSHOT_SOURCES}


def _render_value(value: Any) -> str:
    """Render a scalar or a short inline list, matching ruff format for these shapes."""
    if value is None:
        return 'None'
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, list):
        return '[' + ', '.join(repr(item) for item in value) + ']'
    return repr(value)


def _render_mapping(mapping: dict[str, Any], indent: str) -> str:
    """Render a mapping as an exploded dict literal, keys sorted, matching ruff format."""
    inner_indent = indent + '    '
    lines = []
    for key in sorted(mapping):
        value = mapping[key]
        if isinstance(value, dict):
            lines.append(f'{inner_indent}{key!r}: ' + _render_mapping(value, inner_indent) + ',')
        else:
            lines.append(f'{inner_indent}{key!r}: {_render_value(value)},')
    return '{\n' + '\n'.join(lines) + f'\n{indent}' + '}'


def _render_module(snapshot: dict[str, Any]) -> str:
    """Render the generated module for a snapshot document."""
    generated_at = snapshot.get('generated_at')
    generated_at = generated_at if isinstance(generated_at, str) else None
    models = snapshot.get('models')
    aliases = snapshot.get('aliases')
    models = models if isinstance(models, dict) else {}
    aliases = aliases if isinstance(aliases, dict) else {}
    header = f'''"""Catalog of OpenRouter route capabilities generated from `scripts/openrouter_catalog_snapshot.json`.

Generated by `scripts/update_openrouter_catalog.py`; edit the snapshot and regenerate instead of
editing this file. Import is pure: this module is plain data and performs no I/O.
"""

from __future__ import annotations as _annotations

from typing_extensions import TypedDict


class OpenRouterCatalogEndpoint(TypedDict, total=False):
    """Per-endpoint facts behind the route-level aggregation in `OpenRouterCatalogEntry`."""

    cache_write_1h: bool
    input_modalities: list[str]
    structured_outputs: bool
    thinking_mandatory: bool
    tools: bool


class OpenRouterCatalogEntry(TypedDict, total=False):
    """Profile-used facts for one OpenRouter route, conservatively aggregated across its endpoints.

    A capability is only kept when every endpoint carries it; hazards such as mandatory reasoning
    apply when any endpoint has it. `cache_write_1h` and `endpoints` are carried for review and for
    `_CATALOG_CORRECTIONS` authoring; no profile flag is derived from them yet.
    """

    cache_write_1h: bool
    endpoints: dict[str, OpenRouterCatalogEndpoint]
    input_modalities: list[str]
    structured_outputs: bool
    thinking_mandatory: bool
    tools: bool


CATALOG_GENERATED_AT: str | None = {generated_at!r}
"""UTC date the upstream data was fetched, `None` while the snapshot carries no fetched data."""

OPENROUTER_MODEL_CATALOG: dict[str, OpenRouterCatalogEntry] = {_render_mapping(models, '') if models else '{}'}
"""Route capability facts keyed by OpenRouter model id."""

OPENROUTER_CATALOG_ALIASES: dict[str, str] = {_render_mapping(aliases, '') if aliases else '{}'}
"""Alias id to canonical catalog id, for routes OpenRouter lists under an alias."""
'''
    return header


def _load_snapshot() -> dict[str, Any]:
    with SNAPSHOT_PATH.open(encoding='utf-8') as stream:
        snapshot = json.load(stream)
    if not isinstance(snapshot, dict):
        raise ValueError(f'{SNAPSHOT_PATH} is not a JSON object')
    return snapshot


def _changed_fields(old: Any, new: Any) -> list[str]:
    """Human-readable field-level differences between two entry values."""
    old = old if isinstance(old, dict) else {}
    new = new if isinstance(new, dict) else {}
    lines = []
    for field in sorted(set(old) | set(new)):
        if old.get(field) != new.get(field):
            lines.append(f'{field}: {old.get(field)!r} -> {new.get(field)!r}')
    return lines


def _semantic_changes(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    """Semantic changes between two snapshot documents, ordered so the important ones come first."""
    old_models = old.get('models') if isinstance(old.get('models'), dict) else {}
    new_models = new.get('models') if isinstance(new.get('models'), dict) else {}
    old_aliases = old.get('aliases') if isinstance(old.get('aliases'), dict) else {}
    new_aliases = new.get('aliases') if isinstance(new.get('aliases'), dict) else {}

    lines = []
    for route_id in sorted(set(old_models) - set(new_models)):
        lines.append(f'REMOVED route `{route_id}` — requires a re-probe before accepting this removal')
    for route_id in sorted(set(old_models) | set(new_models)):
        changes = _changed_fields(old_models.get(route_id), new_models.get(route_id))
        if route_id not in old_models:
            lines.append(f'NEW route `{route_id}`: ' + ('; '.join(changes) or 'no flags'))
        elif changes:
            prefix = (
                'MANDATORY-REASONING FLIP'
                if any(c.startswith('thinking_mandatory') for c in changes)
                else 'UPSTREAM CHANGE'
            )
            lines.append(f'{prefix} `{route_id}`: ' + '; '.join(changes))
    for alias_id in sorted(set(old_aliases) | set(new_aliases)):
        if old_aliases.get(alias_id) != new_aliases.get(alias_id):
            lines.append(f'ALIAS CHANGE `{alias_id}`: {old_aliases.get(alias_id)!r} -> {new_aliases.get(alias_id)!r}')
    return lines


def _refresh() -> int:
    """Fetch upstream, update the snapshot, regenerate the module, and print the change report."""
    old = _load_snapshot()
    new = _fetch_snapshot()
    changes = _semantic_changes(old, new)
    SNAPSHOT_PATH.write_text(json.dumps(new, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    MODULE_PATH.write_text(_render_module(new), encoding='utf-8')
    print(f'Snapshot updated: {len(new.get("models", {}))} routes, {len(new.get("aliases", {}))} aliases.')
    if changes:
        print('\n'.join(changes))
    else:
        print('No semantic changes.')
    return 0


def _generate(check: bool) -> int:
    """Regenerate the module from the committed snapshot; with `--check`, verify it is up to date."""
    rendered = _render_module(_load_snapshot())
    if check:
        committed = MODULE_PATH.read_text(encoding='utf-8')
        if committed != rendered:
            print(
                f'{MODULE_PATH} is stale; run `uv run python scripts/update_openrouter_catalog.py generate`',
                file=sys.stderr,
            )
            return 1
        print(f'{MODULE_PATH} is up to date.')
        return 0
    MODULE_PATH.write_text(rendered, encoding='utf-8')
    print(f'Wrote {MODULE_PATH}.')
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to `refresh` or `generate`."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    subparsers.add_parser('refresh', help='fetch OpenRouter and update the snapshot and module')
    generate_parser = subparsers.add_parser('generate', help='regenerate the module from the committed snapshot')
    generate_parser.add_argument(
        '--check', action='store_true', help='fail instead of writing when the module is stale'
    )
    args = parser.parse_args(argv)
    if args.command == 'refresh':
        return _refresh()
    return _generate(check=args.check)


if __name__ == '__main__':
    sys.exit(main())
