"""Pax offers an extra for every `pydantic-ai-slim` extra it does not deliberately skip.

`pydantic_ai_slim/pyproject.toml` is the source of truth: a new slim extra fails here until Pax mirrors it
or `EXCLUDED` names it with a reason. It reads the projects, not the installed metadata, so it is not a VCR test.
"""

import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[2]
THIS = 'tests/pax/test_provider_extras.py'

EXCLUDED: dict[str, str] = {
    'logfire': 'Pax depends on `logfire[httpx]` directly.',
    'sentence-transformers': 'Embeddings only; Pax runs no embedder.',
    'voyageai': 'Embeddings only; Pax runs no embedder.',
    'duckduckgo': 'A tool, not a model provider; plugins install tools through `pydantic-ai-harness` extras.',
    'tavily': 'A tool, not a model provider; plugins install tools through `pydantic-ai-harness` extras.',
    'exa': 'A tool, not a model provider; plugins install tools through `pydantic-ai-harness` extras.',
    'web-fetch': 'A tool, not a model provider; plugins install tools through `pydantic-ai-harness` extras.',
    'cli': 'The `clai` CLI; Pax is its own CLI and depends on what it needs directly.',
    'realtime': 'Speech-to-speech sessions, which Pax does not run.',
    'openai-realtime': 'Speech-to-speech sessions, which Pax does not run.',
    'google-realtime': 'Speech-to-speech sessions, which Pax does not run.',
    'xai-realtime': 'Speech-to-speech sessions, which Pax does not run.',
    'elevenlabs-agents-realtime': 'Speech-to-speech sessions, which Pax does not run.',
    'elevenlabs-realtime': 'Speech-to-speech sessions, which Pax does not run.',
    'mcp': 'Installed by default.',
    'mcp-tasks': 'Explicit MCP task execution pulls the FastMCP server stack; Pax does not use it.',
    'evals': 'Pydantic Evals, which Pax does not use.',
    'ui': 'UI adapters mounted in your own app; Pax serves its own web UI.',
    'ag-ui': 'UI adapters mounted in your own app; Pax serves its own web UI.',
    'web': '`Agent.to_web()`; Pax serves its own web UI with dependencies it declares.',
    'retries': 'The legacy HTTPX retry helpers, which Pax does not use.',
    'temporal': 'Durable execution, which Pax does not use.',
    'dbos': 'Durable execution, which Pax does not use.',
    'prefect': 'Durable execution, which Pax does not use.',
    'spec': 'YAML agent specs, which Pax does not load.',
    'jsonschema': '`StructuredDict(validate=True)`, which Pax does not use.',
}
"""The `pydantic-ai-slim` extras Pax does not mirror, each with the reason."""

LOCAL = 'local'
"""The `/sandbox` extra `all` leaves out: its SDKs have no Windows wheel."""


def _extras(path: Path) -> dict[str, list[str]]:
    return tomllib.loads(path.read_text(encoding='utf-8'))['tool']['hatch']['metadata']['hooks'][
        'uv-dynamic-versioning'
    ]['optional-dependencies']


def _pin(dependencies: list[str], package: str) -> str:
    """The version pin Pax's dependency on `package` uses, such as `==1.{{ minor }}...`."""
    [requirement] = [dependency for dependency in dependencies if dependency.startswith(f'{package}[')]
    return requirement.partition(']')[2]


SLIM = _extras(ROOT / 'pydantic_ai_slim' / 'pyproject.toml')
PAX_PROJECT = tomllib.loads((ROOT / 'src' / 'pydantic_ai_pax' / 'pyproject.toml').read_text())
PAX_HOOK = PAX_PROJECT['tool']['hatch']['metadata']['hooks']['uv-dynamic-versioning']
PAX: dict[str, list[str]] = PAX_HOOK['optional-dependencies']
SLIM_PIN = _pin(PAX_HOOK['dependencies'], 'pydantic-ai-slim')
HARNESS_PIN = _pin(PAX_HOOK['dependencies'], 'pydantic-ai-harness')
MIRRORED = [name for name in SLIM if name not in EXCLUDED]


def test_excluded_extras_exist_in_slim() -> None:
    gone = sorted(set(EXCLUDED) - set(SLIM))
    assert not gone, f'`pydantic-ai-slim` no longer has {gone}: remove them from `EXCLUDED` in {THIS}.'


def test_pax_mirrors_every_slim_extra_it_does_not_exclude() -> None:
    mirrored = {name: deps for name, deps in PAX.items() if name != 'all' and deps[0].startswith('pydantic-ai-slim[')}
    expected = {name: [f'pydantic-ai-slim[{name}]{SLIM_PIN}'] for name in MIRRORED}
    assert mirrored == expected, (
        "Pax's provider extras in src/pydantic_ai_pax/pyproject.toml must be every `pydantic-ai-slim` extra "
        f"except those in `EXCLUDED` in {THIS}, each pinned like Pax's own `pydantic-ai-slim` dependency. "
        'For a new slim extra, add it to Pax (and `all`) or exclude it with a reason.'
    )


def test_all_installs_every_extra_but_local() -> None:
    sandboxes = sorted(
        name for name, deps in PAX.items() if name != LOCAL and deps[0].startswith('pydantic-ai-harness[')
    )
    assert PAX['all'] == [
        f'pydantic-ai-slim[{",".join(sorted(MIRRORED))}]{SLIM_PIN}',
        f'pydantic-ai-harness[{",".join(sandboxes)}]{HARNESS_PIN}',
    ], "Pax's `all` extra in src/pydantic_ai_pax/pyproject.toml must list every other extra but `local`."


def test_every_pax_extra_is_a_provider_a_sandbox_or_all() -> None:
    """So the two tests above see every extra: a new kind of extra needs a decision about `all`."""
    kinds = {name: deps[0].partition('[')[0] for name, deps in PAX.items() if name != 'all' and len(deps) == 1}
    assert set(kinds) | {'all'} == set(PAX)
    assert set(kinds.values()) == {'pydantic-ai-slim', 'pydantic-ai-harness'}
