"""Pax offers an extra for every `pydantic-ai-slim` extra it does not deliberately skip.

`pydantic_ai_slim/pyproject.toml` is the source of truth: a new slim extra fails here until Pax mirrors it
or `EXCLUDED` in `scripts/sync_provider_extras.py` names it with a reason. That script writes the extras;
these tests read the projects, not the installed metadata, so they are not VCR tests.
"""

import importlib.util
import tomllib
from pathlib import Path
from typing import Protocol, runtime_checkable

import pytest

ROOT = Path(__file__).parents[2]


@runtime_checkable
class SyncScript(Protocol):
    EXCLUDED: dict[str, str]
    LOCAL: str
    COMMAND: str

    def main(self, argv: list[str]) -> int: ...


def _load_script() -> SyncScript:
    spec = importlib.util.spec_from_file_location(
        'sync_provider_extras', ROOT / 'src' / 'pydantic_ai_pax' / 'scripts' / 'sync_provider_extras.py'
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert isinstance(module, SyncScript)
    return module


SCRIPT = _load_script()
EXCLUDED = SCRIPT.EXCLUDED
LOCAL = SCRIPT.LOCAL
THIS = 'src/pydantic_ai_pax/scripts/sync_provider_extras.py'


def _extras(path: Path) -> dict[str, list[str]]:
    return tomllib.loads(path.read_text(encoding='utf-8'))['tool']['hatch']['metadata']['hooks'][
        'uv-dynamic-versioning'
    ]['optional-dependencies']


def _pin(dependencies: list[str], package: str) -> str:
    """The version pin Pax's dependency on `package` uses, such as `==0.{{ minor }}...`."""
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
        'For a new slim extra, exclude it there with a reason if Pax should not offer it, then run '
        f'`{SCRIPT.COMMAND}`.'
    )


def test_all_installs_every_extra_but_local() -> None:
    sandboxes = sorted(
        name for name, deps in PAX.items() if name != LOCAL and deps[0].startswith('pydantic-ai-harness[')
    )
    assert PAX['all'] == [
        f'pydantic-ai-slim[{",".join(sorted(MIRRORED))}]{SLIM_PIN}',
        f'pydantic-ai-harness[{",".join(sandboxes)}]{HARNESS_PIN}',
    ], f"Pax's `all` extra is out of date; run `{SCRIPT.COMMAND}`."


def test_every_pax_extra_is_a_provider_a_sandbox_or_all() -> None:
    """So the two tests above see every extra: a new kind of extra needs a decision about `all`."""
    kinds = {name: deps[0].partition('[')[0] for name, deps in PAX.items() if name != 'all' and len(deps) == 1}
    assert set(kinds) | {'all'} == set(PAX)
    assert set(kinds.values()) == {'pydantic-ai-slim', 'pydantic-ai-harness'}


def test_generated_extras_are_current(capsys: pytest.CaptureFixture[str]) -> None:
    """Pax's generated block and the `pydantic-clai2` shim's extras match what the script writes."""
    assert SCRIPT.main(['--check']) == 0, capsys.readouterr().out
