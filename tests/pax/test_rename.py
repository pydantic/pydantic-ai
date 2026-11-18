"""`pydantic-clai2` is now `pydantic-ai-pax`: the new commands, and the old package and command names that still work."""

import importlib
import sys
from importlib import metadata
from types import ModuleType

import pytest
from packaging.requirements import Requirement

import pydantic_ai_pax.commands
import pydantic_ai_pax.plugins
from pydantic_clai2 import __main__ as clai2_main


def _scripts(distribution: str) -> dict[str, str]:
    return {
        entry.name: entry.value
        for entry in metadata.distribution(distribution).entry_points
        if entry.group == 'console_scripts'
    }


def test_pax_is_the_command_and_clai2_its_deprecated_name() -> None:
    assert _scripts('pydantic-ai-pax') == {
        'pax': 'pydantic_ai_pax.__main__:main',
        'pydantic-ai-pax': 'pydantic_ai_pax.__main__:main',
    }
    assert _scripts('pydantic-clai2') == {
        'clai2': 'pydantic_clai2.__main__:main',
        'pydantic-clai2': 'pydantic_clai2.__main__:main',
    }
    [pax] = [Requirement(text) for text in metadata.requires('pydantic-clai2') or [] if 'extra ==' not in text]
    assert (pax.name, str(pax.specifier)) == ('pydantic-ai-pax', f'=={metadata.version("pydantic-ai-pax")}')


def test_clai2_points_at_pax_then_runs_it(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    ran: list[bool] = []
    monkeypatch.setattr(clai2_main, 'pax_main', lambda: ran.append(True))
    clai2_main.main()
    assert ran == [True]
    assert capsys.readouterr().err == (
        '`clai2` is now `pax`: install it with `uv tool install pydantic-ai-pax`. '
        'The `clai2` and `pydantic-clai2` commands will be removed in a future release.\n'
    )


@pytest.mark.parametrize(
    ('old', 'new'),
    [('pydantic_clai2.plugins', pydantic_ai_pax.plugins), ('pydantic_clai2.commands', pydantic_ai_pax.commands)],
)
def test_old_import_paths_are_the_new_modules(monkeypatch: pytest.MonkeyPatch, old: str, new: ModuleType) -> None:
    monkeypatch.delitem(sys.modules, old, raising=False)
    with pytest.warns(DeprecationWarning, match=rf'^`{old}` is deprecated, import `{new.__name__}` instead\.$'):
        assert importlib.import_module(old) is new
