"""`pydantic-clai2` is now `pydantic-ai-pax`: the new commands, and the old package and command names that still work."""

import ast
import importlib
import importlib.util
import io
import sys
import tokenize
import tomllib
from pathlib import Path
from types import ModuleType

import pytest

import pydantic_ai_pax
import pydantic_ai_pax.commands
import pydantic_ai_pax.plugins
from pydantic_ai_pax.cli.self_update import Update
from tests.conftest import try_import

with try_import() as shim_installed:
    import pydantic_clai2
    from pydantic_clai2 import __main__ as clai2_main  # pragma: lax no cover

with try_import() as jinja_installed:
    import jinja2

# The `pydantic-ai-pax` test cell installs no `pydantic-clai2`; the cells with every workspace package do.
# A pull request confined to `pydantic-ai-pax` runs only that cell, so its tests are `lax no cover`.
requires_shim = pytest.mark.skipif(not shim_installed(), reason='pydantic-clai2 not installed')
SOURCES = Path(__file__).parents[2] / 'src'


def test_pax_is_the_command_and_clai2_its_deprecated_name() -> None:
    """Read from the projects, since a test cell installs `pydantic-ai-pax` without the `pydantic-clai2` shim."""
    pax = tomllib.loads((SOURCES / 'pydantic_ai_pax' / 'pyproject.toml').read_text())
    shim = tomllib.loads((SOURCES / 'pydantic_clai2' / 'pyproject.toml').read_text())
    assert pax['project']['scripts'] == {
        'pax': 'pydantic_ai_pax.__main__:main',
        'pydantic-ai-pax': 'pydantic_ai_pax.__main__:main',
    }
    assert shim['project']['scripts'] == {
        'clai2': 'pydantic_clai2.__main__:main',
        'pydantic-clai2': 'pydantic_clai2.__main__:main',
    }
    version = pax['tool']['uv-dynamic-versioning']['format-jinja']
    assert shim['tool']['uv-dynamic-versioning']['format-jinja'] == version
    pax_hook = pax['tool']['hatch']['metadata']['hooks']['uv-dynamic-versioning']
    shim_hook = shim['tool']['hatch']['metadata']['hooks']['uv-dynamic-versioning']
    assert shim_hook['dependencies'] == [
        f'pydantic-ai-pax{{% if env.PAX_SDIST_URL %}} @ {{{{ env.PAX_SDIST_URL }}}}{{% else %}}=={version}{{% endif %}}'
    ]
    assert shim_hook['optional-dependencies'] == {
        extra: [
            f'pydantic-ai-pax[{extra}]{{% if env.PAX_SDIST_URL %}} @ {{{{ env.PAX_SDIST_URL }}}}{{% else %}}=={version}{{% endif %}}'
        ]
        for extra in pax_hook['optional-dependencies']
    }


@pytest.mark.skipif(not jinja_installed(), reason='jinja2 not installed')
def test_main_channel_shim_pins_pax_sdist() -> None:  # pragma: lax no cover
    """The `main` channel's `pydantic-clai2` sdist names Pax's sdist in the same release, as `/update` does.

    A build from before the rename updates `pydantic-clai2` with `--overrides` that do not name Pax, and PyPI has
    no Pax at a `main` commit's version, so an exact version pin there leaves uv nothing to install. Each extra
    names the same sdist, so no Pax requirement in the `main` sdist pins a version.
    It reads build metadata and renders its templates, so it is not a VCR test.
    """
    shim = tomllib.loads((SOURCES / 'pydantic_clai2' / 'pyproject.toml').read_text())
    hook = shim['tool']['hatch']['metadata']['hooks']['uv-dynamic-versioning']
    [template] = hook['dependencies']
    commit = 'c' * 40
    pax = Update(channel='main', target=commit).requirement('pydantic-ai-pax')
    url = pax.partition(' @ ')[2]

    def render(template: str, env: dict[str, str]) -> str:
        return jinja2.Template(template).render(env=env, minor=75, patch=1, distance=3, commit='ccccccccc')

    assert render(template, {'PAX_SDIST_URL': url}) == pax
    assert render(template, {}) == 'pydantic-ai-pax==0.75.1.dev3+ccccccccc'
    extras: dict[str, list[str]] = hook['optional-dependencies']
    assert extras
    for extra, [extra_template] in extras.items():
        assert render(extra_template, {'PAX_SDIST_URL': url}) == f'pydantic-ai-pax[{extra}] @ {url}'
        assert render(extra_template, {}) == f'pydantic-ai-pax[{extra}]==0.75.1.dev3+ccccccccc'
    script = (SOURCES / 'pydantic_ai_pax' / 'scripts' / 'build_bleeding.sh').read_text()
    assert 'PAX_SDIST_URL=${release%/}/pydantic_ai_pax-$commit.tar.gz uv build' in script
    assert shim['tool']['hatch']['metadata']['allow-direct-references'] is True


def test_code_names_say_pax_not_clai2() -> None:
    """Identifiers and `ContextVar` names use the new name.

    Persisted keys, `CLAI2_*` variables, and the `pydantic_clai2` alias keep the old one, in strings only.
    It reads source only, so it is not a VCR test.
    """
    package = SOURCES / 'pydantic_ai_pax' / 'pydantic_ai_pax'
    found: list[str] = []
    for path in sorted(package.rglob('*.py')):
        source = path.read_text()
        if 'clai2' not in source.lower():
            continue
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        names = [token.string for token in tokens if token.type == tokenize.NAME and 'clai2' in token.string.lower()]
        context_vars = [
            node.args[0].value
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == 'ContextVar'
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and 'clai2' in str(node.args[0].value)
        ]
        found += [f'{path.relative_to(package)}: {name}' for name in [*names, *context_vars]]
    assert found == []


@requires_shim
def test_clai2_points_at_pax_then_runs_it(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:  # pragma: lax no cover
    ran: list[bool] = []
    monkeypatch.setattr(clai2_main, 'pax_main', lambda: ran.append(True))
    clai2_main.main()
    assert ran == [True]
    assert capsys.readouterr().err == (
        '`clai2` is now `pax`: install it with `uv tool install pydantic-ai-pax`. '
        'The `clai2` and `pydantic-clai2` commands will be removed in a future release.\n'
    )


@requires_shim
@pytest.mark.parametrize(
    ('old', 'new'),
    [('pydantic_clai2.plugins', pydantic_ai_pax.plugins), ('pydantic_clai2.commands', pydantic_ai_pax.commands)],
)
def test_old_import_paths_are_the_new_modules(
    monkeypatch: pytest.MonkeyPatch, old: str, new: ModuleType
) -> None:  # pragma: lax no cover
    monkeypatch.delitem(sys.modules, old, raising=False)
    with pytest.warns(DeprecationWarning, match=rf'^`{old}` is deprecated, import `{new.__name__}` instead\.$'):
        assert importlib.import_module(old) is new


@requires_shim
def test_a_missing_old_module_is_missing() -> None:  # pragma: lax no cover
    """Probing for an optional submodule must not find one `pydantic_ai_pax` lacks, nor warn about it."""
    assert importlib.util.find_spec('pydantic_clai2.missing') is None
    with pytest.raises(ModuleNotFoundError, match=r"^No module named 'pydantic_clai2\.missing'$"):
        importlib.import_module('pydantic_clai2.missing')


@requires_shim
def test_old_top_level_names_are_the_new_ones() -> None:  # pragma: lax no cover
    with pytest.warns(
        DeprecationWarning, match=r'^`pydantic_clai2.create_agent` is deprecated, import `pydantic_ai_pax.create_agent`'
    ):
        from pydantic_clai2 import create_agent
    assert create_agent is pydantic_ai_pax.create_agent
    assert pydantic_clai2.__all__ == pydantic_ai_pax.__all__
    with pytest.raises(AttributeError):
        pydantic_clai2.missing


@requires_shim
def test_the_old_entry_point_stays_its_own_module() -> None:  # pragma: lax no cover
    """`pydantic_ai_pax.__main__`, once imported, is an attribute of that package but not an export to forward."""
    assert importlib.import_module('pydantic_ai_pax.__main__') is not clai2_main
    assert importlib.import_module('pydantic_clai2.__main__') is clai2_main
    with pytest.raises(AttributeError):
        pydantic_clai2.__getattr__('__main__')
