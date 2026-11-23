"""Every key-reading surface pax constructs is registered, and each key does there what its cell declares."""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import pytest

import pydantic_ai_pax
from tests.manifests import assert_classified
from tests.pax.key_contract import Case, Surface, observed
from tests.pax.key_surfaces import KEY_SURFACES, Site

PACKAGE = Path(pydantic_ai_pax.__file__).parent

WIDGETS = {
    'termflow': {'MenuBuilder', 'TextInputBuilder', 'PagerBuilder', 'Menu', 'TextInput', 'Pager', 'LiveApp'},
    'prompt_toolkit': {'Application', 'PromptSession'},
    'pydantic_ai_pax': {'QuestionMenu', 'LineInput', 'SessionBrowser', 'TaskView', 'LivePrompt'},
}
"""The classes whose instances read keys, by the top-level package that defines them."""


def _widget(name: str, imported: dict[str, tuple[str, str]], module: str) -> str | None:
    source, original = imported.get(name, (module, name))
    return original if original in WIDGETS.get(source.split('.')[0], set()) else None


def construction_sites() -> Iterator[Site]:
    """Each call that constructs one of `WIDGETS`, by module, enclosing qualified name, and class."""
    for path in sorted(PACKAGE.rglob('*.py')):
        module = '.'.join(path.relative_to(PACKAGE).with_suffix('').parts)
        tree = ast.parse(path.read_text(encoding='utf-8'))
        imported = {
            alias.asname or alias.name: (node.module, alias.name)
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
            for alias in node.names
        }
        own = f'pydantic_ai_pax.{module}'
        yield from _sites(tree, module, own, imported, ())


def _sites(
    node: ast.AST, module: str, own: str, imported: dict[str, tuple[str, str]], scope: tuple[str, ...]
) -> Iterator[Site]:
    for child in ast.iter_child_nodes(node):
        inner = scope
        if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            inner = (*scope, child.name)
        elif isinstance(child, ast.Call):
            # `PromptSession[str]()` subscripts the class before calling it.
            func = child.func.value if isinstance(child.func, ast.Subscript) else child.func
            if isinstance(func, ast.Name) and (widget := _widget(func.id, imported, own)) is not None:
                yield module, '.'.join(scope) or '<module>', widget
        yield from _sites(child, module, own, imported, inner)


def test_every_key_reading_surface_is_registered() -> None:
    sites = set(construction_sites())
    assert sites, 'the scan found no widget constructions: check WIDGETS against the imports'
    assert_classified(sites, KEY_SURFACES, complete_walk=True, table_name='KEY_SURFACES in tests/pax/key_surfaces.py')


CASES = [
    pytest.param(case, id=f'{".".join(site[:2])}:{case.name}')
    for site, surface in KEY_SURFACES.items()
    if isinstance(surface, Surface)
    for case in surface.cases
]


@pytest.mark.parametrize('case', CASES)
async def test_each_key_does_what_its_cell_declares(case: Case, tmp_path: Path) -> None:
    """Every key in `KEYS` on a fresh surface: a mismatch prints the whole row as the surface behaves now."""
    assert await observed(case, tmp_path) == case.expected()
