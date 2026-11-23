"""Every key-reading surface pax constructs is registered, and each key does there what its cell declares."""

from __future__ import annotations

import ast
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

import pydantic_ai_pax
from pydantic_ai_pax.ui.menus.dialogs import build_pager
from tests.manifests import assert_classified
from tests.pax.key_contract import PICK, Case, Surface, observed, widget_case
from tests.pax.key_surfaces import KEY_SURFACES, Site

PACKAGE = Path(pydantic_ai_pax.__file__).parent

WIDGETS = {
    'termflow': {'MenuBuilder', 'TextInputBuilder', 'PagerBuilder', 'Menu', 'TextInput', 'Pager', 'LiveApp'},
    'prompt_toolkit': {'Application', 'PromptSession'},
    'pydantic_ai_pax': {'QuestionMenu', 'LineInput', 'SessionBrowser', 'TaskView', 'LivePrompt'},
}
"""The classes whose instances read keys, by the top-level package that defines them."""


def _origins(tree: ast.Module, own: str) -> dict[str, str]:
    """Each name a module imports, by the dotted path it comes from; a relative import resolves against `own`."""
    origins: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            package = own.rsplit('.', node.level)[0] if node.level else ''
            source = '.'.join(part for part in (package, node.module or '') if part)
            origins.update({alias.asname or alias.name: f'{source}.{alias.name}' for alias in node.names})
        elif isinstance(node, ast.Import):
            origins.update(
                {
                    alias.asname or alias.name.split('.')[0]: alias.name if alias.asname else alias.name.split('.')[0]
                    for alias in node.names
                }
            )
    return origins


def _callee(func: ast.expr) -> str | None:
    """The dotted name a call goes through: `Menu`, `menus.Menu`, or `PromptSession` for `PromptSession[str]`."""
    if isinstance(func, ast.Subscript):
        return _callee(func.value)
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute) and (base := _callee(func.value)) is not None:
        return f'{base}.{func.attr}'
    return None


def _widget(callee: str, origins: dict[str, str], own: str) -> str | None:
    head, _, rest = callee.partition('.')
    origin = '.'.join(part for part in (origins.get(head, f'{own}.{head}'), rest) if part)
    name = origin.rpartition('.')[2]
    return name if name in WIDGETS.get(origin.partition('.')[0], set()) else None


def construction_sites() -> Iterator[Site]:
    """Each call that constructs one of `WIDGETS`, by module, enclosing qualified name, and class."""
    for path in sorted(PACKAGE.rglob('*.py')):
        module = '.'.join(path.relative_to(PACKAGE).with_suffix('').parts)
        tree = ast.parse(path.read_text(encoding='utf-8'))
        own = f'pydantic_ai_pax.{module}'
        yield from _sites(tree, module, own, _origins(tree, own), ())


def _sites(node: ast.AST, module: str, own: str, origins: dict[str, str], scope: tuple[str, ...]) -> Iterator[Site]:
    for child in ast.iter_child_nodes(node):
        inner = scope
        if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            inner = (*scope, child.name)
        elif (
            isinstance(child, ast.Call)
            and (callee := _callee(child.func)) is not None
            and (widget := _widget(callee, origins, own)) is not None
        ):
            yield module, '.'.join(scope) or '<module>', widget
        yield from _sites(child, module, own, origins, inner)


@pytest.mark.parametrize(
    ('source', 'widget'),
    [
        ('from termflow.tui.menu import Menu as Picker\n\ndef f():\n    Picker([])\n', 'Menu'),
        ('import termflow.tui.menu as menus\n\ndef f():\n    menus.Menu([])\n', 'Menu'),
        ('import prompt_toolkit\n\ndef f():\n    prompt_toolkit.PromptSession[str]()\n', 'PromptSession'),
        (
            'from pydantic_ai_pax.ui.menus.question_menu import QuestionMenu\n\ndef f():\n    QuestionMenu()\n',
            'QuestionMenu',
        ),
        ('from .question_menu import QuestionMenu\n\ndef f():\n    QuestionMenu()\n', 'QuestionMenu'),
        ('from . import question_menu\n\ndef f():\n    question_menu.QuestionMenu()\n', 'QuestionMenu'),
    ],
)
def test_the_scan_sees_renamed_and_qualified_constructions(source: str, widget: str) -> None:
    tree = ast.parse(source)
    own = 'pydantic_ai_pax.ui.menus.demo'
    assert set(_sites(tree, 'demo', own, _origins(tree, own), ())) == {('demo', 'f', widget)}


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


def _shows_nothing(path: Path) -> None:
    return None


async def _no_widget(path: Path) -> None:
    return None


def _pager_first(path: Path) -> None:
    """A site whose case expects a menu but whose first widget is a pager the case did not script."""
    build_pager('Notes', 'text').run()


@pytest.mark.parametrize(
    ('show', 'message'),
    [
        pytest.param(_shows_nothing, 'returned without running a widget', id='sync site shows nothing'),
        pytest.param(_no_widget, 'never ran a Menu', id='async site shows nothing'),
        pytest.param(_pager_first, 'ran a Pager before its Menu: script it', id='earlier widget unscripted'),
    ],
)
async def test_a_case_that_never_reaches_its_widget_fails(
    show: Callable[[Path], object], message: str, tmp_path: Path
) -> None:
    """A mis-built case fails loudly instead of reporting an outcome for a widget it never pressed a key on."""
    with pytest.raises(AssertionError, match=message):
        await widget_case('broken', show, PICK).press(tmp_path, 'enter')
