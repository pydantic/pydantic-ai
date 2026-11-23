"""Source rules for the hand-written model-name tables in `providers/` and `profiles/` (R23).

Both rules read the source with `ast`, so they cover every module, including those whose optional dependencies aren't
installed.

- Only `profiles/_family.py` splits a `vendor/model` id. A provider that splits the id itself keeps its own vendor
  table, which drifts from the shared family map: #12383 had to route Kimi on Vercel, whose own table lacked the
  `moonshotai` vendor the shared map already had.
- Two string sets in one module may not hold the same names, or one the other's, unless a `# distinct: <reason>` comment
  says why. #13495 had to add GPT-6 and GPT-6.1 to Bedrock's 30-minute cache set, a copy of its list of GPT models that
  had fallen behind.
"""

from __future__ import annotations

import ast
import itertools
import re
from pathlib import Path

import pytest

PACKAGE = Path(__file__).parents[2] / 'pydantic_ai_slim' / 'pydantic_ai'

_SPLITTERS = frozenset({'split', 'partition'})
_HAS_DIGIT = re.compile(r'\d')


def _model_name_splits(path: Path) -> list[str]:
    """The `.split('/')` and `.partition('/')` calls on a model name in `path`, as `path:line source` strings."""
    source = path.read_text(encoding='utf-8')
    hits: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _SPLITTERS
            and node.args
            and isinstance(separator := node.args[0], ast.Constant)
            and separator.value == '/'
            and 'model' in ast.unparse(node.func.value).lower()
        ):
            hits.append(f'{path.relative_to(PACKAGE)}:{node.lineno} {ast.unparse(node)}')
    return hits


def test_only_the_family_map_splits_vendor_model_ids():
    """Providers read a `vendor/model` id through `profiles._family.split_vendor_model_id`, never by splitting it."""
    hits = [hit for path in sorted((PACKAGE / 'providers').glob('*.py')) for hit in _model_name_splits(path)]
    assert not hits, (
        'Split a `vendor/model` id with `split_vendor_model_id()`, and resolve its family with `vendor_family_profile()`'
        ' or `vendor_family_model_profile()` from `pydantic_ai.profiles._family`:\n' + '\n'.join(hits)
    )


def _model_name_sets(tree: ast.Module) -> list[tuple[int, frozenset[str]]]:
    """Each set literal of three or more model names in `tree`, bare or passed to `set()` or `frozenset()`, with its line.

    A model name is a string with a digit in it, which leaves out sets of effort levels, field names and media types.
    """
    found: list[tuple[int, frozenset[str]]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Set):
            elements = node.elts
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in ('set', 'frozenset')
            and len(node.args) == 1
            and isinstance(node.args[0], ast.List | ast.Tuple)
        ):
            elements = node.args[0].elts
        else:
            continue
        names = [element.value for element in elements if isinstance(element, ast.Constant)]
        if len(names) == len(elements) >= 3 and all(
            isinstance(name, str) and _HAS_DIGIT.search(name) for name in names
        ):
            found.append((node.lineno, frozenset(str(name) for name in names)))
    return found


def _overlapping_sets(path: Path) -> list[str]:
    source = path.read_text(encoding='utf-8')
    lines = source.splitlines()

    def distinct(line: int) -> bool:
        # The comment may sit on the literal's first line or the line above it.
        return any('# distinct:' in lines[index] for index in (line - 1, line - 2) if index >= 0)

    sets = [(line, names) for line, names in _model_name_sets(ast.parse(source)) if not distinct(line)]
    problems: list[str] = []
    for (line_a, a), (line_b, b) in itertools.combinations(sets, 2):
        if a == b:
            problems.append(f'{path.relative_to(PACKAGE)}:{line_a} and :{line_b} hold the same {len(a)} names')
        elif a < b or b < a:
            problems.append(f'{path.relative_to(PACKAGE)}:{line_a} and :{line_b}: one holds the other')
    return problems


@pytest.mark.parametrize('package', ['providers', 'profiles'])
def test_no_copied_model_name_sets(package: str):
    """Derive a second set of the same names from one constant, or say with `# distinct: <reason>` why it differs."""
    problems = [problem for path in sorted((PACKAGE / package).glob('*.py')) for problem in _overlapping_sets(path)]
    assert not problems, 'Derive both sets from one constant, or mark one `# distinct: <reason>`:\n' + '\n'.join(
        problems
    )
