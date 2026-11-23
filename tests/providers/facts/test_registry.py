"""Every `ProviderFact` names a fact and a recorded probe that exist (proposal R32).

Not a VCR test: it reads the source, not a provider. The fact's symbol and the probe's test function are found by
parsing their modules, so a module whose optional dependencies are missing is still checked, and a probe counts as
recorded when its cassette is where cassetter looks for it.
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from pathlib import Path

import pytest

from .registry import FACTS, ProviderFact

_REPO = Path(__file__).parents[3]

_UNSAFE_IN_FILE_NAMES = re.compile(r'[<>?%*:|"\'/\\]')
"""The characters cassetter replaces with `-` when it names a cassette after a test."""


def _defines(path: str, symbol: str) -> bool:
    """Whether the module at `path` defines `symbol`: a function, class or assignment, dotted for a class member."""
    body: list[ast.stmt] = ast.parse((_REPO / path).read_text(encoding='utf-8')).body
    for name in symbol.split('.'):
        found: ast.stmt | None = None
        for node in body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                defined = [node.name]
            elif isinstance(node, ast.Assign):
                defined = [t.id for t in node.targets if isinstance(t, ast.Name)]
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                defined = [node.target.id]
            else:
                continue
            if name in defined:
                found = node
        if found is None:
            return False
        body = found.body if isinstance(found, ast.ClassDef) else []
    return True


def test_fact_ids_are_unique():
    assert [fact_id for fact_id, count in Counter(f.id for f in FACTS).items() if count > 1] == []


@pytest.mark.parametrize('fact', FACTS, ids=[f.id for f in FACTS])
def test_the_fact_is_written_where_its_row_says(fact: ProviderFact):
    path, symbol = fact.where.split('::')
    assert path.startswith('pydantic_ai_slim/pydantic_ai/'), path
    assert _defines(path, symbol), fact.where


@pytest.mark.parametrize('fact', FACTS, ids=[f.id for f in FACTS])
def test_the_probe_is_a_recorded_test(fact: ProviderFact):
    path, node_name = fact.probe.split('::')
    function, _, _ = node_name.partition('[')
    assert path.startswith('tests/') and function.startswith('test_'), fact.probe
    assert _defines(path, function), fact.probe

    cassettes = (_REPO / path).parent / 'cassettes' / Path(path).stem
    if function == node_name:
        recorded = any(p.stem.partition('[')[0] == function for p in cassettes.glob('*.yaml'))
    else:
        recorded = (cassettes / f'{_UNSAFE_IN_FILE_NAMES.sub("-", node_name)}.yaml').exists()
    assert recorded, f'no cassette for {fact.probe} in {cassettes.relative_to(_REPO)}'
