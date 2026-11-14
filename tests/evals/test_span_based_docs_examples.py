"""The docs' every-span `HasMatchingSpan` recipes fail on traces that violate their claim.

`tests/test_examples.py` runs these docs blocks, but they only build evaluators, so it never checks what the queries
match. These tests read each query from its docs page and evaluate it with `SpanTree.any`, which is all
`HasMatchingSpan.evaluate` does. Each violating trace also holds a compliant span that an any-span query would match.
Durations come from explicit `SpanNode` timestamps.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import TypeAdapter
from pytest_examples import find_examples

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_evals.otel import SpanNode, SpanQuery, SpanTree

pytestmark = pytest.mark.skipif(not imports_successful(), reason='pydantic-evals not installed')

DOCS = Path(__file__).parents[2] / 'docs' / 'evals' / 'evaluators'
START = datetime(2026, 1, 1, tzinfo=UTC)


def docs_query(page: str, evaluation_name: str) -> SpanQuery:
    """The `query` of the one `HasMatchingSpan(...)` call on a docs page with this `evaluation_name`."""
    queries: list[SpanQuery] = []
    for example in find_examples(DOCS / page):
        for node in ast.walk(ast.parse(example.source)):
            if not (
                isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'HasMatchingSpan'
            ):
                continue
            keywords = {keyword.arg: keyword.value for keyword in node.keywords}
            name = keywords.get('evaluation_name')
            if isinstance(name, ast.Constant) and name.value == evaluation_name:
                queries.append(TypeAdapter(SpanQuery).validate_python(ast.literal_eval(keywords['query'])))
    assert len(queries) == 1, f'expected one {evaluation_name!r} recipe in {page}, found {len(queries)}'
    return queries[0]


@dataclass
class Span:
    name: str
    children: tuple[Span, ...]
    seconds: float


def span(name: str, *children: Span, seconds: float = 0.01) -> Span:
    return Span(name, children, seconds)


def root(*children: Span) -> Span:
    return span('task_execution', *children, seconds=1)


def span_tree(trace: Span) -> SpanTree:
    nodes: list[SpanNode] = []

    def add(s: Span, parent_span_id: int | None) -> None:
        span_id = len(nodes) + 1
        nodes.append(
            SpanNode(
                name=s.name,
                trace_id=1,
                span_id=span_id,
                parent_span_id=parent_span_id,
                start_timestamp=START,
                end_timestamp=START + timedelta(seconds=s.seconds),
                attributes={},
            )
        )
        for child in s.children:
            add(child, span_id)

    add(trace, None)
    tree = SpanTree()
    tree.add_spans(nodes)
    return tree


@dataclass
class Recipe:
    page: str
    evaluation_name: str
    compliant: list[Span]
    violating: list[Span]


def tools(calls: int) -> Span:
    return span('running tools', *(span('search') for _ in range(calls)))


RECIPES = [
    Recipe(
        page='span-based.md',
        evaluation_name='safe_execution',
        compliant=[root(span('search_tool'))],
        violating=[
            root(span('search_tool'), span('delete_database')),
            root(span('step', span('delete_database'))),
        ],
    ),
    Recipe(
        page='span-based.md',
        evaluation_name='fast_db_queries',
        compliant=[
            root(span('database_query', seconds=0.05), span('database_query', seconds=0.1)),
            root(span('search_tool')),
        ],
        violating=[root(span('database_query', seconds=0.05), span('database_query', seconds=0.3))],
    ),
    Recipe(
        page='span-based.md',
        evaluation_name='no_circular_delegation',
        compliant=[
            root(span('master_agent', span('specialist_agent'))),
            root(span('master_agent', span('specialist_agent', span('sub_agent')))),
        ],
        violating=[
            root(span('master_agent', span('specialist_agent', span('master_agent')))),
            root(span('master_agent'), span('master_agent', span('specialist_agent', span('master_agent')))),
        ],
    ),
    Recipe(
        page='span-based.md',
        evaluation_name='reasonable_tool_usage',
        compliant=[root(tools(5), tools(5))],
        violating=[root(tools(1), tools(6))],
    ),
    Recipe(
        page='built-in.md',
        evaluation_name='llm_fast_enough',
        compliant=[root(span('llm_call', seconds=1), span('llm_call', seconds=2))],
        violating=[root(span('llm_call', seconds=1), span('llm_call', seconds=2.5))],
    ),
]


@pytest.mark.parametrize('recipe', [pytest.param(r, id=r.evaluation_name) for r in RECIPES])
def test_docs_recipe_checks_every_span(recipe: Recipe):
    query = docs_query(recipe.page, recipe.evaluation_name)
    assert [span_tree(trace).any(query) for trace in recipe.compliant] == [True] * len(recipe.compliant)
    assert [span_tree(trace).any(query) for trace in recipe.violating] == [False] * len(recipe.violating)
