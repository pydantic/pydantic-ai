from __future__ import annotations

from time import sleep

import pytest

from ..conftest import try_import

with try_import() as imports_successful:
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from pydantic_evals import Case, Dataset
    from pydantic_evals.evaluators import HasMatchingSpan

with try_import() as logfire_import_successful:
    import logfire

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='pydantic-evals or opentelemetry-sdk not installed'),
    pytest.mark.skipif(not logfire_import_successful(), reason='logfire not installed'),
]


@pytest.fixture
def real_clock_logfire() -> None:
    """Configure logfire with real timestamps and an in-memory span exporter.

    The duration thresholds in these recipe examples are only meaningful with a
    real clock: `logfire.testing`'s `TimeGenerator` records span timestamps at
    least one second apart, which would make every span exceed them.
    """
    logfire.configure(
        send_to_logfire=False,
        console=False,
        additional_span_processors=[SimpleSpanProcessor(InMemorySpanExporter())],
    )


def test_span_based_recipes_reject_violating_traces(real_clock_logfire: None) -> None:
    evaluators = [
        HasMatchingSpan(
            query={
                'name_equals': 'task_execution',
                'not_': {'name_contains': 'delete_database'},
                'no_descendant_has': {'name_contains': 'delete_database'},
            },
            evaluation_name='safe_execution',
        ),
        HasMatchingSpan(
            query={
                'name_equals': 'task_execution',
                'not_': {
                    'and_': [
                        {'name_contains': 'database'},
                        {'not_': {'max_duration': 0.1}},
                    ]
                },
                'no_descendant_has': {
                    'and_': [
                        {'name_contains': 'database'},
                        {'not_': {'max_duration': 0.1}},
                    ]
                },
            },
            evaluation_name='fast_db_queries',
        ),
        HasMatchingSpan(
            query={
                'name_equals': 'task_execution',
                'not_': {
                    'and_': [
                        {'name_contains': 'agent'},
                        {'some_descendant_has': {'name_contains': 'agent'}},
                        {'some_ancestor_has': {'name_contains': 'agent'}},
                    ]
                },
                'no_descendant_has': {
                    'and_': [
                        {'name_contains': 'agent'},
                        {'some_descendant_has': {'name_contains': 'agent'}},
                        {'some_ancestor_has': {'name_contains': 'agent'}},
                    ]
                },
            },
            evaluation_name='no_circular_delegation',
        ),
        HasMatchingSpan(
            query={
                'name_equals': 'task_execution',
                'not_': {
                    'and_': [
                        {'name_contains': 'tool'},
                        {'min_child_count': 6},
                    ]
                },
                'no_descendant_has': {
                    'and_': [
                        {'name_contains': 'tool'},
                        {'min_child_count': 6},
                    ]
                },
            },
            evaluation_name='reasonable_tool_usage',
        ),
    ]

    def task(scenario: str) -> str:
        with logfire.span('task_execution'):
            if scenario == 'compliant':
                with logfire.span('search_tool'):
                    pass
                with logfire.span('fast_database_query'):
                    pass
                with logfire.span('master_agent'):
                    with logfire.span('specialist_agent'):
                        pass
                with logfire.span('tool_execute'):
                    with logfire.span('child_operation'):
                        pass
            else:
                with logfire.span('delete_database'):
                    pass
                with logfire.span('slow_database_query'):
                    sleep(0.15)
                with logfire.span('master_agent'):
                    with logfire.span('specialist_agent'):
                        with logfire.span('master_agent'):
                            pass
                with logfire.span('tool_execute'):
                    for child_index in range(6):
                        with logfire.span(f'child_operation_{child_index}'):
                            pass
        return scenario

    dataset = Dataset(
        name='span_based_recipes',
        cases=[
            Case(name='compliant', inputs='compliant'),
            Case(name='violating', inputs='violating'),
        ],
        evaluators=evaluators,
    )
    report = dataset.evaluate_sync(task, progress=False)

    compliant, violating = report.cases
    for evaluation_name in (
        'safe_execution',
        'fast_db_queries',
        'no_circular_delegation',
        'reasonable_tool_usage',
    ):
        assert compliant.assertions[evaluation_name].value is True
        assert violating.assertions[evaluation_name].value is False


def test_builtin_llm_fast_enough_rejects_slow_trace(real_clock_logfire: None) -> None:
    evaluator = HasMatchingSpan(
        query={
            'name_equals': 'task_execution',
            'not_': {
                'and_': [
                    {'name_equals': 'llm_call'},
                    {'not_': {'max_duration': 2.0}},
                ]
            },
            'no_descendant_has': {
                'and_': [
                    {'name_equals': 'llm_call'},
                    {'not_': {'max_duration': 2.0}},
                ]
            },
        },
        evaluation_name='llm_fast_enough',
    )

    def task(is_slow: bool) -> str:
        with logfire.span('task_execution'):
            with logfire.span('llm_call'):
                if is_slow:
                    sleep(2.1)
        return 'done'

    dataset = Dataset(
        name='builtin_llm_fast_enough',
        cases=[
            Case(name='fast_llm', inputs=False),
            Case(name='slow_llm', inputs=True),
        ],
        evaluators=[evaluator],
    )
    report = dataset.evaluate_sync(task, progress=False)

    assert report.cases[0].assertions['llm_fast_enough'].value is True
    assert report.cases[1].assertions['llm_fast_enough'].value is False
