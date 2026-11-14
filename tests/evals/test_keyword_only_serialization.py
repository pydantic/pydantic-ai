from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pytest
import yaml
from pydantic import JsonValue

from ..conftest import try_import

with try_import() as imports_successful:
    from jsonschema.validators import validator_for

    from pydantic_evals import Case, Dataset
    from pydantic_evals.evaluators import (
        Equals,
        EqualsExpected,
        Evaluator,
        EvaluatorContext,
        HasMatchingSpan,
        ReportEvaluator,
        ReportEvaluatorContext,
    )
    from pydantic_evals.reporting.analyses import ScalarResult

    @dataclass(kw_only=True)
    class KeywordOnly(Evaluator[str, str, str]):
        minimum: int

        def evaluate(self, ctx: EvaluatorContext[str, str, str]) -> bool:
            return len(ctx.output) >= self.minimum

    @dataclass
    class KeywordOnlyField(Evaluator[str, str, str]):
        minimum: int = field(default=0, kw_only=True)

        def evaluate(self, ctx: EvaluatorContext[str, str, str]) -> bool:
            return len(ctx.output) >= self.minimum

    @dataclass
    class MixedFields(Evaluator[str, str, str]):
        minimum: int = field(kw_only=True)
        evaluation_name: str = 'length'

        def evaluate(self, ctx: EvaluatorContext[str, str, str]) -> bool:
            return len(ctx.output) >= self.minimum

    @dataclass
    class KeywordOnlyFirstWithDefault(Evaluator[str, str, str]):
        """A positional value would land in `evaluation_name`, the first positional parameter."""

        minimum: int = field(default=0, kw_only=True)
        evaluation_name: str = 'length'

        def evaluate(self, ctx: EvaluatorContext[str, str, str]) -> bool:
            return len(ctx.output) >= self.minimum

    @dataclass
    class Positional(Evaluator[str, str, str]):
        minimum: int

        def evaluate(self, ctx: EvaluatorContext[str, str, str]) -> bool:
            return len(ctx.output) >= self.minimum

    @dataclass
    class PositionalWithKeywordOnlyDefault(Evaluator[str, str, str]):
        minimum: int
        evaluation_name: str = field(default='length', kw_only=True)

        def evaluate(self, ctx: EvaluatorContext[str, str, str]) -> bool:
            return len(ctx.output) >= self.minimum

    @dataclass
    class ObjectPositional(Evaluator[str, str, str]):
        """A string-keyed dict value saves in the keyword form, since the loader reads one as keyword arguments."""

        minimum: dict[str, int]

        def evaluate(self, ctx: EvaluatorContext[str, str, str]) -> bool:
            return len(ctx.output) >= self.minimum['length']

    @dataclass
    class IntKeyed(Evaluator[str, str, str]):
        minimum: dict[int, int]

        def evaluate(self, ctx: EvaluatorContext[str, str, str]) -> bool:
            return len(ctx.output) >= self.minimum[1]

    @dataclass(kw_only=True)
    class KeywordOnlyReport(ReportEvaluator[str, str, str]):
        value: float

        def evaluate(self, ctx: ReportEvaluatorContext[str, str, str]) -> ScalarResult:
            return ScalarResult(title='Configured value', value=self.value)

    @dataclass(frozen=True)
    class RoundTripCase:
        """What `evaluator` saves as, and the spec forms its saved schema offers."""

        evaluator: Evaluator[str, str, str]
        saved_arguments: object
        schema_forms: frozenset[str]

    KEYWORD_FORMS = frozenset({'spec', 'spec_params'})
    ROUND_TRIP_CASES: dict[str, RoundTripCase] = {
        'KeywordOnly': RoundTripCase(KeywordOnly(minimum=4), {'minimum': 4}, KEYWORD_FORMS),
        'KeywordOnlyField': RoundTripCase(KeywordOnlyField(minimum=4), {'minimum': 4}, KEYWORD_FORMS),
        'MixedFields': RoundTripCase(MixedFields(minimum=4), {'minimum': 4}, KEYWORD_FORMS),
        'KeywordOnlyFirstWithDefault': RoundTripCase(
            KeywordOnlyFirstWithDefault(minimum=4), {'minimum': 4}, KEYWORD_FORMS
        ),
        'Positional': RoundTripCase(Positional(minimum=4), 4, frozenset({'short_spec'})),
        'PositionalWithKeywordOnlyDefault': RoundTripCase(
            PositionalWithKeywordOnlyDefault(minimum=4), 4, frozenset({'short_spec'}) | KEYWORD_FORMS
        ),
        'ObjectPositional': RoundTripCase(
            ObjectPositional(minimum={'length': 4}),
            {'minimum': {'length': 4}},
            frozenset({'short_spec'}) | KEYWORD_FORMS,
        ),
    }

    # Each entry with the evaluator it loads as, or `None` when the built-in schema rejects it.
    BUILTIN_SCHEMA_CASES: dict[str, tuple[dict[str, JsonValue], Evaluator[str, str, str] | None]] = {
        'scalar-short-form': ({'Equals': 5}, Equals(value=5)),
        'object-keyword-form': ({'Equals': {'value': {'a': 1}}}, Equals(value={'a': 1})),
        'query-keyword-form': (
            {'HasMatchingSpan': {'query': {'name_equals': 'x'}}},
            HasMatchingSpan(query={'name_equals': 'x'}),
        ),
        'defaulted-short-form': ({'EqualsExpected': 'check'}, EqualsExpected(evaluation_name='check')),
        'unknown-keyword': ({'Equals': {'a': 1}}, None),
        'missing-value': ({'Equals': {}}, None),
        'contains-unknown-keyword': ({'Contains': {'a': 1}}, None),
        'query-fields-as-keywords': ({'HasMatchingSpan': {'name_equals': 'x'}}, None),
        'mistyped-keyword': ({'Equals': {'value': 5, 'evaluation_name': 3}}, None),
        'mistyped-flag': ({'Contains': {'value': 'x', 'case_sensitive': 'no'}}, None),
        'empty-entry': ({}, None),
    }


pytestmark = pytest.mark.skipif(not imports_successful(), reason='pydantic-evals or jsonschema not installed')


@pytest.mark.parametrize('fmt', ['json', 'yaml'])
@pytest.mark.parametrize(
    'evaluator_name',
    [
        'KeywordOnly',
        'KeywordOnlyField',
        'MixedFields',
        'KeywordOnlyFirstWithDefault',
        'Positional',
        'PositionalWithKeywordOnlyDefault',
        'ObjectPositional',
    ],
)
def test_evaluator_file_round_trip(tmp_path: Path, fmt: Literal['json', 'yaml'], evaluator_name: str):
    """File persistence retains constructor names without changing existing positional short forms."""
    case = ROUND_TRIP_CASES[evaluator_name]
    evaluator = case.evaluator
    evaluator_type = type(evaluator)
    dataset = Dataset[str, str, str](
        name='round-trip', cases=[Case(inputs='hello', evaluators=[evaluator])], evaluators=[evaluator]
    )
    path = tmp_path / f'dataset.{fmt}'
    dataset.to_file(path, custom_evaluator_types=[evaluator_type])

    loaded = Dataset[str, str, str].from_file(path, custom_evaluator_types=[evaluator_type])
    assert loaded.evaluators == [evaluator]
    assert loaded.cases[0].evaluators == [evaluator]

    raw = yaml.safe_load(path.read_text())
    name = evaluator_type.get_serialization_name()
    expected = [{name: case.saved_arguments}]
    assert raw['evaluators'] == expected
    assert raw['cases'][0]['evaluators'] == expected

    schema = json.loads((tmp_path / 'dataset_schema.json').read_text())
    validator = validator_for(schema)(schema)
    assert list(validator.iter_errors(raw)) == []
    forms = {key.removesuffix(f'_{name}') for key in schema['$defs'] if key.endswith(f'_{name}')}
    assert forms == case.schema_forms
    report = loaded.evaluate_sync(lambda value: value)
    assert all(result.value is True for result in report.cases[0].assertions.values())


def test_non_string_keyed_short_form():
    """A YAML mapping with a non-string key is a positional short-form value, in the schema as in the loader."""
    raw = yaml.safe_load('name: int-keyed\ncases: [{inputs: hello}]\nevaluators:\n- IntKeyed: {1: 4}\n')
    schema = Dataset[str, str, str].model_json_schema_with_evaluators([IntKeyed])
    assert list(validator_for(schema)(schema).iter_errors(raw)) == []
    loaded = Dataset[str, str, str].from_dict(raw, custom_evaluator_types=[IntKeyed])
    assert loaded.evaluators == [IntKeyed(minimum={1: 4})]
    report = loaded.evaluate_sync(lambda value: value)
    assert all(result.value is True for result in report.cases[0].assertions.values())


@pytest.mark.parametrize(
    'case_id',
    [
        'scalar-short-form',
        'object-keyword-form',
        'query-keyword-form',
        'defaulted-short-form',
        'unknown-keyword',
        'missing-value',
        'contains-unknown-keyword',
        'query-fields-as-keywords',
        'mistyped-keyword',
        'mistyped-flag',
        'empty-entry',
    ],
)
def test_builtin_evaluator_schema(case_id: str):
    """The built-in schema rejects short forms the loader reads as keyword arguments, and empty entries."""
    evaluator, loaded = BUILTIN_SCHEMA_CASES[case_id]
    raw: dict[str, JsonValue] = {'name': 'built-in', 'cases': [{'inputs': 'hello'}], 'evaluators': [evaluator]}
    schema = Dataset[str, str, str].model_json_schema_with_evaluators()
    assert (list(validator_for(schema)(schema).iter_errors(raw)) == []) is (loaded is not None)
    if loaded is not None:
        assert Dataset[str, str, str].from_dict(raw).evaluators == [loaded]


@pytest.mark.parametrize('fmt', ['json', 'yaml'])
def test_keyword_only_report_evaluator_file_round_trip(tmp_path: Path, fmt: Literal['json', 'yaml']):
    """Report evaluators share the keyword-only persistence contract with case evaluators."""
    evaluator = KeywordOnlyReport(value=4.0)
    dataset = Dataset[str, str, str](
        name='report-round-trip', cases=[Case(inputs='hello')], report_evaluators=[evaluator]
    )
    path = tmp_path / f'dataset.{fmt}'
    dataset.to_file(path, custom_report_evaluator_types=[KeywordOnlyReport])

    loaded = Dataset[str, str, str].from_file(path, custom_report_evaluator_types=[KeywordOnlyReport])
    assert loaded.report_evaluators == [evaluator]
    assert yaml.safe_load(path.read_text())['report_evaluators'] == [{'KeywordOnlyReport': {'value': 4.0}}]
    schema = json.loads((tmp_path / 'dataset_schema.json').read_text())
    validator = validator_for(schema)(schema)
    assert list(validator.iter_errors(yaml.safe_load(path.read_text()))) == []
    assert schema['$defs']['spec_KeywordOnlyReport']['properties']['KeywordOnlyReport'] == {
        '$ref': '#/$defs/spec_params_KeywordOnlyReport'
    }
    assert 'short_spec_KeywordOnlyReport' not in schema['$defs']
    report = loaded.evaluate_sync(lambda value: value)
    assert report.analyses == [ScalarResult(title='Configured value', value=4.0)]
