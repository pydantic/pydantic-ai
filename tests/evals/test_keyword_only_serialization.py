from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pytest
import yaml

from ..conftest import try_import

with try_import() as imports_successful:
    from jsonschema.validators import validator_for

    from pydantic_evals import Case, Dataset
    from pydantic_evals.evaluators import Evaluator, EvaluatorContext, ReportEvaluator, ReportEvaluatorContext
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

    @dataclass(kw_only=True)
    class KeywordOnlyReport(ReportEvaluator[str, str, str]):
        value: float

        def evaluate(self, ctx: ReportEvaluatorContext[str, str, str]) -> ScalarResult:
            return ScalarResult(title='Configured value', value=self.value)


pytestmark = pytest.mark.skipif(not imports_successful(), reason='pydantic-evals or jsonschema not installed')


@pytest.mark.parametrize('fmt', ['json', 'yaml'])
@pytest.mark.parametrize(
    'evaluator_name',
    ['KeywordOnly', 'KeywordOnlyField', 'MixedFields', 'Positional', 'PositionalWithKeywordOnlyDefault'],
)
def test_evaluator_file_round_trip(tmp_path: Path, fmt: Literal['json', 'yaml'], evaluator_name: str):
    """File persistence retains constructor names without changing existing positional short forms."""
    evaluator_type = {
        'KeywordOnly': KeywordOnly,
        'KeywordOnlyField': KeywordOnlyField,
        'MixedFields': MixedFields,
        'Positional': Positional,
        'PositionalWithKeywordOnlyDefault': PositionalWithKeywordOnlyDefault,
    }[evaluator_name]
    evaluator = evaluator_type(minimum=4)
    dataset = Dataset[str, str, str](
        name='round-trip', cases=[Case(inputs='hello', evaluators=[evaluator])], evaluators=[evaluator]
    )
    path = tmp_path / f'dataset.{fmt}'
    dataset.to_file(path, custom_evaluator_types=[evaluator_type])

    loaded = Dataset[str, str, str].from_file(path, custom_evaluator_types=[evaluator_type])
    assert loaded.evaluators == [evaluator]
    assert loaded.cases[0].evaluators == [evaluator]

    raw = yaml.safe_load(path.read_text())
    positional = evaluator_type in (Positional, PositionalWithKeywordOnlyDefault)
    arguments = 4 if positional else {'minimum': 4}
    expected = [{evaluator_type.get_serialization_name(): arguments}]
    assert raw['evaluators'] == expected
    assert raw['cases'][0]['evaluators'] == expected

    schema = json.loads((tmp_path / 'dataset_schema.json').read_text())
    validator = validator_for(schema)(schema)
    assert list(validator.iter_errors(raw)) == []
    name = evaluator_type.get_serialization_name()
    if positional:
        assert schema['$defs'][f'short_spec_{name}']['properties'][name]['type'] == 'integer'
    else:
        assert f'short_spec_{name}' not in schema['$defs']
        assert schema['$defs'][f'spec_{name}']['properties'][name] == {'$ref': f'#/$defs/spec_params_{name}'}
        assert schema['$defs'][f'spec_params_{name}']['properties']['minimum'] == {
            'title': 'Minimum',
            'type': 'integer',
        }
    report = loaded.evaluate_sync(lambda value: value)
    assert all(result.value is True for result in report.cases[0].assertions.values())


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
