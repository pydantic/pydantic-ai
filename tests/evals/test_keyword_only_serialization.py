from __future__ import annotations

import json
import re
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import ClassVar, Literal

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

    @dataclass(kw_only=True, init=False)
    class KeywordOnlyVarKeyword(Evaluator[str, str, str]):
        minimum: int

        def __init__(self, *, minimum: int, **kwargs: object) -> None:
            self.minimum = minimum

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

    @dataclass
    class InitFalsePattern(Evaluator[str, str, str]):
        """`__post_init__` compiles `pattern` from the constructor arguments, which are all a file holds."""

        source: str
        ignore_case: bool = False
        pattern: re.Pattern[str] = field(init=False, repr=False)

        def __post_init__(self) -> None:
            self.pattern = re.compile(self.source, re.IGNORECASE if self.ignore_case else 0)

        def evaluate(self, ctx: EvaluatorContext[str, str, str]) -> bool:
            return self.pattern.fullmatch(ctx.output) is not None

    @dataclass
    class InitFalseDefault(Evaluator[str, str, str]):
        """`__post_init__` replaces the default of `label`, which `__init__` doesn't take."""

        minimum: int
        label: str = field(default='unset', init=False)

        def __post_init__(self) -> None:
            self.label = f'at least {self.minimum}'

        def evaluate(self, ctx: EvaluatorContext[str, str, str]) -> bool:
            return len(ctx.output) >= self.minimum

    @dataclass
    class InitFalseFirst(Evaluator[str, str, str]):
        """`calls` is declared before `minimum`, the parameter a short-form value goes to."""

        calls: int = field(default=0, init=False)
        minimum: int = 0

        def evaluate(self, ctx: EvaluatorContext[str, str, str]) -> bool:
            self.calls += 1
            return len(ctx.output) >= self.minimum

    @dataclass
    class WithClassVar(Evaluator[str, str, str]):
        """`unit` is a `ClassVar`, which `__init__` doesn't take."""

        unit: ClassVar[str] = 'characters'
        minimum: int

        def evaluate(self, ctx: EvaluatorContext[str, str, str]) -> bool:
            return len(ctx.output) >= self.minimum

    @dataclass
    class InitFalseTakenByInit(Evaluator[str, str, str]):
        """A hand-written `__init__` takes `label`, so a file keeps it."""

        minimum: int
        label: str = field(init=False)

        def __init__(self, minimum: int, label: str) -> None:
            self.minimum = minimum
            self.label = label

        def evaluate(self, ctx: EvaluatorContext[str, str, str]) -> bool:
            return len(ctx.output) >= self.minimum

    @dataclass
    class InitFalseTakenByKwargs(Evaluator[str, str, str]):
        """A hand-written `__init__` takes `bonus` through `**kwargs`, so a file keeps it, but not the `ClassVar`."""

        unit: ClassVar[str] = 'characters'
        bonus: int = field(default=0, init=False)
        minimum: int

        def __init__(self, **kwargs: int) -> None:
            self.minimum = kwargs['minimum']
            self.bonus = kwargs.get('bonus', 0)

        def evaluate(self, ctx: EvaluatorContext[str, str, str]) -> bool:
            return len(ctx.output) + self.bonus >= self.minimum

    @dataclass
    class InitFalseRenamedByInit(Evaluator[str, str, str]):
        """A hand-written `__init__` fills `_needle` from a parameter of another name, so a file keeps every field."""

        _needle: str = field(init=False)

        def __init__(self, needle: str) -> None:
            self._needle = needle

        def evaluate(self, ctx: EvaluatorContext[str, str, str]) -> bool:
            return self._needle in ctx.output

    @dataclass
    class InitFalseTakenByArgs(Evaluator[str, str, str]):
        """A hand-written `__init__` fills `pattern` from `*args`, so a file keeps every field."""

        pattern: str = field(init=False)

        def __init__(self, *args: str) -> None:
            self.pattern = args[0]

        def evaluate(self, ctx: EvaluatorContext[str, str, str]) -> bool:
            return re.fullmatch(self.pattern, ctx.output) is not None

    @dataclass
    class InitFalseWithInitVar(Evaluator[str, str, str]):
        """The generated `__init__` takes `pattern` and the `InitVar` `flags`, but not `compiled`."""

        pattern: str
        flags: InitVar[int] = 0
        compiled: re.Pattern[str] = field(init=False, repr=False)

        def __post_init__(self, flags: int) -> None:
            self.compiled = re.compile(self.pattern, flags)

        def evaluate(self, ctx: EvaluatorContext[str, str, str]) -> bool:
            return self.compiled.fullmatch(ctx.output) is not None

    @dataclass
    class InitFalseOnly(Evaluator[str, str, str]):
        """Every field is `init=False`, and `build_serialization_arguments` saves the `text` that `__init__` takes."""

        _text: str = field(init=False)

        def __init__(self, text: str) -> None:
            self._text = text

        def build_serialization_arguments(self) -> dict[str, object]:
            return {'text': self._text}

        def evaluate(self, ctx: EvaluatorContext[str, str, str]) -> bool:
            return self._text in ctx.output

    @dataclass(kw_only=True)
    class KeywordOnlyReport(ReportEvaluator[str, str, str]):
        value: float

        def evaluate(self, ctx: ReportEvaluatorContext[str, str, str]) -> ScalarResult:
            return ScalarResult(title='Configured value', value=self.value)

    @dataclass
    class InitFalseReport(ReportEvaluator[str, str, str]):
        value: float
        title: str = field(init=False)

        def __post_init__(self) -> None:
            self.title = f'Configured {self.value}'

        def evaluate(self, ctx: ReportEvaluatorContext[str, str, str]) -> ScalarResult:
            return ScalarResult(title=self.title, value=self.value)

    @dataclass(frozen=True)
    class RoundTripCase:
        """What `evaluator` saves as, and the spec forms its saved schema offers."""

        evaluator: Evaluator[str, str, str]
        saved_arguments: object
        schema_forms: frozenset[str]

    KEYWORD_FORMS = frozenset({'spec', 'spec_params'})
    INIT_FALSE_PATTERN_FORMS = frozenset({'short_spec'}) | KEYWORD_FORMS
    ROUND_TRIP_CASES: dict[str, RoundTripCase] = {
        'KeywordOnly': RoundTripCase(KeywordOnly(minimum=4), {'minimum': 4}, KEYWORD_FORMS),
        'KeywordOnlyField': RoundTripCase(KeywordOnlyField(minimum=4), {'minimum': 4}, KEYWORD_FORMS),
        'MixedFields': RoundTripCase(MixedFields(minimum=4), {'minimum': 4}, KEYWORD_FORMS),
        'KeywordOnlyFirstWithDefault': RoundTripCase(
            KeywordOnlyFirstWithDefault(minimum=4), {'minimum': 4}, KEYWORD_FORMS
        ),
        'KeywordOnlyVarKeyword': RoundTripCase(KeywordOnlyVarKeyword(minimum=4), {'minimum': 4}, KEYWORD_FORMS),
        'Positional': RoundTripCase(Positional(minimum=4), 4, frozenset({'short_spec'})),
        'PositionalWithKeywordOnlyDefault': RoundTripCase(
            PositionalWithKeywordOnlyDefault(minimum=4), 4, frozenset({'short_spec'}) | KEYWORD_FORMS
        ),
        'ObjectPositional': RoundTripCase(
            ObjectPositional(minimum={'length': 4}),
            {'minimum': {'length': 4}},
            frozenset({'short_spec'}) | KEYWORD_FORMS,
        ),
        'InitFalseShortForm': RoundTripCase(InitFalsePattern(source='h.llo'), 'h.llo', INIT_FALSE_PATTERN_FORMS),
        'InitFalseKeywordForm': RoundTripCase(
            InitFalsePattern(source='HELLO', ignore_case=True),
            {'source': 'HELLO', 'ignore_case': True},
            INIT_FALSE_PATTERN_FORMS,
        ),
        'InitFalseDefault': RoundTripCase(InitFalseDefault(minimum=4), 4, frozenset({'short_spec'})),
        'InitFalseFirst': RoundTripCase(InitFalseFirst(minimum=4), 4, frozenset({'short_spec'})),
        'WithClassVar': RoundTripCase(WithClassVar(minimum=4), 4, frozenset({'short_spec'})),
        'InitFalseTakenByInit': RoundTripCase(
            InitFalseTakenByInit(4, 'custom'), {'minimum': 4, 'label': 'custom'}, KEYWORD_FORMS
        ),
        'InitFalseTakenByKwargs': RoundTripCase(
            InitFalseTakenByKwargs(minimum=4, bonus=1), {'minimum': 4, 'bonus': 1}, KEYWORD_FORMS
        ),
        'InitFalseRenamedByInit': RoundTripCase(
            InitFalseRenamedByInit('ell'), 'ell', frozenset({'short_spec'}) | KEYWORD_FORMS
        ),
        'InitFalseTakenByArgs': RoundTripCase(InitFalseTakenByArgs('h.llo'), 'h.llo', frozenset({'short_spec'})),
        'InitFalseOnly': RoundTripCase(
            InitFalseOnly('hello'), {'text': 'hello'}, frozenset({'short_spec'}) | KEYWORD_FORMS
        ),
    }

    @dataclass(frozen=True)
    class SchemaCase:
        """An evaluator entry and the evaluator it loads as, or `None` when the built-in schema rejects it."""

        entry: dict[str, JsonValue]
        loaded: Evaluator[str, str, str] | None = None

    BUILTIN_SCHEMA_CASES: dict[str, SchemaCase] = {
        'scalar-short-form': SchemaCase({'Equals': 5}, Equals(value=5)),
        'object-keyword-form': SchemaCase({'Equals': {'value': {'a': 1}}}, Equals(value={'a': 1})),
        'query-keyword-form': SchemaCase(
            {'HasMatchingSpan': {'query': {'name_equals': 'x'}}}, HasMatchingSpan(query={'name_equals': 'x'})
        ),
        'defaulted-short-form': SchemaCase({'EqualsExpected': 'check'}, EqualsExpected(evaluation_name='check')),
        'unknown-keyword': SchemaCase({'Equals': {'a': 1}}),
        'missing-value': SchemaCase({'Equals': {}}),
        'contains-unknown-keyword': SchemaCase({'Contains': {'a': 1}}),
        'query-fields-as-keywords': SchemaCase({'HasMatchingSpan': {'name_equals': 'x'}}),
        'mistyped-keyword': SchemaCase({'Equals': {'value': 5, 'evaluation_name': 3}}),
        'mistyped-flag': SchemaCase({'Contains': {'value': 'x', 'case_sensitive': 'no'}}),
        'empty-entry': SchemaCase({}),
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
        'KeywordOnlyVarKeyword',
        'Positional',
        'PositionalWithKeywordOnlyDefault',
        'ObjectPositional',
        'InitFalseShortForm',
        'InitFalseKeywordForm',
        'InitFalseDefault',
        'InitFalseFirst',
        'WithClassVar',
        'InitFalseTakenByInit',
        'InitFalseTakenByKwargs',
        'InitFalseRenamedByInit',
        'InitFalseTakenByArgs',
        'InitFalseOnly',
    ],
)
def test_evaluator_file_round_trip(tmp_path: Path, fmt: Literal['json', 'yaml'], evaluator_name: str):
    """File persistence retains constructor names without changing existing positional short forms.

    A file holds `__init__` arguments only, so an `init=False` field `__init__` doesn't take is left out and rebuilt
    on load. An `__init__` that takes `*args` or a parameter no field declares may fill any field, so a file keeps
    every field.
    """
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
    """The built-in schema rejects short forms the loader reads as keyword arguments, and empty entries.

    Schema generation and loading make no model request, so there is nothing to record.
    """
    case = BUILTIN_SCHEMA_CASES[case_id]
    raw: dict[str, JsonValue] = {'name': 'built-in', 'cases': [{'inputs': 'hello'}], 'evaluators': [case.entry]}
    schema = Dataset[str, str, str].model_json_schema_with_evaluators()
    assert (list(validator_for(schema)(schema).iter_errors(raw)) == []) is (case.loaded is not None)
    if case.loaded is not None:
        assert Dataset[str, str, str].from_dict(raw).evaluators == [case.loaded]


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


def test_init_false_state_changed_by_a_run(tmp_path: Path):
    """State an evaluator changes while it runs stays out of a file saved after the run."""
    evaluator = InitFalseFirst(minimum=4)
    dataset = Dataset[str, str, str](name='after-run', cases=[Case(inputs='hello')], evaluators=[evaluator])
    dataset.evaluate_sync(lambda value: value)
    assert evaluator.calls == 1
    path = tmp_path / 'dataset.yaml'
    dataset.to_file(path, custom_evaluator_types=[InitFalseFirst])

    assert yaml.safe_load(path.read_text())['evaluators'] == [{'InitFalseFirst': 4}]
    loaded = Dataset[str, str, str].from_file(path, custom_evaluator_types=[InitFalseFirst])
    assert loaded.evaluators == [InitFalseFirst(minimum=4)]


def test_init_false_field_taken_through_kwargs_declared_first(tmp_path: Path):
    """A one-key spec stays a keyword form when the field before it is `init=False` and `__init__` takes `**kwargs`.

    `__init__` takes no positional value, so a short form would not load. The saved schema of a `**kwargs` target
    requires every annotated field, so this test loads the file without checking it against that schema.
    """
    evaluator = InitFalseTakenByKwargs(minimum=4)
    dataset = Dataset[str, str, str](name='kwargs', cases=[Case(inputs='hello')], evaluators=[evaluator])
    path = tmp_path / 'dataset.yaml'
    dataset.to_file(path, custom_evaluator_types=[InitFalseTakenByKwargs])

    assert yaml.safe_load(path.read_text())['evaluators'] == [{'InitFalseTakenByKwargs': {'minimum': 4}}]
    loaded = Dataset[str, str, str].from_file(path, custom_evaluator_types=[InitFalseTakenByKwargs])
    assert loaded.evaluators == [evaluator]


def test_init_false_field_with_init_var(tmp_path: Path):
    """An `init=False` field stays out of a file when the generated `__init__` also takes an `InitVar`.

    An `InitVar` annotation fails schema generation (https://github.com/pydantic/pydantic-ai/issues/13505), so the
    file is saved without a schema.
    """
    evaluator = InitFalseWithInitVar('h.llo')
    dataset = Dataset[str, str, str](name='init-var', cases=[Case(inputs='hello')], evaluators=[evaluator])
    path = tmp_path / 'dataset.yaml'
    dataset.to_file(path, custom_evaluator_types=[InitFalseWithInitVar], schema_path=None)

    assert yaml.safe_load(path.read_text())['evaluators'] == [{'InitFalseWithInitVar': 'h.llo'}]
    loaded = Dataset[str, str, str].from_file(path, custom_evaluator_types=[InitFalseWithInitVar])
    assert loaded.evaluators == [evaluator]
    report = loaded.evaluate_sync(lambda value: value)
    assert [result.value for result in report.cases[0].assertions.values()] == [True]


@pytest.mark.parametrize('fmt', ['json', 'yaml'])
def test_init_false_report_evaluator_file_round_trip(tmp_path: Path, fmt: Literal['json', 'yaml']):
    """Report evaluators leave `init=False` fields out of a file, like case evaluators."""
    evaluator = InitFalseReport(value=4.0)
    dataset = Dataset[str, str, str](
        name='report-round-trip', cases=[Case(inputs='hello')], report_evaluators=[evaluator]
    )
    path = tmp_path / f'dataset.{fmt}'
    dataset.to_file(path, custom_report_evaluator_types=[InitFalseReport])

    raw = yaml.safe_load(path.read_text())
    assert raw['report_evaluators'] == [{'InitFalseReport': 4.0}]
    schema = json.loads((tmp_path / 'dataset_schema.json').read_text())
    assert list(validator_for(schema)(schema).iter_errors(raw)) == []
    loaded = Dataset[str, str, str].from_file(path, custom_report_evaluator_types=[InitFalseReport])
    assert loaded.report_evaluators == [evaluator]
    report = loaded.evaluate_sync(lambda value: value)
    assert report.analyses == [ScalarResult(title='Configured 4.0', value=4.0)]
