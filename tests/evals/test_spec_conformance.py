"""Every built-in evaluator's `Dataset` schema publishes its parameters, and every form it publishes loads.

See `tests/spec_conformance.py` for what each check does, and `tests/test_spec_conformance.py` for the
capabilities. The loader constructs an evaluator class directly, so each spec's arguments bind to its `__init__`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pytest
import yaml
from pydantic import JsonValue

from ..conftest import try_import
from ..spec_conformance import (
    JsonSchema,
    argument_errors,
    definitions,
    form_instances,
    name_forms,
    published_parameters,
    schema_errors,
    spec_parameters,
)

with try_import() as imports_successful:
    from pydantic_ai._spec import NamedSpec
    from pydantic_evals import Dataset
    from pydantic_evals.evaluators.common import DEFAULT_EVALUATORS
    from pydantic_evals.evaluators.report_common import DEFAULT_REPORT_EVALUATORS

pytestmark = pytest.mark.skipif(not imports_successful(), reason='pydantic-evals not installed')

Kind = Literal['evaluators', 'report_evaluators']
EVALUATORS: dict[str, tuple[Kind, type[object]]] = (
    {
        **{cls.__name__: ('evaluators', cls) for cls in DEFAULT_EVALUATORS},
        **{cls.__name__: ('report_evaluators', cls) for cls in DEFAULT_REPORT_EVALUATORS},
    }
    if imports_successful()
    else {}
)


KNOWN_UNWRITABLE: dict[str, str] = {
    'ConfusionMatrixEvaluator': 'https://github.com/pydantic/pydantic-ai/issues/13979',
}
"""Evaluators `to_file()` can write in a form the schema doesn't publish, which an entry expects until it is fixed."""


@pytest.fixture(scope='module')
def schema() -> JsonSchema:
    return Dataset[str, str, str].model_json_schema_with_evaluators()


def _forms(schema: JsonSchema, kind: Kind, name: str) -> list[JsonSchema]:
    properties = schema['properties']
    assert isinstance(properties, dict)
    entries = properties[kind]
    assert isinstance(entries, dict)
    items = entries['items']
    assert isinstance(items, dict)
    return name_forms(items, name)


@pytest.mark.parametrize('name', sorted(EVALUATORS))
def test_schema_publishes_spec_parameters(name: str, schema: JsonSchema):
    kind, cls = EVALUATORS[name]
    published = published_parameters(cls, name, _forms(schema, kind, name), definitions(schema))
    missing = {p: why for p, why in spec_parameters(cls).items() if p not in published}
    assert not missing, f'The `Dataset` schema leaves out parameters of `{name}`: {missing}'


@pytest.mark.parametrize('name', sorted(EVALUATORS))
def test_spec_forms_load(name: str, schema: JsonSchema, tmp_path: Path):
    """Every instance loads, and the dataset holding them all validates against the schema and survives `to_file()`.

    The written file validates against the schema `to_file()` writes beside it, except for a `KNOWN_UNWRITABLE` entry.
    """
    kind, cls = EVALUATORS[name]
    instances = form_instances(_forms(schema, kind, name), definitions(schema))
    assert instances
    for instance in instances:
        spec = NamedSpec.model_validate(instance)
        errors = argument_errors(cls, spec.args, spec.kwargs)
        assert not errors, f'The published `{name}` form instance {instance!r} does not load: {errors}'

    raw: dict[str, JsonValue] = {'name': 'conformance', 'cases': [{'inputs': 'hello'}], kind: instances}
    assert not schema_errors(schema, raw)
    dataset = Dataset[str, str, str].from_dict(raw)
    path = tmp_path / 'dataset.yaml'
    dataset.to_file(path)
    written = schema_errors(
        json.loads((tmp_path / 'dataset_schema.json').read_text()), yaml.safe_load(path.read_text())
    )
    assert bool(written) is (name in KNOWN_UNWRITABLE), written
    loaded = Dataset[str, str, str].from_file(path)
    assert (loaded.evaluators, loaded.report_evaluators) == (dataset.evaluators, dataset.report_evaluators)


def test_evaluator_names_are_unique():
    assert len(EVALUATORS) == len(DEFAULT_EVALUATORS) + len(DEFAULT_REPORT_EVALUATORS)


def test_unwritable_entries_name_evaluators():
    assert KNOWN_UNWRITABLE.keys() <= EVALUATORS.keys()
