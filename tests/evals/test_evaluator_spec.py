from __future__ import annotations as _annotations

import pytest
from pydantic import ValidationError

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai._spec import _SerializedNamedSpec  # pyright: ignore[reportPrivateUsage]
    from pydantic_evals.evaluators.common import HasMatchingSpan
    from pydantic_evals.evaluators.spec import EvaluatorSpec
    from pydantic_evals.otel.span_tree import SpanQuery

pytestmark = [pytest.mark.skipif(not imports_successful(), reason='pydantic-evals not installed'), pytest.mark.anyio]


def test_evaluator_spec_basic():
    """Test basic EvaluatorSpec functionality."""
    # Test with no arguments
    spec = EvaluatorSpec(name='TestEvaluator', arguments=None)
    assert spec.name == 'TestEvaluator'
    assert spec.arguments is None
    assert spec.args == ()
    assert spec.kwargs == {}

    # Test with single positional argument
    spec = EvaluatorSpec(name='TestEvaluator', arguments=('value',))
    assert spec.name == 'TestEvaluator'
    assert spec.arguments == ('value',)
    assert spec.args == ('value',)
    assert spec.kwargs == {}

    # Test with keyword arguments
    spec = EvaluatorSpec(name='TestEvaluator', arguments={'key': 'value'})
    assert spec.name == 'TestEvaluator'
    assert spec.arguments == {'key': 'value'}
    assert spec.args == ()
    assert spec.kwargs == {'key': 'value'}


def test_evaluator_spec_deserialization():
    """Test EvaluatorSpec deserialization."""
    # Test string form
    spec = EvaluatorSpec.model_validate('TestEvaluator')
    assert spec.name == 'TestEvaluator'
    assert spec.arguments is None

    # Test single argument form
    spec = EvaluatorSpec.model_validate({'TestEvaluator': 'value'})
    assert spec.name == 'TestEvaluator'
    assert spec.arguments == ('value',)

    # Test kwargs form
    spec = EvaluatorSpec.model_validate({'TestEvaluator': {'key': 'value'}})
    assert spec.name == 'TestEvaluator'
    assert spec.arguments == {'key': 'value'}

    # Test invalid form
    with pytest.raises(ValidationError):
        EvaluatorSpec.model_validate({'TestEvaluator1': 'value', 'TestEvaluator2': 'value'})


def test_evaluator_spec_serialization():
    """Test EvaluatorSpec serialization."""
    # Test no arguments
    spec = EvaluatorSpec(name='TestEvaluator', arguments=None)
    assert spec.model_dump(context={'use_short_form': True}) == 'TestEvaluator'

    # Test single argument
    spec = EvaluatorSpec(name='TestEvaluator', arguments=('value',))
    assert spec.model_dump(context={'use_short_form': True}) == {'TestEvaluator': 'value'}

    # Test kwargs
    spec = EvaluatorSpec(name='TestEvaluator', arguments={'key': 'value'})
    assert spec.model_dump(context={'use_short_form': True}) == {'TestEvaluator': {'key': 'value'}}

    # Test without short form
    spec = EvaluatorSpec(name='TestEvaluator', arguments=None)
    assert spec.model_dump() == {'name': 'TestEvaluator', 'arguments': None}


def test_serialized_named_spec():
    """Test _SerializedNamedSpec functionality."""
    # Test string form
    spec = _SerializedNamedSpec.model_validate('TestEvaluator')

    assert spec.to_named_spec().name == 'TestEvaluator'
    assert spec.to_named_spec().arguments is None

    # Test single argument form
    spec = _SerializedNamedSpec.model_validate({'TestEvaluator': 'value'})
    assert spec.to_named_spec().name == 'TestEvaluator'
    assert spec.to_named_spec().arguments == ('value',)

    # Test kwargs form
    spec = _SerializedNamedSpec.model_validate({'TestEvaluator': {'key': 'value'}})
    assert spec.to_named_spec().name == 'TestEvaluator'
    assert spec.to_named_spec().arguments == {'key': 'value'}

    # Test invalid form
    with pytest.raises(ValueError) as exc_info:
        _SerializedNamedSpec.model_validate({'TestEvaluator1': 'value', 'TestEvaluator2': 'value'})
    assert 'Expected a single key' in str(exc_info.value)

    # Test conversion to NamedSpec (EvaluatorSpec)
    spec = _SerializedNamedSpec.model_validate('TestEvaluator')
    named_spec = spec.to_named_spec()
    assert isinstance(named_spec, EvaluatorSpec)
    assert named_spec.name == 'TestEvaluator'
    assert named_spec.arguments is None


def test_evaluator_spec_with_non_string_keys():
    """Test EvaluatorSpec with non-string keys in arguments."""
    # Test with non-string keys in dict
    spec = _SerializedNamedSpec.model_validate({'TestEvaluator': {1: 'value', 2: 'value2'}})
    assert spec.to_named_spec().name == 'TestEvaluator'
    assert spec.to_named_spec().arguments == (
        {1: 'value', 2: 'value2'},
    )  # Should be treated as a single positional argument

    # Test with mixed keys
    spec = _SerializedNamedSpec.model_validate({'TestEvaluator': {'key': 'value', 1: 'value2'}})
    assert spec.to_named_spec().name == 'TestEvaluator'
    assert spec.to_named_spec().arguments == (
        {'key': 'value', 1: 'value2'},
    )  # Should be treated as a single positional argument


def test_has_matching_span_round_trip():
    """Test that HasMatchingSpan with SpanQuery survives serialization round-trip.

    SpanQuery is a TypedDict that serializes to a dict with all-string keys.
    Without special handling, the compact tuple form (arguments=(SpanQuery(...),))
    would serialize as {HasMatchingSpan: {key: value}} which the deserializer
    interprets as kwargs instead of a single positional arg.
    """
    query = SpanQuery(some_descendant_has=SpanQuery(has_attributes={'gen_ai.tool.name': 'calculator'}))
    evaluator = HasMatchingSpan(query=query)

    # Serialize to spec
    spec = evaluator.as_spec()
    assert spec.name == 'HasMatchingSpan'
    # The arguments should use the kwargs form (not compact tuple) since
    # SpanQuery is a dict with all-string keys
    assert isinstance(spec.arguments, dict)
    assert 'query' in spec.arguments

    # Serialize to short form (as it would appear in YAML)
    short_form = spec.model_dump(context={'use_short_form': True})
    assert isinstance(short_form, dict)
    assert 'HasMatchingSpan' in short_form
    serialized_args = short_form['HasMatchingSpan']
    assert isinstance(serialized_args, dict)
    assert 'query' in serialized_args

    # Deserialize back
    deserialized_spec = EvaluatorSpec.model_validate(short_form)
    assert deserialized_spec.name == 'HasMatchingSpan'
    assert deserialized_spec.kwargs == {
        'query': {'some_descendant_has': {'has_attributes': {'gen_ai.tool.name': 'calculator'}}}
    }

    # Reconstruct the evaluator from the deserialized spec
    reconstructed = HasMatchingSpan(**deserialized_spec.kwargs)
    assert reconstructed.query == query
