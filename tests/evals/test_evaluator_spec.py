from __future__ import annotations as _annotations

import pytest
from pydantic import ValidationError

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai._spec import _SerializedNamedSpec  # pyright: ignore[reportPrivateUsage]
    from pydantic_evals.evaluators.spec import EvaluatorSpec

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
