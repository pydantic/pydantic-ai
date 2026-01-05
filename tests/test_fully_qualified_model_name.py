"""Tests for fully_qualified_model_name property across all Model subclasses."""

from pydantic_ai.models.test import TestModel
from pydantic_ai.models.wrapper import WrapperModel


def test_test_model_fully_qualified_name():
    """Test TestModel.fully_qualified_model_name format."""
    model = TestModel()
    assert model.fully_qualified_model_name == 'test:test'
    assert ':' in model.fully_qualified_model_name


def test_test_model_has_provider_prefix():
    """Test that fully_qualified_model_name contains provider:name format."""
    model = TestModel()
    fqn = model.fully_qualified_model_name
    parts = fqn.split(':')
    assert len(parts) >= 2, f'Fully qualified name should contain provider:name format, got {fqn}'
    assert parts[0] == 'test'


def test_wrapper_model_delegates_fully_qualified_name():
    """Test WrapperModel delegates fully_qualified_model_name to wrapped model."""
    wrapped = TestModel()
    wrapper = WrapperModel(wrapped)
    assert wrapper.fully_qualified_model_name == wrapped.fully_qualified_model_name
    assert wrapper.fully_qualified_model_name == 'test:test'


def test_fully_qualified_name_matches_system_and_model_name():
    """Test that fully_qualified_model_name combines system and model_name correctly."""
    model = TestModel()
    fqn = model.fully_qualified_model_name
    assert model.system in fqn, f'System {model.system} should be in {fqn}'
    assert model.model_name in fqn, f'Model name {model.model_name} should be in {fqn}'
