import pytest

from pydantic_evals.dataset import Case, Dataset


def test_add_case_allows_multiple_unnamed():
    dataset = Dataset(name='d', cases=[])
    dataset.add_case(inputs=1)
    dataset.add_case(inputs=2)
    assert len(dataset.cases) == 2


def test_constructor_allows_multiple_unnamed():
    dataset = Dataset(name='d', cases=[Case(inputs=1), Case(inputs=2)])
    assert len(dataset.cases) == 2


def test_add_case_rejects_duplicate_named():
    dataset = Dataset(name='d', cases=[])
    dataset.add_case(name='x', inputs=1)
    with pytest.raises(ValueError, match='Duplicate case name'):
        dataset.add_case(name='x', inputs=2)
