"""Logfire-hosted helpers in offline-simulation mode (push/pull round-trip + managed rubric)."""

from __future__ import annotations

import pytest

from easy_evals import eval_suite
from easy_evals.hosted import hosted_rubric, live, pull, push
from fake_models import qa_agent
from pydantic_evals import Dataset


def test_offline_mode_when_no_token():
    assert live() is False  # no LOGFIRE_TOKEN / experimental client in this env


def test_push_then_pull_roundtrips_a_real_dataset():
    suite = eval_suite(qa_agent())
    suite.case('What is the capital of France?', expect='Paris')
    url = push(suite, 'roundtrip-set')
    assert 'roundtrip-set' in url

    dataset = pull('roundtrip-set')
    assert isinstance(dataset, Dataset)
    assert len(dataset.cases) == 1


def test_pull_unknown_dataset_raises():
    with pytest.raises(KeyError):
        pull('does-not-exist')


def test_hosted_rubric_resolves_managed_variable():
    rubric = hosted_rubric('prompt__haiku_rubric', label='production')
    assert 'haiku' in rubric
    assert hosted_rubric('prompt__unknown', default='be nice') == 'be nice'
