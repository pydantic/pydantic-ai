from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

import pydantic_harness


def test_import():
    assert pydantic_harness.__doc__ is not None
    assert isinstance(pydantic_harness.__all__, list)


def test_test_model_fixture(test_model: TestModel):
    assert isinstance(test_model, TestModel)


def test_test_agent_fixture(test_agent: Agent[None, str]):
    assert test_agent.name == 'test-agent'


def test_tmp_dir_fixture(tmp_dir: Path):
    assert tmp_dir.is_dir()


async def test_allow_model_requests(allow_model_requests: None):
    import pydantic_ai.models

    assert pydantic_ai.models.ALLOW_MODEL_REQUESTS is True
