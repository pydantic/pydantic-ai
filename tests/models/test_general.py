from __future__ import annotations as _annotations

import pytest

from pydantic_ai.models.general import GeneralModel
from ..conftest import TestEnv

pytestmark = pytest.mark.anyio

def test_params_via_args(env: TestEnv):
    env.set('PYDANTIC_AI_API_BASE_URL', 'base-url-via-env-var')
    env.set('PYDANTIC_AI_API_KEY', 'api-key-via-env-var')
    env.set('PYDANTIC_AI_MODEL', 'model-via-env-var')

    m = GeneralModel(
        base_url='base-url-via-args',
        api_key='api-key-via-args',
        model_name='model-via-args'
    )

    assert m.base_url == 'base-url-via-args'
    assert m.api_key == 'api-key-via-args'
    assert m.model_name == 'model-via-args'


def test_params_via_env_var(env: TestEnv):
    env.set('PYDANTIC_AI_API_BASE_URL', 'base-url-via-env-var')
    env.set('PYDANTIC_AI_API_KEY', 'api-key-via-env-var')
    env.set('PYDANTIC_AI_MODEL', 'model-via-env-var')

    m = GeneralModel()

    assert m.base_url == 'base-url-via-env-var'
    assert m.api_key == 'api-key-via-env-var'
    assert m.model_name == 'model-via-env-var'
