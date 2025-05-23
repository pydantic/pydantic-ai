import os
from dataclasses import dataclass

import pytest
from inline_snapshot import snapshot
from pytest_mock import MockerFixture

from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel, GeminiModelSettings

from ..conftest import try_import

with try_import() as imports_successful:
    from google.auth.transport.requests import Request


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='google-auth not installed'),
    pytest.mark.anyio,
]


@pytest.fixture(autouse=True)
def vertex_provider_auth(mocker: MockerFixture) -> None:  # pragma: lax no cover
    # Locally, we authenticate via `gcloud` CLI, so we don't need to patch anything.
    if not os.getenv('CI'):
        return

    @dataclass
    class NoOpCredentials:
        token = 'my-token'

        def refresh(self, request: Request): ...

    return_value = (NoOpCredentials(), 'pydantic-ai')
    mocker.patch('pydantic_ai.providers.google_vertex.google.auth.default', return_value=return_value)


@pytest.mark.skipif(
    not os.getenv('CI', False), reason='Requires properly configured local google vertex config to pass'
)
@pytest.mark.vcr()
async def test_labels(allow_model_requests: None) -> None:
    m = GeminiModel('gemini-2.0-flash', provider='google-vertex')
    agent = Agent(m)

    result = await agent.run(
        'What is the capital of France?',
        model_settings=GeminiModelSettings(gemini_labels={'environment': 'test', 'team': 'analytics'}),
    )
    assert result.output == snapshot('The capital of France is **Paris**.\n')
