"""Strict mode stays off for every tool and output schema CLAI sends, whatever the provider."""

import io
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel
from rich.console import Console

from pydantic_ai import Agent, NativeOutput, Tool
from pydantic_ai.capabilities import Capability
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.output import OutputSpec
from pydantic_clai2 import open_stock_agent
from pydantic_clai2.config import Settings
from pydantic_clai2.config.project_settings import ProjectSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.runtime.session_runtime import create_shell
from pydantic_clai2.runtime.strict_schemas import NON_STRICT_SCHEMAS
from tests.harness._strict_requests import PROVIDERS, capturing_model, strict_tools

pytestmark = pytest.mark.usefixtures('allow_model_requests')


def lookup(query: str) -> str:
    """Look something up."""
    return query  # pragma: no cover


def search(query: str) -> str:
    """Search for something."""
    return query  # pragma: no cover


class Answer(BaseModel):
    text: str


@pytest.mark.parametrize('provider', PROVIDERS)
async def test_stock_agent_sends_every_tool_non_strict(tmp_path: Path, provider: str) -> None:
    bodies: list[dict[str, Any]] = []
    neighbors = Capability[None](tools=[Tool(lookup, strict=True), search])

    async with open_stock_agent(
        workspace=tmp_path, model=capturing_model(provider, bodies), capabilities=[neighbors]
    ) as agent:
        with pytest.raises(ModelHTTPError):
            await agent.run('go')

    sent = strict_tools(provider, bodies[0])
    assert {'read_file', 'shell', 'delegate_task', 'lookup', 'search'} <= sent.keys()
    assert not any(sent.values())


@pytest.mark.parametrize('provider', PROVIDERS)
async def test_tools_that_ask_for_strict_mode_are_sent_non_strict(provider: str) -> None:
    bodies: list[dict[str, Any]] = []
    agent = Agent(
        capturing_model(provider, bodies), tools=[Tool(lookup, strict=True)], capabilities=[NON_STRICT_SCHEMAS]
    )

    with pytest.raises(ModelHTTPError):
        await agent.run('go')

    assert strict_tools(provider, bodies[0]) == {'lookup': False}


async def test_session_runs_bind_non_strict_schemas(tmp_path: Path) -> None:
    shell = create_shell(
        Agent(),
        deps=None,
        plugins=(),
        usage_limits=None,
        console=Console(file=io.StringIO()),
        settings=Settings(model=None),
        store=SettingsStore(tmp_path / 'config.db'),
        builtin_plugins=(),
        project=ProjectSettings(),
        headless=True,
    )

    assert NON_STRICT_SCHEMAS in shell.run_plugins()


@pytest.mark.parametrize('output_type', [Answer, NativeOutput(Answer)], ids=['tool', 'native'])
async def test_output_schemas_are_sent_non_strict(output_type: OutputSpec[Answer]) -> None:
    bodies: list[dict[str, Any]] = []
    agent = Agent(capturing_model('openai', bodies), output_type=output_type, capabilities=[NON_STRICT_SCHEMAS])

    with pytest.raises(ModelHTTPError):
        await agent.run('go')

    (body,) = bodies
    if 'response_format' in body:
        assert body['response_format']['json_schema']['strict'] is False
    else:
        assert strict_tools('openai', body) == {'final_result': False}
