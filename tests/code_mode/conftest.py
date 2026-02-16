"""Shared fixtures and test tools for code mode tests."""

from __future__ import annotations

import shutil
import subprocess
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from typing_extensions import TypedDict

from pydantic_ai._python_signature import FunctionSignature, TypeSignature
from pydantic_ai._run_context import RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.runtime.abstract import CodeRuntime, ToolCallback
from pydantic_ai.runtime.docker import DockerRuntime
from pydantic_ai.toolsets.code_mode import CodeModeToolset
from pydantic_ai.toolsets.function import FunctionToolset
from pydantic_ai.usage import RunUsage


# Define return type as TypedDict for better type hints in signatures
class WeatherResult(TypedDict):
    """Weather data for a city."""

    city: str
    temperature: float
    unit: str
    conditions: str


# Simulated weather data for test cities
_WEATHER_DATA: dict[str, dict[str, Any]] = {
    'London': {'temperature': 15.0, 'conditions': 'cloudy'},
    'Paris': {'temperature': 18.0, 'conditions': 'sunny'},
    'Tokyo': {'temperature': 22.0, 'conditions': 'rainy'},
    'New York': {'temperature': 12.0, 'conditions': 'windy'},
    'Sydney': {'temperature': 25.0, 'conditions': 'sunny'},
}


def get_weather(city: str) -> WeatherResult:
    """Get weather for a city.

    Args:
        city: Name of the city to get weather for.

    Returns:
        Weather data including temperature and conditions.
    """
    data = _WEATHER_DATA.get(city, {'temperature': 20.0, 'conditions': 'unknown'})
    return {'city': city, 'temperature': data['temperature'], 'unit': 'celsius', 'conditions': data['conditions']}


def build_run_context() -> RunContext[None]:
    """Build a minimal RunContext for direct call_tool tests."""
    return RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
        messages=[],
        run_step=0,
    )


async def build_code_mode_toolset(
    runtime: CodeRuntime,
    *tool_specs: tuple[Callable[..., Any], bool],
) -> tuple[CodeModeToolset[None], dict[str, Any]]:
    """Build and initialize a CodeModeToolset, returning it along with its tools dict."""
    toolset: FunctionToolset[None] = FunctionToolset()
    for func, takes_ctx in tool_specs:
        toolset.add_function(func, takes_ctx=takes_ctx)
    code_mode = CodeModeToolset(toolset, runtime=runtime)
    ctx = build_run_context()
    tools = await code_mode.get_tools(ctx)
    return code_mode, tools


async def run_code_with_tools(
    code: str,
    runtime: CodeRuntime,
    *tool_specs: tuple[Callable[..., Any], bool],
) -> Any:
    """Run code through CodeModeToolset. Each tool_spec is (function, takes_ctx)."""
    code_mode, tools = await build_code_mode_toolset(runtime, *tool_specs)
    ctx = build_run_context()
    return await code_mode.call_tool('run_code_with_tools', {'code': code}, ctx, tools['run_code_with_tools'])


class StubRuntime(CodeRuntime):
    """Minimal CodeRuntime for testing CodeModeToolset logic without pydantic-monty."""

    async def run(
        self,
        code: str,
        call_tool: ToolCallback,
        *,
        functions: dict[str, FunctionSignature],
        referenced_types: list[TypeSignature],
    ) -> Any:
        raise NotImplementedError('StubRuntime does not execute code')


def _docker_is_available() -> bool:
    """Check whether Docker is installed and the daemon is reachable."""
    if not shutil.which('docker'):
        return False
    try:
        subprocess.run(['docker', 'info'], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    return True


@pytest.fixture(scope='session')
def _docker_container() -> Iterator[str]:
    """Create a long-lived Docker container for the test session.

    Copies the driver script into the container so that DockerRuntime
    can execute code inside it.
    """
    container_name = f'pydantic-ai-test-{uuid.uuid4().hex[:8]}'
    subprocess.run(
        ['docker', 'run', '-d', '--name', container_name, 'python:3.12-slim', 'sleep', 'infinity'],
        check=True,
        capture_output=True,
    )
    driver_src = Path(__file__).parents[2] / 'pydantic_ai_slim' / 'pydantic_ai' / 'runtime' / '_driver.py'
    subprocess.run(
        ['docker', 'cp', str(driver_src), f'{container_name}:/tmp/pydantic_ai_driver.py'],
        check=True,
        capture_output=True,
    )
    yield container_name
    subprocess.run(['docker', 'rm', '-f', container_name], capture_output=True)


@pytest.fixture
async def managed_docker_runtime() -> AsyncIterator[DockerRuntime]:
    """A managed DockerRuntime that creates and destroys its own container."""
    runtime = DockerRuntime()
    async with runtime:
        yield runtime


@pytest.fixture(params=['monty', 'docker'])
def code_runtime(request: pytest.FixtureRequest) -> CodeRuntime:
    """Parameterized fixture providing each CodeRuntime implementation."""
    if request.param == 'monty':
        try:
            from pydantic_ai.runtime.monty import MontyRuntime
        except ImportError:
            pytest.skip('pydantic-monty is not installed')

        return MontyRuntime()

    if not _docker_is_available():
        pytest.skip('Docker is not available')

    container_id: str = request.getfixturevalue('_docker_container')
    return DockerRuntime(
        container_id=container_id,
        python_path='python3',
        driver_path='/tmp/pydantic_ai_driver.py',
    )
