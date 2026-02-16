"""Docker runtime lifecycle management tests.

Tests managed container creation/destruction, reference counting,
unmanaged mode backward compatibility, security flag application,
and agent auto-lifecycle integration.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from typing import Any

import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.runtime.docker import DockerRuntime
from pydantic_ai.toolsets import CodeModeToolset, FunctionToolset

from .conftest import _docker_is_available, run_code_with_tools  # pyright: ignore[reportPrivateUsage]

pytestmark = pytest.mark.skipif(not _docker_is_available(), reason='Docker is not available')


def _docker_inspect(container_id: str) -> dict[str, Any]:
    """Run docker inspect and return the parsed JSON for a container."""
    result = subprocess.run(
        ['docker', 'inspect', container_id],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f'docker inspect failed: {result.stderr}')
    return json.loads(result.stdout)[0]


def _container_exists(container_id: str) -> bool:
    """Check if a container exists (running or stopped)."""
    result = subprocess.run(
        ['docker', 'inspect', container_id],
        capture_output=True,
    )
    return result.returncode == 0


async def test_managed_lifecycle_security_flags():
    """Managed container is created with all security flags and removed on exit."""
    runtime = DockerRuntime()

    async with runtime:
        assert runtime.container_id is not None
        container_id = runtime.container_id

        info = _docker_inspect(container_id)
        host = info['HostConfig']
        config = info['Config']

        # Init (tini) for zombie reaping
        assert host['Init'] is True

        # Network isolation
        assert host['NetworkMode'] == 'none'

        # Capabilities dropped
        assert 'ALL' in (host.get('CapDrop') or [])

        # Read-only root filesystem
        assert host['ReadonlyRootfs'] is True

        # No new privileges
        assert 'no-new-privileges' in (host.get('SecurityOpt') or [])

        # Memory limits (512MB = 536870912 bytes)
        assert host['Memory'] == 536870912
        # Swap equal to memory (swap disabled)
        assert host['MemorySwap'] == host['Memory']

        # PID limit
        assert host['PidsLimit'] == 256

        # CPU limit (1 CPU = 1e9 NanoCPUs)
        assert host['NanoCpus'] == 1_000_000_000

        # User
        assert config['User'] == 'nobody'

        # Label
        assert config['Labels'].get('pydantic-ai-runtime') == 'true'

        # Tmpfs mount at /tmp
        tmpfs: dict[str, str] = host.get('Tmpfs') or {}
        assert '/tmp' in tmpfs
        tmpfs_opts: str = tmpfs['/tmp']
        assert 'noexec' in tmpfs_opts
        assert 'nosuid' in tmpfs_opts
        assert 'size=64m' in tmpfs_opts

        # Execute simple code to verify the runtime works
        result = await run_code_with_tools('1 + 1', runtime)
        assert result == 2

    # Container should be gone after exiting
    assert not _container_exists(container_id)


async def test_unmanaged_mode(_docker_container: str):
    """Pre-created container is used without lifecycle management."""
    runtime = DockerRuntime(
        container_id=_docker_container,
        python_path='python3',
        driver_path='/tmp/pydantic_ai_driver.py',
    )

    async with runtime:
        result = await run_code_with_tools('"hello"', runtime)
        assert result == 'hello'

    # Container should still exist (not managed)
    assert _container_exists(_docker_container)


async def test_reference_counting():
    """Multiple enters share one container; removed only on last exit."""
    runtime = DockerRuntime()

    # First enter creates the container
    await runtime.__aenter__()
    assert runtime.container_id is not None
    container_id = runtime.container_id

    # Second enter reuses the same container
    await runtime.__aenter__()
    assert runtime.container_id == container_id
    assert runtime._running_count == 2  # pyright: ignore[reportPrivateUsage]

    # First exit keeps container alive
    await runtime.__aexit__(None, None, None)
    assert runtime._running_count == 1  # pyright: ignore[reportPrivateUsage]
    assert _container_exists(container_id)

    # Second exit removes container
    await runtime.__aexit__(None, None, None)
    assert runtime._running_count == 0  # pyright: ignore[reportPrivateUsage]
    assert not _container_exists(container_id)


async def test_agent_auto_lifecycle():
    """Agent manages DockerRuntime lifecycle through CodeModeToolset."""

    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if not any(isinstance(m, ModelResponse) for m in messages):
            return ModelResponse(
                parts=[ToolCallPart(tool_name='run_code_with_tools', args={'code': 'await add(a=2, b=3)'})]
            )
        return ModelResponse(parts=[TextPart('5')])

    tools = FunctionToolset(tools=[add])
    runtime = DockerRuntime()

    agent = Agent(
        FunctionModel(model_function),
        toolsets=[CodeModeToolset(tools, runtime=runtime)],
    )

    # Runtime has no container before the run
    assert runtime.container_id is None

    result = await agent.run('What is 2 + 3?')
    assert result.output == '5'

    # Runtime container is cleaned up after the run
    assert runtime.container_id is None


async def test_concurrent_enter_creates_one_container():
    """Concurrent __aenter__ calls only create a single container."""
    runtime = DockerRuntime()

    # Launch two enters concurrently — _enter_lock ensures only one creates a container
    await asyncio.gather(runtime.__aenter__(), runtime.__aenter__())

    assert runtime.container_id is not None
    assert runtime._running_count == 2  # pyright: ignore[reportPrivateUsage]

    # Clean up
    await runtime.__aexit__(None, None, None)
    await runtime.__aexit__(None, None, None)
    assert runtime._running_count == 0  # pyright: ignore[reportPrivateUsage]


async def test_reuse_after_full_exit():
    """A managed runtime can be reused after a complete exit cycle."""
    runtime = DockerRuntime()

    # First cycle
    async with runtime:
        assert runtime.container_id is not None
        first_container = runtime.container_id
        result = await run_code_with_tools('1 + 1', runtime)
        assert result == 2

    # State is fully reset
    assert runtime.container_id is None
    assert runtime._managed is False  # pyright: ignore[reportPrivateUsage]
    assert runtime._driver_copied is False  # pyright: ignore[reportPrivateUsage]

    # Second cycle creates a fresh container
    async with runtime:
        assert runtime.container_id is not None
        assert runtime.container_id != first_container
        result = await run_code_with_tools('2 + 2', runtime)
        assert result == 4

    assert runtime.container_id is None


async def test_no_container_without_context_manager():
    """Calling run() without entering the context manager raises ValueError."""
    runtime = DockerRuntime()
    with pytest.raises(ValueError, match='Use it as an async context manager'):
        await run_code_with_tools('1 + 1', runtime)
