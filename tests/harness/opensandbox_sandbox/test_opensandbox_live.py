"""Live tests against a real OpenSandbox server, for behavior the conformance suite does not cover.

Opt in with `PYDANTIC_AI_HARNESS_OPENSANDBOX_LIVE=1` and `OPEN_SANDBOX_DOMAIN` (plus `OPEN_SANDBOX_API_KEY` if the
server requires one). A server on its Docker runtime is enough:

    uvx opensandbox-server init-config ~/.sandbox.toml --example docker
    uvx opensandbox-server

On Kubernetes without the ingress gateway, also set `OPENSANDBOX_LIVE_SERVER_PROXY=1`; on a small cluster,
`OPENSANDBOX_LIVE_RESOURCE=small` sizes each sandbox at a quarter CPU.
"""

from __future__ import annotations

import os
import uuid
from datetime import timedelta

import pytest
from opensandbox.config import ConnectionConfig
from pytest_examples import CodeExample

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.workspaces import WorkspaceTimeoutError
from pydantic_ai_harness.opensandbox_sandbox import OpenSandbox, OpenSandboxBackend

from .._docs_examples import documented_cleanup, python_blocks, run_block

LIVE_IMAGE = 'python:3.12-slim'
# Each live sandbox also expires on its own, so one a killed CI job leaves behind does not run on.
_LIVE_SANDBOX_TIMEOUT = 900

pytestmark = [pytest.mark.opensandbox_live, pytest.mark.anyio]


def live_capability() -> OpenSandbox[None]:
    """The capability the live tier uses, configured for the server the environment names."""
    connection = None
    if os.getenv('OPENSANDBOX_LIVE_SERVER_PROXY') == '1':  # pragma: no cover - live job on Kubernetes
        connection = ConnectionConfig(use_server_proxy=True)
    small = os.getenv('OPENSANDBOX_LIVE_RESOURCE') == 'small'
    return OpenSandbox(
        image=LIVE_IMAGE,
        connection=connection,
        resource={'cpu': '250m', 'memory': '512Mi'} if small else None,
        sandbox_timeout=_LIVE_SANDBOX_TIMEOUT,
    )


async def test_a_timeout_kills_the_commands_background_children() -> None:  # pragma: lax no cover
    """execd signals only a command's leader; the backend's stop takes its whole process group."""
    capability = live_capability()
    backend = OpenSandboxBackend(
        image=LIVE_IMAGE, connection=capability.connection, resource=capability.resource, sandbox_timeout=300
    )
    try:
        child = f'/tmp/child-{uuid.uuid4().hex}'
        with pytest.raises(WorkspaceTimeoutError):
            await backend.run(['sh', '-c', f'sleep 300 & echo $! > {child}; wait'], timeout=2)
        alive = await backend.run(['sh', '-c', f'[ -d /proc/$(cat {child}) ] && echo alive || echo gone'])
        assert alive.stdout == 'gone\n'
    finally:
        await backend.aclose()
        if backend.ref is not None:
            await capability.destroy(backend.ref)


async def test_attaching_renews_the_idle_timeout() -> None:  # pragma: lax no cover
    capability = live_capability()
    backend = OpenSandboxBackend(
        image=LIVE_IMAGE, connection=capability.connection, resource=capability.resource, sandbox_timeout=60
    )
    attached = None
    try:
        sandbox = await backend.get_sandbox()
        first = (await sandbox.get_info()).expires_at
        assert backend.ref is not None
        attached = OpenSandboxBackend(ref=backend.ref, connection=capability.connection, sandbox_timeout=600)
        renewed = (await (await attached.get_sandbox()).get_info()).expires_at
        assert first is not None and renewed is not None
        assert renewed - first > timedelta(minutes=8)
    finally:
        if attached is not None:
            await attached.aclose()
        await backend.aclose()
        if backend.ref is not None:
            await capability.destroy(backend.ref)


def _call_pwd_then_answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    returned = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
    if returned:
        return ModelResponse(parts=[TextPart(str(returned[-1].content))])
    return ModelResponse(parts=[ToolCallPart('pwd', {})])


async def test_a_handed_on_workspace_reaches_the_same_sandbox() -> None:  # pragma: lax no cover
    """The first run closes its transport when it ends, so the second run's operations attach again."""
    capability = live_capability()
    agent = Agent(FunctionModel(_call_pwd_then_answer), deps_type=type(None), capabilities=[capability])

    @agent.tool
    async def pwd(ctx: RunContext[None]) -> str:
        return (await ctx.workspace.run(['sh', '-c', 'pwd; hostname'])).stdout

    first = await agent.run('where are you?')
    try:
        second = await agent.run('where are you now?', workspace=first.workspace)
        assert second.output == first.output
        assert first.output.startswith('/workspace\n')
    finally:
        assert first.workspace.ref is not None
        await capability.destroy(first.workspace.ref)


_DOCS_BLOCKS = python_blocks('docs/harness/opensandbox-sandbox.md')


@pytest.mark.parametrize('example', [pytest.param(block, id=f'line {block.start_line}') for block in _DOCS_BLOCKS])
def test_docs_example(example: CodeExample) -> None:  # pragma: lax no cover
    """Every example on the docs page runs as written, and its agent's tools do their work in a real sandbox."""
    _, runs = run_block(example, cleanup=documented_cleanup(_DOCS_BLOCKS, 'remove_sandbox'))
    assert all(run.used_sandbox for run in runs), runs
    # Follow-up runs work in the first run's sandbox.
    assert len({run.ref for run in runs}) <= 1, runs
