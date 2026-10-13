from __future__ import annotations

import importlib.util
import inspect
import json
import os
from collections.abc import Awaitable, Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Concatenate, ParamSpec, Protocol, TypeAlias, TypeGuard, TypeVar, overload

import pytest
from pydantic import TypeAdapter

_render_spec = importlib.util.find_spec('render')

if _render_spec is None and os.environ.get('CI') == 'true':  # pragma: no cover - CI's render extra is installed
    raise pytest.UsageError('Render tests require the `render` SDK in CI. Install the Harness render extra.')

if TYPE_CHECKING:
    from render.workflows import Options, Retry, TaskContext, TaskDefinition, TaskRunMetadata, Workflows

    from pydantic_ai import Agent
    from pydantic_ai.mcp import MCPToolset
    from pydantic_ai.usage import RunUsage
    from pydantic_ai_harness import RenderWorkflows
elif _render_spec is None:  # pragma: no cover - the optional SDK is installed in this test suite
    collect_ignore_glob = ['*.py']

    class Workflows:
        """Placeholder used only while pytest ignores Render-extra tests."""

    class TaskContext:
        """Placeholder used only while pytest ignores Render-extra tests."""
else:
    from render.workflows import Options, Retry, TaskContext, TaskDefinition, TaskRunMetadata, Workflows


def pytest_ignore_collect(collection_path: Path) -> bool:
    """Ignore Render-extra tests only when the top-level optional package is absent."""
    return _render_spec is None and collection_path.suffix == '.py'


P = ParamSpec('P')
R = TypeVar('R')


class RecordingTaskDecorator(Protocol):
    """Typed decorator returned by `RecordingWorkflows.task(...)`."""

    @overload
    def __call__(self, func: Callable[Concatenate[TaskContext, P], Awaitable[R]], /) -> TaskDefinition[P, R]: ...

    @overload
    def __call__(self, func: Callable[Concatenate[TaskContext, P], R], /) -> TaskDefinition[P, R]: ...


class RecordingWorkflows(Workflows):
    """Record task names and registration-time Options through the public decorator."""

    def __init__(self) -> None:
        super().__init__()
        self.options: dict[str, Options] = {}
        self.registered_task_names: list[str] = []

    @overload
    def task(
        self,
        func: Callable[Concatenate[TaskContext, P], Awaitable[R]],
        /,
    ) -> TaskDefinition[P, R]: ...

    @overload
    def task(
        self,
        func: Callable[Concatenate[TaskContext, P], R],
        /,
    ) -> TaskDefinition[P, R]: ...

    @overload
    def task(
        self,
        *,
        name: str | None = None,
        retry: Retry | None = None,
        timeout_seconds: int | None = None,
        plan: str | None = None,
    ) -> RecordingTaskDecorator: ...

    def task(
        self,
        func: Callable[..., object] | None = None,
        *,
        name: str | None = None,
        retry: Retry | None = None,
        timeout_seconds: int | None = None,
        plan: str | None = None,
    ) -> object:
        decorate = super().task(name=name, retry=retry, timeout_seconds=timeout_seconds, plan=plan)

        def record(target: Callable[..., object]) -> TaskDefinition[..., object]:
            definition = decorate(target)
            self.registered_task_names.append(definition.name)
            self.options[definition.name] = Options(retry=retry, timeout_seconds=timeout_seconds, plan=plan)
            return definition

        return record if func is None else record(func)


_RENDER_CREDENTIAL_KEYS = frozenset(
    {
        'RENDER_API_KEY',
        'RENDER_CLI_TOKEN',
        'RENDER_TOKEN',
        'RENDER_WORKSPACE_ID',
    }
)


def renderless_environment() -> dict[str, str]:
    """Return the current environment without Render credentials."""
    return {
        key: value
        for key, value in os.environ.items()
        if key not in _RENDER_CREDENTIAL_KEYS and not key.startswith('COVERAGE_')
    }


class RecordingTaskContext(TaskContext):
    """Run child tasks through JSON in this process and record their public names."""

    def __init__(self) -> None:
        self.task_names: list[str] = []

    @property
    def metadata(self) -> TaskRunMetadata:
        """No Render run IDs exist for tasks executed by this fixture."""
        return TaskRunMetadata()

    async def run(self, task: TaskDefinition[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        self.task_names.append(task.name)
        result = task.func(self, *json_round_trip(args), **json_round_trip(kwargs))
        if inspect.isawaitable(result):
            return json_round_trip(await result)
        return json_round_trip(result)


def json_round_trip(value: R) -> R:
    """Copy JSON payloads while retaining their outer Python call-container type."""
    return TypeAdapter(type(value)).validate_json(json.dumps(value))


async def run_agent_in_task(
    agent: Agent[None, str],
    runtime: RenderWorkflows[None],
    context: TaskContext,
    *,
    prompt: str = 'go',
    usage: RunUsage | None = None,
    model: str | None = None,
) -> str:
    """Run an agent through the public Render entry-task decorator."""

    @runtime.task
    async def root(ctx: TaskContext) -> str:
        del ctx
        return (await agent.run(prompt, usage=usage, model=model)).output

    pending = root.func(context)
    assert inspect.isawaitable(pending)
    return await pending


@dataclass
class ToolTaskConcurrency:
    """Count overlapping tool tasks, including time spent waiting at test barriers."""

    active: int = 0
    maximum: int = 0

    @contextmanager
    def track(self, task_name: str) -> Generator[bool]:
        is_tool = task_name.endswith('.call_tool')
        if is_tool:
            self.active += 1
            self.maximum = max(self.maximum, self.active)
        try:
            yield is_tool
        finally:
            if is_tool:
                self.active -= 1


Tamper: TypeAlias = Callable[[dict[str, object]], None]
_ENVELOPE = TypeAdapter(dict[str, object])


def is_envelope(value: object) -> TypeGuard[dict[str, object]]:
    """Narrow a task argument or result to its JSON object envelope."""
    return isinstance(value, dict)


class TaskBoundary(RecordingTaskContext):
    """Runs child tasks in this process while recording the JSON envelopes that cross.

    A `tamper` hook rewrites that JSON in place, which is how a foreign or future worker's
    bytes reach the reader. Only JSON data is fabricated, only at this public boundary, and
    every envelope is validated by a `TypeAdapter` before it is rewritten.
    """

    def __init__(self, *, tamper_request: Tamper | None = None, tamper_result: Tamper | None = None) -> None:
        super().__init__()
        self.requests: list[dict[str, object]] = []
        self.results: list[dict[str, object]] = []
        self.started: list[str] = []
        self._tamper_request = tamper_request
        self._tamper_result = tamper_result

    async def run(self, task: TaskDefinition[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        self.started.append(task.name)
        request: object = args[0] if args else None
        assert is_envelope(request)
        self.requests.append(dict(request))
        if self._tamper_request is not None:
            self._tamper_request(request)
        result = await super().run(task, *args, **kwargs)
        self._returned(result)
        return result

    def _returned(self, result: object) -> None:
        assert is_envelope(result)
        if self._tamper_result is not None:
            self._tamper_result(result)
        self.results.append(dict(result))


def rewrite_tool_context(tamper: Tamper) -> Tamper:
    """Rewrite only a function worker's serialized context, leaving model tasks intact."""

    def rewrite(request: dict[str, object]) -> None:
        if not str(request['operation']).endswith('.call_tool'):
            return
        payload = _ENVELOPE.validate_python(request['payload'])
        context = _ENVELOPE.validate_python(payload['run_context'])
        fields = _ENVELOPE.validate_python(context['context'])
        tamper(fields)
        context['context'] = fields
        payload['run_context'] = context
        request['payload'] = payload

    return rewrite


MCP_DEPENDENCY_MODULE = 'fastmcp'


def mcp_dependency_installed() -> bool:
    """Whether the optional MCP dependency `MCPToolset` needs is importable at all.

    A present-but-broken MCP installation still has a discoverable module, so the import in
    `mcp_toolset` below raises there instead of being turned into a skip.
    """
    return importlib.util.find_spec(MCP_DEPENDENCY_MODULE) is not None


def mcp_toolset() -> tuple[MCPToolset[None], list[tuple[str, dict[str, Any]]]]:
    """Exercise the real MCP client and server without a provider or network service."""
    if not mcp_dependency_installed():
        pytest.skip(
            f'`MCPToolset` needs the optional `{MCP_DEPENDENCY_MODULE}` client from the `mcp` extra.'
        )  # pragma: no cover - optional local dependency guard

    from fastmcp import FastMCP

    from pydantic_ai.mcp import MCPToolset

    calls: list[tuple[str, dict[str, Any]]] = []
    server = FastMCP('render-tools')

    @server.tool
    async def remote_lookup(query: str) -> str:
        calls.append(('remote_lookup', {'query': query}))
        return 'remote result'

    return MCPToolset[None](server, id='remote-tools'), calls
