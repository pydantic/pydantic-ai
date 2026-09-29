"""Run a docs page's Python blocks as written, with a scripted model in place of the LLM.

A sandbox provider's live tier uses this to check that every example on its docs page works
against the real service: `python_blocks` finds the blocks the same way
`test_doc_snippets.py` does, and `run_block` executes one as a script. Every model the block
names, such as `Agent('anthropic:...')`, becomes a `FunctionModel` that makes the sandbox do
real work through the tools the agent offers:

- a command tool (`shell` or `run_command`) runs a command that echoes `$HOME`;
- `write_file` and `read_file` write a file and read it back.

Then it answers in text. Each run the model finishes is returned as a `ScriptedRun`, and
`ScriptedRun.used_sandbox` says whether the tools came back with what they should have.

Sandboxes a block creates keep running after it, so `run_block` passes every workspace ref
the model saw to the provider's `cleanup`, even when the block fails. `documented_cleanup` takes
that cleanup from the page itself. A block that connects to Temporal at `localhost:7233` runs as
a script file against a local dev server started for it, and a DBOS block runs as a script file.
"""

from __future__ import annotations

import inspect
import os
import runpy
import tempfile
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from unittest import mock

import anyio
from anyio.from_thread import start_blocking_portal
from pytest_examples import CodeExample, find_examples

from pydantic_ai import models
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from pydantic_ai.workspaces import WorkspaceRef

_ROOT = Path(__file__).parents[2]  # the repository root, above `tests/harness`

_TEMPORAL_ADDRESS = 'localhost:7233'

HOME_ECHO = 'sandbox HOME='
"""What the scripted command prints before `$HOME`."""


@dataclass
class ScriptedRun:
    """One agent run the scripted model finished."""

    ref: WorkspaceRef | None
    """The workspace the run's tools used, or `None` if none was created."""
    expected: dict[str, str] = field(default_factory=dict[str, str])
    """For each tool the model called, text its result must contain."""
    outputs: dict[str, str] = field(default_factory=dict[str, str])
    """For each tool the model called, what came back: its result or its retry prompt."""

    @property
    def used_sandbox(self) -> bool:
        """The run created or reattached a workspace, and every tool returned what it should have."""
        return self.ref is not None and all(text in self.outputs.get(name, '') for name, text in self.expected.items())


def python_blocks(*paths: str) -> list[CodeExample]:
    """Every ```python block in the markdown files at `paths`, relative to the repo root."""
    os.chdir(_ROOT)  # `find_examples` wants paths relative to the cwd, like `test_doc_snippets.py`.
    return list(find_examples(*paths))


def run_block(
    example: CodeExample, *, cleanup: Callable[[WorkspaceRef], Awaitable[None]] | None = None
) -> tuple[dict[str, object], list[ScriptedRun]]:
    """Execute `example` as a script, then pass every workspace it created to `cleanup`.

    Returns the script's globals, so a block that only defines a function hands it back, and
    the runs the scripted model finished.
    """
    runs: list[ScriptedRun] = []
    refs: list[WorkspaceRef] = []
    respond = _script(runs, refs)

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
        for part in respond(messages, info).parts:
            if isinstance(part, ToolCallPart):
                yield {0: DeltaToolCall(name=part.tool_name, json_args=part.args_as_json_str())}
            else:
                assert isinstance(part, TextPart)
                yield part.content

    model = FunctionModel(respond, stream_function=stream)
    namespace: dict[str, object] = {'__name__': '__main__'}
    code = compile(example.source, f'{example.path}:{example.start_line}', 'exec')

    def infer_model(*_args: object, **_kwargs: object) -> FunctionModel:
        return model

    try:
        # A durable worker rebuilds a model named by a string with its own `infer_model` import.
        with (
            mock.patch.object(models, 'infer_model', infer_model),
            mock.patch('pydantic_ai.durable_exec._base.infer_model', infer_model),
        ):
            if _TEMPORAL_ADDRESS in example.source:
                namespace = _run_temporal_script(example)
            elif 'from dbos import' in example.source:
                # DBOS reads a workflow function's source when registering it, which `exec` has none of.
                namespace = _run_script(example)
            else:
                exec(code, namespace)
    finally:
        if refs:
            assert cleanup is not None, (
                f'{example.path}:{example.start_line} created a workspace and nothing cleans it up'
            )
            with ThreadPoolExecutor(max_workers=1) as pool:
                for ref in refs:
                    # A thread of its own: the calling test may already have an event loop running.
                    pool.submit(anyio.run, cleanup, ref).result()
    return namespace, runs


def _run_temporal_script(example: CodeExample) -> dict[str, object]:
    """Run `example` as a script file against a local Temporal dev server."""
    from temporalio.testing import WorkflowEnvironment

    # The block runs its own `asyncio.run(main())`, so the server lives on a portal thread's loop.
    async def start() -> WorkflowEnvironment:
        return await WorkflowEnvironment.start_local()  # pyright: ignore[reportUnknownMemberType]

    with start_blocking_portal() as portal:
        env = portal.call(start)
        try:
            # Temporal validates a workflow by re-importing its module, so `__main__` must be a
            # real file, as with `python example.py`; `exec` would leave pytest as `__main__`.
            address = env.client.service_client.config.target_host
            return _run_script(replace(example, source=example.source.replace(_TEMPORAL_ADDRESS, address)))
        finally:
            portal.call(env.shutdown)


def _run_script(example: CodeExample) -> dict[str, object]:
    """Run `example` from a script file, as `python example.py` would."""
    with tempfile.TemporaryDirectory() as directory:
        script = Path(directory) / 'example.py'
        # Padding keeps traceback line numbers equal to the docs page's.
        script.write_text('\n' * (example.start_line - 1) + example.source)
        return runpy.run_path(str(script), run_name='__main__')


def documented_cleanup(examples: list[CodeExample], name: str) -> Callable[[WorkspaceRef], Awaitable[None]]:
    """The async `name(ref)` a docs page defines to clean up a sandbox, to pass to `run_block`.

    Cleaning up every other block's sandboxes with it checks that the documented cleanup works.
    """
    (example,) = [example for example in examples if f'async def {name}(' in example.source]
    documented = run_block(example)[0][name]
    assert callable(documented)

    async def cleanup(ref: WorkspaceRef) -> None:
        done = documented(ref)
        assert inspect.isawaitable(done)
        await done

    return cleanup


def _script(
    runs: list[ScriptedRun], refs: list[WorkspaceRef]
) -> Callable[[list[ModelMessage], AgentInfo], ModelResponse]:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        this_run = messages[_run_start(messages) :]
        for message in this_run:
            if isinstance(message, ModelResponse) and message.workspace_ref and message.workspace_ref not in refs:
                refs.append(message.workspace_ref)

        made = [part for message in this_run for part in message.parts if isinstance(part, ToolCallPart)]
        calls = _calls({tool.name for tool in info.function_tools}, made)
        if len(made) < len(calls):
            name, arguments, _ = calls[len(made)]
            return ModelResponse(parts=[ToolCallPart(name, arguments)])

        run = ScriptedRun(ref=next((m.workspace_ref for m in reversed(this_run) if isinstance(m, ModelResponse)), None))
        run.expected = {name: expected for name, _, expected in calls}
        for message in this_run:
            for part in message.parts:
                if isinstance(part, (ToolReturnPart, RetryPromptPart)) and part.tool_name:
                    run.outputs[part.tool_name] = str(part.content)
        runs.append(run)
        return ModelResponse(parts=[TextPart('Done.')])

    return respond


def _run_start(messages: list[ModelMessage]) -> int:
    """Index of the request that started the current run: the last one with a user prompt."""
    return max(
        index
        for index, message in enumerate(messages)
        if isinstance(message, ModelRequest) and any(isinstance(part, UserPromptPart) for part in message.parts)
    )


def _calls(offered: set[str], made: list[ToolCallPart]) -> list[tuple[str, dict[str, object], str]]:
    """The tool calls a run makes, as `(tool, arguments, text its result must contain)`.

    `made` is the calls the run already made, so the file content stays the one it wrote.
    """
    calls: list[tuple[str, dict[str, object], str]] = []
    command_tool = next((name for name in ('shell', 'run_command') if name in offered), None)
    if command_tool:
        calls.append((command_tool, {'command': f'echo "{HOME_ECHO}$HOME"'}, f'{HOME_ECHO}/'))
    if {'write_file', 'read_file'} <= offered:
        written = next((call.args_as_dict() for call in made if call.tool_name == 'write_file'), None)
        content = str(written['content']) if written else f'written by a docs example {uuid.uuid4().hex}'
        calls.append(('write_file', {'path': 'docs-example.txt', 'content': content}, 'docs-example.txt'))
        calls.append(('read_file', {'path': 'docs-example.txt'}, content))
    return calls
