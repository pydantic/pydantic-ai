"""The `coder` plugin's opt-in `report_stale_files` setting, over harness `ReportStaleFiles`."""

import io
import json
from collections.abc import AsyncIterator
from pathlib import Path

import anyio
import pytest
from pydantic import JsonValue
from rich.console import Console

from pydantic_ai.capabilities import AgentCapability
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.report_stale_files import ReportStaleFiles, StaleFilesPart
from pydantic_ai_pax import Session
from pydantic_ai_pax.app import create_stock_agent
from pydantic_ai_pax.builtin_plugins.coder import CoderPlugin, CoderSettings, CoderSource
from pydantic_ai_pax.plugins import PluginHost, collect
from tests.function_model_prefix import REQUEST_SCOPED_PROMPT_DROPPED

CHANGED = (
    'These files changed in the workspace after you last read or wrote them, so what you saw of them is out of date. '
    'Read them again before relying on them or editing them:\n'
    '- `notes.txt`'
)


def coder_capabilities(**settings: JsonValue) -> list[AgentCapability[None]]:
    """What the `coder` plugin binds with `settings`, without skills or delegation."""
    host = PluginHost[None](
        name='coder',
        console=Console(file=io.StringIO()),
        settings={'repo_context': False, 'sub_agents': False, 'skill_folders': [], **settings},
    )
    return list(collect(CoderPlugin[None].from_host(host)).capabilities)


def notices(messages: list[ModelMessage]) -> list[list[str]]:
    """The stale-file notices each model request carried, one list per request."""
    requests: list[list[str]] = [[]]
    for message in messages:
        if isinstance(message, ModelResponse):
            requests.append([])
        else:
            requests[-1].extend(part.content for part in message.parts if isinstance(part, StaleFilesPart))
    return requests[:-1]


def reads_then_answers(call: ToolCallPart) -> FunctionModel:
    """A model that makes `call` at the start of each prompt, then answers."""

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
        if any(isinstance(part, UserPromptPart) for part in messages[-1].parts):
            yield {0: DeltaToolCall(name=call.tool_name, json_args=call.args_as_json_str())}
        else:
            yield 'done'

    return FunctionModel(stream_function=stream)


def test_off_by_default_binds_only_coder() -> None:
    assert CoderSettings().report_stale_files is False
    [coder] = coder_capabilities()
    assert isinstance(coder, Coder)
    [coder, stale] = coder_capabilities(report_stale_files=True)
    assert isinstance(coder, Coder) and isinstance(stale, ReportStaleFiles)


@pytest.mark.moves_cache_prefix(reason=REQUEST_SCOPED_PROMPT_DROPPED)
@pytest.mark.parametrize('enabled', [False, True])
async def test_a_file_edited_between_prompts_is_named_on_the_next_request(tmp_path: Path, enabled: bool) -> None:
    """The user edits a file the model read; the next prompt's request says so, only with the setting on."""
    notes = tmp_path / 'notes.txt'
    notes.write_text('first draft\n')
    session = Session(
        create_stock_agent(reads_then_answers(ToolCallPart('read_file', {'path': 'notes.txt'}))),
        deps=None,
        workspace=tmp_path,
        plugins=coder_capabilities(report_stale_files=enabled),
    )
    await session.prompt('read the notes')
    notes.write_text('edited by the user\n')
    await session.prompt('and now?')
    # Each prompt makes two requests: one that reads, one after the read that answers.
    assert notices(session.messages) == ([[], [], [CHANGED], []] if enabled else [[], [], [], []])


@pytest.mark.moves_cache_prefix(reason=REQUEST_SCOPED_PROMPT_DROPPED)
async def test_a_file_edited_during_a_run_is_named_until_read_again(tmp_path: Path) -> None:
    notes = tmp_path / 'notes.txt'
    notes.write_text('first draft\n')
    read = {0: DeltaToolCall(name='read_file', json_args=json.dumps({'path': 'notes.txt'}))}

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
        answered = sum(isinstance(message, ModelResponse) for message in messages)
        if answered == 1:
            # The user edits the file while the model works on something else.
            await anyio.Path(notes).write_text('edited by the user\n')
            yield {0: DeltaToolCall(name='list_files', json_args='{}')}
        elif answered in (0, 2):
            yield read
        else:
            yield 'done'

    session = Session(
        create_stock_agent(FunctionModel(stream_function=stream)),
        deps=None,
        workspace=tmp_path,
        plugins=coder_capabilities(report_stale_files=True),
    )
    await session.prompt('read the notes')
    assert notices(session.messages) == [[], [], [CHANGED], []]


@pytest.mark.moves_cache_prefix(reason=REQUEST_SCOPED_PROMPT_DROPPED)
async def test_reads_inside_run_code_are_recorded(tmp_path: Path) -> None:
    """With code mode, `read_file` called from `run_code` is recorded like a direct call."""
    pytest.importorskip('pydantic_monty')
    from pydantic_ai_pax.runtime.speculation import SpeculationCounters
    from pydantic_ai_pax.runtime.speculative_mode import speculative_capabilities

    notes = tmp_path / 'notes.txt'
    notes.write_text('first draft\n')
    capabilities = coder_capabilities(report_stale_files=True)
    model = reads_then_answers(ToolCallPart('run_code', {'code': 'await read_file(path="notes.txt")'}))
    session = Session(
        create_stock_agent(model),
        deps=None,
        workspace=tmp_path,
        plugins=[*capabilities, *speculative_capabilities(SpeculationCounters(), capabilities, speculate=False)],
    )
    await session.prompt('read the notes')
    notes.write_text('edited by the user\n')
    await session.prompt('and now?')
    assert notices(session.messages) == [[], [], [CHANGED], []]


def test_the_menu_saves_the_setting() -> None:
    saved: list[dict[str, JsonValue]] = []
    host = PluginHost[object](
        name='coder', console=Console(file=io.StringIO()), settings={}, save_settings=saved.append
    )
    source = CoderSource(host)
    row = next(row for row in source.rows() if row.key == 'report_stale_files')
    assert (row.label, source.current(row)) == ('Stale file notices', 'false')
    assert source.apply(row, 'true') == 'Saved Stale file notices.'
    assert saved == [{'report_stale_files': True}]
    assert source.current(row) == 'true'
