import importlib
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

import pytest

import pydantic_ai.capabilities.local_workspace
import pydantic_ai_harness.coder
from pydantic_ai import Agent
from pydantic_ai.capabilities import Capability, LocalWorkspace
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.workspaces import LocalWorkspaceBackend, ReadOnlyWorkspace, Workspace, WorkspaceRef
from pydantic_ai_harness.coder import FILE_TOOL_NAMES, Coder, coder_agent
from pydantic_ai_harness.compaction import FallbackCompaction, compact_now, estimate_token_count
from pydantic_ai_harness.filesystem import FileSystem
from pydantic_ai_harness.shell import Shell
from pydantic_ai_harness.subagents import SubAgents


def test_coder_agent_is_model_less_and_composed() -> None:
    assert isinstance(coder_agent, Agent)
    assert coder_agent.model is None
    assert coder_agent.name == 'coder'


async def test_bundled_coder_agent_supplies_import_time_working_directory() -> None:
    result = await coder_agent.run('go', model=TestModel(call_tools=[], custom_output_text='done'))

    assert result.output == 'done'
    assert await result.workspace.working_dir() == Path.cwd().resolve().as_posix()


async def test_bundled_coder_agent_continues_its_local_ref_from_history() -> None:
    first = await coder_agent.run('go', model=TestModel(call_tools=[], custom_output_text='done'))
    assert first.workspace.ref == WorkspaceRef(provider='local', id=str(Path.cwd()))

    second = await coder_agent.run(
        'again', model=TestModel(call_tools=[], custom_output_text='done'), message_history=first.all_messages()
    )

    assert second.workspace.ref == first.workspace.ref


async def test_bundled_coder_agent_declines_a_local_ref_for_another_directory(tmp_path: Path) -> None:
    with pytest.raises(UserError, match="none of the agent's workspace capabilities recognized it"):
        await coder_agent.run(
            'go',
            model=TestModel(call_tools=[], custom_output_text='done'),
            workspace=WorkspaceRef(provider='local', id=str(tmp_path)),
        )


async def test_bundled_coder_agent_preserves_explicit_workspace_identity(tmp_path: Path) -> None:
    backend = LocalWorkspaceBackend(working_dir=tmp_path)
    workspace = ReadOnlyWorkspace(Workspace(backend))
    result = await coder_agent.run('go', model=TestModel(call_tools=[], custom_output_text='done'), workspace=workspace)

    assert result.workspace is workspace


def _posix_only(*_: object, **__: object) -> NoReturn:
    raise NotImplementedError('LocalWorkspaceBackend requires a POSIX host.')


async def test_coder_agent_imports_on_a_non_posix_host(monkeypatch: pytest.MonkeyPatch) -> None:
    # `LocalWorkspace` refuses a non-POSIX host when constructed; the bundled agent must still import there.
    import pydantic_ai_harness.coder._agent as module

    monkeypatch.setattr(pydantic_ai.capabilities.local_workspace, 'LocalWorkspaceBackend', _posix_only)
    try:
        agent = importlib.reload(module).coder_agent
    finally:
        monkeypatch.undo()
        importlib.reload(module)
    assert not any(isinstance(item, LocalWorkspace) for item in agent.root_capability.capabilities)
    with pytest.raises(UserError, match='`Coder` needs a workspace'):
        await agent.run('go', model=TestModel(call_tools=[]))


def test_coder_agent_export_is_lazy() -> None:
    result = subprocess.run(
        [
            sys.executable,
            '-c',
            'import sys; import pydantic_ai_harness.coder; '
            "assert 'pydantic_ai_harness.coder._agent' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_coder_unknown_export() -> None:
    with pytest.raises(AttributeError, match='has no attribute'):
        pydantic_ai_harness.coder.__getattr__('missing')


def test_coder_members_and_parameters() -> None:
    coder = Coder(instructions='Custom instructions')
    assert [type(capability).__name__ for capability in coder.capabilities] == [
        'RequireWorkspace',
        'Capability',
        'FileSystem',
        'Shell',
        'RepoContext',
        'SubAgents',
        'FallbackCompaction',
        '_BoundToolOutputs',
        'RepairToolArguments',
    ]
    files = next(item for item in coder.capabilities if isinstance(item, FileSystem))
    assert (files.root_dir, files.content_hashes, files.tools) == (None, False, FILE_TOOL_NAMES)
    shell = next(item for item in coder.capabilities if isinstance(item, Shell))
    assert (shell.tools, shell.denied_commands, shell.allow_interactive) == (['shell'], [], True)
    assert (shell.env, shell.denied_env_patterns) == (None, [])
    guidance = next(item for item in coder.capabilities if isinstance(item, Capability))
    instructions = str(guidance.get_instructions())
    for text in ('Custom instructions', 'DRY', 'YAGNI', 'SOLID', 'Zen of Python'):
        assert text in instructions
    delegation = next(item for item in coder.capabilities if isinstance(item, SubAgents))
    assert (delegation.include_self, delegation.agents, delegation.agent_folders) == (True, (), None)
    limits = next(item for item in coder.capabilities if type(item).__name__ == '_BoundToolOutputs')
    assert limits.id == 'coder_tool_output_limits'
    assert isinstance(coder.for_agent(Agent(TestModel())), Coder)


def _history(tokens: int) -> list[ModelMessage]:
    """Forty exchanges of about `tokens` in all, so compaction can keep a tail close to its budget."""
    text = 'x' * (tokens * 4 // 40)  # four characters a token
    return [
        message
        for turn in range(40)
        for message in (ModelRequest(parts=[UserPromptPart(text)]), ModelResponse(parts=[TextPart(f'reply {turn}')]))
    ]


@pytest.mark.parametrize('window', [200_000, 32_000])
async def test_coder_compaction_summarizes_with_the_run_model(tmp_path: Path, window: int) -> None:
    """Over 85% of the window, the run's own model summarizes all but a tail that fits a small window too."""
    requests: list[int] = []
    summaries = 0

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal summaries
        if any('<messages>' in str(getattr(part, 'content', '')) for message in messages for part in message.parts):
            summaries += 1
            return ModelResponse(parts=[TextPart('summary')])
        requests.append(estimate_token_count(messages))
        return ModelResponse(parts=[TextPart('done')])

    model = FunctionModel(respond, profile={'context_window': window})
    agent = Agent(model, capabilities=[LocalWorkspace(tmp_path), Coder(repo_context=False, sub_agents=False)])
    await agent.run('go', message_history=_history(int(window * 0.9)))

    assert summaries == 1
    # The kept tail is bounded by the window too, so even a 32,000-token one gets back under the trigger.
    assert requests[0] < window * 0.85


async def test_coder_compaction_truncates_when_the_model_cannot_summarize() -> None:
    compaction = next(item for item in Coder().capabilities if isinstance(item, FallbackCompaction))

    def never_called(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise AssertionError('a model that cannot write text is never asked for a summary')  # pragma: no cover

    textless = FunctionModel(never_called, profile={'supports_text_output': False})

    compacted = await compact_now(compaction, _history(180_000), model=textless)

    assert isinstance(compacted[-1], ModelResponse) and compacted[-1].parts == [TextPart('reply 39')]
    assert estimate_token_count(compacted) < 60_000  # 50,000 tokens kept, give or take a message
    assert not any(
        'summary' in str(getattr(part, 'content', '')).lower() for message in compacted for part in message.parts
    )


def test_coder_compaction_can_be_left_out() -> None:
    assert any(isinstance(item, FallbackCompaction) for item in Coder().capabilities)
    assert not any(isinstance(item, FallbackCompaction) for item in Coder(compaction=False).capabilities)


async def test_no_workspace_fails_the_run_naming_coder() -> None:
    with pytest.raises(UserError, match='`Coder` needs a workspace') as raised:
        await Agent(TestModel(), capabilities=[Coder()]).run('go')
    assert 'from pydantic_ai.capabilities import LocalWorkspace' in str(raised.value)
    assert '- LocalWorkspace: .' in str(raised.value)
