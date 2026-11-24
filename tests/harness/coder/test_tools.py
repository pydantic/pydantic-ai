"""Coder's tool surface as the model sees it; the tools themselves are tested with `FileSystem` and `Shell`."""

import warnings
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai._instructions import LoadedInstructionValues, _recorded_instruction_values, instruction_window
from pydantic_ai.capabilities import (
    AbstractCapability,
    Caching,
    Capability,
    LocalWorkspace,
    ModelFallback,
    ValidatedToolArgs,
)
from pydantic_ai.exceptions import ModelAPIError, UserError
from pydantic_ai.messages import (
    InstructionBaselinePart,
    InstructionDeltaPart,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.instrumented import InstrumentedModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.profiles.openai import openai_model_profile
from pydantic_ai.settings import CacheSetting, ModelSettings
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.workspaces import LocalWorkspaceBackend, ReadOnlyWorkspace, Workspace, WorkspaceBackend
from pydantic_ai_harness.coder import Autonomy, BenchmarkInstructions, Coder
from pydantic_ai_harness.coder._capability import MAX_FILE_TOOL_RETRIES
from pydantic_ai_harness.coder._instructions import coder_instructions
from pydantic_ai_harness.repo_context import RepoContext
from pydantic_ai_harness.sub_agents import TaskControl
from pydantic_ai_harness.tool_output_limits import ToolOutputLimits

from ...workspace_fakes import FakeWorkspace, FilesystemOnlyWorkspaceBackend
from .._recording_durability import RecordingDurability
from .._tool_calls import call_tool, call_tools

_PLAN = 'Propose a plan and wait for approval before editing.'
_DELEGATION_RULE = (
    "Don't spawn sub-agents with `delegate_task` unless the user or AGENTS.md asks, "
    'whatever other instructions say about delegating.'
)
_FINISH = 'You are running a delegated task. Choose sensible defaults and finish.'
_MISREAD_TEXT = (
    'If inputs seem inconsistent or goals impossible, assume you misread. '
    'Re-read, and try other interpretations before settling.'
)
_MISREAD_MAX_TEXT = (
    'If inputs seem inconsistent or a criterion impossible (no feasible plan, matching item, or reachable target), '
    'assume you misread. Re-read the task and inputs, and try interpretations meeting every stated criterion '
    'before settling for a partial answer. A result that leaves a stated criterion unmet means your '
    'interpretation is likely wrong, not that the answer is "none".'
)

_MISREAD_CLASSIFY_TEXT = (
    'If inputs look inconsistent, first check whether the task or its rules classify them '
    '(for example as malformed, conflicting, or rejected) and apply that rule. '
    'Only when no rule covers them, assume you misread: re-read, and try other interpretations before settling.'
)
_HANDOFF_TEXT = (
    'When asked for notes or a handoff, use headed sections: each decision and why, '
    'the exact verification commands with their observed results, the remaining risk, '
    'and what was out of scope or not done.'
)
_RUNTIME_CONSTRAINTS_TEXT = (
    'When a task states constraints on where or how the code will run '
    '(a restricted or sandboxed interpreter, no dependencies, a specific Python version, pure functions, no I/O), '
    "find out what's actually allowed before writing code. Then prefer the simplest constructs that fit, "
    'and if possible check the code under those constraints.'
)
_MAX_GUIDANCE_FLAGS = ('reinterprets_malformed_inputs', 'omits_handoff_evidence', 'ignores_stated_runtime_constraints')


@dataclass
class _Sent:
    """What a scripted model was sent: its instructions, and the system prompts in the request's last message."""

    model: str
    instructions: str | None
    system: list[str]


def _scripted(name: str, sent: list[_Sent], *, delegates: bool, fails: bool = False) -> FunctionModel:
    """A model that logs what it's sent, then fails or answers with its name.

    `delegates` sets every flag `Coder` reads, as `openai_model_profile` does for GPT models.
    """

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        last = messages[-1]
        assert isinstance(last, ModelRequest)
        system = [part.content for part in last.parts if isinstance(part, SystemPromptPart)]
        sent.append(_Sent(name, info.instructions, system))
        if fails:
            raise ModelAPIError(model_name=name, message='down')
        return ModelResponse(parts=[TextPart(name)])

    profile: ModelProfile = {
        'supports_inline_system_prompts': True,
        'delegates_unprompted': delegates,
        'reinterprets_malformed_inputs': delegates,
        'omits_handoff_evidence': delegates,
        'ignores_stated_runtime_constraints': delegates,
    }
    return FunctionModel(respond, model_name=name, profile=profile)


@dataclass
class _ToolLog(AbstractCapability[object]):
    """A host capability bound next to `Coder`, recording every tool call it sees."""

    seen: list[str] = field(default_factory=list[str])

    async def before_tool_execute(
        self, ctx: RunContext[object], *, call: ToolCallPart, tool_def: ToolDefinition, args: ValidatedToolArgs
    ) -> ValidatedToolArgs:
        self.seen.append(call.tool_name)
        return args


def _finish_or_delegate(seen: dict[str, str], messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Record the guidance the run was given, and require it to be a static instruction."""
    first = messages[0].parts[-1]
    seen.setdefault(str(first.content if isinstance(first, UserPromptPart) else None), info.instructions or '')
    guidance = next(
        part
        for part in info.model_request_parameters.instruction_parts or []
        if 'software engineering agent' in part.content
    )
    assert guidance.dynamic is False
    return _delegate_a_command(messages, info)


def _delegate_a_command(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """The parent run delegates, the delegate runs one command, and both then finish."""
    first = messages[0].parts[-1]
    prompt = first.content if isinstance(first, UserPromptPart) else None
    if len(messages) > 1:
        return ModelResponse(parts=[TextPart('done')])
    if prompt == 'Delegate it':
        return ModelResponse(parts=[ToolCallPart('delegate_task', {'agent_name': 'self', 'task': 'Run it'})])
    return ModelResponse(parts=[ToolCallPart('shell', {'command': 'echo delegated'})])


async def call(
    tmp_path: Path,
    name: str,
    arguments: dict[str, object],
    *,
    capabilities: Sequence[AbstractCapability[None]] = (),
    unrestricted_filesystem: bool = False,
) -> str:
    coder = Coder[None](unrestricted_filesystem=unrestricted_filesystem)
    return await call_tool(
        [coder, *capabilities], name, arguments, workspace=LocalWorkspaceBackend(working_dir=tmp_path)
    )


class TestCoder:
    def test_deprecated_workspace_is_ignored(self) -> None:
        with pytest.warns(Warning, match='workspace'):
            coder = Coder(workspace='elsewhere')
        assert coder is not None

    @pytest.mark.parametrize('extra_limits', [False, True])
    async def test_durable_binding(self, tmp_path: Path, extra_limits: bool) -> None:
        (tmp_path / '.git').mkdir()
        durability = RecordingDurability()
        capabilities: list[AbstractCapability[object]] = [
            LocalWorkspace(tmp_path),
            Coder(),
            durability,
        ]
        if extra_limits:
            capabilities.append(ToolOutputLimits())
        agent = Agent(TestModel(call_tools=[], custom_output_text='done'), name='coder', capabilities=capabilities)
        result = await agent.run('Inspect tools')
        assert result.output == 'done'
        assert [(name, getattr(args[0], 'method', None)) for name, args in durability.calls] == [
            ('coder__capability__workspace.call', 'ensure'),
            ('coder__capability__workspace.call', 'exists'),
            *[('coder__capability__workspace.call', 'stat')] * 4,
            ('coder__model.request_stream', None),
        ]

    @pytest.mark.parametrize('unrestricted_filesystem', [False, True])
    async def test_discovered_paths_can_be_read_and_edited(self, tmp_path: Path, unrestricted_filesystem: bool) -> None:
        (tmp_path / 'src').mkdir()
        target = tmp_path / 'src' / 'AGENTS.md'
        target.write_text('Project instructions')
        coder = Coder[None](unrestricted_filesystem=unrestricted_filesystem, repo_context=False)
        workspace = LocalWorkspaceBackend(working_dir=tmp_path)
        path = await call_tool([coder], 'list_files', {'glob': '**/AGENTS.md'}, workspace=workspace)
        assert Path(path) == Path('src/AGENTS.md')
        assert 'Project instructions' in await call_tool([coder], 'read_file', {'path': path}, workspace=workspace)
        await call_tool(
            [coder], 'edit_file', {'path': path, 'old_text': 'Project', 'new_text': 'Updated'}, workspace=workspace
        )
        assert target.read_text() == 'Updated instructions'

    async def test_schema(self, tmp_path: Path) -> None:
        model = TestModel(call_tools=[])
        await Agent(model, capabilities=[Coder()]).run(
            'Inspect tools', workspace=LocalWorkspaceBackend(working_dir=tmp_path)
        )
        assert model.last_model_request_parameters is not None
        tools = {tool.name: tool for tool in model.last_model_request_parameters.function_tools}
        assert list(tools) == ['read_file', 'write_file', 'edit_file', 'list_files', 'grep', 'shell', 'delegate_task']
        assert 'expected_hash' not in str(tools)
        assert 'replacements' in tools['edit_file'].parameters_json_schema['properties']
        assert tools['shell'].parameters_json_schema['properties']['mode']['enum'] == ['foreground', 'background']

    async def test_sub_agents_can_be_left_out(self, tmp_path: Path) -> None:
        model = TestModel(call_tools=[])
        await Agent(model, capabilities=[LocalWorkspace(tmp_path), Coder(sub_agents=False)]).run('Inspect tools')
        assert model.last_model_request_parameters is not None
        tools = [tool.name for tool in model.last_model_request_parameters.function_tools]
        assert tools == ['read_file', 'write_file', 'edit_file', 'list_files', 'grep', 'shell']

    async def test_task_control_adds_the_task_tools(self, tmp_path: Path) -> None:
        model = TestModel(call_tools=[])
        await Agent(model, capabilities=[LocalWorkspace(tmp_path), Coder(task_control=TaskControl())]).run(
            'Inspect tools'
        )
        assert model.last_model_request_parameters is not None
        tools = [tool.name for tool in model.last_model_request_parameters.function_tools]
        assert tools[-6:] == ['delegate_task', 'list_tasks', 'check_task', 'message_task', 'answer_task', 'cancel_task']
        with pytest.raises(ValueError, match='`task_control` requires `sub_agents=True`'):
            Coder(sub_agents=False, task_control=TaskControl())

    async def test_the_delegate_carries_capabilities_bound_next_to_coder(self, tmp_path: Path) -> None:
        """The delegate is the agent `Coder` is bound to, so a host hook sees the commands it runs."""
        log = _ToolLog()
        capabilities: list[AbstractCapability[object]] = [LocalWorkspace(tmp_path), Coder(repo_context=False), log]
        agent = Agent(FunctionModel(_delegate_a_command), capabilities=capabilities)
        assert (await agent.run('Delegate it')).output == 'done'
        assert log.seen == ['delegate_task', 'shell']

    async def test_passing_coder_to_run_is_refused(self, tmp_path: Path) -> None:
        agent = Agent(TestModel(call_tools=[]), capabilities=[LocalWorkspace(tmp_path)])
        with pytest.raises(UserError, match='only carries what is bound to the `Agent`'):
            await agent.run('Inspect tools', capabilities=[Coder()])
        result = await agent.run('Inspect tools', capabilities=[Coder(sub_agents=False)])
        assert result.output == 'success (no tool calls)'

    @pytest.mark.parametrize(
        ('caching', 'before', 'after', 'model_settings', 'agent_settings', 'run_settings', 'expected'),
        [
            pytest.param(True, [], [], None, None, None, True, id='default'),
            pytest.param(False, [], [], None, None, None, None, id='left-out'),
            pytest.param(True, [], [], {'cache': False}, None, None, False, id='model-disables'),
            pytest.param(True, [], [], None, {'cache': False}, None, False, id='agent-disables'),
            pytest.param(True, [], [], None, {'cache': '1h'}, None, '1h', id='agent-configures'),
            pytest.param(True, [Caching('1h')], [], None, None, None, '1h', id='caching-before'),
            pytest.param(True, [], [Caching(False)], None, None, None, False, id='caching-after'),
            pytest.param(True, [], [], None, None, {'cache': False}, False, id='run-disables'),
        ],
    )
    async def test_caching(
        self,
        tmp_path: Path,
        caching: bool,
        before: list[AbstractCapability[object]],
        after: list[AbstractCapability[object]],
        model_settings: ModelSettings | None,
        agent_settings: ModelSettings | None,
        run_settings: ModelSettings | None,
        expected: CacheSetting | None,
    ) -> None:
        """Prompt caching is on unless the model, the agent, a capability, or the run configures `cache`."""
        seen: list[CacheSetting | None] = []

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen.append(info.model_request_parameters.cache)
            return ModelResponse(parts=[TextPart('done')])

        model = FunctionModel(
            respond,
            profile=ModelProfile(supports_cache=True, supported_cache_retentions=('5m', '1h')),
            settings=model_settings,
        )
        coder = Coder[object](repo_context=False, caching=caching)
        agent = Agent(
            model,
            model_settings=agent_settings,
            capabilities=[LocalWorkspace(tmp_path), *before, coder, *after],
        )
        await agent.run('go', model_settings=run_settings)
        assert seen == [expected]

    @pytest.mark.parametrize('extra_instructions', [None, '', 'Keep new files under 400 lines.'])
    async def test_instructions(self, tmp_path: Path, extra_instructions: str | None) -> None:
        model = TestModel(call_tools=[])
        coder = Coder(instructions=extra_instructions, repo_context=False, sub_agents=False)
        await Agent(model, capabilities=[coder]).run('Inspect instructions', workspace=LocalWorkspaceBackend(tmp_path))
        assert model.last_model_request_parameters is not None
        parts = model.last_model_request_parameters.instruction_parts or []
        guidance_part = next(part for part in parts if 'software engineering agent' in part.content)
        assert guidance_part.dynamic is False
        guidance = guidance_part.content
        default = guidance.removesuffix('\n' + extra_instructions) if extra_instructions else guidance
        assert len(default.split()) < 220
        assert "Match the surrounding code's style, structure, and imports." in default
        assert 'Work until done. Ask only for credentials, approvals, or real ambiguity.' in default
        assert 'Read files with `read_file`, not batched `cat`.' in default
        assert "Write checks from the task's stated behavior, not your code." in default
        assert 'go through each requirement and edge case the task states' in default
        assert 'Measure results against any numeric target the task states.' in default
        assert 'assume you misread' in default
        assert 'Check behavior with heredocs for one-offs, not new files.' in default
        assert 'Servers the task needs may keep running; stop the rest.' in default
        assert 'delete your scratch files' not in default
        if extra_instructions:
            assert guidance.endswith('\n' + extra_instructions)

    @pytest.mark.parametrize(
        ('autonomy', 'ask', 'targets', 'cleanup'),
        [
            ('low', 'Propose a plan and wait for approval before editing.', False, False),
            ('medium', 'Work until done; ask when a choice really matters.', False, False),
            ('high', 'Ask only for credentials, approvals, or real ambiguity.', True, False),
            ('max', 'Work fully autonomously; nobody will answer questions.', True, True),
        ],
    )
    async def test_autonomy_sets_how_far_the_agent_goes_alone(
        self, tmp_path: Path, autonomy: Autonomy, ask: str, targets: bool, cleanup: bool
    ) -> None:
        model = TestModel(call_tools=[])
        coder = Coder(autonomy=autonomy, repo_context=False, sub_agents=False)
        await Agent(model, capabilities=[coder]).run('go', workspace=LocalWorkspaceBackend(tmp_path))
        assert model.last_model_request_parameters is not None
        text = '\n'.join(part.content for part in model.last_model_request_parameters.instruction_parts or [])
        assert ask in text
        assert ('Measure results against any numeric target the task states.' in text) is targets
        assert ('Leave only the requested change; delete your scratch files.' in text) is cleanup
        assert 'go through each requirement and edge case the task states' in text
        assert "Write checks from the task's stated behavior, not your code." in text
        assert (_MISREAD_MAX_TEXT in text) is cleanup
        assert (_MISREAD_TEXT in text) is not cleanup

    @pytest.mark.parametrize('autonomy', ['low', 'medium', 'high', 'max'])
    @pytest.mark.parametrize('benchmark', [False, True])
    async def test_benchmark_instructions_are_opt_in(self, tmp_path: Path, autonomy: Autonomy, benchmark: bool) -> None:
        """The clean-directory and every-field sentences come only from `BenchmarkInstructions`, at every level."""
        model = TestModel(call_tools=[])
        coder = Coder(autonomy=autonomy, repo_context=False, sub_agents=False)
        capabilities = [coder, BenchmarkInstructions()] if benchmark else [coder]
        await Agent(model, capabilities=capabilities).run('go', workspace=LocalWorkspaceBackend(tmp_path))
        assert model.last_model_request_parameters is not None
        parts = model.last_model_request_parameters.instruction_parts or []
        texts = [part.content for part in parts if not part.dynamic]
        assert sum('Deliverables may be used outside this workspace' in text for text in texts) == benchmark
        assert sum('uses every field the provided data supplies' in text for text in texts) == benchmark
        assert 'go through each requirement and edge case the task states' in texts[0]

    async def test_a_delegate_gets_benchmark_instructions(self, tmp_path: Path) -> None:
        seen: dict[str, str] = {}

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return _finish_or_delegate(seen, messages, info)

        agent = Agent(
            FunctionModel(respond),
            capabilities=[LocalWorkspace(tmp_path), Coder(repo_context=False), BenchmarkInstructions()],
        )
        assert (await agent.run('Delegate it')).output == 'done'
        assert 'from a clean directory' in seen['Delegate it']
        assert 'from a clean directory' in seen['Run it']

    @pytest.mark.parametrize('autonomy', ['low', 'medium', 'high', 'max'])
    def test_max_expands_the_misread_check_for_parent_and_delegate(self, autonomy: Autonomy) -> None:
        """A delegate follows its parent's autonomy for the misread check, as it does for cleanup."""
        for text in coder_instructions(autonomy=autonomy):
            if autonomy == 'max':
                assert _MISREAD_MAX_TEXT in text
                assert _MISREAD_TEXT not in text
            else:
                assert _MISREAD_TEXT in text
                assert _MISREAD_MAX_TEXT not in text

    async def test_a_delegate_is_told_to_finish_on_its_own(self, tmp_path: Path) -> None:
        """A `'low'` delegate without `ask_parent` would propose a plan to nobody, so it is told to finish instead."""
        seen: dict[str, str] = {}

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return _finish_or_delegate(seen, messages, info)

        agent = Agent(
            FunctionModel(respond), capabilities=[LocalWorkspace(tmp_path), Coder(autonomy='low', repo_context=False)]
        )
        assert (await agent.run('Delegate it')).output == 'done'
        assert _PLAN in seen['Delegate it'] and _FINISH not in seen['Delegate it']
        assert 'Measure results against any numeric target the task states.' not in seen['Delegate it']
        assert _FINISH in seen['Run it'] and _PLAN not in seen['Run it']
        assert 'Measure results against any numeric target the task states.' in seen['Run it']

    async def test_a_durable_delegate_is_told_to_finish_on_its_own(self, tmp_path: Path) -> None:
        """A durable worker does not inherit the caller's context variables, and the delegate text still applies.

        The child run is awaited only after `delegation_depth()` is set, and a recorded delegation sets
        that level again on the worker before the child starts. An isolated operation body sees a fresh
        context, which is where a worker would lose a context variable read too late.
        """
        seen: dict[str, str] = {}

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return _finish_or_delegate(seen, messages, info)

        agent = Agent(
            FunctionModel(respond),
            name='coder',
            capabilities=[
                LocalWorkspace(tmp_path),
                Coder(autonomy='low', repo_context=False),
                RecordingDurability(isolated=True),
            ],
        )
        assert (await agent.run('Delegate it')).output == 'done'
        assert _FINISH in seen['Run it'] and _PLAN not in seen['Run it']

    async def test_a_delegate_with_task_control_is_also_told_to_finish(self, tmp_path: Path) -> None:
        """Whether a delegate gets `ask_parent` depends on the workspace and sub-agent, so every delegate finishes alone.

        Task control streams the child, so the child's guidance is read from the stream function.
        """
        seen: dict[str, str] = {}

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return _finish_or_delegate(seen, messages, info)

        async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
            _finish_or_delegate(seen, messages, info)
            yield 'done'

        agent = Agent(
            FunctionModel(respond, stream_function=stream),
            capabilities=[
                LocalWorkspace(tmp_path),
                Coder(autonomy='low', repo_context=False, task_control=TaskControl()),
            ],
        )
        assert (await agent.run('Delegate it')).output == 'done'
        assert _PLAN in seen['Delegate it'] and _FINISH not in seen['Delegate it']
        assert _FINISH in seen['Run it'] and _PLAN not in seen['Run it']

    def test_unknown_autonomy_is_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"`autonomy` must be one of 'low', 'medium', 'high', 'max', not 'full'"):
            Coder(autonomy='full')  # pyright: ignore[reportArgumentType]

    async def test_file_tool_mistakes_do_not_end_the_run(self, tmp_path: Path) -> None:
        """Consecutive denied writes come back as retries until `MAX_FILE_TOOL_RETRIES` is spent."""
        denied: tuple[str, dict[str, object]] = ('write_file', {'path': '/elsewhere/probe.py', 'content': 'x'})
        coder = Coder[None](repo_context=False, sub_agents=False)
        workspace = LocalWorkspaceBackend(working_dir=tmp_path)
        results = await call_tools([coder], [denied] * MAX_FILE_TOOL_RETRIES, workspace=workspace)
        assert len(results) == MAX_FILE_TOOL_RETRIES
        assert all('the file tools only work inside it' in result for result in results)

    async def test_exhausted_tool_retries_come_back_to_the_model(self, tmp_path: Path) -> None:
        """Once a tool's retries are spent, its error is a failed tool result and the run continues."""
        denied: tuple[str, dict[str, object]] = ('write_file', {'path': '/elsewhere/probe.py', 'content': 'x'})
        allowed: tuple[str, dict[str, object]] = ('write_file', {'path': 'probe.py', 'content': 'x'})
        coder = Coder[None](repo_context=False, sub_agents=False)
        workspace = LocalWorkspaceBackend(working_dir=tmp_path)
        calls = [denied] * (MAX_FILE_TOOL_RETRIES + 2) + [allowed]
        results = await call_tools([coder], calls, workspace=workspace)
        assert len(results) == len(calls)
        assert all('the file tools only work inside it' in result for result in results[:-1])
        assert not any('failed several times in a row' in result for result in results[:MAX_FILE_TOOL_RETRIES])
        assert all('failed several times in a row' in result for result in results[MAX_FILE_TOOL_RETRIES:-1])
        assert (tmp_path / 'probe.py').read_text() == 'x'

    async def test_other_tool_errors_still_end_the_run(self, tmp_path: Path) -> None:
        """Only an exhausted retry comes back to the model; any other tool error still ends the run."""

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[ToolCallPart('boom', {})])

        agent = Agent(
            FunctionModel(respond),
            deps_type=type(None),
            capabilities=[Coder[None](repo_context=False, sub_agents=False)],
        )

        @agent.tool_plain
        def boom() -> str:
            raise RuntimeError('tool crashed')

        with pytest.raises(RuntimeError, match='tool crashed'):
            await agent.run('Use the tools', workspace=LocalWorkspaceBackend(working_dir=tmp_path))

    @pytest.mark.parametrize('repo_context', [False, True])
    @pytest.mark.parametrize('sub_agents', [False, True])
    @pytest.mark.parametrize('extra_instructions', [None, 'Keep new files under 400 lines.'])
    @pytest.mark.parametrize('delegates_unprompted', [False, True])
    async def test_instructions_follow_the_composition(
        self,
        tmp_path: Path,
        repo_context: bool,
        sub_agents: bool,
        extra_instructions: str | None,
        delegates_unprompted: bool,
    ) -> None:
        """The delegation rule comes with `delegate_task`, for models that delegate unprompted, and extra guidance is
        appended, with or without `RepoContext`."""
        model = TestModel(call_tools=[], profile={'delegates_unprompted': delegates_unprompted})
        coder = Coder(instructions=extra_instructions, repo_context=repo_context, sub_agents=sub_agents)
        await Agent(model, capabilities=[coder]).run('go', workspace=LocalWorkspaceBackend(tmp_path))
        assert model.last_model_request_parameters is not None
        parts = model.last_model_request_parameters.instruction_parts or []
        text = '\n'.join(part.content for part in parts)
        guidance = next(part.content for part in parts if 'software engineering agent' in part.content)
        tools = {tool.name for tool in model.last_model_request_parameters.function_tools}
        assert (_DELEGATION_RULE in text) is (sub_agents and delegates_unprompted)
        assert ('delegate_task' in tools) is sub_agents
        if extra_instructions:
            assert guidance.endswith('\n' + extra_instructions)

    @pytest.mark.parametrize('opened_on_gpt', [True, False], ids=['gpt-to-claude', 'claude-to-gpt'])
    async def test_a_fallback_sends_the_new_model_the_instructions_it_would_have_started_with(
        self, tmp_path: Path, opened_on_gpt: bool
    ) -> None:
        """The delegation rule rewrites the prefix on a switch: after a fallback across `delegates_unprompted`, the
        new model is sent the instructions a conversation started on it gets, with no withdrawal or change appended.
        """
        first_name, second_name = ('gpt', 'claude') if opened_on_gpt else ('claude', 'gpt')
        sent: list[_Sent] = []
        workspace = LocalWorkspaceBackend(tmp_path)
        first = await Agent(
            _scripted(first_name, sent, delegates=opened_on_gpt), capabilities=[Coder(repo_context=False)]
        ).run('go', workspace=workspace)
        agent = Agent(
            _scripted(first_name, sent, delegates=opened_on_gpt, fails=True),
            capabilities=[
                Coder(repo_context=False),
                ModelFallback(_scripted(second_name, sent, delegates=not opened_on_gpt)),
            ],
        )
        result = await agent.run('continue', message_history=first.all_messages(), workspace=workspace)
        assert result.output == second_name
        fresh: list[_Sent] = []
        await Agent(
            _scripted(second_name, fresh, delegates=not opened_on_gpt), capabilities=[Coder(repo_context=False)]
        ).run('go', workspace=workspace)
        opened_with, switched_to = sent[0].instructions, fresh[0].instructions
        assert opened_with is not None and switched_to is not None
        assert (_DELEGATION_RULE in opened_with) is opened_on_gpt
        assert (_DELEGATION_RULE in switched_to) is not opened_on_gpt
        assert [(s.model, s.instructions) for s in sent] == [
            (first_name, opened_with),
            (first_name, opened_with),
            (second_name, switched_to),
        ]
        assert [s.system for s in sent] == [[], [], []]
        assert not any(
            isinstance(part, InstructionDeltaPart)
            for message in result.all_messages()
            if isinstance(message, ModelRequest)
            for part in message.parts
        )

    @pytest.mark.parametrize('delegates', [False, True])
    async def test_a_conversation_that_stays_on_one_model_keeps_its_prefix(
        self, tmp_path: Path, delegates: bool
    ) -> None:
        """Without a switch, the delegation block doesn't change, so every request is sent the same instructions."""
        sent: list[_Sent] = []
        workspace = LocalWorkspaceBackend(tmp_path)
        model = _scripted('model', sent, delegates=delegates)
        first = await Agent(model, capabilities=[Coder(repo_context=False)]).run('go', workspace=workspace)
        await Agent(model, capabilities=[Coder(repo_context=False)]).run(
            'continue', message_history=first.all_messages(), workspace=workspace
        )
        assert len(sent) == 2 and sent[0].instructions is not None
        assert (_DELEGATION_RULE in sent[0].instructions) is delegates
        assert sent[1].instructions == sent[0].instructions
        assert [s.system for s in sent] == [[], []]

    async def test_no_delegation_rule_without_sub_agents(self, tmp_path: Path) -> None:
        """The rule names `delegate_task`, so it isn't registered without `SubAgents`, even for a flagged model."""
        model = TestModel(call_tools=[], profile={'delegates_unprompted': True})
        await Agent(model, capabilities=[Coder(repo_context=False, sub_agents=False)]).run(
            'go', workspace=LocalWorkspaceBackend(tmp_path)
        )
        assert model.last_model_request_parameters is not None
        parts = model.last_model_request_parameters.instruction_parts or []
        assert _DELEGATION_RULE not in '\n'.join(part.content for part in parts)
        assert 'delegation' not in {part.name for part in parts}
        assert not any(part.dynamic for part in parts)

    @pytest.mark.parametrize('autonomy', ['high', 'max'])
    @pytest.mark.parametrize('flag', [None, *_MAX_GUIDANCE_FLAGS])
    async def test_max_guidance_follows_the_profile_flags(
        self, tmp_path: Path, autonomy: Autonomy, flag: str | None
    ) -> None:
        """Each `'max'` sentence is a named block that only its flag fills."""
        profile: ModelProfile = {flag: True} if flag is not None else {}  # pyright: ignore[reportAssignmentType]
        model = TestModel(call_tools=[], profile=profile)
        coder = Coder(autonomy=autonomy, repo_context=False, sub_agents=False)
        await Agent(model, capabilities=[coder]).run('go', workspace=LocalWorkspaceBackend(tmp_path))
        assert model.last_model_request_parameters is not None
        parts = model.last_model_request_parameters.instruction_parts or []
        blocks = {part.name: part.content for part in parts if part.name is not None}
        is_max = autonomy == 'max'
        expected = {
            'misread': _MISREAD_CLASSIFY_TEXT if is_max and flag == 'reinterprets_malformed_inputs' else None,
            'handoff': _HANDOFF_TEXT if is_max and flag == 'omits_handoff_evidence' else None,
            'runtime_constraints': _RUNTIME_CONSTRAINTS_TEXT
            if is_max and flag == 'ignores_stated_runtime_constraints'
            else None,
        }
        # An empty block isn't sent; a sent one rewrites the prefix on a switch, so it's plain text.
        assert blocks == {name: text for name, text in expected.items() if text is not None}
        assert all(part.on_change == 'rewrite' for part in parts)
        if not is_max:
            # Below `'max'` the blocks aren't registered, so the instructions stay all static.
            assert not any(part.dynamic for part in parts)
        guidance = next(part.content for part in parts if 'software engineering agent' in part.content)
        assert (_MISREAD_MAX_TEXT in guidance) is is_max
        assert (_MISREAD_TEXT in guidance) is not is_max

    async def test_max_instructions_are_unchanged_for_a_model_without_the_flags(self, tmp_path: Path) -> None:
        """A model without the flags gets exactly the `'max'` instructions it got before they existed: the blocks are
        empty, so not sent, and the default misread check stays inline in the shared guidance."""
        model = TestModel(call_tools=[])
        coder = Coder(autonomy='max', repo_context=False)
        await Agent(model, capabilities=[coder]).run('go', workspace=LocalWorkspaceBackend(tmp_path))
        assert model.last_model_request_parameters is not None
        parts = model.last_model_request_parameters.instruction_parts or []
        assert [part.content for part in parts] == snapshot(
            [
                """\
You are a software engineering agent. Match the surrounding code's style, structure, and imports.

Work fully autonomously; nobody will answer questions. Choose sensible defaults and finish.

Read files with `read_file`, not batched `cat`. Save effort on repeated work, not on understanding the task. Don't re-read files that haven't changed.

Measure results against any numeric target the task states. Before finishing, go through each requirement and edge case the task states and confirm each one explicitly. If inputs seem inconsistent or a criterion impossible (no feasible plan, matching item, or reachable target), assume you misread. Re-read the task and inputs, and try interpretations meeting every stated criterion before settling for a partial answer. A result that leaves a stated criterion unmet means your interpretation is likely wrong, not that the answer is "none". Write checks from the task's stated behavior, not your code. Check behavior with heredocs for one-offs, not new files. Leave only the requested change; delete your scratch files.

Finish long-running work before replying; poll until done or blocked. Servers the task needs may keep running; stop the rest.\
""",
                """\
You can delegate self-contained tasks to these sub-agents using the `delegate_task` tool. Each runs in its own fresh context and does not see this conversation, so pass everything it needs.

Available sub-agents:
- self: A fresh run of this same agent, with all of your tools, instructions, and capabilities. Use it to hand off a self-contained part of the work.\
""",
            ]
        )

    @pytest.mark.parametrize(('model_name', 'flagged'), [('gpt-6.2-sol', True), ('gpt-oss-120b', False)])
    async def test_max_guidance_goes_to_gpt_but_not_gpt_oss(
        self, tmp_path: Path, model_name: str, flagged: bool
    ) -> None:
        """`openai_model_profile` flags OpenAI's GPT models, not the open-weights gpt-oss."""
        model = TestModel(call_tools=[], profile=openai_model_profile(model_name))
        coder = Coder(autonomy='max', repo_context=False, sub_agents=False)
        await Agent(model, capabilities=[coder]).run('go', workspace=LocalWorkspaceBackend(tmp_path))
        assert model.last_model_request_parameters is not None
        text = '\n'.join(part.content for part in model.last_model_request_parameters.instruction_parts or [])
        assert (_MISREAD_CLASSIFY_TEXT in text) is flagged
        assert _MISREAD_MAX_TEXT in text
        assert (_HANDOFF_TEXT in text) is flagged
        assert (_RUNTIME_CONSTRAINTS_TEXT in text) is flagged

    async def test_a_delegate_gets_the_max_guidance(self, tmp_path: Path) -> None:
        seen: dict[str, str] = {}

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            first = messages[0].parts[-1]
            seen.setdefault(str(first.content if isinstance(first, UserPromptPart) else None), info.instructions or '')
            return _delegate_a_command(messages, info)

        profile: ModelProfile = {flag: True for flag in _MAX_GUIDANCE_FLAGS}  # pyright: ignore[reportAssignmentType]
        agent = Agent(
            FunctionModel(respond, profile=profile),
            capabilities=[LocalWorkspace(tmp_path), Coder(autonomy='max', repo_context=False)],
        )
        assert (await agent.run('Delegate it')).output == 'done'
        delegate = seen['Run it']
        assert _FINISH in delegate
        assert _MISREAD_CLASSIFY_TEXT in delegate
        assert _MISREAD_MAX_TEXT in delegate
        assert _HANDOFF_TEXT in delegate
        assert _RUNTIME_CONSTRAINTS_TEXT in delegate

    @pytest.mark.parametrize('sub_agents', [True, False])
    async def test_a_fallback_at_max_sends_the_new_model_its_own_max_instructions(
        self, tmp_path: Path, sub_agents: bool
    ) -> None:
        """When a `'max'` conversation that opened on GPT falls back to Claude, the GPT-only `'max'` sentences rewrite
        the prefix like the delegation rule, so they're left out of Claude's instructions, not withdrawn, also
        without `SubAgents`."""
        sent: list[_Sent] = []
        workspace = LocalWorkspaceBackend(tmp_path)
        first = await Agent(
            _scripted('gpt', sent, delegates=True),
            capabilities=[Coder(autonomy='max', repo_context=False, sub_agents=sub_agents)],
        ).run('go', workspace=workspace)
        agent = Agent(
            _scripted('gpt', sent, delegates=True, fails=True),
            capabilities=[
                Coder(autonomy='max', repo_context=False, sub_agents=sub_agents),
                ModelFallback(_scripted('claude', sent, delegates=False)),
            ],
        )
        result = await agent.run('continue', message_history=first.all_messages(), workspace=workspace)
        assert result.output == 'claude'
        opened_with = sent[0].instructions
        assert opened_with is not None
        assert _MISREAD_CLASSIFY_TEXT in opened_with
        assert _HANDOFF_TEXT in opened_with
        assert _RUNTIME_CONSTRAINTS_TEXT in opened_with
        assert _MISREAD_MAX_TEXT in opened_with
        fresh: list[_Sent] = []
        await Agent(
            _scripted('claude', fresh, delegates=False),
            capabilities=[Coder(autonomy='max', repo_context=False, sub_agents=sub_agents)],
        ).run('go', workspace=workspace)
        assert [(s.model, s.instructions) for s in sent] == [
            ('gpt', opened_with),
            ('gpt', opened_with),
            ('claude', fresh[0].instructions),
        ]
        assert [s.system for s in sent] == [[], [], []]

    @pytest.mark.parametrize('instrumented', [False, True])
    @pytest.mark.parametrize(('autonomy', 'sub_agents'), [('high', True), ('max', False)])
    async def test_fallback_model_gets_no_delegation_rule_and_a_warning(
        self, tmp_path: Path, instrumented: bool, autonomy: Autonomy, sub_agents: bool
    ) -> None:
        """`FallbackModel` has no profile of its own, so even a GPT-first chain gets the default instructions, and the
        run still works.

        `Coder` warns that `ModelFallback` would give each attempt its model's instructions, also when the
        `FallbackModel` is instrumented (the run's `ctx.model` is then still the `FallbackModel`).
        """
        sent: list[_Sent] = []
        fallback = FallbackModel(
            _scripted('gpt', sent, delegates=True, fails=True), _scripted('claude', sent, delegates=False)
        )
        model = InstrumentedModel(fallback) if instrumented else fallback
        coder = Coder(autonomy=autonomy, repo_context=False, sub_agents=sub_agents)
        with pytest.warns(UserWarning, match="`Coder` can't read model-specific instructions from `FallbackModel`"):
            result = await Agent(model, capabilities=[coder]).run('go', workspace=LocalWorkspaceBackend(tmp_path))
        assert result.output == 'claude'
        assert [s.model for s in sent] == ['gpt', 'claude']
        for s in sent:
            assert s.instructions is not None
            assert _DELEGATION_RULE not in s.instructions
            assert _MISREAD_CLASSIFY_TEXT not in s.instructions
            assert _HANDOFF_TEXT not in s.instructions
            assert (_MISREAD_MAX_TEXT in s.instructions) is (autonomy == 'max')

    async def test_fallback_model_gets_no_warning_when_no_instructions_depend_on_the_model(
        self, tmp_path: Path
    ) -> None:
        """Below `'max'` and without `SubAgents`, every model gets the same instructions, so there's nothing to warn
        about."""
        sent: list[_Sent] = []
        fallback = FallbackModel(_scripted('gpt', sent, delegates=True))
        coder = Coder(repo_context=False, sub_agents=False)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            result = await Agent(fallback, capabilities=[coder]).run('go', workspace=LocalWorkspaceBackend(tmp_path))
        assert result.output == 'gpt'

    @pytest.mark.parametrize(
        ('unrestricted', 'files'),
        [(False, 'the file tools only accept paths inside it'), (True, 'relative file paths resolve from it')],
    )
    async def test_instructions_name_the_project_once(self, tmp_path: Path, unrestricted: bool, files: str) -> None:
        model = TestModel(call_tools=[])
        coder = Coder(unrestricted_filesystem=unrestricted)
        await Agent(model, capabilities=[coder]).run('go', workspace=LocalWorkspaceBackend(tmp_path))
        assert model.last_model_request_parameters is not None
        parts = model.last_model_request_parameters.instruction_parts or []
        notes = [part.content for part in parts if 'Your project is' in part.content]
        assert notes == [f'Your project is `{tmp_path.resolve()}`. Shell commands start there, and {files}.']

    async def test_lazy_sandbox_instructions_leave_out_the_project(self, tmp_path: Path) -> None:
        # Without `RepoContext` nothing touches the workspace at run start, so neither do the instructions.
        model = TestModel(call_tools=[])
        await Agent(model, capabilities=[Coder(repo_context=False)]).run(
            'go', workspace=LocalWorkspaceBackend(tmp_path)
        )
        assert model.last_model_request_parameters is not None
        parts = model.last_model_request_parameters.instruction_parts or []
        assert not any('Your project is' in part.content for part in parts)

    async def test_a_given_repo_context_keeps_the_project_note(self, tmp_path: Path) -> None:
        model = TestModel(call_tools=[])
        coder = Coder(repo_context=RepoContext(expose_inventory_tool=False))
        await Agent(model, capabilities=[coder]).run('go', workspace=LocalWorkspaceBackend(tmp_path))
        assert model.last_model_request_parameters is not None
        parts = model.last_model_request_parameters.instruction_parts or []
        assert sum('Your project is' in part.content for part in parts) == 1

    @pytest.mark.parametrize(
        ('config', 'loaded', 'nested'),
        [
            ('bundled', True, True),
            ('walk_up_only', True, False),
            ('nested_pointer', True, False),
            ('no_autoload', False, True),
        ],
    )
    async def test_instructions_describe_what_the_repo_context_loads(
        self, tmp_path: Path, config: str, loaded: bool, nested: bool
    ) -> None:
        if config == 'bundled':
            coder = Coder()
        elif config == 'walk_up_only':
            coder = Coder(repo_context=RepoContext(nested_traversal=False))
        elif config == 'nested_pointer':
            coder = Coder(repo_context=RepoContext(nested_inject='pointer'))
        else:
            coder = Coder(repo_context=RepoContext(autoload_instructions=False))
        model = TestModel(call_tools=[])
        await Agent(model, capabilities=[coder]).run('go', workspace=LocalWorkspaceBackend(tmp_path))
        assert model.last_model_request_parameters is not None
        text = '\n'.join(part.content for part in model.last_model_request_parameters.instruction_parts or [])
        assert ("if any, are already in your context; don't search for them" in text) is loaded
        assert ("A subdirectory's instruction files are added to your context" in text) is nested

    async def test_instructions_name_the_configured_instruction_files(self, tmp_path: Path) -> None:
        model = TestModel(call_tools=[])
        coder = Coder(repo_context=RepoContext(filenames=('RULES.md',)))
        await Agent(model, capabilities=[coder]).run('go', workspace=LocalWorkspaceBackend(tmp_path))
        assert model.last_model_request_parameters is not None
        text = '\n'.join(part.content for part in model.last_model_request_parameters.instruction_parts or [])
        assert 'Instruction files (RULES.md) for the project, if any, are already in your context' in text

    @pytest.mark.parametrize('repo_context', [True, False])
    async def test_repo_context_is_optional(self, tmp_path: Path, repo_context: bool) -> None:
        (tmp_path / 'AGENTS.md').write_text('Always answer in haiku.\n')
        model = TestModel(call_tools=[])
        await Agent(model, capabilities=[Coder(repo_context=repo_context)]).run(
            'Inspect instructions', workspace=LocalWorkspaceBackend(working_dir=tmp_path)
        )
        assert model.last_model_request_parameters is not None
        instructions = model.last_model_request_parameters.instruction_parts or []
        assert any('Always answer in haiku.' in part.content for part in instructions) is repo_context

    async def test_read_write(self, tmp_path: Path) -> None:
        assert 'hash:' not in await call(tmp_path, 'write_file', {'path': 'test.txt', 'content': 'one\ntwo\n'})
        output = await call(tmp_path, 'read_file', {'path': 'test.txt', 'offset': 1, 'limit': 1})
        assert 'two' in output and 'one' not in output and 'hash:' not in output

    async def test_long_reads_page_without_gaps(self, tmp_path: Path) -> None:
        (tmp_path / 'big.py').write_text(''.join(f'line {i} ' + 'x' * 60 + '\n' for i in range(3000)))
        output = await call(tmp_path, 'read_file', {'path': 'big.py'})
        numbers = [int(line.split('\t')[0]) for line in output.splitlines()[1:-1]]
        assert numbers == list(range(1, len(numbers) + 1)) and len(numbers) < 2000
        assert output.endswith(f'Use offset={len(numbers)} to continue reading.)\n')
        assert '[truncated' not in output

    async def test_batch_edit(self, tmp_path: Path) -> None:
        path = tmp_path / 'test.txt'
        path.write_text('one two')
        replacements = [{'old_text': 'one', 'new_text': '1'}, {'old_text': 'two', 'new_text': '2'}]
        assert (
            await call(tmp_path, 'edit_file', {'path': 'test.txt', 'replacements': replacements}) == 'Edited test.txt.'
        )
        assert path.read_text() == '1 2'

    async def test_search(self, tmp_path: Path) -> None:
        (tmp_path / 'test.py').write_text('One\ntwo\n')
        (tmp_path / 'other.txt').write_text('One\n')
        assert await call(tmp_path, 'list_files', {'glob': '*.py'}) == 'test.py'
        output = await call(tmp_path, 'grep', {'pattern': 'one', 'ignore_case': True, 'file_type': 'py'})
        assert output == 'test.py:1:One'

    @pytest.mark.parametrize('read_only', [True, False], ids=['read-only', 'filesystem-only'])
    async def test_search_without_commands(self, tmp_path: Path, read_only: bool) -> None:
        """A reviewer on a read-only or filesystem-only workspace can still list and search files."""
        (tmp_path / 'notes.txt').write_text('needle\n')
        workspace: WorkspaceBackend = (
            ReadOnlyWorkspace(Workspace(LocalWorkspaceBackend(tmp_path)))
            if read_only
            else FilesystemOnlyWorkspaceBackend(FakeWorkspace('files', {'/workspace/notes.txt': b'needle\n'}))
        )
        model = TestModel(call_tools=[])
        await Agent(model, capabilities=[Coder()]).run('Inspect', workspace=workspace)
        assert model.last_model_request_parameters is not None
        names = {tool.name for tool in model.last_model_request_parameters.function_tools}
        assert {'list_files', 'grep'} <= names
        assert 'shell' not in names
        assert await call_tool([Coder[None]()], 'grep', {'pattern': 'needle'}, workspace=workspace) == (
            'notes.txt:1:needle'
        )
        assert await call_tool([Coder[None]()], 'list_files', {}, workspace=workspace) == 'notes.txt'

    async def test_shell_is_unrestricted_and_persistent(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('OPENAI_API_KEY', 'do-not-expose')
        output = await call(tmp_path, 'shell', {'command': 'mkdir child; printf "${OPENAI_API_KEY-unset}"; exit 7'})
        assert 'unset' in output and 'do-not-expose' not in output and '"exit_code": 7' in output
        assert (tmp_path / 'child').is_dir()
        assert 'PID:' in output and 'Status:' in output
        assert 'not in the allowed list' not in await call(tmp_path, 'shell', {'command': 'sudo -n true'})

    async def test_protected_paths_still_apply(self, tmp_path: Path) -> None:
        (tmp_path / '.env').write_text('SECRET=1')
        assert 'protected' in await call(tmp_path, 'edit_file', {'path': '.env', 'old_text': '1', 'new_text': '2'})

    @pytest.mark.parametrize('relative', [False, True])
    async def test_unrestricted_filesystem_keeps_workspace_relative_paths(self, tmp_path: Path, relative: bool) -> None:
        workspace = tmp_path / 'workspace'
        workspace.mkdir()
        outside = tmp_path / '.env'
        outside.write_text('before')
        await call(
            workspace,
            'edit_file',
            {'path': '../.env' if relative else str(outside), 'old_text': 'before', 'new_text': 'after'},
            unrestricted_filesystem=True,
        )
        assert outside.read_text() == 'after'
        await call(workspace, 'write_file', {'path': 'local.txt', 'content': 'local'}, unrestricted_filesystem=True)
        assert (workspace / 'local.txt').read_text() == 'local'
        assert 'after' in await call(workspace, 'read_file', {'path': '../.env'}, unrestricted_filesystem=True)
        refused = await call(workspace, 'read_file', {'path': '../.env'})
        assert f'`{outside}` is outside the project root `{workspace.resolve()}`' in refused
        assert 'Create or clone it inside the project, or use a shell tool if you have one.' in refused


@dataclass
class _SwitchDeps:
    state: str
    count: int = 0


@dataclass
class _SwitchLog:
    """What each request sent: the model, its instructions, and every `<pyai:context>` statement in its history."""

    requests: list[tuple[str, str | None, list[str]]] = field(default_factory=list[tuple[str, str | None, list[str]]])


def _switch_agent() -> Agent[_SwitchDeps, str]:
    """A `'max'` `Coder` agent with an append-mode block of its own and a deferred capability whose text changes."""
    counter = Capability[_SwitchDeps](
        id='counter', defer_loading=True, description='Counting.', instructions='Count carefully.'
    )

    @counter.instructions
    def count(ctx: RunContext[_SwitchDeps]) -> str:
        return f'The count is {ctx.deps.count}.'

    agent = Agent(deps_type=_SwitchDeps, capabilities=[Coder[_SwitchDeps](autonomy='max', repo_context=False), counter])

    @agent.instructions(name='state', on_change='append')
    def state(ctx: RunContext[_SwitchDeps]) -> str:
        return f'The state is {ctx.deps.state}.'

    return agent


def _switch_model(name: str, log: _SwitchLog, *, gpt: bool, script: Sequence[ModelResponse] = ()) -> FunctionModel:
    """GPT sets every flag `Coder` reads; the other model none. `script` answers the first requests in order."""
    responses = list(script)

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        statements = [
            part.content
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, SystemPromptPart) and '<pyai:context' in part.content
        ]
        log.requests.append((name, info.instructions, statements))
        return responses.pop(0) if responses else ModelResponse(parts=[TextPart(name)])

    profile: ModelProfile = {
        'supports_inline_system_prompts': True,
        'delegates_unprompted': gpt,
        **{flag: gpt for flag in _MAX_GUIDANCE_FLAGS},  # pyright: ignore[reportAssignmentType]
    }
    return FunctionModel(respond, model_name=name, profile=profile)


def _instruction_records(messages: Sequence[ModelMessage], kind: type[ModelRequestPart]) -> list[ModelRequestPart]:
    return [
        part
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, kind)
    ]


def _folded(messages: list[ModelMessage]) -> tuple[dict[str, str | None], dict[str, str | None]]:
    """The prefix blocks' values as the instruction window and its changes record them, and the loaded blocks'."""
    baseline = instruction_window(messages)
    assert baseline is not None
    recorded = _recorded_instruction_values(messages, baseline)
    assert recorded is not None
    return {str(instruction_id): value for instruction_id, value in recorded.items()}, dict(
        LoadedInstructionValues.of(messages).values
    )


def _current(state: str) -> dict[str, str | None]:
    """What the prefix's append-mode blocks say: the agent's own and the deferred capability catalog. `Coder`'s
    model-specific blocks rewrite, so none of them is in the window."""
    return {
        'agent:state': f'The state is {state}.',
        'capability:deferred_capabilities': (
            'The following capabilities are deferred and can be loaded using the `load_capability` tool. '
            "A capability's tools stay hidden until it is loaded:\n- counter: Counting."
        ),
    }


_LOAD_COUNTER = ModelResponse(parts=[ToolCallPart('load_capability', {'id': 'counter'}, tool_call_id='load')])


class TestModelSwitches:
    """`Coder`'s model-specific blocks rewrite the prefix on a switch, while other blocks keep appending."""

    async def _fresh(self, tmp_path: Path, *, gpt: bool, deps: _SwitchDeps) -> str | None:
        log = _SwitchLog()
        await _switch_agent().run(
            'go', model=_switch_model('fresh', log, gpt=gpt), deps=deps, workspace=LocalWorkspaceBackend(tmp_path)
        )
        return log.requests[0][1]

    async def test_switching_back_to_gpt_sends_its_original_prefix(self, tmp_path: Path) -> None:
        """GPT → Claude → GPT: switching back opens another window, at GPT's values. With nothing else changed, GPT is
        sent the instructions the conversation opened with, byte for byte, so its cached prefix can still be hit, and
        the same instructions and statements as a GPT run on the same history without instruction records."""
        log = _SwitchLog()
        workspace = LocalWorkspaceBackend(tmp_path)
        agent, deps = _switch_agent(), _SwitchDeps('v1')
        messages: list[ModelMessage] = []
        for name, gpt in [('gpt', True), ('claude', False), ('gpt', True)]:
            result = await agent.run(
                name,
                model=_switch_model(name, log, gpt=gpt),
                deps=deps,
                workspace=workspace,
                message_history=messages,
            )
            messages = result.all_messages()
        opened_with = log.requests[0][1]
        assert opened_with is not None and _DELEGATION_RULE in opened_with
        assert [(name, instructions) for name, instructions, _ in log.requests] == [
            ('gpt', opened_with),
            ('claude', await self._fresh(tmp_path, gpt=False, deps=deps)),
            ('gpt', opened_with),
        ]
        assert [statements for *_, statements in log.requests] == [[], [], []]
        assert _instruction_records(messages, InstructionDeltaPart) == []
        assert len(_instruction_records(messages, InstructionBaselinePart)) == 3

        # A GPT run on the same history, stripped of its instruction records, is sent the same thing.
        stripped = [
            replace(message, parts=[p for p in message.parts if not isinstance(p, InstructionBaselinePart)])
            if isinstance(message, ModelRequest)
            else message
            for message in messages[:-2]
        ]
        replay = _SwitchLog()
        await agent.run(
            'gpt',
            model=_switch_model('gpt', replay, gpt=True),
            deps=deps,
            workspace=workspace,
            message_history=stripped,
        )
        assert replay.requests == log.requests[-1:]

    async def test_other_blocks_keep_appending_across_switches(self, tmp_path: Path) -> None:
        """Between switches, the agent's own block and a loaded capability's block append their changes. A switch opens
        a window that carries each prefix block at its current value, as a fresh run on the new model would, and the
        changes recorded before it stay in history without applying to the new window. A loaded capability's block
        isn't part of any window, so its changes stay in effect and it isn't stated again."""
        log = _SwitchLog()
        workspace = LocalWorkspaceBackend(tmp_path)
        agent = _switch_agent()
        steps: list[tuple[str, bool, _SwitchDeps, Sequence[ModelResponse]]] = [
            ('gpt', True, _SwitchDeps('v1', 0), [_LOAD_COUNTER]),
            ('gpt', True, _SwitchDeps('v2', 1), []),
            ('claude', False, _SwitchDeps('v2', 1), []),
            ('claude', False, _SwitchDeps('v3', 2), []),
            ('gpt', True, _SwitchDeps('v3', 2), []),
        ]
        messages: list[ModelMessage] = []
        deltas_before: list[ModelRequestPart] = []
        for index, (name, gpt, deps, script) in enumerate(steps):
            result = await agent.run(
                name,
                model=_switch_model(name, log, gpt=gpt, script=script),
                deps=deps,
                workspace=workspace,
                message_history=messages,
            )
            messages = result.all_messages()
            new_deltas = _instruction_records(result.new_messages(), InstructionDeltaPart)
            switched = index > 0 and steps[index - 1][1] != gpt
            sent = log.requests[-1][1]
            if switched:
                # The new window is what a fresh run on this model gets, and no change is appended for it.
                assert sent == await self._fresh(tmp_path, gpt=gpt, deps=deps)
                assert new_deltas == []
            elif index > 0:
                assert sent == log.requests[-2][1]
            # Earlier changes stay in history.
            assert _instruction_records(messages, InstructionDeltaPart)[: len(deltas_before)] == deltas_before
            deltas_before = _instruction_records(messages, InstructionDeltaPart)
            # The fold agrees with what the model was sent: the current values, each applied once.
            assert _folded(messages) == (
                _current(deps.state),
                {'counter': f'Count carefully.\n\nThe count is {deps.count}.'},
            )
        # What each request was sent after the prefix, by id: the loaded capability's statements are sent in every
        # window, while the state changes from an earlier window are left out once a switch states the current value.
        assert [
            (name, [statement.split('"')[1] for statement in statements]) for name, _, statements in log.requests
        ] == snapshot(
            [
                ('gpt', []),
                ('gpt', ['capability:counter']),
                ('gpt', ['capability:counter', 'agent:state', 'capability:counter']),
                ('claude', ['capability:counter', 'capability:counter']),
                ('claude', ['capability:counter', 'capability:counter', 'agent:state', 'capability:counter']),
                ('gpt', ['capability:counter', 'capability:counter', 'capability:counter']),
            ]
        )
