"""Tests for `RequireVerification`, driven through `Agent(..., capabilities=[RequireVerification(...)])`."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from pathlib import Path

import anyio
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from pydantic_ai import Agent, DeferredToolRequests, RunContext
from pydantic_ai.capabilities import AbstractCapability, Hooks, LocalWorkspace
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.tools import Tool
from pydantic_ai_harness import (
    DEFAULT_REQUIRE_VERIFICATION_INSTRUCTIONS,
    FileSystem,
    RefineOutput,
    RequireVerification,
    Shell,
)
from pydantic_ai_harness.require_verification import CompletionUnverifiedEvent, VerificationRequiredPart

_SHELL = {'code_arg_language': 'shell', 'code_arg_name': 'command'}

Turn = ToolCallPart | str


def _scripted(*turns: Turn) -> FunctionModel:
    """Answers with each turn in order, then `'done'`."""
    remaining = list(turns)

    def next_turn() -> Turn:
        return remaining.pop(0) if remaining else 'done'

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        turn = next_turn()
        return ModelResponse(parts=[TextPart(turn) if isinstance(turn, str) else turn])

    # An event listener streams the run.
    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        turn = next_turn()
        if isinstance(turn, str):
            yield turn
        else:
            yield {0: DeltaToolCall(name=turn.tool_name, json_args=turn.args_as_json_str())}

    return FunctionModel(model, stream_function=stream)


def _write(path: str = 'app.py') -> ToolCallPart:
    return ToolCallPart('write_file', {'path': path, 'content': 'x = 1\n'})


def _run(command: str) -> ToolCallPart:
    return ToolCallPart('run_command', {'command': command})


def _agent(
    tmp_path: Path,
    *turns: Turn,
    capability: RequireVerification[None] | None = None,
    outputs: Callable[[str], object] = lambda command: '(no output)',
    extra: Sequence[AbstractCapability[None]] = (),
) -> Agent[None, str]:
    """An agent with real file tools and a `run_command` that returns `outputs(command)` instead of running it."""

    def run_command(command: str) -> object:
        return outputs(command)

    return Agent[None, str](
        _scripted(*turns),
        deps_type=type(None),
        tools=[Tool(run_command, metadata=_SHELL)],
        capabilities=[LocalWorkspace(tmp_path), FileSystem(), capability or RequireVerification(), *extra],
    )


def _redirects(messages: Sequence[ModelMessage]) -> list[tuple[str, int]]:
    return [
        (str(part.content), part.attempt)
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, VerificationRequiredPart)
    ]


def _attempts(messages: Sequence[ModelMessage]) -> list[int]:
    return [attempt for _, attempt in _redirects(messages)]


def _unverified() -> tuple[list[CompletionUnverifiedEvent], Hooks[None]]:
    events: list[CompletionUnverifiedEvent] = []
    hooks = Hooks[None]()

    @hooks.on.event(CompletionUnverifiedEvent)
    async def record(ctx: RunContext[None], event: CompletionUnverifiedEvent) -> None:
        events.append(event)

    return events, hooks


class TestRequireVerification:
    async def test_sends_an_unverified_edit_back_until_a_check_passes(self, tmp_path: Path) -> None:
        agent = _agent(tmp_path, _write(), 'Done.', _run('pytest -q'), 'Verified.')

        result = await agent.run('Fix the bug')

        assert result.output == 'Verified.'
        assert (tmp_path / 'app.py').read_text() == 'x = 1\n'
        assert _redirects(result.new_messages()) == [(DEFAULT_REQUIRE_VERIFICATION_INSTRUCTIONS, 1)]

    async def test_a_passing_check_after_the_last_edit_ends_the_run(self, tmp_path: Path) -> None:
        agent = _agent(tmp_path, _write(), _run('pytest -q'), 'Done.')

        result = await agent.run('Fix the bug')

        assert result.output == 'Done.'
        assert _redirects(result.new_messages()) == []

    async def test_a_run_without_edits_is_not_sent_back(self, tmp_path: Path) -> None:
        (tmp_path / 'app.py').write_text('x = 1\n')
        agent = _agent(tmp_path, ToolCallPart('read_file', {'path': 'app.py'}), 'Nothing to change.')

        result = await agent.run('Look around')

        assert result.output == 'Nothing to change.'

    @pytest.mark.parametrize('path', ['README.md', 'docs/guide.mdx', 'docs/index.rst'])
    async def test_edits_to_exempt_paths_need_no_verification(self, tmp_path: Path, path: str) -> None:
        result = await _agent(tmp_path, _write(path), 'Done.').run('Update the docs')

        assert result.output == 'Done.'

    async def test_custom_exempt_paths(self, tmp_path: Path) -> None:
        capability = RequireVerification[None](exempt_paths=['notes/*'])
        result = await _agent(tmp_path, _write('README.md'), 'Done.', capability=capability).run('Update the docs')

        assert _attempts(result.new_messages()) == [1, 2]

    async def test_an_edit_after_a_passing_check_makes_it_stale(self, tmp_path: Path) -> None:
        edit = ToolCallPart('edit_file', {'path': 'app.py', 'old_text': '1', 'new_text': '2'})
        agent = _agent(tmp_path, _write(), _run('pytest -q'), edit, 'Done.', _run('pytest -q'), 'Verified.')

        result = await agent.run('Fix the bug')

        assert result.output == 'Verified.'
        assert len(_redirects(result.new_messages())) == 1

    @pytest.mark.parametrize(
        'command',
        [
            'ruff format .',
            'ruff check --fix',
            'npx prettier --write src',
            "sed -i 's/1/2/' app.py",
            "sed -E -i 's/1/2/' app.py",
            'echo 2 > app.py',
            'echo 2 >> app.py',
            'git checkout -- src/',
            'git stash pop',
            'mv app.py main.py',
            'rm app.py',
            'bash -c "sed -i s/1/2/ app.py"',
            "sh -c 'echo 2 > app.py'",
        ],
    )
    async def test_an_edit_command_makes_checks_stale(self, tmp_path: Path, command: str) -> None:
        agent = _agent(tmp_path, _write(), _run('pytest -q'), _run(command), 'Done.', _run('pytest -q'), 'Verified.')

        result = await agent.run('Fix the bug')

        assert result.output == 'Verified.'
        assert len(_redirects(result.new_messages())) == 1

    async def test_an_edit_command_alone_needs_verification(self, tmp_path: Path) -> None:
        result = await _agent(tmp_path, _run('ruff format .'), 'Done.').run('Format the code')

        assert _attempts(result.new_messages()) == [1, 2]

    @pytest.mark.parametrize(
        'command',
        [
            'command -v rg >/dev/null',
            'ls > /dev/null 2>&1',
            'python -c "def f() -> int: return 1"',
            "git log --format='%h > %s'",
        ],
    )
    async def test_read_only_commands_are_not_edits(self, tmp_path: Path, command: str) -> None:
        result = await _agent(tmp_path, _run(command), 'Done.').run('Look around')

        assert result.output == 'Done.'

    @pytest.mark.parametrize(
        'command',
        [
            'pytest -q',
            'uv run pytest tests/test_app.py',
            'python -m pytest',
            'CI=1 pytest',
            'env CI=1 uv run pyright',
            'cd backend && pytest',
            'ruff check . && pyright',
            'pnpm run test',
            'npx tsc --noEmit',
            'cargo test',
            'go vet ./...',
            'make -C backend typecheck',
            './gradlew build',
            'uv run pytest 2>&1',
            'pytest -q > /dev/null 2>&1',
            "pytest -k 'parse|render'",
            'cd pkg; uv run pytest',
            'export CI=1\npytest -q\n',
        ],
    )
    async def test_recognizes_verification_commands(self, tmp_path: Path, command: str) -> None:
        result = await _agent(tmp_path, _write(), _run(command), 'Done.').run('Fix the bug')

        assert _redirects(result.new_messages()) == []

    @pytest.mark.parametrize(
        'command',
        [
            'echo pytest',
            'git status',
            'pytest || true',
            'pytest; true',
            'pytest; echo done',
            'exit 0; pytest',
            'cd pkg || exit; pytest',
            'pytest | tail -5',
            'pytest &',
            'pytest $(cat args)',
            'pytest `cat args`',
            ';',
            'echo "ok && pytest"',
            'git commit -m "fix parser; pytest passes"',
        ],
    )
    async def test_commands_that_can_hide_a_failure_do_not_count(self, tmp_path: Path, command: str) -> None:
        result = await _agent(tmp_path, _write(), _run(command), 'Done.').run('Fix the bug')

        assert _attempts(result.new_messages()) == [1, 2]

    @pytest.mark.parametrize(
        'output',
        [
            'FAILED test_app.py\n[exit code: 1]',
            '[stdout]\npartial\n[Command timed out after 30s]',
            '[stdout]\nwrote\n[exit code: 153]\n[File-size limit exceeded: max_file_bytes=10. Reduce file output before retrying.]',
            '[exit code: 1]\n[Command failed with max_file_bytes=10; file writes may have reached the per-file limit.]',
            'FAILED test_app.py\n[exited with code 1]',
            'partial\n[output truncated]\n[terminated by signal SIGKILL]',
            'PID: 7 (supervisor)\nOutput: /tmp/o\nStatus: /tmp/s\n{"pid": 7, "exit_code": 1}',
        ],
    )
    async def test_a_failed_check_is_not_evidence(self, tmp_path: Path, output: str) -> None:
        agent = _agent(tmp_path, _write(), _run('pytest -q'), 'Done.', outputs=lambda command: output)

        result = await agent.run('Fix the bug')

        assert _attempts(result.new_messages()) == [1, 2]

    @pytest.mark.parametrize(
        'output',
        [
            'PID: 7 (supervisor)\nOutput: /tmp/o\nStatus: /tmp/s',
            'PID: 7 (supervisor)\nOutput: /tmp/o\nStatus: /tmp/s\n{"pid": 7, "exit_code": null}',
            '{not json',
            "Started background command: 'pytest'\nID: 1",
            {'ok': True},
        ],
    )
    async def test_a_check_that_has_not_finished_is_not_evidence(self, tmp_path: Path, output: object) -> None:
        agent = _agent(tmp_path, _write(), _run('pytest -q'), 'Done.', outputs=lambda command: output)

        result = await agent.run('Fix the bug')

        assert _attempts(result.new_messages()) == [1, 2]

    async def test_the_latest_run_of_each_check_counts(self, tmp_path: Path) -> None:
        outputs = iter(['[exit code: 1]', '(no output)', '(no output)', '[exit code: 1]', '(no output)'])
        agent = _agent(
            tmp_path,
            _write(),
            _run('pytest -q'),
            _run('pytest -q'),
            _run('ruff check .'),
            _run('pyright'),
            'Done.',
            _run('pyright'),
            'Verified.',
            outputs=lambda command: next(outputs),
        )

        result = await agent.run('Fix the bug')

        # The failed `pytest` passed when rerun; `pyright` failed until it too was rerun.
        assert result.output == 'Verified.'
        assert len(_redirects(result.new_messages())) == 1

    async def test_a_rerun_still_running_replaces_an_earlier_pass(self, tmp_path: Path) -> None:
        outputs = iter(['(no output)', 'PID: 7 (supervisor)\nOutput: /tmp/o\nStatus: /tmp/s'])
        agent = _agent(
            tmp_path,
            _write(),
            _run('pytest -q'),
            _run('pytest -q'),
            'Done.',
            capability=RequireVerification(max_attempts=1),
            outputs=lambda command: next(outputs),
        )

        result = await agent.run('Fix the bug')

        assert _attempts(result.new_messages()) == [1]

    async def test_a_check_raced_by_an_edit_in_the_same_response_is_stale(self, tmp_path: Path) -> None:
        edited = anyio.Event()

        async def run_command(command: str) -> str:
            if command == 'pytest -q':
                await edited.wait()
                await anyio.wait_all_tasks_blocked()
            else:
                edited.set()
            return '(no output)'

        turns = iter(
            [
                [_write()],
                [_run('pytest -q'), _run('ruff format .')],
                [TextPart('Done.')],
            ]
        )

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=next(turns, [TextPart('done')]))

        agent = Agent[None, str](
            FunctionModel(model),
            deps_type=type(None),
            tools=[Tool(run_command, metadata=_SHELL)],
            capabilities=[LocalWorkspace(tmp_path), FileSystem(), RequireVerification(max_attempts=1)],
        )

        result = await agent.run('Fix the bug')

        # `pytest` started before `ruff format` changed the code, so its pass is stale.
        assert _attempts(result.new_messages()) == [1]

    async def test_the_unverified_end_is_reported_once_when_another_capability_continues_the_run(
        self, tmp_path: Path
    ) -> None:
        events, hooks = _unverified()
        agent = _agent(
            tmp_path,
            _write(),
            'Done.',
            'Revised.',
            capability=RequireVerification(max_attempts=0),
            extra=[hooks, RefineOutput()],
        )

        result = await agent.run('Fix the bug')

        assert result.output == 'Revised.'
        assert len(events) == 1

    async def test_a_piped_check_is_retried_unpiped_after_the_instructions(self, tmp_path: Path) -> None:
        agent = _agent(
            tmp_path, _write(), _run('uv run pytest 2>&1 | tail -30'), 'Done.', _run('uv run pytest'), 'Verified.'
        )

        result = await agent.run('Fix the bug')

        assert result.output == 'Verified.'
        [(instructions, _)] = _redirects(result.new_messages())
        assert 'not piped' in instructions

    async def test_attempts_are_bounded_and_the_unverified_end_is_reported(self, tmp_path: Path) -> None:
        events, hooks = _unverified()
        agent = _agent(
            tmp_path,
            _write(),
            _run('pytest -q'),
            'Done.',
            'Still done.',
            'Really done.',
            outputs=lambda command: '[exit code: 1]',
            extra=[hooks],
        )

        result = await agent.run('Fix the bug')

        assert result.output == 'Really done.'
        assert _attempts(result.new_messages()) == [1, 2]
        assert [(event.attempts, event.failed_commands) for event in events] == [(2, ['pytest -q'])]

    async def test_no_attempts_reports_the_unverified_end_at_once(self, tmp_path: Path) -> None:
        events, hooks = _unverified()
        agent = _agent(tmp_path, _write(), 'Done.', capability=RequireVerification(max_attempts=0), extra=[hooks])

        result = await agent.run('Fix the bug')

        assert result.output == 'Done.'
        assert _redirects(result.new_messages()) == []
        assert [(event.attempts, event.failed_commands) for event in events] == [(0, [])]

    async def test_custom_instructions_and_patterns(self, tmp_path: Path) -> None:
        capability = RequireVerification[None](
            instructions='Run `just ci`.',
            verification_commands=[r'just\s+ci\b'],
            edit_commands=[r'\bcodegen\b'],
        )
        agent = _agent(
            tmp_path, _run('codegen'), 'Done.', _run('pytest'), 'Done.', _run('just ci'), capability=capability
        )

        result = await agent.run('Regenerate')

        assert result.output == 'done'
        assert _redirects(result.new_messages()) == [('Run `just ci`.', 1), ('Run `just ci`.', 2)]

    async def test_each_run_starts_with_no_edits(self, tmp_path: Path) -> None:
        agent = _agent(tmp_path, _write(), 'Done.', 'Done.', 'Done.', 'Nothing to change.')
        first = await agent.run('Fix the bug')

        second = await agent.run('Anything else?', message_history=first.all_messages())

        assert second.output == 'Nothing to change.'
        assert _redirects(second.new_messages()) == []

    async def test_a_run_paused_for_deferred_tools_is_not_sent_back(self, tmp_path: Path) -> None:
        agent = Agent[None, str | DeferredToolRequests](
            _scripted(_write(), ToolCallPart('deploy', {})),
            deps_type=type(None),
            output_type=[str, DeferredToolRequests],
            capabilities=[LocalWorkspace(tmp_path), FileSystem(), RequireVerification()],
        )

        @agent.tool_plain(requires_approval=True)
        def deploy() -> str:
            raise AssertionError('the call is never approved')  # pragma: no cover

        result = await agent.run('Fix and deploy')

        assert isinstance(result.output, DeferredToolRequests)
        assert _redirects(result.new_messages()) == []

    async def test_shell_tools_without_a_command_argument_are_ignored(self, tmp_path: Path) -> None:
        def odd_shell(command: int) -> str:
            return 'ran'

        agent = Agent[None, str](
            _scripted(_write(), ToolCallPart('odd_shell', {'command': 7}), 'Done.'),
            deps_type=type(None),
            tools=[Tool(odd_shell, metadata=_SHELL)],
            capabilities=[LocalWorkspace(tmp_path), FileSystem(), RequireVerification()],
        )

        result = await agent.run('Fix the bug')

        assert _attempts(result.new_messages()) == [1, 2]

    async def test_reads_the_results_of_the_shell_tools(self, tmp_path: Path) -> None:
        capability = RequireVerification[None](verification_commands=[r'true\b', r'false\b'])
        agent = Agent[None, str](
            _scripted(
                _write(),
                _run('false'),
                'Done.',
                _write(),
                ToolCallPart('shell', {'command': 'true'}),
                'Verified.',
            ),
            deps_type=type(None),
            capabilities=[
                LocalWorkspace(tmp_path),
                FileSystem(),
                Shell(denied_commands=[], tools=['run_command', 'shell']),
                capability,
            ],
        )

        result = await agent.run('Fix the bug')

        # `run_command`'s `false` failed; after the next edit, the `shell` tool's `true` passed.
        assert result.output == 'Verified.'
        assert _attempts(result.new_messages()) == [1]

    async def test_records_spans(self, tmp_path: Path) -> None:
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        agent = _agent(tmp_path, _write(), 'Done.', 'Done.', capability=RequireVerification(max_attempts=1))
        agent.instrument = InstrumentationSettings(tracer_provider=provider)

        await agent.run('Fix the bug')

        spans = [
            (span.name, dict(span.attributes or {}))
            for span in exporter.get_finished_spans()
            if span.name in ('verification required', 'completion unverified')
        ]
        assert spans == [
            ('verification required', {'require_verification.attempt': 1}),
            ('completion unverified', {'require_verification.attempts': 1}),
        ]

    async def test_repetitive_commands_are_neither_edits_nor_checks(self, tmp_path: Path) -> None:
        # Inputs that made unanchored patterns rescan every suffix; kept small, as regression cases.
        commands = ['echo ' + 'sed ' * 200, 'echo ' + 'sed -sed ' * 100, 'make ' + 'x ' * 400, 'echo ' + '1' * 400]
        agent = _agent(tmp_path, *[_run(command) for command in commands], 'Done.')

        result = await agent.run('Look around')

        assert result.output == 'Done.'

    def test_rejects_a_negative_attempt_limit(self) -> None:
        with pytest.raises(ValueError, match='`max_attempts` must be at least 0, got -1'):
            RequireVerification[None](max_attempts=-1)

    def test_rejects_an_invalid_pattern(self) -> None:
        with pytest.raises(UserError, match='RequireVerification has an invalid command pattern'):
            RequireVerification[None](edit_commands=['('])
