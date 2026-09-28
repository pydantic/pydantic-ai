"""Tests for the Macroscope capability and toolset."""

from __future__ import annotations

import json
import re
import shlex
from collections.abc import Sequence
from pathlib import Path

import pytest

from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import ModelRetry, ToolFailed, UserError
from pydantic_ai.messages import ToolReturnPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from pydantic_ai.workspaces import (
    CommandResult,
    LocalWorkspaceBackend,
    ReadOnlyWorkspace,
    Workspace,
    WorkspaceCommand,
    WorkspaceError,
)
from pydantic_ai_harness.macroscope import (
    Macroscope,
    MacroscopeReview,
    MacroscopeToolset,
    parse_macroscope_stream,
)

_ISSUE_LINE = (
    'issue_event={"issue_id":"i1","sequence":1,"path":"a.py","line":4,'
    '"severity":"medium","category":"REVIEW_TYPE_CORRECTNESS","body":"only checks completion"}'
)


def _fake_cli(directory: Path, lines: Sequence[str], *, name: str = 'macroscope', sleep: float | None = None) -> str:
    """Write an executable stand-in for `macroscope` that records argv and cwd and emits `lines` on stderr."""
    script = directory / name
    body = [
        '#!/bin/sh',
        'printf \'%s\\n\' "$@" > "$0.args"',
        'pwd -P > "$0.cwd"',
        "printf 'macroscope starting\\n'",
    ]
    if sleep is not None:
        body.append(f'sleep {sleep}')
    body += [f"printf '%s\\n' {shlex.quote(line)} >&2" for line in lines]
    script.write_text('\n'.join(body) + '\n')
    script.chmod(0o755)
    return str(script)


def _recorded_args(command: str) -> list[str]:
    """Return the argv the fake CLI was invoked with (without the leading program name)."""
    return Path(f'{command}.args').read_text(encoding='utf-8').split()


def _toolset(command: str, *, base: str | None = 'main', timeout: float = 30.0) -> MacroscopeToolset[None]:
    return MacroscopeToolset[None](command=command, base=base, timeout=timeout)


def _ctx(workspace: Workspace | Path) -> RunContext[None]:
    """A run context for calling the tool directly, in a workspace or a local one at a directory."""
    if isinstance(workspace, Path):
        workspace = Workspace(LocalWorkspaceBackend(workspace))
    return RunContext[None](
        deps=None, model=TestModel(), usage=RunUsage(), prompt=None, messages=[], run_step=0, workspace=workspace
    )


class _FailingWorkspace(LocalWorkspaceBackend):
    """A workspace whose commands fail with a plain `WorkspaceError`."""

    async def run(self, command: WorkspaceCommand, **kwargs: object) -> CommandResult:
        raise WorkspaceError('sandbox refused the command')


class TestParseStream:
    def test_parses_review_issue_and_status(self) -> None:
        review = parse_macroscope_stream(['review_id=rev-1', _ISSUE_LINE, 'issue_status=completed'])
        assert review.review_id == 'rev-1'
        assert review.status == 'completed'
        assert len(review.issues) == 1
        issue = review.issues[0]
        assert (issue.issue_id, issue.path, issue.line, issue.severity) == ('i1', 'a.py', 4, 'medium')

    def test_skips_malformed_and_incomplete_issues(self) -> None:
        review = parse_macroscope_stream(
            [
                'review_id=rev-1',
                'issue_event={not json',
                'issue_event={"issue_id":"x"}',  # missing required fields
                _ISSUE_LINE,
                'issue_status=completed',
            ]
        )
        assert [i.issue_id for i in review.issues] == ['i1']

    def test_markers_tolerate_log_prefixes(self) -> None:
        review = parse_macroscope_stream(['2026-07-10 INFO review_id=rev-9 started', 'ts issue_status=failed now'])
        assert review.review_id == 'rev-9'
        assert review.status == 'failed'

    def test_missing_review_id_and_status_default(self) -> None:
        review = parse_macroscope_stream([_ISSUE_LINE])
        assert review.review_id is None
        assert review.status == 'unknown'
        assert len(review.issues) == 1

    def test_empty_marker_tokens_are_ignored(self) -> None:
        review = parse_macroscope_stream(['review_id=', 'issue_status=', 'unrelated line'])
        assert review.review_id is None
        assert review.status == 'unknown'

    def test_marker_text_inside_issue_body_is_not_misparsed(self) -> None:
        # A finding body can contain the literal marker strings (review findings quote code).
        # Matching `issue_event=` first keeps them from being read as a status/review_id line;
        # this guards the branch ordering in `parse_macroscope_stream`.
        body = 'the code sets issue_status=completed early and logs review_id=leaked'
        issue = 'issue_event=' + json.dumps(
            {
                'issue_id': 'i2',
                'sequence': 2,
                'path': 'b.py',
                'line': 9,
                'severity': 'high',
                'category': 'REVIEW_TYPE_CORRECTNESS',
                'body': body,
            }
        )
        review = parse_macroscope_stream([issue])
        assert [i.issue_id for i in review.issues] == ['i2']
        assert review.issues[0].body == body
        assert review.review_id is None  # not pulled out of the body
        assert review.status == 'unknown'  # not pulled out of the body


class TestRunReview:
    async def test_returns_findings(self, tmp_path: Path) -> None:
        command = _fake_cli(tmp_path, ['review_id=rev-1', _ISSUE_LINE, 'issue_status=completed'])
        review = await _toolset(command).run_macroscope_review(_ctx(tmp_path))
        assert isinstance(review, MacroscopeReview)
        assert review.review_id == 'rev-1'
        assert review.status == 'completed'
        assert len(review.issues) == 1
        assert _recorded_args(command) == ['codereview', '--raw', '--base', 'main']

    async def test_clean_review_has_no_issues(self, tmp_path: Path) -> None:
        command = _fake_cli(tmp_path, ['review_id=rev-2', 'issue_status=completed'])
        review = await _toolset(command).run_macroscope_review(_ctx(tmp_path))
        assert review.issues == []

    async def test_per_call_base_overrides_configured_base(self, tmp_path: Path) -> None:
        command = _fake_cli(tmp_path, ['review_id=rev-3', 'issue_status=completed'])
        await _toolset(command, base='develop').run_macroscope_review(_ctx(tmp_path), base='release')
        assert _recorded_args(command) == ['codereview', '--raw', '--base', 'release']

    async def test_runs_in_the_workspace_working_directory(self, tmp_path: Path) -> None:
        repo = tmp_path / 'repo'
        repo.mkdir()
        command = _fake_cli(tmp_path, ['review_id=rev-8', 'issue_status=completed'])
        await _toolset(command).run_macroscope_review(_ctx(repo))
        assert Path(f'{command}.cwd').read_text(encoding='utf-8').strip() == str(repo.resolve())

    async def test_missing_binary_raises_user_error(self, tmp_path: Path) -> None:
        # The model cannot install the CLI, so this is a setup error for the operator, not a retry.
        toolset = _toolset('pai-harness-macroscope-absent')
        with pytest.raises(UserError, match='not found'):
            await toolset.run_macroscope_review(_ctx(tmp_path))

    async def test_no_review_id_raises_user_error(self, tmp_path: Path) -> None:
        # With no model-chosen `base`, a review that never starts is a sign-in/setup problem.
        command = _fake_cli(tmp_path, ['issue_status=failed'])
        with pytest.raises(UserError, match='did not start'):
            await _toolset(command).run_macroscope_review(_ctx(tmp_path))

    async def test_no_review_id_with_model_base_raises_model_retry(self, tmp_path: Path) -> None:
        # A `base` the model passed may be the cause (e.g. a ref that does not exist), so the
        # model gets a retry to drop or change it; a retry without it then surfaces setup errors.
        command = _fake_cli(tmp_path, ['issue_status=failed'])
        with pytest.raises(ModelRetry, match="base='no-such-ref'"):
            await _toolset(command).run_macroscope_review(_ctx(tmp_path), base='no-such-ref')

    async def test_failed_status_with_review_id_is_returned_not_raised(self, tmp_path: Path) -> None:
        # A review that started (has a review_id) but ended `failed` is a real outcome the
        # model should see -- not an error. Setup errors are classified only when the id is missing.
        command = _fake_cli(tmp_path, ['review_id=rev-7', 'issue_status=failed'])
        review = await _toolset(command).run_macroscope_review(_ctx(tmp_path))
        assert review.review_id == 'rev-7'
        assert review.status == 'failed'

    async def test_workspace_timeout_raises_model_retry(self, tmp_path: Path) -> None:
        # sleep(30) far exceeds the 0.2s timeout, so a review that ignored the timeout would
        # complete successfully instead of raising.
        command = _fake_cli(tmp_path, ['review_id=rev-4', 'issue_status=completed'], sleep=30)
        with pytest.raises(ModelRetry, match='timed out'):
            await _toolset(command, timeout=0.2).run_macroscope_review(_ctx(tmp_path))

    async def test_base_omitted_lets_cli_autodetect(self, tmp_path: Path) -> None:
        # With no configured or per-call base, `--base` is dropped so the CLI picks the base itself.
        command = _fake_cli(tmp_path, ['review_id=rev-5', 'issue_status=completed'])
        await _toolset(command, base=None).run_macroscope_review(_ctx(tmp_path))
        assert _recorded_args(command) == ['codereview', '--raw']

    async def test_exec_failure_with_model_base_raises_user_error(self, tmp_path: Path) -> None:
        # The binary exists but cannot run (bad interpreter): `env`'s diagnostic, which names the binary,
        # reaches the user even when the model supplied a base.
        script = tmp_path / 'macroscope'
        script.write_text('#!/nonexistent/interpreter\n')
        script.chmod(0o755)
        with pytest.raises(UserError, match=f'could not be launched(?s:.*)CLI output:\\n.*{re.escape(str(script))}'):
            await _toolset(str(script)).run_macroscope_review(_ctx(tmp_path), base='model-ref')

    async def test_workspace_failure_fails_the_call(self, tmp_path: Path) -> None:
        with pytest.raises(ToolFailed, match='sandbox refused'):
            await _toolset('macroscope').run_macroscope_review(_ctx(Workspace(_FailingWorkspace(tmp_path))))

    async def test_no_tool_on_read_only_workspace(self, tmp_path: Path) -> None:
        read_only = _ctx(ReadOnlyWorkspace(Workspace(LocalWorkspaceBackend(tmp_path))))
        assert await _toolset('macroscope').get_tools(read_only) == {}
        assert list(await _toolset('macroscope').get_tools(_ctx(tmp_path))) == ['run_macroscope_review']


class TestCapability:
    def test_default_instructions_mention_validation(self) -> None:
        instructions = Macroscope().get_instructions()
        assert instructions is not None
        assert 'run_macroscope_review' in instructions
        assert 'untrusted' in instructions

    def test_custom_guidance_replaces_default(self) -> None:
        assert Macroscope(guidance='Review before merging.').get_instructions() == 'Review before merging.'

    def test_empty_guidance_disables_instructions(self) -> None:
        assert Macroscope(guidance='').get_instructions() is None

    def test_agent_spec_roundtrip(self) -> None:
        # The docs promise Macroscope loads from an agent spec via `custom_capability_types`.
        cap = Macroscope.from_spec(base='release', timeout=900.0)
        assert isinstance(cap, Macroscope)
        assert (cap.base, cap.timeout) == ('release', 900.0)
        agent = Agent.from_spec(
            {'model': 'test', 'capabilities': [{'Macroscope': {'base': 'main'}}]},
            custom_capability_types=[Macroscope],
        )
        loaded = [c for c in agent.root_capability.capabilities if isinstance(c, Macroscope)]
        assert len(loaded) == 1
        assert loaded[0].base == 'main'

    async def test_tool_runs_through_agent(self, tmp_path: Path) -> None:
        command = _fake_cli(tmp_path, ['review_id=rev-9', _ISSUE_LINE, 'issue_status=completed'])
        agent = Agent(TestModel(), capabilities=[Macroscope(command=command, base='main')])
        result = await agent.run('review please', workspace=LocalWorkspaceBackend(tmp_path))
        returns = [
            part
            for message in result.all_messages()
            for part in message.parts
            if isinstance(part, ToolReturnPart) and part.tool_name == 'run_macroscope_review'
        ]
        assert len(returns) == 1
        review = returns[0].content
        assert isinstance(review, MacroscopeReview)
        assert review.review_id == 'rev-9'
        assert [i.issue_id for i in review.issues] == ['i1']

    async def test_missing_binary_surfaces_from_run_without_retries(self, tmp_path: Path) -> None:
        # Through a real run, the setup error ends the run on the first call instead of being
        # fed back to the model until tool retries are exhausted.
        agent = Agent(
            TestModel(),
            capabilities=[Macroscope(command='pai-harness-macroscope-absent', base='main')],
        )
        with pytest.raises(UserError, match='not found on PATH'):
            await agent.run('review please', workspace=LocalWorkspaceBackend(tmp_path))

    async def test_not_signed_in_surfaces_from_run_without_retries(self, tmp_path: Path) -> None:
        command = _fake_cli(tmp_path, ['error: not signed in'])
        agent = Agent(TestModel(), capabilities=[Macroscope(command=command, base='main')])
        with pytest.raises(UserError, match='not signed in'):
            await agent.run('review please', workspace=LocalWorkspaceBackend(tmp_path))

    async def test_no_workspace_fails_the_run(self) -> None:
        agent = Agent(TestModel(), capabilities=[Macroscope()])
        with pytest.raises(UserError, match='`Macroscope` needs a workspace'):
            await agent.run('review please')
