"""Offline tests for the Pydantic AI gh-aw shim (.github/scripts/pydantic-ai-runner).

These cover the gh-aw compatibility surface with no network or credentials:
argv tolerance, prompt recovery, model resolution, MCP-config translation and
allow-list filtering, Claude-named tools, `--allowed-tools` /
`--permission-mode` enforcement, structured-error guarantees, and the
stream-json schema.

The single live test is skipped unless an Anthropic-shape endpoint is given
via env: GH_AW_SHIM_LIVE_API_KEY / _BASE_URL / _MODEL.

Run:  uv run --with pytest pytest .github/scripts/test_pydantic_ai_runner.py
"""

import asyncio
import importlib
import inspect
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, cast

import httpcore2
import httpx2
import pytest
import yaml
from anthropic import APITimeoutError, AsyncAnthropic, RateLimitError
from opentelemetry.instrumentation.httpx import HTTPX2ClientInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import TypeAdapter
from pytest import LogCaptureFixture
from tenacity import stop_after_attempt, wait_none

# `.github/scripts/` isn't on sys.path by default — the shim package lives
# there. The runtime equivalent is the PEP-723 launcher script
# (`pydantic-ai-runner`) which inserts the same directory before
# `runpy.run_module`-ing the package.
sys.path.insert(0, str(Path(__file__).parent))

# Tool callables, shared helpers, and the CLI live in distinct submodules;
# tests import each from where it actually lives, not from a re-export.
import agentic_workflow_guard
import pydantic_ai_gh_aw_shim as pkg
from mcp.shared.exceptions import McpError
from mcp.types import ErrorData
from pydantic_ai_gh_aw_shim import (
    cli as shim,
    shared,
)

from pydantic_ai import Agent, ModelRetry
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.exceptions import ModelHTTPError, UsageLimitExceeded
from pydantic_ai.messages import (
    INTERRUPTED_TOOL_RETURN_CONTENT,
    BinaryContent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model as _Model
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.test import TestModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset, PrefixedToolset
from pydantic_ai.usage import RunUsage, UsageLimits


@pytest.mark.subprocess(reason='the staged shim must import without repository sibling modules on PYTHONPATH')
def test_shim_cli_imports_from_isolated_staged_package(tmp_path: Path):
    scripts_dir = Path(__file__).parent
    staged_package = tmp_path / 'pydantic_ai_gh_aw_shim'
    shutil.copytree(scripts_dir / 'pydantic_ai_gh_aw_shim', staged_package)
    result = subprocess.run(
        [sys.executable, '-P', '-c', 'from pydantic_ai_gh_aw_shim import cli'],
        capture_output=True,
        check=False,
        env={**os.environ, 'PYTHONPATH': str(tmp_path)},
        text=True,
    )

    assert result.returncode == 0, result.stderr


# The exact argv shape gh-aw's claude_harness.cjs passes, prompt appended last.
GHAW_ARGV = [
    '--print',
    '--no-chrome',
    '--allowed-tools',
    'Bash,Read,Edit(/tmp/*),mcp__github__get_me,mcp__safeoutputs',
    '--debug-file',
    '/tmp/gh-aw/agent-stdio.log',
    '--verbose',
    '--permission-mode',
    'bypassPermissions',
    '--output-format',
    'stream-json',
    '--mcp-config',
    '/tmp/mcp-servers.json',
    '--prompt-file',
    '/tmp/gh-aw/aw-prompts/prompt.txt',
]


# --------------------------------------------------------------------------- #
# argv / prompt
# --------------------------------------------------------------------------- #
def test_parses_full_claude_argv_without_error():
    args = shim.parse_args([*GHAW_ARGV, 'do the thing'])
    assert args.mcp_config == '/tmp/mcp-servers.json'
    assert args.prompt_file == '/tmp/gh-aw/aw-prompts/prompt.txt'
    assert args.prompt_positional == 'do the thing'
    assert args.permission_mode == 'bypassPermissions'


def test_launcher_rejects_unsupported_continuation_without_output():
    launcher = Path(__file__).with_name('pydantic-ai-runner-launch.sh')
    result = subprocess.run([launcher, '--continue'], text=True, capture_output=True, check=False)
    assert result.returncode == 1
    assert result.stdout == ''
    assert result.stderr == ''


def _run_context_prefetch(
    tmp_path: Path,
    *,
    gh_script: str | None = None,
    env_vars: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    fake_timeout = bin_dir / 'timeout'
    fake_timeout.write_text('#!/bin/sh\nshift\nexec "$@"\n', encoding='utf-8')
    fake_timeout.chmod(0o755)
    if gh_script is not None:
        fake_gh = bin_dir / 'gh'
        fake_gh.write_text(f'#!/bin/sh\n{gh_script}', encoding='utf-8')
        fake_gh.chmod(0o755)

    agent_dir = tmp_path / 'agent'
    env = os.environ.copy()
    env.pop('GH_TOKEN', None)
    env.pop('GITHUB_TOKEN', None)
    env.pop('GITHUB_REPOSITORY', None)
    env.update(
        PATH=f'{bin_dir}:{env["PATH"]}',
        GH_AW_AGENT_DIR=str(agent_dir),
    )
    env.update(env_vars or {})
    script = Path(__file__).with_name('prefetch-github-context.sh')
    result = subprocess.run(['bash', script], text=True, capture_output=True, check=False, env=env)
    return result, agent_dir


_SUCCESSFUL_GH_PREFETCH = """\
case "$1:$2" in
  issue:list) printf '%s\\n' '[{"number":1,"title":"Issue"}]' ;;
  pr:list) printf '%s\\n' '[{"number":2,"title":"PR"}]' ;;
  *) exit 8 ;;
esac
"""


@pytest.mark.parametrize(
    ('env_vars', 'expected_token'),
    [
        ({'GH_TOKEN': 'preferred-token', 'GITHUB_TOKEN': 'fallback-token'}, 'preferred-token'),
        ({'GITHUB_TOKEN': 'fallback-token'}, 'fallback-token'),
    ],
)
def test_prefetch_github_context_selects_token(tmp_path: Path, env_vars: dict[str, str], expected_token: str):
    result, _ = _run_context_prefetch(
        tmp_path,
        gh_script=f'[ "$GH_TOKEN" = "{expected_token}" ] || exit 9\n' + _SUCCESSFUL_GH_PREFETCH,
        env_vars={**env_vars, 'GITHUB_REPOSITORY': 'pydantic/pydantic-ai'},
    )

    assert result.returncode == 0
    assert 'Could not prefetch' not in result.stdout


def test_prefetch_github_context_without_credential_is_non_fatal(tmp_path: Path):
    result, agent_dir = _run_context_prefetch(tmp_path)

    assert result.returncode == 0
    assert 'No GitHub credential' in result.stdout
    assert not (agent_dir / 'github-context/open-issues.json').exists()
    assert not (agent_dir / 'github-context/open-pull-requests.json').exists()


def test_prefetch_github_context_without_repository_is_non_fatal(tmp_path: Path):
    result, agent_dir = _run_context_prefetch(tmp_path, env_vars={'GH_TOKEN': 'test-token'})

    assert result.returncode == 0
    assert 'GITHUB_REPOSITORY is unavailable' in result.stdout
    assert not (agent_dir / 'github-context/open-issues.json').exists()
    assert not (agent_dir / 'github-context/open-pull-requests.json').exists()


@pytest.mark.parametrize(
    ('failed_command', 'preserved_file', 'missing_file', 'warning'),
    [
        (
            'issue:list',
            'open-pull-requests.json',
            'open-issues.json',
            'Could not prefetch open issues',
        ),
        (
            'pr:list',
            'open-issues.json',
            'open-pull-requests.json',
            'Could not prefetch open pull requests',
        ),
    ],
)
def test_prefetch_github_context_preserves_independent_corpus(
    tmp_path: Path,
    failed_command: str,
    preserved_file: str,
    missing_file: str,
    warning: str,
):
    result, agent_dir = _run_context_prefetch(
        tmp_path,
        gh_script=f"""\
[ "$1:$2" = "{failed_command}" ] && exit 7
{_SUCCESSFUL_GH_PREFETCH}
""",
        env_vars={'GH_TOKEN': 'test-token', 'GITHUB_REPOSITORY': 'pydantic/pydantic-ai'},
    )

    context_dir = agent_dir / 'github-context'
    assert result.returncode == 0
    assert warning in result.stdout
    assert (context_dir / preserved_file).exists()
    assert not (context_dir / missing_file).exists()


@pytest.mark.parametrize('malformed_command', ['issue:list', 'pr:list'])
def test_prefetch_github_context_rejects_malformed_corpus(tmp_path: Path, malformed_command: str):
    result, agent_dir = _run_context_prefetch(
        tmp_path,
        gh_script=f"""\
if [ "$1:$2" = "{malformed_command}" ]; then
  printf '%s\\n' '{{"message":"unexpected response"}}'
  exit 0
fi
{_SUCCESSFUL_GH_PREFETCH}
""",
        env_vars={'GH_TOKEN': 'test-token', 'GITHUB_REPOSITORY': 'pydantic/pydantic-ai'},
    )

    rejected_corpus = 'open-issues.json' if malformed_command == 'issue:list' else 'open-pull-requests.json'
    preserved_corpus = 'open-pull-requests.json' if malformed_command == 'issue:list' else 'open-issues.json'
    assert result.returncode == 0
    assert 'Could not prefetch' in result.stdout
    assert not (agent_dir / 'github-context' / rejected_corpus).exists()
    assert (agent_dir / 'github-context' / preserved_corpus).exists()


def test_prefetch_github_context_rejects_capped_corpus(tmp_path: Path):
    result, agent_dir = _run_context_prefetch(
        tmp_path,
        gh_script="""\
if [ "$1:$2" = "issue:list" ]; then
  python3 -c 'import json; print(json.dumps([{}] * 1000))'
  exit 0
fi
"""
        + _SUCCESSFUL_GH_PREFETCH,
        env_vars={'GH_TOKEN': 'test-token', 'GITHUB_REPOSITORY': 'pydantic/pydantic-ai'},
    )

    context_dir = agent_dir / 'github-context'
    assert result.returncode == 0
    assert 'Could not prefetch open issues' in result.stdout
    assert not (context_dir / 'open-issues.json').exists()
    assert (context_dir / 'open-pull-requests.json').exists()


def test_shared_context_setup_is_scoped_and_uses_runtime_paths():
    shared_steps = Path(__file__).parent.parent / 'workflows' / 'shared' / 'pre-agent-steps.md'
    shared_text = shared_steps.read_text(encoding='utf-8')
    context_steps = shared_steps.with_name('issue-filing-context.md')
    context_text = context_steps.read_text(encoding='utf-8')
    tool_hints = context_steps.with_name('tool-hints.md').read_text(encoding='utf-8')
    prewarm = Path(__file__).with_name('prewarm-pydantic-ai-runner.sh').read_text(encoding='utf-8')

    assert 'run: bash .github/scripts/prewarm-pydantic-ai-runner.sh' in shared_text
    assert 'GH_TOKEN' not in shared_text
    assert 'run: bash .github/scripts/prefetch-github-context.sh' in context_text
    assert '      GH_TOKEN: ${{ github.token }}' in context_text
    assert 'prefetch-github-context.sh' not in prewarm
    assert '$GITHUB_WORKSPACE/.review-context/' in tool_hints
    assert '$GITHUB_WORKSPACE/.review-context/' in shim.INSTRUCTIONS


def test_prewarm_leaves_the_runner_lock_as_checked_out(tmp_path: Path):
    """CI Review's `git checkout --detach` of the PR head aborts if the pre-warm left the lock dirty."""
    workspace = tmp_path / 'workspace'
    lock = workspace / '.github' / 'scripts' / 'pydantic-ai-runner.lock'
    lock.parent.mkdir(parents=True)
    lock.write_text('checked out\n', encoding='utf-8')
    git = ['git', '-C', str(workspace), '-c', 'user.name=t', '-c', 'user.email=t@example.com']
    subprocess.run([*git, 'init', '-q'], check=True)
    subprocess.run([*git, 'add', '.'], check=True)
    subprocess.run([*git, 'commit', '-q', '-m', 'init'], check=True)

    # `uv sync --script` rewrites the adjacent lock when it is stale.
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    fake_uv = bin_dir / 'uv'
    fake_uv.write_text('#!/bin/sh\necho relocked > "$3.lock"\n', encoding='utf-8')
    fake_uv.chmod(0o755)

    env = {**os.environ, 'PATH': f'{bin_dir}:{os.environ["PATH"]}', 'GITHUB_WORKSPACE': str(workspace)}
    script = Path(__file__).with_name('prewarm-pydantic-ai-runner.sh')
    result = subprocess.run(['bash', script], text=True, capture_output=True, check=False, env=env)

    assert result.returncode == 0
    assert f'using uv={fake_uv}' in result.stdout
    assert lock.read_text(encoding='utf-8') == 'checked out\n'


@pytest.mark.parametrize(
    'prompt_name',
    [
        'pydantic-ai-bug-hunter.md',
        'pydantic-ai-docs-drift.md',
        'pydantic-ai-provider-mapping-sweep.md',
        'pydantic-ai-provider-parity-explore.md',
        'pydantic-ai-regression-detector.md',
        'pydantic-ai-roundtrip-sweep.md',
        'pydantic-ai-streaming-resilience-sweep.md',
    ],
)
def test_issue_filing_prompts_use_prefetched_context(prompt_name: str):
    prompt = Path(__file__).parent.parent / 'workflows' / 'shared' / 'prompts' / prompt_name
    text = prompt.read_text(encoding='utf-8')

    assert '/tmp/gh-aw/agent/github-context/open-issues.json' in text
    assert 'gh issue list' not in text
    assert 'gh pr list' not in text
    assert 'gh api --paginate' not in text

    workflow_name = prompt_name.removesuffix('.md')
    workflow = prompt.parent.parent.parent / f'{workflow_name}.md'
    assert '  - shared/issue-filing-context.md' in workflow.read_text(encoding='utf-8')


def test_unknown_future_claude_flags_are_tolerated():
    args = shim.parse_args([*GHAW_ARGV, '--some-future-flag', 'x', 'prompt'])
    assert args.prompt_positional == 'prompt'


def test_prompt_recovered_from_trailing_positional():
    args = shim.parse_args([*GHAW_ARGV, 'Investigate the failing CI run.'])
    assert shim.resolve_prompt(args) == 'Investigate the failing CI run.'


def test_prompt_falls_back_to_prompt_file(tmp_path: Path):
    pf = tmp_path / 'prompt.txt'
    pf.write_text('from file', encoding='utf-8')
    args = shim.parse_args(['--prompt-file', str(pf)])
    assert shim.resolve_prompt(args) == 'from file'


def test_prompt_falls_back_to_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pf = tmp_path / 'p.txt'
    pf.write_text('from env path', encoding='utf-8')
    monkeypatch.setenv('GH_AW_PROMPT', str(pf))
    assert shim.resolve_prompt(shim.parse_args(['--print'])) == 'from env path'


# --------------------------------------------------------------------------- #
# --allowed-tools parsing & enforcement
# --------------------------------------------------------------------------- #
def test_allowed_tools_absent_is_none():
    assert shim.parse_args(['--print']).allowed_tools is None
    assert shim._split_allowed_tools(None) is None  # pyright: ignore[reportPrivateUsage]


def test_allowed_tools_parsed_and_scope_stripped():
    args = shim.parse_args([*GHAW_ARGV, 'p'])
    assert args.allowed_tools == frozenset({'Bash', 'Read', 'Edit', 'mcp__github__get_me', 'mcp__safeoutputs'})


async def _toolset_names(
    allowed: frozenset[str] | None,
    permission_mode: str | None,
    *,
    task: shim.TaskCallable | None = None,
) -> list[str]:
    """Resolve a `select_claude_code_toolset(...)` result to its post-filter tool
    name list. The filtered toolset reports its tools through
    `.get_tools(ctx)`, so we drive it with a minimal RunContext.
    """
    toolset = shim.select_claude_code_toolset(allowed, permission_mode, task=task)
    ctx = RunContext(
        deps=None,
        model=cast(_Model[Any], None),
        usage=RunUsage(),
        prompt=None,
        messages=[],
        run_step=0,
    )
    tools = await toolset.get_tools(ctx)
    return list(tools.keys())


def test_select_claude_code_toolset_no_allowlist_keeps_all():
    names = asyncio.run(_toolset_names(None, None, task=shim.task))
    # task=shim.task adds "Task" alongside the base callables. Order is
    # insertion order from `_BASE_TOOLS` + the appended Task entry.
    assert names == [*pkg.CLAUDE_CODE_TOOL_NAMES, 'Task']


def test_select_claude_code_toolset_enforces_allowlist():
    names = asyncio.run(_toolset_names(frozenset({'Bash', 'Read', 'mcp__safeoutputs'}), None))
    assert names == ['Bash', 'Read']


def test_plan_mode_withholds_mutating_tools():
    names = set(asyncio.run(_toolset_names(None, 'plan')))
    assert names.isdisjoint(pkg.MUTATING_TOOLS)
    assert 'Read' in names and 'Grep' in names and 'Glob' in names


def test_plan_mode_and_allowlist_compose():
    names = asyncio.run(_toolset_names(frozenset({'Bash', 'Read'}), 'plan'))
    assert names == ['Read']  # Bash dropped by plan mode


def test_claude_code_tool_names():
    # `WebFetch` is wired separately via a `NativeTool(WebFetchTool())`
    # capability — it's not in the callable Claude Code tool list.
    # `Task` is registered through `build_claude_code_toolset(task=...)` and
    # so isn't part of the static `CLAUDE_CODE_TOOL_NAMES` tuple either.
    assert pkg.CLAUDE_CODE_TOOL_NAMES == (
        'Bash',
        'Read',
        'Write',
        'Edit',
        'MultiEdit',
        'Grep',
        'Glob',
        'LS',
        'TodoWrite',
        'ExitPlanMode',
    )


def test_harness_backed_tools_are_async_and_pin_the_remaining_gaps():
    """Guard the harness-backed vs hand-rolled split — the boundary of the swap.

    The harness-backed tools delegate to pydantic-ai-harness and are async:
    `Bash`/`Read`/`Write`/`Edit`/`Grep`/`Glob`/`LS` to `FileSystemToolset` /
    `ShellToolset`, and `TodoWrite` to the `planning` capability's `render_plan`.

    The remaining tools have no harness equivalent and stay sync:

    - `MultiEdit` — the harness has no atomic multi-replacement primitive
      (`edit_file` is single-unique-occurrence, one call, no batch rollback).
    - `ExitPlanMode` — a plan-mode protocol ack with no capability behind it.

    (`Task`, in the main shim, stays hand-rolled too: the experimental
    `SubAgentToolset.delegate_task` routes to *pre-named* agents, while Claude's
    `Task` spawns an ad-hoc sub-agent from a free-form prompt — a different
    interface, not just an experimental one.)

    If a future change backs one of these with the harness (or accidentally
    de-async's a backed tool), this test trips so the gap list stays honest.
    """
    fn_by_name = {
        'Bash': pkg.bash,
        'Read': pkg.read_file,
        'Write': pkg.write_file,
        'Edit': pkg.edit_file,
        'Grep': pkg.grep,
        'Glob': pkg.glob_search,
        'LS': pkg.list_dir,
        'MultiEdit': pkg.multi_edit,
        'TodoWrite': pkg.todo_write,
        'ExitPlanMode': pkg.exit_plan_mode,
    }
    harness_backed = {'Bash', 'Read', 'Write', 'Edit', 'Grep', 'Glob', 'LS', 'TodoWrite'}
    hand_rolled = {'MultiEdit', 'ExitPlanMode'}
    # Every Claude tool is accounted for in exactly one bucket.
    assert harness_backed.isdisjoint(hand_rolled)
    assert harness_backed | hand_rolled == set(pkg.CLAUDE_CODE_TOOL_NAMES) == set(fn_by_name)
    for name in harness_backed:
        assert inspect.iscoroutinefunction(fn_by_name[name]), f'{name} should be harness-backed (async)'
    for name in hand_rolled:
        assert not inspect.iscoroutinefunction(fn_by_name[name]), f'{name} has no stable harness equivalent (sync)'


# --------------------------------------------------------------------------- #
# Claude Code tool behavior
# --------------------------------------------------------------------------- #
def _ctx() -> RunContext[object]:
    """A run context on the live `GITHUB_WORKSPACE`, the workspace `local_workspace()` gives the shim's agents."""
    from pydantic_ai.usage import RunUsage
    from pydantic_ai.workspaces import LocalWorkspaceBackend, Workspace

    return RunContext(
        deps=None,
        model=cast(_Model[Any], None),
        usage=RunUsage(),
        workspace=Workspace(LocalWorkspaceBackend(shared.workspace())),
    )


def test_file_tools_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # The harness-backed Read/Write/Edit are contained to the workspace root.
    monkeypatch.setenv('GITHUB_WORKSPACE', str(tmp_path))
    f = tmp_path / 'sub' / 'note.txt'
    # Write creates the missing parent directory, mirroring Claude's `Write`.
    assert 'Wrote' in asyncio.run(pkg.write_file(_ctx(), str(f), 'hello\nworld\n'))
    body = asyncio.run(pkg.read_file(_ctx(), str(f)))
    assert 'hello' in body and 'world' in body
    assert 'Edited' in asyncio.run(pkg.edit_file(_ctx(), str(f), 'world', 'there'))
    assert 'there' in asyncio.run(pkg.read_file(_ctx(), str(f)))
    assert 'note.txt' in asyncio.run(pkg.list_dir(_ctx(), str(tmp_path / 'sub')))
    # The harness requires a unique match; a missing string comes back as an error.
    miss = asyncio.run(pkg.edit_file(_ctx(), str(f), 'absent', 'x'))
    assert miss.startswith('error:') and 'not found' in miss


def test_read_file_offset_and_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('GITHUB_WORKSPACE', str(tmp_path))
    f = tmp_path / 'n.txt'
    f.write_text('l1\nl2\nl3\nl4\n', encoding='utf-8')
    # Claude's 1-based offset=2 maps to the harness 0-based offset; limit=2.
    out = asyncio.run(pkg.read_file(_ctx(), str(f), offset=2, limit=2))
    assert 'l2' in out and 'l3' in out
    assert 'l1' not in out and 'l4' not in out


def test_read_continuation_hint_uses_one_based_offset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # The harness's truncation hint carries its own 0-based offset; this tool's
    # `offset` is 1-based, so the hint must be bumped by one. Otherwise a model
    # that follows the hint literally re-reads the last line of the prior chunk.
    monkeypatch.setenv('GITHUB_WORKSPACE', str(tmp_path))
    f = tmp_path / 'big.txt'
    f.write_text('\n'.join(f'L{i}' for i in range(1, 4)) + '\n', encoding='utf-8')  # L1..L3
    first = asyncio.run(pkg.read_file(_ctx(), str(f), limit=2))  # reads L1,L2 + a continuation hint
    assert 'Use offset=3 to continue reading.' in first  # harness emits 2 (0-based); bumped to 3
    assert 'Use offset=2 to continue reading.' not in first
    # Following the (1-based) hint continues at L3 with no duplicated boundary line.
    nxt = asyncio.run(pkg.read_file(_ctx(), str(f), offset=3, limit=2))
    assert 'L3' in nxt and 'L2' not in nxt


def test_read_file_returns_an_image_as_the_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # An image comes back as the harness returns it, header plus the file, not as text.
    monkeypatch.setenv('GITHUB_WORKSPACE', str(tmp_path))
    png = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR'
    (tmp_path / 'chart.png').write_bytes(png)
    out = asyncio.run(pkg.read_file(_ctx(), str(tmp_path / 'chart.png')))
    assert isinstance(out, list)
    assert out[1] == BinaryContent(data=png, media_type='image/png')


def test_read_limit_zero_behaves_like_omitted_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Passed straight to the harness, `limit=0` reads zero lines and emits a
    # same-offset hint that loops. The adapter normalizes `limit<=0` to an omitted
    # limit, so a small file comes back whole with no continuation hint.
    monkeypatch.setenv('GITHUB_WORKSPACE', str(tmp_path))
    f = tmp_path / 'f.txt'
    f.write_text('one\ntwo\nthree\n', encoding='utf-8')
    out = asyncio.run(pkg.read_file(_ctx(), str(f), offset=1, limit=0))
    assert 'one' in out and 'three' in out
    assert 'to continue reading' not in out  # whole short file fit; no looping hint


def test_harness_backed_tools_surface_oserror_as_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # The harness converts only a fixed set of exceptions to `ModelRetry`; a bare
    # `OSError` (here `ENAMETOOLONG` from an over-long path component) is not one
    # of them, so each adapter must catch it and return an `error:` string instead
    # of letting it escape and abort the whole agent run.
    monkeypatch.setenv('GITHUB_WORKSPACE', str(tmp_path))
    long_path = 'a' * 10_000
    read = asyncio.run(pkg.read_file(_ctx(), long_path))
    assert isinstance(read, str) and read.startswith('error:')
    assert asyncio.run(pkg.edit_file(_ctx(), long_path, 'x', 'y')).startswith('error:')
    assert asyncio.run(pkg.list_dir(_ctx(), long_path)).startswith('error:')
    assert asyncio.run(pkg.glob_search(_ctx(), '*.py', long_path)).startswith('error:')
    assert asyncio.run(pkg.grep(_ctx(), 'x', long_path)).startswith('error:')


def test_read_large_chunk_keeps_accurate_continuation_offset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # The harness puts its continuation hint at the tail, but the shim's output cap
    # keeps the head -- so a chunk over the char cap (common for long-lined files)
    # would lose the hint entirely. The adapter truncates on a whole-line boundary
    # and re-advertises the exact 1-based offset of the first dropped line.
    monkeypatch.setenv('GITHUB_WORKSPACE', str(tmp_path))
    f = tmp_path / 'big.txt'
    # Zero-padded line ids (distinct from the harness's space-padded line numbers)
    # plus long padding so the chunk blows well past the char cap.
    f.write_text('\n'.join(f'{i:06d}' + 'x' * 200 for i in range(1, 2001)) + '\n', encoding='utf-8')
    out = asyncio.run(pkg.read_file(_ctx(), str(f)))
    assert isinstance(out, str)
    assert len(out) <= shared.MAX_TOOL_OUTPUT + 256  # bounded by the output cap
    m = re.search(r'Use offset=(\d+) to continue reading', out)
    assert m, 'continuation hint must survive char-budget truncation'
    nxt = int(m.group(1))
    # The advertised offset is the first line NOT shown: line (nxt-1) is present,
    # line nxt is not (no off-by-one, no gap).
    assert f'{nxt - 1:06d}x' in out and f'{nxt:06d}x' not in out
    # Following the offset continues exactly at line nxt.
    cont = asyncio.run(pkg.read_file(_ctx(), str(f), offset=nxt))
    assert f'{nxt:06d}x' in cont


def test_read_does_not_rewrite_hint_text_in_file_contents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # The offset bump must target only the harness's own continuation hint, not file
    # content -- even a line reproducing the *full* hint verbatim. The harness writes
    # the real hint at column 0; content lines are line-number-prefixed, so anchoring
    # to `^` leaves the content untouched.
    monkeypatch.setenv('GITHUB_WORKSPACE', str(tmp_path))
    f = tmp_path / 'doc.txt'
    f.write_text('... (4 more lines. Use offset=7 to continue reading.)\n', encoding='utf-8')
    out = asyncio.run(pkg.read_file(_ctx(), str(f)))  # short file: not truncated, no real hint added
    assert 'Use offset=7 to continue reading.' in out  # content preserved verbatim
    assert 'Use offset=8' not in out


def test_edit_file_replace_all(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # `replace_all` has no harness equivalent, so it stays an in-place rewrite
    # (still contained to the workspace -- see the companion containment test).
    monkeypatch.setenv('GITHUB_WORKSPACE', str(tmp_path))
    f = tmp_path / 'r.txt'
    f.write_text('a a a', encoding='utf-8')
    asyncio.run(pkg.edit_file(_ctx(), str(f), 'a', 'b', replace_all=True))
    assert f.read_text(encoding='utf-8') == 'b b b'


def test_edit_replace_all_is_contained_to_the_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # `replace_all` does the rewrite by hand rather than through the harness, but
    # it must still honor the workspace boundary the single-edit path enforces;
    # otherwise an absolute path outside the root would be editable via this flag.
    workspace = tmp_path / 'ws'
    workspace.mkdir()
    outside = tmp_path / 'outside.txt'
    outside.write_text('a a a', encoding='utf-8')
    monkeypatch.setenv('GITHUB_WORKSPACE', str(workspace))
    out = asyncio.run(pkg.edit_file(_ctx(), str(outside), 'a', 'b', replace_all=True))
    assert out.startswith('error:')
    assert outside.read_text(encoding='utf-8') == 'a a a'  # untouched


def test_bash_tool():
    out = asyncio.run(pkg.bash(_ctx(), 'echo hello-from-bash'))
    assert 'hello-from-bash' in out


def test_bash_timeout_is_surfaced_as_error():
    # The harness *returns* a `[Command timed out ...]` sentinel rather than
    # raising; the adapter must wrap it as an `error:` string (as the old tool
    # did) so the model doesn't read a timeout as a successful, empty result.
    out = asyncio.run(pkg.bash(_ctx(), 'sleep 5', timeout=1))
    assert out.startswith('error:') and 'timed out' in out


def test_bash_timeout_after_partial_output_is_an_error():
    # The timeout sentinel is the *last* line, after whatever the command printed.
    out = asyncio.run(pkg.bash(_ctx(), 'echo partial; sleep 5', timeout=1))
    assert out.startswith('error:') and 'timed out' in out


def test_bash_output_is_capped_keeping_the_tail():
    # The harness caps output only at its own tool dispatch; called directly, the
    # adapter caps it, keeping the tail where errors and the exit code land.
    out = asyncio.run(pkg.bash(_ctx(), f"{sys.executable} -c \"print('x' * 60000); print('THE-END')\""))
    assert out.startswith('[... output truncated')
    assert out.rstrip().endswith('THE-END')
    assert len(out) <= shared.MAX_TOOL_OUTPUT + 100


def test_bash_subprocess_startup_failure_is_an_error_not_a_crash(monkeypatch: pytest.MonkeyPatch):
    # `run_command` can raise a raw `OSError` (not converted to `ModelRetry`) when
    # subprocess startup fails -- e.g. the workspace cwd doesn't exist. That must
    # come back as an `error:` string instead of aborting the whole agent run.
    monkeypatch.setenv('GITHUB_WORKSPACE', '/definitely/not/a/workspace/xyz')
    out = asyncio.run(pkg.bash(_ctx(), 'echo hi', timeout=1))
    assert out.startswith('error:')


def test_grep_tool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    if shutil.which('rg') is None:
        pytest.skip('ripgrep is installed by the gh-aw runtime, not the generic CI runner')

    # grep runs ripgrep through the harness shell capability; the adapter keys off
    # ripgrep's exit code (parsed from `run_command`'s trailing `[exit code: N]`)
    # to unwrap the `[stdout]` framing into `file:line:text` matches, and maps
    # exit-1 ("nothing matched") to the harness's own `No matches found.` sentinel.
    monkeypatch.setenv('GITHUB_WORKSPACE', str(tmp_path))
    (tmp_path / 'a.txt').write_text('alpha\nNEEDLE here\n', encoding='utf-8')
    assert 'a.txt:2:NEEDLE here' in asyncio.run(pkg.grep(_ctx(), 'NEEDLE', '.'))
    assert asyncio.run(pkg.grep(_ctx(), 'ZZZNOPE', '.')) == 'No matches found.'
    # An empty path normalizes to the workspace root rather than reaching `rg`
    # as an empty argument (which would error).
    assert 'a.txt:2:NEEDLE here' in asyncio.run(pkg.grep(_ctx(), 'NEEDLE', ''))


def test_grep_bad_pattern_is_an_error_not_a_match(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # ripgrep exits 2 on a malformed regex. Even though it writes to the framed
    # output, the adapter keys off the exit code, so it surfaces as an error
    # rather than being mistaken for a match (the `[stdout]`-prefix sniff bug).
    monkeypatch.setenv('GITHUB_WORKSPACE', str(tmp_path))
    (tmp_path / 'a.txt').write_text('alpha\n', encoding='utf-8')
    out = asyncio.run(pkg.grep(_ctx(), '(', '.'))
    assert out.startswith('error:')


def test_grep_path_is_contained_to_the_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # `ShellToolset` would happily run `rg -- ../..`; the filesystem preflight
    # rejects a path that escapes the workspace root before ripgrep ever runs.
    workspace = tmp_path / 'ws'
    workspace.mkdir()
    (tmp_path / 'secret.txt').write_text('TOPSECRET\n', encoding='utf-8')
    monkeypatch.setenv('GITHUB_WORKSPACE', str(workspace))
    out = asyncio.run(pkg.grep(_ctx(), 'TOPSECRET', '..'))
    assert out.startswith('error:')
    assert 'TOPSECRET' not in out


def test_grep_large_match_set_is_not_misreported_as_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # A match set larger than the harness output cap is tail-truncated, which
    # elides the `[stdout]` header and prepends a truncation marker. The adapter
    # must still return it as matches, not as an error.

    # `pkg.grep` is the re-exported callable; reach the module to patch its deps.
    grep_mod = importlib.import_module('pydantic_ai_gh_aw_shim.grep')

    monkeypatch.setenv('GITHUB_WORKSPACE', str(tmp_path))  # empty workspace: no context blocks
    truncated = f'{grep_mod._TRUNCATION_PREFIX}, showing last 50000 chars]\nsrc/a.py:1:hit\nsrc/b.py:2:hit\n'

    async def _fake_run_shell(ctx: RunContext[object], command: str, *, timeout_seconds: float) -> str:
        return truncated

    class _FakeFs:
        async def file_info(self, path: str, *, workspace: object) -> str:
            return 'ok'

    monkeypatch.setattr(grep_mod, 'run_shell', _fake_run_shell)
    monkeypatch.setattr(grep_mod, 'filesystem', lambda: _FakeFs())
    out = asyncio.run(grep_mod.grep(_ctx(), 'hit', '.'))
    assert not out.startswith('error:')
    assert 'src/a.py:1:hit' in out and 'src/b.py:2:hit' in out
    assert grep_mod._TRUNCATION_PREFIX not in out


def test_glob_tool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # `Glob` returns matches relative to the search path (workspace root for '.').
    monkeypatch.setenv('GITHUB_WORKSPACE', str(tmp_path))
    (tmp_path / 'x').mkdir()
    (tmp_path / 'x' / 'a.py').write_text('', encoding='utf-8')
    (tmp_path / 'x' / 'b.txt').write_text('', encoding='utf-8')
    res = asyncio.run(pkg.glob_search(_ctx(), '**/*.py', '.'))
    assert 'x/a.py' in res and 'b.txt' not in res


def test_glob_outside_base_is_handled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # An absolute glob pattern can't resolve under the workspace root; the
    # adapter rejects it up front.
    monkeypatch.setenv('GITHUB_WORKSPACE', str(tmp_path))
    out = asyncio.run(pkg.glob_search(_ctx(), '/etc/*', '.'))
    assert out.startswith('error:')


def test_glob_missing_search_path_is_an_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('GITHUB_WORKSPACE', str(tmp_path))
    assert asyncio.run(pkg.glob_search(_ctx(), '*.py', 'nope')).startswith('error:')


def test_ls_and_glob_surface_dotfiles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # The harness `list_directory`/`find_files` walkers hide every dot-prefixed
    # path, which would make `.github/` (where gh-aw's workflows live) invisible.
    # The shim hand-rolls the enumeration so dot-paths stay discoverable.
    monkeypatch.setenv('GITHUB_WORKSPACE', str(tmp_path))
    (tmp_path / '.github' / 'workflows').mkdir(parents=True)
    (tmp_path / '.github' / 'workflows' / 'ci.yml').write_text('', encoding='utf-8')
    assert '.github/' in asyncio.run(pkg.list_dir(_ctx(), '.'))
    assert 'workflows/' in asyncio.run(pkg.list_dir(_ctx(), '.github'))
    assert '.github/workflows/ci.yml' in asyncio.run(pkg.glob_search(_ctx(), '.github/**/*.yml', '.'))


def test_ls_and_glob_paths_are_contained_to_the_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # The hand-rolled enumeration still preflights containment through the
    # filesystem capability, so a path escaping the workspace is rejected.
    workspace = tmp_path / 'ws'
    workspace.mkdir()
    (tmp_path / 'secret.txt').write_text('TOPSECRET\n', encoding='utf-8')
    monkeypatch.setenv('GITHUB_WORKSPACE', str(workspace))
    ls_out = asyncio.run(pkg.list_dir(_ctx(), '..'))
    glob_out = asyncio.run(pkg.glob_search(_ctx(), '*.txt', '..'))
    assert ls_out.startswith('error:') and 'secret.txt' not in ls_out
    assert glob_out.startswith('error:') and 'secret.txt' not in glob_out


def test_glob_reports_matched_symlink_not_its_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Resolving a match is only for the containment decision; the returned path
    # must be the name that matched. An in-workspace symlink (here `CLAUDE.md` ->
    # `AGENTS.md`, as this repo actually has) must be reported as `CLAUDE.md`, and
    # must not be collapsed into its target by dedup.
    monkeypatch.setenv('GITHUB_WORKSPACE', str(tmp_path))
    (tmp_path / 'AGENTS.md').write_text('x', encoding='utf-8')
    (tmp_path / 'CLAUDE.md').symlink_to('AGENTS.md')
    assert asyncio.run(pkg.glob_search(_ctx(), 'CLAUDE.md', '.')).splitlines()[-1] == 'CLAUDE.md'
    both = asyncio.run(pkg.glob_search(_ctx(), '*.md', '.'))
    assert 'AGENTS.md' in both and 'CLAUDE.md' in both


def test_glob_pattern_cannot_escape_via_dotdot_or_symlink(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Reverting to a stdlib glob must not lose containment: a `..` in the pattern,
    # or an in-workspace symlink pointing out, must NOT surface a file outside the
    # workspace -- a purely lexical `relative_to` check would miss both.
    workspace = tmp_path / 'ws'
    workspace.mkdir()
    (tmp_path / 'secret.txt').write_text('TOPSECRET\n', encoding='utf-8')
    (workspace / 'link').symlink_to(tmp_path)  # symlink that climbs out of the workspace
    monkeypatch.setenv('GITHUB_WORKSPACE', str(workspace))
    via_dotdot = asyncio.run(pkg.glob_search(_ctx(), '../secret.txt', '.'))
    via_symlink = asyncio.run(pkg.glob_search(_ctx(), 'link/*.txt', '.'))
    assert 'secret.txt' not in via_dotdot
    assert 'secret.txt' not in via_symlink


def test_multi_edit_atomic(tmp_path: Path):
    f = tmp_path / 'm.txt'
    f.write_text('one two three', encoding='utf-8')
    ok = pkg.multi_edit(str(f), [{'old_string': 'one', 'new_string': '1'}, {'old_string': 'three', 'new_string': '3'}])
    assert 'applied 2 edit(s)' in ok
    assert f.read_text(encoding='utf-8') == '1 two 3'
    # A failing edit writes nothing (atomic).
    res = pkg.multi_edit(str(f), [{'old_string': '1', 'new_string': 'X'}, {'old_string': 'absent', 'new_string': 'Y'}])
    assert 'edit #2' in res and 'not found' in res
    assert f.read_text(encoding='utf-8') == '1 two 3'


def test_multi_edit_replace_all(tmp_path: Path):
    f = tmp_path / 'r.txt'
    f.write_text('a a a', encoding='utf-8')
    pkg.multi_edit(str(f), [{'old_string': 'a', 'new_string': 'b', 'replace_all': True}])
    assert f.read_text(encoding='utf-8') == 'b b b'


def test_web_fetch_only_enabled_on_real_anthropic(monkeypatch: pytest.MonkeyPatch):
    """`web_fetch_20250910` is Anthropic-specific; compatible endpoints may not support it."""
    monkeypatch.delenv('ANTHROPIC_BASE_URL', raising=False)
    caps = shim._anthropic_native_capabilities()  # pyright: ignore[reportPrivateUsage]
    assert len(caps) == 1 and isinstance(caps[0], NativeTool)

    monkeypatch.setenv('ANTHROPIC_BASE_URL', 'https://api.anthropic.com')
    assert len(shim._anthropic_native_capabilities()) == 1  # pyright: ignore[reportPrivateUsage]

    monkeypatch.setenv('ANTHROPIC_BASE_URL', 'https://api.minimax.io/anthropic')
    assert shim._anthropic_native_capabilities() == []  # pyright: ignore[reportPrivateUsage]


def test_todo_write_renders_plan_via_harness():
    # TodoWrite maps Claude's todo schema onto the harness `planning` capability
    # and returns its `render_plan` checklist (with a progress line).
    out = asyncio.run(pkg.todo_write([{'content': 'do x', 'status': 'in_progress', 'activeForm': 'doing x'}]))
    assert 'do x' in out and '[~]' in out and '(0/1 completed)' in out
    # A completed step shows as done; an unknown status falls back to pending.
    out2 = asyncio.run(
        pkg.todo_write(
            [
                {'content': 'a', 'status': 'completed', 'activeForm': ''},
                {'content': 'b', 'status': 'bogus', 'activeForm': ''},
                # `blocked` is a harness status Claude's schema doesn't have.
                {'content': 'c', 'status': 'blocked', 'activeForm': ''},
            ]
        )
    )
    assert '[x] a' in out2 and '[ ] b' in out2 and '[ ] c' in out2 and '(1/3 completed)' in out2


def test_exit_plan_mode_returns_ack():
    assert 'proceeding' in pkg.exit_plan_mode('step 1; step 2').lower()


def test_plan_mode_keeps_new_readonly_tools_drops_multiedit():
    # Note: WebFetch is an Anthropic server-side capability (not in the callable list).
    names = set(asyncio.run(_toolset_names(None, 'plan')))
    assert 'MultiEdit' not in names  # mutating
    assert {'TodoWrite', 'ExitPlanMode'} <= names  # non-mutating callables


@pytest.mark.parametrize(
    ('workflow', 'expected_limit'),
    [
        ('Pydantic AI Attention Triage', 50),
        ('Other Pydantic AI workflow', 400),
    ],
)
def test_run_uses_workflow_request_limit(workflow: str, expected_limit: int, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('GITHUB_WORKFLOW', workflow)
    assert shim.run_request_limit() == expected_limit
    passed_limits: list[int | None] = []

    class _Agent:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def output_validator(self, _validator: object) -> None:
            pass

        async def __aenter__(self) -> object:
            return self

        async def __aexit__(self, *_args: object) -> None:
            pass

        async def run(self, _prompt: str, *, usage_limits: UsageLimits, usage: RunUsage) -> None:
            passed_limits.append(usage_limits.request_limit)
            raise RuntimeError('stop after capturing the run limit')

    monkeypatch.setattr(shim, 'Agent', _Agent)

    def _ignore_emit(_obj: dict[str, object]) -> None:
        pass

    monkeypatch.setattr(shim, 'emit', _ignore_emit)
    rc = asyncio.run(
        shim.run(
            prompt='test',
            model=TestModel(),
            label='test-model',
            claude_code_toolset=shim.select_claude_code_toolset(None, None, task=None),
            mcp_servers=[],
            session_id='request-limit',
        )
    )
    assert rc == 1
    assert passed_limits == [expected_limit]


def test_instructions_encourage_parallel_tool_calls():
    assert shim.INSTRUCTIONS.strip()
    assert 'parallel' in shim.INSTRUCTIONS.lower()


def test_run_routes_workflow_prompt_to_system_instructions(monkeypatch: pytest.MonkeyPatch):
    """Workflow prompt rides in the system instruction; user message is RUN_TRIGGER."""
    seen_instructions: list[str] = []
    received: list[ModelMessage] = []
    emitted: list[dict[str, object]] = []

    def _respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen_instructions.append(info.instructions or '')
        received.extend(messages)
        return ModelResponse(parts=[TextPart('done')])

    async def _stream(messages: list[ModelMessage], info: AgentInfo):
        seen_instructions.append(info.instructions or '')
        received.extend(messages)
        yield 'done'

    monkeypatch.setattr(shim, 'emit', lambda obj: emitted.append(dict(obj)))  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    monkeypatch.setattr(shim, 'log_safe_outputs_state', lambda: None)

    sentinel = '### WORKFLOW TASK SPEC: review the PR per the rules above ###'
    asyncio.run(
        shim.run(
            prompt=sentinel,
            model=FunctionModel(_respond, stream_function=_stream),
            label='test-model',
            claude_code_toolset=shim.select_claude_code_toolset(None, None, task=None),
            mcp_servers=[],
            session_id='test-session',
        )
    )

    instructions = seen_instructions[0]
    user_text = '\n'.join(str(p.content) for m in received for p in m.parts if isinstance(p, UserPromptPart))

    # Order matters for prompt-prefix caching: INSTRUCTIONS must come first.
    assert instructions.startswith(shim.INSTRUCTIONS)
    assert sentinel in instructions
    assert user_text == shim.RUN_TRIGGER
    # `run()` must emit both a `system`/`init` line and a `result` line for gh-aw.
    kinds = [(e.get('type'), e.get('subtype')) for e in emitted]
    assert ('system', 'init') in kinds
    assert any(t == 'result' and s == 'success' for t, s in kinds)
    assert sentinel not in user_text


def _safe_outputs_toolset(sink: Path) -> AbstractToolset[object]:
    """Stand-in for gh-aw's safe-outputs MCP server, which appends each output to `GH_AW_SAFE_OUTPUTS`."""
    toolset: FunctionToolset[object] = FunctionToolset()

    @toolset.tool_plain
    def noop(message: str) -> str:
        with sink.open('a', encoding='utf-8') as f:
            f.write(json.dumps({'type': 'noop', 'message': message}) + '\n')
        return 'ok'

    @toolset.tool_plain
    def probe() -> str:
        return 'nothing new'

    return toolset


def _run_shim(
    respond: Callable[[list[ModelMessage], AgentInfo], ModelResponse], sink: Path, *, mcp_tool_names: bool = False
) -> int:
    """Run the shim on a `FunctionModel` that streams each response `respond` returns.

    `mcp_tool_names` names the safe-output tools as the model sees gh-aw's `safeoutputs` server,
    `mcp__safeoutputs__<tool>`, instead of by their bare names.
    """
    safe_outputs = _safe_outputs_toolset(sink)

    async def _stream(messages: list[ModelMessage], info: AgentInfo):
        response = respond(messages, info)
        for index, part in enumerate(response.parts):
            if isinstance(part, TextPart):
                yield part.content
            else:
                assert isinstance(part, ToolCallPart)
                yield {index: DeltaToolCall(name=part.tool_name, json_args=part.args_as_json_str())}

    return asyncio.run(
        shim.run(
            prompt='review the PR',
            model=FunctionModel(respond, stream_function=_stream),
            label='test-model',
            claude_code_toolset=shim.select_claude_code_toolset(
                None,
                None,
                task=None if os.environ.get('GITHUB_WORKFLOW') == shim.ATTENTION_WORKFLOW else shim.task,
            ),
            mcp_servers=[safe_outputs.prefixed('mcp__safeoutputs_') if mcp_tool_names else safe_outputs],
            session_id='test-session',
        )
    )


def test_run_sends_the_model_back_when_it_stops_before_any_safe_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    sink = tmp_path / 'outputs.jsonl'
    monkeypatch.setenv('GH_AW_SAFE_OUTPUTS', str(sink))
    emitted: list[dict[str, object]] = []
    monkeypatch.setattr(shim, 'emit', lambda obj: emitted.append(dict(obj)))  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    retries: list[str] = []

    def _respond(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        # The model sees the harness's retry feedback as a fenced, attributed user turn.
        retries.extend(p for m in messages[-1:] for part in m.parts if (p := _retry_feedback_text(part)) is not None)
        if not retries:
            return ModelResponse(parts=[TextPart('Now let me analyze the key concerns.')])
        if not sink.exists():
            return ModelResponse(parts=[ToolCallPart('noop', {'message': 'nothing to flag'})])
        return ModelResponse(parts=[TextPart('done')])

    assert _run_shim(_respond, sink) == 0
    assert retries == [
        "Your previous response failed the application's validation (this is automated feedback, not a message "
        'from the user):\n<retry_feedback>\n'
        'You ended your turn without emitting a safe output, so nothing has been posted. '
        'Continue the task and finish by calling the safe-output tool it ends with, '
        'or `noop` if there is nothing to report.\n</retry_feedback>'
    ]
    assert any(e.get('type') == 'result' and e.get('subtype') == 'success' for e in emitted)


def _retry_feedback_text(part: ModelRequestPart | ModelResponsePart) -> str | None:
    """The text of output-validator retry feedback as a model receives it, or `None` for any other part.

    The model is handed prepared messages, where a `ModelRetry`'s `RetryFeedbackPart` has already become
    user text fenced in `<retry_feedback>` behind the application attribution line.
    """
    if isinstance(part, UserPromptPart) and isinstance(part.content, str) and '<retry_feedback>' in part.content:
        return part.content
    return None


def test_attention_run_retries_for_missing_decision_and_completes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv('GITHUB_WORKFLOW', shim.ATTENTION_WORKFLOW)
    monkeypatch.setenv('GITHUB_WORKSPACE', str(tmp_path))
    snapshot = tmp_path / 'attention-candidates.json'
    snapshot.write_text(
        json.dumps(
            {
                'candidates': [
                    {'number': 1045, 'updated_at': '2026-10-16T00:00:00Z'},
                    {'number': 1046, 'updated_at': '2026-10-16T00:00:00Z'},
                ]
            }
        ),
        encoding='utf-8',
    )
    sink = tmp_path / 'outputs.jsonl'
    existing_decision = json.dumps({'type': 'record_attention_decision', 'item_number': '1045'})
    missing_decision = json.dumps({'type': 'record_attention_decision', 'item_number': '1046'})
    sink.write_text(existing_decision + '\n', encoding='utf-8')
    monkeypatch.setenv('GH_AW_SAFE_OUTPUTS', str(sink))
    emitted: list[dict[str, object]] = []
    monkeypatch.setattr(shim, 'emit', lambda obj: emitted.append(dict(obj)))  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    retries: list[str] = []

    def _respond(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        for message in messages:
            for part in message.parts:
                if (text := _retry_feedback_text(part)) is not None:
                    retries.append(text)
        if retries:
            with sink.open('a', encoding='utf-8') as f:
                f.write(missing_decision + '\n')
        return ModelResponse(parts=[TextPart('done')])

    assert _run_shim(_respond, sink) == 0
    assert len(retries) == 1
    assert '1046' in retries[0]
    assert '1045' not in retries[0]
    outputs = sink.read_text(encoding='utf-8').splitlines()
    assert outputs.count(existing_decision) == 1
    assert outputs.count(missing_decision) == 1
    assert any(e.get('type') == 'result' and e.get('subtype') == 'success' for e in emitted)


def test_attention_agent_cannot_shrink_candidates_to_hide_a_missing_decision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setenv('GITHUB_WORKFLOW', shim.ATTENTION_WORKFLOW)
    monkeypatch.setenv('GITHUB_WORKSPACE', str(tmp_path))
    snapshot = tmp_path / 'attention-candidates.json'
    candidates = [{'number': number, 'updated_at': '2026-10-16T00:00:00Z'} for number in range(1040, 1050)]
    snapshot.write_text(json.dumps({'candidates': candidates}), encoding='utf-8')
    sink = tmp_path / 'outputs.jsonl'
    sink.write_text(
        ''.join(
            json.dumps({'type': 'record_attention_decision', 'item_number': str(candidate['number'])}) + '\n'
            for candidate in candidates[:-1]
        ),
        encoding='utf-8',
    )
    monkeypatch.setenv('GH_AW_SAFE_OUTPUTS', str(sink))
    monkeypatch.setattr(shim, 'emit', lambda _obj: None)  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    retries: list[str] = []
    available_tools: list[set[str]] = []

    def _respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        available_tools.append({tool.name for tool in info.function_tools})
        for message in messages:
            for part in message.parts:
                if (text := _retry_feedback_text(part)) is not None:
                    retries.append(text)
        if not available_tools[:-1]:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        'Write',
                        {
                            'file_path': str(snapshot),
                            'content': json.dumps({'candidates': candidates[:-1]}),
                        },
                    )
                ]
            )
        if any('1049' in retry for retry in retries):
            with sink.open('a', encoding='utf-8') as output:
                output.write(json.dumps({'type': 'record_attention_decision', 'item_number': '1049'}) + '\n')
        return ModelResponse(parts=[TextPart('done')])

    assert _run_shim(_respond, sink) == 0
    assert json.loads(snapshot.read_text(encoding='utf-8'))['candidates'] == candidates
    assert {'Write', 'Edit', 'MultiEdit', 'Bash'}.isdisjoint(available_tools[0])
    assert 'Task' not in available_tools[0]
    assert any('1049' in retry for retry in retries)


def test_run_fails_when_the_model_never_emits_a_safe_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    sink = tmp_path / 'outputs.jsonl'
    monkeypatch.setenv('GH_AW_SAFE_OUTPUTS', str(sink))
    emitted: list[dict[str, object]] = []
    monkeypatch.setattr(shim, 'emit', lambda obj: emitted.append(dict(obj)))  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    requests = 0

    def _respond(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        nonlocal requests
        requests += 1
        return ModelResponse(parts=[TextPart('Let me look further.')])

    assert _run_shim(_respond, sink) == 1
    assert requests == shim.NO_SAFE_OUTPUT_RETRIES + 1
    assert any(e.get('type') == 'result' and e.get('is_error') is True for e in emitted)


def test_attention_completion_requests_only_missing_candidate_decisions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setenv('GITHUB_WORKFLOW', 'Pydantic AI Attention Triage')
    monkeypatch.setenv('GITHUB_WORKSPACE', str(tmp_path))
    snapshot = tmp_path / 'attention-candidates.json'
    snapshot.write_text(
        json.dumps(
            {
                'candidates': [
                    {'number': 1045, 'updated_at': '2026-10-16T00:00:00Z'},
                    {'number': 1046, 'updated_at': '2026-10-16T00:00:00Z'},
                ]
            }
        ),
        encoding='utf-8',
    )
    sink = tmp_path / 'outputs.jsonl'
    sink.write_text(
        json.dumps({'type': 'record_attention_decision', 'item_number': '1045'})
        + '\n'
        + json.dumps({'type': 'noop', 'message': 'already recorded'})
        + '\n',
        encoding='utf-8',
    )
    monkeypatch.setenv('GH_AW_SAFE_OUTPUTS', str(sink))

    with pytest.raises(ModelRetry, match='1046') as exc_info:
        shim.require_safe_output('done')
    assert 'only for these missing IDs' in str(exc_info.value)

    with sink.open('a', encoding='utf-8') as f:
        f.write(json.dumps({'type': 'record_attention_decision', 'item_number': '1046'}) + '\n')
        # The completion check leaves exact-set enforcement to the host.
        f.write(json.dumps({'type': 'record_attention_decision', 'item_number': '1045'}) + '\n')
        f.write(json.dumps({'type': 'record_attention_decision', 'item_number': '9999'}) + '\n')
    assert shim.require_safe_output('done') == 'done'


def test_attention_validator_fails_when_candidate_snapshot_is_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv('GITHUB_WORKFLOW', shim.ATTENTION_WORKFLOW)
    monkeypatch.setenv('GITHUB_WORKSPACE', str(tmp_path))
    sink = tmp_path / 'outputs.jsonl'
    sink.write_text(json.dumps({'type': 'noop', 'message': 'complete'}) + '\n', encoding='utf-8')
    monkeypatch.setenv('GH_AW_SAFE_OUTPUTS', str(sink))

    assert shim.safe_output_pending()
    with pytest.raises(ValueError, match='Attention candidate snapshot is missing'):
        shim.require_safe_output('done')


def test_attention_validator_fails_when_safe_output_sink_is_not_configured(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setenv('GITHUB_WORKFLOW', shim.ATTENTION_WORKFLOW)
    monkeypatch.setenv('GITHUB_WORKSPACE', str(tmp_path))
    monkeypatch.delenv('GH_AW_SAFE_OUTPUTS', raising=False)

    assert shim.safe_output_pending()
    with pytest.raises(ValueError, match='Attention safe-output sink is not configured'):
        shim.require_safe_output('done')


def test_ordinary_workflow_accepts_unconfigured_safe_output_sink(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('GITHUB_WORKFLOW', 'CI Review')
    monkeypatch.delenv('GH_AW_SAFE_OUTPUTS', raising=False)

    assert not shim.safe_output_pending()
    assert shim.require_safe_output('done') == 'done'


@pytest.mark.parametrize('sink_configured', [True, False])
def test_attention_run_fails_when_immutable_completion_artifact_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sink_configured: bool
):
    monkeypatch.setenv('GITHUB_WORKFLOW', shim.ATTENTION_WORKFLOW)
    monkeypatch.setenv('GITHUB_WORKSPACE', str(tmp_path))
    sink = tmp_path / 'outputs.jsonl'
    sink.write_text(json.dumps({'type': 'noop', 'message': 'complete'}) + '\n', encoding='utf-8')
    if sink_configured:
        monkeypatch.setenv('GH_AW_SAFE_OUTPUTS', str(sink))
    else:
        monkeypatch.delenv('GH_AW_SAFE_OUTPUTS', raising=False)
    emitted: list[dict[str, object]] = []
    monkeypatch.setattr(shim, 'emit', lambda obj: emitted.append(dict(obj)))  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    requests = 0

    def _respond(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        nonlocal requests
        requests += 1
        return ModelResponse(parts=[TextPart('done')])

    assert _run_shim(_respond, sink) == 1
    assert requests == 1
    assert any(e.get('type') == 'result' and e.get('is_error') is True for e in emitted)


def test_attention_safe_output_pending_tracks_missing_decisions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv('GITHUB_WORKFLOW', shim.ATTENTION_WORKFLOW)
    monkeypatch.setenv('GITHUB_WORKSPACE', str(tmp_path))
    snapshot = tmp_path / 'attention-candidates.json'
    snapshot.write_text(json.dumps({'candidates': [{'number': number} for number in range(1, 11)]}), encoding='utf-8')
    sink = tmp_path / 'outputs.jsonl'
    sink.write_text(
        ''.join(
            json.dumps({'type': 'record_attention_decision', 'item_number': str(number)}) + '\n'
            for number in range(1, 10)
        ),
        encoding='utf-8',
    )
    monkeypatch.setenv('GH_AW_SAFE_OUTPUTS', str(sink))

    assert shim.safe_output_pending()

    with sink.open('a', encoding='utf-8') as f:
        f.write(json.dumps({'type': 'record_attention_decision', 'item_number': '10'}) + '\n')
    assert not shim.safe_output_pending()


def test_attention_request_limit_fails_when_decisions_are_incomplete(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv('GITHUB_WORKFLOW', shim.ATTENTION_WORKFLOW)
    monkeypatch.setenv('GITHUB_WORKSPACE', str(tmp_path))
    snapshot = tmp_path / 'attention-candidates.json'
    snapshot.write_text(json.dumps({'candidates': [{'number': number} for number in range(1, 11)]}), encoding='utf-8')
    sink = tmp_path / 'outputs.jsonl'
    sink.write_text(
        ''.join(
            json.dumps({'type': 'record_attention_decision', 'item_number': str(number)}) + '\n'
            for number in range(1, 10)
        ),
        encoding='utf-8',
    )
    monkeypatch.setenv('GH_AW_SAFE_OUTPUTS', str(sink))
    monkeypatch.setattr(shim, 'run_request_limit', lambda: 1)
    emitted: list[dict[str, object]] = []
    monkeypatch.setattr(shim, 'emit', lambda obj: emitted.append(dict(obj)))  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]

    def _respond(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('done')])

    assert _run_shim(_respond, sink) == 1
    results = [e for e in emitted if e.get('type') == 'result']
    assert len(results) == 1
    assert results[0].get('is_error') is True


def test_attention_request_limit_fails_when_snapshot_is_invalid(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv('GITHUB_WORKFLOW', shim.ATTENTION_WORKFLOW)
    monkeypatch.setenv('GITHUB_WORKSPACE', str(tmp_path))
    (tmp_path / 'attention-candidates.json').write_text('{}', encoding='utf-8')
    sink = tmp_path / 'outputs.jsonl'
    sink.write_text(json.dumps({'type': 'noop', 'message': 'output exists'}) + '\n', encoding='utf-8')
    monkeypatch.setenv('GH_AW_SAFE_OUTPUTS', str(sink))
    monkeypatch.setattr(shim, 'run_request_limit', lambda: 1)
    emitted: list[dict[str, object]] = []
    monkeypatch.setattr(shim, 'emit', lambda obj: emitted.append(dict(obj)))  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]

    def _respond(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart('probe', {})])

    assert _run_shim(_respond, sink) == 1
    results = [e for e in emitted if e.get('type') == 'result']
    assert len(results) == 1
    assert results[0].get('is_error') is True


def test_attention_request_limit_succeeds_when_decisions_are_complete(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv('GITHUB_WORKFLOW', shim.ATTENTION_WORKFLOW)
    monkeypatch.setenv('GITHUB_WORKSPACE', str(tmp_path))
    snapshot = tmp_path / 'attention-candidates.json'
    snapshot.write_text(json.dumps({'candidates': [{'number': 1}]}), encoding='utf-8')
    sink = tmp_path / 'outputs.jsonl'
    sink.write_text(json.dumps({'type': 'record_attention_decision', 'item_number': '1'}) + '\n', encoding='utf-8')
    monkeypatch.setenv('GH_AW_SAFE_OUTPUTS', str(sink))
    monkeypatch.setattr(shim, 'run_request_limit', lambda: 1)
    emitted: list[dict[str, object]] = []
    monkeypatch.setattr(shim, 'emit', lambda obj: emitted.append(dict(obj)))  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]

    def _respond(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart('noop', {'message': 'complete'})])

    assert _run_shim(_respond, sink) == 0
    results = [e for e in emitted if e.get('type') == 'result']
    assert [(result.get('is_error'), result.get('result')) for result in results] == [
        (False, 'request limit reached after the safe output was emitted')
    ]


def test_attention_completion_accepts_noop_for_empty_snapshot(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv('GITHUB_WORKFLOW', 'Pydantic AI Attention Triage')
    monkeypatch.setenv('GITHUB_WORKSPACE', str(tmp_path))
    (tmp_path / 'attention-candidates.json').write_text(json.dumps({'candidates': []}), encoding='utf-8')
    sink = tmp_path / 'outputs.jsonl'
    sink.write_text(json.dumps({'type': 'noop', 'message': 'nothing to classify'}) + '\n', encoding='utf-8')
    monkeypatch.setenv('GH_AW_SAFE_OUTPUTS', str(sink))

    assert shim.require_safe_output('done') == 'done'


def test_run_succeeds_when_the_safe_output_takes_the_last_request(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    sink = tmp_path / 'outputs.jsonl'
    monkeypatch.setenv('GH_AW_SAFE_OUTPUTS', str(sink))
    monkeypatch.setattr(shim, 'run_request_limit', lambda: 20)
    emitted: list[dict[str, object]] = []
    monkeypatch.setattr(shim, 'emit', lambda obj: emitted.append(dict(obj)))  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    narrated = False

    def _respond(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal narrated
        if shim.REQUEST_BUDGET_NOTICE not in (info.instructions or ''):
            return ModelResponse(parts=[ToolCallPart('probe', {})])
        if not narrated:
            # Spends request 19 of 20 narrating, so the retry's `noop` takes the last one.
            narrated = True
            return ModelResponse(parts=[TextPart('Let me wrap up.')])
        return ModelResponse(parts=[ToolCallPart('noop', {'message': 'out of budget'})])

    assert _run_shim(_respond, sink) == 0
    assert sink.exists()
    results = [e for e in emitted if e.get('type') == 'result']
    assert [(r.get('is_error'), r.get('result')) for r in results] == [
        (False, 'request limit reached after the safe output was emitted')
    ]


def test_run_warns_the_model_when_the_request_budget_is_nearly_spent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    sink = tmp_path / 'outputs.jsonl'
    monkeypatch.setenv('GH_AW_SAFE_OUTPUTS', str(sink))
    monkeypatch.setattr(shim, 'run_request_limit', lambda: 20)
    emitted: list[dict[str, object]] = []
    monkeypatch.setattr(shim, 'emit', lambda obj: emitted.append(dict(obj)))  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    warned: list[bool] = []

    def _respond(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        warned.append(shim.REQUEST_BUDGET_NOTICE in (info.instructions or ''))
        if sink.exists():
            return ModelResponse(parts=[TextPart('done')])
        if warned[-1]:
            return ModelResponse(parts=[ToolCallPart('noop', {'message': 'out of budget'})])
        return ModelResponse(parts=[ToolCallPart('probe', {})])

    # With a limit of 20 the last tenth is 2 requests: one to emit the safe output,
    # one to end the run.
    assert _run_shim(_respond, sink) == 0
    assert warned == [False] * 18 + [True, True]


@pytest.mark.parametrize('inside_reserve', [False, True])
def test_run_warns_the_model_when_the_time_budget_is_nearly_spent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, inside_reserve: bool
):
    sink = tmp_path / 'outputs.jsonl'
    monkeypatch.setenv('GH_AW_SAFE_OUTPUTS', str(sink))
    monkeypatch.setattr(shim, 'emit', lambda obj: None)  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    seconds_left = shim.RUN_TIME_RESERVE_SECS + (-1 if inside_reserve else 60)
    monkeypatch.setattr(shim, '_run_deadline', time.monotonic() + seconds_left)
    warned: list[bool] = []

    def _respond(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        warned.append(shim.TIME_BUDGET_NOTICE in (info.instructions or ''))
        if sink.exists():
            return ModelResponse(parts=[TextPart('done')])
        return ModelResponse(parts=[ToolCallPart('mcp__safeoutputs__noop', {'message': 'posted'})])

    assert _run_shim(_respond, sink, mcp_tool_names=True) == 0
    assert warned == [inside_reserve, inside_reserve]


def test_run_offers_only_safe_output_tools_inside_the_time_reserve(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    sink = tmp_path / 'outputs.jsonl'
    monkeypatch.setenv('GH_AW_SAFE_OUTPUTS', str(sink))
    monkeypatch.setenv('GITHUB_WORKSPACE', str(tmp_path))
    monkeypatch.setattr(shim, 'emit', lambda obj: None)  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    monkeypatch.setattr(shim, '_run_deadline', time.monotonic() + shim.RUN_TIME_RESERVE_SECS + 60)
    offered: list[set[str]] = []
    glob_returns: list[str] = []

    def _respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        offered.append({tool_def.name for tool_def in info.function_tools})
        glob_returns.extend(
            str(part.content)
            for part in messages[-1].parts
            if isinstance(part, ToolReturnPart) and part.tool_name == 'Glob'
        )
        if len(offered) == 1:
            # The reserve starts while this request is in flight.
            monkeypatch.setattr(shim, '_run_deadline', time.monotonic() + shim.RUN_TIME_RESERVE_SECS - 1)
            return ModelResponse(parts=[ToolCallPart('Glob', {'pattern': '*'})])
        if sink.exists():
            return ModelResponse(parts=[TextPart('done')])
        return ModelResponse(parts=[ToolCallPart('mcp__safeoutputs__noop', {'message': 'posted'})])

    (tmp_path / 'module.py').write_text('', encoding='utf-8')
    assert _run_shim(_respond, sink, mcp_tool_names=True) == 0
    assert {'Glob', 'Read', 'Task', 'mcp__safeoutputs__noop'} <= offered[0]
    assert offered[1:] == [{'mcp__safeoutputs__noop', 'mcp__safeoutputs__probe'}] * 2
    # The in-flight request's tool call still ran.
    assert len(glob_returns) == 1
    assert 'module.py' in glob_returns[0]


@pytest.mark.parametrize('parallel_tasks', [1, 2])
def test_run_caps_subagents_so_the_budget_notice_still_fires(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, parallel_tasks: int
):
    sink = tmp_path / 'outputs.jsonl'
    monkeypatch.setenv('GH_AW_SAFE_OUTPUTS', str(sink))
    monkeypatch.setenv('GITHUB_WORKSPACE', str(tmp_path))
    monkeypatch.setattr(shim, 'run_request_limit', lambda: 20)
    monkeypatch.setattr(shim, 'emit', lambda obj: None)  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    sub_requests = 0
    warned: list[bool] = []

    def _respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal sub_requests
        instructions = info.instructions or ''
        if shim.SUBAGENT_INSTRUCTIONS in instructions:
            # A sub-agent that never finishes on its own.
            sub_requests += 1
            return ModelResponse(parts=[ToolCallPart('Glob', {'pattern': '*'})])
        warned.append(shim.REQUEST_BUDGET_NOTICE in instructions)
        if sink.exists():
            return ModelResponse(parts=[TextPart('done')])
        if warned[-1]:
            return ModelResponse(parts=[ToolCallPart('noop', {'message': 'out of budget'})])
        return ModelResponse(
            parts=[
                ToolCallPart('Task', {'description': f'scan {i}', 'prompt': 'scan everything'})
                for i in range(parallel_tasks)
            ]
        )

    # Of the 20 requests, the last 2 are the parent's; the first sub-agent gets what is
    # left after the parent's first one, instead of `SUBAGENT_REQUEST_LIMIT` (75), and a
    # parallel second one is refused rather than granted the same headroom.
    assert _run_shim(_respond, sink) == 0
    assert sub_requests == 17
    assert warned == [False, True, True]
    assert shim._subagent_requests_in_flight == 0  # pyright: ignore[reportPrivateUsage]


def test_task_refuses_to_spawn_inside_the_final_tenth():
    class _Ctx:
        model = None
        usage = RunUsage(requests=18)
        usage_limits = UsageLimits(request_limit=20)

    out = asyncio.run(shim.task(cast(RunContext[None], _Ctx()), 'x', 'y'))
    assert out == 'error: the request budget is nearly spent; finish the task from what you already have'


def test_task_refuses_to_spawn_inside_the_run_time_reserve(monkeypatch: pytest.MonkeyPatch):
    class _Ctx:
        model = None
        usage = RunUsage()
        usage_limits = None

    monkeypatch.setattr(shim, '_run_deadline', time.monotonic() + shim.RUN_TIME_RESERVE_SECS - 1)
    out = asyncio.run(shim.task(cast(RunContext[None], _Ctx()), 'x', 'y'))
    assert out == 'error: the time budget is nearly spent; finish the task from what you already have'


def test_task_cuts_off_a_subagent_before_the_run_time_reserve(monkeypatch: pytest.MonkeyPatch):
    class _HangingAgent:
        def __init__(self, *a: object, **k: object) -> None:
            pass

        async def run(self, *a: object, **k: object) -> None:
            await asyncio.Event().wait()

    class _Ctx:
        model = TestModel()
        usage = RunUsage()
        usage_limits = None

    monkeypatch.setattr(shim, 'Agent', _HangingAgent)
    # The sub-agent gets the 0.01s before the reserve starts, not `SUBAGENT_TIMEOUT_SECS`.
    monkeypatch.setattr(shim, '_run_seconds_left', lambda: shim.RUN_TIME_RESERVE_SECS + 0.01)
    out = asyncio.run(shim.task(cast(RunContext[None], _Ctx()), 'x', 'y'))
    assert out == 'error: sub-agent timed out after 1min'


def _cut_delegate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    turns: list[tuple[str, str]],
    resume: Callable[[list[ModelMessage], AgentInfo], ModelResponse],
) -> FunctionModel:
    """A delegate model that plays `turns`, then answers its resume request with `resume`.

    Each turn is `('read', path)`, a `Read` call, or `('cut', partial_text)`, which streams
    `partial_text` and then times out. A read of `cut.py` times out while the tool runs. Raising
    `TimeoutError` where the cut lands takes `task`'s timeout path without expiring its wall-clock cap;
    the run records its partial state as it does for the cap's cancellation, which
    `test_subagent_is_resumed_after_its_cap_cuts_it_off` covers.
    """
    monkeypatch.setenv('GITHUB_WORKSPACE', str(tmp_path))
    tools: FunctionToolset[object] = FunctionToolset()

    @tools.tool_plain(name='Read')
    async def _read(file_path: str) -> str:
        if file_path == 'cut.py':
            raise TimeoutError
        return f'source of {file_path}'

    def _select_tools(
        _allowed: frozenset[str] | None, permission_mode: str | None, *, task: shim.TaskCallable | None
    ) -> AbstractToolset[object]:
        return tools

    monkeypatch.setattr(shim, 'select_claude_code_toolset', _select_tools)

    async def _stream(messages: list[ModelMessage], info: AgentInfo):
        # The resume request is the only one made without tools.
        if not info.function_tools:
            for part in resume(messages, info).parts:
                assert isinstance(part, TextPart)
                yield part.content
            return
        kind, value = turns.pop(0)
        if kind == 'read':
            yield {0: DeltaToolCall(name='Read', json_args=json.dumps({'file_path': value}))}
            return
        if value:
            yield value
        raise TimeoutError

    return FunctionModel(stream_function=_stream)


def _resumed_report(report: str) -> str:
    return (
        'note: the sub-agent timed out after 15min and wrote this report afterwards from what it had '
        f'already found, without further tool calls.\n\n{report}'
    )


def test_task_resumes_a_timed_out_subagent_from_its_history(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    resume_requests: list[tuple[list[ModelMessage], AgentInfo]] = []

    def _resume(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        resume_requests.append((messages, info))
        return ModelResponse(parts=[TextPart('Reviewed files: a.py\nUnreviewed files: b.py')])

    model = _cut_delegate(
        monkeypatch, tmp_path, [('read', 'a.py'), ('cut', 'No findings in a.py. Next, b.py')], _resume
    )
    parent_usage = RunUsage()
    ctx = RunContext[object](deps=object(), model=model, usage=parent_usage, usage_limits=None)
    out = asyncio.run(shim.task(ctx, 'review a.py and b.py', 'Review a.py and b.py.'))

    assert out == _resumed_report('Reviewed files: a.py\nUnreviewed files: b.py')
    [(messages, info)] = resume_requests
    assert info.function_tools == []
    assert info.instructions is not None and 'Review a.py and b.py.' in info.instructions
    parts = [part for message in messages for part in message.parts]
    assert [part.content for part in parts if isinstance(part, ToolReturnPart)] == ['source of a.py']
    # The response the cut left partial is not part of the resumed history.
    assert not any(isinstance(part, TextPart) for part in parts)
    assert isinstance(parts[-1], UserPromptPart) and parts[-1].content == shim.SUBAGENT_RESUME_PROMPT
    # The delegate's two requests, including the one cut off, and the resume request.
    assert parent_usage.requests == 3


def test_task_resume_closes_out_tool_calls_the_timeout_left_unanswered(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    resume_requests: list[list[ModelMessage]] = []

    def _resume(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        resume_requests.append(messages)
        return ModelResponse(parts=[TextPart('Reviewed files: a.py\nUnreviewed files: cut.py')])

    model = _cut_delegate(monkeypatch, tmp_path, [('read', 'a.py'), ('read', 'cut.py')], _resume)
    ctx = RunContext[object](deps=object(), model=model, usage=RunUsage(), usage_limits=None)
    out = asyncio.run(shim.task(ctx, 'review a.py and cut.py', 'Review a.py and cut.py.'))

    assert out == _resumed_report('Reviewed files: a.py\nUnreviewed files: cut.py')
    [messages] = resume_requests
    returns = {
        part.tool_call_id: part.content
        for message in messages
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    }
    calls = {
        part.tool_call_id: part.args_as_dict()['file_path']
        for message in messages
        for part in message.parts
        if isinstance(part, ToolCallPart)
    }
    assert {calls[call_id]: content for call_id, content in returns.items()} == {
        'a.py': 'source of a.py',
        'cut.py': INTERRUPTED_TOOL_RETURN_CONTENT,
    }


@pytest.mark.parametrize('blocked_by', ['time', 'requests', 'nothing to resume'])
def test_task_returns_the_timeout_error_when_no_resume_is_admissible(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, blocked_by: str
):
    turns = [('read', 'a.py'), ('cut', 'No findings in a.py.')]
    usage_limits = None
    if blocked_by == 'time':
        # Read once when the delegate starts, and once when the cut lands inside the parent's time reserve.
        seconds_left = iter([shim.RUN_TIME_RESERVE_SECS + shim.SUBAGENT_TIMEOUT_SECS, shim.RUN_TIME_RESERVE_SECS])
        monkeypatch.setattr(shim, '_run_seconds_left', lambda: next(seconds_left))
    elif blocked_by == 'requests':
        # The delegate's grant of 3 requests is spent: 2 complete and the one cut off.
        turns.insert(0, ('read', 'b.py'))
        usage_limits = UsageLimits(request_limit=20)
    else:
        turns = [('cut', '')]
    resume_requests: list[list[ModelMessage]] = []

    def _resume(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        resume_requests.append(messages)
        return ModelResponse(parts=[TextPart('report')])

    model = _cut_delegate(monkeypatch, tmp_path, turns, _resume)
    ctx = RunContext[object](deps=object(), model=model, usage=RunUsage(requests=15), usage_limits=usage_limits)
    out = asyncio.run(shim.task(ctx, 'review a.py', 'Review a.py.'))
    assert out == 'error: sub-agent timed out after 15min'
    assert turns == []
    assert resume_requests == []


def test_task_returns_the_timeout_error_when_the_resume_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    resume_requests: list[list[ModelMessage]] = []

    def _resume(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        resume_requests.append(messages)
        raise RuntimeError('provider unavailable')

    model = _cut_delegate(monkeypatch, tmp_path, [('read', 'a.py'), ('cut', '')], _resume)
    ctx = RunContext[object](deps=object(), model=model, usage=RunUsage(), usage_limits=None)
    out = asyncio.run(shim.task(ctx, 'review a.py', 'Review a.py.'))
    assert out == 'error: sub-agent timed out after 15min'
    assert len(resume_requests) == 1


def test_read_only_subagent_tools_are_non_mutating_and_exclude_task():
    assert pkg.READ_ONLY_SUBAGENT_TOOLS.isdisjoint(pkg.MUTATING_TOOLS)
    assert 'Task' not in pkg.READ_ONLY_SUBAGENT_TOOLS  # no recursion
    # All entries are real Claude Code tool names.
    assert pkg.READ_ONLY_SUBAGENT_TOOLS <= set(pkg.CLAUDE_CODE_TOOL_NAMES)


def test_task_registered_via_build_claude_code_toolset():
    """`Task` isn't part of the static `CLAUDE_CODE_TOOL_NAMES` tuple — it gets
    appended dynamically by `build_claude_code_toolset(task=...)` only for the
    parent (sub-agents pass `task=None` so they can't recurse).
    """
    parent_names = asyncio.run(_toolset_names(None, None, task=shim.task))
    sub_names = asyncio.run(_toolset_names(None, None, task=None))
    assert 'Task' in parent_names
    assert 'Task' not in sub_names


def test_subagent_without_parent_budget_is_limited_to_configured_default(monkeypatch: pytest.MonkeyPatch):
    """A delegate without inherited headroom gets the configured 150-request cap, merged once."""
    calls = 0

    async def _stream(_messages: list[ModelMessage], _info: AgentInfo):
        nonlocal calls
        calls += 1
        yield {0: DeltaToolCall(name='Glob', json_args='{"pattern": "*"}')}

    monkeypatch.setenv('GITHUB_WORKSPACE', '.')
    parent_usage = RunUsage()

    ctx = RunContext[object](
        deps=object(),
        model=FunctionModel(stream_function=_stream),
        usage=parent_usage,
        usage_limits=None,
    )
    out = asyncio.run(shim.task(ctx, 'exhaust budget', 'keep searching'))
    assert out.startswith('error: sub-agent failed:')
    assert calls == 150
    assert parent_usage.requests == 150


@pytest.mark.parametrize('disable_task', [False, True])
def test_main_can_disable_task(disable_task: bool, monkeypatch: pytest.MonkeyPatch):
    selected_tasks: list[shim.TaskCallable | None] = []

    if disable_task:
        monkeypatch.setenv('GITHUB_WORKFLOW', 'Pydantic AI Attention Triage')
    else:
        monkeypatch.setenv('GITHUB_WORKFLOW', 'Other Pydantic AI workflow')
    monkeypatch.setattr(sys, 'argv', ['pydantic-ai-runner', '--print', 'test prompt'])
    monkeypatch.setattr(shim, 'configure_observability', lambda: None)

    def build_test_model(_args: shim.Args) -> tuple[_Model[Any], str]:
        return cast(_Model[Any], object()), 'test-model'

    monkeypatch.setattr(shim, 'build_model', build_test_model)

    def capture_task(
        _allowed: frozenset[str] | None,
        _permission_mode: str | None,
        *,
        task: shim.TaskCallable | None,
    ) -> AbstractToolset[object]:
        selected_tasks.append(task)
        raise RuntimeError('stop after task selection')

    monkeypatch.setattr(shim, 'select_claude_code_toolset', capture_task)
    with redirect_stdout(io.StringIO()):
        assert shim.main() == 1

    assert selected_tasks == [None if disable_task else shim.task]


def test_attention_workflow_uses_direct_classification():
    workflow = Path(__file__).parent.parent / 'workflows' / 'pydantic-ai-attention-triage.md'
    text = workflow.read_text(encoding='utf-8')

    assert 'Classify every candidate yourself' in text
    assert 'PYDANTIC_AI_DYNAMIC_WORKFLOW' not in text
    assert 'run_workflow' not in text


def test_attention_workflow_fetches_tags_for_runner_version():
    workflow_dir = Path(__file__).parent.parent / 'workflows'
    source = workflow_dir / 'pydantic-ai-attention-triage.md'
    source_steps = agentic_workflow_guard.parse_frontmatter(source)['pre-agent-steps']
    compiled = yaml.safe_load((workflow_dir / 'pydantic-ai-attention-triage.lock.yml').read_text(encoding='utf-8'))
    compiled_steps = compiled['jobs']['agent']['steps']
    expected_checkout_config = {
        'repository': '${{ job.workflow_repository }}',
        'ref': '${{ job.workflow_sha }}',
        'persist-credentials': False,
        'fetch-depth': 0,
    }

    for steps in (source_steps, compiled_steps):
        checkout_index, checkout = next(
            (index, step)
            for index, step in enumerate(steps)
            if str(step.get('uses', '')).startswith('actions/checkout@')
            and step.get('with', {}).get('ref') == expected_checkout_config['ref']
        )
        prewarm_index = next(
            index
            for index, step in enumerate(steps)
            if step.get('name') == 'Pre-warm Pydantic AI gh-aw shim uv environment'
        )
        assert checkout_index < prewarm_index
        assert checkout['with'] == expected_checkout_config


def test_runner_drops_dynamic_workflow_dependencies():
    runner = (Path(__file__).parent / 'pydantic-ai-runner').read_text(encoding='utf-8')
    assert 'dynamic-workflow' not in runner
    assert 'pydantic-monty' not in runner


def test_runner_resolves_pydantic_ai_from_the_workspace():
    """The shim's code is this checkout, so its libraries must be too — see #6998, #7103."""
    runner = (Path(__file__).parent / 'pydantic-ai-runner').read_text(encoding='utf-8')
    assert '# [tool.uv.sources]' in runner
    assert '# pydantic-ai-slim = { path = "../../pydantic_ai_slim", editable = true }' in runner
    assert '# pydantic-ai-harness = { path = "../../src/pydantic_ai_harness", editable = true }' in runner

    lock = (Path(__file__).parent / 'pydantic-ai-runner.lock').read_text(encoding='utf-8')
    assert 'source = { editable = "../../pydantic_ai_slim" }' in lock
    assert 'source = { editable = "../../src/pydantic_ai_harness" }' in lock


def test_compiled_workflows_pin_retry_policy():
    workflow_dir = Path(__file__).parent.parent / 'workflows'
    for compiled_workflow in workflow_dir.glob('pydantic-ai-*.lock.yml'):
        compiled_text = compiled_workflow.read_text(encoding='utf-8')
        assert compiled_text.count('GH_AW_HARNESS_MAX_RETRIES: 0') == 1
        assert compiled_text.count('GH_AW_HARNESS_MAX_RETRIES: 3') == 1


def test_task_runs_subagent_with_run_model_and_read_only_tools(monkeypatch: pytest.MonkeyPatch):
    # The Task tool spawns a sub-Agent on ctx.model with the read-only tool
    # set, runs the given prompt, and returns the sub-agent's output.
    seen_instructions: list[str] = []
    received_messages: list[ModelMessage] = []
    received_tool_names: set[str] = set()

    def _capture(messages: list[ModelMessage], info: AgentInfo) -> None:
        seen_instructions.append(info.instructions or '')
        received_messages.extend(messages)
        received_tool_names.update(td.name for td in info.function_tools)

    def _respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        _capture(messages, info)
        return ModelResponse(parts=[TextPart('SUB: investigated')])

    async def _stream(messages: list[ModelMessage], info: AgentInfo):
        _capture(messages, info)
        yield 'SUB: investigated'

    # The original bug: sharing `ctx.usage` made `SUBAGENT_REQUEST_LIMIT` fire
    # immediately past parent's 75th request. Sub-agent should run regardless.
    parent_usage = RunUsage(requests=100, input_tokens=20_000, output_tokens=10_000)

    class _Ctx:
        model = FunctionModel(_respond, stream_function=_stream)
        usage = parent_usage
        usage_limits = None

    out = asyncio.run(shim.task(cast(RunContext[None], _Ctx()), 'scan models/openai.py', 'find tool_call_id bugs'))
    assert out == 'SUB: investigated'

    instructions = seen_instructions[0]
    user_text = '\n'.join(str(p.content) for m in received_messages for p in m.parts if isinstance(p, UserPromptPart))
    assert shim.INSTRUCTIONS in instructions
    assert shim.SUBAGENT_INSTRUCTIONS in instructions
    assert 'find tool_call_id bugs' in instructions
    assert user_text == shim.RUN_TRIGGER
    assert 'find tool_call_id bugs' not in user_text

    assert received_tool_names == set(pkg.READ_ONLY_SUBAGENT_TOOLS)
    assert 'Task' not in received_tool_names
    assert 'Bash' not in received_tool_names

    # Sub-agent's cost rolls up to the parent without making the parent's
    # request total trip the sub-agent's request_limit.
    assert parent_usage.requests > 100


# --------------------------------------------------------------------------- #
# directory-scoped AGENTS.md / CLAUDE.md auto-loading
# --------------------------------------------------------------------------- #
def test_attach_context_surfaces_files_once_per_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    # AGENTS.md at root of the "workspace" + CLAUDE.md in a subdir.
    monkeypatch.setenv('GITHUB_WORKSPACE', str(tmp_path))
    (tmp_path / 'AGENTS.md').write_text('# repo conventions', encoding='utf-8')
    sub = tmp_path / 'pkg'
    sub.mkdir()
    (sub / 'CLAUDE.md').write_text('# pkg conventions', encoding='utf-8')
    (sub / 'code.py').write_text('x = 1\n', encoding='utf-8')
    shared.reset_context_state()

    first = shared.attach_context('pkg/code.py')
    assert 'context: pkg/CLAUDE.md' in first  # nearest first when walking up
    assert 'context: AGENTS.md' in first
    assert 'pkg conventions' in first and 'repo conventions' in first

    # Subsequent calls in same run dedupe.
    again = shared.attach_context('pkg/code.py')
    assert again == ''

    # A different path under the same dir hits no new context files.
    (sub / 'other.py').write_text('', encoding='utf-8')
    assert shared.attach_context('pkg/other.py') == ''


def test_attach_context_truncates_large_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv('GITHUB_WORKSPACE', str(tmp_path))
    big = 'X' * (shared.MAX_CONTEXT_FILE_CHARS + 5000)
    (tmp_path / 'AGENTS.md').write_text(big, encoding='utf-8')
    shared.reset_context_state()
    out = shared.attach_context('.')
    # Body of the AGENTS.md block is capped to MAX_CONTEXT_FILE_CHARS.
    body = out.split('---\n', 2)[-1]
    assert len(body) <= shared.MAX_CONTEXT_FILE_CHARS + 50  # +slack for trailing markers


def test_attach_context_empty_for_missing_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv('GITHUB_WORKSPACE', str(tmp_path))
    shared.reset_context_state()
    assert shared.attach_context(None) == ''
    assert shared.attach_context('does-not-exist.py') == ''  # parent has no AGENTS.md/CLAUDE.md


def test_read_file_prepends_context(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv('GITHUB_WORKSPACE', str(tmp_path))
    (tmp_path / 'AGENTS.md').write_text('repo rules', encoding='utf-8')
    (tmp_path / 'f.txt').write_text('file body', encoding='utf-8')
    shared.reset_context_state()
    out = asyncio.run(pkg.read_file(_ctx(), 'f.txt'))
    assert 'context: AGENTS.md' in out and 'repo rules' in out and 'file body' in out


# --------------------------------------------------------------------------- #
# history compaction (ProcessHistory capability)
# --------------------------------------------------------------------------- #
def test_compaction_thresholds_are_sane():
    # ~100k tokens at 4 chars/tok = half a 200k-token window. The trigger
    # is hardcoded (no per-knob multiplication) — fewer dials.
    assert shim.COMPACTION_TRIGGER_CHARS == 400_000
    assert shim.COMPACTION_KEEP_RECENT >= 4
    assert shim.TOOL_RESULT_TRIM_THRESHOLD > shim.TOOL_RESULT_HEAD_TAIL_CHARS * 2


def test_history_size_chars_sums_all_part_content():
    msgs: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='hello')]),  # 5
        ModelRequest(parts=[UserPromptPart(content='x' * 20)]),  # 20
    ]
    assert shim._history_size_chars(msgs) == 25  # pyright: ignore[reportPrivateUsage]


def test_compact_history_no_op_below_char_budget(monkeypatch: pytest.MonkeyPatch):
    # Many tiny messages — total chars stays well below the default 80k budget.
    msgs: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content=f'm{i}')]) for i in range(100)]

    class _Ctx:
        model = None

    out = asyncio.run(shim._compact_history(cast(RunContext[None], _Ctx()), msgs))  # pyright: ignore[reportPrivateUsage]
    assert out is msgs  # size-based: count alone never triggers


def test_compact_history_summarises_with_fresh_usage_then_merges():
    """Summariser uses a fresh `RunUsage` (so request_limit doesn't trip on the
    parent's running total) and the parent usage absorbs its cost after."""
    big = 'x' * 50_000
    msgs: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content=f'm{i} {big}')]) for i in range(13)]

    def _respond(_messages: list[ModelMessage], _info: object) -> ModelResponse:
        return ModelResponse(parts=[TextPart('SHORT SUMMARY')])

    async def _stream(_messages: list[ModelMessage], _info: object):
        yield 'SHORT SUMMARY'

    # Parent has already done 50 requests; the old shared-usage bug would
    # trip `UsageLimits(request_limit=2)` immediately. With the fix it runs.
    parent_usage = RunUsage(requests=50)

    class _Ctx:
        model = FunctionModel(_respond, stream_function=_stream)
        usage = parent_usage

    out = asyncio.run(shim._compact_history(cast(RunContext[None], _Ctx()), msgs))  # pyright: ignore[reportPrivateUsage]
    assert len(out) == 1 + shim.COMPACTION_KEEP_RECENT
    assert parent_usage.requests > 50  # summariser's cost merged into parent
    summary_part = out[0].parts[0]
    assert isinstance(summary_part, UserPromptPart)
    assert 'SHORT SUMMARY' in str(summary_part.content)


def test_trim_dedupes_superseded_reads_and_truncates_large_results():
    """The cheap pre-pass should rewrite older tool results without invoking
    the LLM: superseded `Read` returns become a one-line marker, oversized
    returns are head/tail-truncated, and the last KEEP_RECENT messages are
    left untouched."""
    big = 'X' * 20_000
    # Three Read calls for the same file: only the last is current; the first
    # two should be marked superseded.
    msgs: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='start')]),
        ModelResponse(parts=[ToolCallPart(tool_name='Read', args={'file_path': 'a.py'}, tool_call_id='r1')]),
        ModelRequest(parts=[ToolReturnPart(tool_name='Read', content=big, tool_call_id='r1')]),
        ModelResponse(parts=[ToolCallPart(tool_name='Read', args={'file_path': 'a.py'}, tool_call_id='r2')]),
        ModelRequest(parts=[ToolReturnPart(tool_name='Read', content=big, tool_call_id='r2')]),
        # An unrelated big Bash result that should get head/tail trimmed.
        ModelResponse(parts=[ToolCallPart(tool_name='Bash', args={'command': 'ls -la'}, tool_call_id='b1')]),
        ModelRequest(parts=[ToolReturnPart(tool_name='Bash', content=big, tool_call_id='b1')]),
        ModelResponse(parts=[ToolCallPart(tool_name='Read', args={'file_path': 'a.py'}, tool_call_id='r3')]),
    ]
    # Pad to exceed KEEP_RECENT so the older entries are eligible for trimming.
    for i in range(shim.COMPACTION_KEEP_RECENT):
        msgs.append(ModelRequest(parts=[UserPromptPart(content=f'tail{i}')]))

    out = shim._trim_tool_results(msgs)  # pyright: ignore[reportPrivateUsage]
    assert len(out) == len(msgs)

    # r1 (superseded by r2 and r3) → marker.
    r1_return = out[2].parts[0]
    assert isinstance(r1_return, ToolReturnPart)
    assert 'superseded read' in str(r1_return.content) and 'a.py' in str(r1_return.content)

    # r2 (superseded by r3) → also marker.
    r2_return = out[4].parts[0]
    assert isinstance(r2_return, ToolReturnPart)
    assert 'superseded read' in str(r2_return.content)

    # Bash result → head/tail truncated, not dedup-marked.
    bash_return = out[6].parts[0]
    assert isinstance(bash_return, ToolReturnPart)
    bash_content = str(bash_return.content)
    assert 'trimmed' in bash_content
    assert len(bash_content) < len(big)

    # Tail (last KEEP_RECENT) untouched and is the same object identity.
    for i in range(shim.COMPACTION_KEEP_RECENT):
        assert out[-(i + 1)] is msgs[-(i + 1)]


def test_trim_preserves_distinct_read_slices_of_same_file():
    """A `Read` with `offset=N, limit=M` returns different content than a
    `Read` of the same file with no slice (or a different slice). The
    dedup key is the full `(file_path, offset, limit)` tuple, so distinct
    slices stay distinct — only an exact-args re-read is superseded."""
    big = 'Y' * 20_000
    msgs: list[ModelMessage] = [
        # Slice 1 of foo.py — distinct content.
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name='Read', args={'file_path': 'foo.py', 'offset': 1, 'limit': 100}, tool_call_id='s1'
                )
            ]
        ),
        ModelRequest(parts=[ToolReturnPart(tool_name='Read', content=big, tool_call_id='s1')]),
        # Different slice — must NOT be deduped against s1.
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name='Read', args={'file_path': 'foo.py', 'offset': 500, 'limit': 100}, tool_call_id='s2'
                )
            ]
        ),
        ModelRequest(parts=[ToolReturnPart(tool_name='Read', content=big, tool_call_id='s2')]),
        # Same args as s1 — supersedes it.
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name='Read', args={'file_path': 'foo.py', 'offset': 1, 'limit': 100}, tool_call_id='s3'
                )
            ]
        ),
        ModelRequest(parts=[ToolReturnPart(tool_name='Read', content=big, tool_call_id='s3')]),
    ]
    for i in range(shim.COMPACTION_KEEP_RECENT):
        msgs.append(ModelRequest(parts=[UserPromptPart(content=f't{i}')]))

    out = shim._trim_tool_results(msgs)  # pyright: ignore[reportPrivateUsage]

    # s1 (superseded by s3 — same args) → marker that mentions the slice args.
    s1_return = out[1].parts[0]
    assert isinstance(s1_return, ToolReturnPart)
    s1_content = str(s1_return.content)
    assert 'superseded read' in s1_content and 'foo.py' in s1_content and 'offset=1' in s1_content

    # s2 (different slice) is oversized so it gets head/tail-truncated but
    # NOT marked superseded — its content is genuinely distinct.
    s2_return = out[3].parts[0]
    assert isinstance(s2_return, ToolReturnPart)
    s2_content = str(s2_return.content)
    assert 'superseded' not in s2_content
    assert 'trimmed' in s2_content


def test_trim_logs_substitution_counts_only_when_changes_fired(caplog: LogCaptureFixture):
    """Trim logs once when it substitutes; silent on a no-op pass."""
    tiny_msgs: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content=f'm{i}')]) for i in range(shim.COMPACTION_KEEP_RECENT + 5)
    ]
    with caplog.at_level('INFO', logger='pydantic_ai_gh_aw_shim'):
        shim._trim_tool_results(tiny_msgs)  # pyright: ignore[reportPrivateUsage]
    assert not any('trim: deduped' in m for m in caplog.messages)
    caplog.clear()

    big = 'Z' * 20_000
    msgs: list[ModelMessage] = [
        ModelResponse(parts=[ToolCallPart(tool_name='Read', args={'file_path': 'x.py'}, tool_call_id='r1')]),
        ModelRequest(parts=[ToolReturnPart(tool_name='Read', content=big, tool_call_id='r1')]),
        ModelResponse(parts=[ToolCallPart(tool_name='Read', args={'file_path': 'x.py'}, tool_call_id='r2')]),
        ModelRequest(parts=[ToolReturnPart(tool_name='Read', content=big, tool_call_id='r2')]),
        ModelResponse(parts=[ToolCallPart(tool_name='Bash', args={'command': 'ls'}, tool_call_id='b1')]),
        ModelRequest(parts=[ToolReturnPart(tool_name='Bash', content=big, tool_call_id='b1')]),
    ]
    for i in range(shim.COMPACTION_KEEP_RECENT):
        msgs.append(ModelRequest(parts=[UserPromptPart(content=f't{i}')]))
    with caplog.at_level('INFO', logger='pydantic_ai_gh_aw_shim'):
        shim._trim_tool_results(msgs)  # pyright: ignore[reportPrivateUsage]
    log_line = next((m for m in caplog.messages if 'trim: deduped' in m), None)
    assert log_line is not None
    assert 'deduped 1' in log_line and 'truncated 2' in log_line and 'saved' in log_line


def test_compact_history_uses_trim_alone_when_sufficient(monkeypatch: pytest.MonkeyPatch):
    """Trim alone is enough — the LLM summariser must not fire."""
    # 13 messages: a couple of huge superseded reads, then KEEP_RECENT trivial
    # tail messages. The dedup pass should crush the size.
    big = 'Y' * 60_000
    msgs: list[ModelMessage] = [
        ModelResponse(parts=[ToolCallPart(tool_name='Read', args={'file_path': 'big.py'}, tool_call_id='c1')]),
        ModelRequest(parts=[ToolReturnPart(tool_name='Read', content=big, tool_call_id='c1')]),
        ModelResponse(parts=[ToolCallPart(tool_name='Read', args={'file_path': 'big.py'}, tool_call_id='c2')]),
    ]
    for i in range(shim.COMPACTION_KEEP_RECENT):
        msgs.append(ModelRequest(parts=[UserPromptPart(content=f't{i}')]))

    def _fail_agent_ctor(*_a: object, **_kw: object) -> None:
        raise AssertionError('summariser should not be invoked when trim is enough')

    monkeypatch.setattr(shim, 'Agent', _fail_agent_ctor)

    class _Ctx:
        model = None

    out = asyncio.run(shim._compact_history(cast(RunContext[None], _Ctx()), msgs))  # pyright: ignore[reportPrivateUsage]
    assert len(out) == len(msgs)
    # The first read result is now a marker, not the original payload.
    first_return = out[1].parts[0]
    assert isinstance(first_return, ToolReturnPart)
    assert 'superseded read' in str(first_return.content)


def test_compact_history_falls_back_to_truncation_on_failure(monkeypatch: pytest.MonkeyPatch):
    # Same size-driven setup as the previous test — 13 big msgs > trigger.
    big = 'x' * 50_000
    msgs: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content=f'm{i} {big}')]) for i in range(13)]

    class _FailingAgent:
        def __init__(self, *a: object, **k: object) -> None:
            pass

        async def run(self, *a: object, **k: object) -> None:
            raise RuntimeError('boom')

    monkeypatch.setattr(shim, 'Agent', _FailingAgent)

    class _Ctx:
        model = TestModel()
        usage = RunUsage()

    out = asyncio.run(shim._compact_history(cast(RunContext[None], _Ctx()), msgs))  # pyright: ignore[reportPrivateUsage]
    # On failure: keep just the tail (no head, no synthetic summary).
    assert len(out) == shim.COMPACTION_KEEP_RECENT


def test_compact_history_preserves_prior_synthetic_on_fallback(monkeypatch: pytest.MonkeyPatch):
    """A second compaction round whose summary fails (or doesn't fit) must
    keep the earlier round's `[compacted history]` block. Dropping it would
    silently forget the entire run's prior work."""
    big = 'x' * 50_000
    prior_synthetic = ModelRequest(parts=[UserPromptPart(content='[compacted history]\nearlier summary')])
    msgs: list[ModelMessage] = [
        prior_synthetic,
        *(ModelRequest(parts=[UserPromptPart(content=f'm{i} {big}')]) for i in range(12)),
    ]

    class _FailingAgent:
        def __init__(self, *a: object, **k: object) -> None:
            pass

        async def run(self, *a: object, **k: object) -> None:
            raise RuntimeError('boom')

    monkeypatch.setattr(shim, 'Agent', _FailingAgent)

    class _Ctx:
        model = TestModel()
        usage = RunUsage()

    out = asyncio.run(shim._compact_history(cast(RunContext[None], _Ctx()), msgs))  # pyright: ignore[reportPrivateUsage]
    # First element is the preserved prior synthetic; rest is the tail.
    assert len(out) == 1 + shim.COMPACTION_KEEP_RECENT
    assert out[0] is prior_synthetic


def test_task_surfaces_subagent_failure_as_tool_result(monkeypatch: pytest.MonkeyPatch):
    class _FailingAgent:
        def __init__(self, *a: object, **k: object) -> None:
            pass

        async def run(self, *a: object, **k: object) -> None:
            raise RuntimeError('downstream model exploded')

    monkeypatch.setattr(shim, 'Agent', _FailingAgent)

    class _Ctx:
        model = TestModel()
        usage = RunUsage()
        usage_limits = None

    out = asyncio.run(shim.task(cast(RunContext[None], _Ctx()), 'x', 'y'))
    assert out == 'error: sub-agent failed: downstream model exploded'


def test_task_isolates_attach_context_dedupe_set_from_parent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Sub-agents start with a fresh AGENTS.md seen-set, not the parent's."""
    monkeypatch.setenv('GITHUB_WORKSPACE', str(tmp_path))
    (tmp_path / 'AGENTS.md').write_text('# parent-touched guidance', encoding='utf-8')
    (tmp_path / 'f.txt').write_text('parent file', encoding='utf-8')

    # Parent reads f.txt, which marks AGENTS.md as seen in the parent's set.
    shared.reset_context_state()
    parent_first = shared.attach_context('f.txt')
    assert 'AGENTS.md' in parent_first

    def _respond(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('SUB: done')])

    async def _stream(_messages: list[ModelMessage], _info: AgentInfo):
        yield 'SUB: done'

    class _Ctx:
        model = FunctionModel(_respond, stream_function=_stream)
        usage = RunUsage()
        usage_limits = None

    asyncio.run(shim.task(cast(RunContext[None], _Ctx()), 'sub', 'work'))

    # After the sub-agent ran, the parent's seen set still includes AGENTS.md
    # (sub-agent's reset only affected its own context branch). If we re-call
    # attach_context in the *parent's* context, it should still be deduped.
    assert shared.attach_context('f.txt') == ''


# --------------------------------------------------------------------------- #
# live tool-call stream-json emission (`_stream_events`)
# --------------------------------------------------------------------------- #
def test_stream_events_emits_tool_use_and_tool_result_lines():
    """`_stream_events` is the live emitter that turns pydantic-ai events into
    Claude-shape stream-json on stdout — the surface gh-aw's log parser
    reads. Drive it with synthetic events and assert the wire shape."""

    async def _events():
        yield FunctionToolCallEvent(
            part=ToolCallPart(tool_name='Bash', args={'command': 'ls'}, tool_call_id='c1'),
        )
        yield FunctionToolResultEvent(
            part=ToolReturnPart(tool_name='Bash', content='exit=0\nfile1\nfile2', tool_call_id='c1'),
        )

    buf = io.StringIO()
    with redirect_stdout(buf):
        # `_stream_events` discards its `ctx` arg; passing None keeps the test
        # free of pydantic-ai RunContext construction noise.
        asyncio.run(shim._stream_events(cast(RunContext[None], None), _events()))  # pyright: ignore[reportPrivateUsage]

    lines = [json.loads(x) for x in buf.getvalue().splitlines() if x.strip()]
    assert len(lines) == 2

    use_block = lines[0]
    assert use_block['type'] == 'assistant'
    use_content = use_block['message']['content'][0]
    assert use_content == {'type': 'tool_use', 'id': 'c1', 'name': 'Bash', 'input': {'command': 'ls'}}

    result_block = lines[1]
    assert result_block['type'] == 'user'
    result_content = result_block['message']['content'][0]
    assert result_content['type'] == 'tool_result'
    assert result_content['tool_use_id'] == 'c1'
    assert result_content['content'].startswith('exit=0')


def test_stream_events_truncates_long_tool_results():
    """Result content over `MAX_LIVE_TOOL_RESULT_CHARS` is truncated for the
    stream-json view (the model's view is unaffected — this handler is
    observation-only)."""
    huge = 'A' * 5000

    async def _events():
        yield FunctionToolResultEvent(
            part=ToolReturnPart(tool_name='Bash', content=huge, tool_call_id='c1'),
        )

    buf = io.StringIO()
    with redirect_stdout(buf):
        asyncio.run(shim._stream_events(cast(RunContext[None], None), _events()))  # pyright: ignore[reportPrivateUsage]

    line = json.loads(buf.getvalue().strip())
    emitted = line['message']['content'][0]['content']
    assert len(emitted) < len(huge)
    assert '…[+' in emitted and 'chars]' in emitted


def test_stream_events_tags_retried_result_as_error():
    """A `ToolResultEvent` carrying `outcome='retried'` means the call has to be
    made again — gh-aw must see `is_error=True` so it doesn't read it as a
    successful result."""

    async def _events():
        yield FunctionToolResultEvent(
            part=ToolReturnPart(tool_name='Bash', content='ok', tool_call_id='c1'),
        )
        yield FunctionToolResultEvent(
            part=ToolReturnPart(tool_name='Bash', content='Validation failed', tool_call_id='c2', outcome='retried'),
        )

    buf = io.StringIO()
    with redirect_stdout(buf):
        asyncio.run(shim._stream_events(cast(RunContext[None], None), _events()))  # pyright: ignore[reportPrivateUsage]

    lines = [json.loads(x) for x in buf.getvalue().splitlines() if x.strip()]
    assert lines[0]['message']['content'][0]['is_error'] is False
    assert lines[1]['message']['content'][0]['is_error'] is True


# --------------------------------------------------------------------------- #
# disk / IO failure paths for the file tools
# --------------------------------------------------------------------------- #
def test_read_missing_file_returns_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('GITHUB_WORKSPACE', str(tmp_path))
    out = asyncio.run(pkg.read_file(_ctx(), str(tmp_path / 'nope.txt')))
    assert isinstance(out, str) and out.startswith('error:')


def test_edit_missing_file_returns_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('GITHUB_WORKSPACE', str(tmp_path))
    out = asyncio.run(pkg.edit_file(_ctx(), str(tmp_path / 'missing.txt'), 'old', 'new'))
    assert out.startswith('error:')


def test_multi_edit_missing_file_returns_error(tmp_path: Path):
    out = pkg.multi_edit(str(tmp_path / 'absent.txt'), [{'old_string': 'a', 'new_string': 'b'}])
    assert out.startswith('error:')


def test_list_dir_missing_path_returns_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('GITHUB_WORKSPACE', str(tmp_path))
    out = asyncio.run(pkg.list_dir(_ctx(), str(tmp_path / 'nope')))
    assert out.startswith('error:')


def test_write_to_existing_parent_succeeds_otherwise_creates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # The harness `write_file` requires an existing parent; the Write adapter
    # calls `create_directory` first, so nested writes still succeed.
    monkeypatch.setenv('GITHUB_WORKSPACE', str(tmp_path))
    nested = tmp_path / 'a' / 'b' / 'c.txt'
    assert 'Wrote' in asyncio.run(pkg.write_file(_ctx(), str(nested), 'ok'))
    assert nested.read_text(encoding='utf-8') == 'ok'


def test_write_under_a_file_path_returns_error_not_crash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # When a parent segment is an existing *file*, the adapter's `create_directory`
    # makes `Path.mkdir(exist_ok=True)` raise a bare `FileExistsError` -- which the
    # harness does NOT convert to `ModelRetry`. It must come back as an `error:`
    # string, not escape and abort the whole agent run.
    monkeypatch.setenv('GITHUB_WORKSPACE', str(tmp_path))
    (tmp_path / 'afile').write_text('x', encoding='utf-8')
    out = asyncio.run(pkg.write_file(_ctx(), str(tmp_path / 'afile' / 'inner.txt'), 'data'))
    assert out.startswith('error:')


# --------------------------------------------------------------------------- #
# safe-outputs log diagnostic
# --------------------------------------------------------------------------- #
def test_log_safe_outputs_state_reports_entry_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: LogCaptureFixture
):
    safe_path = tmp_path / 'safe-outputs.jsonl'
    safe_path.write_text('{"a": 1}\n{"b": 2}\n\n', encoding='utf-8')
    monkeypatch.setenv('GH_AW_SAFE_OUTPUTS', str(safe_path))
    with caplog.at_level('INFO', logger='pydantic_ai_gh_aw_shim'):
        shim.log_safe_outputs_state()
    joined = '\n'.join(caplog.messages)
    assert 'entries=2' in joined and 'bytes=' in joined


def test_log_safe_outputs_state_handles_missing_env(monkeypatch: pytest.MonkeyPatch, caplog: LogCaptureFixture):
    monkeypatch.delenv('GH_AW_SAFE_OUTPUTS', raising=False)
    with caplog.at_level('INFO', logger='pydantic_ai_gh_aw_shim'):
        shim.log_safe_outputs_state()
    assert any('GH_AW_SAFE_OUTPUTS not set' in m for m in caplog.messages)


# --------------------------------------------------------------------------- #
# model resolution (proxy semantics — unchanged)
# --------------------------------------------------------------------------- #
def test_model_defaults_to_claude_sonnet_4_6(monkeypatch: pytest.MonkeyPatch):
    for v in ('ANTHROPIC_MODEL', 'ANTHROPIC_BASE_URL'):
        monkeypatch.delenv(v, raising=False)
    model, label = shim.build_model(shim.parse_args(['--print']))
    assert label == 'anthropic:claude-sonnet-4-6'
    assert model.__class__.__name__ == 'AnthropicModel'


def test_model_argv_flag_wins(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('ANTHROPIC_MODEL', 'from-env')
    model, label = shim.build_model(shim.parse_args(['--model', 'from-argv']))
    assert label == 'anthropic:from-argv'
    assert model.__class__.__name__ == 'AnthropicModel'


def test_model_anthropic_env_falls_back_when_no_argv(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv('ANTHROPIC_BASE_URL', raising=False)
    # gh-aw v0.74+ sets `ANTHROPIC_MODEL` (the Anthropic SDK standard) from
    # the workflow's `engine.model:` field. The runner picks it up because
    # the Claude-Code CLI does the same.
    monkeypatch.setenv('ANTHROPIC_MODEL', 'MiniMax-M2.7-Highspeed')
    monkeypatch.setenv('ANTHROPIC_AUTH_TOKEN', 'placeholder')
    model, label = shim.build_model(shim.parse_args(['--print']))
    assert label == 'anthropic:MiniMax-M2.7-Highspeed'
    assert model.__class__.__name__ == 'AnthropicModel'


def test_build_model_applies_llm_timeout_and_retries(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv('ANTHROPIC_BASE_URL', raising=False)
    monkeypatch.delenv('ANTHROPIC_MODEL', raising=False)
    model, _ = shim.build_model(shim.parse_args(['--print']))
    # The underlying AsyncAnthropic client should carry our timeout + retries.
    client = model.provider.client  # type: ignore[attr-defined]
    assert client.timeout == shim._LLM_TIMEOUT  # pyright: ignore[reportPrivateUsage]
    assert client.max_retries == shim._LLM_MAX_RETRIES  # pyright: ignore[reportPrivateUsage]


_MESSAGE_RESPONSE = {
    'id': 'msg_1',
    'type': 'message',
    'role': 'assistant',
    'model': 'MiniMax-M3',
    'content': [{'type': 'text', 'text': 'ok'}],
    'stop_reason': 'end_turn',
    'stop_sequence': None,
    'usage': {'input_tokens': 1, 'output_tokens': 1},
}
_RATE_LIMITED_RESPONSE = {
    'type': 'error',
    'error': {'type': 'rate_limit_error', 'message': 'Token Plan rate limit reached (2062)'},
}


def _rate_limited_client(rate_limited_responses: int, calls: list[int]) -> AsyncAnthropic:
    """An `AsyncAnthropic` client whose first requests receive a retryable 429."""

    def _handle(_request: httpx2.Request) -> httpx2.Response:
        calls.append(1)
        if len(calls) <= rate_limited_responses:
            return httpx2.Response(429, json=_RATE_LIMITED_RESPONSE)
        return httpx2.Response(200, json=_MESSAGE_RESPONSE)

    transport = shim.rate_limit_retry_transport(httpx2.MockTransport(_handle))
    transport.config['wait'] = wait_none()
    transport.config['stop'] = stop_after_attempt(4)
    # The shim's own SDK retry count, so a test also sees any SDK retries stacked on top.
    return AsyncAnthropic(
        api_key='x',
        max_retries=shim._LLM_MAX_RETRIES,  # pyright: ignore[reportPrivateUsage]
        http_client=httpx2.AsyncClient(transport=transport),
    )


def test_rate_limit_retry_transport_rides_out_a_429_burst():
    calls: list[int] = []
    client = _rate_limited_client(rate_limited_responses=3, calls=calls)
    message = asyncio.run(
        client.messages.create(model='MiniMax-M3', max_tokens=1, messages=[{'role': 'user', 'content': 'hi'}])
    )
    assert message.content[0].type == 'text'
    assert len(calls) == 4


def test_rate_limit_retry_transport_hands_the_last_429_to_the_sdk():
    calls: list[int] = []
    client = _rate_limited_client(rate_limited_responses=99, calls=calls)
    with pytest.raises(RateLimitError, match='Token Plan rate limit reached'):
        asyncio.run(
            client.messages.create(model='MiniMax-M3', max_tokens=1, messages=[{'role': 'user', 'content': 'hi'}])
        )
    assert len(calls) == 4


@pytest.mark.parametrize(
    'error_code',
    [
        '1113',
        '1308',
        '1309',
        '1310',
        '1311',
        '1313',
        '1314',
        '1315',
        '1316',
        '1317',
        '1318',
        '1319',
        '1320',
        '1321',
        pytest.param(1316, id='numeric-1316'),
    ],
)
def test_rate_limit_retry_transport_does_not_retry_plan_quota_exhaustion(error_code: str | int):
    calls: list[int] = []
    quota_response: dict[str, object] = {'error': {'code': error_code, 'message': 'Usage limit reached'}}

    def _handle(_request: httpx2.Request) -> httpx2.Response:
        calls.append(1)
        return httpx2.Response(
            429,
            headers={'content-type': 'application/json'},
            stream=httpx2.ByteStream(json.dumps(quota_response).encode()),
        )

    transport = shim.rate_limit_retry_transport(httpx2.MockTransport(_handle))
    transport.config['wait'] = wait_none()
    transport.config['stop'] = stop_after_attempt(4)
    client = AsyncAnthropic(
        api_key='x',
        max_retries=shim._LLM_MAX_RETRIES,  # pyright: ignore[reportPrivateUsage]
        http_client=httpx2.AsyncClient(transport=transport),
    )

    async def _request() -> None:
        async with client:
            with pytest.raises(RateLimitError) as exc_info:
                await client.messages.create(
                    model='glm-5.3-flash', max_tokens=1, messages=[{'role': 'user', 'content': 'hi'}]
                )
            assert exc_info.value.body == quota_response

    asyncio.run(_request())
    assert len(calls) == 1


@pytest.mark.parametrize(
    'error_body',
    [
        pytest.param(json.dumps({'error': {'code': '1302', 'message': 'Rate limited'}}).encode(), id='rate-limit-1302'),
        pytest.param(
            json.dumps({'error': {'code': '1305', 'message': 'Temporarily overloaded'}}).encode(), id='overloaded-1305'
        ),
        pytest.param(json.dumps({'error': {'code': '9999', 'message': 'Unknown'}}).encode(), id='unknown-code'),
        pytest.param(
            json.dumps({'error': {'code': '9' * 5000, 'message': 'Unknown'}}).encode(), id='large-numeric-code'
        ),
        pytest.param(b'{invalid json', id='malformed-json'),
    ],
)
def test_rate_limit_retry_transport_retries_transient_unknown_and_malformed_429s(error_body: bytes):
    calls: list[int] = []

    def _handle(_request: httpx2.Request) -> httpx2.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx2.Response(
                429,
                headers={'content-type': 'application/json'},
                stream=httpx2.ByteStream(error_body),
            )
        return httpx2.Response(200, json=_MESSAGE_RESPONSE)

    transport = shim.rate_limit_retry_transport(httpx2.MockTransport(_handle))
    transport.config['wait'] = wait_none()
    transport.config['stop'] = stop_after_attempt(4)
    client = AsyncAnthropic(
        api_key='x',
        max_retries=shim._LLM_MAX_RETRIES,  # pyright: ignore[reportPrivateUsage]
        http_client=httpx2.AsyncClient(transport=transport),
    )

    async def _request() -> None:
        async with client:
            message = await client.messages.create(
                model='glm-5.3-flash', max_tokens=1, messages=[{'role': 'user', 'content': 'hi'}]
            )
            assert message.content[0].type == 'text'
            assert message.content[0].text == 'ok'

    asyncio.run(_request())
    assert len(calls) == 2


def test_rate_limit_retry_attempts_each_get_an_http_span(monkeypatch: pytest.MonkeyPatch):
    """Every retry attempt goes through `httpx2.AsyncHTTPTransport`, which `logfire.instrument_httpx()` patches."""
    transport = shim.rate_limit_retry_transport()
    transport.config['wait'] = wait_none()
    assert isinstance(transport.wrapped, httpx2.AsyncHTTPTransport)
    statuses = iter([429, 429, 200])

    class _Pool:
        async def handle_async_request(self, _request: httpcore2.Request) -> httpcore2.Response:
            status = next(statuses)
            body = _RATE_LIMITED_RESPONSE if status == 429 else _MESSAGE_RESPONSE
            return httpcore2.Response(status, content=json.dumps(body).encode())

    monkeypatch.setattr(transport.wrapped, '_pool', _Pool())
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    instrumentor = HTTPX2ClientInstrumentor()
    instrumentor.instrument(tracer_provider=provider)
    try:
        response = asyncio.run(httpx2.AsyncClient(transport=transport).post('https://api.z.ai/api/anthropic'))
    finally:
        instrumentor.uninstrument()
    assert response.status_code == 200
    span_statuses = [(span.attributes or {}).get('http.status_code') for span in exporter.get_finished_spans()]
    assert span_statuses == [429, 429, 200]


@pytest.fixture
def agent_spans() -> Iterator[InMemorySpanExporter]:
    """Every agent's spans, instrumented as `configure_observability` instruments a gh-aw run."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    Agent.instrument_all(InstrumentationSettings(tracer_provider=provider))
    yield exporter
    Agent.instrument_all(False)


def _span_outcomes(exporter: InMemorySpanExporter) -> list[tuple[object, str, object]]:
    """Each span's GenAI operation, status and `error.type`, independent of agent and model names."""
    outcomes = [
        (
            (span.attributes or {}).get('gen_ai.operation.name'),
            span.status.status_code.name,
            (span.attributes or {}).get('error.type'),
        )
        for span in exporter.get_finished_spans()
    ]
    return sorted(outcomes, key=repr)


class _ClockSkippingLoop(asyncio.SelectorEventLoop):
    """An event loop whose clock a test moves forward, so a wall-clock cap fires without waiting."""

    skipped = 0.0

    def time(self) -> float:
        return super().time() + self.skipped


async def _skip_clock_and_hang(seconds: float) -> None:
    loop = asyncio.get_running_loop()
    assert isinstance(loop, _ClockSkippingLoop)
    loop.skipped += seconds
    await asyncio.Event().wait()


@pytest.mark.parametrize('chunks_before_cutoff', [0, 1])
def test_subagent_timeout_marks_the_spans_it_cuts_off(
    chunks_before_cutoff: int, agent_spans: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
):
    async def _stream(messages: list[ModelMessage], info: AgentInfo):
        if shim.SUBAGENT_INSTRUCTIONS in (info.instructions or ''):
            for _ in range(chunks_before_cutoff):
                yield 'partial'
            await _skip_clock_and_hang(shim.SUBAGENT_TIMEOUT_SECS)
        elif any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            yield 'done'
        else:
            yield {0: DeltaToolCall(name='Task', json_args='{"description": "probe", "prompt": "look"}')}

    monkeypatch.setattr(shim, 'log_safe_outputs_state', lambda: None)
    with redirect_stdout(io.StringIO()), asyncio.Runner(loop_factory=_ClockSkippingLoop) as runner:
        rc = runner.run(
            shim.run(
                prompt='review the PR',
                model=FunctionModel(stream_function=_stream),
                label='test-model',
                claude_code_toolset=shim.select_claude_code_toolset(None, None, task=shim.task),
                mcp_servers=[],
                session_id='subagent-timeout',
            )
        )
    assert rc == 0
    assert _span_outcomes(agent_spans) == [
        ('chat', 'ERROR', 'timeout'),
        ('chat', 'UNSET', None),
        ('chat', 'UNSET', None),
        ('execute_tool', 'ERROR', 'timeout'),
        ('invoke_agent', 'ERROR', 'timeout'),
        ('invoke_agent', 'UNSET', None),
    ]
    [cut_off_chat] = [
        span
        for span in agent_spans.get_finished_spans()
        if span.name.startswith('chat') and span.status.status_code.name == 'ERROR'
    ]
    # Usage the stream had already reported stays on the span.
    assert ('gen_ai.usage.input_tokens' in (cut_off_chat.attributes or {})) == bool(chunks_before_cutoff)


def test_subagent_is_resumed_after_its_cap_cuts_it_off(
    agent_spans: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    task_returns: list[str] = []

    async def _stream(messages: list[ModelMessage], info: AgentInfo):
        returns = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
        if not info.function_tools:
            # The resume request, the only one offered no tools.
            yield 'Reviewed files: none'
        elif shim.SUBAGENT_INSTRUCTIONS in (info.instructions or ''):
            if not returns:
                yield {0: DeltaToolCall(name='Glob', json_args='{"pattern": "*"}')}
            else:
                yield 'partial'
                await _skip_clock_and_hang(shim.SUBAGENT_TIMEOUT_SECS)
        elif returns:
            task_returns.extend(str(part.content) for part in returns if part.tool_name == 'Task')
            yield 'done'
        else:
            yield {0: DeltaToolCall(name='Task', json_args='{"description": "probe", "prompt": "look"}')}

    monkeypatch.setenv('GITHUB_WORKSPACE', str(tmp_path))
    monkeypatch.setattr(shim, 'log_safe_outputs_state', lambda: None)
    with redirect_stdout(io.StringIO()), asyncio.Runner(loop_factory=_ClockSkippingLoop) as runner:
        rc = runner.run(
            shim.run(
                prompt='review the PR',
                model=FunctionModel(stream_function=_stream),
                label='test-model',
                claude_code_toolset=shim.select_claude_code_toolset(None, None, task=shim.task),
                mcp_servers=[],
                session_id='subagent-resume',
            )
        )
    assert rc == 0
    assert task_returns == [_resumed_report('Reviewed files: none')]
    # The cap still marks the delegate and its `Task` call as timed out; the resume's own spans succeed.
    assert _span_outcomes(agent_spans) == [
        ('chat', 'ERROR', 'timeout'),
        ('chat', 'UNSET', None),
        ('chat', 'UNSET', None),
        ('chat', 'UNSET', None),
        ('chat', 'UNSET', None),
        ('execute_tool', 'ERROR', 'timeout'),
        ('execute_tool', 'UNSET', None),
        ('invoke_agent', 'ERROR', 'timeout'),
        ('invoke_agent', 'UNSET', None),
        ('invoke_agent', 'UNSET', None),
    ]
    assert 'invoke_agent delegate-resume' in {span.name for span in agent_spans.get_finished_spans()}


def test_a_subagents_own_timeout_error_is_not_marked_as_timed_out(
    agent_spans: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
):
    async def _stream(messages: list[ModelMessage], info: AgentInfo):
        if shim.SUBAGENT_INSTRUCTIONS in (info.instructions or ''):
            raise TimeoutError('provider read timeout')
        elif any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            yield 'done'
        else:
            yield {0: DeltaToolCall(name='Task', json_args='{"description": "probe", "prompt": "look"}')}

    monkeypatch.setattr(shim, 'log_safe_outputs_state', lambda: None)
    with redirect_stdout(io.StringIO()):
        rc = asyncio.run(
            shim.run(
                prompt='review the PR',
                model=FunctionModel(stream_function=_stream),
                label='test-model',
                claude_code_toolset=shim.select_claude_code_toolset(None, None, task=shim.task),
                mcp_servers=[],
                session_id='subagent-timeout-error',
            )
        )
    assert rc == 0
    assert _span_outcomes(agent_spans) == [
        ('chat', 'UNSET', None),
        ('chat', 'UNSET', None),
        ('chat', 'UNSET', None),
        ('execute_tool', 'UNSET', None),
        ('invoke_agent', 'ERROR', None),
        ('invoke_agent', 'UNSET', None),
    ]


@pytest.mark.parametrize(
    ('stream_chunks', 'expected'),
    [
        # pydantic-ai records the cancelled tool span as an error itself.
        pytest.param(
            None,
            [('chat', 'UNSET', None), ('execute_tool', 'ERROR', None), ('invoke_agent', 'ERROR', 'timeout')],
            id='in-tool',
        ),
        pytest.param(0, [('chat', 'ERROR', 'timeout'), ('invoke_agent', 'ERROR', 'timeout')], id='before-first-chunk'),
        pytest.param(1, [('chat', 'ERROR', 'timeout'), ('invoke_agent', 'ERROR', 'timeout')], id='after-first-chunk'),
    ],
)
def test_run_timeout_marks_the_spans_it_cuts_off(
    stream_chunks: int | None,
    expected: list[tuple[str, str, str | None]],
    agent_spans: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
):
    # A 6-minute run budget, so the run starts outside the parent's time reserve with all its tools.
    budget_secs = 360

    async def hang() -> None:
        await _skip_clock_and_hang(budget_secs)

    async def _stream(_messages: list[ModelMessage], _info: AgentInfo):
        if stream_chunks is None:
            yield {0: DeltaToolCall(name='hang', json_args='{}')}
        else:
            for _ in range(stream_chunks):
                yield 'partial'
            await _skip_clock_and_hang(budget_secs)

    monkeypatch.setenv('PYDANTIC_AI_JOB_TIMEOUT_MINUTES', str(budget_secs // 60 + shim.JOB_TIMEOUT_HEADROOM_MINS))
    with redirect_stdout(io.StringIO()), asyncio.Runner(loop_factory=_ClockSkippingLoop) as runner:
        rc = runner.run(
            shim._run_with_timeout(  # pyright: ignore[reportPrivateUsage]
                'p', FunctionModel(stream_function=_stream), 'lbl', FunctionToolset[object]([hang]), [], 'sess-test'
            )
        )
    assert rc == 1
    assert _span_outcomes(agent_spans) == expected


def test_a_failed_stream_is_not_marked_as_timed_out(agent_spans: InMemorySpanExporter):
    """pydantic-ai cancels a request whose stream failed, under a run cap that has not expired."""

    async def _stream(_messages: list[ModelMessage], _info: AgentInfo):
        yield 'partial'
        raise ModelHTTPError(529, 'glm-5.3-flash')

    with redirect_stdout(io.StringIO()):
        rc = asyncio.run(
            shim._run_with_timeout(  # pyright: ignore[reportPrivateUsage]
                'p', FunctionModel(stream_function=_stream), 'lbl', FunctionToolset[object](), [], 'stream-failure'
            )
        )
    assert rc == 1
    assert _span_outcomes(agent_spans) == [('chat', 'UNSET', None), ('invoke_agent', 'ERROR', None)]


def test_run_with_timeout_emits_error_on_global_timeout(monkeypatch: pytest.MonkeyPatch):
    seconds_left: list[float | None] = []

    async def _hang(*_a: object, **kw: object) -> int:
        usage = kw['usage']
        assert isinstance(usage, RunUsage)
        usage.incr(RunUsage(requests=2, input_tokens=11, output_tokens=4, cache_read_tokens=3))
        seconds_left.append(shim._run_seconds_left())  # pyright: ignore[reportPrivateUsage]
        await asyncio.sleep(9999)
        return 0

    monkeypatch.setattr(shim, 'run', _hang)
    monkeypatch.setattr(shim, '_run_timeout_secs', lambda: 0.01)
    monkeypatch.delenv('PYDANTIC_AI_RUN_ATTEMPT', raising=False)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = asyncio.run(
            shim._run_with_timeout(  # pyright: ignore[reportPrivateUsage]
                'p', cast(_Model[Any], object()), 'lbl', cast(AbstractToolset[object], object()), [], 'sess-test'
            )
        )
    assert rc == 1
    obj = json.loads(buf.getvalue().strip())
    assert obj['type'] == 'result' and obj['is_error'] is True
    assert 'timed out' in obj['result']
    assert obj['usage']['input_tokens'] == 11
    assert obj['usage']['output_tokens'] == 4
    assert obj['usage']['cache_read_input_tokens'] == 3
    assert obj['num_turns'] == 2
    assert obj['provider_health']['failure']['kind'] == 'time_limit'
    assert obj['provider_health']['run_attempt'] is None
    [left] = seconds_left
    assert left is not None and left <= 0.01
    assert shim._run_seconds_left() is None  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(
    ('cut_off', 'num_turns'),
    [
        # 2 root requests and the sub-agent's 2.
        pytest.param(False, 4, id='finished'),
        # The root request in flight at the cutoff never completes.
        pytest.param(True, 3, id='cut-off'),
    ],
)
def test_num_turns_counts_subagent_requests_whether_finished_or_cut_off(
    cut_off: bool, num_turns: int, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    # A 6-minute run budget, so the run starts outside the parent's time reserve with `Task` offered.
    budget_secs = 360

    async def _stream(messages: list[ModelMessage], info: AgentInfo):
        returned = any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts)
        if shim.SUBAGENT_INSTRUCTIONS in (info.instructions or ''):
            if returned:
                yield 'sub done'
            else:
                yield {0: DeltaToolCall(name='Glob', json_args='{"pattern": "*"}')}
        elif not returned:
            yield {0: DeltaToolCall(name='Task', json_args='{"description": "probe", "prompt": "look"}')}
        elif cut_off:
            await _skip_clock_and_hang(budget_secs)
        else:
            yield 'done'

    monkeypatch.setenv('GITHUB_WORKSPACE', str(tmp_path))
    monkeypatch.setenv('PYDANTIC_AI_JOB_TIMEOUT_MINUTES', str(budget_secs // 60 + shim.JOB_TIMEOUT_HEADROOM_MINS))
    monkeypatch.setattr(shim, 'log_safe_outputs_state', lambda: None)
    buf = io.StringIO()
    with redirect_stdout(buf), asyncio.Runner(loop_factory=_ClockSkippingLoop) as runner:
        rc = runner.run(
            shim._run_with_timeout(  # pyright: ignore[reportPrivateUsage]
                'p',
                FunctionModel(stream_function=_stream),
                'lbl',
                shim.select_claude_code_toolset(None, None, task=shim.task),
                [],
                'num-turns',
            )
        )
    result = json.loads(buf.getvalue().splitlines()[-1])
    assert (rc, result['type'], result['is_error']) == (int(cut_off), 'result', cut_off)
    assert result['num_turns'] == num_turns


@pytest.mark.parametrize('outcome', ['success', 'error', 'timeout'])
def test_run_with_timeout_closes_anthropic_client(outcome: str, monkeypatch: pytest.MonkeyPatch):
    client = AsyncAnthropic(api_key='test')
    model = AnthropicModel('test-model', provider=AnthropicProvider(anthropic_client=client))

    async def _fake_run(*_args: object, **_kwargs: object) -> int:
        if outcome == 'error':
            raise RuntimeError('test failure')
        if outcome == 'timeout':
            await asyncio.Event().wait()
        return 0

    monkeypatch.setattr(shim, 'run', _fake_run)
    monkeypatch.setattr(shim, '_run_timeout_secs', lambda: 0.01 if outcome == 'timeout' else 1)
    with redirect_stdout(io.StringIO()):
        rc = asyncio.run(
            shim._run_with_timeout(  # pyright: ignore[reportPrivateUsage]
                'p', model, 'lbl', FunctionToolset[object](), [], 'sess-test'
            )
        )
    assert rc == (0 if outcome == 'success' else 1)
    assert client.is_closed


@pytest.mark.parametrize('attempt', [None, '', 'not-an-int', '0', '-2'])
def test_emit_result_uses_null_for_invalid_run_attempt(attempt: str | None, monkeypatch: pytest.MonkeyPatch):
    if attempt is None:
        monkeypatch.delenv('PYDANTIC_AI_RUN_ATTEMPT', raising=False)
    else:
        monkeypatch.setenv('PYDANTIC_AI_RUN_ATTEMPT', attempt)
    buf = io.StringIO()
    with redirect_stdout(buf):
        shim.emit_result('failed', usage=None, session_id='invalid-attempt', is_error=True)
    result = json.loads(buf.getvalue().strip())
    assert result['provider_health']['run_attempt'] is None


# Both names the budget can come from. `PYDANTIC_AI_JOB_TIMEOUT_MINUTES` is the one that
# runs in production — gh-aw sets `GH_AW_TIMEOUT_MINUTES` only on the failure-handler step,
# so it never reaches the agent container — which is exactly why it must be parametrized
# here rather than left to the other name: a typo in the primary lookup would otherwise
# restore the old hardcoded budget with the suite still green.
JOB_TIMEOUT_ENV_NAMES = ['PYDANTIC_AI_JOB_TIMEOUT_MINUTES', 'GH_AW_TIMEOUT_MINUTES']


@pytest.fixture(params=JOB_TIMEOUT_ENV_NAMES)
def job_timeout_env(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    """Set one budget variable and clear the other, so neither leaks in from the shell."""

    def _set(value: str) -> None:
        for name in JOB_TIMEOUT_ENV_NAMES:
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv(request.param, value)

    return _set


def test_run_timeout_budget_tracks_the_jobs_own_timeout(job_timeout_env: Callable[[str], None]):
    """The agent must stop just under the job's cap, whatever that cap is.

    Hardcoding 28 min silently ignored any workflow that raised its own
    `timeout-minutes`, so the extra time was granted and never used.
    """
    job_timeout_env('45')
    assert shim._run_timeout_secs() == 43 * 60  # pyright: ignore[reportPrivateUsage]

    job_timeout_env('30')
    assert shim._run_timeout_secs() == 28 * 60  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize('value', ['', 'not-a-number', '0', '2'])
def test_run_timeout_budget_falls_back_when_the_env_is_unusable(job_timeout_env: Callable[[str], None], value: str):
    """Absent, malformed, or smaller than the teardown headroom all fall back."""
    job_timeout_env(value)
    assert shim._run_timeout_secs() == 28 * 60  # pyright: ignore[reportPrivateUsage]


def test_run_timeout_budget_prefers_the_variable_that_reaches_the_agent(monkeypatch: pytest.MonkeyPatch):
    """`PYDANTIC_AI_JOB_TIMEOUT_MINUTES` wins: it is the one set on the agent's own job."""
    monkeypatch.setenv('PYDANTIC_AI_JOB_TIMEOUT_MINUTES', '45')
    monkeypatch.setenv('GH_AW_TIMEOUT_MINUTES', '20')
    assert shim._run_timeout_secs() == 43 * 60  # pyright: ignore[reportPrivateUsage]


def test_job_timeout_constants_match_the_guard():
    """The guard mirrors these rather than importing the shim's runtime; pin them together."""
    assert agentic_workflow_guard.DEFAULT_JOB_TIMEOUT_MINS == shim.DEFAULT_JOB_TIMEOUT_MINS
    assert agentic_workflow_guard.JOB_TIMEOUT_HEADROOM_MINS == shim.JOB_TIMEOUT_HEADROOM_MINS


# --------------------------------------------------------------------------- #
# MCP translation & allow-list filtering
# --------------------------------------------------------------------------- #
def test_mcp_missing_config_degrades_gracefully():
    assert shim.build_mcp_servers(shim.Args(mcp_config='/no/such/file.json')) == []


def _mcp_cfg(tmp_path: Path) -> Path:
    cfg = tmp_path / 'mcp.json'
    cfg.write_text(
        json.dumps(
            {
                'mcpServers': {
                    'github': {'command': 'docker', 'args': ['run'], 'env': {'X': '1'}},
                    'safeoutputs': {
                        'type': 'http',
                        'url': 'http://host.docker.internal:1234',
                        'headers': {'Authorization': 'k'},
                    },
                }
            }
        ),
        encoding='utf-8',
    )
    return cfg


def test_mcp_translates_stdio_and_http_unfiltered(tmp_path: Path):
    # `load_mcp_toolsets` wraps each server in a `PrefixedToolset`; the shim
    # then re-prefixes to Claude Code's wire format.
    servers = shim.build_mcp_servers(shim.Args(mcp_config=str(_mcp_cfg(tmp_path))))
    assert len(servers) == 2
    assert {s.__class__.__name__ for s in servers} == {'PrefixedToolset'}


def test_mcp_tools_use_claude_code_wire_format(tmp_path: Path):
    """The re-prefix step makes the model-visible tool name exactly equal to
    gh-aw's `mcp__<server>__<tool>` allow-list entry — the same name Claude
    Code uses on the wire and that Claude was trained to call. With matching
    names the allow-list filter becomes a literal containment check."""
    servers = shim.build_mcp_servers(shim.Args(mcp_config=str(_mcp_cfg(tmp_path))))
    prefixed = [s for s in servers if isinstance(s, PrefixedToolset)]
    assert len(prefixed) == 2
    # `PrefixedToolset` inserts a literal `_` between prefix and tool name;
    # combined with our trailing-underscore prefix this yields the canonical
    # `mcp__<server>__<tool>` double-underscore shape.
    assert {s.prefix for s in prefixed} == {'mcp__github_', 'mcp__safeoutputs_'}


def test_mcp_wrapped_in_filter_when_allowlist_present(tmp_path: Path):
    servers = shim.build_mcp_servers(
        shim.Args(mcp_config=str(_mcp_cfg(tmp_path)), allowed_tools=frozenset({'mcp__safeoutputs'}))
    )
    assert len(servers) == 2
    assert {s.__class__.__name__ for s in servers} == {'FilteredToolset'}


# --------------------------------------------------------------------------- #
# MCP tool-error recovery (the empty-body-`APPROVE` crash: a bare `McpError`
# from the gh-aw gateway escaped `MCPToolset` and killed the whole run).
# --------------------------------------------------------------------------- #
def _mcp_error(message: str) -> McpError:
    return McpError(ErrorData(code=-32602, message=message))


def _error_hook_ctx() -> RunContext[None]:
    return RunContext(
        deps=None,
        model=cast(_Model[Any], None),
        usage=RunUsage(),
        prompt=None,
        messages=[],
        run_step=0,
    )


def test_mcp_protocol_error_message_recognizes_only_mcp_errors():
    err = _mcp_error('review body is empty')
    assert shim._mcp_protocol_error_message(err) == str(err)  # pyright: ignore[reportPrivateUsage]
    # A non-MCP exception (e.g. a real bug in a tool) must not be swallowed as a tool result.
    assert shim._mcp_protocol_error_message(RuntimeError('not mcp')) is None  # pyright: ignore[reportPrivateUsage]


def test_recover_mcp_tool_errors_returns_error_string_instead_of_crashing():
    cap = shim._RecoverMCPToolErrors()  # pyright: ignore[reportPrivateUsage]
    call = ToolCallPart(tool_name='mcp__safeoutputs__submit_pull_request_review', args={}, tool_call_id='c1')
    out = asyncio.run(
        cap.on_tool_execute_error(
            _error_hook_ctx(),
            call=call,
            tool_def=ToolDefinition(name=call.tool_name),
            args={},
            error=_mcp_error('review body is empty and no create_pull_request_review_comment calls were made'),
        )
    )
    assert out.startswith('error:') and 'review body is empty' in out


def test_recover_mcp_tool_errors_reraises_non_mcp_errors():
    cap = shim._RecoverMCPToolErrors()  # pyright: ignore[reportPrivateUsage]
    call = ToolCallPart(tool_name='Bash', args={}, tool_call_id='c2')
    with pytest.raises(RuntimeError, match='boom'):
        asyncio.run(
            cap.on_tool_execute_error(
                _error_hook_ctx(), call=call, tool_def=ToolDefinition(name='Bash'), args={}, error=RuntimeError('boom')
            )
        )


def test_mcp_allow_predicate_server_wildcard_vs_specific():
    # The model-visible tool name is Claude Code's wire form
    # `mcp__<server>__<tool>` (see `_apply_claude_mcp_prefix`), identical to
    # gh-aw's allow-list entries — so the predicate is a literal containment
    # check, with a wildcard for whole-server allows.

    # whole-server allow
    pred = shim._mcp_tool_allowed('safeoutputs', frozenset({'mcp__safeoutputs'}))  # pyright: ignore[reportPrivateUsage]
    assert pred(cast(RunContext[None], None), ToolDefinition(name='mcp__safeoutputs__create_issue')) is True
    assert (
        pred(cast(RunContext[None], None), ToolDefinition(name='mcp__safeoutputs__create_pull_request_review_comment'))
        is True
    )

    # specific-tool allow only
    pred = shim._mcp_tool_allowed('github', frozenset({'mcp__github__get_me'}))  # pyright: ignore[reportPrivateUsage]
    assert pred(cast(RunContext[None], None), ToolDefinition(name='mcp__github__get_me')) is True
    assert pred(cast(RunContext[None], None), ToolDefinition(name='mcp__github__delete_repo')) is False


# --------------------------------------------------------------------------- #
# stream-json schema & structured-error guarantee
# --------------------------------------------------------------------------- #
def test_emit_result_matches_claude_stream_json_schema(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('PYDANTIC_AI_RUN_ATTEMPT', '3')
    buf = io.StringIO()
    with redirect_stdout(buf):
        shim.emit_result('answer', usage=None, session_id='run-1')
    obj = json.loads(buf.getvalue().strip())
    assert obj['type'] == 'result'
    assert obj['subtype'] == 'success'
    assert obj['is_error'] is False
    assert obj['result'] == 'answer'
    assert obj['provider_health']['run_attempt'] == 3
    assert 'failure' not in obj['provider_health']
    for k in ('input_tokens', 'output_tokens', 'cache_creation_input_tokens', 'cache_read_input_tokens'):
        assert k in obj['usage']


def test_emit_result_passes_through_turns_and_duration():
    buf = io.StringIO()
    with redirect_stdout(buf):
        shim.emit_result('x', usage=None, session_id='s', num_turns=3, duration_ms=1234)
    obj = json.loads(buf.getvalue().strip())
    assert obj['num_turns'] == 3 and obj['duration_ms'] == 1234


def test_emit_result_error_subtype():
    buf = io.StringIO()
    with redirect_stdout(buf):
        shim.emit_result('boom', usage=None, session_id='run-1', is_error=True)
    obj = json.loads(buf.getvalue().strip())
    assert obj['subtype'] == 'error' and obj['is_error'] is True


def _zai_429_body(code: str | int) -> dict[str, object]:
    """The error body the Anthropic SDK decodes from a Z.ai 429."""
    return {
        'type': 'error',
        'error': {'type': 'rate_limit_error', 'code': code, 'message': f'[{code}][provider detail]'},
        'request_id': 'request-1',
    }


@pytest.mark.parametrize(
    ('error', 'expected_kind', 'expected_status'),
    [
        (ModelHTTPError(401, 'test-model', {'error': {'type': 'authentication_error'}}), 'authentication', 401),
        (ModelHTTPError(403, 'test-model', {'error': {'type': 'permission_error'}}), 'authentication', 403),
        (ModelHTTPError(429, 'test-model', {'error': {'type': 'rate_limit_error'}}), 'rate_limit', 429),
        (ModelHTTPError(429, 'test-model', None), 'rate_limit', 429),
        (ModelHTTPError(429, 'glm-5.3-flash', _zai_429_body('1302')), 'rate_limit', 429),
        (ModelHTTPError(429, 'glm-5.3-flash', _zai_429_body('1305')), 'rate_limit', 429),
        (ModelHTTPError(429, 'glm-5.3-flash', _zai_429_body('1113')), 'plan_limit', 429),
        (ModelHTTPError(429, 'glm-5.3-flash', _zai_429_body('1310')), 'plan_limit', 429),
        (ModelHTTPError(429, 'glm-5.3-flash', _zai_429_body(1311)), 'plan_limit', 429),
        (ModelHTTPError(429, 'glm-5.3-flash', _zai_429_body('9999')), 'rate_limit', 429),
        (ModelHTTPError(402, 'test-model', {'error': {'type': 'payment_required_error'}}), 'other', 402),
        (UsageLimitExceeded('request limit'), 'request_limit', None),
        (shim.RunTimeLimitExceeded(), 'time_limit', None),
        (TimeoutError(), 'timeout', None),
        (RuntimeError('provider returned a 402'), 'other', None),
    ],
)
def test_failure_metadata_classifies_typed_errors(
    error: BaseException, expected_kind: str, expected_status: int | None
):
    details = shim._failure_details(error)  # pyright: ignore[reportPrivateUsage]
    assert details['kind'] == expected_kind
    assert details.get('http_status') == expected_status


def test_failure_metadata_classifies_wrapped_transport_timeout():
    error = shim.ModelAPIError('test-model', 'ReadTimeout')
    error.__cause__ = httpx2.ReadTimeout('read timed out')

    details = shim._failure_details(error)  # pyright: ignore[reportPrivateUsage]

    assert details == {'kind': 'timeout'}


def test_failure_metadata_classifies_anthropic_timeout_wrapped_by_model_api_error():
    try:
        raise httpx2.ReadTimeout('read timed out')
    except httpx2.ReadTimeout as transport_error:
        try:
            raise APITimeoutError(request=httpx2.Request('POST', 'https://example.com')) from transport_error
        except APITimeoutError as sdk_error:
            error = shim.ModelAPIError('test-model', 'APITimeoutError')
            error.__cause__ = sdk_error

    details = shim._failure_details(error)  # pyright: ignore[reportPrivateUsage]

    assert details == {'kind': 'timeout'}


def test_run_preserves_partial_usage_on_failure_after_model_activity(monkeypatch: pytest.MonkeyPatch):
    emitted: list[dict[str, object]] = []
    monkeypatch.setattr(shim, 'emit', lambda obj: emitted.append(dict(obj)))  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    monkeypatch.setenv('GH_AW_SAFE_OUTPUTS', '/path/that/does/not/exist')
    monkeypatch.setenv('GITHUB_WORKFLOW', 'Pydantic AI CI Review')
    monkeypatch.setenv('PYDANTIC_AI_TASK_KEY', 'pr-123')
    monkeypatch.setenv('PYDANTIC_AI_TRIGGER_EVENT', 'pull_request')
    monkeypatch.setenv('PYDANTIC_AI_RUN_ATTEMPT', '2')
    calls = 0

    def _respond(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse(parts=[TextPart('still working')])
        raise ModelHTTPError(429, 'MiniMax-M3', {'error': {'type': 'rate_limit_error', 'message': 'private detail'}})

    async def _stream(messages: list[ModelMessage], info: AgentInfo):
        response = _respond(messages, info)
        for index, part in enumerate(response.parts):
            if isinstance(part, TextPart):
                yield part.content
            else:
                assert isinstance(part, ToolCallPart)
                yield {index: DeltaToolCall(name=part.tool_name, json_args=part.args_as_json_str())}

    rc = asyncio.run(
        shim.run(
            prompt='review',
            model=FunctionModel(_respond, stream_function=_stream),
            label='test-model',
            claude_code_toolset=shim.select_claude_code_toolset(None, None, task=None),
            mcp_servers=[],
            session_id='usage-failure',
            usage=RunUsage(cache_read_tokens=4, cache_write_tokens=2),
        )
    )
    result = next(event for event in emitted if event.get('type') == 'result')
    assert rc == 1
    assert result['is_error'] is True
    typed_token_usage = TypeAdapter(dict[str, int]).validate_python(result['usage'])
    input_tokens = typed_token_usage['input_tokens']
    output_tokens = typed_token_usage['output_tokens']
    assert input_tokens > 0
    assert output_tokens > 0
    assert typed_token_usage.get('cache_creation_input_tokens') == 2
    assert typed_token_usage.get('cache_read_input_tokens') == 4
    assert TypeAdapter(int).validate_python(result['num_turns']) > 0
    assert result['provider_health'] == {
        'workflow': 'Pydantic AI CI Review',
        'task_key': 'pr-123',
        'trigger_event': 'pull_request',
        'run_attempt': 2,
        'failure': {'kind': 'rate_limit', 'http_status': 429},
    }
    assert 'private detail' not in json.dumps(result['provider_health'])


def test_emit_result_reads_usage_attributes():
    class U:
        input_tokens = 22
        output_tokens = 292
        cache_write_tokens = 5
        cache_read_tokens = 7

    buf = io.StringIO()
    with redirect_stdout(buf):
        shim.emit_result('x', usage=cast(RunUsage, U()), session_id='s')
    usage = json.loads(buf.getvalue().strip())['usage']
    assert usage['input_tokens'] == 22
    assert usage['output_tokens'] == 292
    assert usage['cache_creation_input_tokens'] == 5  # mapped from cache_write_tokens
    assert usage['cache_read_input_tokens'] == 7


def test_main_emits_structured_error_on_empty_prompt(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sys, 'argv', ['pydantic-ai-runner', '--print'])
    monkeypatch.delenv('GH_AW_PROMPT', raising=False)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = shim.main()
    assert rc == 1
    obj = json.loads(buf.getvalue().strip())
    assert obj['type'] == 'result' and obj['is_error'] is True


def test_main_emits_structured_error_on_startup_failure(monkeypatch: pytest.MonkeyPatch):
    # A failure *before* the agent loop (e.g. model build) must still produce a
    # parseable stream-json result, never an opaque "no entries" run.
    def boom(_args: object) -> None:
        raise RuntimeError('kaboom')

    monkeypatch.setattr(shim, 'build_model', boom)
    monkeypatch.delenv('PYDANTIC_AI_RUN_ATTEMPT', raising=False)
    monkeypatch.setattr(sys, 'argv', ['pydantic-ai-runner', '--print', 'hello'])
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = shim.main()
    assert rc == 1
    obj = json.loads(buf.getvalue().strip())
    assert obj['is_error'] is True
    assert 'shim startup failed' in obj['result']
    assert 'kaboom' in obj['result']
    assert obj['usage']['input_tokens'] == 0
    assert obj['provider_health']['failure']['kind'] == 'other'
    assert obj['provider_health']['run_attempt'] is None


def test_main_emits_structured_error_on_argparse_rejection(monkeypatch: pytest.MonkeyPatch):
    # argparse `action='store_true'` raises `SystemExit(2)` on
    # `--print=true`; gh-aw still needs a structured `result` line.
    monkeypatch.setattr(sys, 'argv', ['pydantic-ai-runner', '--print=true', 'hi'])
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = shim.main()
    assert rc == 1
    obj = json.loads(buf.getvalue().strip())
    assert obj['type'] == 'result' and obj['is_error'] is True
    assert 'shim startup failed' in obj['result']


@pytest.mark.skipif(
    not os.environ.get('GH_AW_SHIM_LIVE_API_KEY'),
    reason='set GH_AW_SHIM_LIVE_API_KEY/_BASE_URL/_MODEL to run the live test',
)
def test_live_anthropic_compatible_endpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Verify streamed tool execution and a local safe-output sink against a live Anthropic-shape endpoint."""
    monkeypatch.setenv('ANTHROPIC_API_KEY', os.environ['GH_AW_SHIM_LIVE_API_KEY'])
    monkeypatch.setenv(
        'ANTHROPIC_BASE_URL',
        os.environ.get('GH_AW_SHIM_LIVE_BASE_URL', 'https://api.anthropic.com'),
    )
    sink = tmp_path / 'outputs.jsonl'
    monkeypatch.setenv('GH_AW_SAFE_OUTPUTS', str(sink))

    def build_safe_output_toolsets(_args: shim.Args) -> list[AbstractToolset[object]]:
        return [PrefixedToolset(_safe_outputs_toolset(sink), prefix='mcp__safeoutputs_')]

    monkeypatch.setattr(
        shim,
        'build_mcp_servers',
        build_safe_output_toolsets,
    )
    model = os.environ.get('GH_AW_SHIM_LIVE_MODEL', 'claude-sonnet-4-6')
    argv = list(GHAW_ARGV)
    i = argv.index('--mcp-config')
    del argv[i : i + 2]  # no MCP gateway outside a gh-aw run
    argv += [
        '--model',
        model,
        'Call `mcp__safeoutputs__noop` exactly once with the message '
        '`live Anthropic-compatible runner tool call verified`, then finish.',
    ]
    monkeypatch.setattr(sys, 'argv', ['pydantic-ai-runner', *argv])
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = shim.main()
    assert rc == 0
    lines = [json.loads(x) for x in buf.getvalue().splitlines() if x.strip()]
    result = next(x for x in lines if x['type'] == 'result')
    assert result['is_error'] is False
    assert result['result']
    assert sink.exists()
    assert [json.loads(line) for line in sink.read_text(encoding='utf-8').splitlines()] == [
        {'type': 'noop', 'message': 'live Anthropic-compatible runner tool call verified'}
    ]
    tool_events: list[dict[str, object]] = [
        event
        for line in lines
        if line.get('type') == 'assistant'
        and isinstance(line.get('message'), dict)
        and isinstance(line['message'].get('content'), list)
        for event in line['message']['content']
        if event.get('type') == 'tool_use'
    ]
    assert len(tool_events) == 1
    assert tool_events[0]['name'] == 'mcp__safeoutputs__noop'
    assert any(
        line.get('type') == 'user' and any(event.get('type') == 'tool_result' for event in line['message']['content'])
        for line in lines
    )
    # `input_tokens > 0` proves the prompt round-tripped; `output_tokens > 0`
    # proves the model actually responded.
    assert result['usage']['input_tokens'] > 0
    assert result['usage']['output_tokens'] > 0
