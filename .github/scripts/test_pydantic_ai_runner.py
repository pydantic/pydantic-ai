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
import io
import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, cast

import pytest
from pytest import LogCaptureFixture

# `.github/scripts/` isn't on sys.path by default — the shim package lives
# there. The runtime equivalent is the PEP-723 launcher script
# (`pydantic-ai-runner`) which inserts the same directory before
# `runpy.run_module`-ing the package.
sys.path.insert(0, str(Path(__file__).parent))

# Tool callables, shared helpers, and the CLI live in distinct submodules;
# tests import each from where it actually lives, not from a re-export.
import pydantic_ai_gh_aw_shim as pkg
from pydantic_ai_gh_aw_shim import (
    cli as shim,
    shared,
)

from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import Model as _Model
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import AbstractToolset

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
    from pydantic_ai.usage import RunUsage

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
    import asyncio

    names = asyncio.run(_toolset_names(None, None, task=shim.task))
    # task=shim.task adds "Task" alongside the base callables. Order is
    # insertion order from `_BASE_TOOLS` + the appended Task entry.
    assert names == [*pkg.CLAUDE_CODE_TOOL_NAMES, 'Task']


def test_select_claude_code_toolset_enforces_allowlist():
    import asyncio

    names = asyncio.run(_toolset_names(frozenset({'Bash', 'Read', 'mcp__safeoutputs'}), None))
    assert names == ['Bash', 'Read']


def test_plan_mode_withholds_mutating_tools():
    import asyncio

    names = set(asyncio.run(_toolset_names(None, 'plan')))
    assert names.isdisjoint(pkg.MUTATING_TOOLS)
    assert 'Read' in names and 'Grep' in names and 'Glob' in names


def test_plan_mode_and_allowlist_compose():
    import asyncio

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


# --------------------------------------------------------------------------- #
# Claude Code tool behavior
# --------------------------------------------------------------------------- #
def test_file_tools_roundtrip(tmp_path: Path):
    f = tmp_path / 'sub' / 'note.txt'
    assert 'wrote' in pkg.write_file(str(f), 'hello\nworld\n')
    assert pkg.read_file(str(f)) == 'hello\nworld\n'
    assert 'edited' in pkg.edit_file(str(f), 'world', 'there')
    assert 'there' in pkg.read_file(str(f))
    assert 'note.txt' in pkg.list_dir(str(tmp_path / 'sub'))
    assert pkg.edit_file(str(f), 'absent', 'x') == 'error: `old_string` not found'


def test_read_file_offset_and_limit(tmp_path: Path):
    f = tmp_path / 'n.txt'
    f.write_text('l1\nl2\nl3\nl4\n', encoding='utf-8')
    assert pkg.read_file(str(f), offset=2, limit=2) == 'l2\nl3'


def test_edit_file_replace_all(tmp_path: Path):
    f = tmp_path / 'r.txt'
    f.write_text('a a a', encoding='utf-8')
    pkg.edit_file(str(f), 'a', 'b', replace_all=True)
    assert f.read_text(encoding='utf-8') == 'b b b'


def test_bash_tool():
    out = pkg.bash('echo hello-from-bash')
    assert 'exit=0' in out and 'hello-from-bash' in out


def test_grep_tool(tmp_path: Path):
    (tmp_path / 'a.txt').write_text('alpha\nNEEDLE here\n', encoding='utf-8')
    assert 'NEEDLE here' in pkg.grep('NEEDLE', str(tmp_path))


def test_glob_tool(tmp_path: Path):
    (tmp_path / 'x').mkdir()
    (tmp_path / 'x' / 'a.py').write_text('', encoding='utf-8')
    (tmp_path / 'x' / 'b.txt').write_text('', encoding='utf-8')
    res = pkg.glob_search('**/*.py', str(tmp_path))
    assert 'x/a.py' in res and 'b.txt' not in res


def test_glob_outside_base_is_handled(tmp_path: Path):
    # An absolute pattern resolves outside `base`; must not raise (ValueError
    # from relative_to is caught and reported).
    out = pkg.glob_search('/etc/*', str(tmp_path))
    assert out.startswith('error:') or out == '(no matches)'


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
    """`web_fetch_20250910` is an Anthropic-server-side tool; compat
    endpoints (MiniMax etc.) reject it with HTTP 400. The capability is
    gated by `ANTHROPIC_BASE_URL`."""
    from pydantic_ai.capabilities import NativeTool

    monkeypatch.delenv('ANTHROPIC_BASE_URL', raising=False)
    caps = shim._anthropic_native_capabilities()  # pyright: ignore[reportPrivateUsage]
    assert len(caps) == 1 and isinstance(caps[0], NativeTool)

    monkeypatch.setenv('ANTHROPIC_BASE_URL', 'https://api.anthropic.com')
    assert len(shim._anthropic_native_capabilities()) == 1  # pyright: ignore[reportPrivateUsage]

    monkeypatch.setenv('ANTHROPIC_BASE_URL', 'https://api.minimax.io/anthropic')
    assert shim._anthropic_native_capabilities() == []  # pyright: ignore[reportPrivateUsage]


def test_todo_write_acknowledges():
    out = pkg.todo_write([{'content': 'do x', 'status': 'in_progress', 'activeForm': 'doing x'}])
    assert 'do x' in out and out.startswith('todos recorded')
    assert pkg.todo_write([]) == 'todos recorded (0):\n'


def test_exit_plan_mode_returns_ack():
    assert 'proceeding' in pkg.exit_plan_mode('step 1; step 2').lower()


def test_plan_mode_keeps_new_readonly_tools_drops_multiedit():
    import asyncio

    # Note: WebFetch is an Anthropic server-side capability (not in the callable list).
    names = set(asyncio.run(_toolset_names(None, 'plan')))
    assert 'MultiEdit' not in names  # mutating
    assert {'TodoWrite', 'ExitPlanMode'} <= names  # non-mutating callables


def test_request_limit_is_a_constant():
    assert shim.REQUEST_LIMIT == 200


def test_instructions_encourage_parallel_tool_calls():
    assert shim.INSTRUCTIONS.strip()
    assert 'parallel' in shim.INSTRUCTIONS.lower()


def test_run_routes_workflow_prompt_to_system_instructions(monkeypatch: pytest.MonkeyPatch):
    """Workflow prompt rides in the system instruction; user message is RUN_TRIGGER."""
    import asyncio

    from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, UserPromptPart
    from pydantic_ai.models.function import AgentInfo, FunctionModel

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
    import asyncio

    parent_names = asyncio.run(_toolset_names(None, None, task=shim.task))
    sub_names = asyncio.run(_toolset_names(None, None, task=None))
    assert 'Task' in parent_names
    assert 'Task' not in sub_names


def test_subagent_request_limit_is_a_constant():
    assert shim.SUBAGENT_REQUEST_LIMIT == 75


def test_task_runs_subagent_with_run_model_and_read_only_tools(monkeypatch: pytest.MonkeyPatch):
    # The Task tool spawns a sub-Agent on ctx.model with the read-only tool
    # set, runs the given prompt, and returns the sub-agent's output.
    import asyncio

    from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, UserPromptPart
    from pydantic_ai.models.function import AgentInfo, FunctionModel
    from pydantic_ai.usage import RunUsage

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
    out = pkg.read_file('f.txt')
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
    from pydantic_ai.messages import ModelRequest, UserPromptPart

    msgs: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='hello')]),  # 5
        ModelRequest(parts=[UserPromptPart(content='x' * 20)]),  # 20
    ]
    assert shim._history_size_chars(msgs) == 25  # pyright: ignore[reportPrivateUsage]


def test_compact_history_no_op_below_char_budget(monkeypatch: pytest.MonkeyPatch):
    import asyncio

    from pydantic_ai.messages import ModelRequest, UserPromptPart

    # Many tiny messages — total chars stays well below the default 80k budget.
    msgs: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content=f'm{i}')]) for i in range(100)]

    class _Ctx:
        model = None

    out = asyncio.run(shim._compact_history(cast(RunContext[None], _Ctx()), msgs))  # pyright: ignore[reportPrivateUsage]
    assert out is msgs  # size-based: count alone never triggers


def test_compact_history_summarises_with_fresh_usage_then_merges():
    """Summariser uses a fresh `RunUsage` (so request_limit doesn't trip on the
    parent's running total) and the parent usage absorbs its cost after."""
    import asyncio

    from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
    from pydantic_ai.models.function import FunctionModel
    from pydantic_ai.usage import RunUsage

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
    from pydantic_ai.messages import (
        ModelMessage,
        ModelRequest,
        ModelResponse,
        ToolCallPart,
        ToolReturnPart,
        UserPromptPart,
    )

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
    from pydantic_ai.messages import (
        ModelMessage,
        ModelRequest,
        ModelResponse,
        ToolCallPart,
        ToolReturnPart,
        UserPromptPart,
    )

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
    from pydantic_ai.messages import (
        ModelMessage,
        ModelRequest,
        ModelResponse,
        ToolCallPart,
        ToolReturnPart,
        UserPromptPart,
    )

    tiny_msgs: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content=f'm{i}')]) for i in range(shim.COMPACTION_KEEP_RECENT + 5)
    ]
    with caplog.at_level('INFO', logger='pydantic_ai_gh_aw_shim'):
        shim._trim_tool_results(tiny_msgs)  # pyright: ignore[reportPrivateUsage]
    assert not any('compaction trim' in m for m in caplog.messages)
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
    log_line = next((m for m in caplog.messages if 'compaction trim' in m), None)
    assert log_line is not None
    assert 'deduped 1' in log_line and 'truncated 2' in log_line and 'saved' in log_line


def test_compact_history_uses_trim_alone_when_sufficient(monkeypatch: pytest.MonkeyPatch):
    """Trim alone is enough — the LLM summariser must not fire."""
    import asyncio

    from pydantic_ai.messages import (
        ModelMessage,
        ModelRequest,
        ModelResponse,
        ToolCallPart,
        ToolReturnPart,
        UserPromptPart,
    )

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
    import asyncio

    from pydantic_ai.messages import ModelRequest, UserPromptPart
    from pydantic_ai.models.test import TestModel

    # Same size-driven setup as the previous test — 13 big msgs > trigger.
    big = 'x' * 50_000
    msgs: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content=f'm{i} {big}')]) for i in range(13)]

    class _FailingAgent:
        def __init__(self, *a: object, **k: object) -> None:
            pass

        async def run(self, *a: object, **k: object) -> None:
            raise RuntimeError('boom')

    monkeypatch.setattr(shim, 'Agent', _FailingAgent)

    from pydantic_ai.usage import RunUsage

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
    import asyncio

    from pydantic_ai.messages import ModelRequest, UserPromptPart
    from pydantic_ai.models.test import TestModel

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

    from pydantic_ai.usage import RunUsage

    class _Ctx:
        model = TestModel()
        usage = RunUsage()

    out = asyncio.run(shim._compact_history(cast(RunContext[None], _Ctx()), msgs))  # pyright: ignore[reportPrivateUsage]
    # First element is the preserved prior synthetic; rest is the tail.
    assert len(out) == 1 + shim.COMPACTION_KEEP_RECENT
    assert out[0] is prior_synthetic


def test_task_surfaces_subagent_failure_as_tool_result(monkeypatch: pytest.MonkeyPatch):
    import asyncio

    from pydantic_ai.models.test import TestModel

    class _FailingAgent:
        def __init__(self, *a: object, **k: object) -> None:
            pass

        async def run(self, *a: object, **k: object) -> None:
            raise RuntimeError('downstream model exploded')

    monkeypatch.setattr(shim, 'Agent', _FailingAgent)

    from pydantic_ai.usage import RunUsage

    class _Ctx:
        model = TestModel()
        usage = RunUsage()

    out = asyncio.run(shim.task(cast(RunContext[None], _Ctx()), 'x', 'y'))
    assert out == 'error: sub-agent failed: downstream model exploded'


def test_task_isolates_attach_context_dedupe_set_from_parent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Sub-agents start with a fresh AGENTS.md seen-set, not the parent's."""
    import asyncio

    from pydantic_ai.messages import ModelResponse, TextPart
    from pydantic_ai.models.function import AgentInfo, FunctionModel
    from pydantic_ai.usage import RunUsage

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
    import asyncio

    from pydantic_ai.messages import (
        FunctionToolCallEvent,
        FunctionToolResultEvent,
        ToolCallPart,
        ToolReturnPart,
    )

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
    import asyncio

    from pydantic_ai.messages import FunctionToolResultEvent, ToolReturnPart

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


def test_stream_events_tags_retry_prompt_as_error():
    """`ToolResultEvent.part` is `ToolReturnPart | RetryPromptPart`. A retry
    means tool-call validation failed — gh-aw must see `is_error=True` so it
    doesn't read it as a successful result."""
    import asyncio

    from pydantic_ai.messages import FunctionToolResultEvent, RetryPromptPart, ToolReturnPart

    async def _events():
        yield FunctionToolResultEvent(
            part=ToolReturnPart(tool_name='Bash', content='ok', tool_call_id='c1'),
        )
        yield FunctionToolResultEvent(
            part=RetryPromptPart(content='Validation failed', tool_name='Bash', tool_call_id='c2'),
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
def test_read_missing_file_returns_error(tmp_path: Path):
    out = pkg.read_file(str(tmp_path / 'nope.txt'))
    assert out.startswith('error:')


def test_edit_missing_file_returns_error(tmp_path: Path):
    out = pkg.edit_file(str(tmp_path / 'missing.txt'), 'old', 'new')
    assert out.startswith('error:')


def test_multi_edit_missing_file_returns_error(tmp_path: Path):
    out = pkg.multi_edit(str(tmp_path / 'absent.txt'), [{'old_string': 'a', 'new_string': 'b'}])
    assert out.startswith('error:')


def test_list_dir_missing_path_returns_error(tmp_path: Path):
    out = pkg.list_dir(str(tmp_path / 'nope'))
    assert out.startswith('error:')


def test_write_to_existing_parent_succeeds_otherwise_creates(tmp_path: Path):
    # write_file's own `parent.mkdir(parents=True, exist_ok=True)` handles
    # missing parents; the OSError path triggers only on a real permission
    # error or invalid path. Verify the happy + nested-parent paths here.
    nested = tmp_path / 'a' / 'b' / 'c.txt'
    assert 'wrote' in pkg.write_file(str(nested), 'ok')
    assert nested.read_text(encoding='utf-8') == 'ok'


# --------------------------------------------------------------------------- #
# grep — ripgrep-only after dropping the Python fallback
# --------------------------------------------------------------------------- #
def test_grep_returns_error_when_ripgrep_missing(monkeypatch: pytest.MonkeyPatch):
    # The fallback is gone; if `rg` isn't on PATH we surface a clean error
    # so the agent can fall back to `Bash` (`grep -rn …`) on its own.
    monkeypatch.setattr(pkg.grep.__globals__['shutil'], 'which', lambda _name: None)  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    out = pkg.grep('NEEDLE', '.')
    assert out.startswith('error:') and 'ripgrep' in out


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


def test_run_with_timeout_emits_error_on_global_timeout(monkeypatch: pytest.MonkeyPatch):
    async def _hang(*_a: object, **_kw: object) -> int:
        await asyncio.sleep(9999)
        return 0

    monkeypatch.setattr(shim, 'run', _hang)
    monkeypatch.setattr(shim, 'RUN_TIMEOUT_SECS', 0.01)
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
    from pydantic_ai.toolsets import PrefixedToolset

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


def test_mcp_allow_predicate_server_wildcard_vs_specific():
    from pydantic_ai.tools import ToolDefinition

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
def test_emit_result_matches_claude_stream_json_schema():
    buf = io.StringIO()
    with redirect_stdout(buf):
        shim.emit_result('answer', usage=None, session_id='run-1')
    obj = json.loads(buf.getvalue().strip())
    assert obj['type'] == 'result'
    assert obj['subtype'] == 'success'
    assert obj['is_error'] is False
    assert obj['result'] == 'answer'
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


def test_emit_result_reads_usage_attributes():
    class U:
        input_tokens = 22
        output_tokens = 292
        cache_write_tokens = 5
        cache_read_tokens = 7

    from pydantic_ai.usage import RunUsage as _RunUsage

    buf = io.StringIO()
    with redirect_stdout(buf):
        shim.emit_result('x', usage=cast(_RunUsage, U()), session_id='s')
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
    monkeypatch.setattr(sys, 'argv', ['pydantic-ai-runner', '--print', 'hello'])
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = shim.main()
    assert rc == 1
    obj = json.loads(buf.getvalue().strip())
    assert obj['is_error'] is True
    assert 'shim startup failed' in obj['result']
    assert 'kaboom' in obj['result']


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
def test_live_anthropic_compatible_endpoint(monkeypatch: pytest.MonkeyPatch):
    """End-to-end against a real Anthropic-shape endpoint (api.anthropic.com,
    MiniMax's /anthropic, etc.). Verifies the shim+endpoint integration —
    not the model's instruction-following.
    """
    monkeypatch.setenv('ANTHROPIC_API_KEY', os.environ['GH_AW_SHIM_LIVE_API_KEY'])
    monkeypatch.setenv(
        'ANTHROPIC_BASE_URL',
        os.environ.get('GH_AW_SHIM_LIVE_BASE_URL', 'https://api.anthropic.com'),
    )
    model = os.environ.get('GH_AW_SHIM_LIVE_MODEL', 'claude-sonnet-4-6')
    argv = list(GHAW_ARGV)
    i = argv.index('--mcp-config')
    del argv[i : i + 2]  # no MCP gateway outside a gh-aw run
    argv += ['--model', model, 'Say hi.']
    monkeypatch.setattr(sys, 'argv', ['pydantic-ai-runner', *argv])
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = shim.main()
    assert rc == 0
    lines = [json.loads(x) for x in buf.getvalue().splitlines() if x.strip()]
    result = next(x for x in lines if x['type'] == 'result')
    assert result['is_error'] is False
    assert result['result']
    # `input_tokens > 0` proves the prompt round-tripped; `output_tokens > 0`
    # proves the model actually responded.
    assert result['usage']['input_tokens'] > 0
    assert result['usage']['output_tokens'] > 0
