"""Offline tests for the Pydantic AI gh-aw harness (.github/scripts/pydantic-ai-runner).

These cover the gh-aw compatibility surface with no network or credentials:
argv tolerance, prompt recovery, model resolution, MCP-config translation and
allow-list filtering, Claude-named native tools, `--allowed-tools` /
`--permission-mode` enforcement, structured-error guarantees, and the
stream-json schema.

The single live test is skipped unless an OpenAI-compatible endpoint is given
via env: GH_AW_HARNESS_LIVE_API_KEY / _BASE_URL / _MODEL.

Run:  uv run --with pytest pytest .github/scripts/test_pydantic_ai_runner.py
"""

import importlib.machinery
import importlib.util
import io
import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parent / "pydantic-ai-runner"


def _load_harness():
    # The harness has no .py extension, so use an explicit source loader.
    # Register in sys.modules before exec so dataclasses can resolve the
    # module (required with `from __future__ import annotations`).
    loader = importlib.machinery.SourceFileLoader("par_harness", str(_SCRIPT))
    spec = importlib.util.spec_from_loader("par_harness", loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    loader.exec_module(mod)
    return mod


har = _load_harness()


# The exact argv shape gh-aw's claude_harness.cjs passes, prompt appended last.
GHAW_ARGV = [
    "--print",
    "--no-chrome",
    "--allowed-tools",
    "Bash,Read,Edit(/tmp/*),mcp__github__get_me,mcp__safeoutputs",
    "--debug-file",
    "/tmp/gh-aw/agent-stdio.log",
    "--verbose",
    "--permission-mode",
    "bypassPermissions",
    "--output-format",
    "stream-json",
    "--mcp-config",
    "/tmp/mcp-servers.json",
    "--prompt-file",
    "/tmp/gh-aw/aw-prompts/prompt.txt",
]


# --------------------------------------------------------------------------- #
# argv / prompt
# --------------------------------------------------------------------------- #
def test_parses_full_claude_argv_without_error():
    args = har.parse_args([*GHAW_ARGV, "do the thing"])
    assert args.mcp_config == "/tmp/mcp-servers.json"
    assert args.prompt_file == "/tmp/gh-aw/aw-prompts/prompt.txt"
    assert args.prompt_positional == "do the thing"
    assert args.permission_mode == "bypassPermissions"


def test_unknown_future_claude_flags_are_tolerated():
    args = har.parse_args([*GHAW_ARGV, "--some-future-flag", "x", "prompt"])
    assert args.prompt_positional == "prompt"


def test_prompt_recovered_from_trailing_positional():
    args = har.parse_args([*GHAW_ARGV, "Investigate the failing CI run."])
    assert har.resolve_prompt(args) == "Investigate the failing CI run."


def test_prompt_falls_back_to_prompt_file(tmp_path):
    pf = tmp_path / "prompt.txt"
    pf.write_text("from file", encoding="utf-8")
    args = har.parse_args(["--prompt-file", str(pf)])
    assert har.resolve_prompt(args) == "from file"


def test_prompt_falls_back_to_env(tmp_path, monkeypatch):
    pf = tmp_path / "p.txt"
    pf.write_text("from env path", encoding="utf-8")
    monkeypatch.setenv("GH_AW_PROMPT", str(pf))
    assert har.resolve_prompt(har.parse_args(["--print"])) == "from env path"


# --------------------------------------------------------------------------- #
# --allowed-tools parsing & enforcement
# --------------------------------------------------------------------------- #
def test_allowed_tools_absent_is_none():
    assert har.parse_args(["--print"]).allowed_tools is None
    assert har._split_allowed_tools(None) is None


def test_allowed_tools_parsed_and_scope_stripped():
    args = har.parse_args([*GHAW_ARGV, "p"])
    assert args.allowed_tools == frozenset(
        {"Bash", "Read", "Edit", "mcp__github__get_me", "mcp__safeoutputs"}
    )


def test_select_native_tools_no_allowlist_keeps_all():
    tools = har.select_native_tools(None, None)
    assert [t.name for t in tools] == list(har.NATIVE_TOOL_NAMES)


def test_select_native_tools_enforces_allowlist():
    tools = har.select_native_tools(frozenset({"Bash", "Read", "mcp__safeoutputs"}), None)
    assert [t.name for t in tools] == ["Bash", "Read"]


def test_plan_mode_withholds_mutating_tools():
    tools = har.select_native_tools(None, "plan")
    names = {t.name for t in tools}
    assert names.isdisjoint(har.MUTATING_TOOLS)
    assert "Read" in names and "Grep" in names and "Glob" in names


def test_plan_mode_and_allowlist_compose():
    tools = har.select_native_tools(frozenset({"Bash", "Read"}), "plan")
    assert [t.name for t in tools] == ["Read"]  # Bash dropped by plan mode


def test_native_tools_registry_uses_claude_names():
    assert tuple(har.NATIVE_TOOLS) == har.NATIVE_TOOL_NAMES
    assert har.NATIVE_TOOL_NAMES == (
        "Bash",
        "Read",
        "Write",
        "Edit",
        "MultiEdit",
        "Grep",
        "Glob",
        "LS",
        "WebFetch",
        "TodoWrite",
        "ExitPlanMode",
        "Task",
    )


# --------------------------------------------------------------------------- #
# native tool behavior
# --------------------------------------------------------------------------- #
def test_native_file_tools_roundtrip(tmp_path):
    f = tmp_path / "sub" / "note.txt"
    assert "wrote" in har.write_file(str(f), "hello\nworld\n")
    assert har.read_file(str(f)) == "hello\nworld\n"
    assert "edited" in har.edit_file(str(f), "world", "there")
    assert "there" in har.read_file(str(f))
    assert "note.txt" in har.list_dir(str(tmp_path / "sub"))
    assert har.edit_file(str(f), "absent", "x") == "error: `old_string` not found"


def test_read_file_offset_and_limit(tmp_path):
    f = tmp_path / "n.txt"
    f.write_text("l1\nl2\nl3\nl4\n", encoding="utf-8")
    assert har.read_file(str(f), offset=2, limit=2) == "l2\nl3"


def test_edit_file_replace_all(tmp_path):
    f = tmp_path / "r.txt"
    f.write_text("a a a", encoding="utf-8")
    har.edit_file(str(f), "a", "b", replace_all=True)
    assert f.read_text(encoding="utf-8") == "b b b"


def test_native_bash_tool():
    out = har.bash("echo hello-from-bash")
    assert "exit=0" in out and "hello-from-bash" in out


def test_native_grep_tool(tmp_path):
    (tmp_path / "a.txt").write_text("alpha\nNEEDLE here\n", encoding="utf-8")
    assert "NEEDLE here" in har.grep("NEEDLE", str(tmp_path))


def test_native_glob_tool(tmp_path):
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "a.py").write_text("", encoding="utf-8")
    (tmp_path / "x" / "b.txt").write_text("", encoding="utf-8")
    res = har.glob_search("**/*.py", str(tmp_path))
    assert "x/a.py" in res and "b.txt" not in res


def test_glob_outside_base_is_handled(tmp_path):
    # An absolute pattern resolves outside `base`; must not raise (ValueError
    # from relative_to is caught and reported).
    out = har.glob_search("/etc/*", str(tmp_path))
    assert out.startswith("error:") or out == "(no matches)"


def test_multi_edit_atomic(tmp_path):
    f = tmp_path / "m.txt"
    f.write_text("one two three", encoding="utf-8")
    ok = har.multi_edit(str(f), [{"old_string": "one", "new_string": "1"},
                                 {"old_string": "three", "new_string": "3"}])
    assert "applied 2 edit(s)" in ok
    assert f.read_text(encoding="utf-8") == "1 two 3"
    # A failing edit writes nothing (atomic).
    res = har.multi_edit(str(f), [{"old_string": "1", "new_string": "X"},
                                  {"old_string": "absent", "new_string": "Y"}])
    assert "edit #2" in res and "not found" in res
    assert f.read_text(encoding="utf-8") == "1 two 3"


def test_multi_edit_replace_all(tmp_path):
    f = tmp_path / "r.txt"
    f.write_text("a a a", encoding="utf-8")
    har.multi_edit(str(f), [{"old_string": "a", "new_string": "b", "replace_all": True}])
    assert f.read_text(encoding="utf-8") == "b b b"


def test_page_to_text_strips_html():
    assert har._page_to_text("<html><body>Hi <b>there</b><script>x()</script></body></html>") == "Hi there"


def test_fetch_page_rejects_non_http():
    assert har._fetch_page("ftp://x/y").startswith("error:")
    assert har._fetch_page("file:///etc/passwd").startswith("error:")


def test_fetch_page_success_and_network_error(monkeypatch):
    monkeypatch.setattr(har, "_http_get", lambda url, timeout=20.0: (200, "<p>Hello</p>"))
    out = har._fetch_page("https://example.test")
    assert out.startswith("HTTP 200 for https://example.test") and "Hello" in out

    def boom(url, timeout=20.0):
        raise RuntimeError("blocked by firewall")

    monkeypatch.setattr(har, "_http_get", boom)
    assert har._fetch_page("https://blocked.test") == "error: fetch failed: blocked by firewall"


def test_web_fetch_summarizes_via_run_model(monkeypatch):
    # Faithful WebFetch: fetch then answer the prompt with the run's model.
    # Drive it with a stub ctx exposing .model (a TestModel) — offline.
    import asyncio

    from pydantic_ai.models.test import TestModel

    monkeypatch.setattr(
        har, "_http_get", lambda url, timeout=20.0: (200, "<html>The sky is blue.</html>")
    )

    class _Ctx:
        model = TestModel(custom_output_text="SUMMARY: sky is blue")

    out = asyncio.run(har.web_fetch(_Ctx(), "https://example.test", "what colour is the sky?"))
    assert out == "SUMMARY: sky is blue"
    # A fetch error short-circuits before the model is consulted.
    monkeypatch.setattr(har, "_http_get", lambda url, timeout=20.0: (_ for _ in ()).throw(RuntimeError("nope")))
    err = asyncio.run(har.web_fetch(_Ctx(), "https://blocked.test", "x"))
    assert err == "error: fetch failed: nope"


def test_todo_write_acknowledges():
    out = har.todo_write(
        [{"content": "do x", "status": "in_progress", "activeForm": "doing x"}]
    )
    assert "do x" in out and out.startswith("todo list")
    assert har.todo_write([]) == "todo list cleared"


def test_exit_plan_mode_returns_ack():
    assert "proceeding" in har.exit_plan_mode("step 1; step 2").lower()


def test_plan_mode_keeps_new_readonly_tools_drops_multiedit():
    names = {t.name for t in har.select_native_tools(None, "plan")}
    assert "MultiEdit" not in names  # mutating
    assert {"WebFetch", "TodoWrite", "ExitPlanMode"} <= names  # non-mutating


def test_request_limit_default_and_override(monkeypatch):
    monkeypatch.delenv("GH_AW_HARNESS_REQUEST_LIMIT", raising=False)
    assert har._request_limit() == har.DEFAULT_REQUEST_LIMIT == 200
    monkeypatch.setenv("GH_AW_HARNESS_REQUEST_LIMIT", "350")
    assert har._request_limit() == 350
    for bad in ("0", "-5", "abc", ""):
        monkeypatch.setenv("GH_AW_HARNESS_REQUEST_LIMIT", bad)
        assert har._request_limit() == 200


def test_instructions_encourage_parallel_tool_calls():
    assert har.INSTRUCTIONS.strip()
    assert "parallel" in har.INSTRUCTIONS.lower()


def test_run_routes_workflow_prompt_to_system_instructions(monkeypatch):
    """The workflow prompt rides as a system instruction; the user message
    is just `RUN_TRIGGER`. Weaker models follow system instructions much
    more strictly than user messages, so this wiring is load-bearing for
    multi-step workflow prompts."""
    import asyncio

    seen: dict[str, object] = {}

    class _CapturingAgent:
        def __init__(self, model, instructions=None, **_):  # type: ignore[no-untyped-def]
            seen["instructions"] = instructions

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):  # type: ignore[no-untyped-def]
            return False

        async def run(self, prompt, usage_limits=None, **_):  # type: ignore[no-untyped-def]
            seen["run_prompt"] = prompt

            class _Usage:
                requests = 0
                input_tokens = 0
                output_tokens = 0

            class _Result:
                output = "done"
                usage = _Usage()

                @staticmethod
                def all_messages():
                    return []

            return _Result()

    monkeypatch.setattr(har, "Agent", _CapturingAgent)
    monkeypatch.setattr(har, "emit", lambda *a, **k: None)
    monkeypatch.setattr(har, "log_safe_outputs_state", lambda: None)

    sentinel = "### WORKFLOW TASK SPEC: review the PR per the rules above ###"
    asyncio.run(
        har.run(
            prompt=sentinel,
            model=object(),
            label="test-model",
            native_tools=[],
            mcp_servers=[],
            session_id="test-session",
        )
    )

    instructions = str(seen["instructions"])
    assert har.INSTRUCTIONS in instructions  # baseline parallel-tool guidance
    assert sentinel in instructions  # workflow prompt embedded in system instructions

    # User message is the trivial trigger — task spec MUST NOT leak there.
    assert seen["run_prompt"] == har.RUN_TRIGGER
    assert sentinel not in str(seen["run_prompt"])


def test_read_only_subagent_tools_are_non_mutating_and_exclude_task():
    assert har.READ_ONLY_SUBAGENT_TOOLS.isdisjoint(har.MUTATING_TOOLS)
    assert "Task" not in har.READ_ONLY_SUBAGENT_TOOLS  # no recursion
    # All entries are real native tool names.
    assert har.READ_ONLY_SUBAGENT_TOOLS <= set(har.NATIVE_TOOL_NAMES)


def test_task_registered_in_native_tools():
    assert "Task" in har.NATIVE_TOOLS and "Task" in har.NATIVE_TOOL_NAMES


def test_subagent_request_limit_default_and_override(monkeypatch):
    monkeypatch.delenv("GH_AW_HARNESS_SUBAGENT_REQUEST_LIMIT", raising=False)
    assert har._subagent_request_limit() == har.DEFAULT_SUBAGENT_REQUEST_LIMIT == 75
    monkeypatch.setenv("GH_AW_HARNESS_SUBAGENT_REQUEST_LIMIT", "120")
    assert har._subagent_request_limit() == 120
    for bad in ("0", "-1", "x", ""):
        monkeypatch.setenv("GH_AW_HARNESS_SUBAGENT_REQUEST_LIMIT", bad)
        assert har._subagent_request_limit() == 75


def test_task_runs_subagent_with_run_model_and_read_only_tools(monkeypatch):
    # The Task tool spawns a sub-Agent on ctx.model with the read-only tool
    # set, runs the given prompt, and returns the sub-agent's output.
    import asyncio

    from pydantic_ai.models.test import TestModel

    seen: dict[str, object] = {}

    class _CapturingAgent:
        def __init__(self, model, instructions=None, tools=None, toolsets=None, event_stream_handler=None, **_):  # type: ignore[no-untyped-def]
            seen["model_cls"] = type(model).__name__
            seen["instructions"] = instructions
            seen["tool_names"] = [t.name for t in (tools or [])]
            seen["event_stream_handler"] = event_stream_handler

        async def run(self, prompt, usage_limits=None, usage=None):  # type: ignore[no-untyped-def]
            seen["prompt"] = prompt
            seen["request_limit"] = getattr(usage_limits, "request_limit", None)
            seen["usage_obj"] = usage  # parent's RunUsage forwarded for aggregation

            class _Result:
                output = "SUB: investigated"

            return _Result()

    monkeypatch.setattr(har, "Agent", _CapturingAgent)

    from pydantic_ai.usage import RunUsage

    parent_usage = RunUsage()

    class _Ctx:
        model = TestModel()
        usage = parent_usage

    out = asyncio.run(har.task(_Ctx(), "scan models/openai.py", "find tool_call_id bugs"))
    assert out == "SUB: investigated"
    assert seen["model_cls"] == "TestModel"
    # Sub-agent uses the same system-prompt routing as the parent: the
    # task spec rides in `instructions=` (alongside the general
    # INSTRUCTIONS + SUBAGENT_INSTRUCTIONS preamble), and the user
    # message is the trivial `RUN_TRIGGER`. This makes weaker models
    # follow long structured sub-agent prompts much more reliably.
    sub_instructions = str(seen["instructions"])
    assert har.INSTRUCTIONS in sub_instructions
    assert har.SUBAGENT_INSTRUCTIONS in sub_instructions
    assert "find tool_call_id bugs" in sub_instructions
    assert seen["prompt"] == har.RUN_TRIGGER
    assert "find tool_call_id bugs" not in str(seen["prompt"])
    assert set(seen["tool_names"]) == har.READ_ONLY_SUBAGENT_TOOLS  # type: ignore[arg-type]
    assert "Task" not in seen["tool_names"]  # type: ignore[operator]
    assert "Bash" not in seen["tool_names"]  # type: ignore[operator]
    assert seen["request_limit"] == har.DEFAULT_SUBAGENT_REQUEST_LIMIT
    assert seen["usage_obj"] is parent_usage  # sub-agent tokens roll up into parent
    # Sub-agent gets the same live event handler as the parent so its tool
    # calls stream out interleaved with the parent's (it's spawned inside
    # the parent's `Task` tool execution).
    assert seen["event_stream_handler"] is har._stream_events


# --------------------------------------------------------------------------- #
# directory-scoped AGENTS.md / CLAUDE.md auto-loading
# --------------------------------------------------------------------------- #
def test_attach_context_surfaces_files_once_per_run(monkeypatch, tmp_path):
    # AGENTS.md at root of the "workspace" + CLAUDE.md in a subdir.
    monkeypatch.setattr(har, "WORKSPACE", str(tmp_path))
    (tmp_path / "AGENTS.md").write_text("# repo conventions", encoding="utf-8")
    sub = tmp_path / "pkg"
    sub.mkdir()
    (sub / "CLAUDE.md").write_text("# pkg conventions", encoding="utf-8")
    (sub / "code.py").write_text("x = 1\n", encoding="utf-8")
    har._reset_context_state()

    first = har._attach_context("pkg/code.py")
    assert "context: pkg/CLAUDE.md" in first  # nearest first when walking up
    assert "context: AGENTS.md" in first
    assert "pkg conventions" in first and "repo conventions" in first

    # Subsequent calls in same run dedupe.
    again = har._attach_context("pkg/code.py")
    assert again == ""

    # A different path under the same dir hits no new context files.
    (sub / "other.py").write_text("", encoding="utf-8")
    assert har._attach_context("pkg/other.py") == ""


def test_attach_context_truncates_large_files(monkeypatch, tmp_path):
    monkeypatch.setattr(har, "WORKSPACE", str(tmp_path))
    big = "X" * (har.MAX_CONTEXT_FILE_CHARS + 5000)
    (tmp_path / "AGENTS.md").write_text(big, encoding="utf-8")
    har._reset_context_state()
    out = har._attach_context(".")
    # Body of the AGENTS.md block is capped to MAX_CONTEXT_FILE_CHARS.
    body = out.split("---\n", 2)[-1]
    assert len(body) <= har.MAX_CONTEXT_FILE_CHARS + 50  # +slack for trailing markers


def test_attach_context_empty_for_missing_path(monkeypatch, tmp_path):
    monkeypatch.setattr(har, "WORKSPACE", str(tmp_path))
    har._reset_context_state()
    assert har._attach_context(None) == ""
    assert har._attach_context("does-not-exist.py") == ""  # parent has no AGENTS.md/CLAUDE.md


def test_read_file_prepends_context(monkeypatch, tmp_path):
    monkeypatch.setattr(har, "WORKSPACE", str(tmp_path))
    (tmp_path / "AGENTS.md").write_text("repo rules", encoding="utf-8")
    (tmp_path / "f.txt").write_text("file body", encoding="utf-8")
    har._reset_context_state()
    out = har.read_file("f.txt")
    assert "context: AGENTS.md" in out and "repo rules" in out and "file body" in out


# --------------------------------------------------------------------------- #
# history compaction (ProcessHistory capability)
# --------------------------------------------------------------------------- #
def test_compaction_trigger_chars_resolution(monkeypatch):
    for v in (
        "GH_AW_HARNESS_COMPACTION_TRIGGER_CHARS",
        "GH_AW_HARNESS_MODEL_CONTEXT_CHARS",
        "GH_AW_HARNESS_COMPACTION_FRACTION",
    ):
        monkeypatch.delenv(v, raising=False)
    # Default: 800k * 0.7 = 560k chars (≈ 70% of a 200k-token context).
    assert har._compaction_trigger_chars() == 560_000
    assert har.DEFAULT_COMPACTION_TRIGGER_CHARS == 560_000

    # Component overrides compose.
    monkeypatch.setenv("GH_AW_HARNESS_MODEL_CONTEXT_CHARS", "400000")
    monkeypatch.setenv("GH_AW_HARNESS_COMPACTION_FRACTION", "0.5")
    assert har._compaction_trigger_chars() == 200_000

    # Direct trigger wins over derivation, even when components are also set.
    monkeypatch.setenv("GH_AW_HARNESS_COMPACTION_TRIGGER_CHARS", "12345")
    assert har._compaction_trigger_chars() == 12345

    # Bad fraction values fall back to default.
    monkeypatch.delenv("GH_AW_HARNESS_COMPACTION_TRIGGER_CHARS")
    monkeypatch.delenv("GH_AW_HARNESS_MODEL_CONTEXT_CHARS")
    for bad in ("0", "-1", "1.5", "abc", ""):
        monkeypatch.setenv("GH_AW_HARNESS_COMPACTION_FRACTION", bad)
        assert har._compaction_trigger_chars() == int(
            har.DEFAULT_MODEL_CONTEXT_CHARS * har.DEFAULT_COMPACTION_FRACTION
        )


def test_history_size_chars_sums_all_part_content():
    from pydantic_ai.messages import ModelRequest, UserPromptPart

    msgs = [
        ModelRequest(parts=[UserPromptPart(content="hello")]),  # 5
        ModelRequest(parts=[UserPromptPart(content="x" * 20)]),  # 20
    ]
    assert har._history_size_chars(msgs) == 25


def test_compact_history_no_op_below_char_budget(monkeypatch):
    import asyncio

    from pydantic_ai.messages import ModelRequest, UserPromptPart

    # Many tiny messages — total chars stays well below the default 80k budget.
    msgs = [ModelRequest(parts=[UserPromptPart(content=f"m{i}")]) for i in range(100)]

    class _Ctx:
        model = None

    out = asyncio.run(har._compact_history(_Ctx(), msgs))
    assert out is msgs  # size-based: count alone never triggers


def test_compact_history_summarises_via_run_model(monkeypatch):
    import asyncio

    from pydantic_ai.messages import ModelRequest, UserPromptPart
    from pydantic_ai.models.test import TestModel

    monkeypatch.setenv("GH_AW_HARNESS_COMPACTION_TRIGGER_CHARS", "1000")
    monkeypatch.setenv("GH_AW_HARNESS_COMPACTION_KEEP_RECENT", "3")

    # 12 messages of ~120 chars each ≈ 1.4k chars — exceeds the 1000-char trigger.
    msgs = [
        ModelRequest(parts=[UserPromptPart(content=f"m{i} " + "x" * 120)]) for i in range(12)
    ]

    seen_usage: dict[str, object] = {}

    class _FakeAgent:
        def __init__(self, model, instructions=None):  # type: ignore[no-untyped-def]
            self.model = model

        async def run(self, prompt, usage_limits=None, usage=None):  # type: ignore[no-untyped-def]
            seen_usage["usage"] = usage  # parent's RunUsage forwarded

            class _R:
                output = "SHORT SUMMARY"

            return _R()

    monkeypatch.setattr(har, "Agent", _FakeAgent)

    from pydantic_ai.usage import RunUsage

    parent_usage = RunUsage()

    class _Ctx:
        model = TestModel()
        usage = parent_usage

    out = asyncio.run(har._compact_history(_Ctx(), msgs))
    # synthetic summary (1) + last KEEP_RECENT (3) = 4. We no longer
    # preserve a "head" message — pydantic-ai re-applies the system
    # `instructions=` on every request, so the task spec stays visible
    # without our help.
    assert len(out) == 4
    # Compaction summariser cost rolled up into the parent's usage.
    assert seen_usage["usage"] is parent_usage
    # Summary present in the leading synthetic ModelRequest.
    summary_msg = out[0]
    assert any("SHORT SUMMARY" in getattr(p, "content", "") for p in summary_msg.parts)


def test_compact_history_falls_back_to_truncation_on_failure(monkeypatch):
    import asyncio

    from pydantic_ai.messages import ModelRequest, UserPromptPart
    from pydantic_ai.models.test import TestModel

    monkeypatch.setenv("GH_AW_HARNESS_COMPACTION_TRIGGER_CHARS", "1000")
    monkeypatch.setenv("GH_AW_HARNESS_COMPACTION_KEEP_RECENT", "3")

    msgs = [
        ModelRequest(parts=[UserPromptPart(content=f"m{i} " + "x" * 120)]) for i in range(12)
    ]

    class _FailingAgent:
        def __init__(self, *a, **k):  # type: ignore[no-untyped-def]
            pass

        async def run(self, *a, **k):  # type: ignore[no-untyped-def]
            raise RuntimeError("boom")

    monkeypatch.setattr(har, "Agent", _FailingAgent)

    from pydantic_ai.usage import RunUsage

    class _Ctx:
        model = TestModel()
        usage = RunUsage()

    out = asyncio.run(har._compact_history(_Ctx(), msgs))
    # On failure: keep just the tail (no head, no synthetic summary) = 3
    assert len(out) == 3


def test_task_surfaces_subagent_failure_as_tool_result(monkeypatch):
    import asyncio

    from pydantic_ai.models.test import TestModel

    class _FailingAgent:
        def __init__(self, *a, **k):  # type: ignore[no-untyped-def]
            pass

        async def run(self, *a, **k):  # type: ignore[no-untyped-def]
            raise RuntimeError("downstream model exploded")

    monkeypatch.setattr(har, "Agent", _FailingAgent)

    from pydantic_ai.usage import RunUsage

    class _Ctx:
        model = TestModel()
        usage = RunUsage()

    out = asyncio.run(har.task(_Ctx(), "x", "y"))
    assert out == "error: sub-agent failed: downstream model exploded"


# --------------------------------------------------------------------------- #
# model resolution (proxy semantics — unchanged)
# --------------------------------------------------------------------------- #
def test_model_defaults_to_anthropic(monkeypatch):
    for v in (
        "GH_AW_HARNESS_MODEL",
        "GH_AW_MODEL_AGENT_CLAUDE",
        "ANTHROPIC_MODEL",
        "OPENAI_BASE_URL",
        "ANTHROPIC_BASE_URL",
    ):
        monkeypatch.delenv(v, raising=False)
    model, label = har.build_model(har.parse_args(["--print"]))
    assert label == "anthropic:claude-sonnet-4-5"
    assert model.__class__.__name__ == "AnthropicModel"


def test_model_openai_prefix_builds_openai_model(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    model, label = har.build_model(har.parse_args(["--model", "openai:gpt-4o-mini"]))
    assert label == "openai-compatible:gpt-4o-mini"
    assert model.__class__.__name__ == "OpenAIChatModel"


def test_anthropic_base_url_builds_anthropic_model(monkeypatch):
    for v in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GH_AW_HARNESS_API_KEY"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.minimax.io/anthropic")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "placeholder")
    monkeypatch.setenv("GH_AW_HARNESS_MODEL", "MiniMax-M2.7-highspeed")
    model, label = har.build_model(har.parse_args(["--print"]))
    assert label == "anthropic:MiniMax-M2.7-highspeed"
    assert model.__class__.__name__ == "AnthropicModel"


def test_first_env_precedence(monkeypatch):
    for v in ("A", "B", "C"):
        monkeypatch.delenv(v, raising=False)
    assert har._first_env("A", "B", "C") is None
    monkeypatch.setenv("B", "vb")
    monkeypatch.setenv("C", "vc")
    assert har._first_env("A", "B", "C") == "vb"


def test_use_openai_provider_selection():
    assert har._use_openai_provider("openai", None, None) is True
    assert har._use_openai_provider("anthropic", "http://x", None) is False
    assert har._use_openai_provider("", "http://x", None) is True
    assert har._use_openai_provider("", None, "http://proxy") is False
    assert har._use_openai_provider("", None, None) is False


# --------------------------------------------------------------------------- #
# MCP translation & allow-list filtering
# --------------------------------------------------------------------------- #
def test_mcp_missing_config_degrades_gracefully():
    assert har.build_mcp_servers(har.Args(mcp_config="/no/such/file.json")) == []


def _mcp_cfg(tmp_path):
    cfg = tmp_path / "mcp.json"
    cfg.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "github": {"command": "docker", "args": ["run"], "env": {"X": "1"}},
                    "safeoutputs": {
                        "type": "http",
                        "url": "http://host.docker.internal:1234",
                        "headers": {"Authorization": "k"},
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    return cfg


def test_mcp_translates_stdio_and_http_unfiltered(tmp_path):
    servers = har.build_mcp_servers(har.Args(mcp_config=str(_mcp_cfg(tmp_path))))
    assert len(servers) == 2
    assert {s.__class__.__name__ for s in servers} == {"MCPToolset"}


def test_mcp_wrapped_in_filter_when_allowlist_present(tmp_path):
    servers = har.build_mcp_servers(
        har.Args(mcp_config=str(_mcp_cfg(tmp_path)), allowed_tools=frozenset({"mcp__safeoutputs"}))
    )
    assert len(servers) == 2
    assert {s.__class__.__name__ for s in servers} == {"FilteredToolset"}


def test_mcp_allow_predicate_server_wildcard_vs_specific():
    class _TD:  # minimal stand-in for ToolDefinition (predicate only reads .name)
        def __init__(self, name):
            self.name = name

    # whole-server allow
    pred = har._mcp_tool_allowed("safeoutputs", frozenset({"mcp__safeoutputs"}))
    assert pred(None, _TD("add_comment")) is True
    assert pred(None, _TD("create_issue")) is True

    # specific-tool allow only
    pred = har._mcp_tool_allowed("github", frozenset({"mcp__github__get_me"}))
    assert pred(None, _TD("get_me")) is True
    assert pred(None, _TD("delete_repo")) is False


# --------------------------------------------------------------------------- #
# stream-json schema & structured-error guarantee
# --------------------------------------------------------------------------- #
def test_emit_result_matches_claude_stream_json_schema():
    buf = io.StringIO()
    with redirect_stdout(buf):
        har.emit_result("answer", usage=None, session_id="run-1")
    obj = json.loads(buf.getvalue().strip())
    assert obj["type"] == "result"
    assert obj["subtype"] == "success"
    assert obj["is_error"] is False
    assert obj["result"] == "answer"
    for k in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"):
        assert k in obj["usage"]


def test_emit_result_passes_through_turns_and_duration():
    buf = io.StringIO()
    with redirect_stdout(buf):
        har.emit_result("x", usage=None, session_id="s", num_turns=3, duration_ms=1234)
    obj = json.loads(buf.getvalue().strip())
    assert obj["num_turns"] == 3 and obj["duration_ms"] == 1234


def test_emit_result_error_subtype():
    buf = io.StringIO()
    with redirect_stdout(buf):
        har.emit_result("boom", usage=None, session_id="run-1", is_error=True)
    obj = json.loads(buf.getvalue().strip())
    assert obj["subtype"] == "error" and obj["is_error"] is True


def test_emit_result_reads_usage_attributes():
    class U:
        input_tokens = 22
        output_tokens = 292

    buf = io.StringIO()
    with redirect_stdout(buf):
        har.emit_result("x", usage=U(), session_id="s")
    usage = json.loads(buf.getvalue().strip())["usage"]
    assert usage["input_tokens"] == 22 and usage["output_tokens"] == 292


def test_main_emits_structured_error_on_empty_prompt(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["pydantic-ai-runner", "--print"])
    monkeypatch.delenv("GH_AW_PROMPT", raising=False)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = har.main()
    assert rc == 1
    obj = json.loads(buf.getvalue().strip())
    assert obj["type"] == "result" and obj["is_error"] is True


def test_main_emits_structured_error_on_startup_failure(monkeypatch):
    # A failure *before* the agent loop (e.g. model build) must still produce a
    # parseable stream-json result, never an opaque "no entries" run.
    def boom(_args):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(har, "build_model", boom)
    monkeypatch.setattr(sys, "argv", ["pydantic-ai-runner", "--print", "hello"])
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = har.main()
    assert rc == 1
    obj = json.loads(buf.getvalue().strip())
    assert obj["is_error"] is True
    assert "harness startup failed" in obj["result"]
    assert "kaboom" in obj["result"]


@pytest.mark.skipif(
    not os.environ.get("GH_AW_HARNESS_LIVE_API_KEY"),
    reason="set GH_AW_HARNESS_LIVE_API_KEY/_BASE_URL/_MODEL to run the live test",
)
def test_live_openai_compatible_endpoint(monkeypatch):
    """End-to-end against a real OpenAI-compatible endpoint, using the exact
    argv gh-aw passes. Credentials come from env only — never committed."""
    monkeypatch.setenv("OPENAI_API_KEY", os.environ["GH_AW_HARNESS_LIVE_API_KEY"])
    monkeypatch.setenv(
        "OPENAI_BASE_URL",
        os.environ.get("GH_AW_HARNESS_LIVE_BASE_URL", "https://example.test/v1"),
    )
    model = os.environ.get("GH_AW_HARNESS_LIVE_MODEL", "gpt-4o-mini")
    argv = list(GHAW_ARGV)
    i = argv.index("--mcp-config")
    del argv[i : i + 2]  # no MCP gateway outside a gh-aw run
    argv += ["--model", f"openai:{model}", "Reply with exactly: HARNESS_OK"]
    monkeypatch.setattr(sys, "argv", ["pydantic-ai-runner", *argv])
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = har.main()
    assert rc == 0
    lines = [json.loads(x) for x in buf.getvalue().splitlines() if x.strip()]
    result = next(x for x in lines if x["type"] == "result")
    assert result["is_error"] is False
    assert "HARNESS_OK" in result["result"]
    assert result["usage"]["output_tokens"] > 0
