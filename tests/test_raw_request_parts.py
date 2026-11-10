"""Capability code never hand-builds a system or user part for the model: it defines a `CapabilityPart` kind.

See "Trust tiers map to channels" in `agent_docs/message-history-and-caching.md`. A hand-built `SystemPromptPart`
speaks with the operator's authority whatever wrote its text, and hand-built user text (a `UserPromptPart`, a string
passed to `ctx.enqueue`, or a `TextContent` carrying a `metadata` marker) reads as the user's own words. A
`DataCapabilityPart` or `InstructionCapabilityPart` kind with a real `source` lets the framework own the voice, the
frame and the note, and lets code find the record again by type.

The scan covers the capability sources: Pydantic AI's `capabilities` package, Pydantic AI Harness and CLAI 2. A
construction that is genuinely not capability speech (the user's own words, a translation adapter for another
library, a request for a separate model call that never enters the agent's history, or a throwaway part built to
reuse a helper) is allowlisted below, keyed by file and enclosing function, with the reason it's fine.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import pytest

_REPO = Path(__file__).parents[1]

_ROOTS = {
    'capabilities': _REPO / 'pydantic_ai_slim' / 'pydantic_ai' / 'capabilities',
    'harness': _REPO / 'src' / 'pydantic_ai_harness' / 'pydantic_ai_harness',
    'clai2': _REPO / 'src' / 'pydantic_clai2' / 'pydantic_clai2',
}

_RAW_PART_CONSTRUCTORS = frozenset({'SystemPromptPart', 'UserPromptPart', 'user_text_prompt'})
"""Calls that hand-build a system or user part."""

Site = tuple[str, str, str, str]
"""`(root, path relative to the root, enclosing function's qualified name, what it builds)`."""

_ALLOWLIST: dict[Site, str] = {
    # The user's own words.
    ('harness', 'subagents/_tasks.py', 'DelegationReports.wrap_run', 'UserPromptPart'): (
        "A delegating agent's message to its running task, which is that task's user: it sends the task prompt too."
    ),
    ('harness', 'subagents/_worker.py', '_steer', 'UserPromptPart'): (
        "A delegating agent's message to a task served by a worker, sent as the task's user, like the task prompt."
    ),
    ('clai2', 'runtime/session.py', 'Session.steer', 'UserPromptPart'): 'What the user typed while a run is going.',
    ('clai2', 'runtime/session.py', 'Session.prompt', 'UserPromptPart'): 'What the user typed.',
    ('clai2', 'runtime/imported_history.py', 'HistoryBuilder.prompt', 'UserPromptPart'): (
        "A user message from an imported Claude Code or Codex transcript: the user's words, as they typed them."
    ),
    ('harness', 'compaction/_summarizing_compaction.py', '_mark_kept', 'TextContent'): (
        "Marks a copy of the user's own retained turn, whose text is unchanged, so compaction recognizes the copy."
    ),
    # Translation adapters and throwaway parts that never enter the agent's history.
    ('harness', 'browser_use/_model.py', '_map_messages', 'SystemPromptPart'): (
        "Translates browser-use's own conversation for one model call; it never enters a Pydantic AI agent's history."
    ),
    ('harness', 'browser_use/_model.py', '_map_messages', 'UserPromptPart'): (
        "Translates browser-use's own conversation for one model call; it never enters a Pydantic AI agent's history."
    ),
    ('harness', 'tool_output_limits/_payload.py', 'measure', 'SystemPromptPart'): (
        'Wraps text only to count its tokens with the shared estimator; never sent.'
    ),
    ('harness', 'ask_user/_types.py', '_check_note', 'UserPromptPart'): (
        "Validates a note's items with core's content-type check; the part is discarded."
    ),
    ('harness', 'compaction/_shared.py', 'capability_part_text', 'UserPromptPart'): (
        "Reads a capability part's text with the user-prompt text helper; the part is discarded."
    ),
    ('harness', '_message_text.py', 'with_text', 'TextContent'): (
        "Rewrites an existing item's text (redaction, truncation) and keeps that item's own metadata."
    ),
    ('clai2', 'builtin_plugins/web/channel.py', 'SessionChannel.run_started', 'UserPromptPart'): (
        "Shows the user's prompt in the web UI; never sent to a model."
    ),
    # Pending a design decision: each is capability or application speech on the user channel, but needs a choice of
    # source, or changes a public API. Remove the entry once it's a `CapabilityPart` kind.
    ('harness', 'compaction/_pinning.py', 'pin', 'UserPromptPart'): (
        'TODO: `pin(text)` is public and returns a `UserPromptPart`; the caller, not `pin`, knows whose text it is.'
    ),
    ('harness', 'compaction/_pinning.py', 'pin', 'TextContent'): 'TODO: see `pin` above.',
    ('harness', 'ask_user/_toolset.py', 'ask_user_result', 'TextContent'): (
        "TODO: the user's notes ride `ToolReturn.content` in a hand-built `<answer_note>` frame with an origin marker."
    ),
    ('clai2', 'runtime/imported_history.py', 'imported_context', 'UserPromptPart'): (
        "TODO: content another harness added (reminders, attachments) and CLAI's own import notes, as marked user text."
    ),
    ('clai2', 'runtime/imported_history.py', 'imported_context', 'TextContent'): 'TODO: see `imported_context` above.',
    ('clai2', 'runtime/session_runtime.py', 'SessionRuntime._dispatch_input', 'UserPromptPart'): (
        "TODO: a `!command`'s output, after CLAI's own preamble, as user text; the output is the command's, not the user's."
    ),
}


def _raw_parts_in(root: str, relative: str, source: str) -> Iterator[tuple[Site, int]]:
    """Each hand-built system or user part in *source*, with its line."""

    def visit(node: ast.AST, scope: str) -> Iterator[tuple[Site, int]]:
        for child in ast.iter_child_nodes(node):
            inner = scope
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                inner = f'{scope}.{child.name}' if scope else child.name
            if isinstance(child, ast.Call) and (built := _built(child)) is not None:
                yield (root, relative, scope or '<module>', built), child.lineno
            yield from visit(child, inner)

    yield from visit(ast.parse(source), '')


def _built(call: ast.Call) -> str | None:
    """What *call* hand-builds, if it's a raw system or user part."""
    func = call.func
    name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
    if name in _RAW_PART_CONSTRUCTORS:
        return name
    if name == 'TextContent' and any(keyword.arg == 'metadata' for keyword in call.keywords):
        # A marker on user text is how capabilities tagged their speech before `CapabilityPart`.
        return 'TextContent'
    if name == 'enqueue' and any(isinstance(arg, ast.JoinedStr) or _is_str(arg) for arg in call.args):
        # A string becomes a user part: framework voice read as the user's words.
        return 'enqueue(str)'
    return None


def _is_str(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def _all_raw_parts() -> dict[Site, list[int]]:
    found: dict[Site, list[int]] = {}
    for root, directory in _ROOTS.items():
        for path in sorted(directory.rglob('*.py')):
            relative = path.relative_to(directory).as_posix()
            for site, line in _raw_parts_in(root, relative, path.read_text()):
                found.setdefault(site, []).append(line)
    return found


@pytest.mark.parametrize(
    'statement, built',
    [
        pytest.param("SystemPromptPart('Be brief.')", 'SystemPromptPart', id='system-part'),
        pytest.param('messages.UserPromptPart(text)', 'UserPromptPart', id='qualified-user-part'),
        pytest.param('ModelRequest.user_text_prompt(text)', 'user_text_prompt', id='user-text-prompt'),
        pytest.param("TextContent(text, metadata={'origin': 'x'})", 'TextContent', id='metadata-marker'),
        pytest.param("ctx.enqueue('Stop now.')", 'enqueue(str)', id='enqueue-str'),
        pytest.param("ctx.enqueue(f'{n} left.')", 'enqueue(str)', id='enqueue-fstring'),
        pytest.param('TextContent(text)', None, id='plain-text-content'),
        pytest.param('ctx.enqueue(part)', None, id='enqueue-part'),
        pytest.param('isinstance(part, UserPromptPart)', None, id='isinstance'),
    ],
)
def test_the_raw_part_scan(statement: str, built: str | None) -> None:
    source = f'class Capability:\n    def hook(self, ctx, text, part, n):\n        return {statement}\n'
    expected = [] if built is None else [(('harness', 'example.py', 'Capability.hook', built), 3)]
    assert list(_raw_parts_in('harness', 'example.py', source)) == expected


def test_capability_code_builds_no_raw_system_or_user_parts() -> None:
    """Define a `DataCapabilityPart` or `InstructionCapabilityPart` kind with a real `source` instead.

    If the part genuinely isn't capability speech, allowlist its file and function in `_ALLOWLIST` with the reason.
    """
    unexpected = {
        f'{root}/{path}:{lines[0]} {scope} builds {built}': lines
        for (root, path, scope, built), lines in _all_raw_parts().items()
        if (root, path, scope, built) not in _ALLOWLIST
    }
    assert unexpected == {}


def test_every_allowlist_entry_is_still_needed() -> None:
    assert sorted(set(_ALLOWLIST) - set(_all_raw_parts())) == []
