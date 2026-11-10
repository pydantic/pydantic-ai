#!/usr/bin/env python3
"""Compute /review-branch dispatch preconditions from the prepared snapshot (stdlib only).

Run after `prepare.py`. Reads its artifacts, writes the reviewer input files beside
them plus `preconditions.json`, and prints the table the orchestrator reports. Every
gate only opens a reviewer: each reviewer re-confirms its trigger against the diff and
self-skips, so a gate may over-fire but must not under-fire.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from mechanical.models import is_public_module, package_root

SLIM = 'pydantic_ai_slim/pydantic_ai/'
HUNK = re.compile(rb'@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@')

CORE_SYMBOLS = re.compile(
    r'\b(Agent|Model|AbstractModel|RealtimeModel|ModelSettings|ModelResponse|ModelRequest|ModelResponsePart'
    r'|ModelRequestPart|RunContext|ToolDefinition|ToolCallPart|ToolReturnPart|AbstractCapability|AbstractToolset'
    r'|Workspace)\b'
)
SPEC_LINE = re.compile(r'^(?:Feature spec|Spec|Docs|Reference):\s*(https?://\S+)', re.IGNORECASE | re.MULTILINE)
TOP_LEVEL_DEF = re.compile(r'^\+(?:async )?def |^\+class ')
PUBLIC_DEF = re.compile(r'^\+\s*(?:async\s+)?def\s+[A-Za-z]\w*\s*\(|^\+\s*class\s+[A-Za-z]\w*')
TEST_FN = re.compile(r'^\+(?:    )?(?:async )?def test_')
COMMENT_ONLY = re.compile(r'^[+-]\s*#')
# An added test line that reaches past the public API: a private method call, or a `_name` imported from a package.
PRIVATE_TEST_REACH = re.compile(
    r'\w\._[a-z]\w*\(|^\+\s*from (?:pydantic_ai|pydantic_ai_harness|pydantic_clai2|pydantic_graph|pydantic_evals|clai)'
    r'[\w.]* import .*\b_[a-z]\w*'
)

TRUST_BOUNDARY_PATHS = re.compile(
    r'^(pydantic_ai_slim/pydantic_ai/(ui/|_ssrf\.py|_framing\.py|messages\.py|common_tools/web_fetch\.py'
    r'|realtime/_openai_webrtc\.py|workspaces/|durable_exec/_workspace\.py)'
    r'|src/pydantic_ai_harness/pydantic_ai_harness/([a-z0-9_]+_sandbox/|ssh_workspace/|shell/|_workspace_provider\.py)'
    r'|src/pydantic_clai2/pydantic_clai2/builtin_plugins/web/'
    r'|\.github/(workflows|actions|scripts)/'
    r'|docs/(ui/|input\.md)|tests/(test_vercel_ai|test_ag_ui|test_ui|test_ui_web)\.py)'
)
# Prompt framing: a changed package line that builds tag-framed prompt text (#10544, #10547, #10558, #10559).
FRAMING_TOKENS = re.compile(r'\b_framing\b|\bneutralize_tags\b|</?[a-z][a-z0-9_-]*>')
SUSPEND_RESUME_PATHS = re.compile(
    r'^(pydantic_ai_slim/pydantic_ai/(ui/|messages\.py|conversation\.py|durable_exec/|_agent_graph\.py|result\.py'
    r'|_run_context\.py|agent/)'
    r'|src/pydantic_ai_harness/pydantic_ai_harness/(step_persistence/|absurd/|aws_lambda/|azure_durable_task/|pgtask/'
    r'|render/)'
    r'|src/pydantic_clai2/pydantic_clai2/runtime/(sessions|imported_sessions|claude_code_sessions|codex_sessions)\.py)'
)
SUSPEND_RESUME_TOKENS = re.compile(
    r'dump_messages|load_messages|ModelMessagesTypeAdapter|ConversationTypeAdapter|DeferredTool|CallDeferred'
    r'|ToolApproved|ToolDenied|ApprovalRequired|requires_approval|\binterrupt\b|state=.suspended.|\bsuspended\b'
    r'|cancel_suspended_response|cancel_suspended_job|continuation_delay|CompletedStreamedResponse'
    r'|SuspendedResponseExpired|merge_responses'
)
CONCURRENCY_LIFECYCLE_PATHS = re.compile(r'^pydantic_ai_slim/pydantic_ai/realtime/')
# Async lifecycle primitives on a changed package line: task groups, cancel scopes and deadlines, task
# creation, shields, cancellation, teardown, and the drain helpers `agent_docs/concurrency.md` names.
CONCURRENCY_LIFECYCLE_TOKENS = re.compile(
    r'create_task_group|TaskGroup|CancelScope|move_on_after|fail_after|create_task\(|asyncio\.Task\b|\bshield\b'
    r'|CancelledError|get_cancelled_exc_class|\.cancel\(|\.cancelled\(\)|uncancel\(|__aexit__|aclose\('
    r'|asynccontextmanager|AsyncExitStack'
    r'|cancel_and_drain|asyncio\.gather|_utils\.gather|asyncio\.wait\('
)
# A streaming chunk boundary is a pause boundary. Bare `delta` is excluded: it matches
# nearly all provider streaming code; these tokens name the accumulation surface itself.
STREAMING_STATE = re.compile(
    r'StreamedResponse|allow_partial|partial_json|_parts_manager|handle_text_delta|handle_thinking_delta'
    r'|handle_tool_call_delta|from_json\('
)
SLIM_MODEL_FILE = re.compile(r'^pydantic_ai_slim/pydantic_ai/models/[^/]+\.py$')
# Realtime codecs map the provider wire protocol the way `models/<provider>.py` maps HTTP.
REALTIME_CODEC_FILE = re.compile(
    r'^pydantic_ai_slim/pydantic_ai/realtime/(openai|azure|google|xai|openai_live|elevenlabs|elevenlabs_agents'
    r'|_openai_protocol|_openai_webrtc)\.py$'
)

# Ambient-dimension gate: three narrow arms OR'd together, over changed production lines,
# comment-only lines dropped. Multi-line conditions are missed by design; the reviewer
# re-derives the real branch set. Evidence: fires on #6614, #6521, #6539, #6563, #6052,
# #5326, #5433; stays silent on the ambient-data PRs #6254 and #6547.
# Arm A: a strong ambient read is signal on its own (the read and its branch are often on
# different lines, #6521). `profiles/` and `providers/` define ambient data, so they are excluded.
# `_?` matches realtime's `self._profile.get(...)` (`realtime/_session.py`).
AMBIENT_STRONG = re.compile(r'\b_?profile\.[a-z_]|_model_profile\(|\bModelProfile\b')
AMBIENT_DATA_DIRS = (SLIM + 'profiles/', SLIM + 'providers/')
# Arm B: a weak token fires only when the same line also reads as a condition. Slim only:
# harness `store` / `background` name its own memory and job concepts.
AMBIENT_WEAK = re.compile(
    r'\bmodel_name\b|\bis_retrieve\b|\bbackground\b|\bstore\b|native_tools\b|supported_[a-z_]+|\bmodel_family\b'
)
AMBIENT_COND = re.compile(r'(^|\s)(if|elif|while|case)\s|==|!=|\s(and|or|not|in)\s')
# Arm C: the streaming twin, matched by surface name in `models/` (#6563 had no ambient token).
# Not hunk-header based: local `git diff` labels hunks by class, GitHub by function.
AMBIENT_STREAM = re.compile(
    r'request_stream|_process_streamed_response|StreamedResponse|stream_async|\bstream=True\b|_stream_[a-z_]*\('
)

WRAPPED_CLASS_FILES = re.compile(
    r'^pydantic_ai_slim/pydantic_ai/(mcp\.py|agent/[^/]+\.py|toolsets/[^_/][^/]*\.py'
    r'|models/[^_/][^/]*\.py|capabilities/[^_/][^/]*\.py)$'
)
SIBLING_FAMILY = re.compile(
    r'^(pydantic_ai_slim/pydantic_ai/(models/[^_/][^/]*\.py|providers/[^_/][^/]*\.py|profiles/[^_/][^/]*\.py'
    r'|ui/[^/]+/|durable_exec/[^/]+/|toolsets/[^_/][^/]*\.py|native_tools/[^/]+\.py|capabilities/[^/]+\.py'
    r'|embeddings/[^_/][^/]*\.py|images/[^_/][^/]*\.py'
    r'|realtime/(openai|azure|google|xai|openai_live|elevenlabs|elevenlabs_agents|_openai_protocol)\.py)'
    r'|src/pydantic_ai_harness/pydantic_ai_harness/([a-z0-9_]+_sandbox/|ssh_workspace/|absurd/|aws_lambda/'
    r'|azure_durable_task/|pgtask/|render/)'
    r'|src/pydantic_clai2/pydantic_clai2/builtin_plugins/)'
)
PEER_TYPE_BRANCH = re.compile(r'isinstance\(|^\+?\s*case .+\(\)')

# The classifier and manifest CI's `needs-durable-exec-review` label runs, read at the base-branch tip.
DURABLE_EXEC_CONTRACT = ('scripts/durable_exec_contract.py', 'scripts/durable_exec_contract.json')
# `git diff --name-status` letter -> the status GitHub's `pulls/{number}/files` endpoint reports.
GITHUB_FILE_STATUS: dict[str, str] = {'A': 'added', 'M': 'modified', 'D': 'removed', 'R': 'renamed', 'C': 'copied'}

GATING_LANES = ('review-runtime-behavior', 'review-public-api', 'review-integration-impact')
WAVES = {
    1: (
        'pattern',
        'auto-review',
        'review-public-api',
        'review-integration-impact',
        'review-spec-coverage',
        'review-spec-conformance',
        'review-runtime-behavior',
        'review-code-reuse',
        'review-trust-boundary',
        'review-suspend-resume',
        'review-concurrency-lifecycle',
        'review-ambient-matrix',
        'review-precedent',
    ),
    2: ('review-test-shape', 'review-cassettes', 'review-tests', 'review-docs'),
    3: ('review-cross-cutting', 'review-sibling-consistency'),
}
# Subagents the orchestrator dispatches by id. `pattern` and `auto-review` are skills.
REVIEWER_AGENTS = (
    *(lane for wave in WAVES.values() for lane in wave if lane.startswith('review-')),
    'review-claim-check',
)


@dataclass(frozen=True)
class Change:
    status: str
    path: str
    previous_path: str | None = None


def git(repo: Path, *args: str) -> bytes:
    return subprocess.run(
        ['git', '-C', str(repo), '-c', 'core.quotePath=true', *args], check=True, capture_output=True
    ).stdout


def name_status(repo: Path, merge_base: str, head: str, *pathspec: str) -> list[Change]:
    fields = git(
        repo, 'diff', '--no-ext-diff', '--find-renames', '--name-status', '-z', merge_base, head, '--', *pathspec
    ).split(b'\0')
    changes: list[Change] = []
    index = 0
    while index < len(fields) and fields[index]:
        status = fields[index].decode('ascii')
        paths = 2 if status[0] in 'RC' else 1
        path = fields[index + paths].decode('utf-8', errors='surrogateescape')
        previous = fields[index + 1].decode('utf-8', errors='surrogateescape') if paths == 2 else None
        changes.append(Change(status[0], path, previous))
        index += paths + 1
    return changes


def _header_path(raw: bytes, prefix: bytes) -> str | None:
    raw = raw.split(b'\t', 1)[0]
    if raw.startswith(b'"'):
        decoded = ast.literal_eval('b' + raw.decode('ascii'))
        if not isinstance(decoded, bytes):
            raise ValueError('Malformed quoted path')
        raw = decoded
    if not raw.startswith(prefix):
        return None
    return raw[len(prefix) :].decode('utf-8', errors='surrogateescape')


def changed_lines(diff: bytes) -> dict[str, list[str]]:
    """Map each file to its `+`/`-` hunk lines, prefix kept. Hunk counts keep body text from reading as headers."""
    files: dict[str, list[str]] = {}
    old_path: str | None = None
    new_path: str | None = None
    old_left = new_left = 0
    for line in diff.split(b'\n'):
        if old_left or new_left:
            prefix = line[:1]
            if prefix == b'\\':
                continue
            if prefix != b'+':
                old_left -= 1
            if prefix != b'-':
                new_left -= 1
            path = new_path or old_path
            if prefix in (b'+', b'-') and path is not None:
                files.setdefault(path, []).append(line.decode('utf-8', errors='replace'))
        elif line.startswith(b'diff --git '):
            old_path = new_path = None
        elif line.startswith(b'--- '):
            old_path = _header_path(line[4:], b'a/')
        elif line.startswith(b'+++ '):
            new_path = _header_path(line[4:], b'b/')
        elif match := HUNK.match(line):
            old_left = int(match[1]) if match[1] is not None else 1
            new_left = int(match[2]) if match[2] is not None else 1
    return files


def file_hunks(diff: bytes) -> dict[str, str]:
    """Each file's hunk text from its first `@@` line on, as GitHub's `patch` field holds it.

    Keyed by the head path, or the merge-base path of a removed file. A file without hunks has no key.
    """
    patches: dict[str, str] = {}
    for section in re.split(rb'^diff --git ', diff, flags=re.MULTILINE):
        header, found, hunks = section.partition(b'\n@@')
        if not found:
            continue
        paths = {line[:4]: line[4:] for line in header.split(b'\n') if line.startswith((b'--- ', b'+++ '))}
        path = _header_path(paths[b'+++ '], b'b/') or _header_path(paths[b'--- '], b'a/')
        if path is not None:
            patches[path] = (b'@@' + hunks).removesuffix(b'\n').decode('utf-8', errors='replace')
    return patches


def durable_exec_entries(repo: Path, output: Path, base_sha: str, merge_base: str, head: str) -> list[str] | None:
    """The gating durable-exec contract entries the branch touches, as `core_file::qualname (engines)` lines.

    Runs `touched_entries` from the classifier and manifest at the base-branch tip, as the classifier's own
    `--touched` mode does, with base-side sources read at the merge base. `None` when the base tip lacks either
    file, or has a classifier without `touched_entries`.
    """
    script, manifest_path = DURABLE_EXEC_CONTRACT
    if len(git(repo, 'ls-tree', '--name-only', base_sha, '--', *DURABLE_EXEC_CONTRACT).splitlines()) < 2:
        return None
    classifier = output / 'durable_exec_contract.py'
    classifier.write_bytes(git(repo, 'show', f'{base_sha}:{script}'))
    spec = importlib.util.spec_from_file_location('durable_exec_contract', classifier)
    if spec is None or spec.loader is None:
        raise ImportError(f'cannot load {classifier}')
    contract = importlib.util.module_from_spec(spec)
    # Before Python 3.14, `dataclasses` reads the module's string annotations through `sys.modules`.
    sys.modules[spec.name] = contract
    spec.loader.exec_module(contract)
    # pydantic-ai #11128 added the manifest and classifier; #11499 added `touched_entries` a day later.
    if not hasattr(contract, 'touched_entries'):
        return None
    manifest = json.loads(git(repo, 'show', f'{base_sha}:{manifest_path}'))
    if not manifest:
        return []
    patches = file_hunks(
        git(
            repo,
            'diff',
            '--no-ext-diff',
            '--no-textconv',
            '--no-color',
            '--src-prefix=a/',
            '--dst-prefix=b/',
            '--find-renames',
            '--patch',
            '--unified=0',
            merge_base,
            head,
            '--',
            *manifest,
        )
    )
    files = [
        contract.ChangedFile(
            change.path,
            GITHUB_FILE_STATUS.get(change.status, 'modified'),
            change.previous_path,
            patches.get(change.path),
        )
        for change in name_status(repo, merge_base, head, *manifest)
    ]

    def source_at(commit: str) -> Callable[[str], str]:
        return lambda path: git(repo, 'show', f'{commit}:{path}').decode('utf-8', errors='replace')

    touched = contract.touched_entries(manifest, files, source_at(merge_base), source_at(head))
    return sorted(
        f'{core_file}::{qualname} ({", ".join(entry["engines"])})'
        for core_file, definitions in touched.items()
        for qualname, entry in definitions.items()
    )


def is_python_in_package(path: str) -> bool:
    return path.endswith('.py') and package_root(path) is not None


def is_public_surface(path: str) -> bool:
    """Slim exposes public modules; harness and clai2 re-export from `_` modules through `__all__`."""
    if not is_python_in_package(path):
        return False
    return is_public_module(path) if path.startswith(SLIM) else True


def pr_ready(pr: dict[str, object] | None) -> str:
    if pr is None:
        return 'unknown'
    return 'yes' if pr.get('isDraft') is False and pr.get('reviewDecision') != 'CHANGES_REQUESTED' else 'no'


def load_pr(repo: Path, pr_json: Path | None, pr_number: str | None) -> dict[str, object] | None:
    try:
        if pr_json is not None:
            raw = pr_json.read_text()
        else:
            command = ['gh', 'pr', 'view', *([pr_number] if pr_number else []), '--json', 'body,isDraft,reviewDecision']
            raw = subprocess.run(command, cwd=repo, check=True, capture_output=True, text=True).stdout
        data: dict[str, object] | list[object] | str | float | bool | None = json.loads(raw)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


@dataclass
class Preconditions:
    packages_touched: list[str]
    code_changes: bool
    public_modules: list[str]
    core_symbols: list[str]
    new_py_files: list[str]
    new_top_level_defs: int
    new_public_defs: int
    spec_urls: list[str]
    cassettes_changed: int
    provider_mapping_changed: list[str]
    new_test_fn_count: int
    private_test_reach_lines: int
    trust_boundary: list[str]
    suspend_resume: list[str]
    concurrency_lifecycle: list[str]
    ambient_branch_lines: list[str]
    modified_files: list[str]
    wrapped_class_files: list[str]
    test_files: list[str]
    package_python_files: list[str]
    sibling_family_files: list[str]
    sibling_family_added: list[str]
    peer_type_branches: list[str]
    durable_exec_entries: list[str] | None
    """`None` when the base-branch tip has no durable-exec contract to classify with."""
    pr_ready_for_docs: str


def compute(
    changes: list[Change],
    diff: bytes,
    cassettes: list[str],
    pr: dict[str, object] | None,
    durable_exec_entries: list[str] | None = None,
) -> Preconditions:
    files = [c.path for c in changes]
    modified = [c.path for c in changes if c.status in 'MR']
    added = [c.path for c in changes if c.status == 'A']
    lines = changed_lines(diff)
    all_lines = [(path, line) for path, body in lines.items() for line in body]
    production_lines = [line for path, line in all_lines if is_python_in_package(path) and not COMMENT_ONLY.match(line)]

    trust_boundary = [p for p in files if TRUST_BOUNDARY_PATHS.match(p)]
    if any(FRAMING_TOKENS.search(line) for line in production_lines):
        trust_boundary.append('framing-token-match')
    concurrency_lifecycle = [p for p in files if CONCURRENCY_LIFECYCLE_PATHS.match(p)]
    if any(CONCURRENCY_LIFECYCLE_TOKENS.search(line) for line in production_lines):
        concurrency_lifecycle.append('lifecycle-token-match')

    suspend_resume = [p for p in files if SUSPEND_RESUME_PATHS.match(p)]
    if SUSPEND_RESUME_TOKENS.search(diff.decode('utf-8', errors='replace')):
        suspend_resume.append('diff-token-match')
    if any(SLIM_MODEL_FILE.match(p) for p in files) and any(STREAMING_STATE.search(line) for _, line in all_lines):
        suspend_resume.append('models-streaming-state-match')

    ambient: list[str] = []
    for path, line in all_lines:
        if not is_python_in_package(path) or COMMENT_ONLY.match(line):
            continue
        strong = not path.startswith(AMBIENT_DATA_DIRS) and AMBIENT_STRONG.search(line)
        weak = path.startswith(SLIM) and AMBIENT_WEAK.search(line) and AMBIENT_COND.search(line)
        stream = path.startswith(SLIM + 'models/') and AMBIENT_STREAM.search(line)
        if strong or weak or stream:
            ambient.append(f'{path}: {line}')

    body = pr.get('body') if pr else None
    touched = {
        'slim': any(p.startswith(SLIM) for p in files),
        'graph': any(p.startswith('pydantic_graph/') for p in files),
        'evals': any(p.startswith('pydantic_evals/') for p in files),
        'clai': any(p.startswith('clai/') for p in files),
        'harness': any(p.startswith(('src/pydantic_ai_harness/', 'tests/harness/')) for p in files),
        'clai2': any(p.startswith(('src/pydantic_clai2/', 'tests/clai2/')) for p in files),
    }
    return Preconditions(
        packages_touched=[name for name, hit in touched.items() if hit],
        code_changes=bool(diff.strip()),
        public_modules=[p for p in files if is_public_surface(p)],
        core_symbols=sorted({m for _, line in all_lines for m in CORE_SYMBOLS.findall(line)}),
        new_py_files=[p for p in added if is_python_in_package(p)],
        new_top_level_defs=sum(1 for path, line in all_lines if path.endswith('.py') and TOP_LEVEL_DEF.match(line)),
        new_public_defs=sum(1 for path, line in all_lines if is_python_in_package(path) and PUBLIC_DEF.match(line)),
        spec_urls=SPEC_LINE.findall(body) if isinstance(body, str) else [],
        cassettes_changed=len(cassettes),
        provider_mapping_changed=[
            p for p in files if (p.startswith(SLIM + 'models/') and p.endswith('.py')) or REALTIME_CODEC_FILE.match(p)
        ],
        new_test_fn_count=sum(1 for _, line in all_lines if TEST_FN.match(line)),
        private_test_reach_lines=sum(
            1
            for path, line in all_lines
            if path.startswith('tests/') and line.startswith('+') and PRIVATE_TEST_REACH.search(line)
        ),
        trust_boundary=trust_boundary,
        suspend_resume=suspend_resume,
        concurrency_lifecycle=concurrency_lifecycle,
        ambient_branch_lines=ambient,
        modified_files=modified,
        wrapped_class_files=[p for p in files if WRAPPED_CLASS_FILES.match(p)],
        test_files=[p for p in files if p.startswith('tests/')],
        package_python_files=[p for p in files if is_python_in_package(p)],
        sibling_family_files=[p for p in modified if SIBLING_FAMILY.match(p)],
        sibling_family_added=[p for p in added if SIBLING_FAMILY.match(p)],
        peer_type_branches=[f'{path}: {line}' for path, line in all_lines if PEER_TYPE_BRANCH.search(line)],
        durable_exec_entries=durable_exec_entries,
        pr_ready_for_docs=pr_ready(pr),
    )


def decide(pre: Preconditions) -> dict[str, tuple[bool, str]]:
    """Reviewer id -> (dispatch, reason). The reason names the precondition values either way."""
    code = pre.code_changes
    durable_exec = (
        'durable-exec contract: absent at base'
        if pre.durable_exec_entries is None
        else f'durable-exec entries: {len(pre.durable_exec_entries)}'
    )
    return {
        'pattern': (code, 'code changes'),
        'auto-review': (code, 'code changes'),
        'review-public-api': (bool(pre.public_modules), f'public modules: {len(pre.public_modules)}'),
        'review-integration-impact': (
            bool(
                pre.core_symbols
                or pre.wrapped_class_files
                or pre.durable_exec_entries
                or any(not path.startswith(SLIM) for path in pre.package_python_files)
            ),
            f'core symbols: {", ".join(pre.core_symbols) or "none"}; wrapped-class files: {len(pre.wrapped_class_files)}; '
            f'other package files: {sum(not path.startswith(SLIM) for path in pre.package_python_files)}; {durable_exec}',
        ),
        'review-spec-coverage': (bool(pre.spec_urls), f'spec url: {pre.spec_urls[0] if pre.spec_urls else "none"}'),
        'review-spec-conformance': (code, 'code changes; the reviewer NL-gates itself'),
        'review-runtime-behavior': (
            bool(pre.package_python_files),
            f'package .py files: {len(pre.package_python_files)}',
        ),
        'review-code-reuse': (
            bool(pre.new_py_files) or pre.new_top_level_defs > 0,
            f'new .py files: {len(pre.new_py_files)}; new top-level defs: {pre.new_top_level_defs}',
        ),
        'review-trust-boundary': (
            bool(pre.trust_boundary),
            f'trust-boundary signals: {", ".join(s for s in pre.trust_boundary if "/" not in s) or "none"}; '
            f'paths: {sum(1 for s in pre.trust_boundary if "/" in s)}',
        ),
        'review-suspend-resume': (
            bool(pre.suspend_resume),
            f'pause-boundary signals: {", ".join(s for s in pre.suspend_resume if "/" not in s) or "none"}; '
            f'paths: {sum(1 for s in pre.suspend_resume if "/" in s)}',
        ),
        'review-concurrency-lifecycle': (
            bool(pre.concurrency_lifecycle),
            f'lifecycle signals: {", ".join(s for s in pre.concurrency_lifecycle if "/" not in s) or "none"}; '
            f'realtime paths: {sum(1 for s in pre.concurrency_lifecycle if "/" in s)}',
        ),
        'review-ambient-matrix': (bool(pre.ambient_branch_lines), f'ambient lines: {len(pre.ambient_branch_lines)}'),
        'review-precedent': (bool(pre.modified_files), f'modified files: {len(pre.modified_files)}'),
        'review-test-shape': (
            pre.new_test_fn_count >= 10 and pre.private_test_reach_lines >= 10,
            f'new test fns: {pre.new_test_fn_count} (needs 10); '
            f'added test lines reaching private symbols: {pre.private_test_reach_lines} (needs 10)',
        ),
        'review-cassettes': (
            pre.cassettes_changed > 0 or bool(pre.provider_mapping_changed),
            f'cassettes: {pre.cassettes_changed}; provider mapping files: {len(pre.provider_mapping_changed)}',
        ),
        'review-tests': (
            bool(pre.test_files) or pre.new_public_defs > 0,
            f'test files: {len(pre.test_files)}; new public defs: {pre.new_public_defs}',
        ),
        'review-docs': (True, 'every branch'),
        'review-cross-cutting': (code, 'code changes; the reviewer self-skips without new public surface'),
        'review-sibling-consistency': (
            bool(pre.sibling_family_files or pre.sibling_family_added or pre.peer_type_branches),
            f'family modified: {len(pre.sibling_family_files)}; added: {len(pre.sibling_family_added)}; '
            f'peer-type lines: {len(pre.peer_type_branches)}',
        ),
    }


def write_inputs(output: Path, pre: Preconditions) -> None:
    """Write the files reviewers receive as `*_path` inputs."""
    lists = {
        'core-symbols-touched.txt': pre.core_symbols,
        'wrapped-class-files.txt': pre.wrapped_class_files,
        'new-py-files.txt': pre.new_py_files,
        'ambient-branch-lines.txt': pre.ambient_branch_lines,
        'sibling-family-files.txt': pre.sibling_family_files,
        'peer-type-branches.txt': pre.peer_type_branches,
        'durable-exec-entries.txt': pre.durable_exec_entries or [],
        'spec-url.txt': pre.spec_urls,
    }
    for name, values in lists.items():
        (output / name).write_text(''.join(f'{value}\n' for value in values))
    (output / 'pr-ready-for-docs.txt').write_text(f'{pre.pr_ready_for_docs}\n')


def render(pre: Preconditions, dispatch: dict[str, tuple[bool, str]], base: str, merge_base: str) -> str:
    rows = [
        f'Preconditions: base {base} @ {merge_base[:9]}',
        f'packages touched: {", ".join(pre.packages_touched) or "none"}',
        f'pr_ready_for_docs_review: {pre.pr_ready_for_docs}',
        '',
        '| wave | reviewer | dispatch | precondition |',
        '|---|---|---|---|',
    ]
    for wave, reviewers in WAVES.items():
        for reviewer in reviewers:
            run, reason = dispatch[reviewer]
            gate = ' (gating)' if reviewer in GATING_LANES else ''
            rows.append(f'| {wave} | {reviewer}{gate} | {"run" if run else "skip"} | {reason} |')
    return '\n'.join(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, help='Defaults to worktree-local review-branch/tmp')
    parser.add_argument('--pr', help='PR number for `gh pr view` (default: the current branch)')
    parser.add_argument('--pr-json', type=Path, help='Read PR body/isDraft/reviewDecision from this file instead of gh')
    args = parser.parse_args()
    try:
        repo = Path(git(args.repo, 'rev-parse', '--show-toplevel').decode().strip())
        agents = repo / '.agents/agents'
        present = {path.stem for path in agents.glob('review-*.md')}
        if missing := [agent for agent in REVIEWER_AGENTS if agent not in present]:
            print(
                f'No agent file under {agents} for: {", ".join(missing)}. '
                'Restore the review agents from the commit that holds this skill, then rerun this step.',
                file=sys.stderr,
            )
            return 1
        output = args.output or repo / '.agents/skills/review-branch/tmp'
        merge_base = (output / 'merge-base.txt').read_text().strip()
        head = (output / 'head-sha.txt').read_text().strip()
        base = (output / 'base-ref.txt').read_text().strip()
        base_sha = (output / 'base-sha.txt').read_text().strip()
        diff = (output / 'code.diff').read_bytes()
        cassettes = [line for line in (output / 'cassettes.list').read_text().splitlines() if line.strip()]
        changes = name_status(repo, merge_base, head)
        entries = durable_exec_entries(repo, output, base_sha, merge_base, head)
    except subprocess.CalledProcessError as error:
        print((error.stderr or b'').decode(errors='replace').strip() or str(error), file=sys.stderr)
        return 1
    except (OSError, ValueError, SyntaxError) as error:
        print(f'{error} — run prepare.py first', file=sys.stderr)
        return 1
    pre = compute(changes, diff, cassettes, load_pr(repo, args.pr_json, args.pr), entries)
    dispatch = decide(pre)
    write_inputs(output, pre)
    report = {
        'base_ref': base,
        'merge_base': merge_base,
        'head_sha': head,
        'preconditions': asdict(pre),
        'reviewers': {
            reviewer: {'wave': wave, 'dispatch': dispatch[reviewer][0], 'reason': dispatch[reviewer][1]}
            for wave, reviewers in WAVES.items()
            for reviewer in reviewers
        },
    }
    (output / 'preconditions.json').write_text(json.dumps(report, indent=2) + '\n')
    print(render(pre, dispatch, base, merge_base))
    return 0


if __name__ == '__main__':
    sys.exit(main())
