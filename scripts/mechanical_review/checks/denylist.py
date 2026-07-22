"""Forbidden / removed API surface — high-confidence denylist hits."""

from __future__ import annotations

import re
from pathlib import Path

from ..models import Finding, ScanContext, Severity

CHECK = 'denylist'

# (rule_id, pattern, message, default_for_code_use)
_RULES: list[tuple[str, re.Pattern[str], str]] = [
    (
        'denylist.load_mcp_servers',
        re.compile(r'\bload_mcp_servers\b'),
        'removed API `load_mcp_servers` — use `load_mcp_toolsets`',
    ),
    (
        'denylist.BuiltinToolCallPart',
        re.compile(r'\bBuiltinToolCallPart\b'),
        'removed API `BuiltinToolCallPart` — use `NativeToolCallPart`',
    ),
    (
        'denylist.BuiltinToolReturnPart',
        re.compile(r'\bBuiltinToolReturnPart\b'),
        'removed API `BuiltinToolReturnPart` — use `NativeToolReturnPart`',
    ),
    (
        'denylist.AgentBuiltinTool',
        re.compile(r'\bAgentBuiltinTool\b'),
        'removed API `AgentBuiltinTool` — use `AgentNativeTool` / native tools',
    ),
    (
        'denylist.to_ag_ui',
        re.compile(r'\bto_ag_ui\b'),
        'removed API `Agent.to_ag_ui` — use `pydantic_ai.ui.ag_ui.AGUIAdapter`',
    ),
    (
        'denylist.ag_ui_import',
        re.compile(r'\bfrom\s+pydantic_ai\.ag_ui\b|\bimport\s+pydantic_ai\.ag_ui\b'),
        'removed module `pydantic_ai.ag_ui` — use `pydantic_ai.ui.ag_ui`',
    ),
    (
        'denylist.value_to_type',
        re.compile(r'\bvalue_to_type\b'),
        '`value_to_type` in durable_exec paths — verify intended helper',
    ),
    (
        'denylist.restate_path',
        re.compile(r'durable_exec/restate(?:/|\b)|durable_exec/.*/_mcp_server\b'),
        'stale durable_exec path reference (restate / _mcp_server)',
    ),
]

_SKIP_DIR_NAMES = {
    '.git',
    '.venv',
    'venv',
    'node_modules',
    '__pycache__',
    '.mypy_cache',
    '.ruff_cache',
    '.pytest_cache',
    'dist',
    'build',
    'site',
    'cassettes',
    '.tox',
    'htmlcov',
    # This package itself defines the denylist tokens as patterns/testdata.
    'mechanical_review',
}

_COMMENT_ONLY = re.compile(r'^\s*#')


def _iter_scan_files(repo: Path) -> list[Path]:
    import os

    out: list[Path] = []
    for root_s, dirs, files in os.walk(repo):
        root = Path(root_s)
        dirs[:] = [d for d in dirs if d not in _SKIP_DIR_NAMES and not d.startswith('.')]
        for name in files:
            path = root / name
            if name.endswith('.py'):
                out.append(path)
            elif name.endswith('.md') and any(p in path.parts for p in ('docs', 'examples')):
                out.append(path)
    return out


def _code_portion(line: str) -> str:
    """Text before an inline `#` comment (full line for pure comments → empty)."""
    if _COMMENT_ONLY.match(line):
        return ''
    return line.split('#', 1)[0]


def _severity_for(rel: str, line: str, rule_id: str, matched_in_code: bool) -> Severity | None:
    """Classify hit. Return None to drop (noise)."""
    is_md = rel.endswith('.md')
    is_py = rel.endswith('.py')
    is_slim = rel.startswith('pydantic_ai_slim/')
    is_test = rel.startswith('tests/') or '/tests/' in rel or rel.startswith('clai/')
    is_changelog = 'changelog' in rel.lower() or rel.endswith('HISTORY.md')
    is_durable = 'durable_exec' in rel

    if rule_id == 'denylist.value_to_type' and not is_durable:
        return None
    if rule_id == 'denylist.restate_path':
        return Severity.INFO

    if is_md:
        if is_changelog:
            return Severity.INFO
        # Feature maps / migration notes intentionally document removals.
        if any(tok in line.lower() for tok in ('removed', 'renamed', 'use `', 'use ', '→', '->', 'instead')):
            return Severity.WARNING
        return Severity.INFO

    if is_py:
        # Token only in a trailing comment → not a code use (INFO at most).
        if not matched_in_code:
            return Severity.INFO
        if is_slim or is_test:
            return Severity.ERROR
        # examples, scripts, etc.
        return Severity.WARNING

    return Severity.INFO


def run(ctx: ScanContext) -> list[Finding]:
    findings: list[Finding] = []
    repo = ctx.repo

    files = _iter_scan_files(repo)
    for path in files:
        rel = ctx.rel(path)
        if not ctx.in_scope(rel):
            # In --diff mode still allow docs that changed; skip others.
            if ctx.file_filter is not None:
                continue
        try:
            text = path.read_text(encoding='utf-8', errors='replace')
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if not ctx.is_added_line(rel, lineno):
                continue
            code = _code_portion(line)
            for rule_id, pat, message in _RULES:
                # Prefer match in code portion; fall back to full line for md / comment-only.
                if pat.search(code):
                    matched_in_code = True
                elif pat.search(line):
                    matched_in_code = False
                else:
                    continue
                sev = _severity_for(rel, line, rule_id, matched_in_code)
                if sev is None:
                    continue
                findings.append(
                    Finding(
                        check=CHECK,
                        severity=sev,
                        path=rel,
                        line=lineno,
                        message=message,
                        rule_id=rule_id,
                    )
                )
    return findings
