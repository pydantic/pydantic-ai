"""Forbidden / removed API surface — high-confidence denylist hits."""

from __future__ import annotations

import re
from pathlib import Path

from ..models import Finding, ScanContext, Severity, package_root

CHECK = 'denylist'

_Rule = tuple[str, re.Pattern[str], str]


def _removed_in_v2(name: str, replacement: str) -> _Rule:
    """A name pydantic-ai lists in a `_REMOVED_IN_V2` map, matched outside quotes.

    The maps and `tests/test_removed_v1_names.py` spell each name as a string literal.
    """
    pattern = re.compile(rf'(?<![\'"])\b{name}\b(?![\'"])')
    return f'denylist.{name}', pattern, f'removed API `{name}` — {replacement}'


_USE_MCP_TOOLSET = 'use `MCPToolset`'

# (rule_id, pattern, message)
_RULES: list[_Rule] = [
    _removed_in_v2('load_mcp_servers', 'use `load_mcp_toolsets`'),
    _removed_in_v2('MCPServerStdio', _USE_MCP_TOOLSET),
    _removed_in_v2('MCPServerSSE', _USE_MCP_TOOLSET),
    _removed_in_v2('MCPServerStreamableHTTP', _USE_MCP_TOOLSET),
    _removed_in_v2('MCPServerHTTP', _USE_MCP_TOOLSET),
    _removed_in_v2('cached_async_http_client', 'use `create_async_http_client()`'),
    _removed_in_v2('OpenAIModel', 'use `OpenAIChatModel` or `OpenAIResponsesModel`'),
    _removed_in_v2('OpenAIModelSettings', 'use `OpenAIChatModelSettings`'),
    _removed_in_v2('DeferredToolCalls', 'use `DeferredToolRequests`'),
    (
        # A bare `Usage` also names provider SDK classes, so only the pydantic-ai import counts.
        'denylist.Usage',
        re.compile(r'\bfrom\s+pydantic_ai\.usage\s+import\b[^#]*\bUsage\b|\bpydantic_ai\.usage\.Usage\b'),
        'removed API `pydantic_ai.usage.Usage` — use `RunUsage` or `RequestUsage`',
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


def _is_code_use_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if _COMMENT_ONLY.match(line):
        return False
    # Inline comment after code still counts as code use if symbol appears before #.
    code = line.split('#', 1)[0]
    return bool(code.strip())


def _severity_for(rel: str, line: str, rule_id: str) -> Severity | None:
    """Classify hit. Return None to drop (noise)."""
    is_md = rel.endswith('.md')
    is_py = rel.endswith('.py')
    is_package = package_root(rel) is not None
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
        if not _is_code_use_line(line):
            # Comment-only mention
            if is_package:
                return Severity.INFO
            return Severity.INFO
        # Actual code use
        if is_package:
            return Severity.ERROR
        if is_test:
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
            if ctx.added_lines and not ctx.is_added_line(rel, lineno):
                # Diff mode: only added lines for denylist on code; still whole file for docs? Spec says
                # "added-line aware if --diff". Apply to all.
                continue
            for rule_id, pat, message in _RULES:
                if not pat.search(line):
                    continue
                sev = _severity_for(rel, line, rule_id)
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
