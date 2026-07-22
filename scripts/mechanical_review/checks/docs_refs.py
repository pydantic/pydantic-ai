"""Docs still recommending removed public symbols."""

from __future__ import annotations

import re
from pathlib import Path

from ..models import Finding, ScanContext, Severity

CHECK = 'docs_refs'

# Names that should not be recommended as current API in docs (info/warning).
_REMOVED: list[tuple[str, re.Pattern[str], str, Severity]] = [
    (
        'docs_refs.load_mcp_servers',
        re.compile(r'\bload_mcp_servers\b'),
        'docs mention removed `load_mcp_servers` — prefer `load_mcp_toolsets`',
        Severity.WARNING,
    ),
    (
        'docs_refs.BuiltinToolCallPart',
        re.compile(r'\bBuiltinToolCallPart\b'),
        'docs mention removed `BuiltinToolCallPart` — prefer `NativeToolCallPart`',
        Severity.WARNING,
    ),
    (
        'docs_refs.GeminiModel',
        re.compile(r'\bGeminiModel\b'),
        'docs mention `GeminiModel` — verify current Google model class name',
        Severity.INFO,
    ),
    (
        'docs_refs.OpenAIModel_bare',
        re.compile(r'(?<![\w.])OpenAIModel(?![\w])'),
        'docs mention bare `OpenAIModel` — verify still public / preferred',
        Severity.INFO,
    ),
    (
        'docs_refs.to_ag_ui',
        re.compile(r'\bto_ag_ui\b|\bAgent\.to_ag_ui\b'),
        'docs mention removed `Agent.to_ag_ui` — prefer `AGUIAdapter`',
        Severity.WARNING,
    ),
    (
        'docs_refs.ag_ui_import',
        re.compile(r'pydantic_ai\.ag_ui\b|from pydantic_ai import ag_ui'),
        'docs mention removed `pydantic_ai.ag_ui` — prefer `pydantic_ai.ui.ag_ui`',
        Severity.WARNING,
    ),
    (
        'docs_refs.run_mcp_servers',
        re.compile(r'\brun_mcp_servers\b'),
        'docs mention removed `run_mcp_servers` — use `async with agent:`',
        Severity.WARNING,
    ),
]

_SKIP = {'.git', 'site', 'node_modules', '__pycache__', 'cassettes'}


def run(ctx: ScanContext) -> list[Finding]:
    findings: list[Finding] = []
    docs = ctx.repo / 'docs'
    if not docs.is_dir():
        return findings

    for path in docs.rglob('*.md'):
        if any(part in _SKIP for part in path.parts):
            continue
        rel = ctx.rel(path)
        if not ctx.in_scope(rel):
            continue
        # Changelog intentionally documents removals — demote to info and skip OpenAIModel flood
        is_changelog = 'changelog' in rel.lower()
        try:
            text = path.read_text(encoding='utf-8', errors='replace')
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if ctx.added_lines and not ctx.is_added_line(rel, lineno):
                continue
            for rule_id, pat, message, sev in _REMOVED:
                if not pat.search(line):
                    continue
                use_sev = Severity.INFO if is_changelog else sev
                # Soft-skip pure migration wording on non-changelog pages too for OpenAI/Gemini info flood
                if rule_id in {'docs_refs.OpenAIModel_bare', 'docs_refs.GeminiModel'} and is_changelog:
                    continue
                findings.append(
                    Finding(
                        check=CHECK,
                        severity=use_sev,
                        path=rel,
                        line=lineno,
                        message=message,
                        rule_id=rule_id,
                    )
                )
    return findings
