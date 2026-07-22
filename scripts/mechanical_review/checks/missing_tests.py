"""New public symbols without test mentions — diff mode only."""

from __future__ import annotations

import ast
import re
from pathlib import Path

from ..models import Finding, ScanContext, Severity

CHECK = 'missing_tests'


def run(ctx: ScanContext) -> list[Finding]:
    if not ctx.diff_base:
        return []  # --all: skip (diff-only check)

    findings: list[Finding] = []
    changed = ctx.file_filter or frozenset()
    slim_changed = [
        p
        for p in changed
        if p.startswith('pydantic_ai_slim/pydantic_ai/')
        and p.endswith('.py')
        and not Path(p).name.startswith('_')
        and '/_' not in p.replace('pydantic_ai_slim/pydantic_ai/', '', 1)
    ]

    # Build a cheap tests/ name index: identifiers mentioned as words
    tests_root = ctx.repo / 'tests'
    test_blob = ''
    if tests_root.is_dir():
        chunks: list[str] = []
        for p in tests_root.rglob('*.py'):
            if 'cassettes' in p.parts:
                continue
            try:
                chunks.append(p.read_text(encoding='utf-8', errors='replace'))
            except OSError:
                continue
        test_blob = '\n'.join(chunks)

    for rel in slim_changed:
        path = ctx.repo / rel
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding='utf-8', errors='replace')
            tree = ast.parse(text)
        except (OSError, SyntaxError):
            continue

        # Prefer symbols on added lines when we have that info
        symbols = _public_symbols(tree)
        if not symbols:
            continue

        # If we have a full-file parse from a changed file, still only flag newly added defs when possible
        added = ctx.added_lines.get(rel)
        for name, lineno in symbols:
            if added is not None and lineno not in added:
                # Also accept nearby: definition line might be the def line itself
                if not any(abs(lineno - a) <= 0 for a in added):
                    # Check if any added line is within the def span — skip if no added lines near def
                    if lineno not in added:
                        continue
            # Skip private
            if name.startswith('_'):
                continue
            # Word-boundary search in tests
            if re.search(rf'\b{re.escape(name)}\b', test_blob):
                continue
            findings.append(
                Finding(
                    check=CHECK,
                    severity=Severity.WARNING,
                    path=rel,
                    line=lineno,
                    message=f'new/changed public symbol `{name}` not mentioned under tests/',
                    rule_id='missing_tests.no_test_mention',
                )
            )
    return findings


def _public_symbols(tree: ast.Module) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith('_'):
                out.append((node.name, node.lineno))
    return out
