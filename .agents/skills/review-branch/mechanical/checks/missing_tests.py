"""New public symbols without test mentions — diff mode only."""

from __future__ import annotations

import ast
import re
from pathlib import Path

from ..models import PACKAGE_ROOTS, Finding, ScanContext, Severity, is_public_module, package_root

CHECK = 'missing_tests'
# Harness and clai2 define public classes in `_` modules and re-export them through `__all__`.
_REEXPORT_ROOTS = PACKAGE_ROOTS[1:]


def run(ctx: ScanContext) -> list[Finding]:  # noqa: C901
    if not ctx.diff_base:
        return []  # --all: skip (diff-only check)

    findings: list[Finding] = []
    changed = ctx.file_filter or frozenset()
    changed_modules = [
        p for p in changed if is_public_module(p) or (p.endswith('.py') and package_root(p) in _REEXPORT_ROOTS)
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

    for rel in changed_modules:
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
        if not is_public_module(rel):
            exported = _package_exports(path)
            symbols = [(name, lineno) for name, lineno in symbols if name in exported]
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


def _package_exports(path: Path) -> set[str]:
    """Literal `__all__` names of the package `__init__.py` beside `path`."""
    init = path.parent / '__init__.py'
    try:
        tree = ast.parse(init.read_text(encoding='utf-8', errors='replace'))
    except (OSError, SyntaxError):
        return set()
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == '__all__' for t in node.targets)
            and isinstance(node.value, (ast.List, ast.Tuple))
        ):
            return {e.value for e in node.value.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)}
    return set()


def _public_symbols(tree: ast.Module) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith('_'):
                out.append((node.name, node.lineno))
    return out
