"""High-signal mechanical patterns (regex + light AST) on .py files."""

from __future__ import annotations

import ast
import re
from pathlib import Path

from ..models import Finding, ScanContext, Severity

CHECK = 'patterns'

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
    'mechanical_review',
}

# Bare `# type: ignore` without `[code]`
_BARE_TYPE_IGNORE = re.compile(r'#\s*type:\s*ignore(?!\s*\[)')
# Bare `# pyright: ignore` without `[code]` (file-level or unspecific)
_BARE_PYRIGHT_IGNORE = re.compile(r'#\s*pyright:\s*ignore(?!\s*\[)')
# Line-number refs in comments/docstrings: "line 42", "lines 10-20", " L123 " / "(L123)".
# Avoid GitHub URL fragments (#L36), model ids (MiniLM-L6-v2), and bare #L anchors.
_LINE_REF = re.compile(
    r'(?i)(?:\blines?\s+\d+(?:\s*[-–—]\s*\d+)?\b|(?<![#\w])\bL\d{2,5}\b(?!\w))'
)
_URL_OR_ANCHOR = re.compile(r'https?://|#L\d')
_WARN_CALL = re.compile(r'\bwarnings\.warn\s*\(')
_STACKLEVEL = re.compile(r'\bstacklevel\s*=')
_EMPTY_SNAPSHOT = re.compile(r'\bsnapshot\s*\(\s*\)')
_ANY_ANN = re.compile(r'(->\s*Any\b|:\s*Any\b)')
# Cap Any findings in --all to avoid flood
_ANY_CAP = 40


def _iter_py_files(repo: Path) -> list[Path]:
    roots = [
        repo / 'pydantic_ai_slim',
        repo / 'tests',
        repo / 'clai',
    ]
    out: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for p in root.rglob('*.py'):
            if any(part in _SKIP_DIR_NAMES for part in p.parts):
                continue
            out.append(p)
    return out


def run(ctx: ScanContext) -> list[Finding]:
    findings: list[Finding] = []
    any_count = 0

    for path in _iter_py_files(ctx.repo):
        rel = ctx.rel(path)
        if not ctx.in_scope(rel):
            continue
        try:
            text = path.read_text(encoding='utf-8', errors='replace')
        except OSError:
            continue
        lines = text.splitlines()
        is_test = rel.startswith('tests/') or '/tests/' in rel
        is_public_slim = rel.startswith('pydantic_ai_slim/pydantic_ai/') and not any(
            part.startswith('_') for part in Path(rel).parts[2:-1]
        )

        for lineno, line in enumerate(lines, start=1):
            if ctx.added_lines and not ctx.is_added_line(rel, lineno):
                continue

            if _BARE_TYPE_IGNORE.search(line):
                findings.append(
                    Finding(
                        check=CHECK,
                        severity=Severity.WARNING,
                        path=rel,
                        line=lineno,
                        message='bare `# type: ignore` — use `# pyright: ignore[code]`',
                        rule_id='patterns.bare_type_ignore',
                    )
                )

            if _BARE_PYRIGHT_IGNORE.search(line) and not is_test:
                findings.append(
                    Finding(
                        check=CHECK,
                        severity=Severity.WARNING,
                        path=rel,
                        line=lineno,
                        message='unspecific `# pyright: ignore` — add `[code]` or use inline on the offending line',
                        rule_id='patterns.bare_pyright_ignore',
                    )
                )

            # Line-number refs: comment lines only (avoid traceback/error-string FPs).
            # Also single-line docstrings that are pure documentation.
            stripped = line.lstrip()
            is_comment = stripped.startswith('#')
            is_oneline_doc = bool(
                re.match(r'[ruRU]?("""|\'\'\').*(line|L\d)', stripped, re.I)
                and stripped.count('"""') + stripped.count("'''") >= 2
            )
            if (
                (is_comment or is_oneline_doc)
                and _LINE_REF.search(line)
                and not _URL_OR_ANCHOR.search(line)
                and not re.search(r'(?i)\bline\s+length\b|\blines?\s+of\b|\bline\s+no\b', line)
            ):
                findings.append(
                    Finding(
                        check=CHECK,
                        severity=Severity.ERROR,
                        path=rel,
                        line=lineno,
                        message='line-number reference in comment/docstring goes stale — describe behavior instead',
                        rule_id='patterns.line_number_ref',
                    )
                )

            if _WARN_CALL.search(line) and not _STACKLEVEL.search(line):
                # Single-line heuristic only (multiline warn without stacklevel is harder)
                if line.rstrip().endswith(')') or 'stacklevel' not in line:
                    # If the call clearly continues, skip; if closed on same line without stacklevel, flag.
                    if line.count('(') <= line.count(')') and ')' in line:
                        findings.append(
                            Finding(
                                check=CHECK,
                                severity=Severity.WARNING,
                                path=rel,
                                line=lineno,
                                message='`warnings.warn(...)` missing `stacklevel=`',
                                rule_id='patterns.warn_no_stacklevel',
                            )
                        )

            # Only assert/compare forms — skip docs/helper strings that mention snapshot().
            if is_test and _EMPTY_SNAPSHOT.search(line) and re.search(
                r'\bassert\b.*=\s*snapshot\s*\(\s*\)|\bsnapshot\s*\(\s*\)\s*==', line
            ):
                findings.append(
                    Finding(
                        check=CHECK,
                        severity=Severity.WARNING,
                        path=rel,
                        line=lineno,
                        message='empty `snapshot()` — run with `--inline-snapshot=create` to populate',
                        rule_id='patterns.empty_snapshot',
                    )
                )

            # Any annotations: --diff = added lines; --all = public slim only, info, capped
            if _ANY_ANN.search(line):
                if ctx.diff_base is not None:
                    findings.append(
                        Finding(
                            check=CHECK,
                            severity=Severity.INFO,
                            path=rel,
                            line=lineno,
                            message='`Any` annotation — prefer a specific type or `object`',
                            rule_id='patterns.any_annotation',
                        )
                    )
                elif is_public_slim and any_count < _ANY_CAP:
                    any_count += 1
                    findings.append(
                        Finding(
                            check=CHECK,
                            severity=Severity.INFO,
                            path=rel,
                            line=lineno,
                            message='`Any` annotation on public slim module (capped sample in --all)',
                            rule_id='patterns.any_annotation',
                        )
                    )

        # AST passes: importorskip inside functions
        if is_test:
            findings.extend(_ast_test_patterns(text, rel))

    return findings


def _ast_test_patterns(text: str, rel: str) -> list[Finding]:
    out: list[Finding] = []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return out

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        # Skip if this is module-level... we're iterating FunctionDef so body is nested.
        for child in ast.walk(node):
            if child is node:
                continue
            if isinstance(child, ast.Call):
                name = _call_name(child)
                if name == 'pytest.importorskip' or name == 'importorskip':
                    # Only if the FunctionDef is not a nested helper at module... all function bodies count.
                    # Module-level is not under FunctionDef, so this is correct.
                    out.append(
                        Finding(
                            check=CHECK,
                            severity=Severity.WARNING,
                            path=rel,
                            line=getattr(child, 'lineno', None),
                            message='`pytest.importorskip` inside function body — use module-level `try_import()` + skipif',
                            rule_id='patterns.importorskip_in_function',
                        )
                    )
    return out


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return f'{func.value.id}.{func.attr}'
    return ''
