"""VCR cassette hygiene — size + likely secrets (streamed, fast)."""

from __future__ import annotations

import re
from pathlib import Path

from ..models import Finding, ScanContext, Severity

CHECK = 'cassettes'

SIZE_WARN_BYTES = 1_000_000
SECRET_READ_BYTES = 200_000

_SECRET_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    (
        'cassettes.sk_token',
        # Avoid base64 false positives: do not allow alnum / + / / / = before sk-
        re.compile(r'(?<![A-Za-z0-9+/=])sk-[a-zA-Z0-9]{16,}'),
        'likely OpenAI-style secret `sk-...` in cassette',
    ),
    (
        'cassettes.api_key_assign',
        # Only flag when a non-trivial value sits on the same line
        re.compile(
            r'api[_-]?key[\'\"]?\s*[:=]\s*[\'\"]?(?!REDACTED|xxx|dummy|fake|test|null|none|\s|\[)[A-Za-z0-9_\-]{12,}',
            re.I,
        ),
        'api_key assignment-like text with value in cassette',
    ),
    (
        'cassettes.bearer',
        re.compile(r'(?<![A-Za-z0-9+/=])Bearer\s+[A-Za-z0-9\-._~+/]{16,}=*', re.I),
        'Bearer token in cassette (should be redacted)',
    ),
    (
        'cassettes.akia',
        re.compile(r'(?<![A-Za-z0-9])AKIA[0-9A-Z]{16}(?![A-Za-z0-9])'),
        'likely AWS access key id in cassette',
    ),
    (
        'cassettes.authorization_header',
        re.compile(
            r'(?i)Authorization[\'\"]?\s*:\s*[\'\"]?(?!REDACTED|<.*>|xxx|\*+|\[.*\]|Bearer\s+REDACTED)'
            r'(?:Bearer\s+)?[A-Za-z0-9\-._~+/]{20,}'
        ),
        'Authorization header with long non-redacted value',
    ),
]


def _iter_cassettes(repo: Path) -> list[Path]:
    out: list[Path] = []
    tests = repo / 'tests'
    if tests.is_dir():
        for p in tests.rglob('*'):
            if not p.is_file():
                continue
            parts = p.parts
            if 'cassettes' in parts or p.name.endswith('.cassette.yaml') or p.name.endswith('.cassette.yml'):
                if p.suffix in {'.yaml', '.yml', '.json'} or '.cassette.' in p.name:
                    out.append(p)
    # Also top-level **/*.cassette.yaml
    for p in repo.rglob('*.cassette.yaml'):
        if p not in out:
            out.append(p)
    return out


def run(ctx: ScanContext) -> list[Finding]:
    findings: list[Finding] = []
    for path in _iter_cassettes(ctx.repo):
        rel = ctx.rel(path)
        if not ctx.in_scope(rel):
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > SIZE_WARN_BYTES:
            findings.append(
                Finding(
                    check=CHECK,
                    severity=Severity.WARNING,
                    path=rel,
                    message=f'cassette file size {size} bytes > {SIZE_WARN_BYTES}',
                    rule_id='cassettes.large_file',
                )
            )
        try:
            with path.open('rb') as fh:
                raw = fh.read(SECRET_READ_BYTES)
        except OSError:
            continue
        text = raw.decode('utf-8', errors='replace')
        # Line numbers approximate for first chunk only
        lines = text.splitlines()
        for lineno, line in enumerate(lines, start=1):
            if not ctx.is_added_line(rel, lineno):
                continue
            for rule_id, pat, message in _SECRET_PATTERNS:
                m = pat.search(line)
                if not m:
                    continue
                # Only suppress when the *matched credential* looks redacted —
                # not when an unrelated word (e.g. "example") appears elsewhere.
                matched = m.group(0)
                if re.search(r'(?i)redacted|dummy|fake|example|placeholder|xxxx|\*\*\*', matched):
                    continue
                findings.append(
                    Finding(
                        check=CHECK,
                        severity=Severity.ERROR,
                        path=rel,
                        line=lineno,
                        message=message,
                        rule_id=rule_id,
                    )
                )
    return findings
