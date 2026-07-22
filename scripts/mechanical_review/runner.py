#!/usr/bin/env python3
"""CLI + orchestration for mechanical review checks (stdlib only)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# Support `python3 path/to/mechanical_review/runner.py` as well as
# `python3 -m mechanical_review` (parent of package on sys.path).
if __name__ == '__main__' and (__package__ is None or __package__ == ''):
    _pkg = Path(__file__).resolve().parent
    _parent = str(_pkg.parent)
    if _parent not in sys.path:
        sys.path.insert(0, _parent)
    __package__ = 'mechanical_review'

from .checks import ALL_CHECKS, run_checks
from .models import Finding, ScanContext, Severity, count_by_check, sort_findings


def _git(repo: Path, *args: str) -> str:
    r = subprocess.run(
        ['git', '-C', str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if r.returncode != 0:
        err = (r.stderr or r.stdout or '').strip()
        raise RuntimeError(f'git {" ".join(args)} failed: {err}')
    return r.stdout


def _diff_changed_files(repo: Path, base: str) -> frozenset[str]:
    out = _git(repo, 'diff', '--name-only', base)
    return frozenset(line.strip().replace('\\', '/') for line in out.splitlines() if line.strip())


def _diff_added_lines(repo: Path, base: str) -> dict[str, set[int]]:
    """Map path -> set of new-file line numbers for added lines (`+` hunks)."""
    out = _git(repo, 'diff', '--unified=0', base)
    added: dict[str, set[int]] = {}
    path: str | None = None
    new_line = 0
    for raw in out.splitlines():
        if raw.startswith('+++ '):
            rest = raw[4:].strip()
            if rest == '/dev/null':
                path = None
                continue
            if rest.startswith('b/'):
                rest = rest[2:]
            path = rest.replace('\\', '/')
            added.setdefault(path, set())
            continue
        if raw.startswith('@@'):
            # @@ -a,b +c,d @@
            try:
                plus = raw.split('+')[1].split('@@')[0].strip()
                start = int(plus.split(',')[0])
            except (IndexError, ValueError):
                start = 0
            new_line = start
            continue
        if path is None:
            continue
        if raw.startswith('+') and not raw.startswith('+++'):
            added.setdefault(path, set()).add(new_line)
            new_line += 1
        elif raw.startswith('-') and not raw.startswith('---'):
            pass
        else:
            # context (shouldn't appear with -U0) or `\ No newline`
            if raw.startswith(' '):
                new_line += 1
    return added


def build_context(repo: Path, *, mode_all: bool, diff_base: str | None) -> ScanContext:
    repo = repo.resolve()
    if not repo.is_dir():
        raise SystemExit(f'repo path does not exist: {repo}')
    file_filter: frozenset[str] | None = None
    added: dict[str, set[int]] = {}
    if diff_base:
        try:
            file_filter = _diff_changed_files(repo, diff_base)
            added = _diff_added_lines(repo, diff_base)
        except RuntimeError as e:
            raise SystemExit(str(e)) from e
    return ScanContext(
        repo=repo,
        mode_all=mode_all,
        diff_base=diff_base,
        file_filter=file_filter,
        added_lines=added,
    )


def format_human(findings: list[Finding], elapsed_s: float, checks: list[str]) -> str:
    lines: list[str] = []
    lines.append(f'Mechanical review checks — {len(checks)} check(s) in {elapsed_s:.2f}s')
    lines.append(f'Total findings: {len(findings)}')
    counts = count_by_check(findings)
    if counts:
        lines.append('')
        lines.append('By check:')
        for name in checks:
            c = counts.get(name, {'error': 0, 'warning': 0, 'info': 0, 'total': 0})
            lines.append(
                f'  {name:16} total={c["total"]:4d}  error={c["error"]:3d}  '
                f'warning={c["warning"]:3d}  info={c["info"]:3d}'
            )
    else:
        lines.append('By check: (none)')

    # Severity totals
    sevs = {Severity.ERROR: 0, Severity.WARNING: 0, Severity.INFO: 0}
    for f in findings:
        sevs[f.severity] = sevs.get(f.severity, 0) + 1
    lines.append('')
    lines.append(
        f'Severity totals: error={sevs[Severity.ERROR]}  '
        f'warning={sevs[Severity.WARNING]}  info={sevs[Severity.INFO]}'
    )

    top = sort_findings(findings)[:30]
    if top:
        lines.append('')
        lines.append(f'Top findings ({len(top)} shown):')
        for f in top:
            loc = f'{f.path}:{f.line}' if f.line else f.path
            lines.append(f'  [{f.severity.value:7}] {f.check}/{f.rule_id}  {loc}')
            lines.append(f'            {f.message}')
    else:
        lines.append('')
        lines.append('No findings.')
    return '\n'.join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog='pyai-review-checks',
        description='Fast stdlib-only mechanical review checks for pydantic-ai.',
    )
    p.add_argument('--repo', type=Path, required=True, help='path to pydantic-ai checkout')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        '--all',
        action='store_true',
        help='full-tree scan (default when --diff is omitted)',
    )
    mode.add_argument(
        '--diff',
        metavar='BASE',
        help='only consider changes vs git BASE (e.g. merge-base sha or origin/main)',
    )
    p.add_argument(
        '--checks',
        default=','.join(ALL_CHECKS),
        help=f'comma-separated subset (default: all). Known: {", ".join(ALL_CHECKS)}',
    )
    p.add_argument('--json', action='store_true', help='machine-readable JSON on stdout')
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    diff_base: str | None = args.diff
    mode_all = bool(args.all) or diff_base is None
    if args.diff:
        mode_all = False

    names = [c.strip() for c in args.checks.split(',') if c.strip()]
    for n in names:
        if n not in ALL_CHECKS:
            print(f'unknown check: {n!r}; known: {", ".join(ALL_CHECKS)}', file=sys.stderr)
            return 2

    ctx = build_context(args.repo, mode_all=mode_all, diff_base=diff_base)
    t0 = time.perf_counter()
    findings = sort_findings(run_checks(ctx, names))
    elapsed = time.perf_counter() - t0

    if args.json:
        payload = {
            'repo': str(ctx.repo),
            'mode': 'diff' if diff_base else 'all',
            'diff_base': diff_base,
            'checks': names,
            'elapsed_s': round(elapsed, 3),
            'counts': count_by_check(findings),
            'findings': [f.to_dict() for f in findings],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(format_human(findings, elapsed, names))

    has_error = any(f.severity == Severity.ERROR for f in findings)
    return 1 if has_error else 0


if __name__ == '__main__':
    # Allow `python3 runner.py` when cwd / path includes parent of package.
    # Prefer: PYTHONPATH=scripts python3 -m mechanical_review
    raise SystemExit(main())
