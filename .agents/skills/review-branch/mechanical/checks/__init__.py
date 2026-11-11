"""Registry of mechanical checks."""

from __future__ import annotations

from ..models import CheckFn, Finding, ScanContext
from . import cassettes, denylist, docs_refs, missing_tests, mode_b, patterns

CHECK_REGISTRY: dict[str, CheckFn] = {
    'denylist': denylist.run,
    'patterns': patterns.run,
    'mode_b': mode_b.run,
    'cassettes': cassettes.run,
    'docs_refs': docs_refs.run,
    'missing_tests': missing_tests.run,
}

ALL_CHECKS = tuple(CHECK_REGISTRY.keys())


def run_checks(ctx: ScanContext, names: list[str] | None = None) -> list[Finding]:
    selected = names if names is not None else list(ALL_CHECKS)
    findings: list[Finding] = []
    for name in selected:
        fn = CHECK_REGISTRY.get(name)
        if fn is None:
            raise SystemExit(f'unknown check: {name!r} (known: {", ".join(ALL_CHECKS)})')
        findings.extend(fn(ctx))
    return findings
