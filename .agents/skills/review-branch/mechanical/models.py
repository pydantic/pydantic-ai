"""Finding model and shared scan context for mechanical review checks."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path

# Production package roots in the pydantic-ai monorepo.
PACKAGE_ROOTS = (
    'pydantic_ai_slim/pydantic_ai/',
    'pydantic_graph/pydantic_graph/',
    'pydantic_evals/pydantic_evals/',
    'clai/clai/',
    'src/pydantic_ai_harness/pydantic_ai_harness/',
    'src/pydantic_clai2/pydantic_clai2/',
)


def package_root(rel_path: str) -> str | None:
    return next((root for root in PACKAGE_ROOTS if rel_path.startswith(root)), None)


def is_public_module(rel_path: str) -> bool:
    """A `.py` under a package root with no `_`-prefixed directory or file name, counting `__init__.py` as public."""
    root = package_root(rel_path)
    if root is None or not rel_path.endswith('.py'):
        return False
    *dirs, name = rel_path[len(root) :].split('/')
    return not any(part.startswith('_') for part in dirs) and (name == '__init__.py' or not name.startswith('_'))


class Severity(StrEnum):
    ERROR = 'error'
    WARNING = 'warning'
    INFO = 'info'


@dataclass(frozen=True, slots=True)
class Finding:
    check: str
    severity: Severity
    path: str
    message: str
    rule_id: str
    line: int | None = None

    def to_dict(self) -> dict[str, object]:
        d = asdict(self)
        d['severity'] = self.severity.value
        return d


@dataclass(slots=True)
class ScanContext:
    """Shared scan inputs passed to every check."""

    repo: Path
    mode_all: bool
    diff_base: str | None = None
    # Relative POSIX paths (repo-relative) limited by --diff, or None for --all.
    file_filter: frozenset[str] | None = None
    # path -> 1-based line numbers of added lines (diff mode only).
    added_lines: dict[str, set[int]] = field(default_factory=dict[str, set[int]])

    def rel(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.repo.resolve()).as_posix()
        except ValueError:
            return path.as_posix()

    def in_scope(self, rel_path: str) -> bool:
        if self.file_filter is None:
            return True
        return rel_path in self.file_filter

    def is_added_line(self, rel_path: str, line: int) -> bool:
        if not self.added_lines:
            return True  # --all: treat every line as in-scope
        lines = self.added_lines.get(rel_path)
        if lines is None:
            return False
        return line in lines


CheckFn = Callable[[ScanContext], list[Finding]]


def sort_findings(findings: Iterable[Finding]) -> list[Finding]:
    sev_order = {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}
    return sorted(
        findings,
        key=lambda f: (sev_order.get(f.severity, 9), f.check, f.path, f.line or 0, f.rule_id),
    )


def count_by_check(findings: Sequence[Finding]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for f in findings:
        bucket = out.setdefault(f.check, {'error': 0, 'warning': 0, 'info': 0, 'total': 0})
        bucket[f.severity.value] += 1
        bucket['total'] += 1
    return out
