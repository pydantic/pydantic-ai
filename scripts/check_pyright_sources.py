"""Print a Pyright JSON report as Pyright does, and fail when Pyright read fewer files than its configuration names.

Pyright drops files from its configured scope without a diagnostic: 1.1.414 started applying its default
excludes (`**/.*` among them) over explicit `include` entries, and the twelve `.github/scripts` files the root
configuration listed went unchecked with CI green (https://github.com/pydantic/pydantic-ai/pull/8732).

`make typecheck-pyright` run with `PYRIGHT_REPORT=<report>` saves the root run's `--outputjson` report there and
runs `check_pyright_sources.py print <report>`, which prints its diagnostics in Pyright's own text format and exits 1
when the report holds an error or is not a report. `check_pyright_sources.py <report>` compares the report's
`summary.filesAnalyzed` with the expected count: every tracked `.py` and `.pyi` file under an `include` entry and
under no `exclude` entry, with none of Pyright's own excludes applied, so a file Pyright stops reading on its own
lowers the count it reports below that. The report carries the count because Pyright refuses `--stats` with
`--threads`, which CI passes.
"""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path
from typing import NotRequired

from pydantic import TypeAdapter, ValidationError
from typing_extensions import TypedDict


class _Position(TypedDict):
    line: int
    character: int


class _Range(TypedDict):
    start: _Position


class _Diagnostic(TypedDict):
    file: str
    severity: str
    message: str
    range: NotRequired[_Range]
    """Left out when the diagnostic has no position."""
    rule: NotRequired[str]


class _Summary(TypedDict):
    filesAnalyzed: int
    errorCount: int
    warningCount: int
    informationCount: int


class _Report(TypedDict):
    generalDiagnostics: list[_Diagnostic]
    summary: _Summary


_REPORT = TypeAdapter(_Report)


def configured_sources(pyproject: Path) -> list[str]:
    """The tracked source files the root Pyright configuration names."""
    pyright = tomllib.loads(pyproject.read_text(encoding='utf-8'))['tool']['pyright']
    include: list[str] = [entry.rstrip('/') for entry in pyright['include']]
    exclude: list[str] = [entry.rstrip('/') for entry in pyright.get('exclude', [])]
    tracked = subprocess.run(
        ['git', 'ls-files', '-z', '--', '*.py', '*.pyi'],
        capture_output=True,
        text=True,
        check=True,
        cwd=pyproject.parent,
    ).stdout.split('\0')

    def covered(path: str, entries: list[str]) -> bool:
        return any(path == entry or path.startswith(f'{entry}/') for entry in entries)

    return [path for path in tracked if path and covered(path, include) and not covered(path, exclude)]


def _plural(count: int, noun: str) -> str:
    return f'{count} {noun}' if count == 1 else f'{count} {noun}s'


def print_report(path: str) -> int:
    """Print the report's diagnostics and summary as Pyright's text output does; 1 if it holds an error."""
    text = Path(path).read_text(encoding='utf-8')
    try:
        report = _REPORT.validate_json(text)
    except ValidationError:
        print(text)
        print(f'{path} is not a Pyright JSON report.')
        return 1
    by_file: dict[str, list[_Diagnostic]] = {}
    for diagnostic in report['generalDiagnostics']:
        by_file.setdefault(diagnostic['file'], []).append(diagnostic)
    for file, diagnostics in by_file.items():
        print(file)
        for diagnostic in diagnostics:
            start = diagnostic['range']['start'] if 'range' in diagnostic else None
            where = f'{start["line"] + 1}:{start["character"] + 1} - ' if start else ' '
            rule = f' ({diagnostic["rule"]})' if 'rule' in diagnostic else ''
            message = diagnostic['message'].replace('\n', '\n  ')
            print(f'  {file}:{where}{diagnostic["severity"]}: {message}{rule}')
    summary = report['summary']
    print(
        f'{_plural(summary["errorCount"], "error")}, {_plural(summary["warningCount"], "warning")}, '
        f'{_plural(summary["informationCount"], "information")}'
    )
    return 1 if summary['errorCount'] else 0


def check_count(path: str) -> int:
    """Compare the number of files the report says Pyright read with the configured sources; the exit code."""
    read = _REPORT.validate_json(Path(path).read_text(encoding='utf-8'))['summary']['filesAnalyzed']
    expected = len(configured_sources(Path('pyproject.toml')))
    print(f'Pyright read {read} source files; the root `[tool.pyright]` configuration names {expected}.')
    if read < expected:
        print(
            f'Pyright skipped {expected - read} configured files without a diagnostic. Find them with '
            '`pyright --verbose`, then move them into a project Pyright reads or out of `include`.'
        )
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(print_report(sys.argv[2]) if sys.argv[1] == 'print' else check_count(sys.argv[1]))
