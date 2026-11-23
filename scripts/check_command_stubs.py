"""Fail when a change adds hand-written command output to a gate test under `.github/scripts`.

A gate test that builds a `CompletedProcess` with literal output tests the gate against what its author believed a
command prints. `gh`, `git` and `griffecli` printed something else in shipped defects, for example
https://github.com/pydantic/pydantic-ai/pull/10963. Record the real output through `.github/scripts/_commands.py` and
replay it with `_command_fixtures.Replay` instead.

The check counts, in each changed test file, the `CompletedProcess(...)` calls whose `stdout` or `stderr` is anything
but empty output or an attribute: any other expression, a name or a call such as `json.dumps(...)` included. Pass a
recorded value inline as the attribute, such as `recording.stdout`. Empty output claims nothing about what a command
prints, so it is not counted. A change may not raise a file's count; what the base already has is allowed.

During a merge, `--staged` reports a file only when its count is higher than in both parents.

Usage:
    uv run scripts/check_command_stubs.py --staged                 # the index against HEAD, for pre-commit
    uv run scripts/check_command_stubs.py --base REF               # the working tree against REF
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import PurePosixPath

from _ratchet import Change, Options, escape, merge_head, parse_args, read_changes, worse_than_both

ROOTS = ('.github/scripts',)
GUIDE = '.github/scripts/_commands.py'


@dataclass
class Growth:
    """A test file whose count of hand-written command outputs grew."""

    path: str
    old: int
    new: int
    lines: list[int]


def hand_written_outputs(source: str | None) -> list[int]:
    """The line of each `CompletedProcess` call in `source` whose output is neither empty nor an attribute."""
    if source is None:
        return []
    lines: list[int] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else None
        if name != 'CompletedProcess':
            continue
        outputs = [
            *node.args[2:4],
            *(keyword.value for keyword in node.keywords if keyword.arg in {'stdout', 'stderr'}),
        ]
        if any(
            not isinstance(output, ast.Attribute) and not (isinstance(output, ast.Constant) and not output.value)
            for output in outputs
        ):
            lines.append(node.lineno)
    return sorted(lines)


def growth(changes: list[Change]) -> list[Growth]:
    """Each changed gate test whose count of hand-written command outputs grew."""
    grown: list[Growth] = []
    for change in changes:
        path = change.new_path
        if path is None or not PurePosixPath(path).name.startswith('test_'):
            continue
        old, new = hand_written_outputs(change.old_source), hand_written_outputs(change.new_source)
        if len(new) > len(old):
            grown.append(Growth(path, len(old), len(new), new))
    return grown


def main(argv: Sequence[str] | None = None) -> int:
    """Run the check and return the exit code."""
    options: Options = parse_args(argv, 'Check the hand-written command output a change adds to the gate tests.')
    grown = growth(read_changes(options, ROOTS))
    theirs = merge_head() if options.staged else None
    if theirs:
        their_growth = growth(read_changes(replace(options, base=theirs), ROOTS))
        grown = worse_than_both(grown, their_growth, key=lambda g: g.path, before=lambda g: g.old)
    for g in grown:
        message = (
            f'hand-written command output grew from {g.old} to {g.new}: record the real output with '
            f'`{GUIDE}` and replay it with `_command_fixtures.Replay`'
        )
        if options.github:
            print(f'::error file={escape(g.path, prop=True)},line={g.lines[0]}::{escape(message)}')
        else:
            print(f'{g.path}:{g.lines[0]}: {message} (lines {", ".join(map(str, g.lines))})')
    return 1 if grown else 0


if __name__ == '__main__':
    sys.exit(main())
