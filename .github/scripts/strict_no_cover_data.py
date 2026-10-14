"""Combine the per-job coverage files for `strict-no-cover`, dropping the arcs Python 3.11 jobs borrowed.

On Python 3.11, coverage's C tracer can file another frame's lines under a measured file. When an
exception thrown into a coroutine is handled by the coroutine it awaits, CPython 3.11 resumes the
outer coroutine without a `call` trace event, yet still reports its lines and its `return`. The tracer
pushes an entry per `call` and pops one per `return`, so from then on it records each frame's
remaining lines, such as the rest of `asyncio`'s `run_until_complete`, under its caller's file. A
borrowed line can land on a `# pragma: no cover` line, which `strict-no-cover` then reports as covered.

Borrowed lines arrive as arcs no path through the file takes, so this drops every arc coverage's parser
does not predict from the 3.11 files. It keeps the arcs entering a function, which the parser does not
predict either, because the tracer files a frame's first line under the entry its own `call` pushed. It
leaves other files alone: Python 3.14's `sys.monitoring` core records arcs the parser does not predict
on every run. `coverage report` reads the data as recorded.

Usage: strict_no_cover_data.py DATA_DIR TARGET
"""

from __future__ import annotations

import sys
from pathlib import Path

from coverage import CoverageData
from coverage.exceptions import NoSource, NotPython
from coverage.parser import PythonParser


def predicted_arcs(data: CoverageData) -> dict[str, list[tuple[int, int]]]:
    """The recorded arcs of each measured file that enter a function or that coverage's parser predicts."""
    predicted: dict[str, list[tuple[int, int]]] = {}
    for path in data.measured_files():
        arcs = data.arcs(path) or []
        try:
            parser = PythonParser(filename=path)
            parser.parse_source()
        except (NoSource, NotPython):
            predicted[path] = arcs
            continue
        possible = parser.arcs()
        predicted[path] = [
            (start, end)
            for start, end in arcs
            if start < 0 or (parser.first_line(start), parser.first_line(end)) in possible
        ]
    return predicted


def combine(data_dir: Path, target: Path) -> None:
    """Combine the `.coverage.*` files in `data_dir` into `target`, keeping only predicted arcs of 3.11 files."""
    combined = CoverageData(str(target))
    for path in sorted(data_dir.glob('.coverage.*')):
        data = CoverageData(str(path))
        data.read()
        if path.name.startswith('.coverage.3.11-'):
            combined.add_arcs(predicted_arcs(data))
        else:
            combined.update(data)
    combined.write()


if __name__ == '__main__':
    combine(Path(sys.argv[1]), Path(sys.argv[2]))
