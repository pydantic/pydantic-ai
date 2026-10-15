"""Combine the per-job coverage files for `strict-no-cover`, dropping most lines Python 3.11 jobs borrowed.

On Python 3.11, coverage's C tracer can file another frame's lines under a measured file. When an
exception thrown into a coroutine is handled by the coroutine it awaits, CPython 3.11 resumes the
outer coroutine without a `call` trace event, yet still reports its lines and its `return`
(https://github.com/python/cpython/issues/106749, not fixed in 3.11). The tracer pushes an entry per
`call` and pops one per `return`, so from then on it records each frame's remaining lines, such as the
rest of `asyncio`'s `run_until_complete`, under its caller's file. A borrowed line can land on a
`# pragma: no cover` line, which `strict-no-cover` then reports as covered.

This drops from the 3.11 files every arc coverage's parser does not predict, which removes the
borrowed arcs that leave the caller's control flow. A borrowed arc that matches a real arc of the
caller's file survives: mark a line it reaches `# pragma: lax no cover`. The arcs entering a function
stay, though the parser does not predict them either, because the tracer files a frame's first line
under the entry its own `call` pushed. Other files stay as recorded: Python 3.14's `sys.monitoring`
core records arcs the parser does not predict on every run. `coverage report` reads the data as
recorded.

Delete this once coverage ships a fix for https://github.com/coveragepy/coveragepy/issues/2124
(https://github.com/coveragepy/coveragepy/pull/2125), or once Python 3.11 leaves the test matrix.

Usage: strict_no_cover_data.py DATA_DIR TARGET
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

from coverage import Coverage, CoverageData
from coverage.exceptions import NoSource, NotPython
from coverage.files import PathAliases
from coverage.parser import PythonParser


def predicted_arcs(data: CoverageData, map_path: Callable[[str], str]) -> dict[str, list[tuple[int, int]]]:
    """The recorded arcs of each measured file that enter a function or that coverage's parser predicts."""
    predicted: dict[str, list[tuple[int, int]]] = {}
    for recorded_path in data.measured_files():
        arcs = data.arcs(recorded_path) or []
        path = map_path(recorded_path)
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
    """Combine the `.coverage.*` files in `data_dir` into `target`, keeping only predicted arcs of 3.11 files.

    Paths are remapped through `[tool.coverage.paths]`, as `coverage combine` does, so CI artifacts can be
    combined on another machine.
    """
    config = Coverage().config
    aliases = PathAliases(relative=config.relative_files)
    for result, *patterns in config.paths.values():
        for pattern in patterns:
            aliases.add(pattern, result)
    combined = CoverageData(str(target))
    for path in sorted(data_dir.glob('.coverage.*')):
        data = CoverageData(str(path))
        data.read()
        if path.name.startswith('.coverage.3.11-'):
            combined.add_arcs(predicted_arcs(data, aliases.map))
        else:
            combined.update(data, map_path=aliases.map)
    combined.write()


if __name__ == '__main__':
    combine(Path(sys.argv[1]), Path(sys.argv[2]))
