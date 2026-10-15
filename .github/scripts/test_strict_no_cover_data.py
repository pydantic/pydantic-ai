from __future__ import annotations

import sys
from pathlib import Path

from coverage import CoverageData

sys.path.insert(0, str(Path(__file__).parent))

from strict_no_cover_data import combine

MODULE = """\
def first(run):
    value = run()
    if value is None:
        return None  # pragma: no cover
    return value
"""


def write(path: Path, arcs: dict[str, list[tuple[int, int]]]) -> None:
    """Write a job's coverage file as the test cells upload it."""
    data = CoverageData(str(path))
    data.add_arcs(arcs)
    data.write()


def test_drops_arcs_a_311_job_borrowed(tmp_path: Path) -> None:
    """Arcs a 3.11 job filed under the wrong file go; a 3.14 job's unpredicted arcs stay."""
    module = tmp_path / 'module.py'
    module.write_text(MODULE)
    name = str(module)
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    # While `first` waited on its call at line 2, the rest of `asyncio`'s `run_until_complete` was filed
    # under it, reaching line 4.
    write(data_dir / '.coverage.3.11-all-extras-1', {name: [(-1, 2), (2, 650), (650, 4), (4, -618)]})
    # The `sys.monitoring` core records arcs the parser does not predict, such as a line to itself.
    write(data_dir / '.coverage.3.14-standard', {name: [(3, 3)]})

    combine(data_dir, tmp_path / 'audit')

    audit = CoverageData(str(tmp_path / 'audit'))
    audit.read()
    assert sorted(audit.arcs(name) or []) == [(-1, 2), (3, 3)]
