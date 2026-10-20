from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from postgres_pilot import Event, Evidence, Server, Trial, cells, compare


@pytest.fixture
def evidence() -> list[Evidence]:
    """Build a complete paired batch with deterministic timings and PostgreSQL coverage."""
    result: list[Evidence] = []
    for index, cell in enumerate(cells()):
        shards = 3 if cell.python == '3.11' else 2
        selected = range(1, shards + 1) if cell.selection == 'harness' else [cell.shard]
        nodes = [
            f'tests/harness/{integration}/test_{integration}.py::test_{cell.python}_{shard}'
            for integration in ('absurd', 'pgtask')
            for shard in selected
        ]
        result.append(
            Evidence(
                trial=Trial(
                    cell=cell,
                    source='same-source',
                    runner={'ImageOS': 'ubuntu24', 'ImageVersion': 'fixed'},
                    hashes={'uv.lock': 'same-lock', 'baseline': 'same-baseline'},
                    started_at='same-batch',
                    ready_seconds=float(index // 2 + (10 if cell.strategy == 'docker' else 1)),
                    server=Server(
                        version='PostgreSQL 16',
                        version_num=160015,
                        encoding='UTF8',
                        collate='en_US.utf8',
                        ctype='en_US.utf8',
                        timezone='Etc/UTC',
                        ordering=['A', 'a'],
                    ),
                    image_digest=['postgres@sha256:one-revision'] if cell.strategy == 'docker' else [],
                    probes_passed=True,
                    packages_unchanged=True,
                    test_exit=0,
                    smoke_exit=0,
                    smoke_skips=['absurd:missing URL', 'pgtask:missing URL']
                    if cell.python == '3.14' and cell.shard == 1
                    else [],
                ),
                collection=nodes,
                events=[Event(nodeid=node, phase='call', outcome='passed') for node in nodes],
            )
        )
    return result


def test_complete_paired_batch_and_nearest_rank(evidence: list[Evidence]) -> None:
    """Check the approved statistics against known observations."""
    result = compare(evidence)
    assert all(result.gates.values())
    assert result.median_seconds == {'docker': 17.5, 'native': 8.5}
    assert result.p95_seconds == {'docker': 25.0, 'native': 16.0}


@pytest.mark.parametrize(
    'change', ['missing', 'duplicate', 'collection', 'skip', 'failure', 'locale', 'package', 'digest', 'slow']
)
def test_reject_invalid_adoption_evidence(evidence: list[Evidence], change: str) -> None:
    """Reject each material correctness or timing failure."""
    native = evidence[1]
    if change == 'missing':
        evidence.pop()
    elif change == 'duplicate':
        evidence[-1] = evidence[0]
    elif change == 'collection':
        native.collection.append('new-test')
    elif change == 'skip':
        native.events[0] = Event(
            nodeid=native.collection[0], phase='setup', outcome='skipped', reason='database unavailable'
        )
    elif change == 'failure':
        native.trial.test_exit = 1
    elif change == 'locale':
        assert native.trial.server is not None
        native.trial.server.collate = 'C.UTF-8'
    elif change == 'digest':
        evidence[0].trial.image_digest = ['postgres@sha256:different-revision']
    elif change == 'package':
        native.trial.packages_unchanged = False
    else:
        native.trial.ready_seconds = 1000
    assert not all(compare(evidence).gates.values())


def test_report_order_does_not_change_equivalence(evidence: list[Evidence]) -> None:
    """Ignore interleaving differences from parallel workers."""
    for item in evidence:
        item.events.append(Event(nodeid='another-test', phase='setup', outcome='skipped', reason='known skip'))
    evidence[1].events.reverse()
    assert all(compare(evidence).gates.values())


def test_empty_batch_cannot_pass() -> None:
    """Treat absent evidence as a failed adoption gate."""
    assert not all(compare([]).gates.values())
