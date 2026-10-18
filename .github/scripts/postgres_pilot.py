"""Collect a bounded PostgreSQL startup and test-equivalence experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import time
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, TypeAdapter


class Cell(BaseModel):
    """One fresh runner and production test selection."""

    strategy: Literal['docker', 'native']
    python: str
    selection: Literal['full', 'harness']
    shard: int
    shards: int

    @property
    def key(self) -> str:
        return f'{self.selection}-{self.python}-{self.shard}'

    @property
    def name(self) -> str:
        return f'{self.strategy}-{self.key}'


def cells() -> list[Cell]:
    """Pair every production all-extras and harness-only cell."""
    result: list[Cell] = []
    for version in ['3.11', '3.12', '3.13', '3.14', '3.15']:
        for selection in ('full', 'harness'):
            shards = 1 if selection == 'harness' else 3 if version == '3.11' else 2
            for shard in range(1, shards + 1):
                for strategy in ('docker', 'native'):
                    result.append(
                        Cell(strategy=strategy, python=version, selection=selection, shard=shard, shards=shards)
                    )
    return result


class Server(BaseModel):
    """Observed PostgreSQL version and database defaults."""

    version: str
    version_num: int
    encoding: str
    collate: str
    ctype: str
    timezone: str
    ordering: list[str]


class Trial(BaseModel):
    """Raw observations from one fresh-runner attempt."""

    cell: Cell
    source: str
    runner: dict[str, str]
    hashes: dict[str, str]
    started_at: str
    ready_seconds: float | None = None
    health_seconds: float | None = None
    compressed_layer_bytes_estimate: int = 0
    pull_seconds: float = 0
    pull_interface_rx_bytes: int = 0
    image_digest: list[str] = Field(default_factory=list)
    server: Server | None = None
    probes_passed: bool = False
    packages_unchanged: bool = False
    test_exit: int | None = None
    smoke_exit: int | None = None
    smoke_skips: list[str] = Field(default_factory=list)
    error: str | None = None


class Event(BaseModel):
    """A controller-side pytest phase outcome."""

    nodeid: str
    phase: str
    outcome: str
    reason: str = ''


class Route(BaseModel):
    """The runner interface used for default-route ingress."""

    dev: str


class ImageIndex(BaseModel):
    """The multi-platform index fetched after pulling the immutable image digest."""

    class Descriptor(BaseModel):
        """An image manifest with its platform."""

        class Platform(BaseModel):
            """OCI image platform."""

            architecture: str
            os: str

        digest: str
        platform: Platform

    manifests: list[Descriptor]


class ImageManifest(BaseModel):
    """Compressed registry layer sizes, distinct from measured transfer."""

    class Layer(BaseModel):
        """A compressed layer descriptor."""

        size: int

    layers: list[Layer]


class ArtifactList(BaseModel):
    """Public raw-evidence artifact links."""

    class Artifact(BaseModel):
        """One artifact in this workflow run."""

        id: int
        name: str

    artifacts: list[Artifact]


class Recorder:
    """Run bounded commands and retain their output and duration."""

    def __init__(self, directory: Path):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)

    def run(self, *command: str, password: str = 'postgres', check: bool = True) -> subprocess.CompletedProcess[str]:
        start = time.monotonic()
        result = subprocess.run(
            command, text=True, capture_output=True, env={**os.environ, 'PGPASSWORD': password}, timeout=120
        )
        record: dict[str, object] = {
            'command': command,
            'seconds': time.monotonic() - start,
            'exit': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
        }
        with (self.directory / 'commands.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record) + '\n')
        if check:
            result.check_returncode()
        return result

    def sql(
        self, statement: str, *, password: str = 'postgres', check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        return self.run(
            'psql',
            '-X',
            '-h',
            '127.0.0.1',
            '-U',
            'postgres',
            '-d',
            'postgres',
            '-v',
            'ON_ERROR_STOP=1',
            '-At',
            '-c',
            statement,
            password=password,
            check=check,
        )


def start(cell: Cell, directory: Path, baseline: Path) -> None:
    """Measure cold startup through authenticated readiness, then exercise SQL semantics."""
    recorder = Recorder(directory)
    trial = Trial(
        cell=cell,
        source=os.environ['GITHUB_SHA'],
        started_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        runner={
            key: os.environ.get(key, '')
            for key in ['ImageOS', 'ImageVersion', 'RUNNER_OS', 'RUNNER_ARCH', 'RUNNER_NAME']
        },
        hashes={
            'uv.lock': hashlib.sha256(Path('uv.lock').read_bytes()).hexdigest(),
            'baseline': hashlib.sha256(baseline.read_bytes()).hexdigest(),
        },
    )
    path = directory / 'trial.json'
    path.write_text(trial.model_dump_json(indent=2))
    try:
        trial.runner['platform'] = platform.platform()
        trial.runner['cpu'] = recorder.run('lscpu').stdout
        trial.runner['memory'] = recorder.run('free', '-b').stdout
        trial.runner['psql'] = recorder.run('psql', '--version').stdout.strip()
        trial.runner['docker'] = recorder.run('docker', 'version', '--format', '{{json .}}').stdout.strip()
        packages = recorder.run('dpkg-query', '-W', '-f=$' + '{Package}\t$' + '{Version}\n', 'postgresql*').stdout
        (directory / 'packages-before.txt').write_text(packages)
        images = recorder.run('docker', 'image', 'ls', '--digests', '--no-trunc').stdout
        (directory / 'images-before.txt').write_text(images)
        assert recorder.run('pg_isready', '-h', '127.0.0.1', check=False).returncode != 0, 'Port 5432 is already live'
        assert recorder.run('docker', 'image', 'inspect', 'postgres:16', check=False).returncode != 0, 'Warm image'
        route = TypeAdapter(list[Route]).validate_json(recorder.run('ip', '-j', 'route', 'show', 'default').stdout)[0]
        rx_path = Path('/sys/class/net') / route.dev / 'statistics/rx_bytes'
        startup = time.monotonic()
        if cell.strategy == 'docker':
            before_rx = int(rx_path.read_text())
            pulling = time.monotonic()
            recorder.run('docker', 'pull', 'postgres:16')
            trial.pull_seconds = time.monotonic() - pulling
            trial.pull_interface_rx_bytes = int(rx_path.read_text()) - before_rx
            recorder.run(
                'docker',
                'run',
                '--detach',
                '--name',
                'postgres-pilot',
                '-e',
                'POSTGRES_PASSWORD=postgres',
                '-p',
                '127.0.0.1:5432:5432',
                '--health-cmd',
                'pg_isready -U postgres',
                '--health-interval',
                '5s',
                '--health-timeout',
                '5s',
                '--health-retries',
                '10',
                'postgres:16',
            )
        else:
            recorder.run('sudo', 'systemctl', 'start', 'postgresql.service')
            recorder.run(
                'sudo',
                '-u',
                'postgres',
                'psql',
                '-X',
                '-v',
                'ON_ERROR_STOP=1',
                '-c',
                "ALTER ROLE postgres PASSWORD 'postgres'",
            )
        deadline = time.monotonic() + 60
        while True:
            ready = recorder.sql('SELECT 1', check=False)
            if ready.returncode == 0 and ready.stdout.strip() == '1':
                break
            if time.monotonic() >= deadline:
                raise TimeoutError('Authenticated PostgreSQL readiness exceeded 60 seconds')
            time.sleep(0.25)
        trial.ready_seconds = time.monotonic() - startup
        if cell.strategy == 'docker':
            while (
                recorder.run(
                    'docker', 'inspect', '--format', '{{.State.Health.Status}}', 'postgres-pilot'
                ).stdout.strip()
                != 'healthy'
            ):
                if time.monotonic() >= deadline:
                    raise TimeoutError('Docker health check exceeded 60 seconds')
                time.sleep(0.25)
            trial.health_seconds = time.monotonic() - startup
            trial.image_digest = TypeAdapter(list[str]).validate_json(
                recorder.run('docker', 'inspect', '--format', '{{json .RepoDigests}}', 'postgres:16').stdout
            )
            raw = recorder.run('docker', 'buildx', 'imagetools', 'inspect', '--raw', trial.image_digest[0]).stdout
            (directory / 'image-index.json').write_text(raw)
            index = ImageIndex.model_validate_json(raw)
            descriptor = next(
                item
                for item in index.manifests
                if item.platform.os == 'linux' and item.platform.architecture == 'amd64'
            )
            manifest = ImageManifest.model_validate_json(
                recorder.run(
                    'docker', 'buildx', 'imagetools', 'inspect', '--raw', f'postgres@{descriptor.digest}'
                ).stdout
            )
            trial.compressed_layer_bytes_estimate = sum(layer.size for layer in manifest.layers)
            (directory / 'image-manifest.json').write_text(manifest.model_dump_json(indent=2))
            recorder.run('docker', 'inspect', 'postgres-pilot')
        else:
            recorder.run('pg_isready', '-h', '127.0.0.1', '-U', 'postgres')
            trial.health_seconds = time.monotonic() - startup
            recorder.run('pg_lsclusters')
            recorder.run('sudo', 'systemctl', 'status', 'postgresql.service')
        assert recorder.sql('SELECT 1', password='deliberately-wrong', check=False).returncode != 0, (
            'Host auth trusts passwords'
        )
        trial.server = Server.model_validate_json(
            recorder.sql("""
            SELECT json_build_object(
                'version', version(), 'version_num', current_setting('server_version_num')::int,
                'encoding', current_setting('server_encoding'), 'collate', datcollate,
                'ctype', datctype, 'timezone', current_setting('TimeZone'),
                'ordering', (SELECT json_agg(v ORDER BY v) FROM
                    (VALUES ('a'), ('A'), ('z'), ('Z'), ('é'), ('😀')) AS sample(v)))
            FROM pg_database WHERE datname = current_database()
        """).stdout
        )
        assert trial.server.version_num // 10000 == 16, trial.server.version
        assert trial.server.encoding == 'UTF8', trial.server.encoding
        recorder.sql("""
            CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
            CREATE SCHEMA postgres_pilot;
            CREATE TABLE postgres_pilot.probe (id int PRIMARY KEY, value text, counter int);
            INSERT INTO postgres_pilot.probe VALUES (1, 'café 😀', 0);
        """)
        recorder.sql("""
            BEGIN;
            INSERT INTO postgres_pilot.probe VALUES (2, 'rolled back', 0);
            ROLLBACK;
        """)
        assert recorder.sql('SELECT count(*) FROM postgres_pilot.probe').stdout.strip() == '1'
        assert recorder.sql('SELECT value FROM postgres_pilot.probe').stdout.strip() == 'café 😀'
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    recorder.sql,
                    """
                    BEGIN;
                    UPDATE postgres_pilot.probe SET counter = counter + 1 WHERE id = 1;
                    SELECT pg_sleep(0.1);
                    COMMIT;
                """,
                )
                for _ in range(2)
            ]
            for future in futures:
                future.result()
        assert recorder.sql('SELECT counter FROM postgres_pilot.probe').stdout.strip() == '2'
        recorder.sql('DROP SCHEMA postgres_pilot CASCADE')
        assert recorder.sql("SELECT count(*) FROM pg_namespace WHERE nspname = 'postgres_pilot'").stdout.strip() == '0'
        trial.probes_passed = True
    except (OSError, ValueError, AssertionError, subprocess.SubprocessError, TimeoutError) as exc:
        trial.error = repr(exc)
        raise
    finally:
        path.write_text(trial.model_dump_json(indent=2))


def finish(directory: Path, test_exit: int, smoke_exit: int) -> None:
    """Retain post-test schemas and stop the fresh server, even after a failed test."""
    path = directory / 'trial.json'
    if not path.exists():
        return
    trial = Trial.model_validate_json(path.read_text())
    recorder = Recorder(directory)
    trial.test_exit = test_exit
    trial.smoke_exit = smoke_exit
    smoke = directory / 'missing-url.xml'
    if smoke.exists():
        trial.smoke_skips = sorted(
            f'{case.attrib.get("classname")}::{case.attrib.get("name")}:{skip.attrib.get("message")}'
            for case in ET.parse(smoke).iter('testcase')
            for skip in case.findall('skipped')
        )
    packages = recorder.run('dpkg-query', '-W', '-f=$' + '{Package}\t$' + '{Version}\n', 'postgresql*').stdout
    (directory / 'packages-after.txt').write_text(packages)
    trial.packages_unchanged = packages == (directory / 'packages-before.txt').read_text()
    recorder.sql(
        'SELECT schemaname, tablename FROM pg_tables WHERE schemaname NOT IN '
        "('pg_catalog', 'information_schema') ORDER BY 1, 2",
        check=False,
    )
    if trial.cell.strategy == 'docker':
        recorder.run('docker', 'logs', 'postgres-pilot', check=False)
        recorder.run('docker', 'rm', '--force', '--volumes', 'postgres-pilot', check=False)
    else:
        recorder.run('sudo', 'systemctl', 'stop', 'postgresql.service', check=False)
    stopped = recorder.run('pg_isready', '-h', '127.0.0.1', check=False)
    if stopped.returncode == 0:
        trial.error = 'PostgreSQL still reachable after cleanup'
    path.write_text(trial.model_dump_json(indent=2))


class Evidence(BaseModel):
    """A trial together with its exact pytest evidence."""

    trial: Trial
    collection: list[str]
    events: list[Event]


class Comparison(BaseModel):
    """Adoption gates and descriptive batch statistics."""

    gates: dict[str, bool]
    median_seconds: dict[str, float]
    p95_seconds: dict[str, float]
    pairs: dict[str, bool]


def compare(evidence: list[Evidence]) -> Comparison:
    """Fail adoption gates on missing trials, changed IDs, skips, outcomes or database semantics."""
    expected = {cell.name for cell in cells()}
    names = [item.trial.cell.name for item in evidence]
    gates: dict[str, bool] = {
        'complete_unique_matrix': set(names) == expected and len(names) == len(expected),
        'successful_tests_and_probes': bool(evidence)
        and all(
            item.trial.test_exit == 0
            and item.trial.smoke_exit == 0
            and (
                len(item.trial.smoke_skips) == 2
                if item.trial.cell.python == '3.14' and item.trial.cell.shard == 1
                else True
            )
            and item.trial.probes_passed
            and item.trial.ready_seconds is not None
            and item.trial.error is None
            and bool(item.collection)
            and bool(item.events)
            for item in evidence
        ),
        'zero_native_postgresql_downloads': bool(evidence)
        and all(
            item.trial.packages_unchanged
            and item.trial.pull_seconds == 0
            and item.trial.pull_interface_rx_bytes == 0
            and not item.trial.image_digest
            for item in evidence
            if item.trial.cell.strategy == 'native'
        ),
        'same_docker_image_revision': (
            len({tuple(item.trial.image_digest) for item in evidence if item.trial.cell.strategy == 'docker'}) == 1
            and all(item.trial.image_digest for item in evidence if item.trial.cell.strategy == 'docker')
        ),
        'same_source_and_runner_image': bool(evidence)
        and len(
            {
                (
                    item.trial.source,
                    item.trial.runner.get('ImageOS'),
                    item.trial.runner.get('ImageVersion'),
                    item.trial.runner.get('RUNNER_OS'),
                    item.trial.runner.get('RUNNER_ARCH'),
                )
                for item in evidence
            }
        )
        == 1,
    }
    pairs: dict[str, bool] = {}
    by_name = {item.trial.cell.name: item for item in evidence}
    for cell in cells():
        if cell.strategy != 'docker':
            continue
        docker = by_name.get(cell.name)
        native = by_name.get(f'native-{cell.key}')
        if docker is None or native is None:
            pairs[cell.key] = False
            continue
        docker_server = docker.trial.server
        native_server = native.trial.server
        pairs[cell.key] = (
            docker.collection == native.collection
            and Counter(event.model_dump_json() for event in docker.events)
            == Counter(event.model_dump_json() for event in native.events)
            and docker.trial.hashes == native.trial.hashes
            and docker.trial.smoke_skips == native.trial.smoke_skips
        )
        gates[f'database_semantics_{cell.key}'] = (
            docker_server is not None
            and native_server is not None
            and docker_server.encoding == native_server.encoding
            and docker_server.collate == native_server.collate
            and docker_server.ctype == native_server.ctype
            and docker_server.timezone == native_server.timezone
            and docker_server.ordering == native_server.ordering
        )
    gates['same_test_ids_outcomes_and_skips'] = all(pairs.values())
    for version in ('3.11', '3.12', '3.13', '3.14', '3.15'):
        for strategy in ('docker', 'native'):
            full: set[str] = set()
            harness: set[str] = set()
            for item in evidence:
                if item.trial.cell.python != version or item.trial.cell.strategy != strategy:
                    continue
                target = full if item.trial.cell.selection == 'full' else harness
                target.update(
                    node
                    for node in item.collection
                    if node.startswith(('tests/harness/absurd/', 'tests/harness/pgtask/'))
                )
            gates[f'postgresql_shard_coverage_{strategy}_{version}'] = full == harness and all(
                any(node.startswith(f'tests/harness/{integration}/') for node in full)
                for integration in ('absurd', 'pgtask')
            )
    medians: dict[str, float] = {}
    p95: dict[str, float] = {}
    for strategy in ('docker', 'native'):
        values = sorted(
            item.trial.ready_seconds
            for item in evidence
            if item.trial.cell.strategy == strategy and item.trial.ready_seconds is not None
        )
        gates[f'at_least_ten_ready_{strategy}'] = len(values) >= 10
        if values:
            medians[strategy] = statistics.median(values)
            p95[strategy] = values[math.ceil(len(values) * 0.95) - 1]
    gates['median_no_slower'] = 'native' in medians and 'docker' in medians and medians['native'] <= medians['docker']
    gates['p95_no_slower'] = 'native' in p95 and 'docker' in p95 and p95['native'] <= p95['docker']
    return Comparison(gates=gates, median_seconds=medians, p95_seconds=p95, pairs=pairs)


def summarize(root: Path, output: Path) -> None:
    """Publish every trial, including failures, rather than dropping unfavorable samples."""
    evidence: list[Evidence] = []
    for path in sorted(root.glob('*/trial.json')):
        trial = Trial.model_validate_json(path.read_text())
        directory = path.parent
        collection_path = directory / 'collection.json'
        events_path = directory / 'events.jsonl'
        evidence.append(
            Evidence(
                trial=trial,
                collection=TypeAdapter(list[str]).validate_json(collection_path.read_text())
                if collection_path.exists()
                else [],
                events=[Event.model_validate_json(line) for line in events_path.read_text().splitlines()]
                if events_path.exists()
                else [],
            )
        )
    comparison = compare(evidence)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'comparison.json').write_text(comparison.model_dump_json(indent=2))
    (output / 'trials.json').write_text(TypeAdapter(list[Evidence]).dump_json(evidence, indent=2).decode())
    artifact_data = subprocess.run(
        [
            'gh',
            'api',
            f'repos/{os.environ["GITHUB_REPOSITORY"]}/actions/runs/{os.environ["GITHUB_RUN_ID"]}/artifacts?per_page=100',
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    artifacts = ArtifactList.model_validate_json(artifact_data.stdout)
    links = {
        item.name: f'https://github.com/{os.environ["GITHUB_REPOSITORY"]}/actions/runs/{os.environ["GITHUB_RUN_ID"]}/artifacts/{item.id}'
        for item in artifacts.artifacts
    }
    lines = [
        '# PostgreSQL pilot',
        '',
        f'Adoption gates: **{"PASS" if all(comparison.gates.values()) else "FAIL — retain production Docker service"}**.',
        '',
        'Workflow success means evidence was collected. Adoption requires every gate below to pass.',
        '',
        '| Strategy | Median authenticated-ready seconds | Nearest-rank p95 seconds |',
        '|---|---:|---:|',
        *[
            f'| {strategy} | {comparison.median_seconds.get(strategy, 0):.3f} | {comparison.p95_seconds.get(strategy, 0):.3f} |'
            for strategy in ['docker', 'native']
        ],
        '',
        '| Trial | Runner image | Server | Ready s | Pull s | Pull interface RX bytes | Tests |',
        '|---|---|---|---:|---:|---:|---|',
    ]
    for item in evidence:
        trial = item.trial
        server = trial.server.version if trial.server else 'startup failed'
        lines.append(
            f'| [{trial.cell.name}]({links.get("postgres-pilot-" + trial.cell.name, "")}) | {trial.runner.get("ImageVersion", "")} | {server} | '
            f'{trial.ready_seconds} | {trial.pull_seconds:.3f} | {trial.pull_interface_rx_bytes} | {trial.test_exit} |'
        )
    lines += [
        '',
        '## Gates',
        '',
        *[f'- {"PASS" if passed else "FAIL"}: {gate}' for gate, passed in comparison.gates.items()],
    ]
    lines += [
        '',
        '## Raw evidence',
        '',
        f'All trial artifacts and job logs: https://github.com/{os.environ["GITHUB_REPOSITORY"]}/actions/runs/{os.environ["GITHUB_RUN_ID"]}',
        '',
        'Each trial artifact contains commands with stdout/stderr/timing, runner and server versions, package inventories,',
        'selected node IDs, test reports, and test output. Interface RX during pull is measured ingress including background',
        'traffic; it is not an image-only or billing counter. Image manifests are separate registry metadata.',
        '',
    ]
    (output / 'summary.md').write_text('\n'.join(lines))
    print('\n'.join(lines))
    if summary := os.environ.get('GITHUB_STEP_SUMMARY'):
        Path(summary).write_text('\n'.join(lines), encoding='utf-8')
    assert comparison.gates['complete_unique_matrix'], 'Incomplete evidence; inspect failed jobs'


def main() -> None:
    """Run one experiment stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['matrix', 'start', 'finish', 'summarize'])
    parser.add_argument('--directory', type=Path, default=Path('evidence'))
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--cell')
    parser.add_argument('--test-exit', type=int, default=1)
    parser.add_argument('--smoke-exit', type=int, default=1)
    parser.add_argument('--output', type=Path, default=Path('comparison'))
    args = parser.parse_args()
    if args.command == 'matrix':
        print(json.dumps({'include': [cell.model_dump() for cell in cells()]}))
    elif args.command == 'start':
        assert isinstance(args.cell, str) and isinstance(args.baseline, Path)
        start(Cell.model_validate_json(args.cell), args.directory, args.baseline)
    elif args.command == 'finish':
        finish(args.directory, args.test_exit, args.smoke_exit)
    else:
        summarize(args.directory, args.output)


if __name__ == '__main__':
    main()
