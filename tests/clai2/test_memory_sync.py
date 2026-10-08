"""Hackathon: personal notebooks synced through Logfire, so a user can switch machines without losing notes.

The query API is replaced by `SpanQuery`, which answers `NotebookSync`'s query from the spans the test exported,
so two "machines" (two folders) meet through the same spans as they would through Logfire.
"""

from __future__ import annotations

import os
import re
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from pydantic_clai2.builtin_plugins.fleet_memory import PendingNotes
from pydantic_clai2.builtin_plugins.logfire_sessions import WINDOW, LogfireQuery
from pydantic_clai2.builtin_plugins.memory_command import MemoryCommand
from pydantic_clai2.builtin_plugins.memory_sync import SPAN_NAME, NotebookSync, SyncState

pytestmark = pytest.mark.anyio

OWNER = 'alice@example.com'
NOTE = 'repos/acme/app/personal/MEMORY.md'


class SpanQuery(LogfireQuery):
    """Answers the note query from exported spans, as Logfire's `records` table would."""

    def __init__(self, exporter: InMemorySpanExporter, *, down: bool = False) -> None:
        super().__init__(base_url='https://logfire.example', key='test')
        self.exporter = exporter
        self.down = down

    async def rows(self, sql: str, *, limit: int = 10_000, since: timedelta = WINDOW) -> list[dict[str, Any]]:
        if self.down:
            raise ConnectionError('Logfire is unreachable')
        owner = re.search(r"'clai2.memory.owner' = '([^']+)'", sql)
        assert owner is not None
        return [
            {key.removeprefix('clai2.memory.'): value for key, value in (span.attributes or {}).items()}
            for span in self.exporter.get_finished_spans()
            if span.name == SPAN_NAME and (span.attributes or {}).get('clai2.memory.owner') == owner.group(1)
        ][:limit]


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


def machine(tmp_path: Path, name: str, exporter: InMemorySpanExporter, *, down: bool = False) -> NotebookSync:
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    warnings: list[str] = []
    sync = NotebookSync(
        directory=tmp_path / name / 'memory',
        tracer_provider=provider,
        state=SyncState(tmp_path / name / 'memory_sync.json'),
        owner=lambda: OWNER,
        query=SpanQuery(exporter, down=down),
        warn=warnings.append,
    )
    sync.warnings = warnings  # pyright: ignore[reportAttributeAccessIssue]
    return sync


def write(sync: NotebookSync, path: str, content: str, *, at: float | None = None) -> Path:
    file = sync.directory / path
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(content)
    if at is not None:
        os.utime(file, (at, at))
    return file


def sent(exporter: InMemorySpanExporter) -> list[tuple[str, str, bool]]:
    return [
        (
            str((span.attributes or {})['clai2.memory.path']),
            str((span.attributes or {})['clai2.memory.content']),
            bool((span.attributes or {})['clai2.memory.deleted']),
        )
        for span in exporter.get_finished_spans()
        if span.name == SPAN_NAME
    ]


async def test_a_fresh_machine_restores_notes_written_on_another(tmp_path: Path, exporter: InMemorySpanExporter):
    laptop = machine(tmp_path, 'laptop', exporter)
    write(laptop, NOTE, '- Use uv, not pip.')
    write(laptop, 'global/personal/style.md', '- British English.')
    assert laptop.push_changed() == 2
    assert laptop.push_changed() == 0

    desktop = machine(tmp_path, 'desktop', exporter)
    await desktop.pull()

    assert (desktop.directory / NOTE).read_text() == '- Use uv, not pip.'
    assert (desktop.directory / 'global/personal/style.md').read_text() == '- British English.'
    # Restored notes match Logfire, so nothing is sent back.
    assert len(sent(exporter)) == 2
    assert desktop.push_changed() == 0


async def test_a_newer_local_note_wins_and_is_sent_again(tmp_path: Path, exporter: InMemorySpanExporter):
    laptop = machine(tmp_path, 'laptop', exporter)
    write(laptop, NOTE, 'old', at=time.time() - 3600)
    laptop.push_changed()

    desktop = machine(tmp_path, 'desktop', exporter)
    write(desktop, NOTE, 'newer, written offline')
    await desktop.pull()

    assert (desktop.directory / NOTE).read_text() == 'newer, written offline'
    assert sent(exporter)[-1] == (NOTE, 'newer, written offline', False)

    await laptop.pull()
    assert (laptop.directory / NOTE).read_text() == 'newer, written offline'


async def test_a_deletion_elsewhere_deletes_the_note_here(tmp_path: Path, exporter: InMemorySpanExporter):
    laptop = machine(tmp_path, 'laptop', exporter)
    write(laptop, NOTE, '- Stale advice.', at=time.time() - 3600)
    laptop.push_changed()
    desktop = machine(tmp_path, 'desktop', exporter)
    await desktop.pull()

    (desktop.directory / NOTE).unlink()
    assert desktop.push_changed() == 1
    assert sent(exporter)[-1] == (NOTE, '', True)

    await laptop.pull()
    assert not (laptop.directory / NOTE).exists()
    # The tombstone is now in step here too: nothing left to send, and a later pull leaves it deleted.
    assert laptop.push_changed() == 0
    await laptop.pull()
    assert not (laptop.directory / NOTE).exists()


async def test_logfire_unavailable_keeps_notes_local_with_one_warning(tmp_path: Path, exporter: InMemorySpanExporter):
    sync = machine(tmp_path, 'laptop', exporter, down=True)
    write(sync, NOTE, '- Still here.')

    await sync.pull()
    await sync.pull()

    assert (sync.directory / NOTE).read_text() == '- Still here.'
    assert sync.warnings == [  # pyright: ignore[reportAttributeAccessIssue]
        'Personal notes not synced from Logfire (ConnectionError); keeping the notes on this machine.'
    ]
    # Writing still works, so the note reaches Logfire for the next machine.
    assert sent(exporter) == [(NOTE, '- Still here.', False)]


async def test_notes_outside_personal_notebooks_and_bookkeeping_are_not_synced(
    tmp_path: Path, exporter: InMemorySpanExporter
):
    sync = machine(tmp_path, 'laptop', exporter)
    write(sync, 'pending.json', '[]')
    write(sync, 'repos/acme/app/personal/.memory-operations.json', '[]')
    write(sync, 'repos/acme/app/notes.md', 'not a notebook')
    assert sync.push_changed() == 0


async def test_a_remote_path_cannot_escape_the_notebook_folder(tmp_path: Path, exporter: InMemorySpanExporter):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    provider.get_tracer('test').start_span(
        SPAN_NAME,
        attributes={
            'clai2.memory.owner': OWNER,
            'clai2.memory.path': '../../outside/personal/evil.md',
            'clai2.memory.content': 'x',
            'clai2.memory.deleted': False,
            'clai2.memory.sha256': 'x',
            'clai2.memory.at': time.time(),
        },
    ).end()
    sync = machine(tmp_path, 'laptop', exporter)
    await sync.pull()
    assert not (tmp_path / 'outside').exists()


def test_memory_says_personal_notes_are_synced(tmp_path: Path):
    command = MemoryCommand(
        directory=tmp_path,
        repo=lambda: 'acme/app',
        notes=lambda: [],
        mode=lambda: 'review',
        pending=PendingNotes(tmp_path / 'pending.json'),
        propose=lambda path, content, why: '',
        withdraw=lambda note: None,
        edit=lambda text, title: None,  # pyright: ignore[reportArgumentType]
        synced=True,
    )
    assert 'Personal, acme/app (synced via Logfire; /memory edit):' in command.overview()
