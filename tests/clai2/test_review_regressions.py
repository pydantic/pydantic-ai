"""Review regressions for plugin import and menu ownership."""

import asyncio
import io
import os
from pathlib import Path

import pytest
from pydantic_ai import Agent, PartDeltaEvent, PartStartEvent, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import TextPart, TextPartDelta, ThinkingPart
from pydantic_ai.models.test import TestModel
from rich.console import Console
from test_plugin_loader import Harness

from pydantic_clai2 import StreamRenderer
from pydantic_clai2.command_context import CommandContext
from pydantic_clai2.config import PluginSettings, Settings
from pydantic_clai2.menu_worker import menu_key, run_worker
from pydantic_clai2.model_settings import ModelSettingsForm
from pydantic_clai2.plugin_loader import PluginError
from pydantic_clai2.plugins import PluginHost
from pydantic_clai2.settings_store import SettingsStore


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


async def test_package_relative_import_and_fresh_source(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    package = harness.store.plugins_dir / 'package'
    package.mkdir()
    (package / 'helper.py').write_text('VALUE = 1')
    source = package / '__init__.py'
    source.write_text('from .helper import VALUE\ndef activate(host): host.console.print(VALUE)')
    stamp = source.stat().st_mtime
    await harness.loader.load_all()
    assert '1' in harness.text
    source.write_text('from .helper import VALUE\ndef activate(host): host.console.print(22222)')
    os.utime(source, (stamp, stamp))
    await harness.loader.reload('package')
    assert '22222' in harness.text
    await harness.loader.disable('package')
    with pytest.raises(ValueError, match='disabled'):
        await harness.loader.reload('package')


async def test_cancelled_plugin_start_rolls_back(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    path = harness.write('cancelled')
    path.write_text(
        path.read_text().replace(
            "host.console.print('cancelled started')", 'raise __import__("asyncio").CancelledError()'
        )
    )
    with pytest.raises(asyncio.CancelledError):
        await harness.loader.load('cancelled')
    assert harness.loader.entries()[0].host is None
    assert 'cancelled' not in await harness.commands.execute_async('/help')


async def test_missing_dropin_and_unreadable_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    harness = Harness(tmp_path)
    path = harness.write('gone')
    await harness.loader.disable('gone')
    path.unlink()
    with pytest.raises(PluginError):
        await harness.loader.enable('gone')

    def denied(path: Path) -> list[Path]:
        raise PermissionError('denied')

    monkeypatch.setattr(Path, 'iterdir', denied)
    assert harness.loader.entries()
    assert 'Cannot discover plugins' in harness.text


async def test_explicit_source_wins(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    harness.write('collision')
    harness.store.save_plugin(PluginSettings(id='collision', factory='pydantic_ai.capabilities:AbstractCapability'))
    harness.store.save_plugin(PluginSettings(id='old-id', factory='pydantic_ai.capabilities:AbstractCapability'))
    await harness.loader.load_all()
    assert len(harness.loader.capabilities()) == 2
    with pytest.raises(ValueError, match='already exists'):
        await harness.loader.command(['add', 'collision', 'missing.module'])
    assert 'collision started' not in harness.text


async def test_worker_cancellation_joins_before_return(monkeypatch: pytest.MonkeyPatch) -> None:
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    finished = asyncio.Event()

    def key(*, timeout: float) -> str:
        loop.call_soon_threadsafe(entered.set)
        return ''

    monkeypatch.setattr('pydantic_clai2.menu_worker.read_key', key)
    assert menu_key() == ''

    def worker() -> None:
        while menu_key() != 'ctrl-c':
            pass
        loop.call_soon_threadsafe(finished.set)

    task = asyncio.create_task(run_worker(worker))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished.is_set()


@pytest.mark.parametrize('thinking', [False, True])
async def test_intercepted_start_keeps_streaming(thinking: bool) -> None:
    output = io.StringIO()
    console = Console(file=output)
    host: PluginHost[None] = PluginHost(console=console, name='test', settings={})

    @host.render(PartStartEvent)
    def replace(event: PartStartEvent) -> str:
        return 'replacement'

    renderer = StreamRenderer(console, stop_loading=lambda: None, renderers=host.renderers)
    await renderer.on_stream_event(
        PartStartEvent(index=0, part=ThinkingPart('original') if thinking else TextPart('original'))
    )
    await renderer.on_stream_event(PartDeltaEvent(index=0, delta=TextPartDelta(' subsequent')))
    await renderer.finish()
    assert 'subsequent' in output.getvalue()
    assert renderer.rendered_text == (not thinking)


def test_reset_preserves_other_runtime_overrides(tmp_path: Path) -> None:
    context = CommandContext(
        settings=Settings(model='test', request_limit=999),
        store=SettingsStore(tmp_path / 'config.db'),
        clear_history=lambda: None,
        apply_setting=lambda key, settings: None,
    )
    context.reset_setting('display.thinking')
    assert context.settings.model == 'test'
    assert context.settings.request_limit == 999


async def test_capability_factory_is_resolved_by_core() -> None:
    host: PluginHost[None] = PluginHost(console=Console(file=io.StringIO()), name='test', settings={})
    called: list[bool] = []

    def factory(ctx: RunContext[None]) -> AbstractCapability[None] | None:
        called.append(True)
        return None

    host.add(factory)
    await Agent(TestModel(), deps_type=type(None)).run('hi', capabilities=host.capabilities)
    assert called == [True]


def test_integral_float_settings() -> None:
    assert ModelSettingsForm.model_validate({'temperature': 0, 'top_p': 1, 'timeout': 1}).temperature == 0.0
