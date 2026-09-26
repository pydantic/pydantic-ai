"""Default Logfire tracing stays scoped to the plugin and uses no real exporter."""

import base64
import io
import json
from collections.abc import Generator
from pathlib import Path
from typing import Literal

import anyio
import logfire
import pytest
from opentelemetry import metrics, propagate, trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import JsonValue, ValidationError
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.test import TestModel
from rich.console import Console

from pydantic_clai2 import DEFAULT_PLUGINS
from pydantic_clai2.commands import Commands
from pydantic_clai2.logfire import activate
from pydantic_clai2.plugin_loader import PluginError, PluginLoader
from pydantic_clai2.plugins import PluginHost, SessionEnd, SessionStart
from pydantic_clai2.settings_store import SettingsStore


class Exporter(InMemorySpanExporter):
    closed = False

    def shutdown(self) -> None:
        self.closed = True
        super().shutdown()


class Recorder:
    def __init__(self) -> None:
        self.configure_sdk = logfire.configure
        self.exporters: list[Exporter] = []
        self.instances: list[logfire.Logfire] = []
        self.options: list[dict[str, object]] = []

    def configure(
        self,
        *,
        local: bool,
        send_to_logfire: Literal[False, 'if-token-present'],
        service_name: str,
        console: Literal[False],
        config_dir: Path,
        data_dir: Path,
    ) -> logfire.Logfire:
        self.options.append(
            {
                'local': local,
                'send_to_logfire': send_to_logfire,
                'service_name': service_name,
                'console': console,
                'config_dir': config_dir,
                'data_dir': data_dir,
            }
        )
        exporter = Exporter()
        instance = self.configure_sdk(
            local=local,
            send_to_logfire=False,
            service_name=service_name,
            console=console,
            config_dir=config_dir,
            data_dir=data_dir,
            metrics=False,
            additional_span_processors=[SimpleSpanProcessor(exporter)],
            advanced=logfire.AdvancedOptions(emit_configuration_span=False),
        )
        self.exporters.append(exporter)
        self.instances.append(instance)
        return instance

    def spans(self) -> list[ReadableSpan]:
        return [span for exporter in self.exporters for span in exporter.get_finished_spans()]


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> Generator[Recorder]:
    recorded = Recorder()
    monkeypatch.setattr('pydantic_clai2.logfire.logfire.configure', recorded.configure)
    try:
        yield recorded
    finally:
        for instance in recorded.instances:
            instance.shutdown(timeout_millis=3000)


def make_host(**settings: JsonValue) -> PluginHost[None]:
    return PluginHost(name='logfire', console=Console(file=io.StringIO()), settings=settings)


async def close_host(host: PluginHost[None]) -> None:
    for handler in host.handlers:
        await handler(SessionEnd(reason='exit'))


def operation(span: ReadableSpan) -> object:
    return (span.attributes or {}).get('gen_ai.operation.name')


async def test_default_content_images_tools_and_usage_are_traced(recorder: Recorder, tmp_path: Path) -> None:
    tracer, meter = trace.get_tracer_provider(), metrics.get_meter_provider()
    propagator = propagate.get_global_textmap()
    host = make_host()
    activate(host)
    agent = Agent(TestModel(custom_output_text='The image shows a button'), deps_type=type(None), name='clai_test')

    @agent.tool_plain
    def inspect_button() -> str:
        return 'button details'

    image = BinaryContent(data=b'example image bytes', media_type='image/png')
    try:
        result = await agent.run(['describe the screenshot', image], capabilities=host.capabilities)
        assert result.output == 'The image shows a button'
    finally:
        await close_host(host)
    spans = recorder.spans()
    assert [operation(span) for span in spans].count('invoke_agent') == 1
    assert [operation(span) for span in spans].count('execute_tool') == 1
    assert [operation(span) for span in spans].count('chat') == 2
    run_span = next(span for span in spans if operation(span) == 'invoke_agent')
    run_context = run_span.context
    assert run_context is not None
    assert all(span.context is not None and span.context.trace_id == run_context.trace_id for span in spans)
    serialized = json.dumps([dict(span.attributes or {}) for span in spans])
    assert 'describe the screenshot' in serialized
    assert 'The image shows a button' in serialized
    assert 'button details' in serialized
    assert base64.b64encode(image.data).decode() in serialized
    assert 'gen_ai.usage.input_tokens' in serialized
    assert recorder.options == [
        {
            'local': True,
            'send_to_logfire': 'if-token-present',
            'service_name': 'pydantic-clai2',
            'console': False,
            'config_dir': tmp_path / 'config/pydantic-clai2/logfire',
            'data_dir': tmp_path / 'config/pydantic-clai2/logfire',
        }
    ]
    assert all(exporter.closed for exporter in recorder.exporters)
    assert trace.get_tracer_provider() is tracer
    assert metrics.get_meter_provider() is meter
    assert propagate.get_global_textmap() is propagator
    assert agent.instrument is None


@pytest.mark.parametrize('content', [False, True])
async def test_content_settings(recorder: Recorder, content: bool) -> None:
    host = make_host(
        include_content=content, include_binary_content=False, service_name='custom-clai', send_to_logfire=False
    )
    activate(host)
    try:
        await Agent(TestModel(custom_output_text='plain answer'), deps_type=type(None)).run(
            ['visible caption', BinaryContent(data=b'image bytes', media_type='image/png')],
            capabilities=host.capabilities,
        )
    finally:
        await close_host(host)
    spans = recorder.spans()
    serialized = json.dumps([dict(span.attributes or {}) for span in spans])
    assert ('visible caption' in serialized) == content
    assert ('plain answer' in serialized) == content
    assert base64.b64encode(b'image bytes').decode() not in serialized
    assert spans[0].resource.attributes['service.name'] == 'custom-clai'
    assert recorder.options[0]['send_to_logfire'] is False


@pytest.mark.parametrize('explicit', [False, True])
async def test_disable_reload_and_existing_agent_instrumentation(
    tmp_path: Path, recorder: Recorder, explicit: bool
) -> None:
    existing_exporter = Exporter()
    existing_provider = TracerProvider(shutdown_on_exit=False)
    existing_provider.add_span_processor(SimpleSpanProcessor(existing_exporter))
    original = InstrumentationSettings(tracer_provider=existing_provider)
    agent = Agent(
        TestModel(custom_output_text='done'),
        deps_type=type(None),
        capabilities=[Instrumentation(settings=original)] if explicit else (),
    )
    if not explicit:
        agent.instrument = original
    store = SettingsStore(tmp_path / 'config.db')
    builtin = next(plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'logfire')
    assert builtin.enabled
    loader: PluginLoader[None] = PluginLoader(
        store=store,
        console=Console(file=io.StringIO()),
        commands=Commands(),
        session_start=lambda: SessionStart(agent=agent, settings=store.load()),
        builtin=(builtin,),
    )
    try:
        await loader.load_all()
        assert loader.entries()[0].error is None
        await agent.run('first', capabilities=loader.capabilities())
        assert len([span for span in recorder.spans() if operation(span) == 'invoke_agent']) == 1
        assert not existing_exporter.get_finished_spans()
        await loader.disable('logfire')
        assert not loader.capabilities()
        assert recorder.exporters[0].closed
        before = len(recorder.spans())
        await agent.run('disabled', capabilities=loader.capabilities())
        assert len(recorder.spans()) == before
        assert len([span for span in existing_exporter.get_finished_spans() if operation(span) == 'invoke_agent']) == 1
        await loader.enable('logfire')
        await loader.reload('logfire')
        assert recorder.exporters[1].closed
        await agent.run('reloaded', capabilities=loader.capabilities())
        assert len([span for span in recorder.spans() if operation(span) == 'invoke_agent']) == 2
        assert agent.instrument is (None if explicit else original)
    finally:
        await loader.close('exit')
        existing_provider.shutdown()
    assert all(exporter.closed for exporter in recorder.exporters)


@pytest.mark.parametrize(
    'settings', [{'include_content': 'false'}, {'token': 'do-not-store'}, {'send_to_logfire': True}]
)
def test_settings_validate_before_configuring(recorder: Recorder, settings: dict[str, JsonValue]) -> None:
    with pytest.raises(ValidationError) as error:
        activate(make_host(**settings))
    assert 'do-not-store' not in str(error.value)
    assert not recorder.instances


async def test_unload_finishes_inside_outer_cancellation(recorder: Recorder) -> None:
    host = make_host()
    activate(host)
    with anyio.CancelScope() as scope:
        scope.cancel()
        await close_host(host)
        assert recorder.exporters[0].closed
        await anyio.sleep(0)
    assert scope.cancelled_caught


async def test_shutdown_timeout_is_reported(recorder: Recorder, monkeypatch: pytest.MonkeyPatch) -> None:
    host = make_host()
    activate(host)
    instance = recorder.instances[0]
    shutdown = instance.shutdown

    def timeout(*, timeout_millis: int, flush: bool = True) -> bool:
        shutdown(timeout_millis=timeout_millis, flush=flush)
        return False

    monkeypatch.setattr(instance, 'shutdown', timeout)
    await close_host(host)
    output = host.console.file
    assert isinstance(output, io.StringIO)
    assert 'Logfire shutdown timed out' in output.getvalue()


def test_activation_failure_closes_local_providers(recorder: Recorder, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*, settings: InstrumentationSettings) -> Instrumentation:
        raise RuntimeError('cannot construct instrumentation')

    monkeypatch.setattr('pydantic_clai2.logfire.Instrumentation', fail)
    with pytest.raises(RuntimeError, match='cannot construct'):
        activate(make_host())
    assert recorder.exporters[0].closed


async def test_no_credentials_needs_no_setup_or_console_output(capsys: pytest.CaptureFixture[str]) -> None:
    host = make_host()
    activate(host)
    try:
        await Agent(TestModel(), deps_type=type(None)).run('hello', capabilities=host.capabilities)
    finally:
        await close_host(host)
    captured = capsys.readouterr()
    assert captured.out == captured.err == ''


@pytest.mark.parametrize('cancelled', [False, True])
async def test_failed_and_cancelled_runs_finish_their_spans(recorder: Recorder, cancelled: bool) -> None:
    host = make_host()
    activate(host)
    started = anyio.Event()
    cleaned = anyio.Event()
    agent = Agent(TestModel(), deps_type=type(None), name='failure_test')

    @agent.tool_plain
    async def work() -> str:
        try:
            started.set()
            if cancelled:
                await anyio.sleep_forever()
            raise RuntimeError('tool failure')
        finally:
            cleaned.set()

    async def run() -> None:
        await agent.run('run the tool', capabilities=host.capabilities)

    try:
        if cancelled:
            with anyio.fail_after(10):
                async with anyio.create_task_group() as tasks:
                    tasks.start_soon(run)
                    await started.wait()
                    tasks.cancel_scope.cancel()
        else:
            with pytest.raises(RuntimeError, match='tool failure'):
                await run()
        assert cleaned.is_set()
    finally:
        await close_host(host)
    spans = recorder.spans()
    assert any(operation(span) == 'invoke_agent' for span in spans)
    assert any(operation(span) == 'execute_tool' for span in spans)
    assert all(span.end_time is not None for span in spans)
    if not cancelled:
        assert any(span.status.status_code is trace.StatusCode.ERROR for span in spans)


async def test_flush_timeout_still_stops_providers(recorder: Recorder, monkeypatch: pytest.MonkeyPatch) -> None:
    host = make_host()
    activate(host)

    def timeout(*, timeout_millis: int) -> bool:
        return False

    monkeypatch.setattr(recorder.instances[0], 'force_flush', timeout)
    await close_host(host)
    assert recorder.exporters[0].closed
    output = host.console.file
    assert isinstance(output, io.StringIO)
    assert 'Logfire shutdown timed out' in output.getvalue()


@pytest.mark.parametrize('xdg', ['', 'relative-config'])
async def test_relative_config_home_does_not_read_the_checkout(
    recorder: Recorder, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, xdg: str
) -> None:
    monkeypatch.setenv('XDG_CONFIG_HOME', xdg)
    host = make_host()
    activate(host)
    await close_host(host)
    assert recorder.options[0]['config_dir'] == tmp_path / 'home/.config/pydantic-clai2/logfire'
    assert recorder.options[0]['data_dir'] == recorder.options[0]['config_dir']


async def test_repository_cannot_select_telemetry_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recorder: Recorder
) -> None:
    repo = tmp_path / 'repo'
    repo.mkdir()
    (repo / 'pyproject.toml').write_text('[tool.logfire]\ntoken = "repository-token"\n')
    credentials = repo / '.logfire'
    credentials.mkdir()
    (credentials / 'logfire_credentials.json').write_text(
        json.dumps({'token': 'repository-token', 'project_name': 'untrusted', 'logfire_api_url': 'http://127.0.0.1:1'})
    )
    monkeypatch.chdir(repo)
    monkeypatch.setenv('LOGFIRE_CONFIG_DIR', str(repo))
    monkeypatch.setenv('LOGFIRE_CREDENTIALS_DIR', str(credentials))
    host = make_host()
    activate(host)
    await close_host(host)
    assert recorder.options[0]['config_dir'] == tmp_path / 'config/pydantic-clai2/logfire'
    assert recorder.options[0]['data_dir'] == recorder.options[0]['config_dir']
    assert recorder.instances[0].config.token is None


@pytest.mark.parametrize('cancel', [False, True])
async def test_interrupted_startup_shuts_down_plugin_providers(
    tmp_path: Path, recorder: Recorder, monkeypatch: pytest.MonkeyPatch, cancel: bool
) -> None:
    started = anyio.Event()

    def start(host: PluginHost[None]) -> None:
        activate(host)

        @host.on('session_start')
        async def wait(event: SessionStart) -> None:
            started.set()
            if cancel:
                await anyio.sleep_forever()
            raise RuntimeError('startup failed')

    monkeypatch.setattr('pydantic_clai2.logfire.activate', start)
    store = SettingsStore(tmp_path / 'config.db')
    loader: PluginLoader[None] = PluginLoader(
        store=store,
        console=Console(file=io.StringIO()),
        commands=Commands(),
        session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=store.load()),
        builtin=(next(plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'logfire'),),
    )
    if cancel:
        with anyio.fail_after(10):
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(loader.load, 'logfire')
                await started.wait()
                tasks.cancel_scope.cancel()
    else:
        with pytest.raises(PluginError, match='startup failed'):
            await loader.load('logfire')
    assert recorder.exporters[0].closed
    assert not loader.capabilities()
    assert loader.entries()[0].host is None
