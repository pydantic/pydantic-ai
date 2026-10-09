"""Default Logfire tracing stays scoped to the plugin and uses no real exporter."""

import base64
import io
import json
import warnings
from collections.abc import Generator
from pathlib import Path
from typing import Literal

import anyio
import httpx
import logfire
import pytest
from opentelemetry import metrics, propagate, trace
from opentelemetry.instrumentation.httpx import HTTPX2ClientInstrumentor, HTTPXClientInstrumentor
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import JsonValue, ValidationError
from rich.console import Console
from termflow.tui.menu import MenuResult

from pydantic_ai import Agent, BinaryContent
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.test import TestModel
from pydantic_clai2 import DEFAULT_PLUGINS
from pydantic_clai2.builtin_plugins import logfire as logfire_plugin
from pydantic_clai2.builtin_plugins.logfire import CREDENTIALS_FILE, PROJECT, LogfirePlugin, LogfireSource, logfire_dir
from pydantic_clai2.commands import Commands
from pydantic_clai2.config import Settings
from pydantic_clai2.config.api_keys import save_key
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.plugins import LoadedPlugin, PluginHost, SessionEnd, SessionStart, TurnEnd, load_plugin
from pydantic_clai2.plugins.loader import PluginError, PluginLoader
from pydantic_clai2.ui import telemetry
from tests.clai2.menu_script import Script, pick, typed


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
        self.tokens: list[str | None] = []

    def configure(
        self,
        *,
        local: bool,
        send_to_logfire: Literal[False, 'if-token-present'],
        service_name: str,
        console: Literal[False],
        inspect_arguments: bool,
        config_dir: Path,
        data_dir: Path,
        token: str | None,
        scrubbing: logfire.ScrubbingOptions | None,
        advanced: logfire.AdvancedOptions | None,
    ) -> logfire.Logfire:
        self.tokens.append(token)
        self.options.append(
            {
                'local': local,
                'send_to_logfire': send_to_logfire,
                'service_name': service_name,
                'console': console,
                'inspect_arguments': inspect_arguments,
                'config_dir': config_dir,
                'data_dir': data_dir,
                'base_url': advanced.base_url if advanced else None,
            }
        )
        exporter = Exporter()
        instance = self.configure_sdk(
            local=local,
            send_to_logfire=False,
            service_name=service_name,
            console=console,
            inspect_arguments=inspect_arguments,
            config_dir=config_dir,
            data_dir=data_dir,
            token=token,
            scrubbing=scrubbing,
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
    monkeypatch.setattr('pydantic_clai2.builtin_plugins.logfire.logfire.configure', recorded.configure)
    try:
        yield recorded
    finally:
        for instance in recorded.instances:
            instance.shutdown(timeout_millis=3000)


def make_host(**settings: JsonValue) -> PluginHost[None]:
    return PluginHost(name='observability', console=Console(file=io.StringIO()), settings=settings)


def load_logfire(host: PluginHost[None]) -> LoadedPlugin[None]:
    return load_plugin(LogfirePlugin, host)


async def close(plugin: LoadedPlugin[None]) -> None:
    await plugin.dispatch(SessionEnd(reason='exit'))


def operation(span: ReadableSpan) -> object:
    return (span.attributes or {}).get('gen_ai.operation.name')


async def test_default_content_images_tools_and_usage_are_traced(recorder: Recorder, tmp_path: Path) -> None:
    tracer, meter = trace.get_tracer_provider(), metrics.get_meter_provider()
    propagator = propagate.get_global_textmap()
    host = make_host()
    plugin = load_logfire(host)
    assert not HTTPXClientInstrumentor().is_instrumented_by_opentelemetry
    assert not HTTPX2ClientInstrumentor().is_instrumented_by_opentelemetry
    agent = Agent(TestModel(custom_output_text='The image shows a button'), deps_type=type(None), name='clai_test')

    @agent.tool_plain
    def inspect_button() -> str:
        return 'button details'

    image = BinaryContent(data=b'example image bytes', media_type='image/png')
    try:
        result = await agent.run(['describe the screenshot', image], capabilities=plugin.capabilities)
        assert result.output == 'The image shows a button'
    finally:
        await close(plugin)
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
            'inspect_arguments': False,
            'config_dir': tmp_path / 'config/pydantic-clai2/logfire',
            'data_dir': tmp_path / 'config/pydantic-clai2/logfire',
            'base_url': None,
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
    plugin = load_logfire(host)
    try:
        await Agent(TestModel(custom_output_text='plain answer'), deps_type=type(None)).run(
            ['visible caption', BinaryContent(data=b'image bytes', media_type='image/png')],
            capabilities=plugin.capabilities,
        )
    finally:
        await close(plugin)
    spans = recorder.spans()
    serialized = json.dumps([dict(span.attributes or {}) for span in spans])
    assert ('visible caption' in serialized) == content
    assert ('plain answer' in serialized) == content
    assert base64.b64encode(b'image bytes').decode() not in serialized
    assert spans[0].resource.attributes['service.name'] == 'custom-clai'
    assert recorder.options[0]['send_to_logfire'] is False


@pytest.mark.parametrize('content', [False, True])
async def test_httpx_opt_in_traces_requests_and_unloads(
    recorder: Recorder, monkeypatch: pytest.MonkeyPatch, content: bool
) -> None:
    def respond(transport: httpx.HTTPTransport, request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text='ordinary response body', request=request)

    monkeypatch.setattr(httpx.HTTPTransport, 'handle_request', respond)
    plugin = load_logfire(make_host(httpx=True, include_content=content, ui_events=False))
    await plugin.dispatch(SessionStart(agent=Agent(TestModel()), settings=Settings()))
    assert HTTPXClientInstrumentor().is_instrumented_by_opentelemetry
    assert HTTPX2ClientInstrumentor().is_instrumented_by_opentelemetry
    try:
        with httpx.Client() as client:
            with recorder.instances[-1].span('parent'):
                client.post('https://example.com/test', content='ordinary request body')
    finally:
        await close(plugin)
    assert not HTTPXClientInstrumentor().is_instrumented_by_opentelemetry
    assert not HTTPX2ClientInstrumentor().is_instrumented_by_opentelemetry
    serialized = json.dumps([dict(span.attributes or {}) for span in recorder.spans()])
    assert '"http.method": "POST"' in serialized
    assert ('ordinary request body' in serialized) is content


async def test_httpx_moves_to_remaining_plugin_on_unload(recorder: Recorder, monkeypatch: pytest.MonkeyPatch) -> None:
    def respond(transport: httpx.HTTPTransport, request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, request=request)

    monkeypatch.setattr(httpx.HTTPTransport, 'handle_request', respond)
    first = load_logfire(make_host(httpx=True, ui_events=False))
    second = load_logfire(make_host(httpx=True, ui_events=False))
    try:
        start = SessionStart(agent=Agent(TestModel()), settings=Settings())
        await first.dispatch(start)
        await second.dispatch(start)
        await close(first)
        with httpx.Client() as client:
            client.get('https://example.com/remaining')
        assert HTTPXClientInstrumentor().is_instrumented_by_opentelemetry
    finally:
        await close(second)
    assert not HTTPXClientInstrumentor().is_instrumented_by_opentelemetry
    assert any(
        span.attributes and span.attributes.get('http.url') == 'https://example.com/remaining'
        for span in recorder.exporters[1].get_finished_spans()
    )


async def test_httpx_does_not_uninstrument_an_existing_owner(recorder: Recorder) -> None:
    existing = HTTPXClientInstrumentor()
    existing.instrument()
    plugin = load_logfire(make_host(httpx=True, ui_events=False))
    try:
        await plugin.dispatch(SessionStart(agent=Agent(TestModel()), settings=Settings()))
    finally:
        await close(plugin)
        assert existing.is_instrumented_by_opentelemetry
        existing.uninstrument()
    assert not HTTPX2ClientInstrumentor().is_instrumented_by_opentelemetry


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
    builtin = next(plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'observability')
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
        await loader.disable('observability')
        assert not loader.capabilities()
        assert recorder.exporters[0].closed
        before = len(recorder.spans())
        await agent.run('disabled', capabilities=loader.capabilities())
        assert len(recorder.spans()) == before
        assert len([span for span in existing_exporter.get_finished_spans() if operation(span) == 'invoke_agent']) == 1
        await loader.enable('observability')
        await loader.reload('observability')
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
        load_logfire(make_host(**settings))
    assert 'do-not-store' not in str(error.value)
    assert not recorder.instances


async def test_unload_finishes_inside_outer_cancellation(recorder: Recorder) -> None:
    host = make_host()
    plugin = load_logfire(host)
    with anyio.CancelScope() as scope:
        scope.cancel()
        await close(plugin)
        assert recorder.exporters[0].closed
        await anyio.sleep(0)
    assert scope.cancelled_caught


async def test_shutdown_timeout_is_reported(recorder: Recorder, monkeypatch: pytest.MonkeyPatch) -> None:
    host = make_host()
    plugin = load_logfire(host)
    instance = recorder.instances[0]
    shutdown = instance.shutdown

    def timeout(*, timeout_millis: int, flush: bool = True) -> bool:
        shutdown(timeout_millis=timeout_millis, flush=flush)
        return False

    monkeypatch.setattr(instance, 'shutdown', timeout)
    await close(plugin)
    output = host.console.file
    assert isinstance(output, io.StringIO)
    assert 'Logfire shutdown timed out' in output.getvalue()


def test_activation_failure_closes_local_providers(recorder: Recorder, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*, settings: InstrumentationSettings) -> Instrumentation:
        raise RuntimeError('cannot construct instrumentation')

    monkeypatch.setattr('pydantic_clai2.builtin_plugins.logfire.Instrumentation', fail)
    with pytest.raises(RuntimeError, match='cannot construct'):
        load_logfire(make_host())
    assert recorder.exporters[0].closed


async def test_no_credentials_needs_no_setup_or_console_output(capsys: pytest.CaptureFixture[str]) -> None:
    host = make_host()
    plugin = load_logfire(host)
    try:
        await Agent(TestModel(), deps_type=type(None)).run('hello', capabilities=plugin.capabilities)
    finally:
        await close(plugin)
    captured = capsys.readouterr()
    assert captured.out == captured.err == ''


async def test_logging_never_inspects_caller_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """Source edited on disk mid-session made Logfire warn on stderr, through the live display."""
    monkeypatch.setattr('executing.Source.executing', pytest.fail)
    plugin = load_logfire(make_host(ui_events=True))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            await plugin.dispatch(SessionStart(agent=Agent(TestModel()), settings=Settings()))
            await plugin.dispatch(TurnEnd(text='hello', outcome='completed'))
            telemetry.record('setting {setting} changed', setting='display.theme')
        finally:
            await close(plugin)
    assert [str(warning.message) for warning in caught] == []


@pytest.mark.parametrize('cancelled', [False, True])
async def test_failed_and_cancelled_runs_finish_their_spans(recorder: Recorder, cancelled: bool) -> None:
    host = make_host()
    plugin = load_logfire(host)
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
        await agent.run('run the tool', capabilities=plugin.capabilities)

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
        await close(plugin)
    spans = recorder.spans()
    assert any(operation(span) == 'invoke_agent' for span in spans)
    assert any(operation(span) == 'execute_tool' for span in spans)
    assert all(span.end_time is not None for span in spans)
    if not cancelled:
        assert any(span.status.status_code is trace.StatusCode.ERROR for span in spans)


async def test_flush_timeout_still_stops_providers(recorder: Recorder, monkeypatch: pytest.MonkeyPatch) -> None:
    host = make_host()
    plugin = load_logfire(host)

    def timeout(*, timeout_millis: int) -> bool:
        return False

    monkeypatch.setattr(recorder.instances[0], 'force_flush', timeout)
    await close(plugin)
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
    plugin = load_logfire(host)
    await close(plugin)
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
    plugin = load_logfire(host)
    await close(plugin)
    assert recorder.options[0]['config_dir'] == tmp_path / 'config/pydantic-clai2/logfire'
    assert recorder.options[0]['data_dir'] == recorder.options[0]['config_dir']
    assert recorder.instances[0].config.token is None


@pytest.mark.parametrize('cancel', [False, True])
async def test_interrupted_startup_shuts_down_plugin_providers(
    tmp_path: Path, recorder: Recorder, monkeypatch: pytest.MonkeyPatch, cancel: bool
) -> None:
    started = anyio.Event()

    async def wait(self: object, event: SessionStart) -> None:
        started.set()
        if cancel:
            await anyio.sleep_forever()
        raise RuntimeError('startup failed')

    # A string path: an earlier `loader.reload('logfire')` may have replaced the module's class.
    monkeypatch.setattr('pydantic_clai2.builtin_plugins.logfire.LogfirePlugin.on_session_start', wait)
    store = SettingsStore(tmp_path / 'config.db')
    loader: PluginLoader[None] = PluginLoader(
        store=store,
        console=Console(file=io.StringIO()),
        commands=Commands(),
        session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=store.load()),
        builtin=(next(plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'observability'),),
    )
    if cancel:
        with anyio.fail_after(10):
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(loader.load, 'observability')
                await started.wait()
                tasks.cancel_scope.cancel()
    else:
        with pytest.raises(PluginError, match='startup failed'):
            await loader.load('observability')
    assert recorder.exporters[0].closed
    assert not loader.capabilities()
    assert loader.entries()[0].loaded is None


def messages(recorder: Recorder) -> list[object]:
    return [(span.attributes or {}).get('logfire.msg') for span in recorder.spans()]


async def test_ui_events_can_be_explicitly_disabled(recorder: Recorder) -> None:
    plugin = load_logfire(make_host(ui_events=False))
    try:
        await plugin.dispatch(SessionStart(agent=Agent(TestModel()), settings=Settings()))
        await plugin.dispatch(TurnEnd(text='a private prompt', outcome='cancelled'))
        telemetry.record('setting {setting} changed', setting='display.theme', value='default')
    finally:
        await close(plugin)
    assert messages(recorder) == ['CLAI session']


@pytest.mark.parametrize('model', [Settings().model, None])
@pytest.mark.parametrize('settings', [{}, {'ui_events': True}])
async def test_ui_events_follow_the_plugin_and_keep_setting_names(
    recorder: Recorder, model: str | None, settings: dict[str, JsonValue]
) -> None:
    plugin = load_logfire(make_host(**settings))
    try:
        for event in (
            SessionStart(agent=Agent(TestModel()), settings=Settings(model=model)),
            TurnEnd(text='a private prompt', outcome='cancelled'),
        ):
            await plugin.dispatch(event)
        telemetry.record('setting {setting} changed', setting='sessions.naming', value='password123')
        recorder.instances[0].info('not a UI event', setting='password123')
    finally:
        await close(plugin)
    telemetry.record('after the plugin unloaded')
    assert messages(recorder) == [
        'session started',
        'turn cancelled',
        'setting sessions.naming changed',
        'not a UI event',
        'CLAI session',
    ]
    started, _, changed, other, _ = recorder.spans()
    # The exemption covers only UI records: another span's `setting` is scrubbed as usual.
    assert (other.attributes or {})['setting'] == "[Scrubbed due to 'password']"
    assert (started.attributes or {})['model'] == (model or 'agent default')
    # Names are exempt from scrubbing; any other attribute that looks like a secret is still scrubbed.
    assert (changed.attributes or {})['value'] == "[Scrubbed due to 'password']"
    assert 'a private prompt' not in json.dumps([dict(span.attributes or {}) for span in recorder.spans()])


@pytest.mark.parametrize('content', [False, True])
async def test_ui_events_carry_typed_text_only_with_message_content(recorder: Recorder, content: bool) -> None:
    plugin = load_logfire(make_host(ui_events=True, include_content=content))
    try:
        await plugin.dispatch(SessionStart(agent=Agent(TestModel()), settings=Settings()))
        telemetry.record('prompt submitted', kind='prompt', **telemetry.prompt_text('fix the session bug'))
    finally:
        await close(plugin)
    submitted = next(span for span in recorder.spans() if span.name == 'prompt submitted')
    assert (submitted.attributes or {}).get('prompt') == ('fix the session bug' if content else None)


@pytest.mark.parametrize(
    ('saved', 'send', 'token', 'sent'),
    [
        (True, 'if-token-present', 'lf-shared-write-token', 'if-token-present'),
        (False, 'if-token-present', None, False),
        (False, False, None, False),
    ],
)
async def test_token_from_keys_chooses_the_project(
    recorder: Recorder,
    saved: bool,
    send: Literal[False, 'if-token-present'],
    token: str | None,
    sent: Literal[False, 'if-token-present'],
) -> None:
    if saved:
        save_key(name='CLAI2_LOGFIRE_TOKEN', value='lf-shared-write-token')
    host = make_host(token={'name': 'CLAI2_LOGFIRE_TOKEN'}, send_to_logfire=send)
    await close(load_logfire(host))
    assert recorder.tokens == [token]
    assert recorder.options[0]['send_to_logfire'] == sent
    output = host.console.file
    assert isinstance(output, io.StringIO)
    assert ('CLAI2_LOGFIRE_TOKEN is not in /keys' in output.getvalue()) == (not saved and send is not False)


async def test_self_hosted_base_url_reaches_the_sdk(recorder: Recorder) -> None:
    await close(load_logfire(make_host(base_url='logfire.example.com/')))
    assert recorder.options[0]['base_url'] == 'https://logfire.example.com'


def test_base_url_must_be_an_https_origin(recorder: Recorder) -> None:
    with pytest.raises(ValidationError, match='https URL with no path'):
        load_logfire(make_host(base_url='http://logfire.example.com'))
    assert not recorder.instances


def observability_loader(tmp_path: Path) -> tuple[PluginLoader[None], SettingsStore]:
    store = SettingsStore(tmp_path / 'config.db')
    loader: PluginLoader[None] = PluginLoader(
        store=store,
        console=Console(file=io.StringIO()),
        commands=Commands(),
        session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=store.load()),
        builtin=(next(plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'observability'),),
    )
    return loader, store


async def test_menu_saves_every_option_and_reloads_with_them(
    tmp_path: Path, recorder: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    scripted = Script(
        lists=[
            pick('send_to_logfire'),
            pick('service_name'),
            pick('include_content'),
            pick('include_binary_content'),
            pick('user_tag'),
            pick('httpx'),
            pick('ui_events'),
            MenuResult(cancelled=True),
        ],
        choices=[pick('false'), pick('false'), pick('false'), pick('git-email'), pick('true'), pick('false')],
        texts=[typed('my-clai')],
    )
    monkeypatch.setattr(logfire_plugin, 'RUNNERS', scripted.runners)
    loader, store = observability_loader(tmp_path)
    try:
        await loader.load_all()
        assert loader.configurable('observability')
        assert await loader.command(['configure', 'observability']) == (
            'Saved Send to Logfire.\nSaved Service name.\nSaved Message content.\nSaved Binary content.\n'
            'Saved User tag.\nSaved HTTP requests.\nSaved UI events.'
        )
        [declaration] = store.plugins()
        assert declaration.settings == {
            'service_name': 'my-clai',
            'send_to_logfire': False,
            'include_content': False,
            'include_binary_content': False,
            'user_tag': 'git-email',
            'account': None,
            'token': None,
            'base_url': None,
            'httpx': True,
            'ui_events': False,
        }
        assert [options['service_name'] for options in recorder.options] == ['pydantic-clai2', 'my-clai']
        assert recorder.options[-1]['send_to_logfire'] is False
        assert recorder.exporters[0].closed
    finally:
        await loader.close('exit')


async def test_closing_the_menu_unchanged_keeps_the_running_plugin(
    tmp_path: Path, recorder: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    scripted = Script(lists=[MenuResult(cancelled=True)], choices=[], texts=[])
    monkeypatch.setattr(logfire_plugin, 'RUNNERS', scripted.runners)
    loader, _ = observability_loader(tmp_path)
    try:
        await loader.load_all()
        assert await loader.configure('observability') == 'Logfire settings unchanged.'
        assert len(recorder.instances) == 1
    finally:
        await loader.close('exit')


def test_menu_validates_resets_and_notes_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    source = LogfireSource(make_host(service_name='custom', include_content=False))
    rows = {row.key: row for row in source.rows()}
    assert source.title == 'Observability (Logfire)'
    assert rows[PROJECT].note == 'no LOGFIRE_TOKEN or credentials file'
    assert rows['httpx'].default == 'false'
    assert source.current(rows['httpx']) == 'false'
    assert rows['ui_events'].default == 'true'
    assert source.current(rows['ui_events']) == 'true'
    source.apply(rows['ui_events'], 'false')
    assert source.current(rows['ui_events']) == 'false'
    assert source.reset(rows['ui_events']) == 'Reset UI events.'
    assert source.current(rows['ui_events']) == 'true'
    assert source.current(rows[PROJECT]) == 'LOGFIRE_TOKEN or credentials file'
    assert source.current(rows['send_to_logfire']) == 'if-token-present'
    assert source.current(rows['include_content']) == 'false'
    assert source.problem(rows['service_name'], '') == 'String should have at least 1 character'
    assert source.problem(rows['include_content'], 'maybe') == 'Input should be a valid boolean'
    assert source.problem(rows['send_to_logfire'], 'always') is not None
    assert source.problem(rows['service_name'], 'true') is None
    assert source.reset(rows['service_name']) == 'Reset Service name.'
    assert source.current(rows['service_name']) == 'pydantic-clai2'
    assert source.current(rows['include_content']) == 'false'
    directory = logfire_dir()
    directory.mkdir(parents=True)
    (directory / CREDENTIALS_FILE).write_text('{}')
    assert source.rows()[0].note == 'credentials file found'
    monkeypatch.setenv('LOGFIRE_TOKEN', 'write-token')
    assert source.rows()[0].note == 'LOGFIRE_TOKEN is set'


def test_project_row_names_the_chosen_key_and_resets_to_the_environment() -> None:
    host = make_host(
        token={'name': 'LOGFIRE_TOKEN_TEAM'},
        base_url='https://logfire.example.com',
        account={'email': 'mike@example.com', 'token': {'name': 'LOGFIRE_TOKEN_TEAM'}},
        ui_events=True,
    )
    source = LogfireSource(host)
    project = source.rows()[0]
    assert project.note == '', 'a chosen key replaces the environment, so no note about it'
    assert source.current(project) == 'LOGFIRE_TOKEN_TEAM at https://logfire.example.com'
    hosted = LogfireSource(make_host(token={'name': 'LOGFIRE_TOKEN_US'}))
    assert hosted.current(hosted.rows()[0]) == 'LOGFIRE_TOKEN_US'
    assert source.reset(project) == 'Reset Logfire project.'
    saved = host.settings(logfire_plugin.LogfireSettings)
    assert (saved.token, saved.base_url, saved.account, saved.ui_events) == (None, None, None, True)
