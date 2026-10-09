"""Startup failures reach the configured observability plugin, including failures before it loads."""

import sys
from pathlib import Path

import pytest
from opentelemetry import trace

from pydantic_clai2.config import PluginSettings
from pydantic_clai2.config.api_keys import save_key
from tests.clai2.test_logfire import Recorder, recorder as recorder
from tests.clai2.test_plugin_loader import Harness


@pytest.mark.parametrize('observability_index', [0, 1, 3])
@pytest.mark.parametrize('ui_events', [False, True])
async def test_startup_import_errors_reach_configured_logfire(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recorder: Recorder,
    observability_index: int,
    ui_events: bool,
) -> None:
    # Real imports in drop-in plugins fail even when the test environment has these optional packages.
    monkeypatch.setitem(sys.modules, 'fastmcp', None)
    monkeypatch.setitem(sys.modules, 'anthropic', None)
    failures = [
        ('mcp', 'fastmcp', 'ModuleNotFoundError'),
        ('logfire_mcp', 'fastmcp', 'ModuleNotFoundError'),
        ('claude_code', 'anthropic', 'ImportError'),
    ]
    declarations: list[PluginSettings] = []
    for name, dependency, error_type in failures:
        path = tmp_path / f'{name}.py'
        source = f'import {dependency}\n'
        if error_type == 'ImportError':
            source = (
                f'try:\n    import {dependency}\n'
                f'except ImportError as exc:\n    raise ImportError("Please install {dependency}") from exc\n'
            )
        path.write_text(source)
        declarations.append(PluginSettings(id=name, factory=name, path=str(path)))
    declarations.insert(
        observability_index,
        PluginSettings(id='observability', factory='pydantic_clai2.builtin_plugins.logfire'),
    )
    harness = Harness(tmp_path, builtin=tuple(declarations))
    harness.write('healthy')
    save_key(name='CLAI2_LOGFIRE_TOKEN', value='lf-team-write-token')
    harness.store.save_plugin(
        PluginSettings(
            id='observability',
            factory='pydantic_clai2.builtin_plugins.logfire',
            settings={
                'token': {'name': 'CLAI2_LOGFIRE_TOKEN'},
                'service_name': 'plugin-errors-test',
                'ui_events': ui_events,
            },
        )
    )
    global_provider = trace.get_tracer_provider()
    try:
        await harness.loader.load_all()
        assert 'healthy started' in harness.text
        assert trace.get_tracer_provider() is global_provider
        assert recorder.tokens == ['lf-team-write-token']
        assert recorder.options[0]['service_name'] == 'plugin-errors-test'
        assert recorder.options[0]['send_to_logfire'] == 'if-token-present'
        errors = [span for span in recorder.spans() if (span.attributes or {}).get('logfire.level_num') == 17]
        assert len(errors) == len(failures)
        for span, (name, dependency, error_type) in zip(errors, failures):
            assert (span.attributes or {})['plugin'] == name
            assert len(span.events) == 1
            exception = span.events[0]
            assert exception.name == 'exception'
            attributes = exception.attributes or {}
            assert attributes['exception.type'] == error_type
            assert dependency in str(attributes['exception.message'])
            assert f'{name}.py' in str(attributes['exception.stacktrace'])
            assert f'Plugin {name!r}: {error_type}:' in harness.text
        assert 'ModuleNotFoundError' in str((errors[-1].events[0].attributes or {})['exception.stacktrace'])

        for name, _, _ in failures:
            (tmp_path / f'{name}.py').write_text(
                'from pydantic_clai2.plugins import Plugin\nclass Recovered(Plugin): pass\n'
            )
        await harness.loader.load_all()
        assert all(entry.loaded is not None for entry in harness.loader.entries())
        assert [span for span in recorder.spans() if (span.attributes or {}).get('logfire.level_num') == 17] == errors
    finally:
        await harness.loader.close('exit')
    root = next(span for span in recorder.spans() if span.name == 'CLAI session')
    assert root.parent is None
    assert all(span.parent == root.context for span in errors)
    assert all('logfire.tags' not in (span.attributes or {}) for span in errors)
    assert all(span.instrumentation_scope is not None and span.instrumentation_scope.name == 'clai2' for span in errors)


@pytest.mark.parametrize('enabled', [False, True])
async def test_startup_errors_without_working_observability_still_print(
    tmp_path: Path, recorder: Recorder, enabled: bool
) -> None:
    harness = Harness(
        tmp_path,
        builtin=(
            PluginSettings(
                id='observability',
                factory='pydantic_clai2.builtin_plugins.logfire',
                enabled=enabled,
                settings={'unknown_setting': True},
            ),
        ),
    )
    (harness.store.plugins_dir / 'broken.py').write_text('raise ImportError("missing plugin dependency")\n')
    harness.write('healthy')
    try:
        await harness.loader.load_all()
        assert "Plugin 'broken': ImportError: missing plugin dependency" in harness.text
        assert 'healthy started' in harness.text
        assert recorder.instances == []
        assert recorder.spans() == []
    finally:
        await harness.loader.close('exit')


@pytest.mark.parametrize('module_present', [False, True])
async def test_startup_only_reports_import_errors_inside_available_plugins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recorder: Recorder, module_present: bool
) -> None:
    monkeypatch.setattr(sys, 'path', [str(tmp_path), *sys.path])
    monkeypatch.setitem(sys.modules, 'fastmcp', None)
    if module_present:
        (tmp_path / 'clai2_startup_test_plugin.py').write_text('import fastmcp\n')
    harness = Harness(
        tmp_path,
        builtin=(
            PluginSettings(id='mcp', factory='clai2_startup_test_plugin'),
            PluginSettings(id='observability', factory='pydantic_clai2.builtin_plugins.logfire'),
        ),
    )
    try:
        await harness.loader.load_all()
        entry = next(entry for entry in harness.loader.entries() if entry.name == 'mcp')
        assert entry.error is not None
        assert 'ModuleNotFoundError' in entry.error
        errors = [span for span in recorder.spans() if (span.attributes or {}).get('logfire.level_num') == 17]
        if module_present:
            assert "Plugin 'mcp': ModuleNotFoundError:" in harness.text
            assert len(errors) == 1
            assert (errors[0].attributes or {})['plugin'] == 'mcp'
            assert 'fastmcp' in str((errors[0].events[0].attributes or {})['exception.message'])
        else:
            assert harness.text == ''
            assert errors == []
    finally:
        await harness.loader.close('exit')


async def test_failing_observer_does_not_prevent_error_reporting(tmp_path: Path, recorder: Recorder) -> None:
    observer = tmp_path / 'observer.py'
    observer.write_text(
        'from pydantic_clai2.plugins import Plugin, PluginLoadFailed\n'
        'class Observer(Plugin):\n'
        '    async def on_plugin_load_failed(self, event: PluginLoadFailed) -> None:\n'
        '        raise RuntimeError("observer failed")\n'
    )
    harness = Harness(
        tmp_path,
        builtin=(
            PluginSettings(id='observer', factory='observer', path=str(observer)),
            PluginSettings(
                id='observability',
                factory='pydantic_clai2.builtin_plugins.logfire',
                settings={'send_to_logfire': False},
            ),
        ),
    )
    (harness.store.plugins_dir / 'broken.py').write_text('raise ImportError("missing dependency")\n')
    try:
        await harness.loader.load_all()
        assert "Plugin 'observer': RuntimeError: observer failed" in harness.text
        assert recorder.options[0]['send_to_logfire'] is False
        errors = [span for span in recorder.spans() if (span.attributes or {}).get('logfire.level_num') == 17]
        assert len(errors) == 1
        assert (errors[0].attributes or {})['plugin'] == 'broken'
    finally:
        await harness.loader.close('exit')
