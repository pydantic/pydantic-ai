"""Tests for the CapabilityCreation capability."""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import anyio.to_thread
import pytest

from pydantic_ai import Agent, DeferredToolRequests, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.messages import ToolReturnPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from pydantic_ai.workspaces import (
    LocalWorkspaceBackend,
    ReadOnlyWorkspace,
    Workspace,
    WorkspaceBackend,
    WorkspaceRef,
)
from pydantic_ai_harness.capability_creation import (
    AuthoredCapability,
    CapabilityCreation,
    CapabilityCreationToolset,
    CapabilityStore,
    CapabilityValidationError,
    load_capability_instance,
    validate_capability_file,
)
from pydantic_ai_harness.code_mode import CodeMode, CodeModeToolset
from pydantic_ai_harness.guardrails import GuardrailResult, ToolGuardrail

VALID_CODE = """
from dataclasses import dataclass

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import FunctionToolset

class _MarkerToolset(FunctionToolset):
    def __init__(self):
        super().__init__()
        self.add_function(self.marker, name='marker')

    async def marker(self) -> str:
        return 'marker-result'

@dataclass
class MarkerCapability(AbstractCapability):
    def get_instructions(self):
        return 'MARKER_INSTRUCTION'

    def get_toolset(self):
        return _MarkerToolset()
"""

NO_SUBCLASS_CODE = 'value = 1\n'

MULTI_CODE = """
from dataclasses import dataclass

from pydantic_ai.capabilities import AbstractCapability

@dataclass
class CapA(AbstractCapability):
    pass

@dataclass
class CapB(AbstractCapability):
    pass
"""

CONSTRUCT_FAILS_CODE = """
from dataclasses import dataclass

from pydantic_ai.capabilities import AbstractCapability

@dataclass
class NeedsArg(AbstractCapability):
    value: int
"""

GETTER_FAILS_CODE = """
from dataclasses import dataclass

from pydantic_ai.capabilities import AbstractCapability

@dataclass
class BadGetter(AbstractCapability):
    def get_toolset(self):
        raise ValueError('boom')
"""

SYNTAX_ERROR_CODE = 'def (:\n'

BAD_TOOLSET_RETURN_CODE = """
from dataclasses import dataclass

from pydantic_ai.capabilities import AbstractCapability

@dataclass
class BadToolset(AbstractCapability):
    def get_toolset(self):
        return 42
"""


def _marker_code(class_name: str, instruction: str) -> str:
    """A valid capability whose class name and static instructions are controllable."""
    return f"""
from dataclasses import dataclass

from pydantic_ai.capabilities import AbstractCapability

@dataclass
class {class_name}(AbstractCapability):
    def get_instructions(self):
        return {instruction!r}
"""


BAD_INSTRUCTIONS_RETURN_CODE = """
from dataclasses import dataclass

from pydantic_ai.capabilities import AbstractCapability

@dataclass
class BadInstructions(AbstractCapability):
    def get_instructions(self):
        return 12345
"""

BAD_MODEL_SETTINGS_RETURN_CODE = """
from dataclasses import dataclass

from pydantic_ai.capabilities import AbstractCapability

@dataclass
class BadModelSettings(AbstractCapability):
    def get_model_settings(self):
        return 'not-a-mapping'
"""


def _write(directory: Path, name: str, code: str) -> Path:
    path = directory / f'{name}.py'
    path.write_text(code, encoding='utf-8')
    return path


def _gate_module(monkeypatch: pytest.MonkeyPatch, *names: str) -> ModuleType:
    """A module of `threading.Event`s that authored code imports to pause where a test needs it."""
    module = ModuleType('_capability_creation_test_gate')
    module.__dict__.update({name: threading.Event() for name in names})
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


def _gate(module: ModuleType, name: str) -> threading.Event:
    event = module.__dict__[name]
    assert isinstance(event, threading.Event)
    return event


class TestValidate:
    def test_valid_returns_class_name(self, tmp_path: Path) -> None:
        assert validate_capability_file(_write(tmp_path, 'm', VALID_CODE)) == 'MarkerCapability'

    def test_no_subclass(self, tmp_path: Path) -> None:
        with pytest.raises(CapabilityValidationError, match='no `AbstractCapability` subclass'):
            validate_capability_file(_write(tmp_path, 'm', NO_SUBCLASS_CODE))

    def test_multiple_subclasses(self, tmp_path: Path) -> None:
        with pytest.raises(CapabilityValidationError, match='found 2: CapA, CapB'):
            validate_capability_file(_write(tmp_path, 'm', MULTI_CODE))

    def test_construct_failure_wrapped(self, tmp_path: Path) -> None:
        with pytest.raises(CapabilityValidationError, match='TypeError'):
            validate_capability_file(_write(tmp_path, 'm', CONSTRUCT_FAILS_CODE))

    def test_getter_failure_wrapped(self, tmp_path: Path) -> None:
        with pytest.raises(CapabilityValidationError, match='ValueError: boom'):
            validate_capability_file(_write(tmp_path, 'm', GETTER_FAILS_CODE))

    def test_bad_instructions_return_rejected(self, tmp_path: Path) -> None:
        # A non-instructions return is caught at author time, not at the next agent.run.
        with pytest.raises(CapabilityValidationError):
            validate_capability_file(_write(tmp_path, 'm', BAD_INSTRUCTIONS_RETURN_CODE))

    def test_bad_model_settings_return_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(CapabilityValidationError, match='ModelSettings mapping'):
            validate_capability_file(_write(tmp_path, 'm', BAD_MODEL_SETTINGS_RETURN_CODE))

    def test_syntax_error_wrapped(self, tmp_path: Path) -> None:
        with pytest.raises(CapabilityValidationError):
            validate_capability_file(_write(tmp_path, 'm', SYNTAX_ERROR_CODE))

    def test_bad_toolset_return_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(CapabilityValidationError, match='get_toolset\\(\\) returned int'):
            validate_capability_file(_write(tmp_path, 'm', BAD_TOOLSET_RETURN_CODE))

    def test_system_exit_wrapped(self, tmp_path: Path) -> None:
        # Authored code that exits must fail validation, not end the host process.
        with pytest.raises(CapabilityValidationError, match='SystemExit'):
            validate_capability_file(_write(tmp_path, 'm', 'raise SystemExit(0)\n'))

    def test_future_annotations_dataclass_field(self, tmp_path: Path) -> None:
        # `dataclasses` resolves string annotations through `sys.modules`, so the authored module must be findable there.
        code = 'from __future__ import annotations\n' + _marker_code('WithField', 'X') + '    value: int = 0\n'
        assert validate_capability_file(_write(tmp_path, 'm', code)) == 'WithField'

    def test_future_annotations_pydantic_forward_ref_after_load(self, tmp_path: Path) -> None:
        # Pydantic resolves a forward reference when the model is first used, after the module body has run.
        code = """from __future__ import annotations

from pydantic import BaseModel

from pydantic_ai.capabilities import AbstractCapability

class A(BaseModel):
    b: B | None = None

class B(BaseModel):
    x: int = 0

class UsesModels(AbstractCapability):
    def get_instructions(self):
        return str(A(b=B(x=1)).b)
"""
        assert validate_capability_file(_write(tmp_path, 'm', code)) == 'UsesModels'

    def test_reload_replaces_module_registration(self, tmp_path: Path) -> None:
        path = _write(tmp_path, 'm', VALID_CODE)
        before = set(sys.modules)
        load_capability_instance(path)
        load_capability_instance(path)
        added = [name for name in set(sys.modules) - before if name.startswith('_authored_capability_m_')]
        assert len(added) == 1

    def test_validation_keeps_no_module_registered(self, tmp_path: Path) -> None:
        before = set(sys.modules)
        validate_capability_file(_write(tmp_path, 'm', VALID_CODE))
        with pytest.raises(CapabilityValidationError):
            validate_capability_file(_write(tmp_path, 'n', NO_SUBCLASS_CODE))
        assert set(sys.modules) - before == set()

    def test_load_leaves_bytecode_settings_and_cache_alone(self, tmp_path: Path) -> None:
        before = sys.dont_write_bytecode
        validate_capability_file(_write(tmp_path, 'm', VALID_CODE))
        assert sys.dont_write_bytecode is before
        assert not (tmp_path / '__pycache__').exists()

    def test_load_instance_success(self, tmp_path: Path) -> None:
        instance = load_capability_instance(_write(tmp_path, 'm', VALID_CODE))
        assert isinstance(instance, AbstractCapability)

    def test_load_instance_passthrough_validation_error(self, tmp_path: Path) -> None:
        with pytest.raises(CapabilityValidationError, match='no `AbstractCapability` subclass'):
            load_capability_instance(_write(tmp_path, 'm', NO_SUBCLASS_CODE))

    def test_load_instance_generic_wrapped(self, tmp_path: Path) -> None:
        with pytest.raises(CapabilityValidationError, match='TypeError'):
            load_capability_instance(_write(tmp_path, 'm', CONSTRUCT_FAILS_CODE))

    def test_load_instance_checks_getter_returns(self, tmp_path: Path) -> None:
        # Loading runs the same checks as authoring, so a capability that failed validation stays out.
        with pytest.raises(CapabilityValidationError, match='ModelSettings mapping'):
            load_capability_instance(_write(tmp_path, 'm', BAD_MODEL_SETTINGS_RETURN_CODE))


class TestCapabilityStore:
    def test_write_valid(self, tmp_path: Path) -> None:
        store = CapabilityStore(tmp_path)
        record = store.write('marker', VALID_CODE)
        assert record.last_error is None
        assert record.class_name == 'MarkerCapability'
        assert (tmp_path / 'marker.py').exists()
        assert (tmp_path / 'manifest.json').exists()
        assert [r.name for r in store.list_all()] == ['marker']

    @pytest.mark.parametrize('name', ['Bad-Name', 'trailing_newline\n', 'a' * 65])
    def test_write_invalid_name_raises(self, tmp_path: Path, name: str) -> None:
        store = CapabilityStore(tmp_path)
        with pytest.raises(ValueError, match='invalid capability name'):
            store.write(name, VALID_CODE)
        assert list(tmp_path.iterdir()) == []

    def test_write_longest_name(self, tmp_path: Path) -> None:
        assert CapabilityStore(tmp_path).write('a' * 64, VALID_CODE).last_error is None

    def test_write_system_exit_records_error(self, tmp_path: Path) -> None:
        # Authored code that exits fails validation instead of ending the process or leaving a healthy-looking entry.
        store = CapabilityStore(tmp_path)
        record = store.write('cap', 'raise SystemExit(0)\n')
        assert record.last_error == 'SystemExit: 0'
        assert store.list_all() == [record]
        assert store.load_active() == []

    def test_write_validation_failure_records_error(self, tmp_path: Path) -> None:
        store = CapabilityStore(tmp_path)
        record = store.write('bad', NO_SUBCLASS_CODE)
        assert record.last_error is not None
        assert record.class_name == ''
        assert (tmp_path / 'bad.py').exists()

    def test_failed_reauthor_keeps_previous_version(self, tmp_path: Path) -> None:
        store = CapabilityStore(tmp_path)
        good = store.write('marker', _marker_code('Foo', 'V1'))
        rejected = store.write('marker', 'raise SystemExit(0)\n')
        assert rejected.last_error == 'SystemExit: 0'
        assert rejected.module_file == 'marker.rejected.py'
        assert (tmp_path / 'marker.rejected.py').read_text(encoding='utf-8') == 'raise SystemExit(0)\n'
        assert store.list_all() == [good]
        active = store.load_active()
        assert [capability.get_instructions() for capability in active] == ['V1']

    def test_successful_reauthor_removes_rejected_source(self, tmp_path: Path) -> None:
        store = CapabilityStore(tmp_path)
        store.write('marker', VALID_CODE)
        store.write('marker', NO_SUBCLASS_CODE)
        assert store.write('marker', _marker_code('Bar', 'V2')).class_name == 'Bar'
        assert sorted(path.name for path in tmp_path.iterdir() if path.suffix == '.py') == ['marker.py']

    def test_replacing_the_source_removes_an_older_rejected_source(self, tmp_path: Path) -> None:
        store = CapabilityStore(tmp_path)
        store.write('marker', VALID_CODE)
        store.write('marker', NO_SUBCLASS_CODE)
        (tmp_path / 'marker.py').write_text(SYNTAX_ERROR_CODE, encoding='utf-8')
        store.load_active()  # the kept version now fails, so the next attempt replaces it
        assert store.write('marker', NO_SUBCLASS_CODE).module_file == 'marker.py'
        assert not (tmp_path / 'marker.rejected.py').exists()

    def test_write_reports_the_same_error_as_load_active(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Validation runs on a staged copy; its error must name the saved file, so a reload has nothing to update.
        store = CapabilityStore(tmp_path)
        record = store.write('bad', SYNTAX_ERROR_CODE)
        assert record.last_error is not None
        assert 'bad.py' in record.last_error
        save_spy = Mock()
        monkeypatch.setattr(CapabilityStore, '_save_manifest', save_spy)
        assert store.load_active() == []
        save_spy.assert_not_called()
        assert sorted(path.name for path in tmp_path.iterdir()) == ['bad.py', 'manifest.json']

    def test_failed_reauthor_of_broken_entry_replaces_it(self, tmp_path: Path) -> None:
        # With no version that passed validation to keep, the new attempt is what gets recorded.
        store = CapabilityStore(tmp_path)
        store.write('bad', NO_SUBCLASS_CODE)
        record = store.write('bad', SYNTAX_ERROR_CODE)
        assert record.module_file == 'bad.py'
        assert store.list_all() == [record]
        assert (tmp_path / 'bad.py').read_text(encoding='utf-8') == SYNTAX_ERROR_CODE

    def test_reauthor_replaces_stale_source(self, tmp_path: Path) -> None:
        # Re-authoring the same name must serve the new source, not cached bytecode.
        store = CapabilityStore(tmp_path)
        store.write('marker', _marker_code('Foo', 'V1'))
        record = store.write('marker', _marker_code('Bar', 'V2'))
        assert record.class_name == 'Bar'
        active = store.load_active()
        assert len(active) == 1
        assert active[0].get_instructions() == 'V2'

    def test_write_bad_getter_return_records_error(self, tmp_path: Path) -> None:
        # A wrong getter return type is rejected at author time, with last_error set.
        store = CapabilityStore(tmp_path)
        record = store.write('bad', BAD_INSTRUCTIONS_RETURN_CODE)
        assert record.last_error is not None
        assert record.class_name == ''

    def test_write_append_distinct(self, tmp_path: Path) -> None:
        store = CapabilityStore(tmp_path)
        store.write('a', VALID_CODE)
        store.write('b', VALID_CODE)
        assert [r.name for r in store.list_all()] == ['a', 'b']

    def test_write_upsert_later_entry(self, tmp_path: Path) -> None:
        store = CapabilityStore(tmp_path)
        store.write('a', VALID_CODE)
        store.write('b', NO_SUBCLASS_CODE)
        # Re-authoring the second entry skips past the first before matching.
        store.write('b', VALID_CODE)
        records = store.list_all()
        assert [r.name for r in records] == ['a', 'b']
        assert records[1].last_error is None

    def test_stores_over_one_directory_share_a_lock(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Two agents over one directory each build their own store; their manifest updates must not interleave.
        saving = [threading.Event(), threading.Event()]
        release = threading.Event()
        real_replace = os.replace

        def held_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
            if Path(dst).name == 'manifest.json':
                first = not saving[0].is_set()
                saving[0 if first else 1].set()
                if first:
                    release.wait(timeout=30)  # hang guard; the test sets it
            real_replace(src, dst)

        monkeypatch.setattr(os, 'replace', held_replace)
        writers = [
            threading.Thread(target=CapabilityStore(tmp_path).write, args=('first', VALID_CODE)),
            threading.Thread(target=CapabilityStore(tmp_path / '.').write, args=('second', VALID_CODE)),
        ]
        writers[0].start()
        try:
            assert saving[0].wait(timeout=30)
            writers[1].start()
            # Correct code never sets it; overlapping updates would within this window.
            assert not saving[1].wait(timeout=0.2)
        finally:
            release.set()
            for writer in writers:
                writer.join(timeout=30)
        assert [record.name for record in CapabilityStore(tmp_path).list_all()] == ['first', 'second']

    def test_blocking_authored_code_does_not_hold_up_other_writes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        gate = _gate_module(monkeypatch, 'entered', 'release')
        code = f'import {gate.__name__} as g\ng.entered.set()\ng.release.wait(60)\n' + VALID_CODE
        writer = threading.Thread(target=CapabilityStore(tmp_path).write, args=('slow', code))
        done = threading.Event()

        def quick() -> None:
            CapabilityStore(tmp_path).write('quick', VALID_CODE)
            CapabilityStore(tmp_path).disable('quick')
            CapabilityStore(tmp_path).load_active()
            done.set()

        writer.start()
        try:
            assert _gate(gate, 'entered').wait(timeout=30)
            threading.Thread(target=quick).start()
            assert done.wait(timeout=30)  # well inside the slow write's 60s hang guard
        finally:
            _gate(gate, 'release').set()
            writer.join(timeout=30)
        assert [record.name for record in CapabilityStore(tmp_path).list_all()] == ['quick', 'slow']

    def test_load_active_keeps_an_entry_rewritten_while_loading(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # An error found by a load that started before a re-author must not overwrite the new entry.
        store = CapabilityStore(tmp_path)
        store.write('x', VALID_CODE)
        gate = _gate_module(monkeypatch, 'entered', 'release')
        broken = f'import {gate.__name__} as g\ng.entered.set()\ng.release.wait(30)\nraise ValueError("broken")\n'
        (tmp_path / 'x.py').write_text(broken, encoding='utf-8')
        loader = threading.Thread(target=store.load_active)
        loader.start()
        try:
            assert _gate(gate, 'entered').wait(timeout=30)
            record = store.write('x', _marker_code('Fixed', 'V2'))
        finally:
            _gate(gate, 'release').set()
            loader.join(timeout=30)
        assert store.list_all() == [record]

    def test_load_manifest_missing_returns_empty(self, tmp_path: Path) -> None:
        assert CapabilityStore(tmp_path / 'absent').list_all() == []

    def test_load_manifest_corrupt_returns_empty(self, tmp_path: Path) -> None:
        (tmp_path / 'manifest.json').write_text('not json{', encoding='utf-8')
        assert CapabilityStore(tmp_path).list_all() == []
        assert CapabilityStore(tmp_path).load_active() == []

    @pytest.mark.parametrize(
        'manifest',
        [
            '{"capabilities": [',
            '{"capabilities": [{"name": "a", "module_file": "a.py", "class_name": "A", "status": "paused"}]}',
        ],
    )
    def test_unreadable_manifest_is_not_replaced(self, tmp_path: Path, manifest: str) -> None:
        # Overwriting a manifest this version cannot read would drop every entry it lists.
        (tmp_path / 'manifest.json').write_text(manifest, encoding='utf-8')
        store = CapabilityStore(tmp_path)
        with pytest.raises(UserError, match='cannot be read'):
            store.write('marker', VALID_CODE)
        with pytest.raises(UserError, match='cannot be read'):
            store.disable('a')
        assert (tmp_path / 'manifest.json').read_text(encoding='utf-8') == manifest
        assert not (tmp_path / 'marker.py').exists()

    def test_load_active_returns_instances(self, tmp_path: Path) -> None:
        store = CapabilityStore(tmp_path)
        store.write('marker', VALID_CODE)
        active = store.load_active()
        assert len(active) == 1
        assert isinstance(active[0], AbstractCapability)

    def test_load_active_skips_disabled(self, tmp_path: Path) -> None:
        store = CapabilityStore(tmp_path)
        store.write('marker', VALID_CODE)
        store.disable('marker')
        assert store.load_active() == []

    def test_load_active_skips_broken(self, tmp_path: Path) -> None:
        store = CapabilityStore(tmp_path)
        store.write('marker', VALID_CODE)
        # The manifest entry stays active, but the source on disk goes bad.
        (tmp_path / 'marker.py').write_text(NO_SUBCLASS_CODE, encoding='utf-8')
        assert store.load_active() == []

    def test_save_manifest_cleans_temp_on_failure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # If the atomic replace fails, the temp file must not be left behind.
        store = CapabilityStore(tmp_path)
        store.write('marker', VALID_CODE)

        def _boom(*args: object, **kwargs: object) -> None:
            raise OSError('replace failed')

        monkeypatch.setattr(os, 'replace', _boom)
        with pytest.raises(OSError, match='replace failed'):
            store.write('second', VALID_CODE)
        leftovers = [p.name for p in tmp_path.iterdir() if p.name.startswith('manifest.') and p.name.endswith('.tmp')]
        assert leftovers == []

    def test_load_active_skips_entry_that_failed_validation(self, tmp_path: Path) -> None:
        # A getter with a wrong return type fails at authoring and must not be loaded into the next run.
        store = CapabilityStore(tmp_path)
        assert store.write('bad', BAD_INSTRUCTIONS_RETURN_CODE).last_error is not None
        assert store.load_active() == []
        assert store.list_all()[0].last_error is not None

    @pytest.mark.parametrize('module_file', ['../outside.py', 'sub/inner.py', 'marker.txt'])
    def test_load_active_refuses_module_file_outside_directory(self, tmp_path: Path, module_file: str) -> None:
        directory = tmp_path / 'authored'
        (directory / 'sub').mkdir(parents=True)
        for path in (tmp_path / 'outside.py', directory / 'sub' / 'inner.py', directory / 'marker.txt'):
            path.write_text(VALID_CODE, encoding='utf-8')
        manifest = f'{{"capabilities": [{{"name": "x", "module_file": "{module_file}", "class_name": "M"}}]}}'
        (directory / 'manifest.json').write_text(manifest, encoding='utf-8')
        store = CapabilityStore(directory)
        assert store.load_active() == []
        last_error = store.list_all()[0].last_error
        assert last_error is not None
        assert 'is not a `.py` file in the directory' in last_error

    def test_load_active_persists_new_error(self, tmp_path: Path) -> None:
        # A capability that validated once but later fails to load gets its error
        # persisted to the manifest, so list_all reflects the real state.
        store = CapabilityStore(tmp_path)
        store.write('marker', VALID_CODE)
        assert store.list_all()[0].last_error is None
        (tmp_path / 'marker.py').write_text(NO_SUBCLASS_CODE, encoding='utf-8')
        assert store.load_active() == []
        record = store.list_all()[0]
        assert record.last_error is not None
        assert 'no `AbstractCapability` subclass' in record.last_error

    def test_load_active_broken_twice_does_not_rewrite(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # A still-broken capability whose error is already persisted must not
        # trigger another manifest write on the next reload.
        store = CapabilityStore(tmp_path)
        store.write('marker', VALID_CODE)
        (tmp_path / 'marker.py').write_text(NO_SUBCLASS_CODE, encoding='utf-8')
        store.load_active()
        assert store.list_all()[0].last_error is not None

        save_spy = Mock()
        monkeypatch.setattr(CapabilityStore, '_save_manifest', save_spy)
        assert store.load_active() == []
        save_spy.assert_not_called()

    def test_load_active_clears_error_on_refix(self, tmp_path: Path) -> None:
        # Re-fixing a broken capability clears its persisted last_error on reload.
        store = CapabilityStore(tmp_path)
        store.write('marker', VALID_CODE)
        (tmp_path / 'marker.py').write_text(NO_SUBCLASS_CODE, encoding='utf-8')
        store.load_active()
        assert store.list_all()[0].last_error is not None
        (tmp_path / 'marker.py').write_text(VALID_CODE, encoding='utf-8')
        active = store.load_active()
        assert len(active) == 1
        assert store.list_all()[0].last_error is None

    def test_load_active_healthy_does_not_rewrite_manifest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A reload with no error transitions must not touch the manifest on disk.
        store = CapabilityStore(tmp_path)
        store.write('marker', VALID_CODE)

        save_spy = Mock()
        monkeypatch.setattr(CapabilityStore, '_save_manifest', save_spy)
        active = store.load_active()
        assert len(active) == 1
        save_spy.assert_not_called()

    def test_disable_found(self, tmp_path: Path) -> None:
        store = CapabilityStore(tmp_path)
        store.write('marker', VALID_CODE)
        assert store.disable('marker') is True
        assert store.list_all()[0].status == 'disabled'

    def test_disable_not_found(self, tmp_path: Path) -> None:
        store = CapabilityStore(tmp_path)
        store.write('marker', VALID_CODE)
        # An existing non-matching entry is skipped, and the result is still False.
        assert store.disable('nope') is False


class TestCapabilityCreationToolset:
    async def test_author_success_message(self, tmp_path: Path) -> None:
        toolset = CapabilityCreationToolset(CapabilityStore(tmp_path))
        result = await toolset.author_capability('marker', VALID_CODE)
        assert 'authored, validated and saved' in result
        assert 'MarkerCapability' in result
        assert 'does not take effect in this run' in result
        assert 'depends on how this agent is set up' in result
        assert 'load_active' not in result  # host wiring is for the docs, not the model

    async def test_author_capability_is_cancellable(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Authored code that blocks must not hold the run past cancellation; the write still completes afterwards.
        gate = _gate_module(monkeypatch, 'entered', 'release')
        code = f'import {gate.__name__} as g\ng.entered.set()\ng.release.wait(30)\n' + VALID_CODE
        written = threading.Event()

        class SignallingStore(CapabilityStore):
            def write(self, name: str, code: str) -> AuthoredCapability:
                try:
                    return super().write(name, code)
                finally:
                    written.set()

        toolset = CapabilityCreationToolset(SignallingStore(tmp_path))
        try:
            with anyio.CancelScope() as scope:
                async with anyio.create_task_group() as tg:
                    tg.start_soon(toolset.author_capability, 'slow', code)
                    assert await anyio.to_thread.run_sync(_gate(gate, 'entered').wait, 30)
                    scope.cancel()
            assert scope.cancelled_caught
            assert not written.is_set()  # the run stopped waiting before the write finished
        finally:
            _gate(gate, 'release').set()
        assert await anyio.to_thread.run_sync(written.wait, 30)
        assert [record.name for record in CapabilityStore(tmp_path).list_all()] == ['slow']

    async def test_author_validation_failure_message(self, tmp_path: Path) -> None:
        toolset = CapabilityCreationToolset(CapabilityStore(tmp_path))
        result = await toolset.author_capability('bad', NO_SUBCLASS_CODE)
        assert 'failed validation' in result
        assert 'previous version' not in result

    async def test_author_failure_message_names_kept_version(self, tmp_path: Path) -> None:
        toolset = CapabilityCreationToolset(CapabilityStore(tmp_path))
        await toolset.author_capability('marker', VALID_CODE)
        result = await toolset.author_capability('marker', NO_SUBCLASS_CODE)
        assert "The previous version of 'marker' is unchanged." in result

    async def test_author_invalid_name_model_retry(self, tmp_path: Path) -> None:
        toolset = CapabilityCreationToolset(CapabilityStore(tmp_path))
        with pytest.raises(ModelRetry, match='invalid capability name'):
            await toolset.author_capability('Bad', VALID_CODE)

    async def test_list_empty(self, tmp_path: Path) -> None:
        toolset = CapabilityCreationToolset(CapabilityStore(tmp_path))
        assert await toolset.list_authored_capabilities() == 'No capabilities authored yet.'

    async def test_list_with_entries(self, tmp_path: Path) -> None:
        toolset = CapabilityCreationToolset(CapabilityStore(tmp_path))
        await toolset.author_capability('marker', VALID_CODE)
        await toolset.author_capability('bad', NO_SUBCLASS_CODE)
        listing = await toolset.list_authored_capabilities()
        assert '- marker [active] MarkerCapability' in listing
        assert '- bad [active] ?' in listing
        assert 'ERROR:' in listing

    async def test_disable_found_message(self, tmp_path: Path) -> None:
        toolset = CapabilityCreationToolset(CapabilityStore(tmp_path))
        await toolset.author_capability('marker', VALID_CODE)
        result = await toolset.disable_authored_capability('marker')
        assert 'disabled' in result

    async def test_disable_not_found_message(self, tmp_path: Path) -> None:
        toolset = CapabilityCreationToolset(CapabilityStore(tmp_path))
        result = await toolset.disable_authored_capability('nope')
        assert 'No authored capability' in result

    async def test_author_capability_stays_native_under_code_mode(self, tmp_path: Path) -> None:
        """`author_capability` takes Python source, so CodeMode exposes it beside `run_code`, not inside it.

        Folding it in would make the model write a Monty script whose argument is a second
        Python module quoted as a string literal. The list/disable tools carry no code and
        stay sandboxed.
        """
        toolset = CapabilityCreationToolset[None](CapabilityStore(tmp_path))
        wrapper = CodeMode[None]().get_wrapper_toolset(toolset)
        assert isinstance(wrapper, CodeModeToolset)

        ctx = RunContext[None](deps=None, model=TestModel(), usage=RunUsage(), prompt=None, messages=[], run_step=1)
        tools = await wrapper.get_tools(ctx)

        assert set(tools) == {'author_capability', 'run_code'}


class TestCapabilityCreationCapability:
    def test_get_instructions_default(self, tmp_path: Path) -> None:
        instructions = CapabilityCreation[object](directory=tmp_path).get_instructions()
        assert isinstance(instructions, str)
        assert 'author_capability' in instructions
        assert 'do not take effect in this run' in instructions
        assert 'load_active' not in instructions

    def test_get_instructions_custom(self, tmp_path: Path) -> None:
        assert CapabilityCreation[object](directory=tmp_path, guidance='X').get_instructions() == 'X'

    def test_get_instructions_empty_omitted(self, tmp_path: Path) -> None:
        assert CapabilityCreation[object](directory=tmp_path, guidance='').get_instructions() is None

    def test_get_toolset_type(self, tmp_path: Path) -> None:
        assert isinstance(CapabilityCreation[object](directory=tmp_path).get_toolset(), CapabilityCreationToolset)

    async def test_invalid_name_model_retry_through_the_capability(self, tmp_path: Path) -> None:
        toolset = CapabilityCreation[object](directory=tmp_path).get_toolset()
        assert isinstance(toolset, CapabilityCreationToolset)
        with pytest.raises(ModelRetry, match='invalid capability name'):
            await toolset.author_capability('Bad', VALID_CODE)

    def test_serialization_name_none(self) -> None:
        assert CapabilityCreation.get_serialization_name() is None

    def test_store_property(self, tmp_path: Path) -> None:
        store = CapabilityCreation[object](directory=tmp_path).store
        assert isinstance(store, CapabilityStore)
        assert store.directory == tmp_path


class TestEndToEnd:
    async def test_authored_capability_injected_and_runs(self, tmp_path: Path) -> None:
        store = CapabilityStore(tmp_path)
        await anyio.to_thread.run_sync(store.write, 'marker', VALID_CODE)
        agent = Agent(TestModel(), capabilities=await anyio.to_thread.run_sync(store.load_active))
        result = await agent.run('go')
        returns = [
            part.content
            for message in result.all_messages()
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        assert 'marker-result' in returns

    async def test_capability_creation_tools_wired(self, tmp_path: Path) -> None:
        agent = Agent(TestModel(), capabilities=[CapabilityCreation(directory=tmp_path)])
        result = await agent.run('go', workspace=LocalWorkspaceBackend(tmp_path))
        assert result.output is not None

    async def test_tool_guardrail_can_require_approval_before_authoring(self, tmp_path: Path) -> None:
        # The documented way to have a person approve authored code before it runs.
        agent = Agent(
            TestModel(call_tools=['author_capability']),
            capabilities=[
                CapabilityCreation(directory=tmp_path),
                ToolGuardrail(guard=lambda call: GuardrailResult.approve(), tools=['author_capability']),
            ],
            output_type=[str, DeferredToolRequests],
        )
        result = await agent.run('go', workspace=LocalWorkspaceBackend(tmp_path))
        assert isinstance(result.output, DeferredToolRequests)
        assert [call.tool_name for call in result.output.approvals] == ['author_capability']
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.parametrize('workspace', ['none', 'read-only', 'sandbox'])
    async def test_refuses_a_workspace_that_is_not_this_machine_and_writable(
        self, tmp_path: Path, workspace: str
    ) -> None:
        backends: dict[str, WorkspaceBackend | None] = {
            'none': None,
            'read-only': ReadOnlyWorkspace(Workspace(LocalWorkspaceBackend(tmp_path))),
            'sandbox': _Sandbox(tmp_path),
        }
        agent = Agent(TestModel(), capabilities=[CapabilityCreation(directory=tmp_path)])
        with pytest.raises(UserError, match='imports model-written Python into the agent process'):
            await agent.run('go', workspace=backends[workspace])


class _Sandbox(LocalWorkspaceBackend):
    """A backend that names a provider other than this machine."""

    @property
    def ref(self) -> WorkspaceRef:
        return WorkspaceRef(provider='sandbox', id='box-1')
