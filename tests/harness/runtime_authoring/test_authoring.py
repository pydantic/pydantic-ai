"""Tests for the RuntimeAuthoring capability."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import ToolReturnPart
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.runtime_authoring import (
    AuthoringToolset,
    CapabilityStore,
    CapabilityValidationError,
    RuntimeAuthoring,
)
from pydantic_ai_harness.runtime_authoring._validate import (
    load_capability_instance,
    validate_capability_file,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    """Run async tests on the asyncio backend (matching upstream pydantic-ai)."""
    return 'asyncio'


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


def _return_none(*args: object, **kwargs: object) -> None:
    return None


def _return_spec_without_loader(*args: object, **kwargs: object) -> SimpleNamespace:
    return SimpleNamespace(loader=None)


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

    def test_load_module_no_spec(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        path = _write(tmp_path, 'm', VALID_CODE)
        monkeypatch.setattr(importlib.util, 'spec_from_file_location', _return_none)
        with pytest.raises(CapabilityValidationError, match='import spec'):
            validate_capability_file(path)

    def test_load_module_no_loader(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        path = _write(tmp_path, 'm', VALID_CODE)
        monkeypatch.setattr(importlib.util, 'spec_from_file_location', _return_spec_without_loader)
        with pytest.raises(CapabilityValidationError, match='import spec'):
            validate_capability_file(path)

    def test_load_instance_success(self, tmp_path: Path) -> None:
        instance = load_capability_instance(_write(tmp_path, 'm', VALID_CODE))
        assert isinstance(instance, AbstractCapability)

    def test_load_instance_passthrough_validation_error(self, tmp_path: Path) -> None:
        with pytest.raises(CapabilityValidationError, match='no `AbstractCapability` subclass'):
            load_capability_instance(_write(tmp_path, 'm', NO_SUBCLASS_CODE))

    def test_load_instance_generic_wrapped(self, tmp_path: Path) -> None:
        with pytest.raises(CapabilityValidationError, match='TypeError'):
            load_capability_instance(_write(tmp_path, 'm', CONSTRUCT_FAILS_CODE))


class TestCapabilityStore:
    def test_write_valid(self, tmp_path: Path) -> None:
        store = CapabilityStore(tmp_path)
        record = store.write('marker', VALID_CODE)
        assert record.last_error is None
        assert record.class_name == 'MarkerCapability'
        assert (tmp_path / 'marker.py').exists()
        assert (tmp_path / 'manifest.json').exists()
        assert [r.name for r in store.list_all()] == ['marker']

    def test_write_invalid_name_raises(self, tmp_path: Path) -> None:
        store = CapabilityStore(tmp_path)
        with pytest.raises(ValueError, match='invalid capability name'):
            store.write('Bad-Name', VALID_CODE)

    def test_write_validation_failure_records_error(self, tmp_path: Path) -> None:
        store = CapabilityStore(tmp_path)
        record = store.write('bad', NO_SUBCLASS_CODE)
        assert record.last_error is not None
        assert record.class_name == ''
        assert (tmp_path / 'bad.py').exists()

    def test_write_upsert_replaces(self, tmp_path: Path) -> None:
        store = CapabilityStore(tmp_path)
        store.write('marker', VALID_CODE)
        store.write('marker', NO_SUBCLASS_CODE)
        records = store.list_all()
        assert len(records) == 1
        assert records[0].last_error is not None

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
        store.write('b', VALID_CODE)
        # Re-authoring the second entry skips past the first before matching.
        store.write('b', NO_SUBCLASS_CODE)
        records = store.list_all()
        assert [r.name for r in records] == ['a', 'b']
        assert records[1].last_error is not None

    def test_load_manifest_missing_returns_empty(self, tmp_path: Path) -> None:
        assert CapabilityStore(tmp_path / 'absent').list_all() == []

    def test_load_manifest_corrupt_returns_empty(self, tmp_path: Path) -> None:
        (tmp_path / 'manifest.json').write_text('not json{', encoding='utf-8')
        assert CapabilityStore(tmp_path).list_all() == []

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

    def test_save_manifest_atomic_over_partial_prior_file(self, tmp_path: Path) -> None:
        # A prior interrupted save left a partial/corrupt manifest; the next save
        # must replace it atomically, leaving a valid manifest and no temp files.
        (tmp_path / 'manifest.json').write_text('{"capabilities": [', encoding='utf-8')
        store = CapabilityStore(tmp_path)
        store.write('marker', VALID_CODE)
        assert [r.name for r in store.list_all()] == ['marker']
        leftovers = [p.name for p in tmp_path.iterdir() if p.name.startswith('manifest.') and p.name.endswith('.tmp')]
        assert leftovers == []

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


class TestAuthoringToolset:
    async def test_author_success_message(self, tmp_path: Path) -> None:
        toolset = AuthoringToolset(CapabilityStore(tmp_path))
        result = await toolset.author_capability('marker', VALID_CODE)
        assert 'authored and validated' in result
        assert 'MarkerCapability' in result
        assert 'next agent run' in result

    async def test_author_validation_failure_message(self, tmp_path: Path) -> None:
        toolset = AuthoringToolset(CapabilityStore(tmp_path))
        result = await toolset.author_capability('bad', NO_SUBCLASS_CODE)
        assert 'failed validation' in result

    async def test_author_invalid_name_model_retry(self, tmp_path: Path) -> None:
        toolset = AuthoringToolset(CapabilityStore(tmp_path))
        with pytest.raises(ModelRetry, match='invalid capability name'):
            await toolset.author_capability('Bad', VALID_CODE)

    async def test_list_empty(self, tmp_path: Path) -> None:
        toolset = AuthoringToolset(CapabilityStore(tmp_path))
        assert await toolset.list_authored_capabilities() == 'No capabilities authored yet.'

    async def test_list_with_entries(self, tmp_path: Path) -> None:
        toolset = AuthoringToolset(CapabilityStore(tmp_path))
        await toolset.author_capability('marker', VALID_CODE)
        await toolset.author_capability('bad', NO_SUBCLASS_CODE)
        listing = await toolset.list_authored_capabilities()
        assert '- marker [active] MarkerCapability' in listing
        assert '- bad [active] ?' in listing
        assert 'ERROR:' in listing

    async def test_disable_found_message(self, tmp_path: Path) -> None:
        toolset = AuthoringToolset(CapabilityStore(tmp_path))
        await toolset.author_capability('marker', VALID_CODE)
        result = await toolset.disable_authored_capability('marker')
        assert 'disabled' in result

    async def test_disable_not_found_message(self, tmp_path: Path) -> None:
        toolset = AuthoringToolset(CapabilityStore(tmp_path))
        result = await toolset.disable_authored_capability('nope')
        assert 'No authored capability' in result


class TestRuntimeAuthoringCapability:
    def test_get_instructions_default(self, tmp_path: Path) -> None:
        instructions = RuntimeAuthoring[object](directory=tmp_path).get_instructions()
        assert isinstance(instructions, str)
        assert 'author_capability' in instructions

    def test_get_instructions_custom(self, tmp_path: Path) -> None:
        assert RuntimeAuthoring[object](directory=tmp_path, guidance='X').get_instructions() == 'X'

    def test_get_instructions_empty_omitted(self, tmp_path: Path) -> None:
        assert RuntimeAuthoring[object](directory=tmp_path, guidance='').get_instructions() is None

    def test_get_toolset_type(self, tmp_path: Path) -> None:
        assert isinstance(RuntimeAuthoring[object](directory=tmp_path).get_toolset(), AuthoringToolset)

    def test_serialization_name_none(self) -> None:
        assert RuntimeAuthoring.get_serialization_name() is None

    def test_store_property(self, tmp_path: Path) -> None:
        store = RuntimeAuthoring[object](directory=tmp_path).store
        assert isinstance(store, CapabilityStore)
        assert store.directory == tmp_path


class TestEndToEnd:
    async def test_authored_capability_injected_and_runs(self, tmp_path: Path) -> None:
        store = CapabilityStore(tmp_path)
        store.write('marker', VALID_CODE)
        agent = Agent(TestModel(), capabilities=store.load_active())
        result = await agent.run('go')
        returns = [
            part.content
            for message in result.all_messages()
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        assert 'marker-result' in returns

    async def test_runtime_authoring_tools_wired(self, tmp_path: Path) -> None:
        agent = Agent(TestModel(), capabilities=[RuntimeAuthoring(directory=tmp_path)])
        result = await agent.run('go')
        assert result.output is not None
