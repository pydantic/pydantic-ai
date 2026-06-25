"""Disk-backed store for runtime-authored capabilities.

Each authored capability is one `<name>.py` file under `directory`; a sibling
`manifest.json` indexes them (name, module file, class, status, last validation
error) and is the surface a UI can read. The store mirrors Loopy's persona
persistence: a read-modify-write upsert keyed by name, fail-soft loading that
skips corrupt entries rather than raising.
"""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError
from pydantic_ai.capabilities import AbstractCapability

from pydantic_ai_harness.experimental.authoring._validate import (
    CapabilityValidationError,
    load_capability_instance,
    validate_capability_file,
)

_NAME_RE = re.compile(r'^[a-z][a-z0-9_]*$')


class AuthoredCapability(BaseModel):
    """Manifest entry for one authored capability."""

    name: str
    module_file: str
    class_name: str
    status: Literal['active', 'disabled'] = 'active'
    last_error: str | None = None


class _Manifest(BaseModel):
    capabilities: list[AuthoredCapability] = []


@dataclass
class CapabilityStore:
    """Read/write index of authored capability `.py` files under `directory`."""

    directory: Path

    @property
    def _manifest_path(self) -> Path:
        return self.directory / 'manifest.json'

    def _load_manifest(self) -> _Manifest:
        path = self._manifest_path
        if not path.exists():
            return _Manifest()
        try:
            return _Manifest.model_validate_json(path.read_text(encoding='utf-8'))
        except (OSError, ValidationError, ValueError):
            return _Manifest()

    def _save_manifest(self, manifest: _Manifest) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        # Write to a temp file in the same directory, then atomically replace the
        # manifest, so a crash mid-write never leaves a partial/corrupt file that
        # `_load_manifest` would read as "no capabilities".
        fd, tmp_name = tempfile.mkstemp(dir=self.directory, prefix='manifest.', suffix='.json.tmp')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as tmp:
                tmp.write(manifest.model_dump_json(indent=2))
            os.replace(tmp_name, self._manifest_path)
        except BaseException:
            os.unlink(tmp_name)
            raise

    def _upsert(self, record: AuthoredCapability) -> None:
        manifest = self._load_manifest()
        for index, existing in enumerate(manifest.capabilities):
            if existing.name == record.name:
                manifest.capabilities[index] = record
                break
        else:
            manifest.capabilities.append(record)
        self._save_manifest(manifest)

    def write(self, name: str, code: str) -> AuthoredCapability:
        """Write `code` to `<name>.py`, validate it, and upsert the manifest entry.

        Raises `ValueError` for an invalid name (before writing anything). A code
        that imports but fails validation is still written (so it can be
        inspected) and recorded with `last_error` set; `load_active` skips it.
        """
        if not _NAME_RE.match(name):
            raise ValueError(
                f'invalid capability name {name!r}: use lowercase letters, digits, and underscores, '
                f'starting with a letter'
            )
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f'{name}.py'
        path.write_text(code, encoding='utf-8')
        try:
            class_name = validate_capability_file(path)
            record = AuthoredCapability(name=name, module_file=path.name, class_name=class_name)
        except CapabilityValidationError as exc:
            record = AuthoredCapability(name=name, module_file=path.name, class_name='', last_error=str(exc))
        self._upsert(record)
        return record

    def disable(self, name: str) -> bool:
        """Mark the named capability disabled so `load_active` stops returning it. Returns whether it existed."""
        manifest = self._load_manifest()
        found = False
        for record in manifest.capabilities:
            if record.name == name:
                record.status = 'disabled'
                found = True
        if found:
            self._save_manifest(manifest)
        return found

    def list_all(self) -> list[AuthoredCapability]:
        """Return every manifest entry, in insertion order."""
        return self._load_manifest().capabilities

    def load_active(self) -> list[AbstractCapability[object]]:
        """Construct every active authored capability for per-run injection.

        Re-imports and re-constructs each active entry. Entries that fail to load
        (corrupt source, construction error) are skipped, not raised, so one bad
        capability never blocks the rest. A load outcome that disagrees with the
        record's `last_error` is persisted back to the manifest: a newly broken
        entry records its error, a re-fixed entry clears it, so the manifest stays
        truthful about which capabilities are actually active.
        """
        manifest = self._load_manifest()
        instances: list[AbstractCapability[object]] = []
        changed = False
        for record in manifest.capabilities:
            if record.status != 'active':
                continue
            try:
                instances.append(load_capability_instance(self.directory / record.module_file))
            except CapabilityValidationError as exc:
                if record.last_error != str(exc):
                    record.last_error = str(exc)
                    changed = True
                continue
            if record.last_error is not None:
                record.last_error = None
                changed = True
        if changed:
            self._save_manifest(manifest)
        return instances
