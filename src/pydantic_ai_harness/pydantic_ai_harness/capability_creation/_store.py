"""Disk-backed store for runtime-authored capabilities.

Each authored capability is one `<name>.py` file under `directory`; a sibling
`manifest.json` indexes them (name, module file, class, status, last validation
error) and is the surface a UI can read. Updates are a read-modify-write upsert
keyed by name. Reads are fail-soft: a broken entry is skipped rather than raised.
Writes refuse an unreadable manifest rather than replacing it, so its entries
are not lost.

Every store over one directory in this process shares a lock around each
manifest update, so two agents or toolsets over the same directory do not lose
each other's updates. Authored code runs outside that lock. Separate processes
are not coordinated.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import re
import shutil
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai_harness.capability_creation._validate import (
    CapabilityValidationError,
    load_capability_instance,
    validate_capability_source,
)

_NAME_RE = re.compile(r'[a-z][a-z0-9_]{0,63}')

_DIRECTORY_LOCKS: dict[Path, threading.Lock] = {}
_DIRECTORY_LOCKS_GUARD = threading.Lock()


def rejected_module_file(name: str) -> str:
    """The file `CapabilityStore.write` saves rejected code for `name` in while it keeps the previous version."""
    return f'{name}.rejected.py'


def _directory_lock(directory: Path) -> threading.Lock:
    """The lock shared by every store over `directory` in this process."""
    key = directory.resolve()
    with _DIRECTORY_LOCKS_GUARD:
        return _DIRECTORY_LOCKS.setdefault(key, threading.Lock())


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

    def _module_path(self, record: AuthoredCapability) -> Path:
        """The record's module file, refused unless it is a `.py` file directly in `directory`.

        The check is on the path as written, so a symlink in `directory` is followed wherever it points.
        """
        path = self.directory / record.module_file
        if path.parent != self.directory or path.suffix != '.py':
            raise CapabilityValidationError(f'module file {record.module_file!r} is not a `.py` file in the directory')
        return path

    def _load_manifest(self, *, for_update: bool = False) -> _Manifest:
        """Read the manifest; an unreadable one reads as empty, or raises `UserError` when `for_update` is set."""
        path = self._manifest_path
        if not path.exists():
            return _Manifest()
        try:
            return _Manifest.model_validate_json(path.read_text(encoding='utf-8'))
        except (OSError, ValidationError, ValueError) as exc:
            if for_update:
                raise UserError(
                    f'The capability manifest {str(path)!r} cannot be read ({type(exc).__name__}), so it was left '
                    'unchanged. Repair or remove it before authoring or disabling capabilities.'
                ) from exc
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

    def write(self, name: str, code: str) -> AuthoredCapability:
        """Validate `code` as the capability `name` and save it as `<name>.py`.

        Raises `ValueError` for an invalid name and `UserError` for an unreadable
        manifest, both before writing anything.

        The code is validated before it replaces anything. Code that passes
        replaces `<name>.py` and its manifest entry. Code that fails is kept on
        disk so it can be inspected: when the current entry for `name` has no
        recorded error and its `<name>.py` still exists, that version and its
        manifest entry stay as they are, the rejected code is saved as
        `<name>.rejected.py`, and the returned record (with that `module_file`)
        is not saved to the manifest. Otherwise the code is saved as `<name>.py`
        and recorded with `last_error` set, so `load_active` skips it.

        If saving the manifest fails, the previous `<name>.py` (or its absence)
        is put back before the error is raised, so the source and its manifest
        entry change together or not at all.
        """
        if not _NAME_RE.fullmatch(name):
            raise ValueError(
                f'invalid capability name {name!r}: use at most 64 lowercase letters, digits, and underscores, '
                f'starting with a letter'
            )
        self._load_manifest(for_update=True)
        self.directory.mkdir(parents=True, exist_ok=True)
        live = self.directory / f'{name}.py'
        rejected = self.directory / rejected_module_file(name)
        # Authored code runs outside the directory lock, so code that blocks holds up only this call.
        staging = Path(tempfile.mkdtemp(dir=self.directory, prefix='.candidate-'))
        candidate = staging / live.name
        try:
            candidate.write_text(code, encoding='utf-8')
            try:
                class_name, error = validate_capability_source(candidate, file=live), None
            except CapabilityValidationError as exc:
                class_name, error = '', str(exc)
            with _directory_lock(self.directory):
                manifest = self._load_manifest(for_update=True)
                previous = next((record for record in manifest.capabilities if record.name == name), None)
                if error is not None and previous is not None and previous.last_error is None and live.exists():
                    os.replace(candidate, rejected)
                    return AuthoredCapability(name=name, module_file=rejected.name, class_name='', last_error=error)
                record = AuthoredCapability(name=name, module_file=live.name, class_name=class_name, last_error=error)
                if previous is None:
                    manifest.capabilities.append(record)
                else:
                    manifest.capabilities[manifest.capabilities.index(previous)] = record
                # Capability names start with a letter, so no candidate is named like the backup.
                backup = staging / '.previous'
                replaced = live.exists()
                if replaced:
                    shutil.copy2(live, backup)
                os.replace(candidate, live)
                try:
                    self._save_manifest(manifest)
                except BaseException:
                    if replaced:
                        os.replace(backup, live)
                    else:
                        live.unlink()
                    raise
                # The new version is committed, so a leftover rejected file must not fail the write.
                with contextlib.suppress(OSError):
                    rejected.unlink(missing_ok=True)
                return record
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def disable(self, name: str) -> bool:
        """Mark the named capability disabled so `load_active` stops returning it. Returns whether it existed."""
        with _directory_lock(self.directory):
            manifest = self._load_manifest(for_update=True)
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

        Re-imports, re-constructs, and re-validates each active entry with the same
        checks as `write`. Entries that fail (corrupt source, construction error, a
        getter with a wrong return type, a module file outside `directory`) are
        skipped, not raised. A load outcome that disagrees with the record's
        `last_error` is persisted back to the manifest: a newly broken entry records
        its error, a re-fixed entry clears it, so the manifest stays truthful about
        which capabilities are actually active. An outcome is not persisted when the
        entry or its source was rewritten while it was loading, keeping the newer
        write's state.

        The returned instances are built from each file as this call read it. A
        `write` that lands during the call may be in them or not, and the next call
        picks it up. Making them always match the manifest would mean holding the
        directory lock while authored code runs, which would let one module that
        blocks hold up every write.
        """
        manifest = self._load_manifest()
        instances: list[AbstractCapability[object]] = []
        changed: dict[str, tuple[AuthoredCapability, str | None, AuthoredCapability]] = {}
        for record in manifest.capabilities:
            if record.status != 'active':
                continue
            digest = self._source_digest(record)
            try:
                instances.append(load_capability_instance(self._module_path(record)))
                error = None
            except CapabilityValidationError as exc:
                error = str(exc)
            if record.last_error != error:
                changed[record.name] = (record, digest, record.model_copy(update={'last_error': error}))
        if changed:
            self._record_load_errors(changed)
        return instances

    def _source_digest(self, record: AuthoredCapability) -> str | None:
        """A hash of the record's module source, or `None` when there is none to read."""
        try:
            return hashlib.sha256(self._module_path(record).read_bytes()).hexdigest()
        except (CapabilityValidationError, OSError):
            return None

    def _record_load_errors(
        self, changed: dict[str, tuple[AuthoredCapability, str | None, AuthoredCapability]]
    ) -> None:
        """Save each new `last_error` whose entry and source no write replaced while it was loading."""
        with _directory_lock(self.directory):
            manifest = self._load_manifest()
            updated = False
            for index, record in enumerate(manifest.capabilities):
                if record.name not in changed:
                    continue
                loaded, digest, result = changed[record.name]
                if record == loaded and self._source_digest(record) == digest:
                    manifest.capabilities[index] = result
                    updated = True
            if updated:
                self._save_manifest(manifest)
