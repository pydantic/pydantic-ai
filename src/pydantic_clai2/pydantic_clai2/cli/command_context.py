"""Conversation-local settings and actions behind `/set`."""

import threading
from collections.abc import Callable, Collection, Iterable, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import JsonValue, TypeAdapter, ValidationError

from pydantic_ai.settings import ModelSettings
from pydantic_clai2.commands import Command, set_completions
from pydantic_clai2.config import SETTING_FIELDS, SETTING_KEYS, STRING_SETTINGS, Settings, resolve_settings
from pydantic_clai2.config.project_settings import ProjectSettings, repository_directories, with_current_approvals
from pydantic_clai2.config.settings_store import SettingsStore, setting_json
from pydantic_clai2.ui import telemetry
from pydantic_clai2.ui.rendering.tool_output import terminal_text

if TYPE_CHECKING:
    from pydantic_clai2.runtime.session import FamilyDefaults


@runtime_checkable
class CommandProvider(Protocol):
    """Compatibility interface for capability-provided terminal commands."""

    def get_commands(self, context: 'CommandContext') -> Sequence[Command]:
        """Declare commands at startup."""
        ...


@dataclass(kw_only=True)
class CommandContext:
    """Conversation-local settings and actions, without global mutable state."""

    settings: Settings
    store: SettingsStore
    apply_setting: Callable[[str, Settings], None]
    project: ProjectSettings = field(default_factory=ProjectSettings)
    """Read-only here: `/set` writes the user store, and the project file wins again at next start."""
    plugin_models: Callable[[], Sequence[str]] = lambda: ()
    """Models loaded plugins offer with `PluginHost.model_provider`, as `PREFIX:NAME`."""
    settings_model: Callable[[str], str] = lambda model: model
    """The model whose `/model settings` controls and defaults a model takes.

    Differs for a plugin's `settings_from`, and for a chain, which takes its first model's.
    """
    edits: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    """Held by each save, so the terminal's `/set` menu, typed commands, and the web UI, each on its own thread, never lose one."""
    settings_for: Callable[[ProjectSettings], Settings] | None = None
    """The settings CLAI starts with beside a project file, launch flags included; without it, saved ones, then the project's."""
    workspace: Callable[[], Path] = Path.cwd
    """The directory this session works in, where `/project` looks for approvals when no file sets them."""
    saved: Callable[[str], None] = lambda key: None
    """Told the key `/set` saved or reset, once applied here, so the process's other live sessions can follow."""

    def __post_init__(self) -> None:
        """Keep the configured model selectable, including preferences saved before the model list existed."""
        if self.settings.model:
            self.store.add_model(name=self.settings.model)

    def from_project(self, key: str) -> bool:
        """Whether the project file sets `key`, so a saved value only lasts for this session."""
        return key in self.project.overrides

    def use_project(self, project: ProjectSettings, settings: Settings) -> None:
        """Adopt another directory's project file and the settings CLAI starts with there, applying each one."""
        self.project = project
        self.settings = settings
        if settings.model:
            self.store.add_model(name=settings.model)
        for key in SETTING_FIELDS:
            self.apply_setting(key, settings)

    def project_command(self, args: list[str]) -> str:
        """`/project` lists what the project file sets; `approve` and `revoke` decide its other keys for this repository.

        The model's tools never reach `/commands`, and a file a repository ships cannot approve itself. This
        does not hold against a process that drives CLAI's input or writes the user's settings database, such as
        an agent's shell. An approval keeps the value approved, so a changed value waits again.
        """
        project, root = self.project, self.project.root
        stored = self._stored_approvals()
        if root is None and not stored:
            raise ValueError('No project file: CLAI looks for .clai/settings.json from here up to the repository root.')
        match args:
            case [] | ['list']:
                return _project_listing(project, stored)
            case ['approve', _, *_] if root is None:
                raise ValueError('No project file to approve: /project revoke removes the approvals saved here.')
            case ['approve', *names] if names and root is not None:
                keys = _project_keys(names, project.pending, verb='approve')
                with self.edits:
                    self.store.approve_project_settings(root, {key: project.pending[key] for key in keys})
                    self._use_project(with_current_approvals(project, store=self.store))
                return f'Approved {_file_names(keys)} for {_one_line(root)}. Applied, unless a launch flag sets it.'
            case ['revoke', *names] if names:
                return self._revoke_project(names, stored)
            case _:
                raise ValueError(PROJECT_USAGE)

    def _revoke_project(self, names: list[str], stored: dict[Path, dict[str, str]]) -> str:
        """`/project revoke`: remove the `stored` approvals `names` choose, wherever they were saved."""
        keys = _project_keys(names, {key for rows in stored.values() for key in rows}, verb='revoke')
        touched = [folder for folder, rows in stored.items() if rows.keys() & set(keys)]
        with self.edits:
            for folder in touched:
                self.store.revoke_project_settings(folder, keys)
            self._use_project(with_current_approvals(self.project, store=self.store))
        folders = ', '.join(_one_line(folder) for folder in touched)
        return f'Revoked {_file_names(keys)} for {folders}.'

    def project_completions(self, args: list[str]) -> Iterable[str]:
        """`approve` and `revoke`, then `all` and the names each one takes."""
        if len(args) <= 1:
            return ('approve', 'revoke')
        if args[0] == 'approve':
            return ('all', *(SETTING_FIELDS[key] for key in self.project.pending))
        stored = {key for rows in self._stored_approvals().values() for key in rows}
        return ('all', *(SETTING_FIELDS.get(key, key) for key in sorted(stored)))

    def _stored_approvals(self) -> dict[Path, dict[str, str]]:
        """Every approval saved for this repository, by folder, whether or not the file sets that key now.

        The file may have dropped a key, changed its value, or gone, and may bring it back later, when a
        leftover approval would apply again. So these folders are the project's, and those up to the git root.
        """
        root = self.project.root
        folders = dict.fromkeys((*(() if root is None else (root,)), *repository_directories(self.workspace())))
        return {folder: rows for folder in folders if (rows := self.store.approved_project_settings(folder))}

    def _use_project(self, project: ProjectSettings) -> None:
        """Adopt `project` for this session, applying each key that starts or stops applying, as CLAI would start with it.

        That includes a key another live session approved or revoked since; such a session is not told at the time.
        """
        changed = sorted(self.project.overrides.keys() ^ project.overrides.keys())
        self.project = project
        starting = (
            self.settings_for(project)
            if self.settings_for is not None
            else resolve_settings(self.store.overrides() | project.overrides)
        )
        for key in changed:
            field = SETTING_FIELDS[key]
            updated = self.settings.model_dump(exclude_unset=True)
            if field in starting.model_fields_set:
                updated[field] = starting.model_dump(include={field})[field]
            else:
                updated.pop(field, None)
            self.settings = Settings.model_validate(updated)
            if key == 'model' and self.settings.model:
                self.store.add_model(name=self.settings.model)
            self.apply_setting(key, self.settings)

    def setting(self, key: str) -> JsonValue:
        """The effective value of `key`, such as `run.code_mode`, as JSON; `KeyError` for an unknown key."""
        field = SETTING_FIELDS[key]
        return self.settings.model_dump(mode='json', include={field})[field]

    def set_setting(self, args: list[str]) -> str:
        """Validate, persist, and apply a preference to the current conversation."""
        if len(args) == 1 and args[0] in SETTING_FIELDS:
            return str(self.settings.model_dump()[SETTING_FIELDS[args[0]]])
        if len(args) != 2 or args[0] not in SETTING_FIELDS:
            raise ValueError('Usage: /set SETTING VALUE. Press Tab for suggestions.')
        key, raw = args
        with self.edits:
            value, settings = self.validate(key, raw)
            self.store.set(key, value)
            self._apply(key, settings)
        return f'Saved {key}. ' + self._when(key)

    def validate(self, key: str, raw: str) -> tuple[JsonValue, Settings]:
        """Parse typed text for `key` and check it against the whole settings model; nothing is saved.

        A model whose provider SDK is not installed raises `ValueError` here, not on the next turn.
        """
        adapter: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)
        value: JsonValue = (
            raw
            if key in STRING_SETTINGS or (key == 'sessions.naming_model' and raw != 'null')
            else adapter.validate_json(raw)
        )
        # Unset fields stay unset, so a field the user never chose keeps reading as CLAI's default.
        updated = self.settings.model_dump(exclude_unset=True)
        updated[SETTING_FIELDS[key]] = value
        settings = Settings.model_validate(updated)
        if key == 'model' and settings.model:
            from pydantic_clai2.models.model_catalog import check_installed

            check_installed(settings.model)
        return value, settings

    def model_settings(self, model: str) -> ModelSettings | None:
        """Family defaults plus saved overrides: what CLAI applies to a run with `model`."""
        from pydantic_clai2.models.model_settings import model_settings_from_json

        form = model_settings_from_json(self.store.model_settings(model), model=self.settings_model(model))
        return form.to_model_settings()

    def model_overrides(self, model: str) -> ModelSettings | None:
        """The saved overrides alone, for `agent.run(model_settings=...)`, where they beat every other setting."""
        from pydantic_clai2.models.model_settings import model_settings_from_json

        return model_settings_from_json(self.store.model_settings(model)).to_model_settings()

    def live_model_overrides(self, model: str) -> Callable[[object], ModelSettings]:
        """`model_overrides(model)` for a run, read again before each model request.

        Core resolves a callable per request, so edits saved while a turn runs reach its next
        request. Invalid saved settings raise `ValidationError` now, before the run starts; an
        invalid read later keeps the last valid settings instead of failing the running turn.
        """
        current = self.model_overrides(model) or ModelSettings()

        def resolve(_: object) -> ModelSettings:
            nonlocal current
            with suppress(ValidationError):
                current = self.model_overrides(model) or ModelSettings()
            return current

        return resolve

    def model_defaults(self, model: str) -> 'FamilyDefaults':
        """Family defaults for a model name; for `model`, only those its saved overrides leave unset.

        Merged beneath `live_model_overrides(model)`, they make `model_settings(model)`. The saved
        overrides are read again on each call, so a field saved or reset mid-turn moves between the two.
        """
        from pydantic_clai2.models.model_settings import default_model_settings

        return lambda name: default_model_settings(
            model=name, saved=self.store.model_settings(model) if name == model else {}
        )

    def reset_setting(self, key: str) -> str:
        """Forget the saved override and apply the default now."""
        with self.edits:
            self.store.reset(key)
            updated = self.settings.model_dump(exclude_unset=True)
            updated.pop(SETTING_FIELDS[key], None)
            self._apply(key, Settings.model_validate(updated))
        return f'Reset {key}. ' + self._when(key)

    def adopt(self, key: str, source: Settings) -> None:
        """Apply `key` as `source`, another live session's settings, holds it now, unless the project file sets it here."""
        if self.from_project(key):
            return
        field = SETTING_FIELDS[key]
        updated = self.settings.model_dump(exclude_unset=True)
        if field in source.model_fields_set:
            updated[field] = source.model_dump(include={field})[field]
        else:
            updated.pop(field, None)
        self.settings = Settings.model_validate(updated)
        self.apply_setting(key, self.settings)

    def _apply(self, key: str, settings: Settings) -> None:
        self.settings = settings
        self.apply_setting(key, settings)
        telemetry.record('setting {setting} changed', setting=key, value=_shown(key, settings))
        self.saved(key)

    def _when(self, key: str) -> str:
        when = 'Applies at next startup.' if key == 'display.splash' else 'Applied.'
        if self.from_project(key):
            when += ' The project file sets it again at next start.'
        return when


PROJECT_USAGE = 'Usage: /project [approve NAME...|all | revoke NAME...|all]'


def _one_line(path: Path) -> str:
    """A path as inert text: directory names come from the repository, and a line break would forge a row."""
    return terminal_text(str(path), keep='')


def _file_names(keys: Iterable[str]) -> str:
    """Keys as the project file spells them."""
    return ', '.join(SETTING_FIELDS.get(key, key) for key in keys)


def _project_keys(names: list[str], candidates: Collection[str], *, verb: str) -> list[str]:
    """The `/set` keys `names` choose among `candidates`: `all`, or each by its file name or `/set` key."""
    if names == ['all']:
        if not candidates:
            raise ValueError(f'Nothing to {verb}.')
        return sorted(candidates)
    keys = [SETTING_KEYS.get(name, name) for name in names]
    if missing := [name for name, key in zip(names, keys) if key not in candidates]:
        state = 'waiting for approval' if verb == 'approve' else 'approved'
        raise ValueError(f'Not {state} here: {", ".join(missing)}. /project lists them.')
    return list(dict.fromkeys(keys))


def _project_listing(project: ProjectSettings, stored: dict[Path, dict[str, str]]) -> str:
    """What `/project` shows: the file, each key it sets and whether it applies, and approvals it no longer matches."""
    lines = [f'Project settings: {_one_line(project.path)}' if project.path is not None else 'No project file.']
    for key, value in project.overrides.items():
        state = 'approved' if key in project.approved else 'applied'
        lines.append(f'  {SETTING_FIELDS[key]} ({state}): {setting_json(value)}')
    for key, value in project.pending.items():
        lines.append(f'  {SETTING_FIELDS[key]} (waiting for approval): {setting_json(value)}')
    for rows in stored.values():
        lines.extend(
            f'  {SETTING_FIELDS.get(key, key)} (approved earlier, the file does not set this value now): {value}'
            for key, value in rows.items()
            if key not in project.approved
        )
    if project.pending:
        lines.append('/project approve NAME... (or all) applies them here and in later sessions in this repository.')
    # A value's JSON still holds C1 controls.
    return terminal_text('\n'.join(lines))


def _shown(key: str, settings: Settings) -> bool | int | float | str:
    """A setting's new value for UI telemetry: flags, numbers, and listed choices, but never typed text."""
    value: object = getattr(settings, SETTING_FIELDS[key])
    if value is None:
        return 'null'
    if isinstance(value, bool | int | float):
        return value
    return value if isinstance(value, str) and value in set(set_completions([key, ''])) else 'custom'
