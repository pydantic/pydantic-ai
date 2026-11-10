"""Project-local settings: `.clai/settings.json`, found by walking up from the workspace to the git root."""

from dataclasses import dataclass, field, replace
from pathlib import Path

from pydantic import JsonValue, TypeAdapter

from pydantic_clai2.config import (
    PERMISSION_MODES,
    PERMISSIONS_SETTING,
    SETTING_FIELDS,
    SETTING_KEYS,
    USER_ONLY_SETTINGS,
    PluginSettings,
    resolve_settings,
)
from pydantic_clai2.config.hooks import HOOKS_FACTORY, PROJECT_HOOKS, HookTable
from pydantic_clai2.config.settings_store import SettingsStore, setting_json

PROJECT_FILE = Path('.clai') / 'settings.json'
"""The file name, relative to any directory between the workspace and the git root."""

_FILE: TypeAdapter[dict[str, JsonValue]] = TypeAdapter(dict[str, JsonValue])
_PLUGINS: TypeAdapter[list[PluginSettings]] = TypeAdapter(list[PluginSettings])
_HOOKS: TypeAdapter[HookTable] = TypeAdapter(HookTable)
SAFE_PROJECT_KEYS = frozenset(key for key in SETTING_FIELDS if key.startswith('display.'))
"""Keys a project file sets at once: how CLAI shows things, which cannot change what the agent may do or is told.

Every other key a project file may set waits for `/project approve`: it grants permissions, adds
instructions, picks the model or account, or configures how runs execute. A new key waits too unless
it is added here. The exception is a `run.permissions` no wider than your own, which applies at once.
"""


@dataclass(frozen=True, kw_only=True)
class ProjectSettings:
    """What the project file contributes. The default instance is what "no file" looks like."""

    path: Path | None = None
    overrides: dict[str, JsonValue] = field(default_factory=dict[str, JsonValue])
    """Validated setting values by `/set` key, ready to layer over the user store.

    The `SAFE_PROJECT_KEYS`, the values `/project approve` approved for this project's root, and a
    `run.permissions` no wider than your own.
    """
    pending: dict[str, JsonValue] = field(default_factory=dict[str, JsonValue])
    """Validated values by `/set` key that are not applied: no approval for this root, or one of another value."""
    approved: frozenset[str] = frozenset()
    """The `/set` keys `/project approve` approved for this project's root with the value the file sets now."""
    plugins: tuple[PluginSettings, ...] = ()
    """Declarations from the file, every one disabled: the user approves each with `/plugins enable`.

    A relative `path` is already made absolute against the project root, the folder holding `.clai`.
    A `hooks` table is one more, `project_hooks`, which runs it from that root.
    """
    unknown: tuple[str, ...] = ()
    """Keys the `Settings` model does not know; reported once at startup and ignored."""

    @property
    def root(self) -> Path | None:
        """The folder holding `.clai`, which approvals are kept for; `None` without a file."""
        return None if self.path is None else self.path.parent.parent

    @property
    def pending_instructions(self) -> str | None:
        """The file's `instructions` while they wait for approval, which CLAI sends as the repository's, not yours."""
        text = self.pending.get('run.instructions')
        return text if isinstance(text, str) and text else None

    @property
    def pending_notice(self) -> str | None:
        """The startup notice naming the `pending` keys, or `None` when every key applies."""
        if not self.pending:
            return None
        names = ', '.join(SETTING_FIELDS[key] for key in self.pending)
        return f'Project settings not applied until you approve them with /project approve in a CLAI session: {names}'


def _present(path: Path) -> bool:
    """Like `exists()`, except a permission failure counts as present: it is a boundary, not an absence."""
    try:
        path.stat()
    except (FileNotFoundError, NotADirectoryError):
        return False
    except OSError:
        return True
    return True


def find_project_file(workspace: Path, name: Path = PROJECT_FILE) -> Path | None:
    """The nearest `name` at or above `workspace`. The search stops at the first `.git`.

    An unreadable project file is still returned, so loading it fails loudly rather than silently
    dropping the repository's settings; an unreadable `.git` still ends the walk.
    """
    start = workspace.resolve()
    for directory in (start, *start.parents):
        candidate = directory / name
        if _present(candidate):
            return candidate
        if _present(directory / '.git'):
            return None
    return None


def repository_directories(workspace: Path) -> list[Path]:
    """`workspace` and its parents up to the nearest git root, nearest first; only `workspace` outside a repository.

    Like `find_project_file`, the walk stops at the first `.git`, and an unreadable one still ends it.
    """
    start = workspace.resolve()
    walked: list[Path] = []
    for directory in (start, *start.parents):
        walked.append(directory)
        if _present(directory / '.git'):
            return walked
    return [start]


def _unapproved(plugin: PluginSettings, *, root: Path) -> PluginSettings:
    """`plugin` switched off, with its file anchored to the project `root`.

    Approval saves the declaration to the user store, which every repository shares. A path left
    relative there would resolve against whichever directory CLAI starts in next, so another
    repository could put its own file at that path and have it run under this approval.
    """
    update: dict[str, object] = {'enabled': False}
    if plugin.path is not None:
        update['path'] = str(root / plugin.path)
    return plugin.model_copy(update=update)


def _hooks(table: HookTable, *, root: Path) -> tuple[PluginSettings, ...]:
    """The `project_hooks` declaration running `table` from `root`, switched off, or none for an empty table."""
    settings = table.model_dump(mode='json', exclude_defaults=True)
    if not settings:
        return ()
    settings['project'] = str(root)
    return (PluginSettings(id=PROJECT_HOOKS, factory=HOOKS_FACTORY, enabled=False, settings=settings),)


def with_current_approvals(project: ProjectSettings, *, store: SettingsStore | None) -> ProjectSettings:
    """`project` with its keys split again by what `store` approves now; the file is not read again.

    Without a `store`, nothing is approved. Without a file, there is nothing to approve, so `project` stays as it is.
    """
    root = project.root
    if root is None:
        return project
    rows = {} if store is None else store.approved_project_settings(root)
    declared = {**project.overrides, **project.pending}
    approved = frozenset(key for key, value in declared.items() if rows.get(key) == setting_json(value))
    # Later modes ask for more, so one at or after yours cannot widen what the agent may do.
    no_wider = (
        store is not None
        and PERMISSIONS_SETTING in declared
        and PERMISSION_MODES.index(declared[PERMISSIONS_SETTING]) >= PERMISSION_MODES.index(store.load().permissions)
    )
    applied = {
        key: value
        for key, value in declared.items()
        if key in SAFE_PROJECT_KEYS or key in approved or (no_wider and key == PERMISSIONS_SETTING)
    }
    return replace(
        project,
        overrides=applied,
        pending={key: value for key, value in declared.items() if key not in applied},
        approved=approved,
    )


def load_project_settings(workspace: Path, *, store: SettingsStore | None = None) -> ProjectSettings:
    """Read and validate the project file for `workspace`.

    An unreadable file or a bad value fails startup with the path in the message; an unknown key does not.
    A key outside `SAFE_PROJECT_KEYS` applies only with the value `store` approved for the project's
    root, or as a `run.permissions` no wider than `store` holds; without a `store`, none does.
    """
    path = find_project_file(workspace)
    if path is None:
        return ProjectSettings()
    try:
        raw = _FILE.validate_json(path.read_bytes())
        plugins = _PLUGINS.validate_python(raw.pop('plugins', []))
        hooks = _HOOKS.validate_python(raw.pop('hooks', {}))
        if any(plugin.id == PROJECT_HOOKS or plugin.factory.partition(':')[0] == HOOKS_FACTORY for plugin in plugins):
            raise ValueError(
                f'declare hooks with the `hooks` key, not as a plugin ({PROJECT_HOOKS} or {HOOKS_FACTORY})'
            )
        for name in sorted(USER_ONLY_SETTINGS):
            if name in raw:
                raise ValueError(f'`{name}` is user-only and cannot be set in a project file')
        overrides = {SETTING_KEYS[name]: value for name, value in raw.items() if name in SETTING_KEYS}
        resolve_settings(overrides)
    except (OSError, ValueError) as exc:
        raise ValueError(f'{path}: {exc}') from exc
    root = path.parent.parent
    project = ProjectSettings(
        path=path,
        pending=overrides,
        plugins=(*(_unapproved(plugin, root=root) for plugin in plugins), *_hooks(hooks, root=root)),
        unknown=tuple(sorted(raw.keys() - SETTING_KEYS.keys())),
    )
    return with_current_approvals(project, store=store)
