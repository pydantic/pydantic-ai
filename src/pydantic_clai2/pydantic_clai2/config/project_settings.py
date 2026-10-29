"""Project-local settings: `.clai/settings.json`, found by walking up from the workspace to the git root."""

from dataclasses import dataclass, field
from pathlib import Path

from pydantic import JsonValue, TypeAdapter

from pydantic_clai2.config import SETTING_FIELDS, PluginSettings, resolve_settings
from pydantic_clai2.config.hooks import HOOKS_FACTORY, PROJECT_HOOKS, HookTable

PROJECT_FILE = Path('.clai') / 'settings.json'
"""The file name, relative to any directory between the workspace and the git root."""

_FILE: TypeAdapter[dict[str, JsonValue]] = TypeAdapter(dict[str, JsonValue])
_PLUGINS: TypeAdapter[list[PluginSettings]] = TypeAdapter(list[PluginSettings])
_HOOKS: TypeAdapter[HookTable] = TypeAdapter(HookTable)
_SET_KEYS = {name: key for key, name in SETTING_FIELDS.items()}
"""`Settings` field name to `/set` key, so the file uses the model's own names."""


@dataclass(frozen=True, kw_only=True)
class ProjectSettings:
    """What the project file contributes. The default instance is what "no file" looks like."""

    path: Path | None = None
    overrides: dict[str, JsonValue] = field(default_factory=dict[str, JsonValue])
    """Validated setting values by `/set` key, ready to layer over the user store."""
    plugins: tuple[PluginSettings, ...] = ()
    """Declarations from the file, every one disabled: the user approves each with `/plugins enable`.

    A relative `path` is already made absolute against the project root, the folder holding `.clai`.
    A `hooks` table is one more, `project_hooks`, which runs it from that root.
    """
    unknown: tuple[str, ...] = ()
    """Keys the `Settings` model does not know; reported once at startup and ignored."""


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


def load_project_settings(workspace: Path) -> ProjectSettings:
    """Read and validate the project file for `workspace`.

    An unreadable file or a bad value fails startup with the path in the message; an unknown key does not.
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
        overrides = {_SET_KEYS[name]: value for name, value in raw.items() if name in _SET_KEYS}
        resolve_settings(overrides)
    except (OSError, ValueError) as exc:
        raise ValueError(f'{path}: {exc}') from exc
    root = path.parent.parent
    return ProjectSettings(
        path=path,
        overrides=overrides,
        plugins=(*(_unapproved(plugin, root=root) for plugin in plugins), *_hooks(hooks, root=root)),
        unknown=tuple(sorted(raw.keys() - _SET_KEYS.keys())),
    )
