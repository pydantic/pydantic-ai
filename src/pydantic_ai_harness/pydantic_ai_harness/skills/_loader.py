"""Discover and parse Agent Skill packages from libraries in the run's workspace."""

from __future__ import annotations

import posixpath
import re
import unicodedata
from collections.abc import Collection, Hashable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from pydantic_ai.workspaces import FileEntry, Workspace
from pydantic_ai_harness._workspace import stat_or_none, workspace_path
from pydantic_ai_harness.artifact_scanning import ArtifactScanResult

# Imported with the module rather than on first parse: a durable engine such as Temporal parses skills in
# workflow code, where importing a module for the first time fails the workflow task.
try:
    import yaml
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'PyYAML is required to load Agent Skills. Install it with: pip install "pydantic-ai-harness[skills]"'
    ) from _import_error

# These fields affect invocation, permissions, model selection, execution, or
# prompt rendering in clients that implement them. Skills accepts their files
# for compatibility but reports that the behavior is not active.
_BEHAVIORAL_FRONTMATTER_FIELDS = frozenset(
    {
        'agent',
        'allowed-tools',
        'argument-hint',
        'arguments',
        'context',
        'dependencies',
        'disallowed-tools',
        'effort',
        'hooks',
        'model',
        'paths',
        'shell',
        'tools',
    }
)


class _SkillFrontmatter(BaseModel):
    model_config = ConfigDict(extra='allow')

    name: str | None = None
    description: str
    when_to_use: str | None = None
    disable_model_invocation: bool = Field(default=False, alias='disable-model-invocation')
    user_invocable: bool = Field(default=True, alias='user-invocable')

    @field_validator('disable_model_invocation', 'user_invocable', mode='before')
    @classmethod
    def _yaml_boolean(cls, value: object) -> object:
        # Frontmatter is loaded without implicit types, so a YAML boolean arrives as text. Only YAML 1.2's
        # spellings count: `yes` or `1` failing the skill is safer than guessing for a field that gates invocation.
        if isinstance(value, str) and value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        raise ValueError(f'must be `true` or `false`, not {value!r}')

    @field_validator('description', mode='after')
    @classmethod
    def _strip_description(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError('must not be empty')
        return stripped

    @field_validator('when_to_use', mode='after')
    @classmethod
    def _strip_when_to_use(cls, value: str) -> str | None:
        # Frontmatter is loaded without implicit types, so a set field is always text. An empty trigger has
        # nothing to append to the catalog description.
        return value.strip() or None


MissingDirectories: TypeAlias = Literal['error', 'skip']
"""What `Skills` does with a library path that is not a library: fail the run, or skip it.

A path is not a library when it does not exist, is not a directory, is itself a skill package, or cannot be read
(the workspace raises `PermissionError` for it).
"""

DuplicateNames: TypeAlias = Literal['error', 'keep_first']
"""What `Skills` does when two valid skills with different `SKILL.md` files share a name.

Fail the run, or keep the first with a warning.
"""

UnknownNames: TypeAlias = Literal['error', 'skip']
"""What `Skills` does with an `include` or `exclude` name that no library has: fail the run, or skip the name.

Skipping suits a selection shared by several projects, such as a person's list of skills to hide.
"""

_ARGUMENTS_PLACEHOLDER = re.compile(r'\$ARGUMENTS(?![\[\w])')
"""`$ARGUMENTS` itself, not Claude Code's indexed `$ARGUMENTS[0]` or a longer name such as `$ARGUMENTS_LIST`."""

_PLACEHOLDERS = re.compile(rf'{_ARGUMENTS_PLACEHOLDER.pattern}|\$\{{(?:CLAUDE_)?SKILL_DIR\}}')
"""`$ARGUMENTS`, Claude Code's `${CLAUDE_SKILL_DIR}`, and the `${SKILL_DIR}` other clients use."""


@dataclass(frozen=True, kw_only=True)
class SkillDefinition:
    """Validated skill metadata and body from a single `SKILL.md`."""

    name: str
    """The skill's name: its `name` frontmatter field, or else its directory's name."""

    description: str
    """The `description` frontmatter field, listed in the model's catalog."""

    when_to_use: str | None = None
    """The `when_to_use` frontmatter field, appended to the catalog description."""

    body: str
    """The Markdown after the frontmatter, without leading and trailing blank lines."""

    ignored_behavioral_fields: tuple[str, ...]
    """Frontmatter fields such as `allowed-tools` whose behavior `Skills` does not implement."""

    model_invocable: bool = True
    """Whether the model may load it; `False` when its frontmatter has `disable-model-invocation: true`.

    A run leaves such a skill out of the model's catalog. `Skills.load` still returns it, for a host
    that lets a person invoke it.
    """

    user_invocable: bool = True
    """Whether a host should let a person invoke it; `False` when its frontmatter has `user-invocable: false`.

    `Skills` itself only offers skills to the model, so this is for a host, such as one with a `/skill-name` command.
    """

    path: str
    """Absolute workspace path of the `SKILL.md` it was read from."""

    in_run_workspace: bool = True
    """Whether it was read from the run's workspace, where the model's file and shell tools can reach its directory.

    `False` for a library read from `Skills(workspace=...)`; `render` then leaves the directory out.
    """

    @property
    def directory(self) -> str:
        """Absolute workspace path of the skill's directory, which holds any `scripts/` or `references/`."""
        return posixpath.dirname(self.path)

    def render(self, arguments: str | None = None) -> str:
        """The skill's instructions as the model receives them: a heading, the skill's directory, and the body.

        Pass `arguments` when a user invokes the skill with a command such as `/code-review src/app.py`:
        every `$ARGUMENTS` in the body becomes `arguments`, and a body without `$ARGUMENTS` gets
        `ARGUMENTS: <arguments>` appended. Other placeholders, such as Claude Code's indexed `$0` and
        `$ARGUMENTS[0]`, are left unchanged. Without `arguments`, the result is the instructions the
        model gets when it loads the skill.

        When `in_run_workspace`, `${CLAUDE_SKILL_DIR}` and `${SKILL_DIR}` in the body become the skill's
        directory. Otherwise they are left unchanged, and the directory line is left out.
        """
        directory = self.directory if self.in_run_workspace else None

        # One pass over the original body, so a value put in for one placeholder is not read as another.
        def substitute(match: re.Match[str]) -> str:
            value = arguments if match.group() == '$ARGUMENTS' else directory
            return match.group() if value is None else value

        body = _PLACEHOLDERS.sub(substitute, self.body)
        if arguments and not _ARGUMENTS_PLACEHOLDER.search(self.body):
            body = '\n\n'.join(part for part in (body, f'ARGUMENTS: {arguments}') if part)
        heading = f'# Skill: {self.name}'
        if self.in_run_workspace:
            heading += f'\n\nSkill directory: `{self.directory}`. Relative paths in this skill resolve against it.'
        return f'{heading}\n\n{body}' if body else heading


SkipReason: TypeAlias = Literal[
    'invalid', 'duplicate_name', 'not_a_directory', 'skill_package', 'unreadable', 'flagged'
]
"""Why `Skills` left a path out of the catalog.

- `'invalid'`: a `SKILL.md` with malformed frontmatter, invalid UTF-8, or an invalid or mismatched name.
- `'duplicate_name'`: a `SKILL.md` named like a valid skill found earlier, with `duplicate_names='keep_first'`.
- `'not_a_directory'`: a library path that is a file, unless `missing_directories='error'`.
- `'skill_package'`: a library path that holds a `SKILL.md` itself, unless `missing_directories='error'`.
- `'unreadable'`: a skill directory or `SKILL.md` the workspace raises `PermissionError` for, skipped like an invalid
  `SKILL.md`, or such a library directory, such as a local one without permission, unless
  `missing_directories='error'`.
- `'flagged'`: a skill whose description, `when_to_use` or body the `Skills(scanner=...)` artifact scan flagged.
"""


@dataclass(frozen=True, kw_only=True)
class SkippedSkill:
    """A `SKILL.md` or skill library path that `Skills` left out, and why.

    A skipped library path, such as one that is not a directory, skips every skill it would have
    held, so it is an entry too, with the library's `path`.
    """

    path: str
    """Absolute workspace path of the skipped `SKILL.md`, or of the skipped library."""

    reason: SkipReason
    """Why it was left out."""

    message: str
    """A sentence naming the path and the problem, as a run's `UserWarning` says it."""

    scan: ArtifactScanResult | None = None
    """The artifact scan's findings when `reason` is `'flagged'`, otherwise `None`."""

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, kw_only=True)
class SkillCatalog:
    """The skills `Skills.load` read, each `SKILL.md` or library it left out, and advice about the skills it read."""

    skills: tuple[SkillDefinition, ...]
    """The selected, valid skills, in catalog order, including those the model may not load."""

    skipped: tuple[SkippedSkill, ...]
    """Each `SKILL.md` or library path left out, in the order it was found.

    A run emits each `message` as a `UserWarning`; a host showing skills to a person can show them instead.
    """

    advice: Sequence[str] = ()
    """Advice about the skills read, such as descriptions over the Agent Skills length limit or frontmatter
    fields whose behavior is not implemented.

    A run emits each as a `UserWarning`; a host that hides warnings can show them instead.
    """


def _extract_frontmatter(text: str, source: str) -> tuple[str, str]:
    lines = text.splitlines()
    if not lines or lines[0] != '---':
        raise ValueError(f'{source} must start with YAML frontmatter delimited by `---`.')

    closing = next((index for index, line in enumerate(lines[1:], start=1) if line == '---'), None)
    if closing is None:
        raise ValueError(f'{source} has unclosed YAML frontmatter.')

    frontmatter = '\n'.join(lines[1:closing])
    body_lines = lines[closing + 1 :]
    while body_lines and not body_lines[0].strip():
        body_lines.pop(0)
    while body_lines and not body_lines[-1].strip():
        body_lines.pop()
    body = '\n'.join(body_lines)
    return frontmatter, body


def _parse_frontmatter(frontmatter: str, source: str) -> _SkillFrontmatter:
    # Agent Skills frontmatter fields are strings. BaseLoader preserves valid
    # scalar names such as `123` and `on` instead of applying YAML implicit types.
    # PyYAML otherwise also accepts duplicate mapping keys and keeps the last value.
    class UniqueKeyLoader(yaml.BaseLoader):
        def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Hashable, object]:
            keys: set[str] = set()
            for key_node, _ in node.value:
                if not isinstance(key_node, yaml.ScalarNode):
                    raise yaml.constructor.ConstructorError(
                        'while constructing a mapping',
                        node.start_mark,
                        'found a non-scalar key',
                        key_node.start_mark,
                    )
                key: str = key_node.value
                if key in keys:
                    raise yaml.constructor.ConstructorError(
                        'while constructing a mapping',
                        node.start_mark,
                        f'found duplicate key {key!r}',
                        key_node.start_mark,
                    )
                keys.add(key)
            return super().construct_mapping(node, deep=deep)

    try:
        parsed: object = yaml.load(frontmatter, Loader=UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f'Invalid YAML frontmatter in {source}: {exc}') from exc

    if not isinstance(parsed, dict):
        raise ValueError(f'YAML frontmatter in {source} must be a mapping.')
    try:
        return _SkillFrontmatter.model_validate(parsed)
    except ValidationError as exc:
        raise ValueError(f'Invalid Agent Skill frontmatter in {source}: {exc}') from exc


def _normalize_name(name: str) -> str:
    return unicodedata.normalize('NFKC', name)


async def _is_file(workspace: Workspace, path: str) -> bool:
    entry = await stat_or_none(workspace, path)
    return entry is not None and not entry.is_dir


async def _discover_skills(
    workspace: Workspace, libraries: Sequence[tuple[str, Sequence[FileEntry]]]
) -> tuple[list[tuple[str, str]], list[SkippedSkill]]:
    """Each skill package's name and `SKILL.md`, and each skill directory left out because it cannot be read."""
    discovered: list[tuple[str, str]] = []
    skipped: list[SkippedSkill] = []
    for library, children in libraries:
        # Sorted by name so the catalog, and so the instructions, are the same for every run over the same files.
        for child in sorted(children, key=lambda entry: entry.name):
            if not child.is_dir:
                continue
            directory = posixpath.join(library, child.name)
            skill_file = posixpath.join(directory, 'SKILL.md')
            try:
                is_skill = await _is_file(workspace, skill_file)
            except PermissionError:
                # Like an invalid `SKILL.md`, a skill the workspace denies reading must not make the others unusable.
                message = f'Skipping {directory}: it is not readable.'
                skipped.append(SkippedSkill(path=directory, reason='unreadable', message=message))
                continue
            if is_skill:
                discovered.append((_normalize_name(child.name), skill_file))
    return discovered, skipped


async def _libraries(
    workspace: Workspace, directories: Sequence[str | Path], missing_directories: MissingDirectories
) -> tuple[list[tuple[str, Sequence[FileEntry]]], list[SkippedSkill]]:
    """The configured libraries that exist, in order, each listed once with its entries, and the paths skipped."""
    libraries: list[tuple[str, Sequence[FileEntry]]] = []
    skipped: list[SkippedSkill] = []
    seen: set[str] = set()
    for configured in directories:
        library = await workspace.resolve(workspace_path(Path(configured)))
        if library in seen:
            continue
        seen.add(library)
        try:
            entry, children = await _read_library(workspace, library)
        except PermissionError:
            # Permissions can change mid-session, so with 'skip' it must not fail every later run. A shell-backed
            # workspace raises this only when `stat` is denied; a listing it is denied raises `WorkspaceError`.
            if missing_directories == 'error':
                raise
            message = f'Skipping {library}: it is not readable.'
            skipped.append(SkippedSkill(path=library, reason='unreadable', message=message))
            continue
        if entry is None:
            if missing_directories == 'skip':
                continue
            raise ValueError(f'Skill library directory does not exist in the workspace: {configured}')
        if children is not None:
            libraries.append((library, children))
            continue
        skipped.append(
            _skip_non_library(configured, library, is_dir=entry.is_dir, missing_directories=missing_directories)
        )
    return libraries, skipped


async def _read_library(workspace: Workspace, library: str) -> tuple[FileEntry | None, Sequence[FileEntry] | None]:
    """The entry at `library`, and its children when it is a library: a directory without a `SKILL.md` of its own."""
    entry = await stat_or_none(workspace, library)
    if entry is None or not entry.is_dir or await _is_file(workspace, posixpath.join(library, 'SKILL.md')):
        return entry, None
    return entry, await workspace.list_dir(library)


def _skip_non_library(
    configured: str | Path, library: str, *, is_dir: bool, missing_directories: MissingDirectories
) -> SkippedSkill:
    """The entry for `library`, a skill package or a file; raises unless `missing_directories` is `'skip'`."""
    # The model's tools can create either mid-session, so with 'skip' they must not fail every later run.
    if missing_directories == 'skip':
        reason: SkipReason = 'skill_package' if is_dir else 'not_a_directory'
        problem = 'it is a skill package, not a library' if is_dir else 'it is not a directory'
        return SkippedSkill(path=library, reason=reason, message=f'Skipping {library}: {problem}.')
    if is_dir:
        raise ValueError(
            f'Skill library path points to a skill package: {configured}. Pass its parent directory instead.'
        )
    raise ValueError(f'Skill library path is not a directory: {configured}')


async def same_skill(first: tuple[Workspace, str], second: tuple[Workspace, str]) -> bool:
    """Whether two `SKILL.md` files, each in its workspace, hold one skill: the same file, or a byte-identical copy.

    The same file is recognized through symlinks within one workspace. Only `SKILL.md` is compared,
    not bundled files. Asked only when two skills share a name, so a catalog without clashes costs
    no extra round trips.
    """
    (first_workspace, first_path), (second_workspace, second_path) = first, second
    if first_workspace is second_workspace and await first_workspace.realpath(
        first_path
    ) == await second_workspace.realpath(second_path):
        return True
    return await first_workspace.read_bytes(first_path) == await second_workspace.read_bytes(second_path)


def skip_duplicate_name(name: str, kept: str, skipped: str, duplicate_names: DuplicateNames) -> SkippedSkill:
    """The entry for `skipped`, whose name an earlier `SKILL.md`, `kept`, has; raises when duplicates are errors."""
    if duplicate_names == 'error':
        raise ValueError(f'Duplicate skill name {name!r}: {kept} and {skipped}.')
    return SkippedSkill(
        path=skipped,
        reason='duplicate_name',
        message=f'Skipping {skipped}: skill name {name!r} is already taken by {kept}.',
    )


def _validate_name(name: str, source: str) -> str:
    normalized = _normalize_name(name)
    if (
        not normalized
        or len(normalized) > 64
        or normalized != normalized.lower()
        or normalized.startswith('-')
        or normalized.endswith('-')
        or '--' in normalized
        or not all(character.isalnum() or character == '-' for character in normalized)
    ):
        raise ValueError(
            f'Invalid skill name {name!r} in {source}; expected at most 64 lowercase Unicode letters or numbers '
            'and single hyphens, without a leading or trailing hyphen.'
        )
    return normalized


def parse_skill(text: str, skill_file: str) -> SkillDefinition:
    """Parse one `SKILL.md`, read from `skill_file`, into a validated definition."""
    frontmatter_text, body = _extract_frontmatter(text, skill_file)
    frontmatter = _parse_frontmatter(frontmatter_text, skill_file)

    directory_name = posixpath.basename(posixpath.dirname(skill_file))
    name = frontmatter.name if frontmatter.name is not None else directory_name
    normalized_name = _validate_name(name, skill_file)
    if frontmatter.name is not None and normalized_name != _normalize_name(directory_name):
        raise ValueError(f'Skill name {name!r} in {skill_file} must match its parent directory {directory_name!r}.')

    ignored_fields = tuple(sorted(_BEHAVIORAL_FRONTMATTER_FIELDS.intersection(frontmatter.model_extra or {})))
    return SkillDefinition(
        name=normalized_name,
        description=frontmatter.description,
        when_to_use=frontmatter.when_to_use,
        body=body,
        ignored_behavioral_fields=ignored_fields,
        model_invocable=not frontmatter.disable_model_invocation,
        user_invocable=frontmatter.user_invocable,
        path=skill_file,
    )


def catalog_description(skill: SkillDefinition) -> str:
    """The text the model is shown for `skill`: its description, then its `when_to_use`.

    The same in the catalog and from `list_skills`.
    """
    if skill.when_to_use is None:
        return skill.description
    # Appended verbatim, Claude Code style: the author writes the trigger clause, e.g. 'Use when ...'.
    return f'{skill.description} {skill.when_to_use}'


async def load_skill_libraries(
    workspace: Workspace,
    directories: Sequence[str | Path],
    *,
    include: Collection[str] | None,
    exclude: Collection[str],
    missing_directories: MissingDirectories,
    duplicate_names: DuplicateNames,
    unknown_names: UnknownNames,
) -> tuple[list[SkillDefinition], list[SkippedSkill]]:
    """Discover immediate child skill packages under configured directories in `workspace`.

    Relative directories resolve against the workspace's working directory. Returns the parsed
    skills and an entry for each skipped `SKILL.md` or library path. A skill found twice, through a symlinked
    library or skill directory or as a byte-identical `SKILL.md`, counts once. Names are compared
    after parsing, so an invalid `SKILL.md` does not hide a valid one with the same name.
    """
    libraries, skipped = await _libraries(workspace, directories, missing_directories)
    discovered, unreadable = await _discover_skills(workspace, libraries)
    # An unreadable skill directory still names a skill, so selecting it reports it as skipped, not unknown.
    unreadable_by_name = {_normalize_name(posixpath.basename(entry.path)): entry for entry in unreadable}
    available_names = frozenset(name for name, _ in discovered).union(unreadable_by_name)
    selected_names = _selected_names(
        available_names, include=include, exclude=exclude, unknown_names=unknown_names, skipped=[*skipped, *unreadable]
    )
    # Like an invalid `SKILL.md` below, an unreadable skill the user left out stays quiet.
    skipped.extend(entry for name, entry in unreadable_by_name.items() if name in selected_names)

    by_name: dict[str, SkillDefinition] = {}
    for name, skill_file in discovered:
        if name not in selected_names:
            continue
        skill = await _read_or_skip(workspace, skill_file)
        if isinstance(skill, SkippedSkill):
            skipped.append(skill)
            continue
        if (previous := by_name.get(skill.name)) is None:
            by_name[skill.name] = skill
        elif not await same_skill((workspace, previous.path), (workspace, skill_file)):
            skipped.append(skip_duplicate_name(skill.name, previous.path, skill_file, duplicate_names))
    return list(by_name.values()), skipped


def _selected_names(
    available_names: frozenset[str],
    *,
    include: Collection[str] | None,
    exclude: Collection[str],
    unknown_names: UnknownNames,
    skipped: Sequence[SkippedSkill],
) -> frozenset[str]:
    """The normalized names `include` and `exclude` select from `available_names`, validated per `unknown_names`."""
    normalized_include = None if include is None else frozenset(_normalize_name(name) for name in include)
    normalized_exclude = frozenset(_normalize_name(name) for name in exclude)
    if unknown_names == 'error':
        _validate_selection('include', normalized_include, available_names, skipped)
        _validate_selection('exclude', normalized_exclude, available_names, skipped)
    return normalized_include if normalized_include is not None else available_names.difference(normalized_exclude)


async def _read_or_skip(workspace: Workspace, skill_file: str) -> SkillDefinition | SkippedSkill:
    """The skill at `skill_file`, or the entry that leaves it out because it is invalid or cannot be read.

    A model-editable skill should not make unrelated valid skills unusable.
    """
    try:
        return await _read_skill(workspace, skill_file)
    except ValueError as error:
        return SkippedSkill(path=skill_file, reason='invalid', message=f'Skipping {skill_file}: {error}')
    except PermissionError:
        return SkippedSkill(path=skill_file, reason='unreadable', message=f'Skipping {skill_file}: it is not readable.')


async def _read_skill(workspace: Workspace, skill_file: str) -> SkillDefinition:
    try:
        text = await workspace.read_text(skill_file)
    except UnicodeDecodeError as error:
        raise ValueError(f'{skill_file} is not valid UTF-8: {error}') from error
    return parse_skill(text, skill_file)


def _validate_selection(
    name: str,
    selected: Collection[str] | None,
    available: Collection[str],
    skipped: Sequence[SkippedSkill] = (),
) -> None:
    if selected is None:
        return
    unknown = sorted(set(selected).difference(available))
    if not unknown:
        return
    noun = 'skill' if len(unknown) == 1 else 'skills'
    available_text = ', '.join(sorted(available)) or '(none)'
    message = f'Unknown {noun} in {name}: {", ".join(unknown)}. Available skills: {available_text}.'
    if skipped:
        # A skipped library or skill may be where the name is, so its reason is the likelier cause.
        message += ' Some paths were skipped: ' + ' '.join(entry.message for entry in skipped)
    raise ValueError(message)
