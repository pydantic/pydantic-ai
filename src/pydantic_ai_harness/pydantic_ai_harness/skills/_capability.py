"""Load Agent Skill instructions from the run's workspace as deferred capabilities."""

from __future__ import annotations

import warnings
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import get_args, overload

from pydantic_ai._utils import replace_no_init
from pydantic_ai.capabilities import AbstractCapability, CombinedCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.workspaces import Workspace, WorkspaceBackend
from pydantic_ai_harness._workspace import require_workspace, secondary_workspace
from pydantic_ai_harness.skills._loader import (
    DuplicateNames,
    MissingDirectories,
    SkillCatalog,
    SkillDefinition,
    load_skill_libraries,
    same_skill,
    skip_duplicate_name,
)

_MAX_DESCRIPTION_LENGTH = 1024


@dataclass(frozen=True)
class _SkillSource:
    """One `Skills(...)` configuration: libraries, the selection applied to them, and where they live."""

    directories: tuple[str | Path, ...]
    include: frozenset[str] | None
    exclude: frozenset[str]
    missing_directories: MissingDirectories
    duplicate_names: DuplicateNames
    workspace: Workspace | None
    """The `workspace=` the libraries are read from, or `None` for the run's workspace."""


class _Skill(AbstractCapability[AgentDepsT]):
    """One skill: its instructions, loaded by the model with `load_capability`.

    Instructions only, with no toolset, so a durable engine accepts it although it is built per run.
    """

    def __init__(self, skill: SkillDefinition) -> None:
        self.id = skill.name
        # Continuation lines are indented so a multiline description doesn't read as separate catalog entries.
        self.description = skill.description.replace('\n', '\n  ')
        self.defer_loading = True
        self.instructions = skill.render()

    def get_instructions(self) -> str:
        return self.instructions


@dataclass(init=False, repr=False)
class Skills(AbstractCapability[AgentDepsT]):
    """Offer Agent Skill instructions from the run's workspace as deferred capabilities.

    Skill libraries are directories in the run's workspace (`ctx.workspace`), read at the
    start of every run; relative paths resolve against its working directory. Attach
    `LocalWorkspace` to read directories on this machine, or pass `workspace=` to read them
    from a workspace of their own. A run with neither fails at its start.

    Each selected immediate child containing `SKILL.md` becomes a deferred capability named after
    the skill: the model sees its name and description, and loads its Markdown body, with the
    skill's directory in the run's workspace, with `load_capability`. Bundled files are not loaded
    or executed. Descriptions longer than the Agent Skills limit are preserved and emit a warning.
    A skill found twice, through a symlink or as a byte-identical `SKILL.md`, counts once.

    `load` reads the same skills outside a run, for a host that lets a person invoke a skill
    itself; render one with `SkillDefinition.render`.

    Two `Skills` on one agent combine, so every library either names stays reachable.
    """

    directories: tuple[str | Path, ...]
    """Skill-library paths in the workspace, read at the start of each run."""

    include: frozenset[str] | None
    """Exact skill names to expose, or `None` to expose all discovered skills."""

    exclude: frozenset[str]
    """Exact skill names to omit from the catalog."""

    missing_directories: MissingDirectories
    """`'error'` fails the run on a library directory that does not exist; `'skip'` leaves it out."""

    duplicate_names: DuplicateNames
    """`'error'` fails the run when two different `SKILL.md` files share a name; `'keep_first'` keeps the first."""

    workspace: WorkspaceBackend | None
    """Where the libraries live, when not in the run's workspace; see `__init__`."""

    id: str | None = 'skills'
    """One per agent: two `Skills` combine into one catalog."""

    _sources: tuple[_SkillSource, ...] = field(default=(), init=False, repr=False, compare=False)
    """Every configuration this instance serves: its own, plus those of any `Skills` combined into it."""

    @overload
    def __init__(  # pragma: no cover - overload is enforced by static type checking
        self,
        directories: str | Path | Sequence[str | Path],
        *,
        include: Collection[str],
        exclude: None = None,
        missing_directories: MissingDirectories = 'error',
        duplicate_names: DuplicateNames = 'error',
        workspace: WorkspaceBackend | None = None,
    ) -> None: ...

    @overload
    def __init__(  # pragma: no cover - overload is enforced by static type checking
        self,
        directories: str | Path | Sequence[str | Path],
        *,
        include: None = None,
        exclude: Collection[str] | None = None,
        missing_directories: MissingDirectories = 'error',
        duplicate_names: DuplicateNames = 'error',
        workspace: WorkspaceBackend | None = None,
    ) -> None: ...

    def __init__(
        self,
        directories: str | Path | Sequence[str | Path],
        *,
        include: Collection[str] | None = None,
        exclude: Collection[str] | None = None,
        missing_directories: MissingDirectories = 'error',
        duplicate_names: DuplicateNames = 'error',
        workspace: WorkspaceBackend | None = None,
    ) -> None:
        """Configure the skill libraries to read at the start of each run.

        Args:
            directories: One skill-library path or a sequence of paths in the workspace, in precedence
                order. `~` is not expanded: the workspace may be a sandbox with a home of its own.
            include: Exact names to expose. Omit to expose all discovered skills.
            exclude: Exact names to omit. Cannot be combined with `include`.
            missing_directories: `'error'` (the default) fails the run when a library directory does
                not exist. `'skip'` leaves it out, for conventional locations such as `.agents/skills`
                that a project may not have.
            duplicate_names: `'error'` (the default) fails the run when two different `SKILL.md` files
                share a name. `'keep_first'` keeps the one from the earlier directory and skips the
                other with a warning, the way coding agents layer project skills over personal ones.
            workspace: A workspace backend to read the libraries from instead of the run's, such as
                `LocalWorkspaceBackend('/app')` for skills shipped with the code while the agent
                works in a sandbox. It is read in-process only in this release: a durable engine
                does not route it through its workflow machinery.
        """
        if include is not None and exclude is not None:
            raise ValueError('include and exclude cannot be used together.')
        # Typed as literals, but agent specs and untyped callers reach here with any string.
        if missing_directories not in get_args(MissingDirectories):
            raise ValueError(f"missing_directories must be 'error' or 'skip', not {missing_directories!r}.")
        if duplicate_names not in get_args(DuplicateNames):
            raise ValueError(f"duplicate_names must be 'error' or 'keep_first', not {duplicate_names!r}.")

        self.directories = self._normalize_directories(directories)
        self.include = self._normalize_selection('include', include) if include is not None else None
        self.exclude = self._normalize_selection('exclude', exclude) if exclude is not None else frozenset()
        self.missing_directories = missing_directories
        self.duplicate_names = duplicate_names
        self.workspace = workspace
        own = secondary_workspace(workspace, 'Skills')
        self._sources = (
            _SkillSource(self.directories, self.include, self.exclude, missing_directories, duplicate_names, own),
        )

    def __repr__(self) -> str:
        """Show only the `Skills` configuration that callers control."""
        return (
            f'{type(self).__name__}('
            f'directories={self.directories!r}, include={self.include!r}, exclude={self.exclude!r}, '
            f'missing_directories={self.missing_directories!r}, duplicate_names={self.duplicate_names!r})'
        )

    @staticmethod
    def _normalize_directories(
        directories: str | Path | Sequence[str | Path],
    ) -> tuple[str | Path, ...]:
        if isinstance(directories, (str, Path)):
            return (directories,)
        normalized = tuple(directories)
        if not normalized:
            raise ValueError('Skills requires at least one skill-library directory.')
        return normalized

    @staticmethod
    def _normalize_selection(name: str, values: Collection[object]) -> frozenset[str]:
        if isinstance(values, str):
            raise TypeError(f'{name} must be a collection of skill names, not a string.')
        normalized: set[str] = set()
        for value in values:
            if not isinstance(value, str):
                raise TypeError(f'{name} must contain only skill names as strings.')
            normalized.add(value)
        return frozenset(normalized)

    @classmethod
    def combine(cls, capabilities: Sequence[AbstractCapability[AgentDepsT]]) -> AbstractCapability[AgentDepsT]:
        """Serve every combined configuration's libraries through one catalog.

        The field-by-field default would keep only the last configuration's directories, dropping the
        other libraries. A skill name selected by two configurations must name the same `SKILL.md`, or
        a byte-identical copy, unless the later configuration has `duplicate_names='keep_first'`.
        """
        first = capabilities[0]
        assert isinstance(first, cls)
        sources: list[_SkillSource] = []
        for capability in capabilities:
            assert isinstance(capability, cls)
            sources.extend(source for source in capability._sources if source not in sources)
        merged = replace_no_init(first)
        merged._sources = tuple(sources)
        return merged

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractCapability[AgentDepsT]:
        """Read the selected skills, from each configuration's `workspace=` or else the run's workspace.

        Returns one deferred capability per skill, or, without skills, this capability, which adds
        no instructions or tools. Emits a `UserWarning` for each skipped `SKILL.md`, an overlong
        description, or unsupported frontmatter. Raises `UserError` when a configuration without
        `workspace=` meets a run without a workspace.
        """
        if any(source.workspace is None for source in self._sources):
            require_workspace(ctx.workspace, 'Skills', ctx.messages)
        catalog = await self._load(ctx.workspace)
        for message in (*catalog.skipped, *self._advice(catalog.skills)):
            warnings.warn(message, UserWarning, stacklevel=2)
        return CombinedCapability([_Skill[AgentDepsT](skill) for skill in catalog.skills]) if catalog.skills else self

    async def load(self, workspace: WorkspaceBackend | None = None) -> SkillCatalog:
        """Read the selected skills now, as a run does at its start, without emitting warnings.

        For a host that offers skills to a person as well as to the model, such as a `/code-review`
        command that sends `skill.render(arguments)` as a prompt. The catalog's `skipped` messages
        say which `SKILL.md` files were left out, for the host to show.

        Args:
            workspace: Where libraries without their own `workspace=` live, as the run's workspace
                would be, such as `LocalWorkspaceBackend('.')`.

        Raises:
            UserError: A configuration without `workspace=` needs `workspace`, and none was passed.
            ValueError: A configuration problem that would also fail a run, such as an unknown
                `include` name or, with `duplicate_names='error'`, two skills with one name.
        """
        run_workspace = secondary_workspace(workspace, 'Skills.load')
        if run_workspace is None and any(source.workspace is None for source in self._sources):
            raise UserError(
                '`Skills.load()` needs the workspace the libraries are in, such as `LocalWorkspaceBackend(".")`.'
            )
        return await self._load(run_workspace)

    async def _load(self, run_workspace: Workspace | None) -> SkillCatalog:
        by_name: dict[str, tuple[Workspace, SkillDefinition]] = {}
        messages: list[str] = []
        for source in self._sources:
            workspace = source.workspace or run_workspace
            assert workspace is not None, 'a configuration without `workspace=` is only loaded with a run workspace'
            skills, skipped = await load_skill_libraries(
                workspace,
                source.directories,
                include=source.include,
                exclude=source.exclude,
                missing_directories=source.missing_directories,
                duplicate_names=source.duplicate_names,
            )
            messages.extend(skipped)
            for skill in skills:
                previous = by_name.get(skill.name)
                if previous is None:
                    # A library read from `workspace=` is not where the model's file tools work.
                    by_name[skill.name] = (workspace, replace(skill, in_run_workspace=source.workspace is None))
                    continue
                previous_workspace, previous_skill = previous
                if not await same_skill((previous_workspace, previous_skill.path), (workspace, skill.path)):
                    messages.append(
                        skip_duplicate_name(skill.name, previous_skill.path, skill.path, source.duplicate_names)
                    )
        return SkillCatalog(skills=tuple(skill for _, skill in by_name.values()), skipped=tuple(messages))

    @staticmethod
    def _advice(definitions: tuple[SkillDefinition, ...]) -> list[str]:
        advice: list[str] = []
        overlong_descriptions = [
            f'{skill.name} ({len(skill.description):,} characters)'
            for skill in definitions
            if len(skill.description) > _MAX_DESCRIPTION_LENGTH
        ]
        if overlong_descriptions:
            advice.append(
                f'Agent Skill descriptions exceed the {_MAX_DESCRIPTION_LENGTH:,}-character limit: '
                + '; '.join(overlong_descriptions)
            )
        ignored = [
            f'{skill.name}: {", ".join(skill.ignored_behavioral_fields)}'
            for skill in definitions
            if skill.ignored_behavioral_fields
        ]
        if ignored:
            advice.append('Ignoring unsupported Agent Skill behavioral frontmatter fields: ' + '; '.join(ignored))
        return advice
