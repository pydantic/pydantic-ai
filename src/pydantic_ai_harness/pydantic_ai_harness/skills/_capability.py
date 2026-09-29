"""Load Agent Skill instructions from the run's workspace as deferred capabilities."""

from __future__ import annotations

import warnings
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import overload

from pydantic_ai._utils import replace_no_init
from pydantic_ai.capabilities import AbstractCapability, CombinedCapability
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.workspaces import Workspace, WorkspaceBackend
from pydantic_ai_harness._workspace import require_workspace, secondary_workspace
from pydantic_ai_harness.skills._loader import SkillDefinition, load_skill_libraries

_MAX_DESCRIPTION_LENGTH = 1024


@dataclass(frozen=True)
class _SkillSource:
    """One `Skills(...)` configuration: libraries, the selection applied to them, and where they live."""

    directories: tuple[str | Path, ...]
    include: frozenset[str] | None
    exclude: frozenset[str]
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
        self.instructions = f'# Skill: {skill.name}\n\n{skill.body}' if skill.body else f'# Skill: {skill.name}'

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
    the skill: the model sees its name and description, and loads its Markdown body with
    `load_capability`. Bundled files are not loaded or executed. Descriptions longer than the
    Agent Skills limit are preserved and emit a warning.

    Two `Skills` on one agent combine, so every library either names stays reachable.
    """

    directories: tuple[str | Path, ...]
    """Skill-library paths in the workspace, read at the start of each run."""

    include: frozenset[str] | None
    """Exact skill names to expose, or `None` to expose all discovered skills."""

    exclude: frozenset[str]
    """Exact skill names to omit from the catalog."""

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
        workspace: WorkspaceBackend | None = None,
    ) -> None: ...

    @overload
    def __init__(  # pragma: no cover - overload is enforced by static type checking
        self,
        directories: str | Path | Sequence[str | Path],
        *,
        include: None = None,
        exclude: Collection[str] | None = None,
        workspace: WorkspaceBackend | None = None,
    ) -> None: ...

    def __init__(
        self,
        directories: str | Path | Sequence[str | Path],
        *,
        include: Collection[str] | None = None,
        exclude: Collection[str] | None = None,
        workspace: WorkspaceBackend | None = None,
    ) -> None:
        """Configure the skill libraries to read at the start of each run.

        Args:
            directories: One skill-library path or a sequence of paths in the workspace.
            include: Exact names to expose. Omit to expose all discovered skills.
            exclude: Exact names to omit. Cannot be combined with `include`.
            workspace: A workspace backend to read the libraries from instead of the run's, such as
                `LocalWorkspaceBackend('/app')` for skills shipped with the code while the agent
                works in a sandbox. It is read in-process only in this release: a durable engine
                does not route it through its workflow machinery.
        """
        if include is not None and exclude is not None:
            raise ValueError('include and exclude cannot be used together.')

        self.directories = self._normalize_directories(directories)
        self.include = self._normalize_selection('include', include) if include is not None else None
        self.exclude = self._normalize_selection('exclude', exclude) if exclude is not None else frozenset()
        self.workspace = workspace
        own = secondary_workspace(workspace, 'Skills')
        self._sources = (_SkillSource(self.directories, self.include, self.exclude, own),)

    def __repr__(self) -> str:
        """Show only the `Skills` configuration that callers control."""
        return (
            f'{type(self).__name__}('
            f'directories={self.directories!r}, include={self.include!r}, exclude={self.exclude!r})'
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
        other libraries. A skill name selected by two configurations must name the same `SKILL.md`.
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

        Returns one deferred capability per skill. Raises `UserError` when a configuration without
        `workspace=` meets a run without a workspace.
        """
        if any(source.workspace is None for source in self._sources):
            require_workspace(ctx.workspace, 'Skills', ctx.messages)
        skills = await self._load(ctx.workspace)
        self._warn(skills)
        return CombinedCapability([_Skill[AgentDepsT](skill) for skill in skills]) if skills else self

    async def _load(self, run_workspace: Workspace) -> tuple[SkillDefinition, ...]:
        by_name: dict[str, tuple[Workspace, SkillDefinition]] = {}
        for source in self._sources:
            workspace = source.workspace or run_workspace
            for skill in await load_skill_libraries(
                workspace, source.directories, include=source.include, exclude=source.exclude
            ):
                previous_workspace, previous = by_name.setdefault(skill.name, (workspace, skill))
                if previous_workspace is not workspace or previous.path != skill.path:
                    raise ValueError(f'Duplicate skill name {skill.name!r}: {previous.path} and {skill.path}.')
        return tuple(skill for _, skill in by_name.values())

    @staticmethod
    def _warn(definitions: tuple[SkillDefinition, ...]) -> None:
        overlong_descriptions = [
            f'{skill.name} ({len(skill.description):,} characters)'
            for skill in definitions
            if len(skill.description) > _MAX_DESCRIPTION_LENGTH
        ]
        if overlong_descriptions:
            warnings.warn(
                f'Agent Skill descriptions exceed the {_MAX_DESCRIPTION_LENGTH:,}-character limit: '
                + '; '.join(overlong_descriptions),
                UserWarning,
                stacklevel=3,
            )
        ignored = [
            f'{skill.name}: {", ".join(skill.ignored_behavioral_fields)}'
            for skill in definitions
            if skill.ignored_behavioral_fields
        ]
        if ignored:
            warnings.warn(
                'Ignoring unsupported Agent Skill behavioral frontmatter fields: ' + '; '.join(ignored),
                UserWarning,
                stacklevel=3,
            )
