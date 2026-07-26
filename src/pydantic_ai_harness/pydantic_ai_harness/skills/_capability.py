"""Load Agent Skill instructions as deferred Pydantic AI capabilities."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import overload

from pydantic_ai.capabilities import AbstractCapability, Capability
from pydantic_ai.tools import AgentDepsT

from pydantic_ai_harness.skills._loader import SkillDefinition, load_skill_libraries

_MAX_DESCRIPTION_LENGTH = 1024


@dataclass(init=False, repr=False)
class Skills(AbstractCapability[AgentDepsT]):
    """Load Agent Skill instructions as deferred capabilities.

    Libraries are scanned once during construction. Each selected immediate
    child containing `SKILL.md` becomes a deferred capability using the skill's
    name, description, and Markdown body. Bundled files are not loaded or
    executed. Descriptions longer than the Agent Skills limit are preserved and
    emit a warning.
    """

    directories: tuple[str | Path, ...]
    """Skill-library paths scanned during construction."""

    include: frozenset[str] | None
    """Exact skill names to expose, or `None` to expose all discovered skills."""

    exclude: frozenset[str]
    """Exact skill names to omit from the deferred capability catalog."""

    _deferred_capabilities: tuple[Capability[AgentDepsT], ...] = field(init=False, repr=False, compare=False)
    """Deferred leaf capabilities generated from the selected skills."""

    @overload
    def __init__(  # pragma: no cover - overload is enforced by static type checking
        self,
        directories: str | Path | Sequence[str | Path],
        *,
        include: Collection[str],
        exclude: None = None,
    ) -> None: ...

    @overload
    def __init__(  # pragma: no cover - overload is enforced by static type checking
        self,
        directories: str | Path | Sequence[str | Path],
        *,
        include: None = None,
        exclude: Collection[str] | None = None,
    ) -> None: ...

    def __init__(
        self,
        directories: str | Path | Sequence[str | Path],
        *,
        include: Collection[str] | None = None,
        exclude: Collection[str] | None = None,
    ) -> None:
        """Build a snapshot of the selected Agent Skills.

        Args:
            directories: One skill-library path or a sequence of paths.
            include: Exact names to expose. Omit to expose all discovered skills.
            exclude: Exact names to omit. Cannot be combined with `include`.
        """
        if include is not None and exclude is not None:
            raise ValueError('include and exclude cannot be used together.')

        self.directories = self._normalize_directories(directories)
        self.include = self._normalize_selection('include', include) if include is not None else None
        self.exclude = self._normalize_selection('exclude', exclude) if exclude is not None else frozenset()

        definitions = load_skill_libraries(self.directories, include=self.include, exclude=self.exclude)
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
                stacklevel=2,
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
                stacklevel=2,
            )
        self._deferred_capabilities = tuple(self._to_capability(skill) for skill in definitions)

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

    def apply(self, visitor: Callable[[AbstractCapability[AgentDepsT]], None]) -> None:
        """Expose each selected skill as a deferred leaf capability."""
        for capability in self._deferred_capabilities:
            capability.apply(visitor)

    def _to_capability(self, skill: SkillDefinition) -> Capability[AgentDepsT]:
        # TODO: Pydantic AI core should render multiline deferred-capability descriptions without making
        # continuation lines look like separate catalog entries. Keep valid SKILL.md descriptions unchanged here.
        return Capability[AgentDepsT](
            id=skill.name,
            description=skill.description,
            instructions=f'# Skill: {skill.name}\n\n{skill.body}' if skill.body else f'# Skill: {skill.name}',
            defer_loading=True,
        )
