"""Logfire as the fleet's control plane: company config, the catalog, and notices when either changes.

Hackathon code. The `observability` plugin builds this next to its instrumentation when it has a Logfire API key
that can read managed variables. Three things come down from Logfire:

- `agent__<name>`: the Agent Control config, applied by `AgentControl`, plus two sections this module applies
  itself: company `skills` (deferred capabilities the model loads on demand) and company `mcp_servers`.
- `catalog__<name>`: the marketplace. Items marked `default: on` are active unless the user opted out with
  `/catalog`; items marked `off` are active only once the user opts in. `plugin` items must name a capability
  class on the plugin's allowlist.
- What changed since the user last saw it, shown as a notice at the start of a turn and in the status bar.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import logfire
from logfire.agent_control import AgentConfig
from logfire.variables import Variable
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from pydantic_ai import AgentRunResult, RunContext
from pydantic_ai.capabilities import (
    AbstractCapability,
    Capability,
    CapabilityOrdering,
    CombinedCapability,
    Instrumentation,
    WrapRunHandler,
)
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.toolsets import AbstractToolset

ItemKind = Literal['skill', 'mcp_server', 'plugin', 'instruction']
INSTRUCTION_PREFIX = 'fleet:'
"""Pushed instruction blocks are named `fleet:<slug>`; AgentControl adds them rather than matching code blocks."""


class FleetSkill(BaseModel):
    """A skill pushed to every agent: the model sees its name and description and loads the body on demand."""

    model_config = ConfigDict(extra='ignore')
    name: str
    description: str = ''
    instructions: str = ''
    source: str | None = None
    proposal_id: str | None = None


class FleetMCPServer(BaseModel):
    """A Streamable HTTP MCP server pushed to every agent; header values may reference `${env:NAME}`."""

    model_config = ConfigDict(extra='ignore')
    name: str
    url: str
    description: str | None = None
    headers: dict[str, str] = Field(default_factory=dict[str, str])


class FleetAgentConfig(AgentConfig):
    """Agent Control's `AgentConfig` plus the hackathon's company `skills` and `mcp_servers` sections."""

    skills: list[FleetSkill] | None = None
    mcp_servers: list[FleetMCPServer] | None = None


class CatalogItem(BaseModel):
    """One marketplace entry; `payload` is the skill, MCP server, or plugin declaration it activates."""

    model_config = ConfigDict(extra='ignore')
    kind: ItemKind
    name: str
    description: str = ''
    default: Literal['on', 'off'] = 'off'
    payload: dict[str, Any] = Field(default_factory=dict[str, Any])


class Catalog(BaseModel):
    """The `catalog__<name>` variable."""

    model_config = ConfigDict(extra='ignore')
    items: list[CatalogItem] = Field(default_factory=list[CatalogItem])


class _UserState(BaseModel):
    opted_in: list[str] = Field(default_factory=list[str])
    opted_out: list[str] = Field(default_factory=list[str])
    seen: dict[str, str] = Field(default_factory=dict[str, str])
    """What the user was last shown, as item key to content digest."""


class _State(BaseModel):
    users: dict[str, _UserState] = Field(default_factory=dict[str, _UserState])


def _key(kind: str, name: str) -> str:
    return f'{kind}:{name}'


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:12]


_ENV_REF = re.compile(r'\$\{env:([A-Za-z_][A-Za-z0-9_]*)\}')


def _resolve_env(value: str) -> str:
    return _ENV_REF.sub(lambda match: os.environ.get(match.group(1), ''), value)


@dataclass(frozen=True)
class ActiveItem:
    """An item in force for this user, with where it came from."""

    kind: ItemKind
    name: str
    description: str
    tier: Literal['company', 'catalog']
    payload: Mapping[str, Any]

    @property
    def key(self) -> str:
        return _key(self.kind, self.name)


@dataclass(frozen=True)
class Change:
    """Something that arrived from Logfire since the user last saw the fleet config."""

    action: Literal['added', 'updated', 'removed']
    kind: str
    name: str
    tier: str

    def describe(self) -> str:
        verb = {'added': 'New', 'updated': 'Updated', 'removed': 'Removed'}[self.action]
        if self.kind == 'instructions':
            return f'{verb} company instructions from Logfire'
        if self.kind == 'instruction':
            return f'{verb} company instruction from Logfire: {self.name}'
        noun = {'skill': 'skill', 'mcp_server': 'MCP server', 'plugin': 'plugin'}
        where = 'company' if self.tier == 'company' else 'catalog'
        return f'{self.action.capitalize()} {where} {noun.get(self.kind, self.kind)} from Logfire: {self.name}'


@dataclass
class Fleet:
    """Reads the fleet config for one user, and remembers what that user opted into and has seen."""

    instance: logfire.Logfire
    name: str
    state_file: Path
    allowed_plugins: Sequence[str] = ()
    attributes: Callable[[], Mapping[str, Any]] = dict
    targeting_key: Callable[[], str | None] = lambda: None
    user: Callable[[], str] = lambda: 'local'
    agent_variable: Variable[FleetAgentConfig] = field(init=False)
    catalog_variable: Variable[Catalog] = field(init=False)
    _mcp: dict[str, AbstractToolset[None]] = field(default_factory=dict[str, AbstractToolset[None]], init=False)
    warnings: list[str] = field(default_factory=list[str], init=False)

    def __post_init__(self) -> None:
        self.agent_variable = Variable(
            f'agent__{self.name}', type=FleetAgentConfig, default=FleetAgentConfig(), logfire_instance=self.instance
        )
        self.catalog_variable = Variable(
            f'catalog__{self.name}', type=Catalog, default=Catalog(), logfire_instance=self.instance
        )

    # Resolution

    def config(self) -> tuple[FleetAgentConfig, str | None]:
        resolved = self.agent_variable.get(targeting_key=self.targeting_key(), attributes=self.attributes())
        version = getattr(resolved, 'version', None)
        return resolved.value, None if version is None else str(version)

    def catalog(self) -> Catalog:
        return self.catalog_variable.get(targeting_key=self.targeting_key(), attributes=self.attributes()).value

    def active(self) -> list[ActiveItem]:
        """Company items, then catalog items the user has on."""
        config, _ = self.config()
        items = [
            *(
                ActiveItem('instruction', name, '', 'company', {'instructions': text})
                for name, text in _named_instructions(config)
            ),
            *(
                ActiveItem('skill', skill.name, skill.description, 'company', skill.model_dump())
                for skill in config.skills or ()
            ),
            *(
                ActiveItem('mcp_server', server.name, server.description or '', 'company', server.model_dump())
                for server in config.mcp_servers or ()
            ),
        ]
        state = self._user_state()
        company = {item.key for item in items}
        for item in self.catalog().items:
            key = _key(item.kind, item.name)
            if key in company or not self._enabled(item, state):
                continue
            items.append(ActiveItem(item.kind, item.name, item.description, 'catalog', item.payload))
        return items

    def _enabled(self, item: CatalogItem, state: _UserState) -> bool:
        key = _key(item.kind, item.name)
        if item.default == 'on':
            return key not in state.opted_out
        return key in state.opted_in

    # Capabilities

    def capabilities(self) -> list[AbstractCapability[None]]:
        self.warnings.clear()
        built: list[AbstractCapability[None]] = []
        for item in self.active():
            try:
                capability = self._build(item)
            except Exception as error:  # noqa: BLE001 -- one bad pushed item must not stop the run
                self.warnings.append(f'Skipped {item.kind} {item.name!r} from Logfire: {error}')
                continue
            if capability is not None:
                built.append(capability)
        return built

    def _build(self, item: ActiveItem) -> AbstractCapability[None] | None:
        if item.kind == 'instruction':
            return None  # AgentControl adds these to the prompt itself.
        if item.kind == 'skill':
            skill = FleetSkill.model_validate({'name': item.name, **item.payload})
            body = f'# Skill: {skill.name}\n\n{skill.instructions}' if skill.instructions else f'# Skill: {skill.name}'
            return Capability[None](
                id=_capability_id(skill.name),
                description=(skill.description or skill.name).replace('\n', '\n  '),
                instructions=body,
                defer_loading=True,
            )
        if item.kind == 'mcp_server':
            server = FleetMCPServer.model_validate({'name': item.name, **item.payload})
            headers = {key: _resolve_env(value) for key, value in server.headers.items()}
            cache_key = _digest([server.url, headers])
            toolset = self._mcp.get(cache_key)
            if toolset is None:
                toolset = MCPToolset[None](server.url, id=server.name, headers=headers or None).prefixed(
                    _capability_id(server.name)
                )
                self._mcp[cache_key] = toolset
            return Capability[None](toolsets=[toolset])
        factory = str(item.payload.get('factory', ''))
        if factory not in self.allowed_plugins:
            raise ValueError(f'{factory or "(no factory)"} is not on the allowlist of plugins Logfire may enable')
        module_name, _, attr = factory.partition(':')
        target = getattr(importlib.import_module(module_name), attr)
        if not (isinstance(target, type) and issubclass(target, AbstractCapability)):
            raise TypeError(f'{factory} is not a capability class')
        settings: dict[str, Any] = dict(item.payload.get('settings') or {})
        return target(**settings)  # pyright: ignore[reportUnknownVariableType]

    # Opt-in and notices

    def set_opt(self, key: str, on: bool) -> str:
        catalog = {_key(item.kind, item.name): item for item in self.catalog().items}
        if key not in catalog:
            matches = [k for k in catalog if k.split(':', 1)[1] == key]
            if len(matches) != 1:
                return f'No catalog item {key!r}. Run /catalog to list them.'
            key = matches[0]
        item = catalog[key]
        state = self._load()
        user = state.users.setdefault(self.user(), _UserState())
        user.opted_in = [k for k in user.opted_in if k != key]
        user.opted_out = [k for k in user.opted_out if k != key]
        if on and item.default == 'off':
            user.opted_in.append(key)
        elif not on and item.default == 'on':
            user.opted_out.append(key)
        # The user made this change themselves, so it is not news on their next prompt.
        if on:
            user.seen[key] = _digest(dict(item.payload))
        else:
            user.seen.pop(key, None)
        self._save(state)
        return f'{"Enabled" if on else "Disabled"} {item.name} ({item.kind}); it applies from your next prompt.'

    def changes(self, *, mark_seen: bool = True) -> list[Change]:
        """What was added, updated, or removed since the user last saw the fleet config."""
        config, _ = self.config()
        current: dict[str, tuple[str, str, str, str]] = {}
        for item in self.active():
            current[item.key] = (item.kind, item.name, item.tier, _digest(dict(item.payload)))
        added_instructions = [block for block in config.instructions or () if _is_added(block)]
        if added_instructions:
            digest = _digest([_dump_block(block) for block in added_instructions])
            current['instructions:company'] = ('instructions', 'company instructions', 'company', digest)
        state = self._load()
        user = state.users.setdefault(self.user(), _UserState())
        changes: list[Change] = []
        for key, (kind, name, tier, digest) in current.items():
            previous = user.seen.get(key)
            if previous is None:
                changes.append(Change('added', kind, name, tier))
            elif previous != digest:
                changes.append(Change('updated', kind, name, tier))
        for key in user.seen.keys() - current.keys():
            kind, _, name = key.partition(':')
            changes.append(Change('removed', kind, name, 'company'))
        if mark_seen and changes:
            user.seen = {key: value[3] for key, value in current.items()}
            self._save(state)
        return changes

    def listing(self) -> str:
        """The `/catalog` listing: company items, then the catalog with each item's state for this user."""
        config, version = self.config()
        state = self._user_state()
        lines = [f'Company config from Logfire (agent__{self.name}{f" v{version}" if version else ""}):']
        company = [
            *(f'  skill       {skill.name}: {skill.description}' for skill in config.skills or ()),
            *(f'  mcp_server  {server.name}: {server.url}' for server in config.mcp_servers or ()),
        ]
        lines.extend(company or ['  (none)'])
        lines.append(f'Catalog (catalog__{self.name}):')
        items = self.catalog().items
        for item in items:
            on = self._enabled(item, state)
            default = 'default on' if item.default == 'on' else 'optional'
            lines.append(f'  [{"x" if on else " "}] {item.kind:<10} {item.name} ({default}): {item.description}')
        if not items:
            lines.append('  (empty)')
        lines.append('Toggle with /catalog enable NAME or /catalog disable NAME.')
        return '\n'.join(lines)

    def _user_state(self) -> _UserState:
        return self._load().users.get(self.user(), _UserState())

    def _load(self) -> _State:
        try:
            return _State.model_validate_json(self.state_file.read_text())
        except (OSError, ValidationError):
            return _State()

    def _save(self, state: _State) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(state.model_dump_json(indent=2))


def _capability_id(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9_-]', '_', name) or 'item'


def _named_instructions(config: AgentConfig) -> list[tuple[str, str]]:
    """The pushed instructions that carry a name, as `(slug, text)`."""
    named: list[tuple[str, str]] = []
    for block in config.instructions or ():
        block_id = getattr(block, 'id', None)
        text = getattr(block, 'instructions', None)
        if isinstance(block_id, str) and block_id.startswith(INSTRUCTION_PREFIX) and text:
            named.append((block_id.removeprefix(INSTRUCTION_PREFIX), text))
    return named


def _is_added(block: object) -> bool:
    if isinstance(block, str):
        return True
    return getattr(block, 'id', None) is None


def _dump_block(block: object) -> object:
    return block.model_dump() if isinstance(block, BaseModel) else block


@dataclass(kw_only=True)
class FleetControl(AbstractCapability[None]):
    """Contributes the company skills, MCP servers, and enabled catalog items, read afresh for every run."""

    fleet: Fleet
    id: str | None = 'clai2_fleet'

    async def for_run(self, ctx: RunContext[None]) -> AbstractCapability[None]:
        capabilities = self.fleet.capabilities()
        active = ','.join(sorted(item.key for item in self.fleet.active()))
        return CombinedCapability([_AdoptionBaggage(active=active), *capabilities])


ACTIVE_ITEMS_ATTRIBUTE = 'clai2.fleet.active'
"""Every span of a run lists the fleet items in force for it, as sorted `kind:name` keys joined by commas."""


@dataclass(kw_only=True)
class _AdoptionBaggage(AbstractCapability[None]):
    """Puts which company and catalog items this run had on every span, so Logfire can show who adopted what."""

    active: str

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position='outermost', wraps=(Instrumentation,))

    async def wrap_run(self, ctx: RunContext[None], *, handler: WrapRunHandler) -> AgentRunResult[Any]:
        with logfire.set_baggage(**{ACTIVE_ITEMS_ATTRIBUTE: self.active}):
            return await handler()
