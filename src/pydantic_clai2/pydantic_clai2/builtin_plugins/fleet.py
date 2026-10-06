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
from pydantic_ai_harness.logfire import AgentControlConfig
from pydantic_ai_harness.policy import Approver, Policy, PolicyDecision, PolicyRules
from pydantic_clai2 import policy_state

ItemKind = Literal['skill', 'mcp_server', 'plugin', 'instruction']


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


class FleetAgentConfig(AgentControlConfig):
    """Agent Control's config (with named added instructions) plus company `skills` and `mcp_servers` sections."""

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
class Snapshot:
    """The company config and catalog, resolved once, so everything a run or a turn reads agrees."""

    config: FleetAgentConfig
    version: str | None
    catalog: Catalog

    @property
    def policy(self) -> Policy | None:
        return self.config.policy

    @property
    def locked(self) -> frozenset[str]:
        return frozenset(self.config.policy.locked) if self.config.policy is not None else frozenset()


@dataclass(frozen=True)
class Build:
    """One build of the fleet config for a run: the capabilities, the items they came from, and what failed."""

    snapshot: Snapshot
    capabilities: list[AbstractCapability[None]]
    loaded: list[ActiveItem]
    failed: list[tuple[ActiveItem, str]]


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
        # A removed item is no longer anywhere to say which tier it came from.
        where = {'company': 'company ', 'catalog': 'catalog '}.get(self.tier, '')
        return f'{self.action.capitalize()} {where}{noun.get(self.kind, self.kind)} from Logfire: {self.name}'


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
    _prepared: Build | None = field(default=None, init=False)
    latest: Snapshot | None = field(default=None, init=False)
    """The most recent snapshot, for decisions made outside a run (the MCP allowlist, locked items)."""

    def __post_init__(self) -> None:
        self.agent_variable = Variable(
            f'agent__{self.name}', type=FleetAgentConfig, default=FleetAgentConfig(), logfire_instance=self.instance
        )
        self.catalog_variable = Variable(
            f'catalog__{self.name}', type=Catalog, default=Catalog(), logfire_instance=self.instance
        )

    # Resolution

    def snapshot(self) -> Snapshot:
        """Resolve the company config and the catalog once (one `Resolve variable` span each)."""
        targeting_key, attributes = self.targeting_key(), self.attributes()
        resolved = self.agent_variable.get(targeting_key=targeting_key, attributes=attributes)
        version = getattr(resolved, 'version', None)
        catalog = self.catalog_variable.get(targeting_key=targeting_key, attributes=attributes).value
        self.latest = Snapshot(
            config=resolved.value, version=None if version is None else str(version), catalog=catalog
        )
        return self.latest

    def fingerprint(self) -> tuple[str | None, ...]:
        """The raw published values, read from the provider's cache without a span, to notice a push cheaply."""
        provider = self.instance.config.get_variable_provider()
        targeting_key, attributes = self.targeting_key(), self.attributes()
        return tuple(
            provider.get_serialized_value(variable.name, targeting_key, attributes).value
            for variable in (self.agent_variable, self.catalog_variable)
        )

    def current_policy(self) -> Policy | None:
        """The policy as last resolved, for decisions outside a run."""
        return (self.latest or self.snapshot()).policy

    def active(self, snapshot: Snapshot) -> list[ActiveItem]:
        """Company items, then catalog items the user has on."""
        config = snapshot.config
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
        for item in snapshot.catalog.items:
            key = _key(item.kind, item.name)
            if key in company or not self._enabled(item, state, snapshot.locked):
                continue
            items.append(ActiveItem(item.kind, item.name, item.description, 'catalog', item.payload))
        return items

    def compliance(self, snapshot: Snapshot, active: Sequence[ActiveItem]) -> dict[str, str]:
        """The compliance attributes for a run: policy version, what the user opted out of, locked items held."""
        state = self._user_state()
        defaults_on = {_key(item.kind, item.name) for item in snapshot.catalog.items if item.default == 'on'}
        opted_out = sorted(key for key in state.opted_out if key in defaults_on)
        keys = {item.key for item in active} | {f'plugin:{name}' for name in policy_state.loaded_plugins()}
        return {
            'clai2.policy.version': snapshot.version or '',
            'clai2.catalog.opted_out': ','.join(opted_out),
            'clai2.policy.locked_ok': 'true' if snapshot.locked <= keys else 'false',
        }

    def _enabled(self, item: CatalogItem, state: _UserState, locked: frozenset[str]) -> bool:
        key = _key(item.kind, item.name)
        if key in locked:
            return True
        if item.default == 'on':
            return key not in state.opted_out
        return key in state.opted_in

    # Capabilities

    def build(self, snapshot: Snapshot | None = None) -> Build:
        """Build every active item; one that fails is reported, not counted as loaded."""
        snapshot = snapshot or self.snapshot()
        capabilities: list[AbstractCapability[None]] = []
        loaded: list[ActiveItem] = []
        failed: list[tuple[ActiveItem, str]] = []
        for item in self.active(snapshot):
            try:
                capability = self._build(item)
            except Exception as error:  # noqa: BLE001 -- one bad pushed item must not stop the run
                failed.append((item, f'{type(error).__name__}: {error}'))
                continue
            loaded.append(item)
            if capability is not None:
                capabilities.append(capability)
        return Build(snapshot=snapshot, capabilities=capabilities, loaded=loaded, failed=failed)

    def prepare(self) -> Build:
        """Build for the turn about to start, so its notices and its run agree on what loaded."""
        self._prepared = self.build()
        return self._prepared

    def take(self) -> Build:
        """The build `prepare` made for this turn's run, or a fresh one (a nested or headless run)."""
        build, self._prepared = self._prepared or self.build(), None
        return build

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
                leaf = MCPToolset[None](server.url, id=server.name, headers=headers or None)
                policy_state.mark_gated(leaf)  # Pushed by Logfire, so always allowed.
                toolset = leaf.prefixed(_capability_id(server.name))
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
        snapshot = self.snapshot()
        catalog = {_key(item.kind, item.name): item for item in snapshot.catalog.items}
        if key not in catalog:
            matches = [k for k in catalog if k.split(':', 1)[1] == key]
            if len(matches) != 1:
                return f'No catalog item {key!r}. Run /catalog to list them.'
            key = matches[0]
        item = catalog[key]
        if not on and key in snapshot.locked:
            return f'{item.name} is {policy_state.LOCKED_MESSAGE}; it cannot be disabled.'
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

    def changes(self, build: Build, *, mark_seen: bool = True) -> list[Change]:
        """What was added, updated, or removed since the user last saw the fleet config, among what loaded.

        An item that failed to build is neither news nor a removal: the user sees the failure instead, and
        the item is announced once it loads.
        """
        config = build.snapshot.config
        failed = {item.key for item, _ in build.failed}
        current: dict[str, tuple[str, str, str, str]] = {}
        for item in build.loaded:
            current[item.key] = (item.kind, item.name, item.tier, _digest(dict(item.payload)))
        named_texts = {text for _, text in _named_instructions(config)}
        added_instructions = [
            block
            for block in config.instructions or ()
            if _is_added(block) and (block if isinstance(block, str) else block.instructions) not in named_texts
        ]
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
        for key in user.seen.keys() - current.keys() - failed:
            kind, _, name = key.partition(':')
            changes.append(Change('removed', kind, name, ''))
        if mark_seen and changes:
            kept = {key: digest for key, digest in user.seen.items() if key in failed}
            user.seen = {**kept, **{key: value[3] for key, value in current.items()}}
            self._save(state)
        return changes

    def listing(self) -> str:
        """The `/catalog` listing: company items, then the catalog with each item's state for this user."""
        snapshot = self.snapshot()
        config, version = snapshot.config, snapshot.version
        state = self._user_state()
        lines = [f'Company config from Logfire (agent__{self.name}{f" v{version}" if version else ""}):']
        company = [
            *(f'  skill       {skill.name}: {skill.description}' for skill in config.skills or ()),
            *(f'  mcp_server  {server.name}: {server.url}' for server in config.mcp_servers or ()),
        ]
        lines.extend(company or ['  (none)'])
        lines.append(f'Catalog (catalog__{self.name}):')
        items = snapshot.catalog.items
        for item in items:
            on = self._enabled(item, state, snapshot.locked)
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


def _named_instructions(config: FleetAgentConfig) -> list[tuple[str, str]]:
    """The added instructions published with a `name`, as `(name, text)`; unnamed ones are grouped separately."""
    named: list[tuple[str, str]] = []
    for block in config.instructions or ():
        text = block if isinstance(block, str) else block.instructions if block.id is None else None
        name = config.instruction_name(text) if text else None
        if name and text:
            named.append((name, text))
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

    approver: Approver | None = None
    record: Callable[[PolicyDecision], None] | None = None

    async def for_run(self, ctx: RunContext[None]) -> AbstractCapability[None]:
        build = self.fleet.take()
        capabilities, items, snapshot = build.capabilities, build.loaded, build.snapshot
        baggage = {
            ACTIVE_ITEMS_ATTRIBUTE: ','.join(sorted(item.key for item in items)),
            **self.fleet.compliance(snapshot, items),
        }
        # The run's policy is the one this run resolved, so a push mid-run applies from the next run.
        rules = PolicyRules(
            policy=lambda: snapshot.policy, approver=self.approver, record=self.record, attribute_prefix='clai2.policy'
        )
        return CombinedCapability([_AdoptionBaggage(baggage=baggage), rules, _PluginMCPAllowlist(), *capabilities])


@dataclass(kw_only=True)
class _PluginMCPAllowlist(AbstractCapability[None]):
    """Apply the MCP allowlist to MCP toolsets other plugins contribute (the `mcp` plugin gates its own).

    In `enforce` mode a server outside the list keeps its connection but offers the model no tools.
    """

    def get_wrapper_toolset(self, toolset: AbstractToolset[None]) -> AbstractToolset[None]:
        return toolset.visit_and_replace(_gate_plugin_mcp)


def _gate_plugin_mcp(toolset: AbstractToolset[None]) -> AbstractToolset[None]:
    if not isinstance(toolset, MCPToolset) or policy_state.is_gated(toolset):
        return toolset
    transport = getattr(toolset.client, 'transport', None)
    url = str(getattr(transport, 'url', '') or '')
    name = toolset.id or url or 'unnamed'
    if policy_state.mcp_allowed(name, url, subject=url or name):
        return toolset
    return toolset.filtered(lambda ctx, tool_def: False)


ACTIVE_ITEMS_ATTRIBUTE = 'clai2.fleet.active'
"""Every span of a run lists the fleet items in force for it, as sorted `kind:name` keys joined by commas."""


@dataclass(kw_only=True)
class _AdoptionBaggage(AbstractCapability[None]):
    """Puts which company and catalog items this run had on every span, so Logfire can show who adopted what."""

    baggage: dict[str, str]

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position='outermost', wraps=(Instrumentation,))

    async def wrap_run(self, ctx: RunContext[None], *, handler: WrapRunHandler) -> AgentRunResult[Any]:
        with logfire.set_baggage(**self.baggage):
            return await handler()
