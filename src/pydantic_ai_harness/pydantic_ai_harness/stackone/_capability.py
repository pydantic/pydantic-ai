"""StackOne capability that gives agents access to actions on a linked SaaS account."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AbstractToolset, DynamicToolset
from pydantic_ai_harness._warn import warn_argument_renamed
from pydantic_ai_harness.stackone._toolset import (
    STACKONE_BASE_URL,
    MCPToolsetClient,
    StackOneToolset,
    ToolMode,
    resolve_tool_mode,
    validate_configuration,
)

_ID_PREFIX = 'stackone'
_DEFAULT_DESCRIPTION = 'Use actions from a linked business application through StackOne.'
_INDIVIDUAL_INSTRUCTIONS = (
    "The StackOne tools operate on the user's linked SaaS account (HRIS, ATS, CRM, and more). "
    'Tool names follow `{connector}_{action}_{entity}`, for example `bamboohr_list_employees`. '
    "Results follow each tool's output schema. Prefer list actions with filters over unbounded listings."
)
_SEARCH_EXECUTE_INSTRUCTIONS = (
    'StackOne is available through two tools: a search tool (name ending in `_search_actions`) that finds '
    'available actions from a natural-language query, and an execute tool (name ending in `_execute_action`) that '
    'runs one action by id. Always search first: `action_id` values are runtime identifiers returned by the search '
    'tool and must never be guessed.'
)


@dataclass
class StackOne(AbstractCapability[AgentDepsT]):
    """Actions on the user's SaaS account (HRIS, ATS, CRM, and more) via StackOne.

    Connects an agent to one linked account's actions over StackOne's MCP
    endpoint, with authentication, tool filtering, and usage instructions.
    """

    account_id: str
    """The linked account to act on (one account is one provider connection)."""

    _: KW_ONLY

    id: str | None = None
    """Stable capability and toolset ID, derived from `account_id` when not given.

    One account is one provider connection, so the account is what identifies this capability --
    the same way an MCP server is identified by its URL. Deriving it rather than fixing it to
    `'stackone'` is what lets one agent reach two linked accounts: their ids differ, so they stay
    two capabilities. Two under the *same* account are a mistake, and collide.
    """

    description: str | None = _DEFAULT_DESCRIPTION
    """Routing description used when the capability is loaded on demand."""

    auth: str | Callable[[RunContext[AgentDepsT]], str | None] | None = field(default=None, repr=False)
    """A StackOne API key or a function of the run context that returns one.

    Unset, it uses `STACKONE_API_KEY`. A function never does: if it returns `None` or `''`, that run has no
    StackOne tools.
    """

    base_url: str = STACKONE_BASE_URL
    """HTTPS StackOne API host. Point at a regional or staging host if needed."""

    actions: str | Sequence[str] = ()
    """`fnmatch` globs over full tool names (case-insensitive), e.g. `['*_list_*']`.
    Giving `actions` switches the default `tool_mode` to `individual`, where the globs apply."""

    tool_mode: ToolMode | None = None
    """`individual` registers one tool per enabled action; `search_execute` registers two
    server-side meta-tools (search the catalog, execute an action by id) whose prompt
    footprint stays constant however large the catalog is. `None` picks `search_execute`,
    or `individual` when `actions` are given."""

    include_instructions: bool = True
    """Inject StackOne usage instructions into the system prompt."""

    metadata: Mapping[str, object] | None = None
    """Metadata merged onto every tool, available to tool-selection machinery such as
    `CodeMode(tools={'code_mode': True})` or custom `prepare_tools` hooks."""

    client: MCPToolsetClient | None = field(default=None, repr=False)
    """Replacement for the default `{base_url}/mcp` connection. URL values must use HTTPS;
    prebuilt clients keep their own transport, auth, and account selection, so `account_id`
    is not applied to them."""

    api_key: str | None = field(default=None, repr=False)
    """Deprecated: renamed to `auth`."""

    def __post_init__(self) -> None:
        if self.api_key is not None:
            if self.auth is not None:
                raise UserError('Pass `auth` only: `api_key` is its deprecated name.')
            warn_argument_renamed('StackOne', 'api_key', 'auth', stacklevel=4)
            self.auth, self.api_key = self.api_key, None
        self.tool_mode, self.actions = validate_configuration(self.tool_mode, self.actions)
        self.id = self._derived_id()

    def get_toolset(self) -> AbstractToolset[AgentDepsT]:
        """Build the StackOne toolset."""
        if callable(self.auth):
            # Registered once under a fixed `id`, as durable execution requires; filled per run.
            return DynamicToolset(self._connect_for_run, per_run_step=False, id=self._derived_id())
        return self._connect(self.auth)

    def _connect_for_run(self, ctx: RunContext[AgentDepsT]) -> StackOneToolset[AgentDepsT] | None:
        auth = self.auth(ctx) if callable(self.auth) else self.auth
        return self._connect(auth) if auth else None

    def _connect(self, api_key: str | None) -> StackOneToolset[AgentDepsT]:
        return StackOneToolset[AgentDepsT](
            account_id=self.account_id,
            api_key=api_key,
            base_url=self.base_url,
            actions=self.actions,
            tool_mode=self.tool_mode,
            metadata=self.metadata,
            client=self.client,
            id=self._derived_id(),
        )

    def _derived_id(self) -> str:
        """This capability's `id`, falling back to the one the account names."""
        return self.id if self.id is not None else f'{_ID_PREFIX}-{self.account_id}'

    def get_instructions(self) -> str | None:
        """StackOne usage guidance; the underlying MCP toolset provides none itself."""
        if not self.include_instructions:
            return None
        mode = resolve_tool_mode(self.tool_mode, self.actions)
        return _SEARCH_EXECUTE_INSTRUCTIONS if mode == 'search_execute' else _INDIVIDUAL_INSTRUCTIONS

    @classmethod
    def from_spec(
        cls,
        account_id: str,
        *,
        id: str | None = None,
        description: str | None = _DEFAULT_DESCRIPTION,
        defer_loading: bool = False,
        auth: str | None = None,
        base_url: str = STACKONE_BASE_URL,
        actions: str | Sequence[str] = (),
        tool_mode: ToolMode | None = None,
        include_instructions: bool = True,
        metadata: Mapping[str, object] | None = None,
        api_key: str | None = None,
    ) -> StackOne[AgentDepsT]:
        """Construct from serializable options, excluding the runtime-only `client`."""
        return cls(
            account_id=account_id,
            id=id,
            description=description,
            defer_loading=defer_loading,
            auth=auth,
            api_key=api_key,
            base_url=base_url,
            actions=actions,
            tool_mode=tool_mode,
            include_instructions=include_instructions,
            metadata=metadata,
        )

    @classmethod
    def get_serialization_name(cls) -> str:
        """Return the agent-spec capability name."""
        return 'StackOne'
