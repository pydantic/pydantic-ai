"""StackOne wire contract and toolset.

Wire contract, verified 2026-07-29:

- `POST {base_url}/mcp` -- MCP over streamable HTTP; lists and executes tools.
  `?tool-mode=search_execute` switches from one-tool-per-action to two
  search/execute meta-tools.
- Auth is `Authorization: Basic base64('{api_key}:')` plus an `x-account-id`
  header selecting the linked account. StackOne requires HTTPS.
- Tool names follow `{connector}_{action}_{entity}`, e.g.
  `bamboohr_list_employees`.
- Search/execute names end in `_search_actions` and `_execute_action`;
  the first returns runtime `action_id` values consumed by the second.

Sources: https://docs.stackone.com/mcp/quickstart and
https://docs.stackone.com/mcp/auth-security. Re-check with the documented
initialize and tools/list requests in both tool modes.
"""

from __future__ import annotations

import base64
import os
from collections.abc import Mapping, Sequence
from dataclasses import replace
from fnmatch import fnmatch
from typing import Literal
from urllib.parse import parse_qsl, urlsplit, urlunsplit

from pydantic import AnyUrl
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import ToolsetTool

try:
    from pydantic_ai.mcp import MCPToolset, MCPToolsetClient
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'MCP support is required for the StackOne capability. Install it with: uv add "pydantic-ai-harness[stackone]"'
    ) from _import_error

__all__ = (
    # Re-exported so `_capability` can take it from here instead of repeating the optional-import
    # guard above. Naming it makes the re-export explicit, which is what `reportPrivateImportUsage`
    # asks for; without it the import only type-checks when Pyright happens to analyze this module
    # in the same invocation as the one importing from it.
    'MCPToolsetClient',
    'STACKONE_API_KEY_ENV',
    'STACKONE_BASE_URL',
    'StackOneToolset',
    'ToolMode',
)

STACKONE_API_KEY_ENV = 'STACKONE_API_KEY'
"""Environment variable consulted when `api_key` is not passed explicitly."""

STACKONE_BASE_URL = 'https://api.stackone.com'
"""Default StackOne API host."""

ToolMode = Literal['individual', 'search_execute']
"""How StackOne registers tools: one tool per action, or two search/execute meta-tools."""

_MCP_PATH = '/mcp'
_SEARCH_EXECUTE_QUERY = 'tool-mode=search_execute'


def validate_configuration(
    tool_mode: ToolMode | None, actions: str | Sequence[str]
) -> tuple[ToolMode | None, tuple[str, ...]]:
    if tool_mode not in (None, 'individual', 'search_execute'):
        raise UserError('`tool_mode` must be `individual`, `search_execute`, or `None`.')

    resolved_actions = (actions,) if isinstance(actions, str) else tuple(actions)

    if tool_mode == 'search_execute' and resolved_actions:
        raise UserError(
            '`actions` filters cannot apply in `search_execute` mode because that mode registers '
            'only the search and execute tools. Use `individual` mode to filter action tools.'
        )
    return tool_mode, resolved_actions


def resolve_tool_mode(tool_mode: ToolMode | None, actions: Sequence[str]) -> ToolMode:
    """Resolve the default tool mode: `search_execute`, or `individual` when `actions` are given.

    `search_execute` keeps the prompt footprint constant regardless of catalog
    size (provider catalogs can exceed model context windows in `individual`
    mode), while `actions` globs only apply to individually registered tools.
    """
    if tool_mode is not None:
        return tool_mode
    return 'individual' if actions else 'search_execute'


def resolve_api_key(api_key: str | None) -> str:
    """Return the given API key, or the one from `STACKONE_API_KEY`.

    Raises:
        UserError: If neither is set.
    """
    resolved = api_key or os.environ.get(STACKONE_API_KEY_ENV)
    if not resolved:
        raise UserError(
            f'A StackOne API key is required: pass `api_key` or set the `{STACKONE_API_KEY_ENV}` environment variable.'
        )
    return resolved


def _basic_auth(api_key: str) -> str:
    token = base64.b64encode(f'{api_key}:'.encode()).decode()
    return f'Basic {token}'


def _with_tool_mode(url: str, tool_mode: ToolMode) -> str:
    parts = urlsplit(url)
    tool_mode_values = [value for key, value in parse_qsl(parts.query, keep_blank_values=True) if key == 'tool-mode']
    if tool_mode_values:
        if tool_mode_values == [tool_mode]:
            return url
        raise UserError(
            "The URL's `tool-mode` conflicts with the configured `tool_mode`; the URL is not rewritten because "
            'rewriting would invalidate signed URLs.'
        )
    if tool_mode == 'individual':
        return url
    query = f'{parts.query}&{_SEARCH_EXECUTE_QUERY}' if parts.query else _SEARCH_EXECUTE_QUERY
    return urlunsplit(parts._replace(query=query))


def _validate_https_url(url: str, *, name: str) -> None:
    parts = urlsplit(url)
    if parts.scheme.lower() != 'https' or parts.hostname is None:
        raise UserError(f'`{name}` must be an absolute HTTPS URL.')


class StackOneToolset(MCPToolset[AgentDepsT]):
    """StackOne actions on one linked SaaS account, as an agent toolset.

    An `MCPToolset` connected to StackOne's MCP endpoint. URL clients receive
    StackOne auth and account headers. Action filtering and metadata merging
    apply to the tools listed by the server. Prebuilt clients keep their own
    transport configuration. Instances support the full `MCPToolset` surface.

    Use the `StackOne` capability for usage instructions and agent-spec support.
    Use this class for toolset combinators such as `approval_required()`.
    """

    def __init__(
        self,
        *,
        account_id: str,
        api_key: str | None = None,
        base_url: str = STACKONE_BASE_URL,
        actions: Sequence[str] = (),
        tool_mode: ToolMode | None = None,
        metadata: Mapping[str, object] | None = None,
        client: MCPToolsetClient | None = None,
        id: str = 'stackone',
    ) -> None:
        """Build a StackOne MCP toolset.

        Args:
            account_id: Linked account used for StackOne requests.
            api_key: API key, or `STACKONE_API_KEY` when omitted.
            base_url: HTTPS StackOne API host.
            actions: Case-insensitive globs over individual action tool names. Selects
                `individual`; incompatible with explicit `search_execute`.
            tool_mode: Individual tools or the search/execute pair. Inferred when omitted.
            metadata: Metadata merged onto each tool definition.
            client: URL, `FastMCP`, or prebuilt client accepted by `MCPToolset`.
                Non-URL clients keep their own transport, auth, and account selection,
                so `account_id` is not applied to them.
            id: Toolset ID; use distinct values for multiple accounts.
        """
        tool_mode, actions = validate_configuration(tool_mode, actions)
        mode = resolve_tool_mode(tool_mode, actions)
        if client is None:
            _validate_https_url(base_url, name='base_url')
            parts = urlsplit(base_url)
            if parts.query or parts.fragment:
                raise UserError('`base_url` must not contain a query or fragment.')
            resolved: MCPToolsetClient = urlunsplit(parts._replace(path=f'{parts.path.rstrip("/")}{_MCP_PATH}'))
        else:
            resolved = client
        headers: dict[str, str] | None = None
        is_url = isinstance(resolved, AnyUrl) or (
            isinstance(resolved, str) and urlsplit(resolved).scheme.lower() in ('http', 'https')
        )
        if is_url:
            _validate_https_url(str(resolved), name='client')
            resolved = _with_tool_mode(str(resolved), mode)
            headers = {'Authorization': _basic_auth(resolve_api_key(api_key)), 'x-account-id': account_id}
        super().__init__(resolved, id=id, headers=headers)
        self._lowered_actions: tuple[str, ...] = tuple(glob.lower() for glob in actions)
        self._metadata = metadata

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        tools = await super().get_tools(ctx)
        if self._lowered_actions:
            tools = {
                name: tool
                for name, tool in tools.items()
                if any(fnmatch(tool.tool_def.name.lower(), glob) for glob in self._lowered_actions)
            }
        if self._metadata:
            tools = {
                name: replace(
                    tool,
                    tool_def=replace(
                        tool.tool_def,
                        metadata={**(tool.tool_def.metadata or {}), **self._metadata},
                    ),
                )
                for name, tool in tools.items()
            }
        return tools
