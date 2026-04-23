from __future__ import annotations

import importlib.metadata
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

from pydantic.json_schema import JsonSchemaValue

from pydantic_ai.tools import Tool
from pydantic_ai.toolsets.function import FunctionToolset

try:
    from stackone_ai import StackOneToolSet
except ImportError as _import_error:
    raise ImportError('Please install `stackone-ai` to use StackOne tools.') from _import_error

if TYPE_CHECKING:
    from stackone_ai import ExecuteToolsConfig, SearchConfig
    from stackone_ai.models import StackOneTool

__all__ = ('tool_from_stackone', 'StackOneToolset')

_SEARCH_AND_EXECUTE_MIN_VERSION = (2, 5, 0)


def _require_search_and_execute_support() -> None:
    """Raise `ImportError` if the installed `stackone-ai` is too old for `mode='search_and_execute'`."""
    try:
        version = importlib.metadata.version('stackone-ai')
    except importlib.metadata.PackageNotFoundError:
        return
    parts = tuple(int(p) for p in version.split('.')[:3] if p.isdigit())
    if parts < _SEARCH_AND_EXECUTE_MIN_VERSION:
        raise ImportError(
            f"mode='search_and_execute' requires `stackone-ai>=2.5.0` (found {version}). "
            'Install with `pip install -U stackone-ai`.'
        )


def _resolve_account_ids(
    account_ids: str | list[str] | None,
    *,
    required: bool = True,
) -> list[str]:
    """Return account IDs from the explicit arg or `STACKONE_ACCOUNT_ID` env var.

    Accepts a single string, a list of strings, or `None`. Raises `ValueError`
    when nothing is provided and `required=True`.
    """
    if isinstance(account_ids, str):
        return [account_ids]
    if account_ids is not None:
        return list(account_ids)
    env = os.getenv('STACKONE_ACCOUNT_ID')
    if env:
        return [env]
    if required:
        raise ValueError(
            'StackOne account ID(s) required. '
            "Pass `account_ids='acct-1'` or `account_ids=['acct-1', 'acct-2']`, "
            'or set the `STACKONE_ACCOUNT_ID` environment variable.'
        )
    return []


def _tool_from_stackone_tool(stackone_tool: StackOneTool) -> Tool:
    openai_function = stackone_tool.to_openai_function()
    json_schema: JsonSchemaValue = openai_function['function']['parameters']

    def implementation(**kwargs: Any) -> Any:
        return stackone_tool.execute(kwargs)

    return Tool.from_schema(
        function=implementation,
        name=stackone_tool.name,
        description=stackone_tool.description or '',
        json_schema=json_schema,
    )


def tool_from_stackone(
    tool_name: str,
    *,
    account_ids: str | list[str] | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Tool:
    """Creates a Pydantic AI tool proxy from a StackOne tool.

    Args:
        tool_name: The name of the StackOne tool to wrap (e.g., `"workday_list_workers"`).
        account_ids: One or more StackOne account IDs. Pass a string for a single account
            or a list for multiple. If not provided, uses the `STACKONE_ACCOUNT_ID`
            environment variable.
        api_key: The StackOne API key. If not provided, uses the `STACKONE_API_KEY`
            environment variable.
        base_url: Optional custom base URL for the StackOne API.

    Returns:
        A Pydantic AI tool that corresponds to the StackOne tool.
    """
    resolved = _resolve_account_ids(account_ids)
    stackone_toolset = StackOneToolSet(api_key=api_key, base_url=base_url)
    stackone_toolset.set_accounts(resolved)
    tools = stackone_toolset.fetch_tools(actions=[tool_name])
    stackone_tool = tools.get_tool(tool_name)
    if stackone_tool is None:
        raise ValueError(f"Tool '{tool_name}' not found in StackOne")
    return _tool_from_stackone_tool(stackone_tool)


class StackOneToolset(FunctionToolset):
    """A toolset that wraps StackOne tools."""

    def __init__(
        self,
        tools: Sequence[str] | None = None,
        *,
        account_ids: str | list[str] | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        filter_pattern: str | list[str] | None = None,
        mode: Literal['search_and_execute'] | None = None,
        search_config: SearchConfig | None = None,
        execute_config: ExecuteToolsConfig | None = None,
        id: str | None = None,
    ):
        if mode != 'search_and_execute' and (search_config is not None or execute_config is not None):
            raise ValueError("'search_config' and 'execute_config' require mode='search_and_execute'")

        if mode == 'search_and_execute':
            _require_search_and_execute_support()
            if tools is not None or filter_pattern is not None:
                raise ValueError("Cannot combine mode='search_and_execute' with 'tools' or 'filter_pattern'")
            has_execute_accounts = execute_config is not None and 'account_ids' in execute_config
            if has_execute_accounts and account_ids is not None:
                raise ValueError("Cannot specify both 'account_ids' and 'execute_config[\"account_ids\"]'")
            stackone_toolset = StackOneToolSet(
                api_key=api_key,
                base_url=base_url,
                search=search_config or {},
                execute=execute_config,
            )
            if not has_execute_accounts:
                stackone_toolset.set_accounts(_resolve_account_ids(account_ids))
            # TODO: switch to a public API once `stackone-ai` exposes native support for pydantic tools.
            meta_tools = stackone_toolset._build_tools()  # pyright: ignore[reportPrivateUsage]
            pydantic_tools = [_tool_from_stackone_tool(t) for t in meta_tools]
            super().__init__(pydantic_tools, id=id)
            return

        if tools is not None and filter_pattern is not None:
            raise ValueError("Cannot specify both 'tools' and 'filter_pattern'")

        resolved = _resolve_account_ids(account_ids)
        stackone_toolset = StackOneToolSet(api_key=api_key, base_url=base_url)
        stackone_toolset.set_accounts(resolved)

        if tools is not None:
            actions = list(tools)
        elif isinstance(filter_pattern, str):
            actions = [filter_pattern]
        else:
            actions = filter_pattern

        fetched_tools = stackone_toolset.fetch_tools(actions=actions)
        pydantic_tools = [_tool_from_stackone_tool(tool) for tool in fetched_tools]
        super().__init__(pydantic_tools, id=id)
