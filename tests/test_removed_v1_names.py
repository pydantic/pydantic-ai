"""Importing a name Pydantic AI V2 removed names its replacement instead of failing with a bare `cannot import name`."""

from __future__ import annotations

import importlib

import pytest

MIGRATION_URL = 'https://pydantic.dev/docs/ai/overview/migration/'

# The modules under test, each with the optional dependency it needs, or `None` when it has none.
MODULES = {
    'pydantic_ai.models.openai': 'openai',
    'pydantic_ai.mcp': 'fastmcp',
    'pydantic_ai.usage': None,
    'pydantic_ai.output': None,
    'pydantic_ai.models': None,
}


def _import(module: str) -> object:
    if dependency := MODULES[module]:
        pytest.importorskip(dependency)
    return importlib.import_module(module)


@pytest.mark.parametrize(
    'module, name, advice',
    [
        (
            'pydantic_ai.models.openai',
            'OpenAIModel',
            'use `OpenAIChatModel` (the same Chat Completions API), or `OpenAIResponsesModel`.',
        ),
        ('pydantic_ai.models.openai', 'OpenAIModelSettings', 'use `OpenAIChatModelSettings`.'),
        *(
            (
                'pydantic_ai.mcp',
                name,
                'use `MCPToolset`, which picks the transport from its arguments; its timeout and retry defaults differ.',
            )
            for name in ('MCPServerStdio', 'MCPServerSSE', 'MCPServerStreamableHTTP', 'MCPServerHTTP')
        ),
        ('pydantic_ai.mcp', 'load_mcp_servers', 'use `load_mcp_toolsets`.'),
        ('pydantic_ai.usage', 'Usage', "use `RunUsage` for a run's usage, or `RequestUsage` for a single request's."),
        (
            'pydantic_ai.output',
            'DeferredToolCalls',
            'use `pydantic_ai.tools.DeferredToolRequests`, whose `calls` replaces `tool_calls`.',
        ),
        ('pydantic_ai.models', 'cached_async_http_client', 'use `create_async_http_client()`.'),
    ],
)
def test_importing_a_removed_v1_name_names_its_replacement(module: str, name: str, advice: str):
    _import(module)

    with pytest.raises(ImportError) as exc_info:
        exec(f'from {module} import {name}', {})

    assert str(exc_info.value) == f'`{module}.{name}` was removed in Pydantic AI V2: {advice} See {MIGRATION_URL}'


@pytest.mark.parametrize('module', list(MODULES))
def test_other_missing_names_still_raise_attribute_error(module: str):
    imported = _import(module)

    assert not hasattr(imported, 'NotARealName')
    with pytest.raises(ImportError, match="cannot import name 'NotARealName'"):
        exec(f'from {module} import NotARealName', {})
