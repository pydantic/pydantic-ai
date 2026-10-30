"""Importing a name Pydantic AI V2 removed names its replacement instead of failing with a bare `cannot import name`."""

from __future__ import annotations

import importlib

import pytest

MIGRATION_URL = 'https://pydantic.dev/docs/ai/overview/migration/'

# The modules under test, each with the optional dependency it needs, or `None` when it has none.
MODULES = {
    'pydantic_ai.models.openai': 'openai',
    'pydantic_ai.mcp': 'fastmcp',
    'pydantic_ai.providers.google': 'google.genai',
    'pydantic_ai.usage': None,
    'pydantic_ai.output': None,
    'pydantic_ai.models': None,
}


def _import(module: str) -> object:
    if dependency := MODULES[module]:
        pytest.importorskip(dependency)
    return importlib.import_module(module)


@pytest.mark.parametrize(
    'module, name, replacement',
    [
        ('pydantic_ai.models.openai', 'OpenAIModel', '`OpenAIChatModel`'),
        ('pydantic_ai.models.openai', 'OpenAIModelSettings', '`OpenAIChatModelSettings`'),
        ('pydantic_ai.mcp', 'MCPServerStdio', '`MCPToolset`'),
        ('pydantic_ai.mcp', 'MCPServerSSE', '`MCPToolset`'),
        ('pydantic_ai.mcp', 'MCPServerStreamableHTTP', '`MCPToolset`'),
        ('pydantic_ai.mcp', 'MCPServerHTTP', '`MCPToolset`'),
        ('pydantic_ai.mcp', 'FastMCPToolset', '`MCPToolset`'),
        ('pydantic_ai.mcp', 'load_mcp_servers', '`load_mcp_toolsets`'),
        ('pydantic_ai.usage', 'Usage', '`RunUsage`'),
        ('pydantic_ai.output', 'DeferredToolCalls', '`pydantic_ai.tools.DeferredToolRequests`'),
        ('pydantic_ai.providers.google', 'GoogleGLAProvider', '`GoogleProvider`'),
        (
            'pydantic_ai.providers.google',
            'GoogleVertexProvider',
            '`pydantic_ai.providers.google_cloud.GoogleCloudProvider`',
        ),
        ('pydantic_ai.models', 'cached_async_http_client', '`create_async_http_client()`'),
    ],
)
def test_importing_a_removed_v1_name_names_its_replacement(module: str, name: str, replacement: str):
    _import(module)

    with pytest.raises(ImportError) as exc_info:
        exec(f'from {module} import {name}', {})

    message = str(exc_info.value)
    assert message.startswith(f'`{module}.{name}` was removed in Pydantic AI V2: ')
    assert replacement in message
    assert message.endswith(f'See {MIGRATION_URL}')


@pytest.mark.parametrize('module', list(MODULES))
def test_other_missing_names_still_raise_attribute_error(module: str):
    imported = _import(module)

    assert not hasattr(imported, 'NotARealName')
    with pytest.raises(ImportError, match="cannot import name 'NotARealName'"):
        exec(f'from {module} import NotARealName', {})
