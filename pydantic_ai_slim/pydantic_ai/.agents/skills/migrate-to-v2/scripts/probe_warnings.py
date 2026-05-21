"""Probe v1.100.0 to capture EXACT deprecation warning text for every codemod
row in references/DEPRECATIONS.md. Prints a JSON-ish record of (row_id, message).

Run with the v1 venv:
    .venv-v1/bin/python probe_warnings.py
"""

from __future__ import annotations

import asyncio
import warnings
from typing import Any


def probe(name: str, fn) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            res = fn()
            if asyncio.iscoroutine(res):
                asyncio.get_event_loop().run_until_complete(res)
        except Exception as e:  # noqa: BLE001
            print(f'[{name}] EXC {type(e).__name__}: {e}')
        msgs = [str(w.message) for w in caught if 'pydantic' in str(w.filename).lower() or True]
        # Filter to pydantic_ai-originated warnings by category name when possible
        rel = [str(w.message) for w in caught
               if 'Deprecation' in type(w.message).__name__ or 'deprecated' in str(w.message).lower()]
        if rel:
            for m in rel:
                print(f'[{name}] {m}')
        else:
            print(f'[{name}] (no deprecation warning fired)')


from pydantic_ai import Agent


def _agent_instrument():
    return Agent('openai:gpt-4o', instrument=True)


def _agent_history_processors():
    return Agent('openai:gpt-4o', history_processors=[lambda h: h])


def _agent_prepare_tools():
    return Agent('openai:gpt-4o', prepare_tools=lambda ctx, defs: defs)


def _agent_prepare_output_tools():
    return Agent('openai:gpt-4o', prepare_output_tools=lambda ctx, defs: defs)


def _agent_event_stream_handler():
    async def h(ctx, ev):
        return None
    return Agent('openai:gpt-4o', event_stream_handler=h)


def _agent_tool_retries():
    return Agent('openai:gpt-4o', tool_retries=3)


def _agent_output_retries():
    return Agent('openai:gpt-4o', output_retries=3)


def _agent_mcp_servers():
    from pydantic_ai.mcp import MCPServerStdio
    return Agent('openai:gpt-4o', mcp_servers=[MCPServerStdio('python', args=['-c', 'pass'])])


def _agent_bare_model_name():
    return Agent('gpt-4o')


def _agent_openai_string():
    # The "openai:" string itself is not deprecated; only the behavior flips in v2.
    return Agent('openai:gpt-4o')


def _agent_gateway_gemini():
    return Agent('gateway/gemini:gemini-1.5-pro')


def _openai_model():
    from pydantic_ai.models.openai import OpenAIModel
    return OpenAIModel('gpt-4o')


def _gemini_model():
    from pydantic_ai.models.gemini import GeminiModel
    return GeminiModel('gemini-1.5-pro')


def _google_gla_provider():
    from pydantic_ai.providers.google_gla import GoogleGLAProvider
    return GoogleGLAProvider(api_key='x')


def _google_vertex_provider():
    try:
        from pydantic_ai.providers.google_vertex import GoogleVertexProvider
    except ImportError as e:
        print(f'  (google_vertex import error: {e})')
        return None
    return GoogleVertexProvider(project_id='x')


def _grok_provider():
    try:
        from pydantic_ai.providers.grok import GrokProvider
    except ImportError as e:
        print(f'  (grok import error: {e})')
        return None
    return GrokProvider(api_key='x')


def _mcp_server_stdio():
    from pydantic_ai.mcp import MCPServerStdio
    return MCPServerStdio('python', args=['-c', 'pass'])


def _mcp_server_sse():
    from pydantic_ai.mcp import MCPServerSSE
    return MCPServerSSE('http://localhost:8000/sse')


def _mcp_server_streamable_http():
    from pydantic_ai.mcp import MCPServerStreamableHTTP
    return MCPServerStreamableHTTP('http://localhost:8000/mcp')


def _mcp_server_http():
    from pydantic_ai.mcp import MCPServerHTTP
    return MCPServerHTTP('http://localhost:8000/sse')


def _fastmcp_toolset():
    try:
        from pydantic_ai.mcp import FastMCPToolset
    except ImportError as e:
        print(f'  (FastMCPToolset import: {e})')
        return None
    from fastmcp import Client  # noqa: F401
    return None


def _agent_to_a2a():
    a = Agent('openai:gpt-4o')
    return a.to_a2a()


def _agui_app():
    try:
        from pydantic_ai.ag_ui import AGUIApp
    except ImportError as e:
        print(f'  (AGUIApp import: {e})')
        return None
    a = Agent('openai:gpt-4o')
    return AGUIApp(a)


def _to_ag_ui():
    a = Agent('openai:gpt-4o')
    return a.to_ag_ui()


def _ag_ui_shim_import():
    import importlib
    return importlib.import_module('pydantic_ai.ag_ui')


def _outlines_model():
    try:
        from pydantic_ai.models.outlines import OutlinesModel
    except ImportError as e:
        print(f'  (OutlinesModel import: {e})')
        return None
    return OutlinesModel('gpt-4o')


def _aci_import():
    try:
        from pydantic_ai.ext.aci import tool_from_aci  # noqa: F401
    except ImportError as e:
        print(f'  (aci import: {e})')
        return None
    return None


def _usage_class():
    from pydantic_ai.usage import Usage
    u = Usage()
    return u


def _usage_request_tokens():
    from pydantic_ai.usage import RunUsage, RequestUsage
    u = RequestUsage()
    # touch deprecated field
    _ = u.request_tokens
    return u


def _vendor_details():
    from pydantic_ai.messages import ModelResponse, TextPart
    r = ModelResponse(parts=[TextPart(content='x')])
    _ = r.vendor_details
    _ = r.vendor_id
    return r


def _function_tool_call_id():
    from pydantic_ai.messages import FunctionToolCallEvent, ToolCallPart
    ev = FunctionToolCallEvent(part=ToolCallPart(tool_name='x', args={}, tool_call_id='t1'))
    _ = ev.call_id
    return ev


def _dataset_no_name():
    from pydantic_evals import Dataset, Case
    return Dataset(cases=[Case(name='c', inputs='i', expected_output='o')], evaluators=[])


def _graph_top_level():
    from pydantic_graph import Graph  # noqa: F401
    return None


def _graph_beta():
    try:
        from pydantic_graph.beta.decision import Decision  # noqa: F401
    except ImportError as e:
        print(f'  (graph.beta import: {e})')
        return None
    return None


def _deferred_tool_calls():
    try:
        from pydantic_ai.output import DeferredToolCalls
    except ImportError as e:
        print(f'  (DeferredToolCalls: {e})')
        return None
    return DeferredToolCalls(tool_calls=[])


def _deferred_toolset():
    try:
        from pydantic_ai.toolsets import DeferredToolset
    except ImportError as e:
        print(f'  (DeferredToolset: {e})')
        return None
    return DeferredToolset(tool_defs=[])


def _history_processor_alias():
    try:
        from pydantic_ai import HistoryProcessor  # noqa: F401
    except ImportError as e:
        print(f'  (HistoryProcessor: {e})')
        return None
    return None


def _cached_http_client():
    try:
        from pydantic_ai.providers import cached_async_http_client
    except ImportError as e:
        print(f'  (cached_async_http_client: {e})')
        return None
    return cached_async_http_client()


PROBES = [
    ('A1_instrument', _agent_instrument),
    ('A2_history_processors', _agent_history_processors),
    ('A3_prepare_tools', _agent_prepare_tools),
    ('A4_prepare_output_tools', _agent_prepare_output_tools),
    ('A5_event_stream_handler', _agent_event_stream_handler),
    ('A6a_tool_retries', _agent_tool_retries),
    ('A6b_output_retries', _agent_output_retries),
    ('A7_mcp_servers', _agent_mcp_servers),
    ('B1_openai_model', _openai_model),
    ('B2_gemini_model', _gemini_model),
    ('B3a_gla_provider', _google_gla_provider),
    ('B3b_vertex_provider', _google_vertex_provider),
    ('B3c_gateway_gemini', _agent_gateway_gemini),
    ('B4_bare_model_name', _agent_bare_model_name),
    ('B5_grok_provider', _grok_provider),
    ('D1_mcp_stdio', _mcp_server_stdio),
    ('D2_mcp_sse', _mcp_server_sse),
    ('D3_mcp_streamable', _mcp_server_streamable_http),
    ('D4_mcp_http', _mcp_server_http),
    ('D5_fastmcp_toolset', _fastmcp_toolset),
    ('E1_to_a2a', _agent_to_a2a),
    ('E2a_agui_app', _agui_app),
    ('E2b_to_ag_ui', _to_ag_ui),
    ('E2c_ag_ui_shim', _ag_ui_shim_import),
    ('E3_outlines_model', _outlines_model),
    ('E4_aci_import', _aci_import),
    ('F1_usage_class', _usage_class),
    ('F1b_request_tokens', _usage_request_tokens),
    ('F2_vendor_details', _vendor_details),
    ('F3_call_id', _function_tool_call_id),
    ('J1_dataset_no_name', _dataset_no_name),
    ('I1_graph_top_level', _graph_top_level),
    ('I2_graph_beta', _graph_beta),
    ('H1_deferred_tool_calls', _deferred_tool_calls),
    ('H2_deferred_toolset', _deferred_toolset),
    ('H3_history_processor_alias', _history_processor_alias),
    ('H4_cached_http_client', _cached_http_client),
]


def main() -> None:
    for name, fn in PROBES:
        try:
            probe(name, fn)
        except Exception as e:  # noqa: BLE001
            print(f'[{name}] OUTER-EXC {type(e).__name__}: {e}')


if __name__ == '__main__':
    main()
