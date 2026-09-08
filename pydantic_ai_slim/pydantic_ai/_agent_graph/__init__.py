# pyright: reportUnusedImport=false
# ruff: noqa: F401

from __future__ import annotations as _annotations

# This package replaced the former `_agent_graph.py` module. Keep its attributes available so
# moving the implementation into focused modules remains a behavior-preserving refactor.
from pydantic_ai._history_processor import HistoryProcessor
from pydantic_ai._tool_execution import process_tool_calls

from . import (
    graph as _graph,
    model_request as _model_request,
    model_response as _model_response,
    user_prompt as _user_prompt,
)
from .graph import (
    AgentNode,
    SetFinalResult,
    build_agent_graph,
    drain_node_event_stream,
    is_agent_node,
)
from .history import (
    SYNTHESIZED_TOOL_RETURN_METADATA_KEY,
    _clean_message_history,
    _dangling_tool_calls_by_response,
    _first_new_message_index,
    _first_run_id_index,
    _repair_dangling_tool_calls,
    capture_run_messages,
    get_captured_run_messages,
)
from .model_call import (
    MAX_BACKGROUND_POLLS,
    MAX_GENERATION_CONTINUATIONS,
    _check_continuation_usage,
    _resolve_interrupted_stream_state,
    fill_response_cost,
    model_request,
    model_request_stream,
)
from .model_request import ModelRequestNode
from .model_response import CallToolsNode, _with_event_stream_buffer
from .state import (
    NEW_CONVERSATION,
    AgentGraphSleepFunc,
    EndStrategy,
    GraphAgentDeps,
    GraphAgentState,
    build_run_context,
    build_validation_context,
    resolve_conversation_id,
    resolve_run_id,
    run_cancelled_snapshot,
    set_agent_graph_sleep,
)
from .user_prompt import UserPromptNode

# The modules avoid importing one another at initialization time, but their postponed annotations
# still need the cross-node names when users resolve them at runtime.
_graph.ModelRequestNode = ModelRequestNode
_graph.CallToolsNode = CallToolsNode
_graph.UserPromptNode = UserPromptNode
_model_request.CallToolsNode = CallToolsNode
_model_response.ModelRequestNode = ModelRequestNode
_user_prompt.ModelRequestNode = ModelRequestNode
_user_prompt.CallToolsNode = CallToolsNode

__all__ = (
    'GraphAgentState',
    'GraphAgentDeps',
    'UserPromptNode',
    'ModelRequestNode',
    'CallToolsNode',
    'build_run_context',
    'capture_run_messages',
    'HistoryProcessor',
    'resolve_conversation_id',
    'process_tool_calls',
    'resolve_run_id',
)
