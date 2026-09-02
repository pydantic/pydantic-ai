from __future__ import annotations as _annotations

from pydantic_ai._history_processor import HistoryProcessor
from pydantic_ai._tool_execution import process_tool_calls

from . import (
    graph as _graph,
    model_request as _model_request,
    model_response as _model_response,
    user_prompt as _user_prompt,
)
from .graph import (
    AgentNode as AgentNode,
    SetFinalResult as SetFinalResult,
    build_agent_graph as build_agent_graph,
    drain_node_event_stream as drain_node_event_stream,
    is_agent_node as is_agent_node,
)
from .history import (
    SYNTHESIZED_TOOL_RETURN_METADATA_KEY as SYNTHESIZED_TOOL_RETURN_METADATA_KEY,
    _clean_message_history as _clean_message_history,
    _dangling_tool_calls_by_response as _dangling_tool_calls_by_response,
    _first_new_message_index as _first_new_message_index,
    _first_run_id_index as _first_run_id_index,
    _repair_dangling_tool_calls as _repair_dangling_tool_calls,
    capture_run_messages as capture_run_messages,
    get_captured_run_messages as get_captured_run_messages,
)
from .model_call import (
    MAX_BACKGROUND_POLLS as MAX_BACKGROUND_POLLS,
    MAX_GENERATION_CONTINUATIONS as MAX_GENERATION_CONTINUATIONS,
    _check_continuation_usage as _check_continuation_usage,
    _resolve_interrupted_stream_state as _resolve_interrupted_stream_state,
    fill_response_cost as fill_response_cost,
    model_request as model_request,
    model_request_stream as model_request_stream,
)
from .model_request import ModelRequestNode as ModelRequestNode
from .model_response import CallToolsNode as CallToolsNode, _with_event_stream_buffer as _with_event_stream_buffer
from .state import (
    NEW_CONVERSATION as NEW_CONVERSATION,
    AgentGraphSleepFunc as AgentGraphSleepFunc,
    EndStrategy as EndStrategy,
    GraphAgentDeps as GraphAgentDeps,
    GraphAgentState as GraphAgentState,
    build_run_context as build_run_context,
    build_validation_context as build_validation_context,
    resolve_conversation_id as resolve_conversation_id,
    resolve_run_id as resolve_run_id,
    run_cancelled_snapshot as run_cancelled_snapshot,
    set_agent_graph_sleep as set_agent_graph_sleep,
)
from .user_prompt import UserPromptNode as UserPromptNode

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
