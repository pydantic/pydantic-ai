from __future__ import annotations as _annotations

from . import ModelProfile, merge_profile
from .anthropic import AnthropicModelProfile, anthropic_model_profile


def claude_code_model_profile(model_name: str) -> ModelProfile:
    """Get the model profile for Claude models used through Claude Code with the user's own Claude login.

    Claude Code sends the Messages API request itself, from a system prompt, tool list and settings fixed when its
    session starts, so the request features it doesn't let a client control are turned off: cache breakpoints (it
    places its own), mid-conversation and request-scoped system messages, forced tool choice, native JSON schema output,
    Anthropic's server-side tools, file and container IDs, and Anthropic's batch, thread, compaction, inference geo,
    fast mode and task budget features. Documents in a user prompt reach the model, but a tool's result reaches Claude Code
    over MCP, which carries only text and images, so a document a tool returns is replaced by a note.

    Claude Code turns thinking off with its own `--thinking disabled` flag, including on the models whose API takes
    `{'type': 'between_tools'}` for that instead.
    """
    return merge_profile(
        anthropic_model_profile(model_name),
        AnthropicModelProfile(
            supports_cache=False,
            supports_auto_cache=False,
            supported_cache_retentions=(),
            supports_inline_system_prompts=False,
            supports_request_scoped_system_prompts=False,
            supports_mid_conversation_effort=False,
            supports_forced_tool_choice=False,
            supports_forced_tool_choice_with_thinking=False,
            supports_json_schema_output=False,
            supports_document_input=False,
            supported_native_tools=frozenset(),
            tool_addition_mode=None,
            tool_deferral_mode=None,
            anthropic_supports_on_demand_compaction=False,
            anthropic_supports_message_batches=False,
            anthropic_supports_message_threads=False,
            anthropic_supports_inference_geo=False,
            anthropic_supports_fast_speed=False,
            anthropic_supports_task_budgets=False,
            anthropic_binds_thinking_blocks=False,
            anthropic_accepts_file_and_container_ids=False,
            anthropic_count_tokens_accepts_server_tools=False,
            anthropic_disables_thinking_with_between_tools=False,
        ),
    )
