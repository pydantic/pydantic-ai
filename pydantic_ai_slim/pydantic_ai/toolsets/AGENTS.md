# toolsets/ Guidelines

Toolsets are reusable tool collections with lifecycle, instructions, and execution boundaries.

- Prefer wrapper toolsets for cross-cutting behavior such as filtering, prefixing, approval, deferral, metadata, or schema changes; avoid modifying every concrete toolset for behavior that can compose.
- Keep `get_tools`, `get_instructions`, lifecycle, and `call_tool` semantics aligned. If a wrapper forwards one of these, check whether it should forward the others.
- Preserve tool identity and stable naming across wrappers, MCP, deferred execution, message history, and durable engines.
- Return tool output through the tool result channel, never as system voice, and keep the tools the model is shown stable mid-run: today only revealing a deferred tool (through tool search or a capability load) is recorded in history, as a `ToolAvailabilityDeltaPart`, and any other change re-renders the tool list; once [#9405](https://github.com/pydantic/pydantic-ai/pull/9405) lands, every mid-run tool change is recorded as a delta. See [Message History & Prompt Caching](../../../agent_docs/message-history-and-caching.md).
- Do not put feature-specific state on `Agent` when a toolset or capability can own it.
- Test through public agent/toolset behavior where possible; snapshot message/tool-call history when behavior affects protocol shape.
