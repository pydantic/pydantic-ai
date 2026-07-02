# toolsets/ Guidelines

Toolsets are reusable tool collections with lifecycle, instructions, and execution boundaries.

- Prefer wrapper toolsets for cross-cutting behavior such as filtering, prefixing, approval, deferral, metadata, or schema changes; avoid modifying every concrete toolset for behavior that can compose.
- Keep `get_tools`, `get_instructions`, lifecycle, and `call_tool` semantics aligned. If a wrapper forwards one of these, check whether it should forward the others.
- Preserve tool identity and stable naming across wrappers, MCP, deferred execution, message history, and durable engines.
- Do not put feature-specific state on `Agent` when a toolset or capability can own it.
- Test through public agent/toolset behavior where possible; snapshot message/tool-call history when behavior affects protocol shape.
