# Brief: Move builtin_tools to Capability System

## Problem

`builtin_tools` is currently a separate field on `GraphAgentDeps` and a parameter on `Agent.__init__`. It should be handled through the capability system, like `history_processors` already is (wrapped in `HistoryProcessor` capability).

## Design

Create a `BuiltinTools` capability that:
- Takes builtin tool instances or definitions
- Should be able to take serialized `AbstractBuiltinToolDef` for spec support
- Returns them from `get_builtin_tools()`
- Note: PR #4770 is adding dedicated capabilities for specific builtin tools (e.g. WebSearch with fallback to local tools)

## Scope

- Create `BuiltinTools` capability class
- Handle the `Agent(builtin_tools=...)` parameter by wrapping in `BuiltinTools` capability (like history_processors → HistoryProcessor)
- Remove `builtin_tools` from `GraphAgentDeps` (or keep for backward compat)
- Coordinate with PR #4770

## Reference

- PR #4640 comment on `_agent_graph.py:136`
- PR #4770 — dedicated builtin tool capabilities
