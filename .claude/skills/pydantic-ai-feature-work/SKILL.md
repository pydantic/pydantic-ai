---
name: pydantic-ai-feature-work
description: Use before nontrivial Pydantic AI slim work that changes public APIs, provider behavior, capabilities, toolsets, output/tools, message history, streaming, UI adapters, or durable execution compatibility.
allowed-tools: Read, Glob, Grep, Bash(gh issue view:*), Bash(gh pr view:*), Bash(gh api:*)
---

# Pydantic AI Feature Work

Load this skill when a task is likely to affect the shape of `pydantic-ai-slim`, not for tiny mechanical edits.

## Read First

Read these references before planning or editing:

- `references/feature-map.md` — user-facing feature surface and v2 execution direction.
- `references/internals-model.md` — implementation architecture and layer ownership.
- `references/maintainer-mindset.md` — maintainer review heuristics extracted from recent PRs.

Also read the relevant local instructions:

- `agent_docs/index.md`
- nearest `AGENTS.md` files for the changed paths
- related docs pages for the feature surface being changed

## Dispatch Pattern

For nontrivial changes, delegate focused research before implementation:

- Provider/API work: one agent checks provider docs and SDK types; one reads existing provider/profile patterns.
- Tool/output/capability work: one agent traces graph/tool/output internals; one checks docs/tests/snapshots that define the public contract.
- Durable/UI/MCP work: one agent checks downstream adapters and durable wrappers; one checks message serialization/replay expectations.
- Large branch review: use `review-branch` rather than a single broad review pass.

Each subagent should receive only the files, PR/issue links, and reference sections it needs. Do not pass the whole conversation unless the subtask requires it.

## Design Checks

Before editing, answer these briefly:

- Which abstraction owns the behavior: `Agent`, graph, toolset, capability, model/provider, profile, message model, UI adapter, or durable wrapper?
- Which public contract changes: API, docs, message history, event stream, provider request/response shape, or serialized spec?
- Which compatibility targets need checking: provider parity, durable execution, UI adapters, MCP/toolsets, streaming, or v2 migrations?
- What tests should pin the behavior: public API integration, provider cassette, message/event snapshot, docs example, or durable workflow?

If the answer is unclear, research or ask before implementing.
