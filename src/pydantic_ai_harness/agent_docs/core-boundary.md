# Core Boundary

Harness should extend Pydantic AI through public primitives. It should not
quietly create a second runtime.

## Belongs In Pydantic AI Core

Start in Pydantic AI core when the change needs:

- new agent loop semantics
- new normalized message parts or message-history behavior
- provider API compatibility or provider wire mapping
- model/profile/provider capability facts
- generic tool execution semantics
- output-mode semantics
- durable execution primitives
- generic capability hooks or hook ordering changes
- MCP protocol behavior that should apply to all users

Harness can depend on these once core exposes the right primitive.

## Belongs In Harness

Harness is the right home for:

- reusable capability compositions
- coding-agent tools and repo workflows
- guardrails built on hooks
- memory and persistence policies
- context management policies
- tool-output handling policies
- planning, skills, task tracking, and sub-agent compositions
- opinionated defaults that are useful but not fundamental to every Pydantic AI
  user

## Decision Rule

If a feature would be hard for a third-party capability package to implement
correctly through public Pydantic AI APIs, do not work around that in harness.
Identify the missing core primitive and propose that change first.

If a feature is mostly a policy decision over existing hooks/toolsets/messages,
build it in harness.
