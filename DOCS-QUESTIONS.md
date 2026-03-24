# Capabilities Docs — Uncovered User Questions

Questions users are likely to have that the current `docs/capabilities.md` doesn't cover (or barely touches).

## 1. What's the generic type parameter on `AbstractCapability[T]`?

Every example uses `AbstractCapability[Any]`. Users will wonder: what is `T`? How does it relate to the agent's dependency type? Can a capability declare that it needs specific dependencies, and what happens if the agent's dep type doesn't match?

## 2. Can capabilities be added per-run?

The docs only show capabilities at `Agent(...)` construction time. Users will want to know: can I pass extra capabilities to `agent.run()` or `agent.iter()`? Or override/disable specific ones for a single run?

## 3. `BuiltinTool`, `Toolset`, and `HistoryProcessor` — how to use them?

These three are listed in the built-in table but have zero documentation beyond the one-line row. A user reading top-to-bottom will see `PrepareTools` and `PrefixTools` get dedicated sections but these don't.

## 4. `ImageGeneration` local fallback

`WebSearch` auto-detects DuckDuckGo, `MCP` auto-detects transport — but what does `ImageGeneration(local=...)` expect? There's no example of providing a local image generation fallback.

## 5. When to use a capability vs direct agent parameters

The intro paragraph hints at this ('instructions and model settings are configured directly...'), but there's no decision framework. When should I use `Agent(instructions=...)` vs a capability that provides instructions? When should I register tools with `@agent.tool` vs bundling them in a capability?

## 6. Ordering guidance for composition

The composition section explains *mechanics* (before fires in order, after in reverse, wrap nests) but gives no *practical guidance*. If I have a guardrail and a logger, which should come first? What's the mental model for deciding order?

## 7. How does `for_run` interact with `get_toolset`?

`get_toolset` is called at agent construction, `for_run` is called per-run. If my capability returns a toolset *and* overrides `for_run`, does the toolset come from the original instance or the fresh one? This matters a lot for stateful toolsets.

## 8. Concurrency / thread safety

If the same agent is used for concurrent runs (common in web servers), what happens to shared capability instances? `for_run` helps, but the docs don't mention concurrency at all. Users need to know if `for_run` is required or just recommended for concurrent usage.

## 9. Can capabilities interact with each other?

No mention of inter-capability communication. If capability A needs to read state from capability B (e.g., a cost tracker that a budget limiter reads), how do they coordinate? Through dependencies? Through some other mechanism?

## 10. Skip exceptions and hook chain behavior

`SkipModelRequest`, `SkipToolValidation`, and `SkipToolExecution` are mentioned one sentence each. Users will want to know: when a skip is raised from `before_*`, do other capabilities' `before_*` hooks still fire? Do `after_*` hooks fire with the skip result? What about `wrap_*`?

## 11. Error hook ordering with multiple capabilities

When multiple capabilities define `on_*_error`, which fires first? If one recovers (returns a result), do the others still see the error? Or does recovery short-circuit?

## 12. Message history interaction

If `message_history` is passed to `agent.run()`, do `before_model_request` hooks see the historical messages in `request_context.messages`? Can they modify them?

## 13. Testing custom capabilities

No guidance on testing. Users building non-trivial capabilities (guardrails, approval workflows) will want patterns for: testing that hooks fire in the right order, testing error recovery, testing with `TestModel`, etc.

## 14. Complete lifecycle flow diagram

The error hook section has a small `before -> wrap -> after/on_error` ASCII diagram, but there's no full picture showing how all the hooks interact across a complete run (run -> node -> model request -> tool validate -> tool execute), especially with multiple capabilities composed.

## Assessment

Most of these are 'what happens when I combine X with Y' questions — the docs do a good job explaining each feature in isolation but leave composition, edge cases, and the full lifecycle under-documented. The biggest gaps are **#3** (undocumented builtins), **#1** (the type parameter), and **#5** (decision framework for when to use capabilities vs direct params).
