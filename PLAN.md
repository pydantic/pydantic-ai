# Capabilities documentation improvements

Doc gaps surfaced during the #4640 review, greenlit by DouweM as worth addressing.
Each item below is a self-contained improvement — pick any subset.

## 1. Document `AbstractCapability[AgentDepsT]` generic parameter

**Gap:** every example uses `AbstractCapability[Any]`. Users don't know what the type param is, how it relates to the agent's deps, or what happens on mismatch.

**DouweM:** "We should make it clear it's generic in the agent's dependency type. `AbstractToolset` is as well, not sure if it's explicitly documented there."

**Scope:** short paragraph + typed example in `docs/capabilities.md`.

## 2. Show examples for `BuiltinTool`, `Toolset`, `HistoryProcessor` capabilities

**Gap:** listed in the built-in table but zero usage examples beyond the one-line row.

**DouweM:** "Yeah we should show examples, although the API docs should make it pretty clear how to use them. In #4828 I'm also making the `BuiltinOrLocalTool` capability public."

**Scope:** small code example per capability in `docs/capabilities.md`. Coordinate with #4828 landing.

## 3. Document `ImageGeneration` local fallback

**Gap:** `WebSearch` auto-detects DuckDuckGo, `MCP` auto-detects transport — but what does `ImageGeneration(local=...)` expect?

**DouweM:** "Any user tool that the agent should use instead of builtin image generation. That could be a tool that directly uses an image generation model. Would be worth mentioning explicitly."

**Scope:** one paragraph + example in the `ImageGeneration` section.

## 4. Verify `for_run` -> `get_toolset`/`get_instructions` ordering

**Gap:** if a capability returns a toolset *and* overrides `for_run`, does the toolset come from the original or the fresh instance?

**DouweM:** "The intended behavior would be that `for_run` is called first, and then that new instance's `get_toolset` is used. Same for `get_instructions` etc; they should run on the new capability instance. Please verify if that's how it currently works, and let me know if it's not!"

**Scope:** code verification first, then doc clarification + fix if behavior doesn't match intent.

## 5. Document capability-to-capability communication

**Gap:** no mention of inter-capability coordination (e.g. cost tracker + budget limiter).

**DouweM:** "The easiest way is to create a contextvar and set it around the run (using before/after or wrap run hooks), and then other capabilities can read it. That means that the order of capabilities in the `capabilities` list matters: the contextvar writer should come before the contextvar reader. I'm not against adding explicit dependencies between capabilities. Through dependencies can make sense in your own codebase if you can control the deps type, but not for capabilities that are meant to be used with arbitrary agents / deps."

**Scope:** new subsection in `docs/capabilities.md` with contextvar pattern example.

## 6. Document skip exception chain behavior

**Gap:** `SkipModelRequest`, `SkipToolValidation`, `SkipToolExecution` get one sentence each. Users need to know: do other `before_*` hooks still fire? Do `after_*` hooks fire?

**DouweM:** "Good point, should be mentioned explicitly."

**Scope:** verify actual behavior in code, then document in the relevant hook sections.

## 7. Document `on_*_error` ordering and recovery semantics

**Gap:** when multiple capabilities define error hooks, which fires first? Does recovery short-circuit others?

**DouweM:** "I'm not sure if `on_*_error` is fired in capability order, or reverse order like `after_*` hooks are. Reverse order probably makes most sense since they're the failure counterpart to after hooks, but Claude would need to verify. I believe that recovery should result in other capabilities getting `after_*` called, instead of `on_*_error`, so that the recovery is 'complete'."

**Scope:** verify actual behavior in code, document explicitly, fix if behavior doesn't match DouweM's intent.

## 8. Capabilities on `agent.run`/`agent.override` (WIP)

**Gap:** capabilities can only be set at `Agent(...)` construction time.

**DouweM:** "Capabilities can't currently be passed into agent.run or agent.override directly, although it can be done indirectly via `spec=AgentSpec(capabilities=...)`. In #4807 I started working on letting them be overridden/disabled one by one."

**Scope:** no doc work needed now — will be addressed when #4807 lands.
