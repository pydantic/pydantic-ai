---
name: adding-a-provider-api-feature
description: Add a new provider API capability (prompt caching, strict/structured tool calling, thinking/reasoning effort, service tier, safety settings, logprobs, etc.) to Pydantic AI. Use when wiring a provider feature through the library — it enforces reasoning from the existing cross-provider abstraction before designing anything, and picking default-on vs opt-in deliberately. Not for adding a new model id (that's a different flow) or a bug fix.
user-invocable: true
allowed-tools: Bash(git:*), Bash(gh:*), Bash(rg:*), Bash(ls:*), Bash(uv:*), Read, Write, Edit, Glob, Grep, WebFetch, AskUserQuestion, Agent
---

# Adding a provider API feature

Use this when exposing a **new provider API capability** through Pydantic AI — prompt caching, strict/structured tool calling, thinking/reasoning effort, service tier, safety settings, logprobs, cache breakpoints, and the like. The output is a change that is *consistent with how sibling providers already expose the same concept*, defaults deliberately, and gates support with a capability flag.

Not for: adding a new model id (that's `add-new-model`), a bug fix, or a refactor.

## The one rule that prevents the most rework

**Before designing anything, find the existing cross-provider abstraction that governs this capability and let its shape decide the API.** Most "how should I expose this?" questions are already answered by an abstraction the codebase has — reaching for a new provider-specific knob when one exists is the single most common thing maintainers reject. When a feature routes through an existing abstraction, the abstraction's shape pre-decides the API surface, the opt-out, and often the default.

The tell that you skipped this: you find yourself listing 2-3 "options" for how a user controls the feature. If one of those options duplicates an existing cross-provider control, it isn't a real option — the existing abstraction wins.

## Step 0 — Enumerate sibling precedent

For the capability you're adding, list **how every provider that already has an analog exposes it**, and name the governing existing abstraction. It is one of:

- a **per-tool flag** — `ToolDefinition.strict: bool | None` (`tools.py`), resolved in `models/__init__.py::_customize_tool_def`;
- a **shared `ModelSettings` field** — `thinking`, `service_tier` (`settings.py`), each with per-model resolvers mapping to native concepts;
- a **provider-prefixed `{Provider}ModelSettings` field** — `anthropic_cache`, `openai_prompt_cache_key`, `groq_reasoning_effort`;
- a **message-stream marker** — `CachePoint` in `UserPromptPart.content` (`messages.py`);
- a **`ModelProfile` capability flag** — `openai_supports_strict_tool_definition`, `bedrock_supports_prompt_caching`.

Read the actual sibling implementations and any review threads on the PRs that added them (`gh pr view <n> --comments`). Contributors who skipped this got redirected: raw `google_tool_config` → use `strict` ([#5366](https://github.com/pydantic/pydantic-ai/issues/5366#issuecomment-4909047000)); string-prefix model detection → use a profile flag ([#4604](https://github.com/pydantic/pydantic-ai/pull/4604#discussion_r2967152134)). **Only open a design fork if no existing abstraction covers the capability.**

## Step 1 — Pick the API shape

1. **An existing cross-provider abstraction covers it → reuse it.** Add a provider mapping (a `_translate_*` resolver, a `JsonSchemaTransformer` subclass, a `CachePoint` translation). Do **not** add a provider-prefixed knob for something the shared abstraction already expresses.
2. **No shared abstraction, but ≥3 providers now have the concept → promote to a shared `ModelSettings` field** with a deliberately narrow common vocabulary and per-model resolvers, keeping per-provider fields underneath as precedence-winning escape hatches (the `service_tier` promotion, [#4926](https://github.com/pydantic/pydantic-ai/pull/4926#discussion_r3028738692)). Don't delete provider knobs; deprecate only genuinely-misnamed ones with a `TODO(v3)`.
3. **The provider's native values can't be expressed by the shared enum** (family-disjoint value sets, platform-only request shaping) → a `{provider}_*` knob is justified, and it **coexists with and outranks** the unified field (`groq_reasoning_effort`, [#5797](https://github.com/pydantic/pydantic-ai/pull/5797)).
4. **Choose the locus by "can the user naturally point at it?":** a boundary *inside* the message stream → a marker (`CachePoint`); a structural region (system prompt, tool defs, whole-request setting) → a setting ([#3363](https://github.com/pydantic/pydantic-ai/pull/3363#discussion_r2505366576)).
5. **Wire it into every request path** the provider has (chat *and* responses APIs); put a setting shared across a provider's APIs on the **base** settings class ([#3678](https://github.com/pydantic/pydantic-ai/pull/3678#discussion_r2602911025)).
6. **Type it.** Reuse the provider SDK's own types where they exist; type knobs as `Literal`, never `extra_body` or untyped `**kwargs` (`models/AGENTS.md`; *"kwargs are a big no no… I'd rather be repetitive but type safe"* — DouweM, [#3457](https://github.com/pydantic/pydantic-ai/pull/3457#discussion_r2547862149)).

## Step 2 — Default: default-on (silent) vs opt-in

Default the feature **on** only when enabling it **cannot change observable behavior and cannot cost the user** — a pure, backward-compatible improvement (caching only *lowers* cost; a validation mode that needs no schema rewrite and can't reject a previously-valid request). A backward-compatible, purely-better default is welcome and needs no opt-in.

Keep it **opt-in** when it:

- changes observable output or wire behavior;
- is a **preview** feature (provider may change semantics);
- can *raise* cost — and choose the default *value* so a shared field never silently upgrades anyone to a pricier tier (`service_tier` `'auto'` vs `'default'`, [#4926](https://github.com/pydantic/pydantic-ai/pull/4926));
- can hit provider limits at scale (auto-promoting strict silently broke agents with >20 tools; reverted in [#5580](https://github.com/pydantic/pydantic-ai/pull/5580));
- applies lossy schema rewrites (OpenAI/Anthropic strict transforms).

When the default isn't obvious, **decide it with a live probe, not an opinion** ([#5897](https://github.com/pydantic/pydantic-ai/pull/5897) flipped a default on 0/5 → 5/5 recovery). Beware "automatic" language — verify whether it means a Pydantic AI default or just provider-side management of an opt-in feature.

## Step 3 — Capability gating

Detect support with a provider-prefixed **`ModelProfile` flag** set in `Provider.model_profile()` — never inline `isinstance`/model-name checks (`profiles/AGENTS.md`). Gate at the layer(s) that actually vary:

- **model/family** → profile flag (`google_supports_strict_tool_definition`, `bedrock_supports_prompt_caching`);
- **per-schema** → the `JsonSchemaTransformer.is_strict_compatible` signal (default conservative unless the mode needs no rewrites);
- **SDK version** → probe the SDK shape and degrade with a `UserWarning` ([#5580](https://github.com/pydantic/pydantic-ai/pull/5580), botocore strict param);
- **unknowable client-side** → defer to the runtime API error.

**Unsupported → silently ignore** the setting (best-effort so as many requests as possible succeed), documented in the docstring — never hard-error. Conflicting user settings → `UserWarning`, not `UserError`. Profiles are layered: developer-keyed base + thin provider overlay resolved per family, with the provider overlay layering **last** ([#5934](https://github.com/pydantic/pydantic-ai/issues/5934), [#6231](https://github.com/pydantic/pydantic-ai/pull/6231)).

## Step 4 — Tests

Live-recorded **wire-contract cassette** asserting the exact outbound body (mode/field is sent for supported models, absent/ignored for unsupported), plus a test exercising the new setting. Prefer case-based parametrized VCR over mocks; a unit test is still right for asserting an internal request shape a cassette matcher wouldn't catch. This is also where a review bot's "this will fail" claim gets refuted with a recorded cassette.

## Step 5 — Docs & skills

The new public symbol's docstring lists **which providers support it** and how each interprets the value. Add/refresh the `docs/**.md` section and any sibling docstrings that now under-claim ("only OpenAI" → the full provider list). Describe the mechanism only as far as the provider documents it — don't assert a mechanism a provider's docs leave unstated. Update the relevant agent skill.

## Recurring maintainer principles (quoted)

1. **Reuse the cross-provider abstraction over a provider knob** — *"this maps to what the Anthropic and OpenAI APIs call `strict`… already represented on `ToolDefinition`… a more complete, consistent, 'doesn't require the user to do something special' implementation would be to automatically use this mode."* ([#5366](https://github.com/pydantic/pydantic-ai/issues/5366#issuecomment-4909047000))
2. **Promote to a shared setting once several providers have the concept; keep per-provider overrides underneath.** ([#4926](https://github.com/pydantic/pydantic-ai/pull/4926#discussion_r3028738692))
3. **Best-effort — silently ignore unsupported settings, don't error.** *"we typically do a 'best effort' so that as many requests as possible succeed."* ([#3438](https://github.com/pydantic/pydantic-ai/pull/3438#issuecomment-3553448605))
4. **Hide provider complexity** — the feature should be useful to people who don't want to become experts in that provider's limitations.
5. **Capability facts belong on `ModelProfile` flags, layered base + overlay.** ([#5934](https://github.com/pydantic/pydantic-ai/issues/5934#issuecomment-4858653237))
6. **Type safety over repetition; no untyped kwargs.** ([#3457](https://github.com/pydantic/pydantic-ai/pull/3457#discussion_r2547862149))
7. **Put shared fields on the base settings class; cover every API surface.** ([#3678](https://github.com/pydantic/pydantic-ai/pull/3678#discussion_r2602911025))
8. **Verify against the real API; defer validation to runtime.** *"If they're allowed by the SDK types and we can try it out and it doesn't fail, I'm fine with it."* ([#3678](https://github.com/pydantic/pydantic-ai/pull/3678#discussion_r2604746387))
9. **Name the eventual unification even when deferring it** — a `{provider}_*` knob today can note the future `Caching`/`Thinking`-style capability it should fold into. ([#4604](https://github.com/pydantic/pydantic-ai/pull/4604#discussion_r3277672350))

## Precedent map

| Capability | Reused abstraction | Default | Gating | PR |
|---|---|---|---|---|
| service tier (cross-provider) | promoted to `ModelSettings.service_tier` | opt-in, never silent upgrade | map-and-drop | [#4926](https://github.com/pydantic/pydantic-ai/pull/4926) |
| strict — OpenAI (origin) | `ToolDefinition.strict` | auto-promote per compatible schema | per-schema `is_strict_compatible` | [#1304](https://github.com/pydantic/pydantic-ai/pull/1304) |
| strict — Anthropic | reused `strict` | conservative opt-in | transformer + profile flag | [#3457](https://github.com/pydantic/pydantic-ai/pull/3457) |
| strict — Bedrock (+ fix) | reused `strict` | opt-in (auto-promote reverted) | transformer + profile + SDK probe | [#4237](https://github.com/pydantic/pydantic-ai/pull/4237), [#5580](https://github.com/pydantic/pydantic-ai/pull/5580) |
| strict — Gemini `VALIDATED` | reused `strict`; rejected raw `google_tool_config` | **default-on** (mode needs no schema rewrite) | profile flag; `is_strict_compatible = True` | [#6353](https://github.com/pydantic/pydantic-ai/pull/6353) |
| thinking (cross-provider) | `ModelSettings.thinking` + per-provider maps | opt-in, graceful degradation | `supports_thinking` flags | [#4640](https://github.com/pydantic/pydantic-ai/pull/4640) |
| reasoning effort — Groq | provider knob coexists with `thinking`, outranks it | opt-in | per-family profile flag | [#5797](https://github.com/pydantic/pydantic-ai/pull/5797), [#6231](https://github.com/pydantic/pydantic-ai/pull/6231) |
| prompt caching — Anthropic → Bedrock/OpenRouter | `CachePoint` marker + settings | opt-in | `{provider}_supports_prompt_caching` | [#3363](https://github.com/pydantic/pydantic-ai/pull/3363), [#3438](https://github.com/pydantic/pydantic-ai/pull/3438), [#4604](https://github.com/pydantic/pydantic-ai/pull/4604) |
