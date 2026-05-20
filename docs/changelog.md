# Upgrade Guide

In September 2025, Pydantic AI reached V1, which means we're committed to API stability: we will not introduce changes that break your code until V2. For more information, review our [Version Policy](version-policy.md).

## Breaking Changes

Here's a filtered list of the breaking changes for each version to help you upgrade Pydantic AI.

### v2.0.0 (unreleased)

- Drop `pydantic_ai.providers.grok.GrokProvider` (use `pydantic_ai.providers.xai.XaiProvider` with `pydantic_ai.models.xai.XaiModel('grok-N', ...)` instead). The `'grok:'` model-string prefix is also removed; use `'xai:'` instead. `pydantic_ai.providers.grok.GrokModelName` is removed; use `pydantic_ai.models.xai.XaiModelName`.

#### Tool-call execution: `end_strategy` default and `sequential=True` semantics

The default [`end_strategy`][pydantic_ai.agent.EndStrategy] changed from `'early'` to `'graceful'`. This only affects responses where a model calls function tools in the *same* response as an [output tool](output.md#tool-output) (the call that ends the run). When that output tool **succeeds**, the function tools requested alongside it now **run** by default instead of being skipped, so their side effects happen and their results reach the model if the run continues; and a function tool's [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] now suppresses the output result so the model can correct itself on the next round. The case where *every* output tool fails is unchanged: function tools run and the run continues either way. Most agents don't need any change. If you relied on the run ending the instant an output tool succeeds — skipping any function tools requested in the same response — set `end_strategy='early'` explicitly.

`sequential=True` on a tool is now a per-tool **barrier** rather than a batch-wide serial switch: a sequential tool runs alone, but other tools in the same response still run in parallel around it. To run *all* of a run's tools serially, wrap the run in [`agent.parallel_tool_call_execution_mode('sequential')`][pydantic_ai.agent.AbstractAgent.parallel_tool_call_execution_mode] or set `parallel_tool_calls=False` on the [model settings][pydantic_ai.settings.ModelSettings].

See [Parallel Output Tool Calls](output.md#parallel-output-tool-calls) for the full behavior of all three strategies.

#### [`ModelProfile`][pydantic_ai.profiles.ModelProfile] is now a `TypedDict`

See the [Model Profile guide](models/openai.md#model-profile) for an overview of what a model profile is and how to configure one.

[`ModelProfile`][pydantic_ai.profiles.ModelProfile] and all its subclasses ([`OpenAIModelProfile`][pydantic_ai.profiles.openai.OpenAIModelProfile], [`AnthropicModelProfile`][pydantic_ai.profiles.anthropic.AnthropicModelProfile], [`GoogleModelProfile`][pydantic_ai.profiles.google.GoogleModelProfile], `BedrockModelProfile`, etc.) are now `TypedDict(total=False)` instead of `@dataclass`. This unifies the mental model with [`ModelSettings`][pydantic_ai.settings.ModelSettings] (also a `TypedDict`) and enables direct dict-spread for cross-class merging.

`ModelProfile.update()` and `ModelProfile.from_profile()` are removed; use the module-level [`merge_profile`][pydantic_ai.profiles.merge_profile] (later argument wins per key).

Migration recipes:

| v1 (dataclass) | v2 (TypedDict) |
|---|---|
| `OpenAIModelProfile(field=value)` | Same syntax; returns a partial `dict` instead of a fully-defaulted instance. |
| `profile.field` (attribute read) | `profile.get('field', <default>)` — non-trivial defaults are exported from [`pydantic_ai.profiles`][pydantic_ai.profiles] (e.g. [`DEFAULT_THINKING_TAGS`][pydantic_ai.profiles.DEFAULT_THINKING_TAGS], [`DEFAULT_PROMPTED_OUTPUT_TEMPLATE`][pydantic_ai.profiles.DEFAULT_PROMPTED_OUTPUT_TEMPLATE]); the fully-merged base is [`DEFAULT_PROFILE`][pydantic_ai.profiles.DEFAULT_PROFILE]. |
| `profile.field = value` (attribute write) | `profile['field'] = value` |
| `dataclasses.replace(profile, field=value)` | `{**profile, 'field': value}` or `merge_profile(profile, ModelProfile(field=value))` |
| `profile.update(other)` | `merge_profile(profile, other)` |
| `OpenAIModelProfile.from_profile(p)` | Just `p` — no upcasting needed |
| `Model(name, profile=full_profile)` (full replace) | Now merges on top of the provider's default profile — usually what you want. For a hard replace use `Model(name, profile=lambda _default: full_profile)`. |
| `Model(name, profile=fn)` where `fn: Callable[[str], ModelProfile \| None]` | Removed — the user-passed callable is now `Callable[[ModelProfile], ModelProfile]`, receiving the resolved default and returning the final profile. The `(model_name: str) -> ModelProfile \| None` shape is still accepted internally by `Provider.model_profile`. |
| `isinstance(profile, OpenAIModelProfile)` | Not supported by `TypedDict` at runtime — raises `TypeError`. Use `isinstance(profile, dict)` or check key presence (`'openai_chat_supports_web_search' in profile`). Pyright still narrows correctly via the TypedDict subclass annotation. |

`Model.profile` is now the single source of truth for the **resolved** profile. It is composed by [`merge_profile`][pydantic_ai.profiles.merge_profile] in this order (later wins):

1. [`DEFAULT_PROFILE`][pydantic_ai.profiles.DEFAULT_PROFILE] — base defaults for every documented key.
2. `Provider.model_profile(model_name)` — provider/model-specific resolution.
3. The user's `profile=` argument — either a partial dict (merged on top) or a `Callable[[ModelProfile], ModelProfile]` (full control: receives the resolved default, returns the final profile).

#### Resolved profiles now carry cross-class fields

In v1, `ModelProfile.update()` silently filtered out fields not declared on the target class. In v2, dict-spread preserves every key.

This means e.g. a Bedrock-hosted Anthropic model's resolved profile now carries the upstream `anthropic_*` fields alongside the `bedrock_*` fields, where v1 dropped them. No in-tree model class reads cross-class fields, so behavior is unchanged in the standard providers; but custom model classes that do `profile.get('anthropic_supports_adaptive_thinking', False)` on a non-Anthropic route will now see the value the upstream Anthropic profile set, where v1 always returned the default.

See [PR #5481](https://github.com/pydantic/pydantic-ai/pull/5481) for the full ModelProfile redesign.

### v1.0.1 (2025-09-05)

The following breaking change was accidentally left out of v1.0.0:

- See [#2808](https://github.com/pydantic/pydantic-ai/pull/2808) - Remove `Python` evaluator from `pydantic_evals` for security reasons

### v1.0.0 (2025-09-04)

- See [#2725](https://github.com/pydantic/pydantic-ai/pull/2725) - Drop support for Python 3.9
- See [#2738](https://github.com/pydantic/pydantic-ai/pull/2738) - Make many dataclasses require keyword arguments
- See [#2715](https://github.com/pydantic/pydantic-ai/pull/2715) - Remove `cases` and `averages` attributes from `pydantic_evals` spans
- See [#2798](https://github.com/pydantic/pydantic-ai/pull/2798) - Change `ModelRequest.parts` and `ModelResponse.parts` types from `list` to `Sequence`
- See [#2726](https://github.com/pydantic/pydantic-ai/pull/2726) - Default `InstrumentationSettings` version to 2
- See [#2717](https://github.com/pydantic/pydantic-ai/pull/2717) - Remove errors when passing `AsyncRetrying` or `Retrying` object to `AsyncTenacityTransport` or `TenacityTransport` instead of `RetryConfig`

### v0.x.x

Before V1, minor versions were used to introduce breaking changes:

**v0.8.0 (2025-08-26)**

See [#2689](https://github.com/pydantic/pydantic-ai/pull/2689) - `AgentStreamEvent` was expanded to be a union of `ModelResponseStreamEvent` and `HandleResponseEvent`, simplifying the `event_stream_handler` function signature. Existing code accepting `AgentStreamEvent | HandleResponseEvent` will continue to work.

**v0.7.6 (2025-08-26)**

The following breaking change was inadvertently released in a patch version rather than a minor version:

See [#2670](https://github.com/pydantic/pydantic-ai/pull/2670) - `TenacityTransport` and `AsyncTenacityTransport` now require the use of `pydantic_ai.retries.RetryConfig` (which is just a `TypedDict` containing the kwargs to `tenacity.retry`) instead of `tenacity.Retrying` or `tenacity.AsyncRetrying`.

**v0.7.0 (2025-08-12)**

See [#2458](https://github.com/pydantic/pydantic-ai/pull/2458) - `pydantic_ai.models.StreamedResponse` now yields a `FinalResultEvent` along with the existing `PartStartEvent` and `PartDeltaEvent`. If you're using `pydantic_ai.direct.model_request_stream` or `pydantic_ai.direct.model_request_stream_sync`, you may need to update your code to account for this.

See [#2458](https://github.com/pydantic/pydantic-ai/pull/2458) - `pydantic_ai.models.Model.request_stream` now receives a `run_context` argument. If you've implemented a custom `Model` subclass, you will need to account for this.

See [#2458](https://github.com/pydantic/pydantic-ai/pull/2458) - `pydantic_ai.models.StreamedResponse` now requires a `model_request_parameters` field and constructor argument. If you've implemented a custom `Model` subclass and implemented `request_stream`, you will need to account for this.

**v0.6.0 (2025-08-06)**

This release was meant to clean some old deprecated code, so we can get a step closer to V1.

See [#2440](https://github.com/pydantic/pydantic-ai/pull/2440) - The `next` method was removed from the `Graph` class. Use `async with graph.iter(...) as run:  run.next()` instead.

See [#2441](https://github.com/pydantic/pydantic-ai/pull/2441) - The `result_type`, `result_tool_name` and `result_tool_description` arguments were removed from the `Agent` class. Use `output_type` instead.

See [#2441](https://github.com/pydantic/pydantic-ai/pull/2441) - The `result_retries` argument was also removed from the `Agent` class. Use `output_retries` instead.

See [#2443](https://github.com/pydantic/pydantic-ai/pull/2443) - The `data` property was removed from the `FinalResult` class. Use `output` instead.

See [#2445](https://github.com/pydantic/pydantic-ai/pull/2445) - The `get_data` and `validate_structured_result` methods were removed from the
`StreamedRunResult` class. Use `get_output` and `validate_response_output` instead.

See [#2446](https://github.com/pydantic/pydantic-ai/pull/2446) - The `format_as_xml` function was moved to the `pydantic_ai.format_as_xml` module.
Import it via `from pydantic_ai import format_as_xml` instead.

See [#2451](https://github.com/pydantic/pydantic-ai/pull/2451) - Removed deprecated `Agent.result_validator` method, `Agent.last_run_messages` property, `AgentRunResult.data` property, and `result_tool_return_content` parameters from result classes.

**v0.5.0 (2025-08-04)**

See [#2388](https://github.com/pydantic/pydantic-ai/pull/2388) - The `source` field of an `EvaluationResult` is now of type `EvaluatorSpec` rather than the actual source `Evaluator` instance, to help with serialization/deserialization.

See [#2163](https://github.com/pydantic/pydantic-ai/pull/2163) - The `EvaluationReport.print` and `EvaluationReport.console_table` methods now require most arguments be passed by keyword.

**v0.4.0 (2025-07-08)**

See [#1799](https://github.com/pydantic/pydantic-ai/pull/1799) - Pydantic Evals `EvaluationReport` and `ReportCase` are now generic dataclasses instead of Pydantic models. If you were serializing them using `model_dump()`, you will now need to use the `EvaluationReportAdapter` and `ReportCaseAdapter` type adapters instead.

See [#1507](https://github.com/pydantic/pydantic-ai/pull/1507) - The `ToolDefinition` `description` argument is now optional and the order of positional arguments has changed from `name, description, parameters_json_schema, ...` to `name, parameters_json_schema, description, ...` to account for this.

**v0.3.0 (2025-06-18)**

See [#1142](https://github.com/pydantic/pydantic-ai/pull/1142) — Adds support for thinking parts.

We now convert the thinking blocks (`"<think>..."</think>"`) in provider specific text parts to
Pydantic AI `ThinkingPart`s. Also, as part of this release, we made the choice to not send back the
`ThinkingPart`s to the provider - the idea is to save costs on behalf of the user. In the future, we
intend to add a setting to customize this behavior.

**v0.2.0 (2025-05-12)**

See [#1647](https://github.com/pydantic/pydantic-ai/pull/1647) — usage makes sense as part of `ModelResponse`, and could be really useful in "messages" (really a sequence of requests and response). In this PR:

- Adds `usage` to `ModelResponse` (field has a default factory of `Usage()` so it'll work to load data that doesn't have usage)
- changes the return type of `Model.request` to just `ModelResponse` instead of `tuple[ModelResponse, Usage]`

**v0.1.0 (2025-04-15)**

See [#1248](https://github.com/pydantic/pydantic-ai/pull/1248) — the attribute/parameter name `result` was renamed to `output` in many places. Hopefully all changes keep a deprecated attribute or parameter with the old name, so you should get many deprecation warnings.

See [#1484](https://github.com/pydantic/pydantic-ai/pull/1484) — `format_as_xml` was moved and made available to import from the package root, e.g. `from pydantic_ai import format_as_xml`.

## Full Changelog

<div id="display-changelog">
  For the full changelog, see <a href="https://github.com/pydantic/pydantic-ai/releases">GitHub Releases</a>.
</div>

<script>
  fetch('/changelog.html').then(r => {
    if (r.ok) {
      r.text().then(t => {
        document.getElementById('display-changelog').innerHTML = t;
      });
    }
  });
</script>
