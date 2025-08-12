# Upgrade Guide

Pydantic AI is still pre-version 1, so breaking changes will occur, however:

- We try to minimize them as much as possible.
- We use minor version bumps to signify breaking changes.
- Wherever possible we deprecate old features so code continues to work with deprecation warnings when changing the public API.
- We intend to release V1 in summer 2025, and then follow strict semantic versioning, e.g. no intentional breaking changes except in minor or patch versions.

## Breaking Changes

!!! note
    Here's a filtered list of the breaking changes for each version to help you upgrade Pydantic AI.

### v0.7.0 (2025-08-12)

See [#2458](https://github.com/pydantic/pydantic-ai/pull/2458) - `pydantic_ai.models.StreamedResponse` now yields a `FinalResultEvent` along with the existing `PartStartEvent` and `PartDeltaEvent`. If you're using `pydantic_ai.direct.model_request_stream` or `pydantic_ai.direct.model_request_stream_sync`, you may need to update your code to account for this.

See [#2458](https://github.com/pydantic/pydantic-ai/pull/2458) - `pydantic_ai.models.Model.request_stream` now receives a `run_context` argument. If you've implemented a custom `Model` subclass, you will need to account for this.

See [#2458](https://github.com/pydantic/pydantic-ai/pull/2458) - `pydantic_ai.models.StreamedResponse` now requires a `model_request_parameters` field and constructor argument. If you've implemented a custom `Model` subclass and implemented `request_stream`, you will need to account for this.

### v0.6.0 (2025-08-06)

This release was meant to clean some old deprecated code, so we can get a step closer to V1.

See [#2440](https://github.com/pydantic/pydantic-ai/pull/2440) - The `next` method was removed from the `Graph` class. Use `async with graph.iter(...) as run:  run.next()` instead.

See [#2441](https://github.com/pydantic/pydantic-ai/pull/2441) - The `result_type`, `result_tool_name` and `result_tool_description` arguments were removed from the `Agent` class. Use `output_type` instead.

See [#2441](https://github.com/pydantic/pydantic-ai/pull/2441) - The `result_retries` argument was also removed from the `Agent` class. Use `output_retries` instead.

See [#2443](https://github.com/pydantic/pydantic-ai/pull/2443) - The `data` property was removed from the `FinalResult` class. Use `output` instead.

See [#2445](https://github.com/pydantic/pydantic-ai/pull/2445) - The `get_data` and `validate_structured_result` methods were removed from the
`StreamedRunResult` class. Use `get_output` and `validate_structured_output` instead.

See [#2446](https://github.com/pydantic/pydantic-ai/pull/2446) - The `format_as_xml` function was moved to the `pydantic_ai.format_as_xml` module.
Import it via `from pydantic_ai import format_as_xml` instead.

See [#2451](https://github.com/pydantic/pydantic-ai/pull/2451) - Removed deprecated `Agent.result_validator` method, `Agent.last_run_messages` property, `AgentRunResult.data` property, and `result_tool_return_content` parameters from result classes.

### v0.5.0 (2025-08-04)

See [#2388](https://github.com/pydantic/pydantic-ai/pull/2388) - The `source` field of an `EvaluationResult` is now of type `EvaluatorSpec` rather than the actual source `Evaluator` instance, to help with serialization/deserialization.

See [#2163](https://github.com/pydantic/pydantic-ai/pull/2163) - The `EvaluationReport.print` and `EvaluationReport.console_table` methods now require most arguments be passed by keyword.

### v0.4.0 (2025-07-08)

See [#1799](https://github.com/pydantic/pydantic-ai/pull/1799) - Pydantic Evals `EvaluationReport` and `ReportCase` are now generic dataclasses instead of Pydantic models. If you were serializing them using `model_dump()`, you will now need to use the `EvaluationReportAdapter` and `ReportCaseAdapter` type adapters instead.

See [#1507](https://github.com/pydantic/pydantic-ai/pull/1507) - The `ToolDefinition` `description` argument is now optional and the order of positional arguments has changed from `name, description, parameters_json_schema, ...` to `name, parameters_json_schema, description, ...` to account for this.

### v0.3.0 (2025-06-18)

See [#1142](https://github.com/pydantic/pydantic-ai/pull/1142) — Adds support for thinking parts.

We now convert the thinking blocks (`"<think>..."</think>"`) in provider specific text parts to
Pydantic AI `ThinkingPart`s. Also, as part of this release, we made the choice to not send back the
`ThinkingPart`s to the provider - the idea is to save costs on behalf of the user. In the future, we
intend to add a setting to customize this behavior.

### v0.2.0 (2025-05-12)

See [#1647](https://github.com/pydantic/pydantic-ai/pull/1647) — usage makes sense as part of `ModelResponse`, and could be really useful in "messages" (really a sequence of requests and response). In this PR:

- Adds `usage` to `ModelResponse` (field has a default factory of `Usage()` so it'll work to load data that doesn't have usage)
- changes the return type of `Model.request` to just `ModelResponse` instead of `tuple[ModelResponse, Usage]`

### v0.1.0 (2025-04-15)

See [#1248](https://github.com/pydantic/pydantic-ai/pull/1248) — the attribute/parameter name `result` was renamed to `output` in many places. Hopefully all changes keep a deprecated attribute or parameter with the old name, so you should get many deprecation warnings.

See [#1484](https://github.com/pydantic/pydantic-ai/pull/1484) — `format_as_xml` was moved and made available to import from the package root, e.g. `from pydantic_ai import format_as_xml`.

---

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
