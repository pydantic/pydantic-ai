# Upgrade Guide

PydanticAI is still pre-version 1, so breaking changes will occur, however:

- We try to minimize them as much as possible.
- We use minor version bumps to signify breaking changes.
- Wherever possible we deprecate old features so code continues to work with deprecation warnings when changing the public API.
- We intend to release V1 in summer 2025, and then follow strict semantic versioning, e.g. no intentional breaking changes except in minor or patch versions.

## Breaking Changes

!!! note
    Here's a filtered list of the breaking changes for each version to help you upgrade PydanticAI.

### v0.1.0 (2025-04-15)

See [#1248](https://github.com/pydantic/pydantic-ai/pull/1248) — the attribute/parameter name `result` was renamed to `output` in many places. Hopefully all changes keep a deprecated attribute or parameter with the old name, so you should get many deprecation warnings.

See [#1484](https://github.com/pydantic/pydantic-ai/pull/1484) — `format_as_xml` was moved and made available to import from the package root, e.g. `from pydantic_ai import format_as_xml`.

### v0.2.0 (2025-05-12)

See [#1647](https://github.com/pydantic/pydantic-ai/pull/1647) — usage makes sense as part of `ModelResponse`, and could be really useful in "messages" (really a sequence of requests and response). In this PR:

* Adds `usage` to `ModelResponse` (field has a default factory of `Usage()` so it'll work to load data that doesn't have usage)
* changes the return type of `Model.request` to just `ModelResponse` instead of `tuple[ModelResponse, Usage]`

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
