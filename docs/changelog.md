# Changelog

## Breaking Changes

PydanticAI is still pre-version 1, so breaking changes will occur, however:

- We try to minimize them as much as possible.
- We will use minor version bumps to signify breaking changes.
- Wherever possible we'll deprecate old features so code continues to work with deprecation warnings when changing the public API.
- We intend to release V1 in summer 2025, and then follow strict semantic versioning, e.g. no intentional breaking changes except in major versions.

### V0.1.0

See [#1248](https://github.com/pydantic/pydantic-ai/pull/1248) — the attribute/parameter name `result` was renamed to `output` in many places. Hopefully all changes keep a deprecated attribute or parameter with the old name, so you should get many deprecation warnings.

See [#1484](https://github.com/pydantic/pydantic-ai/pull/1484) — `format_as_xml` was moved and made available to import from the package root, e.g. `from pydantic_ai import format_as_xml`.

## Full Changelog

<div id="display-changelog">
  For the full changelog, see <a href="https://github.com/pydantic/pydantic-ai/releases">GitHub Releases</a>.
</div>

<script>
  async function getChangelog() {
    const fullChangelog = document.getElementById('display-changelog');
    const r = await fetch('/changelog.html');
    fullChangelog.innerHTML = await r.text();
  }
  getChangelog();
</script>
