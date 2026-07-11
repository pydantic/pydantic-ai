# feat: add Amazon Bedrock Knowledge Base tool

- Created `create_bedrock_kb_tool()` factory returning typed async function
- `RetrievalResult` Pydantic model for type-safe responses
- Supports managed and vector knowledge base types
- Agentic retrieval with fallback to standard Retrieve API
- Uses pydantic-ai's `tool_plain()` registration pattern
- Unit tests included
- Added BEDROCK_MANAGED_KB.md design doc

<!-- Thank you for contributing to Pydantic AI! -->

<!-- Please read the contributing guide: https://ai.pydantic.dev/contributing/ -->

<!-- For non-trivial changes, link an issue that a maintainer has agreed on -->
<!-- and assigned to you. Unassigned PRs may be auto-closed. -->
<!-- Trivial fixes (typos, broken links, small doc improvements) don't need an issue. -->

<!-- Adding a capability? Most capabilities belong in Pydantic AI Harness, not here. -->
<!-- See: https://ai.pydantic.dev/harness/overview/#what-goes-where -->

- Closes N/A (new feature)

### Checklist

- [x] Any **AI generated code** has been reviewed line-by-line by the human PR author, who stands by it.
- [x] No **breaking changes** in accordance with the [version policy](https://github.com/pydantic/pydantic-ai/blob/main/docs/version-policy.md).
- [x] **PR title** is fit for the [release changelog](https://github.com/pydantic/pydantic-ai/releases).
