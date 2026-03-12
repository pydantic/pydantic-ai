<!-- braindump: rules extracted from PR review patterns -->

# tests/ Guidelines

## Testing

<!-- rule:177 -->
- Test through public APIs, not private methods (prefixed with `_`) or helpers — Prevents brittle tests tied to implementation details, reduces maintenance burden when refactoring internals, and validates actual user-facing behavior rather than isolated units
<!-- rule:173 -->
- Maintain 1:1 correspondence between test files and source modules (`test_{module}.py`) — consolidate related tests instead of splitting by feature, config, or test type — Prevents test suite fragmentation and makes tests easier to locate by matching source structure; use fixtures/markers to distinguish test types within the file
<!-- rule:86 -->
- Use `snapshot()` for complex structured outputs (objects, message sequences, API responses, nested dicts, span attributes) — prevents brittle field-by-field assertions and improves test maintainability — Snapshot testing catches unexpected changes in complex structures more reliably than manual assertions, and `IsStr` matchers handle variable values gracefully
<!-- rule:318 -->
- Use `pytest-vcr` cassettes (not mocks) in `tests/models/` — records real HTTP interactions for deterministic replay, captures both success and error cases — Ensures integration tests validate real API behavior without live calls on every run, making tests faster and preventing flakiness from network issues or rate limits
<!-- rule:334 -->
- Assert meaningful behavior in tests, not just code execution or type checks — validates correctness and data flow — Prevents false confidence from tests that pass without verifying actual functionality works as intended
<!-- rule:194 -->
- In agent/model/stream tests, assert on final output AND snapshot `result.all_messages()` — validates complete execution trace, not just end result — Catches regressions in tool calls, intermediate steps, and message flow that final output assertions miss
<!-- rule:363 -->
- Test through real APIs, not mocks — mock only slow/external dependencies outside your control — Improves refactoring safety, documents real usage patterns, and catches integration issues — use lightweight local infrastructure (test servers, in-memory DBs) for systems you control (provider APIs, Temporal workflows, frameworks) in files like `test_{provider}.py`; reserve mocks for third-party HTTP APIs and unreliable external services
<!-- rule:11 -->
- Parametrize tests across all providers that support the feature (or at minimum OpenAI, Anthropic, Google) — catches provider-specific regressions and ensures cross-provider compatibility — Prevents breaking unchanged providers when modifying shared model logic, and surfaces integration issues across different provider APIs before they reach production
<!-- rule:385 -->
- Ensure test assertions match test names and docstrings — prevents false confidence in test coverage and catches actual regressions — Tests without proper assertions or that verify opposite behavior create false positives and fail to catch bugs they claim to prevent.
<!-- rule:89 -->
- Test both positive and negative cases for optional capabilities (model features, server features, streaming) — ensures features work when supported AND fail gracefully when absent — Prevents false confidence from tests that only check unsupported cases, catching both implementation bugs and missing error handling
<!-- rule:630 -->
- Test MCP against real `tests.mcp_server` instance, not mocks — extend test server with helper tools to expose runtime context (instructions, client info, session state) — Verifies actual data flow and integration behavior rather than just testing mock interfaces, catching real-world issues that mocks would miss

## General

<!-- rule:463 -->
- Remove stale test docstrings, comments, and historical provider bug notes when behavior changes — Outdated test documentation misleads developers about what's actually being tested and why

<!-- /braindump -->
