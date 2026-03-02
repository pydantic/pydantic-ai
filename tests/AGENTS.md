<!-- braindump: rules extracted from PR review patterns -->

# tests/ Guidelines

## Testing

<!-- rule:177 -->
- Test through public APIs, not private methods/attributes (prefixed with `_`) — prevents brittle tests coupled to implementation details — Testing implementation details like `_registered_model_instances` or `_resolve_model()` creates fragile tests that break during refactoring; public API tests validate actual user-facing behavior and survive internal changes
<!-- rule:86 -->
- Use `assert result == snapshot()` for complex outputs (objects, message sequences, API responses, nested dicts, span attributes, telemetry) — easier to maintain than field-by-field assertions; use `IsStr` for variable values — Snapshot testing reduces brittleness and maintenance overhead when verifying structured data, while matchers handle dynamic values cleanly
<!-- rule:318 -->
- Use `pytest-vcr` to record/replay API calls in `tests/models/` — captures real provider behavior for regression testing without live API dependency — Recording actual API interactions with VCR ensures tests reflect real provider behavior and can run offline, while avoiding brittle manual mocks that drift from reality.
<!-- rule:319 -->
- Use `pytest --record-mode=rewrite` to regenerate cassettes when modifying VCR tests or API interactions — Ensures test cassettes in `tests/models/cassettes/` stay in sync with test code and actual API behavior, preventing false passes from stale recordings
<!-- rule:334 -->
- Verify meaningful behavior with explicit assertions — validate actual data flow, state changes, and outputs, not just that code runs without errors — Prevents false confidence from trivial tests (import checks, isinstance tautologies, exception-free execution) that don't actually validate correctness or catch regressions
<!-- rule:194 -->
- Snapshot test `result.all_messages()` in agent/model/stream tests — catches regressions in tool calls, reasoning steps, and message flow, not just final output — Asserting only on final output misses bugs in tool calls, intermediate reasoning, and message sequences that can break agent behavior
<!-- rule:363 -->
- Prefer integration tests over mocked unit tests for external systems — test real API interactions in `test_{provider}.py` files — Integration tests catch real-world failures, API changes, and serialization issues that mocks miss, ensuring components work with actual external dependencies
<!-- rule:205 -->
- Place tests in `test_{component}.py` files that mirror the component being tested, not the utilities used — test agent behavior in `test_agent.py` even if it uses function models as fixtures — Prevents test fragmentation and maintains clear test organization as the suite scales, making tests discoverable by what they test rather than implementation details
<!-- rule:173 -->
- Keep 1:1 test-to-module mapping: all tests for a module go in `test_{module}.py`, not split by config/type/feature — Prevents fragmentation and makes tests easier to find; use fixtures/markers to distinguish test variations instead of separate files.
<!-- rule:11 -->
- Test all supported providers and variants (endpoints, auth methods, tiers) — catches provider-specific issues and ensures cross-provider compatibility — Prevents regressions in specific provider implementations and verifies features work consistently across different backends like `OpenAIProvider`, `OllamaProvider`, etc.
<!-- rule:89 -->
- Test both positive (feature works) and negative (graceful failure) cases for optional capabilities — Ensures features work correctly where supported AND fail gracefully where unsupported, preventing silent failures and improving error handling across providers
<!-- rule:630 -->
- Test MCP against real `tests.mcp_server` instance, not mocks — extend test server with helper tools to expose runtime context (instructions, client info, session state) — Verifies actual data flow and integration behavior that mocks would hide, ensuring MCP functionality works end-to-end

## Documentation

<!-- rule:386 -->
- Test docstrings must describe what the test actually validates, not aspirational behavior — Misleading test documentation causes confusion during debugging and makes it unclear what's actually being tested when failures occur
<!-- rule:463 -->
- Keep test docstrings/comments current with behavior changes; remove resolved historical bug notes — Outdated test documentation creates confusion and maintenance burden; historical bug notes add no actionable value once resolved

<!-- /braindump -->
