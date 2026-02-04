<!-- braindump: rules extracted from PR review patterns -->

# tests/ Guidelines

## Testing

- Use `assert result == snapshot()` for complex structures — more maintainable than manual field assertions — Snapshot testing catches all fields/changes automatically, reduces test brittleness, and makes tests easier to update when structure evolves
- Delete tests when removing features or when behavior is already covered — prevents test bloat and maintains meaningful coverage — Redundant or feature-orphaned tests waste CI time and create maintenance burden without adding safety
- Add tests to existing test files covering the same module — avoids fragmentation and keeps related tests discoverable — Prevents test suite fragmentation and ensures developers can find all tests for a feature in one place, improving maintainability
- Reuse test patterns, helpers, and fixtures across providers — ensures consistent coverage and reduces maintenance burden — Consistent test structures make it easier to verify equivalent coverage across providers and prevent divergence in how features are validated
- Test through public APIs, not private methods or mocked internals — prevents brittle tests — Tests coupled to implementation details break during refactoring; testing public interfaces validates actual behavior and makes tests resilient to internal changes
- Use VCR to record real API requests in tests — don't mock responses — Ensures tests validate actual API response formats including error conditions, preventing mismatches between mocked and real provider behavior
- Snapshot both intermediate state (messages, transitions) and final results in agent/async tests — Catches bugs in execution flow and state management, not just output correctness
- Stack `@pytest.mark.parametrize` decorators (one per param) for Cartesian product testing — Prevents accidentally omitting test combinations and makes it easier to add new values to independent variables

<!-- /braindump -->
