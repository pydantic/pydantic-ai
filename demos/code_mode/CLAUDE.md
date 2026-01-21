# Code Mode Eval Goals

## Success Criteria (Future)

1. LLM calls tools directly when simple (single tool, no chaining)
2. LLM uses code mode for chaining multiple tools or applying logic
3. LLM doesn't abuse code_mode to call tools one-by-one
4. LLM can call tools to explore their return types (applies to `Any` types when the JSON schema is not specific)

## Current Metrics

- Request count: code mode should reduce round-trips for complex queries
- Token usage: should reduce total tokens for multi-tool scenarios
- Correctness: both modes produce same results

## Test Scenarios

### Ideal for Code Mode
- Multi-tool chaining: 'Get user, then get their orders, then aggregate totals'
- Data transformation: 'Fetch records and compute statistics'
- Conditional logic: 'If user is premium, apply discount'

### Should Use Direct Tools
- Single tool call: 'What's the weather in London?'
- Simple lookup: 'Get user profile for ID 123'

## Measuring Success

When running comparison tests:
1. Same prompt to agent with direct tools vs code mode
2. Compare: request count, total tokens, result correctness
3. Code mode wins if: fewer requests AND same/better correctness
