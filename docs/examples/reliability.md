# Production Reliability with Arsenal

Pydantic AI agents can fail due to provider outages, rate limits, or transient errors.
[Arsenal](https://github.com/darshjme/arsenal) provides drop-in reliability primitives.

## Circuit Breaker

```python
from pydantic_ai import Agent
from kavacha import CircuitBreaker
from punarjanma import retry

cb = CircuitBreaker(name="llm", threshold=3, timeout=30)

@retry(max_attempts=3, backoff=2.0)
def run_agent(prompt: str) -> str:
    with cb:
        agent = Agent("anthropic:claude-sonnet-4-6")
        return agent.run_sync(prompt).data
```

Install: `pip install kavacha punarjanma`
Arsenal: [github.com/darshjme/arsenal](https://github.com/darshjme/arsenal) — 100 reliability libs, 4,375 tests.
