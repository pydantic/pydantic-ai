# Synap Memory Tools for Pydantic AI

[Synap](https://maximem.ai) is a managed memory layer for AI agents. The Pydantic AI
integration provides `SynapDeps` — a typed dependency class — and `register_synap_tools`,
which adds `search_memory` and `store_memory` tools to any Pydantic AI `Agent` so it can
recall and persist information about users across sessions.

## Installation

```bash
pip install maximem-synap-pydantic-ai
```

Get an API key at [synap.maximem.ai](https://synap.maximem.ai).

## Usage

```python
from pydantic_ai import Agent
from maximem_synap import MaximemSynapSDK
from synap_pydantic_ai import SynapDeps, register_synap_tools

sdk = MaximemSynapSDK(api_key="sk-...")

agent = Agent('openai:gpt-4o', deps_type=SynapDeps)
register_synap_tools(agent)

result = agent.run_sync(
    "What are my dietary restrictions?",
    deps=SynapDeps(sdk=sdk, user_id="user_123", customer_id="acme_corp"),
)

print(result.data)
```

`register_synap_tools` adds two tools to the agent:

- **`search_memory`** — retrieves facts, preferences, and past interactions relevant
  to a query from the user's Synap memory
- **`store_memory`** — persists a new piece of information to Synap for retrieval
  in future sessions

The agent decides when to call each tool. Memory is scoped to the `user_id` and
`customer_id` in `SynapDeps`, ensuring strict isolation between users and customers.

## More Resources

- [Synap Documentation](https://docs.maximem.ai)
- [Pydantic AI Integration Guide](https://docs.maximem.ai/integrations/pydantic-ai)
- [Dashboard](https://synap.maximem.ai)
- [PyPI: maximem-synap-pydantic-ai](https://pypi.org/project/maximem-synap-pydantic-ai/)
