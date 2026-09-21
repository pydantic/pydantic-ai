---
title: Run the same agent in a terminal and browser
description: Define an agent once, then expose it through Pydantic AI's CLI and web chat interfaces.
---

# Run the same agent in a terminal and browser

Keep the agent in a normal importable module so different interfaces can load the same tools, instructions, dependencies, and output behavior.

```python {title="support_agent.py" test="skip"}
from pydantic_ai import Agent

agent = Agent(
    'openai:gpt-5.6-sol',
    instructions='Help operators diagnose incidents. Ask before suggesting a destructive action.',
)
```

Run it as an interactive terminal agent:

```bash
clai --agent support_agent:agent
```

Serve the same object in the built-in browser chat:

```bash
clai web --agent support_agent:agent
```

For a product-specific frontend, expose the same agent through an AG-UI or Vercel AI adapter rather than creating a second agent definition. Interface code should translate transport events; agent behavior should remain in the shared agent and its capabilities.
