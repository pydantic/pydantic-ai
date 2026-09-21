---
title: Redact personal data before a model request
description: Remove known sensitive fields in application code so raw values never reach the model provider.
---

# Redact personal data before a model request

Redact deterministic identifiers before calling the agent. Asking the same model to redact and then process a prompt still sends the original data to the provider.

```bash
pip/uv-add "pydantic-ai-slim[openai]"
export OPENAI_API_KEY=your-api-key
```

```python {dunder_name="not_main"}
import asyncio
import re

from pydantic_ai import Agent

EMAIL = re.compile(r'\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b')
CARD = re.compile(r'\b(?:\d[ -]*?){13,19}\b')

agent = Agent(
    'openai:gpt-5.6-sol',
    instructions='Classify the support request. Treat bracketed values as redacted data.',
)


def redact_pii(text: str) -> str:
    text = EMAIL.sub('[EMAIL]', text)
    return CARD.sub('[PAYMENT_CARD]', text)


async def main() -> None:
    request = 'alice@example.com says card 4111 1111 1111 1111 was charged twice.'
    safe_request = redact_pii(request)
    result = await agent.run(safe_request)
    print(safe_request)
    #> [EMAIL] says card [PAYMENT_CARD] was charged twice.
    print(result.output)
    #> This is a duplicate card charge request involving redacted customer data.


if __name__ == '__main__':
    asyncio.run(main())
```

Regexes only cover identifiers with predictable formats. Use a dedicated detector for names, addresses, and domain-specific identifiers, and test it against representative data. Keep raw input in an access-controlled system rather than logs or message history.

## Related

See [Messages and chat history](../message-history.md) before storing or accepting conversation data outside the agent.
