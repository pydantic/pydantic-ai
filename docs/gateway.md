# Pydantic AI Gateway

**Pydantic AI Gateway** (PAIG) is a unified interface for accessing multiple AI providers with a single key. Features include built-in OpenTelemetry observability, real-time cost monitoring, failover management , and native integration with the Pydantic stack.

!!! note "Currently in Beta"
    The Pydantic AI Gateway is currently in Beta. No charges will be applied during this period.

Sign up at [gateway.pydantic.dev](https://gateway.pydantic.dev/).

For questions and feedback, contact us on [Slack](https://logfire.pydantic.dev/docs/join-slack/).

## Key features
- **API key management**: access multiple LLM providers with a single API key.
- **Cost Limits**: set spending limits at project, user, and API key levels with daily, weekly, and monthly caps.
- **BYOK and managed providers:** Bring your own API keys (BYOK) from LLM providers, or pay for API usage directly through the platform (_coming soon_).
- **Multi-provider support:** Access models from OpenAI, Anthropic, Google Vertex, Groq, and AWS Bedrock. _More providers coming soon_.
- **Backend observability:** Log every request through [Pydantic Logfire](https://pydantic.dev/logfire) or any OpenTelemetry backend (_coming soon_).
- **Zero translation**: Unlike traditional AI gateways that translate everything to one common schema, PAIG allows requests to flow through directly in each provider's native format. This gives you immediate access to the new model features as soon as they are released.
- **Open source with self-hosting**: PAIG's core is [open source](https://github.com/pydantic/pydantic-ai-gateway/) (under [AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0.en.html)), allowing self-hosting with file-based configuration, instead of using the managed service.
- **Enterprise ready**: Includes SSO (with OIDC support), granular permissions, and flexible deployment options. Deploy to your Cloudflare account, or run on-premises with our [consulting support](https://pydantic.dev/contact).


```python {title="hello_word.py"}
from pydantic_ai import Agent

agent = Agent(
    'gateway/openai:gpt-4.1',
    instructions='Be concise, reply with one sentence.'
)

result = agent.run_sync('Hello World')
print(result.output)
```
