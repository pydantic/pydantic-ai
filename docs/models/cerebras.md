# Cerebras

Cerebras provides ultra-fast inference using their Wafer-Scale Engine (WSE), delivering predictable performance for any workload.

## Installation

To use Cerebras, you need to either install `pydantic-ai`, or install `pydantic-ai-slim` with the `cerebras` optional group:

```bash
# pip
pip install "pydantic-ai-slim[cerebras]"

# uv
uv add "pydantic-ai-slim[cerebras]"
```

## Configuration

To use Cerebras, go to [cloud.cerebras.ai](https://cloud.cerebras.ai/?utm_source=3pi_pydantic-ai&utm_campaign=partner_doc) to get an API key.

### Environment Variable

Set your API key as an environment variable:

```bash
export CEREBRAS_API_KEY='your-api-key'
```

### Available Models

Cerebras supports the following models:

- `llama-3.3-70b` (recommended) - Latest Llama 3.3 model
- `llama-3.1-8b` - Llama 3.1 8B (faster, smaller)
- `qwen-3-235b-a22b-instruct-2507` - Qwen 3 235B
- `qwen-3-32b` - Qwen 3 32B
- `gpt-oss-120b` - GPT-OSS 120B
- `zai-glm-4.6` - GLM 4.6 model


See the [Cerebras documentation](https://inference-docs.cerebras.ai/introduction?utm_source=3pi_pydantic-ai&utm_campaign=partner_doc) for the latest models.

## Usage

```python
from pydantic_ai import Agent

agent = Agent('cerebras:llama-3.3-70b')
result = agent.run_sync('What is the capital of France?')
print(result.output)
#> The capital of France is Paris.
```

## Why Cerebras?

- **Ultra-fast inference** - Powered by the world's largest AI chip (WSE)
- **Predictable performance** - Consistent latency for any workload
- **OpenAI-compatible** - Drop-in replacement for OpenAI API
- **Cost-effective** - Competitive pricing with superior performance

## Resources

- [Cerebras Inference Documentation](https://inference-docs.cerebras.ai?utm_source=3pi_pydantic-ai&utm_campaign=partner_doc)
- [Get API Key](https://cloud.cerebras.ai/?utm_source=3pi_pydantic-ai&utm_campaign=partner_doc)
- [Model Pricing](https://cerebras.ai/pricing?utm_source=3pi_pydantic-ai&utm_campaign=partner_doc)