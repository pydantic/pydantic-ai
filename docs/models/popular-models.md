# Popular Models

This page provides detailed information about the most popular AI models supported by Pydantic AI, including pricing, specifications, and available inference providers. Each model card lists the **shorthand** (ID) that can be passed to the [`Agent`](../api/agent.md) class to use that model and provider combination:

```python
from pydantic_ai import Agent

agent = Agent('anthropic:claude-opus-4-5')
```

All popular models can be used via the [Gateway](https://gateway.pydantic.dev).

For an overview of all supported providers, see [Models and Providers](overview.md).

!!! note "About Modalities"
    Unless otherwise specified, "Modalities" refer to the types of data a model can accept as input. Output modality is implicitly Text for all models unless noted otherwise.

## Anthropic Claude

<div class="model-card" markdown>

### Claude Opus 4.5

ID: `anthropic:claude-opus-4-5`

Anthropic's most intelligent model for complex reasoning and research.

<div class="model-details" markdown>
<div markdown>

**Specifications:**

- Context: 200K tokens
- Modalities: Text
- Release: November 2025

**Performance:**

<ul class="perf-list">
<li><span>Cost</span> <span class="rating-badge rating-5">5</span></li>
<li><span>Speed</span> <span class="rating-badge rating-2">2</span></li>
<li><span>Intelligence</span> <span class="rating-badge rating-5">5</span></li>
</ul>

</div>
<div markdown>

**Alternative Providers:**

- [Gateway](https://gateway.pydantic.dev):<br>
    `'gateway/anthropic:claude-opus-4-5'`
- [AWS Bedrock](bedrock.md):<br>
    `'bedrock:us.anthropic.claude-opus-4-5-20251101-v1:0'`
- [OpenRouter](openrouter.md):<br>
    `'openrouter:anthropic/claude-opus-4.5'`

**Pricing:** $5.00 / $25.00 per 1M tokens (in/out)

</div>
</div>
</div>

<div class="model-card" markdown>

### Claude Sonnet 4.5

ID: `anthropic:claude-sonnet-4-5`

Best balance of speed, cost, and capability. Ideal for agents and coding.

<div class="model-details" markdown>
<div markdown>

**Specifications:**

- Context: 1M tokens
- Modalities: Text
- Release: September 2025

**Performance:**

<ul class="perf-list">
<li><span>Cost</span> <span class="rating-badge rating-3">3</span></li>
<li><span>Speed</span> <span class="rating-badge rating-4">4</span></li>
<li><span>Intelligence</span> <span class="rating-badge rating-4">4</span></li>
</ul>

</div>
<div markdown>

**Alternative Providers:**

- [Gateway](https://gateway.pydantic.dev):<br>
    `'gateway/anthropic:claude-sonnet-4-5'`
- [AWS Bedrock](bedrock.md):<br>
    `'bedrock:us.anthropic.claude-sonnet-4-5-20250929-v1:0'`
- [OpenRouter](openrouter.md):<br>
    `'openrouter:anthropic/claude-sonnet-4.5'`

**Pricing:** $3.00-6.00 / $15.00-22.50 per 1M tokens (in/out)

</div>
</div>
</div>

<div class="model-card" markdown>

### Claude Haiku 4.5

ID: `anthropic:claude-haiku-4-5`

Fastest and most cost-effective. Ideal for high-volume tasks.

<div class="model-details" markdown>
<div markdown>

**Specifications:**

- Context: 200K tokens
- Modalities: Text
- Release: October 2025

**Performance:**

<ul class="perf-list">
<li><span>Cost</span> <span class="rating-badge rating-1">1</span></li>
<li><span>Speed</span> <span class="rating-badge rating-5">5</span></li>
<li><span>Intelligence</span> <span class="rating-badge rating-3">3</span></li>
</ul>

</div>
<div markdown>

**Alternative Providers:**

- [Gateway](https://gateway.pydantic.dev):<br>
    `'gateway/anthropic:claude-haiku-4-5'`
- [AWS Bedrock](bedrock.md):<br>
    `'bedrock:us.anthropic.claude-haiku-4-5-20251001-v1:0'`
- [OpenRouter](openrouter.md):<br>
    `'openrouter:anthropic/claude-haiku-4.5'`

**Pricing:** $1.00 / $5.00 per 1M tokens (in/out)

</div>
</div>
</div>

---

## Google Gemini

<div class="model-card" markdown>

### Gemini 3 Pro

ID: `google-gla:gemini-3-pro-preview`

Google's most capable model for complex multimodal tasks.

<div class="model-details" markdown>
<div markdown>

**Specifications:**

- Context: 1M tokens
- Modalities: Text, Vision
- Release: Preview 2025

**Performance:**

<ul class="perf-list">
<li><span>Cost</span> <span class="rating-badge rating-4">4</span></li>
<li><span>Speed</span> <span class="rating-badge rating-3">3</span></li>
<li><span>Intelligence</span> <span class="rating-badge rating-5">5</span></li>
</ul>

</div>
<div markdown>

**Alternative Providers:**

- [Gateway](https://gateway.pydantic.dev):<br>
    `'gateway/google-gla:gemini-3-pro-preview'`
- [Vertex AI](google.md):<br>
    `'google-vertex:gemini-3-pro-preview'`
- [OpenRouter](openrouter.md):<br>
    `'openrouter:google/gemini-3-pro-preview'`

**Pricing:** $2.00-4.00 / $12.00-18.00 per 1M tokens (in/out)

</div>
</div>
</div>

<div class="model-card" markdown>

### Gemini 3 Flash

ID: `google-gla:gemini-3-flash-preview`

Fast and efficient with excellent performance-to-cost ratio.

<div class="model-details" markdown>
<div markdown>

**Specifications:**

- Context: 1M tokens
- Modalities: Text, Audio
- Release: Preview 2025

**Performance:**

<ul class="perf-list">
<li><span>Cost</span> <span class="rating-badge rating-2">2</span></li>
<li><span>Speed</span> <span class="rating-badge rating-5">5</span></li>
<li><span>Intelligence</span> <span class="rating-badge rating-4">4</span></li>
</ul>

</div>
<div markdown>

**Alternative Providers:**

- [Gateway](https://gateway.pydantic.dev):<br>
    `'gateway/google-gla:gemini-3-flash-preview'`
- [Vertex AI](google.md):<br>
    `'google-vertex:gemini-3-flash-preview'`
- [OpenRouter](openrouter.md):<br>
    `'openrouter:google/gemini-3-flash-preview'`

**Pricing:** $0.50 / $3.00 per 1M tokens (in/out)

</div>
</div>
</div>

<div class="model-card" markdown>

### Gemini 2.5 Flash Lite

ID: `google-gla:gemini-2.5-flash-lite`

Most cost-effective with ultra-low latency for high-volume apps.

<div class="model-details" markdown>
<div markdown>

**Specifications:**

- Context: 1M tokens
- Modalities: Text, Audio
- Release: 2025

**Performance:**

<ul class="perf-list">
<li><span>Cost</span> <span class="rating-badge rating-1">1</span></li>
<li><span>Speed</span> <span class="rating-badge rating-5">5</span></li>
<li><span>Intelligence</span> <span class="rating-badge rating-2">2</span></li>
</ul>

</div>
<div markdown>

**Alternative Providers:**

- [Gateway](https://gateway.pydantic.dev):<br>
    `'gateway/google-gla:gemini-2.5-flash-lite'`
- [Vertex AI](google.md):<br>
    `'google-vertex:gemini-2.5-flash-lite'`
- [OpenRouter](openrouter.md):<br>
    `'openrouter:google/gemini-2.5-flash-lite'`

**Pricing:** $0.10 / $0.40 per 1M tokens (in/out)

</div>
</div>
</div>

---

## OpenAI GPT

<div class="model-card" markdown>

### GPT-5.2 Pro

ID: `openai:gpt-5.2-pro`

Most intelligent reasoning model for complex tasks.

<div class="model-details" markdown>
<div markdown>

**Specifications:**

- Context: 400K tokens
- Modalities: Text
- Release: December 2025

**Performance:**

<ul class="perf-list">
<li><span>Cost</span> <span class="rating-badge rating-4">4</span></li>
<li><span>Speed</span> <span class="rating-badge rating-3">3</span></li>
<li><span>Intelligence</span> <span class="rating-badge rating-5">5</span></li>
</ul>

</div>
<div markdown>

**Alternative Providers:**

- [Gateway](https://gateway.pydantic.dev):<br>
    `'gateway/openai:gpt-5.2-pro'`
- [Azure](azure.md):<br>
    `'azure:gpt-5.2-pro'`
- [OpenRouter](openrouter.md):<br>
    `'openrouter:openai/gpt-5.2-pro'`

**Pricing:** $21.00 / $168.00 per 1M tokens (in/out)

</div>
</div>
</div>

<div class="model-card" markdown>

### GPT-5.2

ID: `openai:gpt-5.2`

Flagship general-purpose model with excellent balance.

<div class="model-details" markdown>
<div markdown>

**Specifications:**

- Context: 400K tokens
- Modalities: Text
- Release: December 2025

**Performance:**

<ul class="perf-list">
<li><span>Cost</span> <span class="rating-badge rating-3">3</span></li>
<li><span>Speed</span> <span class="rating-badge rating-4">4</span></li>
<li><span>Intelligence</span> <span class="rating-badge rating-4">4</span></li>
</ul>

</div>
<div markdown>

**Alternative Providers:**

- [Gateway](https://gateway.pydantic.dev):<br>
    `'gateway/openai:gpt-5.2'`
- [Azure](azure.md):<br>
    `'azure:gpt-5.2'`
- [OpenRouter](openrouter.md):<br>
    `'openrouter:openai/gpt-5.2'`

**Pricing:** $1.75 / $14.00 per 1M tokens (in/out)

</div>
</div>
</div>

---

## xAI Grok

<div class="model-card" markdown>

### Grok 4

ID: `grok:grok-4`

Flagship reasoning model with excellent math and reasoning.

<div class="model-details" markdown>
<div markdown>

**Specifications:**

- Context: 256K tokens
- Modalities: Text, Vision
- Release: July 2025

**Performance:**

<ul class="perf-list">
<li><span>Cost</span> <span class="rating-badge rating-2">2</span></li>
<li><span>Speed</span> <span class="rating-badge rating-4">4</span></li>
<li><span>Intelligence</span> <span class="rating-badge rating-4">4</span></li>
</ul>

</div>
<div markdown>

**Alternative Providers:**

- [Gateway](https://gateway.pydantic.dev):<br>
    `'gateway/grok:grok-4'`
- [OpenRouter](openrouter.md):<br>
    `'openrouter:x-ai/grok-4'`

**Pricing:** $3.00 / $15.00 per 1M tokens (in/out)

</div>
</div>
</div>

<div class="model-card" markdown>

### Grok 4.1 Fast

ID: `grok:grok-4-1-fast`

Optimized for agentic tool calling with massive 2M context.

<div class="model-details" markdown>
<div markdown>

**Specifications:**

- Context: 2M tokens
- Modalities: Text, Vision
- Release: 2025

**Performance:**

<ul class="perf-list">
<li><span>Cost</span> <span class="rating-badge rating-1">1</span></li>
<li><span>Speed</span> <span class="rating-badge rating-5">5</span></li>
<li><span>Intelligence</span> <span class="rating-badge rating-3">3</span></li>
</ul>

</div>
<div markdown>

**Alternative Providers:**

- [Gateway](https://gateway.pydantic.dev):<br>
    `'gateway/grok:grok-4-1-fast'`
- [OpenRouter](openrouter.md):<br>
    `'openrouter:x-ai/grok-4-1-fast'`

**Pricing:** $0.20 / $0.50 per 1M tokens (in/out)

</div>
</div>
</div>

---

<small>**Performance Ratings:** <span class="rating-badge rating-1">1</span> = lowest, <span class="rating-badge rating-5">5</span> = highest. Cost based on $/1M tokens, Speed on tokens/sec, Intelligence on benchmarks.<br>
**Sources:** [OpenRouter Rankings](https://openrouter.ai/rankings), [Artificial Analysis](https://artificialanalysis.ai/), [pydantic/genai-prices](https://github.com/pydantic/genai-prices)</small>
