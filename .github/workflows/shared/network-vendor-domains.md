---
# Shared network allowlist for provider SDK/API hosts and vendor docs.
#
# This list is intended for research/review workflows that may need to consult:
# - provider API endpoints
# - provider SDK/docs/console pages
# - provider model catalogs and related reference docs
#
# Sources:
# - Existing allowlist from pydantic-ai-pr-review.md
# - Domains referenced under docs/models/
#
# Keep this list host-only (no schemes/paths) and deduplicated.
network:
  allowed:
    - defaults
    - python
    - github
    - chrome

    # Core Pydantic docs
    - ai.pydantic.dev
    - pydantic.dev

    # Anthropic
    - anthropic.com
    - console.anthropic.com
    - docs.anthropic.com
    - platform.claude.com
    - api.minimax.io

    # OpenAI-compatible ecosystem
    - api.openai.com
    - platform.openai.com
    - developers.openai.com
    - api.deepseek.com
    - api-docs.deepseek.com
    - deepseek.com
    - openrouter.ai
    - api.perplexity.ai
    - docs.perplexity.ai
    - api.studio.nebius.com
    - studio.nebius.com
    - dashscope-intl.aliyuncs.com
    - dashscope.aliyuncs.com
    - www.alibabacloud.com
    - api.fireworks.ai
    - fireworks.ai
    - api.together.xyz
    - www.together.ai
    - api.sambanova.ai
    - cloud.sambanova.ai
    - docs.sambanova.ai
    - api.cerebras.ai
    - cerebras.ai
    - cloud.cerebras.ai
    - inference-docs.cerebras.ai

    # xAI
    - api.x.ai
    - x.ai
    - console.x.ai
    - docs.x.ai

    # Groq
    - api.groq.com
    - groq.com
    - console.groq.com

    # Mistral
    - api.mistral.ai
    - mistral.ai
    - console.mistral.ai

    # Cohere
    - api.cohere.com
    - cohere.com
    - dashboard.cohere.com

    # Google Gemini / Vertex
    - ai.google.dev

    # AWS Bedrock
    - amazonaws.com
    - aws.amazon.com
    - docs.aws.amazon.com
    - boto3.amazonaws.com

    # GitHub Models
    - models.github.ai

    # Hugging Face
    - huggingface.co
    - hf.co

    # Azure / Microsoft Foundry
    - ai.azure.com
    - learn.microsoft.com

    # Additional provider docs/endpoints used in models docs
    - vercel.com
    - ollama.com
    - platform.moonshot.ai
    - ovh.com
    - endpoints.ai.cloud.ovh.net

    # Logfire ingestion/tenant endpoints
    - logfire-api.pydantic.dev
    - logfire-eu.pydantic.dev
    - logfire-us.pydantic.dev
    - logfire-api.pydantic.info
    - logfire-eu.pydantic.info
    - logfire-us.pydantic.info

    # UI adapter protocol specs (Vercel AI SDK, AG-UI)
    - ai-sdk.dev
    - docs.ag-ui.com
---
