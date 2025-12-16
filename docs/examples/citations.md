# Citations

Examples demonstrating how to access citations from model responses across different providers.

Citations are references to sources that the model used to generate its response. They typically include URLs, titles, and text ranges indicating which parts of the response are supported by each citation.

## Overview

Pydantic AI supports citations from multiple providers:

- **OpenAI** (Chat Completions and Responses APIs): `URLCitation` with URL, title, and character indices
- **Anthropic**: `ToolResultCitation` from tool execution results
- **Google/Gemini**: `GroundingCitation` from grounding metadata
- **OpenRouter**: Uses OpenAI-compatible citation format
- **Perplexity**: Uses OpenAI-compatible citation format

For more information, see the [Citations Overview](../citations/overview.md) and [Accessing Citations](../citations/accessing_citations.md) guides.

## Examples

### OpenAI Chat Completions

Demonstrates accessing URL citations from OpenAI's Chat Completions API:

```snippet {path="/examples/pydantic_ai_examples/citations/openai_chat_completions.py"}
```

Run with:
```bash
uv run -m pydantic_ai_examples.citations.openai_chat_completions
```

### OpenAI Responses API

Demonstrates accessing URL citations from OpenAI's Responses API:

```snippet {path="/examples/pydantic_ai_examples/citations/openai_responses.py"}
```

Run with:
```bash
uv run -m pydantic_ai_examples.citations.openai_responses
```

### Anthropic Claude

Demonstrates accessing tool result citations from Anthropic's Claude models:

```snippet {path="/examples/pydantic_ai_examples/citations/anthropic.py"}
```

Run with:
```bash
uv run -m pydantic_ai_examples.citations.anthropic
```

### Google Gemini

Demonstrates accessing grounding citations from Google's Gemini models:

```snippet {path="/examples/pydantic_ai_examples/citations/google.py"}
```

Run with:
```bash
uv run -m pydantic_ai_examples.citations.google
```

### OpenRouter

Demonstrates accessing URL citations from OpenRouter (OpenAI-compatible format):

```snippet {path="/examples/pydantic_ai_examples/citations/openrouter.py"}
```

Run with:
```bash
uv run -m pydantic_ai_examples.citations.openrouter
```

### Perplexity AI

Demonstrates accessing URL citations from Perplexity AI (OpenAI-compatible format):

```snippet {path="/examples/pydantic_ai_examples/citations/perplexity.py"}
```

Run with:
```bash
uv run -m pydantic_ai_examples.citations.perplexity
```

## Notes

- Citations are only available when the model includes them in its response
- Not all responses will contain citations
- Availability depends on:
  - The model's capabilities
  - The query type (some queries are more likely to trigger citations)
  - Provider-specific settings (e.g., enabling web search for Google)

## See Also

- [Citations Overview](../citations/overview.md) - Introduction to citations
- [Accessing Citations](../citations/accessing_citations.md) - Comprehensive guide on accessing citations
- [API Reference](../api/messages.md#citations) - Complete API documentation
