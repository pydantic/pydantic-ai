# Accessing Citations

This guide shows how to access citations from model responses in Pydantic AI.

## Basic Access

Citations are attached to [`TextPart`][pydantic_ai.messages.TextPart] objects in the model's response. Each `TextPart` has an optional `citations` field that contains a list of citation objects.

### From Run Results

After running an agent, you can access citations from the response messages:

```python {title="basic_citations.py"}
from pydantic_ai import Agent, TextPart, URLCitation

agent = Agent('openai:gpt-4o')
result = agent.run_sync('What is the capital of France?')

# Access citations from new messages
for message in result.new_messages():
    if message.role == 'assistant':
        for part in message.parts:
            if isinstance(part, TextPart) and part.citations:
                print(f"Text: {part.content}")
                for citation in part.citations:
                    if isinstance(citation, URLCitation):
                        print(f"  Citation: {citation.title or citation.url}")
                        print(f"  URL: {citation.url}")
                        print(f"  Range: {citation.start_index}-{citation.end_index}")
```

### From All Messages

You can also access citations from the full message history:

```python {title="all_messages_citations.py"}
from pydantic_ai import Agent, TextPart

agent = Agent('openai:gpt-4o')

# First turn
result1 = agent.run_sync('What is the capital of France?')

# Second turn (continues conversation)
result2 = agent.run_sync('What about Germany?')

# Access citations from all messages
for message in result2.all_messages():
    if message.role == 'assistant':
        for part in message.parts:
            if isinstance(part, TextPart) and part.citations:
                print(f"Found {len(part.citations)} citations in message")
```

## Citation Types

### URLCitation (OpenAI)

`URLCitation` objects contain URL-based citations with character indices:

```python {title="url_citation.py"}
from pydantic_ai import Agent, TextPart, URLCitation

agent = Agent('openai:gpt-4o')
result = agent.run_sync('Tell me about Python programming.')

for message in result.new_messages():
    if message.role == 'assistant':
        for part in message.parts:
            if isinstance(part, TextPart) and part.citations:
                for citation in part.citations:
                    if isinstance(citation, URLCitation):
                        # Extract the cited text
                        cited_text = part.content[citation.start_index:citation.end_index]
                        print(f"Cited text: {cited_text}")
                        print(f"Source: {citation.title or citation.url}")
                        print(f"URL: {citation.url}")
```

### ToolResultCitation (Anthropic)

`ToolResultCitation` objects contain citations from tool execution results:

```python {title="tool_result_citation.py"}
from pydantic_ai import Agent, TextPart, ToolResultCitation

agent = Agent('anthropic:claude-3-5-sonnet-20241022')
result = agent.run_sync('Search for information about Python.')

for message in result.new_messages():
    if message.role == 'assistant':
        for part in message.parts:
            if isinstance(part, TextPart) and part.citations:
                for citation in part.citations:
                    if isinstance(citation, ToolResultCitation):
                        print(f"Tool: {citation.tool_name}")
                        print(f"Tool Call ID: {citation.tool_call_id}")
                        if citation.citation_data:
                            print(f"Citation Data: {citation.citation_data}")
```

### GroundingCitation (Google)

`GroundingCitation` objects contain citations from Google's grounding metadata:

```python {title="grounding_citation.py"}
from pydantic_ai import Agent, TextPart, GroundingCitation

agent = Agent('google-gla:gemini-1.5-flash')
result = agent.run_sync('What is the capital of France?')

for message in result.new_messages():
    if message.role == 'assistant':
        for part in message.parts:
            if isinstance(part, TextPart) and part.citations:
                for citation in part.citations:
                    if isinstance(citation, GroundingCitation):
                        if citation.citation_metadata:
                            print(f"Citation Metadata: {citation.citation_metadata}")
                        if citation.grounding_metadata:
                            print(f"Grounding Metadata: {citation.grounding_metadata}")
```

## Working with Multiple Citations

A single `TextPart` can have multiple citations:

```python {title="multiple_citations.py"}
from pydantic_ai import Agent, TextPart, URLCitation

agent = Agent('openai:gpt-4o')
result = agent.run_sync('Compare Python and JavaScript.')

for message in result.new_messages():
    if message.role == 'assistant':
        for part in message.parts:
            if isinstance(part, TextPart) and part.citations:
                print(f"Text has {len(part.citations)} citations:")
                for i, citation in enumerate(part.citations, 1):
                    if isinstance(citation, URLCitation):
                        print(f"  {i}. {citation.title or citation.url}")
                        print(f"     URL: {citation.url}")
```

## Filtering Citations

You can filter citations by type:

```python {title="filter_citations.py"}
from pydantic_ai import Agent, TextPart, URLCitation, ToolResultCitation, GroundingCitation

agent = Agent('openai:gpt-4o')
result = agent.run_sync('Tell me about Python.')

for message in result.new_messages():
    if message.role == 'assistant':
        for part in message.parts:
            if isinstance(part, TextPart) and part.citations:
                # Filter by type
                url_citations = [c for c in part.citations if isinstance(c, URLCitation)]
                tool_citations = [c for c in part.citations if isinstance(c, ToolResultCitation)]
                grounding_citations = [c for c in part.citations if isinstance(c, GroundingCitation)]

                print(f"URL citations: {len(url_citations)}")
                print(f"Tool citations: {len(tool_citations)}")
                print(f"Grounding citations: {len(grounding_citations)}")
```

## Citations in Streaming Responses

Citations are also available in streaming responses. They are attached to `TextPart` objects as they arrive:

```python {title="streaming_citations.py"}
from pydantic_ai import Agent, TextPart

agent = Agent('openai:gpt-4o')

async def stream_with_citations():
    async for response in agent.run_stream('Tell me about Python.'):
        for part in response.parts:
            if isinstance(part, TextPart):
                if part.citations:
                    print(f"Found {len(part.citations)} citations")
                    for citation in part.citations:
                        print(f"  Citation: {citation}")

# Run the async function
import asyncio
asyncio.run(stream_with_citations())
```

## Citations in Message History

Citations persist in message history and survive serialization/deserialization:

```python {title="citations_in_history.py"}
from pydantic_ai import Agent, TextPart
from pydantic_ai.messages import ModelMessagesTypeAdapter

agent = Agent('openai:gpt-4o')
result = agent.run_sync('What is Python?')

# Serialize messages
messages_json = result.all_messages_json()

# Deserialize messages
adapter = ModelMessagesTypeAdapter()
messages = adapter.validate_json(messages_json)

# Citations are preserved
for message in messages:
    if message.role == 'assistant':
        for part in message.parts:
            if isinstance(part, TextPart) and part.citations:
                print(f"Citations preserved: {len(part.citations)}")
```

## Citations in OpenTelemetry

Citations are included in OpenTelemetry events for observability:

```python {title="otel_citations.py"}
from pydantic_ai import Agent, TextPart
from pydantic_ai.models.instrumented import InstrumentationSettings

agent = Agent('openai:gpt-4o')
result = agent.run_sync('Tell me about Python.')

# Get OTEL events
for message in result.new_messages():
    if message.role == 'assistant':
        settings = InstrumentationSettings(include_content=True)
        events = message.otel_events(settings)

        for event in events:
            content = event.body.get('content', [])
            if isinstance(content, list):
                for item in content:
                    if 'citations' in item:
                        print(f"OTEL event includes {len(item['citations'])} citations")
```

## Common Patterns

### Extract All URLs from Citations

```python {title="extract_urls.py"}
from pydantic_ai import Agent, TextPart, URLCitation

agent = Agent('openai:gpt-4o')
result = agent.run_sync('Tell me about Python.')

urls = []
for message in result.new_messages():
    if message.role == 'assistant':
        for part in message.parts:
            if isinstance(part, TextPart) and part.citations:
                for citation in part.citations:
                    if isinstance(citation, URLCitation):
                        urls.append(citation.url)

print(f"Found {len(urls)} unique URLs: {set(urls)}")
```

### Map Citations to Text Ranges

```python {title="map_citations.py"}
from pydantic_ai import Agent, TextPart, URLCitation

agent = Agent('openai:gpt-4o')
result = agent.run_sync('Tell me about Python.')

for message in result.new_messages():
    if message.role == 'assistant':
        for part in message.parts:
            if isinstance(part, TextPart) and part.citations:
                for citation in part.citations:
                    if isinstance(citation, URLCitation):
                        cited_text = part.content[citation.start_index:citation.end_index]
                        print(f"'{cited_text}' is cited from {citation.url}")
```

## See Also

- [Citations Overview](overview.md) - Introduction to citations
- [Provider-Specific Examples](../examples/citations/) - Detailed examples for each provider
- [API Reference](../api/messages.md#citations) - Complete API documentation
