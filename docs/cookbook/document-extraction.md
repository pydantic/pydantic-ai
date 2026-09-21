---
title: Extract structured data from a PDF
description: Send a document to a multimodal model and validate the extracted fields before application code uses them.
---

# Extract structured data from a PDF

Use `DocumentUrl` with a Pydantic output type for the fields your application requires. The model provider downloads the URL, so use a signed URL with a short expiry for private documents and confirm that provider-side fetching is permitted.

```bash
pip/uv-add "pydantic-ai-slim[openai]"
export OPENAI_API_KEY=your-api-key
```

```python {dunder_name="not_main"}
import asyncio

from pydantic import BaseModel

from pydantic_ai import Agent, DocumentUrl


class Paper(BaseModel):
    title: str
    document_type: str
    published_year: int


agent = Agent(
    'openai:gpt-5.6-sol',
    output_type=Paper,
    instructions='Extract only fields visible in the supplied document. Do not infer missing values.',
)


async def main() -> None:
    result = await agent.run(
        [
            'Extract this paper.',
            DocumentUrl(url='https://arxiv.org/pdf/2403.05530'),
        ]
    )
    print(result.output.title)
    #> The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits
    print(result.output.published_year)
    #> 2024


if __name__ == '__main__':
    asyncio.run(main())
```

Keep the source URL or content hash with the validated result for auditability. Add optional fields when a value may legitimately be absent; making every field optional hides extraction failures. Use `BinaryContent` instead when the document must not be fetched from a public or signed URL.
