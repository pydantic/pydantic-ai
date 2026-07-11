# Bedrock Managed Knowledge Base Support

## Overview
Adds a Pydantic AI tool extension that queries Amazon Bedrock Knowledge Bases for managed retrieval within agents.

## Usage
```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.ext.bedrock_kb import create_bedrock_kb_tool

agent = Agent(
    'anthropic:claude-sonnet-4-20250514',
    tools=[create_bedrock_kb_tool(knowledge_base_id='YOUR_KB_ID')],
)
result = agent.run_sync('What are the deployment requirements?')
print(result.output)
```

## Configuration
| Variable | Description | Default |
|---|---|---|
| KNOWLEDGE_BASE_ID | Bedrock Knowledge Base ID | None |
| AWS_REGION | AWS region for the KB | us-east-1 |
| AWS_PROFILE | AWS credentials profile | None |
| USE_AGENTIC_RETRIEVAL | Enable agentic retrieval | true |
| MAX_RESULTS | Maximum retrieval results | 5 |

## Features
- Managed search (no vector store needed)
- Agentic retrieval with query decomposition + reranking
- Automatic fallback to plain Retrieve if agentic fails
- Multi-source support (S3, Web, Confluence, SharePoint)
- Type-safe results with Pydantic models

## SDK Requirements
- boto3 >= 1.43
- pydantic-ai-slim[bedrock] >= 0.1

## Required IAM Permissions
```json
{
  "Effect": "Allow",
  "Action": [
    "bedrock:Retrieve",
    "bedrock:AgenticRetrieve"
  ],
  "Resource": "arn:aws:bedrock:<region>:<account-id>:knowledge-base/<kb-id>"
}
```

## References
- [Build a Managed Knowledge Base](https://docs.aws.amazon.com/bedrock/latest/userguide/kb-build-managed.html)
- [Retrieve API](https://docs.aws.amazon.com/bedrock/latest/userguide/kb-test-retrieve.html)
- [Agentic Retrieval](https://docs.aws.amazon.com/bedrock/latest/userguide/kb-test-agentic.html)
