# Embeddings

Embeddings are vector representations of text that capture semantic meaning. They're essential for building:

- **Semantic search** — Find documents based on meaning, not just keyword matching
- **RAG (Retrieval-Augmented Generation)** — Retrieve relevant context for your AI agents
- **Similarity detection** — Find similar documents, detect duplicates, or cluster content
- **Classification** — Use embeddings as features for downstream ML models

Pydantic AI provides a unified interface for generating embeddings across multiple providers.

## Quick Start

The [`Embedder`][pydantic_ai.embeddings.Embedder] class is the high-level interface for generating embeddings:

```python {title="embeddings_quickstart.py"}
from pydantic_ai import Embedder

embedder = Embedder('openai:text-embedding-3-small')


async def main():
    # Embed a search query
    result = await embedder.embed_query('What is machine learning?')
    print(f'Embedding dimensions: {len(result.embeddings[0])}')
    #> Embedding dimensions: 1536

    # Embed multiple documents at once
    docs = [
        'Machine learning is a subset of AI.',
        'Deep learning uses neural networks.',
        'Python is a programming language.',
    ]
    result = await embedder.embed_documents(docs)
    print(f'Embedded {len(result.embeddings)} documents')
    #> Embedded 3 documents
```

_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)_

!!! tip "Queries vs Documents"
    Some embedding models optimize differently for queries and documents. Use
    [`embed_query()`][pydantic_ai.embeddings.Embedder.embed_query] for search queries and
    [`embed_documents()`][pydantic_ai.embeddings.Embedder.embed_documents] for content you're indexing.

## Embedding Result

All embed methods return an [`EmbeddingResult`][pydantic_ai.embeddings.EmbeddingResult] containing the embeddings along with useful metadata.

For convenience, you can access embeddings either by index (`result[0]`) or by the original input text (`result['Hello world']`).

```python {title="embedding_result.py"}
from pydantic_ai import Embedder

embedder = Embedder('openai:text-embedding-3-small')


async def main():
    result = await embedder.embed_query('Hello world')

    # Access embeddings - each is a sequence of floats
    embedding = result.embeddings[0]  # By index via .embeddings
    embedding = result[0]  # Or directly via __getitem__
    embedding = result['Hello world']  # Or by original input text
    print(f'Dimensions: {len(embedding)}')
    #> Dimensions: 1536

    # Check usage
    print(f'Tokens used: {result.usage.input_tokens}')
    #> Tokens used: 2

    # Calculate cost (requires `genai-prices` to have pricing data for the model)
    cost = result.cost()
    print(f'Cost: ${cost.total_price:.6f}')
    #> Cost: $0.000000
```

_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)_

## Providers

### OpenAI

[`OpenAIEmbeddingModel`][pydantic_ai.embeddings.openai.OpenAIEmbeddingModel] works with OpenAI's embeddings API and any [OpenAI-compatible provider](models/openai.md#openai-compatible-models).

#### Install

To use OpenAI embedding models, you need to either install `pydantic-ai`, or install `pydantic-ai-slim` with the `openai` optional group:

```bash
pip/uv-add "pydantic-ai-slim[openai]"
```

#### Configuration

To use `OpenAIEmbeddingModel` with the OpenAI API, go to [platform.openai.com](https://platform.openai.com/) and follow your nose until you find the place to generate an API key. Once you have the API key, you can set it as an environment variable:

```bash
export OPENAI_API_KEY='your-api-key'
```

You can then use the model:

```python {title="openai_embeddings.py"}
from pydantic_ai import Embedder

embedder = Embedder('openai:text-embedding-3-small')


async def main():
    result = await embedder.embed_query('Hello world')
    print(len(result.embeddings[0]))
    #> 1536
```

_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)_

See [OpenAI's embedding models](https://platform.openai.com/docs/guides/embeddings) for available models.

#### Dimension Control

OpenAI's `text-embedding-3-*` models support dimension reduction via the `dimensions` setting:

```python {title="openai_dimensions.py"}
from pydantic_ai import Embedder
from pydantic_ai.embeddings import EmbeddingSettings

embedder = Embedder(
    'openai:text-embedding-3-small',
    settings=EmbeddingSettings(dimensions=256),
)


async def main():
    result = await embedder.embed_query('Hello world')
    print(len(result.embeddings[0]))
    #> 256
```

_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)_

#### OpenAI-Compatible Providers {#openai-compatible}

Since [`OpenAIEmbeddingModel`][pydantic_ai.embeddings.openai.OpenAIEmbeddingModel] uses the same provider system as [`OpenAIChatModel`][pydantic_ai.models.openai.OpenAIChatModel], you can use it with any [OpenAI-compatible provider](models/openai.md#openai-compatible-models):

```python {title="openai_compatible_embeddings.py"}
# Using Azure OpenAI
from openai import AsyncAzureOpenAI

from pydantic_ai import Embedder
from pydantic_ai.embeddings.openai import OpenAIEmbeddingModel
from pydantic_ai.providers.openai import OpenAIProvider

azure_client = AsyncAzureOpenAI(
    azure_endpoint='https://your-resource.openai.azure.com',
    api_version='2024-02-01',
    api_key='your-azure-key',
)
model = OpenAIEmbeddingModel(
    'text-embedding-3-small',
    provider=OpenAIProvider(openai_client=azure_client),
)
embedder = Embedder(model)


# Using any OpenAI-compatible API
model = OpenAIEmbeddingModel(
    'your-model-name',
    provider=OpenAIProvider(
        base_url='https://your-provider.com/v1',
        api_key='your-api-key',
    ),
)
embedder = Embedder(model)
```

For providers with dedicated provider classes (like [`OllamaProvider`][pydantic_ai.providers.ollama.OllamaProvider] or [`AzureProvider`][pydantic_ai.providers.azure.AzureProvider]), you can use the shorthand syntax:

```python
from pydantic_ai import Embedder

embedder = Embedder('azure:text-embedding-3-small')
embedder = Embedder('ollama:nomic-embed-text')
```

See [OpenAI-compatible Models](models/openai.md#openai-compatible-models) for the full list of supported providers.

### Google

[`GoogleEmbeddingModel`][pydantic_ai.embeddings.google.GoogleEmbeddingModel] works with Google's embedding models via the Gemini API (Google AI Studio) or Vertex AI.

#### Install

To use Google embedding models, you need to either install `pydantic-ai`, or install `pydantic-ai-slim` with the `google` optional group:

```bash
pip/uv-add "pydantic-ai-slim[google]"
```

#### Configuration

To use `GoogleEmbeddingModel` with the Gemini API, go to [aistudio.google.com](https://aistudio.google.com/) and generate an API key. Once you have the API key, you can set it as an environment variable:

```bash
export GOOGLE_API_KEY='your-api-key'
```

You can then use the model:

```python {title="google_embeddings.py"}
from pydantic_ai import Embedder

embedder = Embedder('google-gla:gemini-embedding-001')


async def main():
    result = await embedder.embed_query('Hello world')
    print(len(result.embeddings[0]))
    #> 3072
```

_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)_

See the [Google Embeddings documentation](https://ai.google.dev/gemini-api/docs/embeddings) for available models.

##### Vertex AI

To use Google's embedding models via Vertex AI instead of the Gemini API, use the `google-vertex` provider prefix:

```python {title="google_vertex_embeddings.py"}
from pydantic_ai import Embedder
from pydantic_ai.embeddings.google import GoogleEmbeddingModel
from pydantic_ai.providers.google import GoogleProvider

# Using provider prefix
embedder = Embedder('google-vertex:gemini-embedding-001')

# Or with explicit provider configuration
model = GoogleEmbeddingModel(
    'gemini-embedding-001',
    provider=GoogleProvider(vertexai=True, project='my-project', location='us-central1'),
)
embedder = Embedder(model)
```

See the [Google provider documentation](models/google.md#vertex-ai-enterprisecloud) for more details on Vertex AI authentication options, including application default credentials, service accounts, and API keys.

#### Dimension Control

Google's embedding models support dimension reduction via the `dimensions` setting:

```python {title="google_dimensions.py"}
from pydantic_ai import Embedder
from pydantic_ai.embeddings import EmbeddingSettings

embedder = Embedder(
    'google-gla:gemini-embedding-001',
    settings=EmbeddingSettings(dimensions=768),
)


async def main():
    result = await embedder.embed_query('Hello world')
    print(len(result.embeddings[0]))
    #> 768
```

_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)_

#### Google-Specific Settings

Google models support additional settings via [`GoogleEmbeddingSettings`][pydantic_ai.embeddings.google.GoogleEmbeddingSettings]:

```python {title="google_settings.py"}
from pydantic_ai import Embedder
from pydantic_ai.embeddings.google import GoogleEmbeddingSettings

embedder = Embedder(
    'google-gla:gemini-embedding-001',
    settings=GoogleEmbeddingSettings(
        dimensions=768,
        google_task_type='SEMANTIC_SIMILARITY',  # Optimize for similarity comparison
    ),
)
```

See [Google's task type documentation](https://ai.google.dev/gemini-api/docs/embeddings#task-types) for available task types. By default, `embed_query()` uses `RETRIEVAL_QUERY` and `embed_documents()` uses `RETRIEVAL_DOCUMENT`.

### Cohere

[`CohereEmbeddingModel`][pydantic_ai.embeddings.cohere.CohereEmbeddingModel] provides access to Cohere's embedding models, which offer multilingual support and various model sizes.

#### Install

To use Cohere embedding models, you need to either install `pydantic-ai`, or install `pydantic-ai-slim` with the `cohere` optional group:

```bash
pip/uv-add "pydantic-ai-slim[cohere]"
```

#### Configuration

To use `CohereEmbeddingModel`, go to [dashboard.cohere.com/api-keys](https://dashboard.cohere.com/api-keys) and follow your nose until you find the place to generate an API key. Once you have the API key, you can set it as an environment variable:

```bash
export CO_API_KEY='your-api-key'
```

You can then use the model:

```python {title="cohere_embeddings.py"}
from pydantic_ai import Embedder

embedder = Embedder('cohere:embed-v4.0')


async def main():
    result = await embedder.embed_query('Hello world')
    print(len(result.embeddings[0]))
    #> 1024
```

_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)_

See the [Cohere Embed documentation](https://docs.cohere.com/docs/cohere-embed) for available models.

#### Cohere-Specific Settings

Cohere models support additional settings via [`CohereEmbeddingSettings`][pydantic_ai.embeddings.cohere.CohereEmbeddingSettings]:

```python {title="cohere_settings.py"}
from pydantic_ai import Embedder
from pydantic_ai.embeddings.cohere import CohereEmbeddingSettings

embedder = Embedder(
    'cohere:embed-v4.0',
    settings=CohereEmbeddingSettings(
        dimensions=512,
        cohere_truncate='END',  # Truncate long inputs instead of erroring
        cohere_max_tokens=256,  # Limit tokens per input
    ),
)
```

### VoyageAI

[`VoyageAIEmbeddingModel`][pydantic_ai.embeddings.voyageai.VoyageAIEmbeddingModel] provides access to VoyageAI's embedding models, which are optimized for retrieval with specialized models for code, finance, and legal domains.

#### Install

To use VoyageAI embedding models, you need to install `pydantic-ai-slim` with the `voyageai` optional group:

```bash
pip/uv-add "pydantic-ai-slim[voyageai]"
```

#### Configuration

To use `VoyageAIEmbeddingModel`, go to [dash.voyageai.com](https://dash.voyageai.com/) to generate an API key. Once you have the API key, you can set it as an environment variable:

```bash
export VOYAGE_API_KEY='your-api-key'
```

You can then use the model:

```python {title="voyageai_embeddings.py"}
from pydantic_ai import Embedder

embedder = Embedder('voyageai:voyage-3.5')


async def main():
    result = await embedder.embed_query('Hello world')
    print(len(result.embeddings[0]))
    #> 1024
```

_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)_

See the [VoyageAI Embeddings documentation](https://docs.voyageai.com/docs/embeddings) for available models.

#### VoyageAI-Specific Settings

VoyageAI models support additional settings via [`VoyageAIEmbeddingSettings`][pydantic_ai.embeddings.voyageai.VoyageAIEmbeddingSettings]:

```python {title="voyageai_settings.py"}
from pydantic_ai import Embedder
from pydantic_ai.embeddings.voyageai import VoyageAIEmbeddingSettings

embedder = Embedder(
    'voyageai:voyage-3.5',
    settings=VoyageAIEmbeddingSettings(
        dimensions=512,  # Reduce output dimensions
        voyageai_input_type='document',  # Override input type for all requests
    ),
)
```

### Bedrock

[`BedrockEmbeddingModel`][pydantic_ai.embeddings.bedrock.BedrockEmbeddingModel] provides access to embedding models through AWS Bedrock, including Amazon Titan, Cohere, and Amazon Nova models.

#### Install

To use Bedrock embedding models, you need to either install `pydantic-ai`, or install `pydantic-ai-slim` with the `bedrock` optional group:

```bash
pip/uv-add "pydantic-ai-slim[bedrock]"
```

#### Configuration

Authentication with AWS Bedrock uses standard AWS credentials. See the [Bedrock provider documentation](models/bedrock.md#environment-variables) for details on configuring credentials via environment variables, AWS credentials file, or IAM roles.

Ensure your AWS account has access to the Bedrock embedding models you want to use. See [AWS Bedrock model access](https://docs.aws.amazon.com/bedrock/latest/userguide/model-access.html) for details.

#### Basic Usage

```python {title="bedrock_embeddings.py" test="skip"}
from pydantic_ai import Embedder

# Using Amazon Titan
embedder = Embedder('bedrock:amazon.titan-embed-text-v2:0')


async def main():
    result = await embedder.embed_query('Hello world')
    print(len(result.embeddings[0]))
    #> 1024
```

_(This example requires AWS credentials configured)_

#### Supported Models

Bedrock supports three families of embedding models. See the [AWS Bedrock documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html) for the full list of available models.

**Amazon Titan:**

- `amazon.titan-embed-text-v1` — 1536 dimensions (fixed), 8K tokens
- `amazon.titan-embed-text-v2:0` — 256/384/1024 dimensions (configurable, default: 1024), 8K tokens

**Cohere Embed:**

- `cohere.embed-english-v3` — English-only, 1024 dimensions (fixed), 512 tokens
- `cohere.embed-multilingual-v3` — Multilingual, 1024 dimensions (fixed), 512 tokens
- `cohere.embed-v4:0` — 256/512/1024/1536 dimensions (configurable, default: 1536), 128K tokens

**Amazon Nova:**

- `amazon.nova-2-multimodal-embeddings-v1:0` — 256/384/1024/3072 dimensions (configurable, default: 3072), 8K tokens

#### Titan-Specific Settings

Titan v2 supports vector normalization for direct similarity calculations via `bedrock_titan_normalize` (default: `True`). Titan v1 does not support this setting.

```python {title="bedrock_titan.py"}
from pydantic_ai import Embedder
from pydantic_ai.embeddings.bedrock import BedrockEmbeddingSettings

embedder = Embedder(
    'bedrock:amazon.titan-embed-text-v2:0',
    settings=BedrockEmbeddingSettings(
        dimensions=512,
        bedrock_titan_normalize=True,
    ),
)
```

!!! note
    Titan models do not support the `truncate` setting. The `dimensions` setting is only supported by Titan v2.

#### Cohere-Specific Settings

Cohere models on Bedrock support additional settings via [`BedrockEmbeddingSettings`][pydantic_ai.embeddings.bedrock.BedrockEmbeddingSettings]:

- `bedrock_cohere_input_type` — By default, `embed_query()` uses `'search_query'` and `embed_documents()` uses `'search_document'`. Also accepts `'classification'` or `'clustering'`.
- `bedrock_cohere_truncate` — Fine-grained truncation control: `'NONE'` (default, error on overflow), `'START'`, or `'END'`. Overrides the base `truncate` setting.
- `bedrock_cohere_max_tokens` — Limits tokens per input (default: 128000). Only supported by Cohere v4.

```python {title="bedrock_cohere.py"}
from pydantic_ai import Embedder
from pydantic_ai.embeddings.bedrock import BedrockEmbeddingSettings

embedder = Embedder(
    'bedrock:cohere.embed-v4:0',
    settings=BedrockEmbeddingSettings(
        dimensions=512,
        bedrock_cohere_max_tokens=1000,
        bedrock_cohere_truncate='END',
    ),
)
```

!!! note
    The `dimensions` and `bedrock_cohere_max_tokens` settings are only supported by Cohere v4. Cohere v3 models have fixed 1024 dimensions.

#### Nova-Specific Settings

Nova models on Bedrock support additional settings via [`BedrockEmbeddingSettings`][pydantic_ai.embeddings.bedrock.BedrockEmbeddingSettings]:

- `bedrock_nova_truncate` — Fine-grained truncation control: `'NONE'` (default, error on overflow), `'START'`, or `'END'`. Overrides the base `truncate` setting.
- `bedrock_nova_embedding_purpose` — By default, `embed_query()` uses `'GENERIC_RETRIEVAL'` and `embed_documents()` uses `'GENERIC_INDEX'`. Also accepts `'TEXT_RETRIEVAL'`, `'CLASSIFICATION'`, or `'CLUSTERING'`.

```python {title="bedrock_nova.py"}
from pydantic_ai import Embedder
from pydantic_ai.embeddings.bedrock import BedrockEmbeddingSettings

embedder = Embedder(
    'bedrock:amazon.nova-2-multimodal-embeddings-v1:0',
    settings=BedrockEmbeddingSettings(
        dimensions=1024,
        bedrock_nova_embedding_purpose='TEXT_RETRIEVAL',
        truncate=True,
    ),
)
```

#### Concurrency Settings

Models that don't support batch embedding (Titan and Nova) make individual API requests for each input text. By default, these requests run concurrently with a maximum of 5 parallel requests.

You can adjust this with the `bedrock_max_concurrency` setting:

```python {title="bedrock_concurrency.py"}
from pydantic_ai import Embedder
from pydantic_ai.embeddings.bedrock import BedrockEmbeddingSettings

# Increase concurrency for faster throughput
embedder = Embedder(
    'bedrock:amazon.titan-embed-text-v2:0',
    settings=BedrockEmbeddingSettings(bedrock_max_concurrency=10),
)

# Or reduce concurrency to avoid rate limits
embedder = Embedder(
    'bedrock:amazon.nova-2-multimodal-embeddings-v1:0',
    settings=BedrockEmbeddingSettings(bedrock_max_concurrency=2),
)
```

#### Regional Prefixes (Cross-Region Inference)

Bedrock supports cross-region inference using geographic prefixes like `us.`, `eu.`, or `apac.`:

```python {title="bedrock_regional.py"}
from pydantic_ai import Embedder

embedder = Embedder('bedrock:us.amazon.titan-embed-text-v2:0')
```

#### Using a Custom Provider

For advanced configuration like explicit credentials or a custom boto3 client, you can create a [`BedrockProvider`][pydantic_ai.providers.bedrock.BedrockProvider] directly. See the [Bedrock provider documentation](models/bedrock.md#provider-argument) for more details.

```python {title="bedrock_provider.py"}
from pydantic_ai import Embedder
from pydantic_ai.embeddings.bedrock import BedrockEmbeddingModel
from pydantic_ai.providers.bedrock import BedrockProvider

provider = BedrockProvider(
    region_name='us-west-2',
    aws_access_key_id='your-access-key',
    aws_secret_access_key='your-secret-key',
)

model = BedrockEmbeddingModel('amazon.titan-embed-text-v2:0', provider=provider)
embedder = Embedder(model)
```

!!! note "Token Counting"
    Bedrock embedding models do not support the `count_tokens()` method because AWS Bedrock's token counting API only works with text generation models (Claude, Llama, etc.), not embedding models. Calling `count_tokens()` will raise `NotImplementedError`.

### Sentence Transformers (Local)

[`SentenceTransformerEmbeddingModel`][pydantic_ai.embeddings.sentence_transformers.SentenceTransformerEmbeddingModel] runs embeddings locally using the [sentence-transformers](https://www.sbert.net/) library. This is ideal for:

- **Privacy** — Data never leaves your infrastructure
- **Cost** — No API charges for high-volume workloads
- **Offline use** — No internet connection required after model download

#### Install

To use Sentence Transformers embedding models, you need to install `pydantic-ai-slim` with the `sentence-transformers` optional group:

```bash
pip/uv-add "pydantic-ai-slim[sentence-transformers]"
```

#### Usage

```python {title="sentence_transformers_embeddings.py"}
from pydantic_ai import Embedder

# Model is downloaded from Hugging Face on first use
embedder = Embedder('sentence-transformers:all-MiniLM-L6-v2')


async def main():
    result = await embedder.embed_query('Hello world')
    print(len(result.embeddings[0]))
    #> 384
```

_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)_

See the [Sentence-Transformers pretrained models](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html) documentation for available models.

#### Device Selection

Control which device to use for inference:

```python {title="sentence_transformers_device.py"}
from pydantic_ai import Embedder
from pydantic_ai.embeddings.sentence_transformers import (
    SentenceTransformersEmbeddingSettings,
)

embedder = Embedder(
    'sentence-transformers:all-MiniLM-L6-v2',
    settings=SentenceTransformersEmbeddingSettings(
        sentence_transformers_device='cuda',  # Use GPU
        sentence_transformers_normalize_embeddings=True,  # L2 normalize
    ),
)
```

#### Using an Existing Model Instance

If you need more control over model initialization:

```python {title="sentence_transformers_instance.py"}
from sentence_transformers import SentenceTransformer

from pydantic_ai import Embedder
from pydantic_ai.embeddings.sentence_transformers import (
    SentenceTransformerEmbeddingModel,
)

# Create and configure the model yourself
st_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

# Wrap it for use with Pydantic AI
model = SentenceTransformerEmbeddingModel(st_model)
embedder = Embedder(model)
```

## Settings

[`EmbeddingSettings`][pydantic_ai.embeddings.EmbeddingSettings] provides common configuration options that work across providers:

- `dimensions`: Reduce the output embedding dimensions (supported by OpenAI, Google, Cohere, Bedrock, VoyageAI)
- `truncate`: When `True`, truncate input text that exceeds the model's context length instead of raising an error (supported by Cohere, Bedrock, VoyageAI)

Settings can be specified at the embedder level (applied to all calls) or per-call:

```python {title="embedding_settings.py"}
from pydantic_ai import Embedder
from pydantic_ai.embeddings import EmbeddingSettings

# Default settings for all calls
embedder = Embedder(
    'openai:text-embedding-3-small',
    settings=EmbeddingSettings(dimensions=512),
)


async def main():
    # Override for a specific call
    result = await embedder.embed_query(
        'Hello world',
        settings=EmbeddingSettings(dimensions=256),
    )
    print(len(result.embeddings[0]))
    #> 256
```

_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)_

## Token Counting

You can check token counts before embedding to avoid exceeding model limits:

```python {title="token_counting.py"}
from pydantic_ai import Embedder

embedder = Embedder('openai:text-embedding-3-small')


async def main():
    text = 'Hello world, this is a test.'

    # Count tokens in text
    token_count = await embedder.count_tokens(text)
    print(f'Tokens: {token_count}')
    #> Tokens: 7

    # Check model's maximum input tokens (returns None if unknown)
    max_tokens = await embedder.max_input_tokens()
    print(f'Max tokens: {max_tokens}')
    #> Max tokens: 1024
```

_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)_

## Testing

Use [`TestEmbeddingModel`][pydantic_ai.embeddings.TestEmbeddingModel] for testing without making API calls:

```python {title="testing_embeddings.py"}
from pydantic_ai import Embedder
from pydantic_ai.embeddings import TestEmbeddingModel


async def test_my_rag_system():
    embedder = Embedder('openai:text-embedding-3-small')
    test_model = TestEmbeddingModel()

    with embedder.override(model=test_model):
        result = await embedder.embed_query('test query')

        # TestEmbeddingModel returns deterministic embeddings
        assert result.embeddings[0] == [1.0] * 8

        # Check what settings were used
        assert test_model.last_settings is not None
```

## Instrumentation

Enable OpenTelemetry instrumentation for debugging and monitoring:

```python {title="instrumented_embeddings.py"}
import logfire

from pydantic_ai import Embedder

logfire.configure()

# Instrument a specific embedder
embedder = Embedder('openai:text-embedding-3-small', instrument=True)

# Or instrument all embedders globally
Embedder.instrument_all()
```

See the [Debugging and Monitoring guide](logfire.md) for more details on using Logfire with Pydantic AI.

## Building Custom Embedding Models

To integrate a custom embedding provider, subclass [`EmbeddingModel`][pydantic_ai.embeddings.EmbeddingModel]:

```python {title="custom_embedding_model.py"}
from collections.abc import Sequence

from pydantic_ai.embeddings import EmbeddingModel, EmbeddingResult, EmbeddingSettings
from pydantic_ai.embeddings.result import EmbedInputType


class MyCustomEmbeddingModel(EmbeddingModel):
    @property
    def model_name(self) -> str:
        return 'my-custom-model'

    @property
    def system(self) -> str:
        return 'my-provider'

    async def embed(
        self,
        inputs: str | Sequence[str],
        *,
        input_type: EmbedInputType,
        settings: EmbeddingSettings | None = None,
    ) -> EmbeddingResult:
        inputs, settings = self.prepare_embed(inputs, settings)

        # Call your embedding API here
        embeddings = [[0.1, 0.2, 0.3] for _ in inputs]  # Placeholder

        return EmbeddingResult(
            embeddings=embeddings,
            inputs=inputs,
            input_type=input_type,
            model_name=self.model_name,
            provider_name=self.system,
        )
```

Use [`WrapperEmbeddingModel`][pydantic_ai.embeddings.WrapperEmbeddingModel] if you want to wrap an existing model to add custom behavior like caching or logging.
