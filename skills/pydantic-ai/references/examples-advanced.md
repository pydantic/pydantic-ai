# Advanced Examples Reference

Real-world application patterns demonstrating PydanticAI capabilities.

## RAG Application (Retrieval-Augmented Generation)

A complete RAG pipeline using embeddings for semantic search.

```python {title="rag_example.py" test="skip"}
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel

from pydantic_ai import Agent, RunContext
from pydantic_ai.embeddings import OpenAIEmbedder


@dataclass
class Document:
    content: str
    embedding: list[float]
    metadata: dict[str, str]


@dataclass
class RAGDeps:
    embedder: OpenAIEmbedder
    documents: list[Document]
    top_k: int = 5


class Answer(BaseModel):
    answer: str
    sources: list[str]
    confidence: float


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr, b_arr = np.array(a), np.array(b)
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))


rag_agent = Agent(
    'openai:gpt-5',
    deps_type=RAGDeps,
    output_type=Answer,
    instructions='''
    Answer questions using only the retrieved context.
    If the context doesn't contain enough information, say so.
    Always cite your sources.
    ''',
)


@rag_agent.tool
async def retrieve_context(ctx: RunContext[RAGDeps], query: str) -> str:
    """Retrieve relevant documents for the query."""
    # Embed the query
    query_embedding = await ctx.deps.embedder.embed([query])

    # Find most similar documents
    similarities = [
        (doc, cosine_similarity(query_embedding[0], doc.embedding))
        for doc in ctx.deps.documents
    ]
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Return top-k results
    top_docs = similarities[: ctx.deps.top_k]
    context_parts = []
    for doc, score in top_docs:
        source = doc.metadata.get('source', 'unknown')
        context_parts.append(f'[Source: {source}]\n{doc.content}')

    return '\n\n---\n\n'.join(context_parts)


async def answer_question(question: str, documents: list[Document]) -> Answer:
    embedder = OpenAIEmbedder('text-embedding-3-small')
    deps = RAGDeps(embedder=embedder, documents=documents, top_k=5)

    result = await rag_agent.run(question, deps=deps)
    return result.output
```

## Customer Support Bot

A multi-agent support system with routing and escalation.

```python {title="support_bot.py" test="skip"}
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel

from pydantic_ai import Agent, RunContext, UsageLimits


class Category(str, Enum):
    BILLING = 'billing'
    TECHNICAL = 'technical'
    GENERAL = 'general'
    ESCALATE = 'escalate'


class RouteDecision(BaseModel):
    category: Category
    confidence: float
    reason: str


class SupportResponse(BaseModel):
    message: str
    resolved: bool
    follow_up_needed: bool
    ticket_id: str | None = None


@dataclass
class SupportDeps:
    customer_id: str
    customer_tier: str  # 'free', 'pro', 'enterprise'
    conversation_history: list[str]


# Router agent - classifies incoming requests
router_agent = Agent(
    'openai:gpt-5-mini',  # Fast, cheap for classification
    output_type=RouteDecision,
    instructions='''
    Classify customer support requests into categories.
    Route to 'escalate' if the customer is angry, threatening to cancel,
    or the issue is complex and requires human intervention.
    ''',
)

# Specialist agents
billing_agent = Agent(
    'openai:gpt-5',
    deps_type=SupportDeps,
    output_type=SupportResponse,
    instructions='''
    You handle billing questions: invoices, payments, refunds, plan changes.
    Be empathetic and solution-oriented.
    For enterprise customers, offer more flexibility.
    ''',
)

technical_agent = Agent(
    'anthropic:claude-sonnet-4-5',
    deps_type=SupportDeps,
    output_type=SupportResponse,
    instructions='''
    You handle technical support: bugs, how-to questions, integrations.
    Provide clear, step-by-step instructions.
    Ask clarifying questions if needed.
    ''',
)

general_agent = Agent(
    'openai:gpt-5-mini',
    deps_type=SupportDeps,
    output_type=SupportResponse,
    instructions='''
    You handle general inquiries: product info, feature requests, feedback.
    Be friendly and informative.
    ''',
)


@billing_agent.tool
async def lookup_subscription(ctx: RunContext[SupportDeps]) -> str:
    """Look up the customer's current subscription."""
    # In real app: query database
    return f'Customer {ctx.deps.customer_id} is on {ctx.deps.customer_tier} plan'


@billing_agent.tool
async def check_payment_history(ctx: RunContext[SupportDeps]) -> str:
    """Check recent payment history."""
    # In real app: query payment provider
    return 'Last payment: $99 on 2024-01-15, status: successful'


@technical_agent.tool
async def search_knowledge_base(ctx: RunContext[SupportDeps], query: str) -> str:
    """Search the technical knowledge base."""
    # In real app: semantic search over docs
    return f'Found 3 articles related to: {query}'


async def handle_support_request(
    message: str,
    customer_id: str,
    customer_tier: str,
    history: list[str],
) -> SupportResponse:
    deps = SupportDeps(
        customer_id=customer_id,
        customer_tier=customer_tier,
        conversation_history=history,
    )

    # Step 1: Route the request
    route = await router_agent.run(
        f'Customer message: {message}\n\nConversation history: {history[-3:]}',
        usage_limits=UsageLimits(request_limit=1),
    )

    # Step 2: Handle based on route
    if route.output.category == Category.ESCALATE:
        return SupportResponse(
            message='I understand this is important to you. Let me connect you with a specialist who can help.',
            resolved=False,
            follow_up_needed=True,
            ticket_id='ESC-12345',
        )

    agent = {
        Category.BILLING: billing_agent,
        Category.TECHNICAL: technical_agent,
        Category.GENERAL: general_agent,
    }[route.output.category]

    result = await agent.run(message, deps=deps)
    return result.output
```

## Code Assistant

An agent that can analyze and generate code with safety checks.

```python {title="code_assistant.py" test="skip"}
from dataclasses import dataclass, field

from pydantic import BaseModel

from pydantic_ai import Agent, ModelRetry, RunContext


class CodeAnalysis(BaseModel):
    language: str
    complexity: str  # 'simple', 'moderate', 'complex'
    issues: list[str]
    suggestions: list[str]


class GeneratedCode(BaseModel):
    code: str
    language: str
    explanation: str
    test_cases: list[str]


@dataclass
class CodeDeps:
    allowed_languages: list[str] = field(default_factory=lambda: ['python', 'javascript', 'typescript'])
    max_code_length: int = 5000
    include_tests: bool = True


code_agent = Agent(
    'anthropic:claude-sonnet-4-5',  # Strong at code
    deps_type=CodeDeps,
    instructions='''
    You are a code assistant. You can analyze code for issues,
    suggest improvements, and generate new code.

    Always:
    - Follow best practices for the language
    - Include error handling
    - Write clear, maintainable code
    - Add helpful comments for complex logic
    ''',
)


@code_agent.tool
async def analyze_code(ctx: RunContext[CodeDeps], code: str, language: str) -> CodeAnalysis:
    """Analyze code for issues and improvements."""
    if language not in ctx.deps.allowed_languages:
        raise ModelRetry(f'Language {language} not supported. Use: {ctx.deps.allowed_languages}')

    # In real app: use AST parsing, linters, etc.
    issues = []
    suggestions = []

    # Basic checks
    if len(code) > ctx.deps.max_code_length:
        issues.append('Code exceeds maximum length')

    if 'TODO' in code:
        suggestions.append('Complete TODO items before production')

    if 'print(' in code and language == 'python':
        suggestions.append('Consider using logging instead of print statements')

    complexity = 'simple'
    if code.count('if ') + code.count('for ') + code.count('while ') > 10:
        complexity = 'complex'
    elif code.count('if ') + code.count('for ') + code.count('while ') > 5:
        complexity = 'moderate'

    return CodeAnalysis(
        language=language,
        complexity=complexity,
        issues=issues,
        suggestions=suggestions,
    )


@code_agent.tool
async def execute_code_sandbox(ctx: RunContext[CodeDeps], code: str, language: str) -> str:
    """Execute code in a sandboxed environment (Python only)."""
    if language != 'python':
        raise ModelRetry('Only Python execution is supported')

    if len(code) > 1000:
        raise ModelRetry('Code too long for sandbox execution')

    # Security checks
    forbidden = ['import os', 'import subprocess', 'open(', '__import__', 'eval(', 'exec(']
    for term in forbidden:
        if term in code:
            raise ModelRetry(f'Forbidden operation: {term}')

    # In real app: use proper sandboxing (e.g., Docker, RestrictedPython)
    try:
        # NEVER do this in production without proper sandboxing!
        local_vars: dict = {}
        exec(code, {'__builtins__': {}}, local_vars)  # noqa: S102
        return f'Execution successful. Variables: {list(local_vars.keys())}'
    except Exception as e:
        return f'Execution error: {e}'


@code_agent.output_validator
def validate_generated_code(ctx: RunContext[CodeDeps], output: GeneratedCode) -> GeneratedCode:
    """Validate generated code meets requirements."""
    if output.language not in ctx.deps.allowed_languages:
        raise ModelRetry(f'Generated code in unsupported language: {output.language}')

    if len(output.code) > ctx.deps.max_code_length:
        raise ModelRetry(f'Generated code too long ({len(output.code)} chars)')

    if ctx.deps.include_tests and not output.test_cases:
        raise ModelRetry('Please include test cases for the generated code')

    return output
```

## Data Analysis Agent

An agent for analyzing datasets and generating insights.

```python {title="data_analyst.py" test="skip"}
from dataclasses import dataclass

import pandas as pd
from pydantic import BaseModel

from pydantic_ai import Agent, RunContext


class DataInsight(BaseModel):
    title: str
    description: str
    metric_value: float | None = None
    visualization_type: str | None = None  # 'bar', 'line', 'scatter', 'pie'


class AnalysisReport(BaseModel):
    summary: str
    insights: list[DataInsight]
    recommendations: list[str]
    data_quality_score: float  # 0-1


@dataclass
class AnalysisDeps:
    df: pd.DataFrame
    column_descriptions: dict[str, str]


analyst_agent = Agent(
    'openai:gpt-5',
    deps_type=AnalysisDeps,
    output_type=AnalysisReport,
    instructions='''
    You are a data analyst. Analyze datasets and provide actionable insights.

    Always:
    - Start by understanding the data structure
    - Look for patterns, anomalies, and trends
    - Provide specific, quantified insights
    - Suggest visualizations for key findings
    - Make recommendations based on the data
    ''',
)


@analyst_agent.tool
async def get_data_summary(ctx: RunContext[AnalysisDeps]) -> str:
    """Get a summary of the dataset structure and statistics."""
    df = ctx.deps.df

    summary_parts = [
        f'Rows: {len(df)}, Columns: {len(df.columns)}',
        f'\nColumn types:\n{df.dtypes.to_string()}',
        f'\nMissing values:\n{df.isnull().sum().to_string()}',
    ]

    # Add descriptions if available
    if ctx.deps.column_descriptions:
        desc_str = '\n'.join(f'  {k}: {v}' for k, v in ctx.deps.column_descriptions.items())
        summary_parts.append(f'\nColumn descriptions:\n{desc_str}')

    return '\n'.join(summary_parts)


@analyst_agent.tool
async def get_numeric_statistics(ctx: RunContext[AnalysisDeps], column: str) -> str:
    """Get detailed statistics for a numeric column."""
    df = ctx.deps.df

    if column not in df.columns:
        return f'Column {column} not found'

    if not pd.api.types.is_numeric_dtype(df[column]):
        return f'Column {column} is not numeric'

    stats = df[column].describe()
    return f'Statistics for {column}:\n{stats.to_string()}'


@analyst_agent.tool
async def get_value_counts(ctx: RunContext[AnalysisDeps], column: str, top_n: int = 10) -> str:
    """Get value counts for a categorical column."""
    df = ctx.deps.df

    if column not in df.columns:
        return f'Column {column} not found'

    counts = df[column].value_counts().head(top_n)
    return f'Top {top_n} values in {column}:\n{counts.to_string()}'


@analyst_agent.tool
async def calculate_correlation(ctx: RunContext[AnalysisDeps], col1: str, col2: str) -> str:
    """Calculate correlation between two numeric columns."""
    df = ctx.deps.df

    for col in [col1, col2]:
        if col not in df.columns:
            return f'Column {col} not found'
        if not pd.api.types.is_numeric_dtype(df[col]):
            return f'Column {col} is not numeric'

    corr = df[col1].corr(df[col2])
    return f'Correlation between {col1} and {col2}: {corr:.4f}'


@analyst_agent.tool
async def find_outliers(ctx: RunContext[AnalysisDeps], column: str, method: str = 'iqr') -> str:
    """Find outliers in a numeric column using IQR or Z-score method."""
    df = ctx.deps.df

    if column not in df.columns:
        return f'Column {column} not found'

    if not pd.api.types.is_numeric_dtype(df[column]):
        return f'Column {column} is not numeric'

    if method == 'iqr':
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = df[(df[column] < lower) | (df[column] > upper)]
    else:  # z-score
        mean = df[column].mean()
        std = df[column].std()
        outliers = df[abs((df[column] - mean) / std) > 3]

    return f'Found {len(outliers)} outliers in {column} ({len(outliers) / len(df) * 100:.1f}% of data)'


async def analyze_dataset(
    df: pd.DataFrame,
    column_descriptions: dict[str, str] | None = None,
) -> AnalysisReport:
    deps = AnalysisDeps(
        df=df,
        column_descriptions=column_descriptions or {},
    )

    result = await analyst_agent.run(
        'Analyze this dataset and provide key insights and recommendations.',
        deps=deps,
    )
    return result.output
```

## Webhook Event Handler

An agent that processes incoming webhooks and takes appropriate actions.

```python {title="webhook_handler.py" test="skip"}
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pydantic import BaseModel

from pydantic_ai import Agent, RunContext


class WebhookAction(BaseModel):
    action_type: str  # 'notify', 'update_db', 'trigger_workflow', 'ignore'
    details: dict[str, Any]
    priority: str  # 'low', 'medium', 'high', 'critical'
    reason: str


@dataclass
class WebhookDeps:
    source: str  # 'github', 'stripe', 'slack', etc.
    event_type: str
    payload: dict[str, Any]
    timestamp: datetime


webhook_agent = Agent(
    'openai:gpt-5-mini',  # Fast for event processing
    deps_type=WebhookDeps,
    output_type=WebhookAction,
    instructions='''
    Process incoming webhook events and determine the appropriate action.

    Priority levels:
    - critical: Security issues, payment failures, system errors
    - high: Important user actions, significant state changes
    - medium: Regular updates, non-urgent notifications
    - low: Informational events, analytics

    Always explain your reasoning for the chosen action and priority.
    ''',
)


@webhook_agent.tool
async def get_event_context(ctx: RunContext[WebhookDeps]) -> str:
    """Get context about the webhook event."""
    return f'''
Source: {ctx.deps.source}
Event Type: {ctx.deps.event_type}
Timestamp: {ctx.deps.timestamp.isoformat()}
Payload Keys: {list(ctx.deps.payload.keys())}
'''


@webhook_agent.tool
async def extract_payload_field(ctx: RunContext[WebhookDeps], field_path: str) -> str:
    """Extract a specific field from the payload using dot notation."""
    parts = field_path.split('.')
    value = ctx.deps.payload

    for part in parts:
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return f'Field {field_path} not found'

    return str(value)


async def process_webhook(
    source: str,
    event_type: str,
    payload: dict[str, Any],
) -> WebhookAction:
    deps = WebhookDeps(
        source=source,
        event_type=event_type,
        payload=payload,
        timestamp=datetime.now(),
    )

    result = await webhook_agent.run(
        f'Process this {source} webhook event: {event_type}',
        deps=deps,
    )
    return result.output
```

## Patterns Demonstrated

| Pattern | Example | Key Features |
|---------|---------|--------------|
| RAG | rag_example.py | Embeddings, similarity search, source citation |
| Multi-agent routing | support_bot.py | Router agent, specialist delegation, escalation |
| Code generation | code_assistant.py | Output validation, sandbox execution, safety checks |
| Data analysis | data_analyst.py | DataFrame tools, statistical analysis, insights |
| Event processing | webhook_handler.py | Fast classification, structured actions |

## See Also

- [multi-agent.md](multi-agent.md) — Multi-agent patterns
- [tools.md](tools.md) — Tool registration
- [output.md](output.md) — Structured output
- [embeddings.md](embeddings.md) — Embeddings for RAG
- [dependencies.md](dependencies.md) — Dependency injection
