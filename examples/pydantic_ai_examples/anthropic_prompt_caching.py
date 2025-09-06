#!/usr/bin/env python3
"""Example script to test prompt caching with AWS Bedrock.

This script demonstrates:
1. Manual cache point insertion
2. Cache usage metrics
3. Cost savings from caching

Prerequisites:
- AWS credentials configured
- Access to Claude 4 Sonnet on Bedrock EU region
- Install dependencies: pip install "pydantic-ai-slim[bedrock]"
"""

import asyncio
import os
from collections.abc import Callable

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import (
    CachePoint,
    ModelMessage,
    ModelRequest,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.bedrock import BedrockConverseModel
from pydantic_ai.providers.bedrock import BedrockProvider

# Some long context for testing
# Note that Bedrock prompt caching has a minimum token length of 1024 tokens.
LONG_CONTEXT = """
This is a comprehensive company handbook containing detailed information about our organization.

COMPANY BACKGROUND AND HISTORY:
We are a leading technology company founded in 2018 that specializes in cutting-edge artificial intelligence and machine learning solutions. Our company was started by a team of former researchers from top universities including MIT, Stanford, and Carnegie Mellon. We began as a small startup with just 5 people working out of a garage in Palo Alto, and have since grown to over 250 employees across multiple offices worldwide.

Our initial focus was on natural language processing, but we quickly expanded into computer vision, predictive analytics, and reinforcement learning. We've raised over $150 million in funding across Series A, B, and C rounds from leading venture capital firms including Andreessen Horowitz, Sequoia Capital, and Google Ventures.

PRODUCTS AND SERVICES:
Our flagship products include:

1. NLP Suite - A comprehensive natural language processing platform that includes:
   - Sentiment analysis with 99.2% accuracy across 50+ languages
   - Named entity recognition supporting multilingual content
   - Text summarization using transformer models and neural architectures
   - Question answering systems for enterprise knowledge bases
   - Conversational AI chatbots with advanced context understanding
   - Document classification and content moderation tools

2. Vision AI Platform - Computer vision solutions featuring:
   - Object detection and classification with real-time processing capabilities
   - Facial recognition and emotion detection systems (privacy-compliant)
   - Medical image analysis for diagnostic assistance in healthcare
   - Quality control systems for manufacturing and production lines
   - Autonomous vehicle perception systems for transportation
   - Augmented reality content recognition and spatial mapping

3. Predictive Analytics Engine - Advanced forecasting tools including:
   - Demand forecasting for retail and e-commerce optimization
   - Financial risk assessment and fraud detection algorithms
   - Customer churn prediction and retention strategies
   - Supply chain optimization and logistics planning
   - Energy consumption prediction for smart buildings
   - Weather impact modeling for agricultural planning

INDUSTRY VERTICALS AND CLIENT BASE:
We serve clients across multiple industries with customized AI solutions:

Healthcare Sector:
- Diagnostic assistance systems for radiologists and pathologists
- Drug discovery acceleration using machine learning algorithms
- Patient outcome prediction models for treatment optimization
- Medical record digitization and automated analysis systems
- Telemedicine platform integration and remote monitoring
- Clinical trial optimization and patient recruitment systems

Financial Services:
- Risk assessment platforms for loan approvals and credit scoring
- Algorithmic trading systems with advanced market analysis
- Fraud detection and prevention for payment processors
- Regulatory compliance monitoring and automated reporting
- Customer service automation for banking operations
- Investment portfolio optimization using quantitative algorithms

Retail and E-commerce:
- Demand forecasting and inventory management optimization
- Personalized recommendation engines for online platforms
- Dynamic pricing algorithms for revenue optimization
- Customer behavior analysis and market segmentation
- Supply chain visibility and logistics coordination
- Visual search and product discovery technologies

Manufacturing and Industrial:
- Computer vision quality control systems for production lines
- Predictive maintenance algorithms for equipment monitoring
- Process optimization for increased efficiency and waste reduction
- Safety monitoring systems using IoT sensors and AI analytics
- Energy consumption optimization for sustainable operations
- Robotic automation and intelligent manufacturing systems

TEAM AND ORGANIZATIONAL STRUCTURE:
Our diverse team consists of world-class talent:
- 150+ engineers and data scientists with advanced degrees
- 50+ product managers, designers, and business development professionals
- 30+ sales, marketing, and customer success specialists
- 20+ operations, legal, and administrative support staff

We maintain offices in strategic locations:
- San Francisco, California (Headquarters) - 120 employees
- New York, New York (East Coast Operations) - 60 employees
- London, United Kingdom (European Operations) - 40 employees
- Toronto, Canada (AI Research Laboratory) - 30 employees

Our company culture emphasizes innovation, ethical AI development, diversity and inclusion, work-life balance, open source contributions, and environmental sustainability.

TECHNOLOGY STACK AND INFRASTRUCTURE:
Machine Learning Development:
- Python ecosystem with TensorFlow, PyTorch, and scikit-learn
- CUDA and distributed computing for GPU acceleration
- MLflow for experiment tracking and model management
- Apache Airflow for ML pipeline orchestration
- Jupyter notebooks and collaborative development environments

Cloud Infrastructure:
- Multi-cloud architecture using AWS, Google Cloud, and Azure
- Kubernetes for container orchestration and microservices
- Docker for containerization and application packaging
- Terraform for infrastructure as code automation
- Jenkins and GitLab CI/CD for continuous deployment

Data Engineering:
- Apache Kafka for real-time data streaming
- Apache Spark for large-scale data processing
- Elasticsearch for search and analytics
- PostgreSQL, MongoDB, and Redis databases
- Apache Hadoop for distributed storage

Security and Compliance:
- Zero-trust security architecture with end-to-end encryption
- Multi-factor authentication and role-based access control
- GDPR, HIPAA, and SOC2 compliance frameworks
- Regular security audits and penetration testing
- Privacy-preserving machine learning techniques

MISSION AND VALUES:
Our mission is to democratize artificial intelligence and make advanced machine learning capabilities accessible to businesses of all sizes. We believe AI should augment human capabilities and are committed to developing ethical, explainable, and trustworthy systems that solve real-world problems while considering societal impact.
"""

# TODO: add simple example for manual cache point insertion


async def demo_manual_cache_points() -> None:
    """Demonstrate manual cache point insertion."""
    print('=== Manual Cache Points Demo ===')

    bedrock_model = BedrockConverseModel(
        'eu.anthropic.claude-sonnet-4-20250514-v1:0',
        provider=BedrockProvider(profile_name='co2-s3'),
    )

    _amazon_model = BedrockConverseModel(
        'eu.amazon.nova-pro-v1:0',
        provider=BedrockProvider(profile_name='co2-s3'),
    )

    # anthropic_model = AnthropicModel(
    #     'claude-sonnet-4-20250514',
    # )

    agent = Agent(
        model=bedrock_model,
        system_prompt='You are a helpful assistant with access to company information.',
    )

    print('Running first query with cache point...')
    # First query with cache point - this should cache the long context
    result1 = await agent.run(
        [
            LONG_CONTEXT,
            CachePoint(),  # Cache everything above this point
            'Read this, and then I will ask you a question.',
        ]
    )

    print(f'Response: {result1.output}')
    result_1_usage = result1.usage()
    if result_1_usage:
        print(f'Usage: {result_1_usage}')
        if result_1_usage.cache_write_tokens:
            print(f'Cache write tokens: {result_1_usage.cache_write_tokens}')
        if result_1_usage.cache_read_tokens:
            print(f'Cache read tokens: {result_1_usage.cache_read_tokens}')

    print('\nRunning second query (should use cache)...')
    # Second query with same cached context
    result2 = await agent.run(
        [
            LONG_CONTEXT,
            CachePoint(),
            'What technology stack does the company use?',
        ]
    )

    print(f'Response: {result2.output}')
    result_2_usage = result2.usage()
    if result_2_usage:
        print(f'Usage: {result_2_usage}')
        if result_2_usage.cache_write_tokens:
            print(f'Cache write tokens: {result_2_usage.cache_write_tokens}')
        if result_2_usage.cache_read_tokens:
            print(f'Cache read tokens: {result_2_usage.cache_read_tokens}')

        # Calculate potential savings
        if result_2_usage.cache_read_tokens and result_2_usage.input_tokens:
            cache_percentage = (
                result_2_usage.cache_read_tokens / result_2_usage.input_tokens
            ) * 100
            print(f'Cache hit rate: {cache_percentage:.1f}% of input tokens')
            # Cached tokens typically cost ~10% of normal tokens
            savings = result_2_usage.cache_read_tokens * 0.9
            print(f'Estimated savings: ~{savings:.0f} token-equivalents')


# TODO(larryhudson): For this to work in a long thread, you also need to add a processor to remove the cache points from the non-last messages
def cache_long_tool_returns(
    min_tokens: int = 1024,
) -> Callable[[list[ModelMessage]], list[ModelMessage]]:
    """Add cache points after long tool results.

    This processor only examines the last ModelRequest in the message history
    and automatically adds cache points after UserPromptPart content that
    likely came from ToolReturn.content when the content exceeds the token threshold.

    Args:
        min_tokens: Minimum estimated tokens before adding a cache point

    Returns:
        A processor function that adds cache points after long tool results
    """

    def processor(messages: list[ModelMessage]) -> list[ModelMessage]:
        if not messages:
            return messages

        last_message = messages[-1]
        if not isinstance(last_message, ModelRequest):
            return messages

        tool_return_part = next(
            (part for part in last_message.parts if isinstance(part, ToolReturnPart)),
            None,
        )

        if not tool_return_part:
            return messages

        if len(tool_return_part.model_response_str()) > min_tokens * 4:
            parts_list = list(last_message.parts)
            parts_list.append(UserPromptPart(content=[CachePoint()]))
            last_message.parts = parts_list

        return messages

    return processor


async def demo_tool_result_caching() -> None:
    """Demonstrate caching of long tool results using ToolReturn.content with CachePoint."""
    print('\n=== Tool Result Caching Demo ===')

    bedrock_model = BedrockConverseModel(
        'eu.anthropic.claude-sonnet-4-20250514-v1:0',
        provider=BedrockProvider(profile_name='co2-s3'),
    )

    # _anthropic_model = AnthropicModel(
    #     'claude-sonnet-4-20250514',
    # )

    _amazon_model = BedrockConverseModel(
        'eu.amazon.nova-pro-v1:0',
        provider=BedrockProvider(profile_name='co2-s3'),
    )

    agent = Agent(
        model=bedrock_model,
        system_prompt='You are a helpful assistant that processes large datasets and documents. When I ask you to use multiple tools, call them all in sequence.',
        history_processors=[cache_long_tool_returns(min_tokens=1024)],
    )

    @agent.tool
    def fetch_api_documentation(ctx: RunContext) -> str:
        """Fetch comprehensive API documentation."""
        # Simulate fetching large API documentation
        api_docs = (
            f"""
API Reference Documentation - MyService v2.0

{'=' * 60}
AUTHENTICATION
{'=' * 60}
All endpoints require Bearer token authentication.
Header: Authorization: Bearer <your-token>

{'=' * 60}
ENDPOINTS
{'=' * 60}
"""
            + 'Detailed endpoint documentation with examples... ' * 150
        )

        return api_docs

    @agent.tool
    def fetch_user_guide(ctx: RunContext) -> str:
        """Fetch comprehensive user guide that references API docs."""
        # Simulate fetching large user guide
        user_guide = (
            f"""
User Guide - MyService Integration

{'=' * 60}
GETTING STARTED
{'=' * 60}
This guide assumes you have read the API documentation above.

{'=' * 60}
STEP-BY-STEP TUTORIALS
{'=' * 60}
"""
            + 'Detailed step-by-step tutorials and examples... ' * 120
        )

        return user_guide

    @agent.tool
    def generate_code_examples(ctx: RunContext) -> str:
        """Generate code examples that reference both cached docs."""
        # Simulate generating code examples
        code_examples = (
            f"""
Code Examples - MyService Integration

{'=' * 60}
BASIC AUTHENTICATION EXAMPLE
{'=' * 60}
Based on the API documentation and user guide above:

import requests

# Use the authentication method described in the API docs
headers = {{'Authorization': 'Bearer your-token'}}
response = requests.get('https://api.myservice.com/data', headers=headers)

{'=' * 60}
ADVANCED EXAMPLES
{'=' * 60}
"""
            + 'More detailed code examples and best practices... ' * 80
        )

        return code_examples

    print('Demo: Request that triggers multiple tool calls with automatic caching...')
    print(
        'The history processor will automatically add cache points after long tool results.'
    )
    print('Expected: cache write → cache read + write → cache read + write')

    result = await agent.run(
        'Please help me integrate with MyService by: 1) fetching the API documentation, '
        + 'then stop and think before get the user guide, then read the user guide and then finally generating code examples. Use all three tools in sequence, one at a time.'
    )

    print(f'Response: {result.output[:200]}...')

    usage = result.usage()
    if usage:
        print(f'\nUsage Summary: {usage}')
        if usage.cache_write_tokens:
            print(
                f'✅ Cache writes: {usage.cache_write_tokens} tokens (automatically cached tool results)'
            )
        if usage.cache_read_tokens:
            print(
                f'✅ Cache reads: {usage.cache_read_tokens} tokens (reusing previous tool results)'
            )
        if usage.cache_read_tokens and usage.input_tokens:
            cache_percentage = (usage.cache_read_tokens / usage.input_tokens) * 100
            print(f'Cache efficiency: {cache_percentage:.1f}% of input was cached')
            savings = usage.cache_read_tokens * 0.9  # Cached tokens cost ~10% of normal
            print(f'Estimated savings: ~{savings:.0f} token-equivalents')

    print('\n🎯 How automatic tool result caching works:')
    print('1. Tools return large content via ToolReturn.content (no manual CachePoint)')
    print('2. History processor detects long content and automatically adds CachePoint')
    print('3. First tool → cache write only (new content)')
    print('4. Second tool → cache read (previous results) + cache write (new content)')
    print('5. Third tool → cache read (all previous) + cache write (new content)')
    print('6. Result shows both cache_read_tokens and cache_write_tokens')


async def demo_system_prompt_caching() -> None:
    """Demonstrate system prompt caching."""
    print('\n=== System Prompt Caching Demo ===')

    bedrock_model = BedrockConverseModel(
        'eu.anthropic.claude-sonnet-4-20250514-v1:0',
        provider=BedrockProvider(profile_name='co2-s3'),
    )

    agent = Agent(model=bedrock_model, system_prompt=LONG_CONTEXT)

    print('Running first financial query...')
    result1 = await agent.run(
        [
            # By adding a CachePoint at the start of the first user prompt part, the system prompt will be cached.
            CachePoint(),
            "What are the key ratios to analyze when evaluating a tech company's financial health?",
        ]
    )
    print(f'Response length: {len(result1.output)} characters')
    result_1_usage = result1.usage()
    if result_1_usage:
        print(f'Usage: {result_1_usage}')

    print('\nRunning second financial query...')
    result2 = await agent.run(
        'How should I approach valuing a SaaS startup with recurring revenue?'
    )
    print(f'Response length: {len(result2.output)} characters')
    result_2_usage = result2.usage()
    if result_2_usage:
        print(f'Usage: {result_2_usage}')
        if result_2_usage.cache_read_tokens:
            print(f'Cache read tokens: {result_2_usage.cache_read_tokens}')


# TODO(larryhudson): Make this more minimal and clear
async def main() -> None:
    """Main function to run all demos."""
    print('Prompt Caching Demo with AWS Bedrock')
    print('====================================')

    # Check if AWS credentials are available
    if not any(
        key in os.environ for key in ['AWS_ACCESS_KEY_ID', 'AWS_PROFILE']
    ) and not os.path.exists(os.path.expanduser('~/.aws/credentials')):
        print('⚠️  Warning: No AWS credentials found!')
        print('Please configure AWS credentials using one of:')
        print('1. AWS CLI: aws configure')
        print('2. Environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY')
        print('3. IAM role (if running on EC2)')
        return

    try:
        await demo_manual_cache_points()
        await demo_tool_result_caching()
        await demo_system_prompt_caching()

        print('\n✅ Demo completed successfully!')
        print('\nKey takeaways:')
        print('- Cache points reduce costs by up to 90% for repeated context')
        print('- Cache read tokens appear in usage metrics')
        print('- Same context + CachePoint = cache hit')
        print('- Great for long system prompts, documents, conversation history')
        print('- ToolReturn.content supports CachePoint for caching tool results')
        print('- Message processors can automatically add cache points to tool results')

    except Exception as e:
        print(f'❌ Error running demo: {e}')
        print('\nTroubleshooting:')
        print('1. Check AWS credentials and region configuration')
        print('2. Verify access to Claude 3.5 Sonnet on Bedrock')
        print('3. Ensure your AWS account has Bedrock permissions')
        raise e


if __name__ == '__main__':
    asyncio.run(main())
