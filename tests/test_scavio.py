"""Tests for the Scavio search tool."""

from __future__ import annotations

import pytest
from inline_snapshot import snapshot
from scavio import AsyncScavioClient

from pydantic_ai._run_context import RunContext
from pydantic_ai.common_tools.scavio import ScavioSearchTool, scavio_search_tool
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage


@pytest.mark.vcr()
async def test_basic_search(scavio_api_key: str):
    """Test basic search with default parameters."""
    tool = ScavioSearchTool(client=AsyncScavioClient(api_key=scavio_api_key))
    results = await tool('What is Pydantic AI?')
    assert results == snapshot(
        [
            {
                'position': 1,
                'title': 'Pydantic AI | Pydantic Docs',
                'url': 'https://pydantic.dev/docs/ai/overview/',
                'domain': 'pydantic.dev',
                'content': 'Pydantic AI is a Python agent framework designed to help you quickly, confidently, and painlessly build production grade applications and workflows with ...Read more',
                'date': None,
            },
            {
                'position': 2,
                'title': 'Pydantic | The end-to-end AI engineering stack',
                'url': 'https://pydantic.dev/',
                'domain': 'pydantic.dev',
                'content': '3 days ago -- Pydantic is an end-to-end AI engineering stack, focused on developer experience. Build in Python, TypeScript, Rust, and Go. Monitor on the ...',
                'date': None,
            },
            {
                'position': 3,
                'title': 'AI Agent Framework, the Pydantic way',
                'url': 'https://github.com/pydantic/pydantic-ai',
                'domain': 'github.com',
                'content': 'Pydantic AI is a Python agent framework designed to help you quickly, confidently, and painlessly build production grade applications and workflows with ...Read more',
                'date': None,
            },
            {
                'position': 4,
                'title': 'Pydantic AI : r/LLMDevs',
                'url': 'https://www.reddit.com/r/LLMDevs/comments/1iih8az/pydantic_ai/',
                'domain': 'www.reddit.com',
                'content': 'Pydantic AI to build some basic agents and multi agents how to build powerful python tools. Pydantic ai is free,except',
                'date': None,
            },
            {
                'position': 5,
                'title': 'Pydantic AI: Agent Framework',
                'url': 'https://medium.com/ai-agent-insider/pydantic-ai-agent-framework-02b138e8db71',
                'domain': 'medium.com',
                'content': 'PydanticAI is a Python Agent Framework designed to make building production-grade applications with Generative AI less painful.',
                'date': None,
            },
            {
                'position': 6,
                'title': "Pydantic AI: A Beginner's Guide With Practical Examples",
                'url': 'https://www.datacamp.com/tutorial/pydantic-ai-guide',
                'domain': 'www.datacamp.com',
                'content': 'Sep 3, 2025 -- Pydantic AI is a Python agent framework that brings structure and type safety to LLM applications.',
                'date': 'Sep 3, 2025',
            },
            {
                'position': 7,
                'title': 'Type-safe LLM agents with PydanticAI',
                'url': 'https://simmering.dev/blog/pydantic-ai/',
                'domain': 'simmering.dev',
                'content': 'Dec 16, 2024 -- Pydantic AI is a new agent framework by the company behind Pydantic, the popular data validation library. developers define workflows wherein ...',
                'date': 'Dec 16, 2024',
            },
            {
                'position': 8,
                'title': 'Pydantic AI: Type-Safe Python Framework for AI Agents & ...',
                'url': 'https://pydantic.dev/pydantic-ai',
                'domain': 'pydantic.dev',
                'content': 'Build production-grade AI applications with Pydantic AI - a model-agnostic Python framework featuring type safety, structured outputs, validation, ...',
                'date': None,
            },
        ]
    )


@pytest.mark.vcr()
async def test_factory_with_bound_params(scavio_api_key: str):
    """Test factory-bound params are forwarded through FunctionSchema.call and hidden from the schema."""
    tool = scavio_search_tool(scavio_api_key, country_code='us', language='en')
    # Developer-fixed params must not appear in the LLM tool schema.
    assert 'query' in tool.function_schema.json_schema['properties']
    assert 'country_code' not in tool.function_schema.json_schema['properties']
    assert 'language' not in tool.function_schema.json_schema['properties']

    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())
    results = await tool.function_schema.call({'query': 'attention mechanisms'}, ctx)
    assert results == snapshot(
        [
            {
                'position': 1,
                'title': 'Attention (machine learning)',
                'url': 'https://en.wikipedia.org/wiki/Attention_(machine_learning)',
                'domain': 'en.wikipedia.org',
                'content': 'In machine learning, attention is a method that determines the importance of each component in a sequence relative to the other components in that sequence.Read more',
                'date': None,
            },
            {
                'position': 2,
                'title': 'What is an attention mechanism?',
                'url': 'https://www.ibm.com/think/topics/attention-mechanism',
                'domain': 'www.ibm.com',
                'content': 'An attention mechanism is a machine learning technique that directs deep learning models to prioritize (or attend to) the most relevant parts of input data.Read more',
                'date': None,
            },
            {
                'position': 3,
                'title': 'Attention Mechanism in ML',
                'url': 'https://www.geeksforgeeks.org/artificial-intelligence/ml-attention-mechanism/',
                'domain': 'www.geeksforgeeks.org',
                'content': 'May 11, 2026 -- The attention mechanism allows models to focus on the most important parts of input data by assigning different weights to different elements.',
                'date': 'May 11, 2026',
            },
            {
                'position': 4,
                'title': '11. Attention Mechanisms and Transformers',
                'url': 'http://www.d2l.ai/chapter_attention-mechanisms-and-transformers/index.html',
                'domain': 'www.d2l.ai',
                'content': 'The core idea behind the Transformer model is the attention mechanism, an innovation that was originally envisioned as an enhancement for encoder-decoder RNNs.Read more',
                'date': None,
            },
            {
                'position': 5,
                'title': 'Attention Mechanisms in Deep Learning: Enhancing Model ...',
                'url': 'https://medium.com/@zhonghong9998/attention-mechanisms-in-deep-learning-enhancing-model-performance-32a91006092a',
                'domain': 'medium.com',
                'content': 'Attention mechanisms enable neural networks to mimic human-like selective focus, improving their ability to process and understand complex data.Read more',
                'date': None,
            },
            {
                'position': 6,
                'title': '[D] How to truly understand attention mechanism in ...',
                'url': 'https://www.reddit.com/r/MachineLearning/comments/qidpqx/d_how_to_truly_understand_attention_mechanism_in/',
                'domain': 'www.reddit.com',
                'content': 'Attention seems to be a core concept for language modeling these days. However it is not that easy to fully understand, and in my opinion, somewhat unintuitive. ...',
                'date': None,
            },
            {
                'position': 7,
                'title': 'Coding Self-Attention From Scratch',
                'url': 'https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html',
                'domain': 'sebastianraschka.com',
                'content': 'Feb 9, 2023 -- In this article, we are going to understand how self-attention works from scratch. This means we will code it ourselves one step at a time.Read more',
                'date': 'Feb 9, 2023',
            },
        ]
    )
