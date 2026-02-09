"""Tests for the Tavily search tool."""

from __future__ import annotations

import pytest
from inline_snapshot import snapshot

from pydantic_ai._run_context import RunContext
from pydantic_ai.usage import RunUsage

from .conftest import try_import

with try_import() as imports_successful:
    from tavily import AsyncTavilyClient

    from pydantic_ai.common_tools.tavily import TavilySearchTool, tavily_search_tool

from pydantic_ai.models.test import TestModel

pytestmark = pytest.mark.skipif(not imports_successful(), reason='tavily-python not installed')


@pytest.mark.vcr()
async def test_basic_search(tavily_api_key: str):
    """Test basic search with default parameters."""
    tool = TavilySearchTool(client=AsyncTavilyClient(tavily_api_key))
    results = await tool('What is Pydantic AI?')
    assert len(results) > 0
    assert results == snapshot(
        [
            {
                'title': 'Pydantic AI: Agent Framework',
                'url': 'https://medium.com/ai-agent-insider/pydantic-ai-agent-framework-02b138e8db71',
                'content': '## AI Agent Insider. # Pydantic AI: Agent Framework. Introducing **Pydantic AI**-- a groundbreaking Python framework specifically designed to simplify the creation of production-grade AI agents. **Pydantic AI** is a Python framework that acts as a bridge between developers and LLMs, providing tools to create **agents** -- entities that execute specific tasks based on system prompts, functions, and structured outputs. Here\'s a basic example of using Pydantic AI to create an agent that responds to user queries:. from pydantic_ai import Agentagent = Agent("openai:gpt-4", system_prompt="Be a helpful assistant.")result = await agent.run("Hello, how are you?")print(result.data) # Outputs the response. from pydantic_ai import ModelRetry@agent.tooldef validate_data(ctx): if not ctx.input_data: raise ModelRetry("Data missing, retrying..."). Pydantic AI is transforming how developers build AI agents. Install Pydantic AI today and build your first agent!**. ### Agent Framework / shim to use Pydantic with LLMs. Contribute to pydantic/pydantic-ai development by creating an account.... ## GitHub - pydantic/pydantic-ai: Agent Framework / shim to use Pydantic with LLMs. ## Published in AI Agent Insider.',
                'score': 0.9999875,
            },
            {
                'title': 'Pydantic AI - Pydantic AI',
                'url': 'https://ai.pydantic.dev/',
                'content': 'Pydantic AI is a Python agent framework designed to help you quickly, confidently, and painlessly build production grade applications and workflows with',
                'score': 0.99997807,
            },
            {
                'title': 'Build Production-Ready AI Agents in Python with Pydantic AI',
                'url': 'https://www.youtube.com/watch?v=-WB0T0XmDrY',
                'content': 'Pydantic AI lets you integrate large language models like GPT-5 **directly into your Python applications**.',
                'score': 0.9999398,
            },
            {
                'title': 'What is Pydantic AI?. Build production-ready AI agents with...',
                'url': 'https://medium.com/@tahirbalarabe2/what-is-pydantic-ai-15cc81dea3c3',
                'content': '*"Pydantic AI is a Python agent framework designed to help you quickly, confidently, and painlessly build production grade applications and workflows with Generative AI."*. So they built Pydantic AI with a single, simple aim, to bring that FastAPI feeling to building applications and workflows with generative AI. A: Pydantic AI is a Python agent framework from the creators of Pydantic Validation. Q: How is Pydantic AI different from other agent frameworks like LangChain or LlamaIndex? Q: What LLM providers does Pydantic AI support? Q: Can I use my own custom models with Pydantic AI? Q: Can I use Pydantic AI with existing FastAPI applications? A: Yes, Pydantic AI is designed to integrate well with FastAPI and other Python web frameworks. A: Pydantic AI uses Pydantic models to define structured output types, ensuring LLM responses are validated and type-safe. A: Yes, Pydantic AI is designed specifically for production-grade applications with features like durable execution, observability integration, and type safety.',
                'score': 0.9999125,
            },
            {
                'title': 'Pydantic AI : r/LLMDevs',
                'url': 'https://www.reddit.com/r/LLMDevs/comments/1iih8az/pydantic_ai/',
                'content': "I've been using Pydantic AI to build some basic agents and multi agents and it seems quite straight forward and I'm quite pleased with it.",
                'score': 0.9999001,
            },
        ]
    )


@pytest.mark.vcr()
async def test_search_with_include_domains(tavily_api_key: str):
    """Test search with include_domains filtering."""
    tool = TavilySearchTool(client=AsyncTavilyClient(tavily_api_key))
    results = await tool('transformer architectures', include_domains=['arxiv.org'])
    assert len(results) > 0
    assert results == snapshot(
        [
            {
                'title': 'Deep Dive into Transformer Architectures for Long-Term ...',
                'url': 'https://arxiv.org/abs/2507.13043',
                'content': "# Title:The Power of Architecture: Deep Dive into Transformer Architectures for Long-Term Time Series Forecasting. Authors:Lefei Shen, Mouxiang Chen, Han Fu, Xiaoxue Ren, Xiaoyun Joy Wang, Jianling Sun, Zhuo Li, Chenghao Liu. View a PDF of the paper titled The Power of Architecture: Deep Dive into Transformer Architectures for Long-Term Time Series Forecasting, by Lefei Shen and 7 other authors. View a PDF of the paper titled The Power of Architecture: Deep Dive into Transformer Architectures for Long-Term Time Series Forecasting, by Lefei Shen and 7 other authors. > Abstract:Transformer-based models have recently become dominant in Long-term Time Series Forecasting (LTSF), yet the variations in their architecture, such as encoder-only, encoder-decoder, and decoder-only designs, raise a crucial question: What Transformer architecture works best for LTSF tasks? | Cite as: | arXiv:2507.13043 [cs.LG] |. |  | (or  arXiv:2507.13043v1 [cs.LG] for this version) |. # Bibliographic and Citation Tools. Have an idea for a project that will add value for arXiv's community?",
                'score': 0.7923522,
            },
            {
                'title': '[2505.13499] Optimal Control for Transformer Architectures',
                'url': 'https://arxiv.org/abs/2505.13499',
                'content': "# Computer Science > Machine Learning. # Title:Optimal Control for Transformer Architectures: Enhancing Generalization, Robustness and Efficiency. | Subjects: | Machine Learning (cs.LG); Artificial Intelligence (cs.AI); Optimization and Control (math.OC) |. | Cite as: | arXiv:2505.13499 [cs.LG] |. |  | (or  arXiv:2505.13499v2 [cs.LG] for this version) |. |  |  Focus to learn more  arXiv-issued DOI via DataCite |. ### References & Citations. ## BibTeX formatted citation. # Bibliographic and Citation Tools. # Code, Data and Media Associated with this Article. # Recommenders and Search Tools. # arXivLabs: experimental projects with community collaborators. arXivLabs is a framework that allows collaborators to develop and share new arXiv features directly on our website. Both individuals and organizations that work with arXivLabs have embraced and accepted our values of openness, community, excellence, and user data privacy. arXiv is committed to these values and only works with partners that adhere to them. Have an idea for a project that will add value for arXiv's community?",
                'score': 0.783542,
            },
            {
                'title': 'Deep Dive into Transformer Architectures for Long-Term ...',
                'url': 'https://arxiv.org/html/2507.13043v1',
                'content': 'Transformer-based models have recently become dominant in Long-term Time Series Forecasting (LTSF), yet the variations in their architecture, such as encoder-only, encoder-decoder, and decoder-only designs, raise a crucial question: What Transformer architecture works best for LTSF tasks? In recent years, Transformer-based models have become dominant in long-term time series forecasting (LTSF) tasks (Informer, ; Autoformer, ; FEDformer, ; PatchTST, ; iTransformer, ; TimeXer, ; ARMA\\_Attention, ; Pyraformer, ; TFT, ; PDFormer, ; BasisFormer, ; SAMformer, ; Scaleformer, ; Quatformer, ), demonstrating strong performance across various real-world applications (TSF\\_Energy\\_1, ; TSF\\_Energy\\_2, ; TSF\\_Economics\\_1, ; TSF\\_Web\\_1, ; TSF\\_Web\\_2, ; TSF\\_Web\\_3, ; TSF\\_Weather\\_1, ; TSF\\_Weather\\_2, ; TSF\\_Finance\\_1, ). We examine Transformer-based LTSF models from multiple perspectives, including attention mechanisms, forecasting aggregation strategies, forecasting paradigms, and normalization layers. Based on the above conclusions, we construct an optimal Transformer architecture by combining the best choices, including bi-directional attention with joint-attention, complete forecasting aggregation, direct-mapping paradigm, and the BatchNorm layer.',
                'score': 0.77731717,
            },
            {
                'title': 'Lightweight Transformer Architectures for Edge Devices in ...',
                'url': 'https://www.arxiv.org/abs/2601.03290',
                'content': '# Title:Lightweight Transformer Architectures for Edge Devices in Real-Time Applications. View a PDF of the paper titled Lightweight Transformer Architectures for Edge Devices in Real-Time Applications, by Hema Hariharan Samson. > Abstract:The deployment of transformer-based models on resource-constrained edge devices represents a critical challenge in enabling real-time artificial intelligence applications. This comprehensive survey examines lightweight transformer architectures specifically designed for edge deployment, analyzing recent advances in model compression, quantization, pruning, and knowledge distillation techniques. Experimental results demonstrate that modern lightweight transformers can achieve 75-96% of full-model accuracy while reducing model size by 4-10x and inference latency by 3-9x, enabling deployment on devices with as little as 2-5W power consumption. Comprehensive study of lightweight transformer architectures for edge computing with novel findings on memory-bandwidth tradeoffs, quantization strategies, and hardware-specific optimizations. |  | (or  arXiv:2601.03290v1 [cs.LG] for this version) |. View a PDF of the paper titled Lightweight Transformer Architectures for Edge Devices in Real-Time Applications, by Hema Hariharan Samson.',
                'score': 0.76826453,
            },
            {
                'title': 'Study of Lightweight Transformer Architectures for Single ...',
                'url': 'https://arxiv.org/abs/2505.21057',
                'content': "# Title:Study of Lightweight Transformer Architectures for Single-Channel Speech Enhancement. View a PDF of the paper titled Study of Lightweight Transformer Architectures for Single-Channel Speech Enhancement, by Haixin Zhao and Nilesh Madhu. Networks integrating stacked temporal and spectral modelling effectively leverage improved architectures such as transformers; however, they inevitably incur substantial computational complexity and model expansion. The proposed lightweight, causal, transformer-based architecture with adversarial training (LCT-GAN) yields SoTA performance on instrumental metrics among contemporary lightweight models, but with far less overhead. | Cite as: | arXiv:2505.21057 [eess.AS] |. |  | (or  arXiv:2505.21057v1 [eess.AS] for this version) |. View a PDF of the paper titled Study of Lightweight Transformer Architectures for Single-Channel Speech Enhancement, by Haixin Zhao and Nilesh Madhu. # Bibliographic and Citation Tools. arXiv is committed to these values and only works with partners that adhere to them. Have an idea for a project that will add value for arXiv's community?",
                'score': 0.7663815,
            },
        ]
    )


@pytest.mark.vcr()
async def test_factory_with_bound_params(tavily_api_key: str):
    """Test factory-bound params are forwarded through FunctionSchema.call."""
    tool = tavily_search_tool(tavily_api_key, max_results=2, include_domains=['arxiv.org'])
    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())
    results = await tool.function_schema.call({'query': 'attention mechanisms'}, ctx)
    assert len(results) <= 2
    assert results == snapshot(
        [
            {
                'title': '[2601.03329] Attention mechanisms in neural networks',
                'url': 'https://arxiv.org/abs/2601.03329',
                'content': 'by H Hays · 2026 · Cited by 1 -- Attention mechanisms represent a fundamental paradigm shift in neural network architectures, enabling models to selectively focus on relevant',
                'score': 0.81770587,
            },
            {
                'title': 'A General Survey on Attention Mechanisms in Deep ...',
                'url': 'https://arxiv.org/abs/2203.14263',
                'content': "# Title:A General Survey on Attention Mechanisms in Deep Learning. View a PDF of the paper titled A General Survey on Attention Mechanisms in Deep Learning, by Gianni Brauwers and Flavius Frasincar. > Abstract:Attention is an important mechanism that can be employed for a variety of deep learning models across many different domains and tasks. The various attention mechanisms are explained by means of a framework consisting of a general attention model, uniform notation, and a comprehensive taxonomy of attention mechanisms. | Subjects: | Machine Learning (cs.LG) |. | Cite as: | arXiv:2203.14263 [cs.LG] |. |  | (or  arXiv:2203.14263v1 [cs.LG] for this version) |. View a PDF of the paper titled A General Survey on Attention Mechanisms in Deep Learning, by Gianni Brauwers and Flavius Frasincar. ### References & Citations. # Bibliographic and Citation Tools. # Recommenders and Search Tools. Have an idea for a project that will add value for arXiv's community?",
                'score': 0.8138313,
            },
        ]
    )


class TestTavilySearchToolFactory:
    """Schema-level tests for tavily_search_tool factory."""

    def test_no_params_bound_exposes_all_in_schema(self, tavily_api_key: str):
        """Test that with no factory params, all parameters appear in the tool schema."""
        tool = tavily_search_tool(tavily_api_key)

        assert tool.name == 'tavily_search'
        schema_props = tool.function_schema.json_schema['properties']
        assert 'max_results' in schema_props
        assert 'include_domains' in schema_props
        assert 'exclude_domains' in schema_props

    def test_bound_params_hidden_from_schema(self, tavily_api_key: str):
        """Test that factory-provided params are excluded from the tool schema."""
        tool = tavily_search_tool(
            tavily_api_key,
            search_deep='advanced',
            topic='news',
            time_range='week',
            max_results=5,
            include_domains=['arxiv.org'],
            exclude_domains=['medium.com'],
        )

        schema_props = tool.function_schema.json_schema['properties']
        assert 'search_deep' not in schema_props
        assert 'topic' not in schema_props
        assert 'time_range' not in schema_props
        assert 'max_results' not in schema_props
        assert 'include_domains' not in schema_props
        assert 'exclude_domains' not in schema_props
        # query should always be visible
        assert 'query' in schema_props
