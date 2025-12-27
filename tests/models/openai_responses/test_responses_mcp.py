"""Tests for MCP server tool integration with OpenAI Responses model."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest
from inline_snapshot import snapshot

from pydantic_ai import (
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    ThinkingPart,
    ToolCallPartDelta,
    UserPromptPart,
)
from pydantic_ai.agent import Agent
from pydantic_ai.builtin_tools import MCPServerTool
from pydantic_ai.usage import RequestUsage

from ...conftest import IsDatetime, IsNow, IsStr, try_import

with try_import() as imports_successful:
    from pydantic_ai.models.openai import (
        OpenAIResponsesModel,
    )
    from pydantic_ai.providers.openai import OpenAIProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]


async def test_openai_responses_model_mcp_server_tool(allow_model_requests: None, openai_api_key: str):
    m = OpenAIResponsesModel(
        'o4-mini',
        provider=OpenAIProvider(api_key=openai_api_key),
    )
    agent = Agent(
        m,
        instructions='You are a helpful assistant.',
        builtin_tools=[
            MCPServerTool(
                id='deepwiki',
                url='https://mcp.deepwiki.com/mcp',
                description='DeepWiki MCP server',
                allowed_tools=['ask_question'],
                headers={'custom-header-key': 'custom-header-value'},
            ),
        ],
    )

    result = await agent.run('Can you tell me more about the pydantic/pydantic-ai repo? Keep your answer short')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Can you tell me more about the pydantic/pydantic-ai repo? Keep your answer short',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                instructions='You are a helpful assistant.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(
                        tool_name='mcp_server:deepwiki',
                        args={'action': 'list_tools'},
                        tool_call_id='mcpl_0083938b3a28070e0068fabd81d51081a09d4b183ced693273',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='mcp_server:deepwiki',
                        content={
                            'tools': [
                                {
                                    'input_schema': {
                                        'type': 'object',
                                        'properties': {
                                            'repoName': {
                                                'type': 'string',
                                                'description': 'GitHub repository: owner/repo (e.g. "facebook/react")',
                                            },
                                            'question': {
                                                'type': 'string',
                                                'description': 'The question to ask about the repository',
                                            },
                                        },
                                        'required': ['repoName', 'question'],
                                        'additionalProperties': False,
                                        '$schema': 'http://json-schema.org/draft-07/schema#',
                                    },
                                    'name': 'ask_question',
                                    'annotations': {'read_only': False},
                                    'description': 'Ask any question about a GitHub repository',
                                }
                            ],
                            'error': None,
                        },
                        tool_call_id='mcpl_0083938b3a28070e0068fabd81d51081a09d4b183ced693273',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_0083938b3a28070e0068fabd84727c81a0a52c171d2568a947',
                        signature='gAAAAABo-r2bs6ChS2NtAXH6S8ZWRHzygQvAZrQGsb5ziJKg6dINF9TQnq4llBquiZh-3Ngx2Ha4S-2_TLSbgcsglradULI8c8N2CnilghcqlLE90MXgHWzGfMDbmnRVpTW9iJsOnBn4ferQtNLIsXzfGWq4Ov0Bbvlw_fCm9pQsqOavcJ5Kop2lJ9Xqb__boYMcBCPq3FcNlfC3aia2wZkacS4qKZGqytqQP13EX3q6LwFVnAMIFuwn5XLrh4lFf-S5u8UIw3C6wvVIXEUatY6-awgHHJKXxWUxqRQPJegatMb8KE-QtuKQUfdvEE0ykdHtWqT7nnC3qTY67UaSCCvJ9SdXj-t806GVei9McSUe8riU3viHnfY0R0u9GIXsVnfVthIDRnX7KzpF5ot_CpCrgbCmD9Rj2AAos5pCdSzpc08G5auUuuMZfoiWANADTHHhO2OvflSEpmO8pb-QAYfMoK9exYVQ8Oig-Nj35unupcYy7A2bDCViXzqy32aw9QHmH7rErI4v72beWQxRVdX15Z7VS2c6L1dD7cU18K35CWqlSz9hEX5AcGqEEtIDVu1TdF3m1m2u4ooc4TjYpRecjYoG8Ib-vVKoX5C65a7G1cTbCo8dO0DYKGgM8jM7ZDubxbCcZ22Sxk58f8cer7WxHyp7WRo5-6zvMwMCk8uEY44RJmg-m0Oxl_6qxdr4Md80xZah_6tCCB62agQmYwCrR75_r93xOckQAK0R_37khvQD5gWVlE5Rg-01eUTboiPGqYmIsqWvOkziMGnxgKVw_yUf8swHU1ciWr7O1EdVPHLG7YXlVQTHTE_CX3uOsE2FoZnpS_MgpxGfjb76majV50h7mJ6ySVPF_3NF3RQXx64W08SW4eVFD8JJf0yChqXDmlwu2CDZN1n99xdaE9QbMODNEOmfTQOPhQ9g-4LhstNTKCCxWDh0qiv_dq2qAd0I9Gupoit33xGpb66mndc0nuuNFe8-16iC_KzQtHBNzgasgYK-r83KFVmiYK3Jxvz_2dfdwe0M1q7NLBvbnWc6k9LIf8iDUF6Q1J-cfC7SsncCbROtzIPlKpQwxhP-M09Xy3RVxlH9dcvuk3_qqEAartUQC8ZbuLRbhiq66eE1RvQzdNd2tsoBQ85cdNs57Penio7w9zILUf1JP5O8-zCe5GPC3W3EXTIEvHR-kiuxJvhcsySijpldGmuygRx05ARNOIT7VDCZvF23RfmnRduY1X1FAqb_i_aMStK7iyHr_2ohwOWLuklpyuoG0Y1ulvq1A9-hyCZ0mpvTEF6om2tAZ9_7h8W9ksiOkey0yA-6ze17MCjfnK2XcbqmSMgOngW1PrD81oKoheMnIeJdcWgF2mk8VDqmAwaDTxMxdnXkzK74rA43a4rWk3d2bUts8dAUkuYXTwJwKQw4LfXtu-mwwgJ6BkT_GiBcBJ6ulBuPsNZfpwPuxox6PS6KpzVTQ94cKNqSIIyFCD4xZsEvPALud09-gmAEDHxdnPjqLSi2U8xd0j-6XYKN0JtZ45kwEIRsOrFu-SYLz1OcYFKI5A5P-vYlzGx1WhEnoeUlyooJBhNj6ZBfj9f63SByxm7sgh260vf1t-4OGzVTIUKFluxkI4ubigLZ-g4q4dSwiEWXn50JFPrtuPs5VxsIIz_lXbh1SrKeQ647KdDSAQZFgEfzOOt3el5K97V1x7V7gEWCCgmqDIz3yZPpwD6qmUQKqlj_p8-OQrniamGULkXrmrgbNQVfV-Qw7Hg6ELw4aHF_IZME9Qnyn7peFhH6ai_YapuNF7FK-MBtPYoMaqBf05U2-uJAVUas3VuT_-pTyHvhtFmB7vc0-qgf_CtVNIXSPq2_vXdQdEwwCVPPwW6xWm-invrzhyQR_mf3OQqZT6_zOHIMPBJUaXcQKT0KTdoBZUDamAR-ECZl8r6wdLCn0HjAEwj3ifUCNMzQ7CZHUQG46rj61YyasNWO__4Ef4kTcApKgljosuABqP4HAdmkP5eEnX-6nutrL50iv-Mms_R-T7SKtmEEf9wihTu4Meb441cU9DI4WwSyiBSnsYdGy9FJKmHwP7HD0FmpmWkOrtROkQVMlMVKQFlKK8OBtxafHYsZkWDawbA1eetzMBzQ3PP8PSvva6SJWjbgURHVm5RjXV8Hk6toIBEDx9r9vAIczSp49eDCkQbzPkGAVilO3KLQpNx2itBbZzgE36uV0neZZsVs7aqafI4qCTQOLzYA8YFDKz92yhgdIzl5VPFLFNHqRS4duPRQImQ7vb6yKSxjDThiyQQUTPBX_EXUAAR7JHwJI1i8la3V',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='mcp_server:deepwiki',
                        args={
                            'action': 'call_tool',
                            'tool_name': 'ask_question',
                            'tool_args': {
                                'repoName': 'pydantic/pydantic-ai',
                                'question': 'Provide a brief summary of the repository, including purpose, main features, and status.',
                            },
                        },
                        tool_call_id='mcp_0083938b3a28070e0068fabd88db5c81a08e56f163bbc6088b',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='mcp_server:deepwiki',
                        content={
                            'output': """\
Pydantic AI is a Python agent framework designed to build production-grade applications using Generative AI, emphasizing an ergonomic developer experience and type-safety . It provides type-safe agents, a model-agnostic design supporting over 15 LLM providers, structured outputs with Pydantic validation, comprehensive observability, and production-ready tooling . The project is structured as a UV workspace monorepo, including core framework components, an evaluation system, a graph execution engine, examples, and a CLI tool .

## Purpose <cite/>

The primary purpose of Pydantic AI is to simplify the development of reliable AI applications by offering a robust framework that integrates type-safety and an intuitive developer experience . It aims to provide a unified approach to interacting with various LLM providers and managing complex agent workflows .

## Main Features <cite/>

### Type-Safe Agents <cite/>
Pydantic AI agents are generic `Agent[Deps, Output]` for compile-time validation, utilizing `RunContext[Deps]` for dependency injection and Pydantic `output_type` for output validation  . This ensures that the inputs and outputs of agents are strictly typed and validated .

### Model-Agnostic Design <cite/>
The framework supports over 15 LLM providers through a unified `Model` interface, allowing developers to switch between different models without significant code changes  . Implementations for providers like OpenAI, Anthropic, and Google are available .

### Structured Outputs <cite/>
Pydantic AI leverages Pydantic for automatic validation and self-correction of structured outputs from LLMs . This is crucial for ensuring data integrity and reliability in AI applications .

### Comprehensive Observability <cite/>
The framework includes comprehensive observability features via OpenTelemetry and native Logfire integration . This allows for tracing agent runs, model requests, tool executions, and monitoring token usage and costs  .

### Production-Ready Tooling <cite/>
Pydantic AI offers an evaluation framework, durable execution capabilities, and protocol integrations .
*   **Tool System**: Tools can be registered using the `@agent.tool` decorator, with automatic JSON schema generation from function signatures and docstrings .
*   **Graph Execution**: The `pydantic_graph.Graph` module provides a graph-based state machine for orchestrating agent execution, using nodes like `UserPromptNode`, `ModelRequestNode`, and `CallToolsNode` .
*   **Evaluation Framework**: The `pydantic-evals` package provides tools for creating datasets, running evaluators (e.g., `ExactMatch`, `LLMEvaluator`), and generating reports .
*   **Integrations**: It integrates with various protocols and environments, including Model Context Protocol (MCP) for external tool servers, AG-UI for interactive frontends, and Temporal/DBOS for durable execution .

## Status <cite/>
The project is actively maintained and considered "Production/Stable"  . It supports Python versions 3.10 through 3.13  . The documentation is built using MkDocs and includes API references and examples  .

## Notes <cite/>
The repository is organized as a monorepo using `uv` for package management  . Key packages include `pydantic-ai-slim` (core framework), `pydantic-evals` (evaluation system), `pydantic-graph` (graph execution engine), `examples` (example applications), and `clai` (CLI tool) .

Wiki pages you might want to explore:
- [Overview (pydantic/pydantic-ai)](/wiki/pydantic/pydantic-ai#1)

View this search on DeepWiki: https://deepwiki.com/search/provide-a-brief-summary-of-the_a5712f6e-e928-4886-bcea-b9b75761aac5
""",
                            'error': None,
                        },
                        tool_call_id='mcp_0083938b3a28070e0068fabd88db5c81a08e56f163bbc6088b',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_0083938b3a28070e0068fabd97008081a0ad1b2362bcb153c9',
                        signature='gAAAAABo-r2bD-v0Y3pAlyAEK1Sb8qJJcJRKSRtYwymHwLNXY-SKCqd_Q5RbN0DLCclspuPCAasGLm1WM1Q2Y_3szaEEr_OJalXTVEfRvhCJE1iTgoz2Uyf7KttZ4W92hlYjE8cjgdo5tKtSVkNyzTs4JUHKRHoDMutL2KivjZKuK_4n-lo9paJC_jmz6RWO8wUoXo3_fGxjliOGnWyRXwEPmgAcEWNOSVgCgAEO3vXerXRPLie02HegWcLMtK6WORDHd02Kr86QSK3W30bnvU7glAFX6VhSSnR8G0ceAM-ImoomQ8obEDyedX1-pYDKPOa4pZ5iTjD24ABYOwz-0L7SNziQJLycwwsr11Fj0_Au9yJph8YkNb2nAyFeiNVCRjKul51B7dZgz-UZ9juWO2ffeI0GNtQTYzf46_Y1t0qykGW6w59xjmBHTKf5SiSe0pqWxZ6LOLoPx01rX2gLaKgNZZiERSbO0iwbA4tpxb9ur-qeFVv5tS7xy8KFYOa8SPrypvFWDoY6CjSwTS3ir0vyfpbJy-n6bcYP_pTwDZxy_1aVkciim8Tmm_9wYgI0uY5kcA9VYJuyc4cg7S7ykTUxMZz7xiLMf8FoXl1gHbVJrYriyZzh2poYTWlcCuSCiUaXhQKxcxMRrt_P7WANx0n68ENQ40HkoJ6rThvWUuwtmEYqZ0ldh3XSFtyNrqha4PQ5eg_DudlU_5CxyykuzWmi_o5MEW4_XW4b9vdXg1laqx4189_jEuV_JPGNeL3Ke4EbMbKHzsiaGePRZGgNutnlERagmU4VFTeoE5bN3oHlR_Au4PeQxdb7BuBmZRDDCnnIRd2NfSWb7bgfUozkA4S6rm_089OlRBeRVoLtA8zZZinNGtOZl7MtkLnoJVIWpF1rr7D_47eWSyyegUIIS2e5UKLJfCLkNgSlWPU9VquHEzSfqeHfzoN5ccoVwrvrHmeveTjI-wIJygdfuyti5cMgOOkAtLzjWmbs4CjmlWcbZKeidtDj5YpCSmYAGFuZze-cSbNjMv4th639dCu_jmRMze-l2Y5npbRwMqEJr7VLXghmLc1vhOsaQM3gxoF0CJJlmvtR4jxPqhE3694YRva6LS1WjR4oueM6zfpVeB2kC0hQgqaL6MiwtTRYFfuCzEHi18TwA5bqqkfgrDXedmjAzlEGSZFe2EBRlF_ZtagrVVTCagHQArnH3DkVQMEDCHCqDxA_PINR_997IxeNgGPsvazVdOOBef7sO4rvAWrC94nIlt7d4aViqbTNMW-W8rqjGFOqj1swrM0yoX5y6LY5oXPc3Mu35xeitn_paqtGPkvuH6WeGzAiNZFDoQkUdLkZ4SIH2lr4ZXmMI3nuTzCrwyshwcEu-hhVtGAEQEqVrIn8J75IzYTs1UGLBvhmcpHxCfG04MFNoVf-EPI4SgjNEgV61861TYshxCRrydVhaJmbLqYh8yzLYBHK6oIymv-BrIJ0LX222LwoGbSc0gMTMaudtthlFXrHdnswKf81ubhF7viiD3Y=',
                        provider_name='openai',
                    ),
                    TextPart(
                        content=IsStr(),
                        id='msg_0083938b3a28070e0068fabd989bb481a08c61416ab343ef49',
                    ),
                ],
                usage=RequestUsage(input_tokens=1207, output_tokens=535, details={'reasoning_tokens': 320}),
                model_name='o4-mini-2025-04-16',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 10, 23, 23, 42, 57, tzinfo=timezone.utc),
                },
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    messages = result.all_messages()
    result = await agent.run('What packages does the repo contain?', message_history=messages)
    assert result.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What packages does the repo contain?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                instructions='You are a helpful assistant.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='rs_0083938b3a28070e0068fabd9de42881a08fbb49a65d0f9b06',
                        signature='gAAAAABo-r2izZacxe_jVh_p3URhewxBJuyLNqkJOd0owsDPt9uCE7MXn06WHhO_mp6gLDAqcF1uhMhXCqwztJ1Nbpc0cEDAxUpUCUn2bKSgG6r8Snc_FPtKGgQWDsByvW_Nigx55CyPuNeDO_MiDgYee_WeUw7ASLPfiGOx_9YNc_BFYo1ngsb8CKZcJn3AoponMheLoxkVAPgOjMgteRVaQTr13MljTDUlBIZLIOhVbtIu_dI23saXPigbgwR4RhGn5mCHG_a9ILNkXDJUmGy5TKklIEi2HuJM3ZJ3gfoGYS3OONvzmU4AgMP2UrU17YKZAYKxUBKSpyAqigd4RJSYWzxBCoYzCTmiITwdZ6Cpsw1X9Wox_TQSGt5G2Xu0UY2TQZGRNNH8knJpWs-UQxBBV4L3alMwJuIeV-uzqeKr5fKO5rL_c9as-qQIW_EGQItjvR5z80Hi-S9VXthWCmtqZFIJkgLB5JfTYuFL86valsFVLzSavUIWJAG5qOcxag2mbZMwMRRNfvR__BBtoqBoeGIqveQAbIeZbG0ymw30PH1a2v1mmSrpkK6PB3AHYRDdpkezXLkbyGYgidyV2DAAtPaFplsubWCh_74UxmOuk4BH-9cWkE15mRUBrvtnbTb793RsPzOe7nPmkMpdgqa3nqc6RcQZ_M30lFLUViAbfpEpMVrCzz2cv1RklT1JUzpuVXBTKqQ4FxVCfnvzSgQ2INQ8K50E1X5w_7TAWhrHbNg6LetCa-4KWe9ps0GH6r1x9FWvGyVxSwa7SIdPq3sGpxjOydluPECbBOnHWFUB-3rI2DcUl4rGWYbv2FEFNeCH9Zr67uUvMc4Doi8nVMoeb1lJxFCrfziGhbEXY0FepH3zIzlj-_dXqLAL1qqhfCznT_xkDMVYg-D5gMu-_p3r2SirjJbeaz5UFmP-Dihd9v7jWgD6hx_Mq1uIdzIPE8ImGiDPR7PK64svkvwYg1Czdrc_7GmrKRuzsBL0720UXe19NQqCZfYvUJAjgbEqr3tuS_RkhuEQeeVORn88xkhkrGCEgBS0LHFpe4tcnUEXKnaYYRnoYtk5xo4EyOGVKR2yhF9ht2zrMTo83YuRAPcNT38Jk4gMtVhBaJw_GOfee-IWN_F258rpmU4p8sRV-1iSuQI3Arm4JBU66QuyjoY-KJmTcE9ft3Bfm9If3yG5W0RFRJrsVb--GjHmiiXDGWiR5Q8L1of_RnSD5QDEbXXxhn4dsDejtCXUaQXE9Ty-NvkvA7G6Ru8cMvIKqP2fXS9SmiW6ePJ2Znrlyafxx6L58pT26RF42h90BVrSldf6SjxQApK3AKZW6q8AkuJnYWTtkR9-qfIDl7W94BsgOFoEd-SDQGxWzGJV9YqAu6_SQKiNDQoZZHrJkRSOPEW_b3-BAdrpwL700I92Rye4-BdhlgeK1RwhT3w1Z-z1tvGZXJtPwdpPa3iIw2TIlesMbC1ZJ22iT3CB_r0lnlZhMtIH6o50l50UGfSDuv8HZ_RNgGnYEPqP3FW-o_VD_Yu_KBqGSA0Eb5xAJjl0vpin2vFGO1P4RdgI17eZXRsCp1KvkpWjbEQTWAvJz39yr7wFQ4BrPfgxUqMP0-ZI_h1DkdPBzWs1uKqHw-4qC77sZXgxgHGEIU1tfKosTy_fK4c-WAbdqIHNTh9VdlM1EdrUJQ4rs2rsUG8o9WXwnGTFchI9Ao64LiCFTFTiFL_dvKI4ZraNNXXprfPhxsdLBaNfgj2CIfUwBMJ9xMGmHKQKLtwZdHpQNVqi8DNm1qjvs3CxbSXGKtkl5K8UhJtI1g4OnEnbq3jDO8DGIyDl0NH-0bcCDqS2yAkh8I3IobzxTg16mqU3roXLQ4pGXnWbx26A_9zb4Y1jV7rzCq24VIfNJzMUtW4fVMYzlrp3X1l32I5hF3YP-tU2paD98xobgc2Cn2RWXd3OirrdjKAE088KhXYLZZY59y4LYRLC6MDMHSX0cbEXbBvl6mKmbaFig2_7ICiSa7rR_Ij6PpQRxIW7NfS7ZMu5w7TnhLJyg5nuwMI8A5pVxfy3gYg2L60wepuX7UUV0USaHNKi8qxbp4RJj4nO-GdE8TbLJtvPw-OzrH9Qiv7iDHVMHOe1CDPLD5IeGqmVB0tuLqlyASuIe3oPxTU7QdctyxHa1z-sO8nN6kpPnzmVmS6XK8bY-h5do28dkZvefomSquXwKeiVg9VAMWVziKLPWWg5iWp2x-spLkWcQsQle2T7xizyETaF1t6YbecXtSoVFmu90_o6ns07etU3RVK1YpQLgqUIJwwF3ZwP65MaWPwqDuWCuoQErlApdhRptxId67KE3UC4j8cAaGSoG0kXnws-jzpPyAg1GU8c-Gu_K0F-h-KFbHPMiWCrrQqzVfvoA2wLaQz3NPAqpq-kbFmrXRGkzLIeIvRVxck-sKkxQIcg3amSV5Dykl-lRCXGxlWNiFG_1SFrTSfp5VKyg7l1KjJzXUXHtqAErsPtMyhxaMmlh4An5a8NIaM9W6tafJrBXpUh85DfwZ8W92OAi1WOgoJIwWXSSeSuo6ECDstjVWW3OQQh9183jliwS7Bis3eu9jgAF3q8sYILBdwjrJRa6aAna2GirNwqZMEIg60kIlvmf1U6S2PgYaPm9UDzvMxjpzwjhXhzxHJitfU1tfl0vo-ATaTV8CxmKerNzy2AjlIZnjknG3xLyonCHbGbAe33QQTclb98y_vr5nA4WKlrls413o0a0f8GL8GjINCOd1RHVMjV',
                        provider_name='openai',
                    ),
                    TextPart(
                        content="""\
The monorepo is organized into these main packages:  \n\

• pydantic-ai-slim\u2003– core agent framework (type-safe agents, model interface, tooling)  \n\
• pydantic-evals\u2003\u2003– evaluation system (datasets, metrics, evaluators, reports)  \n\
• pydantic-graph\u2003\u2003– graph-based execution engine (state-machine orchestration)  \n\
• clai\u2003\u2003\u2003\u2003\u2003\u2003\u2003– CLI for scaffolding and running agents  \n\
• examples\u2003\u2003\u2003\u2003– sample apps & demos showing real-world usage\
""",
                        id='msg_0083938b3a28070e0068fabda04de881a089010e6710637ab3',
                    ),
                ],
                usage=RequestUsage(input_tokens=1109, output_tokens=444, details={'reasoning_tokens': 320}),
                model_name='o4-mini-2025-04-16',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 10, 23, 23, 43, 25, tzinfo=timezone.utc),
                },
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_openai_responses_model_mcp_server_tool_stream(allow_model_requests: None, openai_api_key: str):
    m = OpenAIResponsesModel('o4-mini', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(
        m,
        instructions='You are a helpful assistant.',
        builtin_tools=[
            MCPServerTool(
                id='deepwiki',
                url='https://mcp.deepwiki.com/mcp',
                allowed_tools=['ask_question', 'read_wiki_structure'],
            ),
        ],
    )

    event_parts: list[Any] = []

    async with agent.iter(
        user_prompt='Can you tell me more about the pydantic/pydantic-ai repo? Keep your answer short'
    ) as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for event in request_stream:
                        if (
                            isinstance(event, PartStartEvent)
                            and isinstance(event.part, BuiltinToolCallPart | BuiltinToolReturnPart)
                        ) or (isinstance(event, PartDeltaEvent) and isinstance(event.delta, ToolCallPartDelta)):
                            event_parts.append(event)

    assert agent_run.result is not None
    messages = agent_run.result.all_messages()
    assert messages == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Can you tell me more about the pydantic/pydantic-ai repo? Keep your answer short',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                instructions='You are a helpful assistant.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(
                        tool_name='mcp_server:deepwiki',
                        args={'action': 'list_tools'},
                        tool_call_id='mcpl_00b9cc7a23d047270068faa0e29804819fb060cec0408ffbcd',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='mcp_server:deepwiki',
                        content={
                            'tools': [
                                {
                                    'input_schema': {
                                        'type': 'object',
                                        'properties': {
                                            'repoName': {
                                                'type': 'string',
                                                'description': 'GitHub repository: owner/repo (e.g. "facebook/react")',
                                            }
                                        },
                                        'required': ['repoName'],
                                        'additionalProperties': False,
                                        '$schema': 'http://json-schema.org/draft-07/schema#',
                                    },
                                    'name': 'read_wiki_structure',
                                    'annotations': {'read_only': False},
                                    'description': 'Get a list of documentation topics for a GitHub repository',
                                },
                                {
                                    'input_schema': {
                                        'type': 'object',
                                        'properties': {
                                            'repoName': {
                                                'type': 'string',
                                                'description': 'GitHub repository: owner/repo (e.g. "facebook/react")',
                                            },
                                            'question': {
                                                'type': 'string',
                                                'description': 'The question to ask about the repository',
                                            },
                                        },
                                        'required': ['repoName', 'question'],
                                        'additionalProperties': False,
                                        '$schema': 'http://json-schema.org/draft-07/schema#',
                                    },
                                    'name': 'ask_question',
                                    'annotations': {'read_only': False},
                                    'description': 'Ask any question about a GitHub repository',
                                },
                            ],
                            'error': None,
                        },
                        tool_call_id='mcpl_00b9cc7a23d047270068faa0e29804819fb060cec0408ffbcd',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_00b9cc7a23d047270068faa0e4cd5c819f8855c183ff0fe957',
                        signature='gAAAAABo-qDma-ZMjX6meVDoCLYMqgkbQoEVzx_VFnmBFRLqsq37MiF7LP1HrMpqXqtrZ0R2Knb6lUiGSKhsOjOUAn9IFNUCuJx23cPLObF2CKt86wGLb7vccbCrp8bx-I6-kUtZASjlJx7_eJnvwyr24FLZlaDyGDuqRecGA8H4tXnQSAQTT9fJqy8h8dXvxvYzNj5rgOUWgRGn1NBph164KpiEzVWHADzZ_K0l4fX-DFHgtNFssPDYqOKLs_nU0XO8xaIZOgJ8QTf0XmHYF02GA_KciV6sIlSzVricQkwmu1XfJbjpME8XmRMIzlnLRqC8SAJs2kiaYnA8ObfI-s0RbRd3ztIUrzmAsdeo13ualD3tqC1w1_H6S5F47BB47IufTTbpwe_P6f5dLGpOzcrDPbtfHXv-aAW5YEsGyusXqxk51Wp7EONtADmPmVLJffFbRgnwfvPslbxxpNGfxNkN2pIs3U1FW7g1VvmxUfrF84LJpPKvs3xOaWXGorrPBY5nUyeRckhDFt6hGdS59VICmVy8lT4dL_LNswq7dVRS74HrrkfraXDDm2EhL2rtkwhiMqZtuYFsyIK2ys0lZuhNAkhtfgIoV8IwY6O4Y7iXbODxXUr48oZyvLdgV2J2TCcyqIbWClh3-q8MXMmP5wUJdrqajJ8lMVyhQt0UtMJKyk6EWY1DayGpSEW6t8vkqmuYdhyXQOstluONd31LqnEq58Sh8aHCzrypjcLfjDRo5Om1RlxIa-y8S-6rEIXahcJCX_juSg8uYHzDNJffYdBbcLSVQ5mAVl6OM9hE8gHs7SYqw-k-MCeoYsZwt3MqSV7piAu91SMZqB0gXrRDD67bdhmcLBYKmZYKNmLce60WkLH0eZMPSls-n2yyvmwflJA---IZQZOvYXpNUuS7FgMrh3c7n9oDVp15bUgJ8jDx6Mok4pq9E-MHxboblGUpMlFCJDH3NK_7_iHetcqC6Mp2Vc5KJ0OMpDFhCfT3Bvohsee5dUYZezxAkM67qg0BUFyQykulYLHoayemGxzi1YhiX1Of_PEfijmwV2qkUJodq5-LeBVIv8Nj0WgRO-1Y_QW3AWNfQ80Iy6AVa8j9YfsvQU1vwwE9qiAhzSIEeN1Pm2ub8PaRhVIFRgyMOLPVW7cDoNN8ibcOpX-k9p_SfKA9WSzSXuorAs80CTC9OwJibfcPzFVugnnBjBENExTQRfn4l7nWq-tUQNrT4UNGx-xdNeiSeEFCNZlH50Vr5dMaz5sjQQEw_lcTrvxKAV5Zs1mtDf6Kf29LkqhuUEdlMLEJwnAdz2IHLIy41zWLQctSnzBl9HB3mkw8eHZ1LdaRBQRFH4o7Rumhb3D1HdIqDLWeE3jkA6ZBAh2KadGx1u3AIIh4g3dHUS6UREkmzyRIuImbdTsoin1DrQbuYbaqZwIqU4TTIEmA8VeohMfff0rIL5yyFy7cfgGYurgAyMhARPGAAMAoTrR8ldWwymzPkGOJ_SQlzfNGV8weHOEYUl2BgQe57EDX4n1Uk294GIbvGR7eLRL_TLBUyHQErCaOCi8TkBNlLXIobw4ScN_jqqtURmC0mjRDVZeBi6hfrVShWChpQR8A2HxxHrcuHi2hi_2akgUea3zz6_zbUYVoIRdOa9DvZuN015E8ZSL-v_1_vOzUGvt0MuWPazjiRDWgpgcISYzT8N-Xzu_EbwO1OsaOFIeUqrD8mZ6MKOuBQts68og0DWo8KQaHmCaWi4O-c8-5fbB2q3H6oiIoZtSJIoowAmFGOwyWxn_OPS9svDgEaeFYEYhXZ5wZDphxoHkjJ703opxrWoEfQw==',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='mcp_server:deepwiki',
                        args='{"action":"call_tool","tool_name":"ask_question","tool_args":{"repoName":"pydantic/pydantic-ai","question":"What is the pydantic/pydantic-ai repository about?"}}',
                        tool_call_id='mcp_00b9cc7a23d047270068faa0e67fb0819fa9e21302c398e9ac',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='mcp_server:deepwiki',
                        content={
                            'error': None,
                            'output': """\
The `pydantic/pydantic-ai` repository is a Python agent framework designed to simplify the development of production-grade applications using Generative AI . It aims to bring the ergonomic developer experience and type-safety philosophy of Pydantic and FastAPI to AI agent development .

## Core Purpose and Features

The framework focuses on providing a robust and type-safe environment for building AI agents . Key features include:

*   **Type-safe Agents**: Agents are generic `Agent[Deps, Output]` for compile-time validation, leveraging Pydantic for output validation and dependency injection .
*   **Model-agnostic Design**: It supports over 15 LLM providers through a unified `Model` interface, allowing for easy switching between different models and providers  .
*   **Structured Outputs**: Automatic Pydantic validation and reflection/self-correction ensure structured and reliable outputs from LLMs .
*   **Comprehensive Observability**: Integration with OpenTelemetry and native Logfire provides real-time debugging, performance monitoring, and cost tracking  .
*   **Production-ready Tooling**: This includes an evaluation framework (`pydantic-evals`), durable execution capabilities, and various protocol integrations like MCP, A2A, and AG-UI  .
*   **Graph Support**: It provides a way to define graphs using type hints for complex applications .

## Framework Architecture

The framework is structured as a UV workspace monorepo, containing several packages .

### Core Packages

*   `pydantic-ai-slim`: Contains the core framework components such as `Agent`, `Model`, and tools .
*   `pydantic-ai`: A meta-package that includes all optional extras .

### Supporting Packages

*   `pydantic-graph`: Provides the graph execution engine with `Graph` and `BaseNode` .
*   `pydantic-evals`: An evaluation framework for datasets and evaluators .
*   `examples`: Contains example applications .
*   `clai`: Provides a CLI interface .

## Agent Execution Flow

The `Agent` class serves as the primary orchestrator . Agent execution is graph-based, utilizing a state machine from `pydantic_graph.Graph` . The execution involves three core node types:

*   `UserPromptNode`: Processes user input and creates initial `ModelRequest` .
*   `ModelRequestNode`: Calls `model.request()` or `model.request_stream()` and handles retries .
*   `CallToolsNode`: Executes tool functions via `RunContext[Deps]` .

The `Agent` provides methods like `run()`, `run_sync()`, and `run_stream()` for different execution scenarios .

## Model Provider Support

The framework offers a unified `Model` abstract base class for various LLM providers . This includes native support for providers like OpenAI, Anthropic, Google, Groq, Mistral, Cohere, and Bedrock . Additionally, many OpenAI-compatible providers can be used with `OpenAIChatModel` .

## Tool System

Tools are registered using the `@agent.tool` decorator . The system automatically generates JSON schemas from function signatures and docstrings, validates tool call arguments, and provides context injection via `RunContext[Deps]` .

## Observability Integration

Pydantic AI integrates with OpenTelemetry, allowing for instrumentation of agent runs, model requests, and tool executions . It has native integration with Pydantic Logfire for enhanced monitoring and visualization .

## Evaluation Framework

The `pydantic-evals` package provides a framework for systematically testing and evaluating AI systems . It supports defining datasets with `Case` objects and using various evaluators, including built-in and custom ones .

## Integration Ecosystem

Pydantic AI supports various integrations for development and production:

*   **Model Context Protocol (MCP)**: For external tool server access .
*   **AG-UI Protocol**: For interactive application frontends .
*   **Agent2Agent (A2A)**: For multi-agent communication and workflows .
*   **Temporal**: For durable workflow execution .
*   **DBOS**: For database-backed execution and state persistence .

## Notes

The `CLAUDE.md` file provides guidance for Claude Code when working with the repository, including development commands and an overview of core components and design patterns . The `mkdocs.yml` file defines the structure and content of the project's documentation, including navigation, plugins, and watch directories for various packages  . The `docs/install.md` file details how to install the `pydantic-ai` package and its optional components, including a "slim" installation option for specific model dependencies .

Wiki pages you might want to explore:
- [Overview (pydantic/pydantic-ai)](/wiki/pydantic/pydantic-ai#1)

View this search on DeepWiki: https://deepwiki.com/search/what-is-the-pydanticpydanticai_e234e9cf-d4aa-4c67-a257-56034816dd56
""",
                        },
                        tool_call_id='mcp_00b9cc7a23d047270068faa0e67fb0819fa9e21302c398e9ac',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_00b9cc7a23d047270068faa0f4ff54819f9fb9ff25bebe7f5f',
                        signature='gAAAAABo-qD2WTMmhASwWVtFPlo7ILZP_OxHfRvHhda5gZeKL20cUyt0Np6wAHsJ6pyAsXCkLlKBVz3Vwm52JrJuUbqmw-zlXL19rbpvTPRMkiv_GdSfvmxKKNJvSm417OznBDVjsIAqmes2bMq03nRf6Pq2C0oUJnIbpbMwtWzs3jMQqUb0IwyopqXGhn3MWKctLPKZS89nyL4E9kJAx_TyWTQvME8bf8UrV8y2yrNz9odjSQQyZq5YXrlHzpOJjDTfLofVFjsEzM8J29SdLcWnqlv4djJ8xeMpP2ByXuHRnTEyNNuxpYJB7uQbYT0T_eLhwcLv2ZzDZ_hf2Msv7ZdyuPc7Yxc5YWlChB0iaHqQ_8UuMjIVurfgSIjSq2lTvJwdaA365-ZoBMpo4mG04jQDP3XM-0xEM6JTFWc4jZ1OjIXVpkjaXxdOOkYq3t3j8cqBQH69shFCEQr5tnM8jOEl3WHnkvaBg4xEMcd61hiLOKnWbQiYisbFucA8z5ZNbdohUZd-4ww0R8kSjIE5veiyT66gpIte0ItUnTyhIWy8SZYF9bnZGeS-2InDhv5UgjF2iXzgl6dmUrS-_ITgJkwu4Rdf9SBDJhji3_GUO9Za0sBKW8WohP142qY0Tbq4I6-7W1wJ3_gHJqiXVwDLcY90ODSyyC5_I3MgaALRC1wt55sHSeSsDjmNGmiH-m0snaqsI0JnAZwycnWCK17NamjQ9SxVM5tTqJgemkGFQNH1XhZPWvVj56mlj74KKbCJALQpdXD27C8LfdrlBd0v_zEmF1dh7e12I95fYeAlO51xOglBaMCgcMWSDHMGHsJBbJ04eVQSwYTl72rmkASTMaybD-aAm1m8qZnKU-f3xQradhs9l1x9eOfQDIsfWMr1aVMiZi59--VsrgYCbqBj7AGf8n6VNbQWkhO2etozwYZcdGIyiu4TaULX1Xp89Gb28M-tVkIrkQoHO_Z7wzKU1HRBViES1wRKUJ-Sa6wc8UP5orDxeOTFPUr7JL-qaj49cpKzvdlfuoIdbYwpsNvAg69sNbFI3w4jLxOT4yxS6thra1Bit6SY5wAEfrrjtzofLeg49aFqFVGIHeJ8kE3spc1rctpETkdHNyP9fEjZaM3mxR4yz0tPmEgUsd-sdw5BbOKDAVzwconmbeGBmf9KLXMEpRRH7-qSIWUscCi5qIdHXGYoQkStsNGrnhucn_hwqZCSti3Kbzfosud3zQPjW6NyuJCdeTxbDbsnrV7Lkge5j92pyxCHw9j0iuzofRW55_KToBtIvRoPr_37G_6d6TxK42mKqdbgk9GHrcXf27mXszCEzX-VfRVTxyc6JLfEy1iikdo-J2AzXPd4m3zE-zazBU3Z5ey596g8gxwXMkHakLrvwp4_-fQfcvs7sIH34xkEhz7BRdNok3Aqbu_zCt2np69jjHqfPQWZzAy1C-bmMuhAaItPYkkw-LgSu-YP6L89zNofK9Q_S3JwVsLN-fq-9OwhSjy_rQu22Gn4KD6saAu61QMXBPa6z0QJSFUZHJQ_megq1tENfB6wRVtQ0DdAvUwhUsMwx6yE9CT20bma4CloGW__aZuD9gikdQrQ1DCHOvTrfEpvHkl6-wuCImeNjsCvbRFAkx6Xgpc6fdbq4j6WyEVW_4VePNknFWYZ1cw795ka5uJMLc3hVughVlGwDbw60Q3utsjHPbu03pxPle5pdcVEYSQWa0WbFDCrF4ysK0lpmlF7',
                        provider_name='openai',
                    ),
                    TextPart(
                        content=IsStr(),
                        id='msg_00b9cc7a23d047270068faa0f63798819f83c5348ca838d252',
                    ),
                ],
                usage=RequestUsage(input_tokens=1401, output_tokens=480, details={'reasoning_tokens': 256}),
                model_name='o4-mini-2025-04-16',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 10, 23, 21, 40, 50, tzinfo=timezone.utc),
                },
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    assert event_parts == snapshot(
        [
            PartStartEvent(
                index=0,
                part=BuiltinToolCallPart(
                    tool_name='mcp_server:deepwiki',
                    args={'action': 'list_tools'},
                    tool_call_id='mcpl_00b9cc7a23d047270068faa0e29804819fb060cec0408ffbcd',
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=1,
                part=BuiltinToolReturnPart(
                    tool_name='mcp_server:deepwiki',
                    content={
                        'tools': [
                            {
                                'input_schema': {
                                    'type': 'object',
                                    'properties': {
                                        'repoName': {
                                            'type': 'string',
                                            'description': 'GitHub repository: owner/repo (e.g. "facebook/react")',
                                        }
                                    },
                                    'required': ['repoName'],
                                    'additionalProperties': False,
                                    '$schema': 'http://json-schema.org/draft-07/schema#',
                                },
                                'name': 'read_wiki_structure',
                                'annotations': {'read_only': False},
                                'description': 'Get a list of documentation topics for a GitHub repository',
                            },
                            {
                                'input_schema': {
                                    'type': 'object',
                                    'properties': {
                                        'repoName': {
                                            'type': 'string',
                                            'description': 'GitHub repository: owner/repo (e.g. "facebook/react")',
                                        },
                                        'question': {
                                            'type': 'string',
                                            'description': 'The question to ask about the repository',
                                        },
                                    },
                                    'required': ['repoName', 'question'],
                                    'additionalProperties': False,
                                    '$schema': 'http://json-schema.org/draft-07/schema#',
                                },
                                'name': 'ask_question',
                                'annotations': {'read_only': False},
                                'description': 'Ask any question about a GitHub repository',
                            },
                        ],
                        'error': None,
                    },
                    tool_call_id='mcpl_00b9cc7a23d047270068faa0e29804819fb060cec0408ffbcd',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                ),
                previous_part_kind='builtin-tool-call',
            ),
            PartStartEvent(
                index=3,
                part=BuiltinToolCallPart(
                    tool_name='mcp_server:deepwiki',
                    tool_call_id='mcp_00b9cc7a23d047270068faa0e67fb0819fa9e21302c398e9ac',
                    provider_name='openai',
                ),
                previous_part_kind='thinking',
            ),
            PartDeltaEvent(
                index=3,
                delta=ToolCallPartDelta(
                    args_delta='{"action":"call_tool","tool_name":"ask_question","tool_args":',
                    tool_call_id='mcp_00b9cc7a23d047270068faa0e67fb0819fa9e21302c398e9ac',
                ),
            ),
            PartDeltaEvent(
                index=3,
                delta=ToolCallPartDelta(
                    args_delta='{"repoName":"pydantic/pydantic-ai","question":"What is the pydantic/pydantic-ai repository about?"}',
                    tool_call_id='mcp_00b9cc7a23d047270068faa0e67fb0819fa9e21302c398e9ac',
                ),
            ),
            PartDeltaEvent(
                index=3,
                delta=ToolCallPartDelta(
                    args_delta='}', tool_call_id='mcp_00b9cc7a23d047270068faa0e67fb0819fa9e21302c398e9ac'
                ),
            ),
            PartStartEvent(
                index=4,
                part=BuiltinToolReturnPart(
                    tool_name='mcp_server:deepwiki',
                    content={
                        'output': """\
The `pydantic/pydantic-ai` repository is a Python agent framework designed to simplify the development of production-grade applications using Generative AI . It aims to bring the ergonomic developer experience and type-safety philosophy of Pydantic and FastAPI to AI agent development .

## Core Purpose and Features

The framework focuses on providing a robust and type-safe environment for building AI agents . Key features include:

*   **Type-safe Agents**: Agents are generic `Agent[Deps, Output]` for compile-time validation, leveraging Pydantic for output validation and dependency injection .
*   **Model-agnostic Design**: It supports over 15 LLM providers through a unified `Model` interface, allowing for easy switching between different models and providers  .
*   **Structured Outputs**: Automatic Pydantic validation and reflection/self-correction ensure structured and reliable outputs from LLMs .
*   **Comprehensive Observability**: Integration with OpenTelemetry and native Logfire provides real-time debugging, performance monitoring, and cost tracking  .
*   **Production-ready Tooling**: This includes an evaluation framework (`pydantic-evals`), durable execution capabilities, and various protocol integrations like MCP, A2A, and AG-UI  .
*   **Graph Support**: It provides a way to define graphs using type hints for complex applications .

## Framework Architecture

The framework is structured as a UV workspace monorepo, containing several packages .

### Core Packages

*   `pydantic-ai-slim`: Contains the core framework components such as `Agent`, `Model`, and tools .
*   `pydantic-ai`: A meta-package that includes all optional extras .

### Supporting Packages

*   `pydantic-graph`: Provides the graph execution engine with `Graph` and `BaseNode` .
*   `pydantic-evals`: An evaluation framework for datasets and evaluators .
*   `examples`: Contains example applications .
*   `clai`: Provides a CLI interface .

## Agent Execution Flow

The `Agent` class serves as the primary orchestrator . Agent execution is graph-based, utilizing a state machine from `pydantic_graph.Graph` . The execution involves three core node types:

*   `UserPromptNode`: Processes user input and creates initial `ModelRequest` .
*   `ModelRequestNode`: Calls `model.request()` or `model.request_stream()` and handles retries .
*   `CallToolsNode`: Executes tool functions via `RunContext[Deps]` .

The `Agent` provides methods like `run()`, `run_sync()`, and `run_stream()` for different execution scenarios .

## Model Provider Support

The framework offers a unified `Model` abstract base class for various LLM providers . This includes native support for providers like OpenAI, Anthropic, Google, Groq, Mistral, Cohere, and Bedrock . Additionally, many OpenAI-compatible providers can be used with `OpenAIChatModel` .

## Tool System

Tools are registered using the `@agent.tool` decorator . The system automatically generates JSON schemas from function signatures and docstrings, validates tool call arguments, and provides context injection via `RunContext[Deps]` .

## Observability Integration

Pydantic AI integrates with OpenTelemetry, allowing for instrumentation of agent runs, model requests, and tool executions . It has native integration with Pydantic Logfire for enhanced monitoring and visualization .

## Evaluation Framework

The `pydantic-evals` package provides a framework for systematically testing and evaluating AI systems . It supports defining datasets with `Case` objects and using various evaluators, including built-in and custom ones .

## Integration Ecosystem

Pydantic AI supports various integrations for development and production:

*   **Model Context Protocol (MCP)**: For external tool server access .
*   **AG-UI Protocol**: For interactive application frontends .
*   **Agent2Agent (A2A)**: For multi-agent communication and workflows .
*   **Temporal**: For durable workflow execution .
*   **DBOS**: For database-backed execution and state persistence .

## Notes

The `CLAUDE.md` file provides guidance for Claude Code when working with the repository, including development commands and an overview of core components and design patterns . The `mkdocs.yml` file defines the structure and content of the project's documentation, including navigation, plugins, and watch directories for various packages  . The `docs/install.md` file details how to install the `pydantic-ai` package and its optional components, including a "slim" installation option for specific model dependencies .

Wiki pages you might want to explore:
- [Overview (pydantic/pydantic-ai)](/wiki/pydantic/pydantic-ai#1)

View this search on DeepWiki: https://deepwiki.com/search/what-is-the-pydanticpydanticai_e234e9cf-d4aa-4c67-a257-56034816dd56
""",
                        'error': None,
                    },
                    tool_call_id='mcp_00b9cc7a23d047270068faa0e67fb0819fa9e21302c398e9ac',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                ),
                previous_part_kind='builtin-tool-call',
            ),
        ]
    )


async def test_openai_responses_model_mcp_server_tool_with_connector(allow_model_requests: None, openai_api_key: str):
    m = OpenAIResponsesModel(
        'o4-mini',
        provider=OpenAIProvider(api_key=openai_api_key),
    )
    agent = Agent(
        m,
        instructions='You are a helpful assistant.',
        builtin_tools=[
            MCPServerTool(
                id='google_calendar',
                url='x-openai-connector:connector_googlecalendar',
                authorization_token='fake',
                description='Google Calendar',
                allowed_tools=['search_events'],
            ),
        ],
    )

    result = await agent.run('What do I have on my Google Calendar for today?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(content='What do I have on my Google Calendar for today?', timestamp=IsDatetime())
                ],
                timestamp=IsNow(tz=timezone.utc),
                instructions='You are a helpful assistant.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(
                        tool_name='mcp_server:google_calendar',
                        args={'action': 'list_tools'},
                        tool_call_id='mcpl_0558010cf1416a490068faa0f9679481a082dc4ac08889f104',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='mcp_server:google_calendar',
                        content={
                            'tools': [
                                {
                                    'input_schema': {
                                        'properties': {
                                            'calendar_id': {
                                                'anyOf': [{'type': 'string'}, {'type': 'null'}],
                                                'default': None,
                                                'description': "The ID of the calendar to search. Default one is 'primary'",
                                                'title': 'Calendar Id',
                                            },
                                            'max_results': {'default': 50, 'title': 'Max Results', 'type': 'integer'},
                                            'next_page_token': {
                                                'anyOf': [{'type': 'string'}, {'type': 'null'}],
                                                'default': None,
                                                'title': 'Next Page Token',
                                            },
                                            'query': {
                                                'anyOf': [{'type': 'string'}, {'type': 'null'}],
                                                'default': None,
                                                'title': 'Query',
                                            },
                                            'time_max': {
                                                'anyOf': [{'type': 'string'}, {'type': 'null'}],
                                                'default': None,
                                                'description': "Time in the ISO-8601 format. You can also use 'now' or leave null.",
                                                'title': 'Time Max',
                                            },
                                            'time_min': {
                                                'anyOf': [{'type': 'string'}, {'type': 'null'}],
                                                'default': None,
                                                'description': "Time in the ISO-8601 format. You can also use 'now' or leave null.",
                                                'title': 'Time Min',
                                            },
                                            'timezone_str': {
                                                'anyOf': [{'type': 'string'}, {'type': 'null'}],
                                                'default': None,
                                                'description': "Timezone of the event. Default is 'America/Los_Angeles'",
                                                'title': 'Timezone Str',
                                            },
                                        },
                                        'title': 'search_events_input',
                                        'type': 'object',
                                    },
                                    'name': 'search_events',
                                    'annotations': {'read_only': True},
                                    'description': 'Look up Google Calendar events using various filters.',
                                }
                            ],
                            'error': None,
                        },
                        tool_call_id='mcpl_0558010cf1416a490068faa0f9679481a082dc4ac08889f104',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_0558010cf1416a490068faa0fb684081a0a0b70f55d8194bb5',
                        signature='gAAAAABo-qEE669V-_c3vkQAeRtSj9pi72OLJweRJe4IRZkLcFfnuwdxSeJM5DVDLzb3LbfzU0ee6a4KAae0XsETU3hELT1hn3LZPwfFku5zl7CVgsc1DmYBf41Qki1EPHFyIlMj937K8TbppAAqMknfLHHwV1FLb8TapccSEhJbzGutqD3c2519P9f6XHKcuDa8d-sjyUejF0QuSjINFcjifJ8DiU40cL_-K6OJotlx6e0FqOivz6Nlj13QZxQ0I3FiiSi03mYKy240jYMpOpjXr7yPmEXLdCJdP5ycmTiJLxf4Bugww6u4F2uxy22978ACyFGSLHBiQyjczj_can7qKXAkMwYJKcGNjaNi8jG5iTIwsGswRjD1hvY-AGUotMFbPCszX3HW1M_ar-livaheiZauCfKV-Uc1ZeI3gijWEwtWQ0jye29FyQPCCpOBvT6RbUvFEpfqpwcMQuUhOyEfgzli2dpuOAgkSjCPE6ctoxjbYa62YzE-yrXAGc5_ptQy_2vw7t0k3jUzSo2Tv0aKnqvvKcj9SIilkZV4Nf-TL_d2E7d48bBJDlqbAv7fkhhd2YlkLqwdR1MqZtygcR1Jh8p2Y1pFAa4mSj7hh4M-zfSu--6dij2iKIbnKQ4DbXyGpMZXBAqTHMe9PPOwGxWKShlN5a5T89B04d_GwJYBDJx2ctecqZxDMjkTn3wVGl_5wuDnrEgd0I91vmAoYuWldR_h8M_FjDFiHefdbZjw1TxVKjkp6wk6zQiXCvvCZYJa9XkhytcllWvUI4C0gbxHrEzZRy9Vii3buqnbiIM9Qj0VPx-Q-FKM_usZBBmlvmk9PMQ8rH9vVT8dRFNQEj-aqudB5yUcTx8XaUFwYAts04OObGBqXoazYtxh6WvHwrf09pb_g0dwzE_rlcQdYxcFLOpYD-AentRAjOuIr4bLRM9BMERBxPvvPCxZ2Mva8YqV2TIOtxzMY08freim6du1IuYprO6CoejPaBdULhct-nsPubOdjLBikZt_bwumvmqGXnxI_uu51b9HtzPeDpWIjF6pi88bcsOk0qglA9GAu3wwX-iIdaV19VdVCO4KJjxiVrbTY1IVgWSdz98Alb_HzpXsoS6i2PRAjjsYOe4RBX3etxjsY07XXLlmXAM_vuYXc8Y6STxvBk4ST4OkaCvUk9DoZbVL5KmVcT6TaFpbVCOB_eHkHIvMjXc35kzxCdqEMG3FpRzL_UkY8pPridvq2z1Xw0al2KEBvdKPlInB8-zX5ANGeRkMGZ6ZfyX1zCIdYLe3wrC8xqr5nUZ-ueWmtqYLavSg8mQKphp4QyVaiwtbxEt5GEiVG7_LR754mGQYPdr9Shh3ECAp8wmSfDVO8MHaLmzgo3RXeqlqFldRjQzDHtCaGhjD9bHKF3yWF2LtH4gUN-Sf--86lcq7iwHDSDm656P_FBfYmE7rA0svH-m3hQoBhza4CKJ7s7f7ZymEhcHAfH7SPImZ3Y-kT_Sy1mbCCf3Yg8uitrpX7ukO6_bIANS_R4oiOPcuLixbWY0ZSyq8ERB5fa5EsIUm7PpGxbO96nmk5rPkewyB4gCtslwJI0Ye7zHtqrDBz1j1nsjIKsRCfFWlUdRF8J1JPiiBSvP8SraQ_94cnKBCsl34BGsVm-R1_ULbuyahBzSHq2Kwr0XQuNLdGChyLKS_FZVT58kbRFsvjZnbalAZ-k9alMeZ-pdWX5f9nSn3w7fz675zOxnBaqiZmoWHXFNOBVGH7gkz05ynJ2B8j_RpdRNJKXUN8pAvf595HGl2IPdaDhqoeS2_3jixO5mmxZuPEdzopoBFRarWud99mxH-mYxWJzKiA1pLNqj7SO93p2-jB-jtsCfZfk6bVEWpRRkIEz0XvxffFTVuGUCqpGS7FiFZc4pQU24pCrdpg2w3xeDSrmfHDAx2vUvv0iRBnQxTTWx2-de2TQQTpR5tjFNyOhYGVn1OXqkbkNtIUHdnNGA1QBCU0Qs0471Ss1CrxXIeeNVSTd00jiu4_ELk6nJYgSpmS8G_crrDza8mRLV5Yk0ItRrZj6pwKUOEaYeyM-RHyhrjf09yaf7Qc3sAozQF0aXFCQjSYiVb98DuGH28HLUxW9ulmSKKR4pYKlCOLNGm0h_gWCpSa0H1HXCgEoPn68HyaJogv_xH3k4ERYyJnxu8zVbVPMGoa9q9nNRQQ9Ks2AvxYRQeGFSCTACBmuookvHsO1zjYfHNuSCD7pCLRFE76KlmSiAX6l9LNOq_xe9Oos-1AvcZHkmVsuh-mjTVkBOjG6zmnHiNJirBpORs_UWL5lmlQBeaXgdHxcb4tHIn8XYXFkQiC4b4pw==',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='mcp_server:google_calendar',
                        args={
                            'action': 'call_tool',
                            'tool_name': 'search_events',
                            'tool_args': {
                                'time_min': '2025-10-23T00:00:00',
                                'time_max': '2025-10-23T23:59:59',
                                'timezone_str': 'America/Los_Angeles',
                                'max_results': 50,
                                'query': None,
                                'calendar_id': 'primary',
                                'next_page_token': None,
                            },
                        },
                        tool_call_id='mcp_0558010cf1416a490068faa0fdf64481a085a3e5b8f7d6559a',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='mcp_server:google_calendar',
                        content={
                            'output': None,
                            'error': {
                                'code': 500,
                                'message': 'An unknown error occurred while executing the tool.',
                                'type': 'http_error',
                            },
                        },
                        tool_call_id='mcp_0558010cf1416a490068faa0fdf64481a085a3e5b8f7d6559a',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_0558010cf1416a490068faa0ff5c9081a0b156a84d46e5d787',
                        signature='gAAAAABo-qEE72KCH4RlulMdH6cOTaOQwFy4of4dPd8YlZ-zF9MIsPbumWO2qYlZdGjIIXDJTrlRh_5FJv2LtTmMbdbbECA20AzFMwE4pfNd2aNLC5RhcHKa4M9acC1wYKAddqEOPP7ETVNBj-GMx-tMT_CY8XnBLWvSwPpcfde9E--kSrfsgvRn1umqDsao4sLlAtV-9Gc6hmW1P9CSJDQbHWkdTKMV-cjQ-wZHFCly5kSdIW4OKluFuFRPkrXs7kVmlGnMr8-Q5Zuu1ZOFR9mPvpu2JdxAFohjioM-ftjeBuBWVJvOrIF4nV-yIVHVT-_psAZaPUUB5cyPAtqpoxxIV3iPKPU8DHctP03g_0R6pSWWHhggvO5PBw3zyPwtBwOrHBipc4nQEWEMxZxLH5SYJauTKwHNOx9NyCq8JUjZXM_v4xsGxNa4cAp7GuXqR2YyW2sx7syRUiDwtebh0xk_YOQtkv8tAjzCofmaz3n8FJ2nGSXkilaV5Q8LUNO-9-D2tsAaScDVMuLMMAHFNp_GPplWrmGES4mTCNtTXWyF1GLcQBw8dYYctV66Ocy2_zxyDoB7SsR5htlV77nJ6u1Hbp3tk26LutDrhAhe55xcki8iblHbXNY9MRzR1SS5Zk3-dv0ex4QOzC663NvS9aK3olQbKYko5TvM7Pq4MFYfaxwFTVFVEdaskoDJieVyikz0ZzBjTsItIwL-Q2BVN2F_P_wgCV5hyDclNMPEGTMxajxfIFv-oEunmHY1_RJavl47iXWS8H3JWAvp-9YYQdTS4Aa6m5zPndvHOvEV355UawLHRPctHFUS7rE7rYmcU6KQaqC96JRM0KRfXNIgYtNfw6cxgnyqGxzTF7qeeVzObOqoQmz59Rh0U9ti37vqHb8Ca43-q2Gx2KaVZFj7MBQK8UodfaDRIEuyMB3XNfckxCefwHs7FeAj5NuNDBrm0uDcwJjs2JfY2i54gAES8kAPLGJgRpq_qdjVXqpO6W0H9E1vBdRem7zLPYbA8OOo-KCkRW4AFCVbgCpgIvo4GDNvFOMksl-d8zgQU2qroUWJRu58j1bdaar7Zlfxk0UR33nROmJpXGb_R-RCNAN1ZxJTdEU_dVfyLCeuIXPsnO-FlfO8J6Un3WWPNLuN_bDS5RocniI_ms71qLsisJQiPTs-JDFl-eMM2Hk3QqSCC6OT0CLG9XMmI_zva9yp2joQ8HdGMddE3FDCbLejRrx8fV-9Nd0tZ7SYjFG78_fre8IfL0L67CK1JIPYzhgRZgCb-FFwUy-stR_BstIn0sRr_tDCoHdxuoVCh0dZfTY1p27xbKQ50svHxp1caNp3uze0wLXP9STNouFjFpdIHMsDRaGfO9R9mMmUsFcmBMK3aikuHTpebyL1CeZsIzH2cbZLPRx3pN2IqJ-5h6-cORHuMqf3ysEEFCjXnqmzvWPuBjYDsxnxA1awaGkYKsKhqchgakrfplOjdG5tSkklggBJA93iRaUWIR-4oV6HkkrnpdK1w7BL_VT8upqZmkpHZtZCDSgINk5S5hoYPLBTtS3dcCmQIbLvPXPuGzdAZxl0bhD4Rm3GPDFszaDoFK0Jszcjlaf4SJqyZABKEf71dDbi1as-2Qwr4fxBiQIOsF8ChbYo6Z2iFtUpBnbruFUIwB5QyKfWnwEZbOgf4UbIvIqNMkTzMc8tJgz6Ddqfih8VeNH3v8_84J6vHU0SVm_gvkgQ6P6N_6r5LwNdlAEff0hFwn-aTHWZ3s8MICckUZj97lKoZxAl91WlsKa0yrLw24dxvJ6bhZf0FsOitUJGd7vFPx0TxSobUkzE2RrbQ3hziPxw2Gins4aI6YG3M1gfumd3MgdH-fYBvZulJ9vmw0ZC1Dqh6BkCWHOFKsnpQvHmYuyTzUmnYuJf8N5j_b9XNw0krmxouOCPQClFmIOBLw8XPbe3xf0F5JP7BC0PpjlPT33A5Z6Za5zlA5O-DE_Wp0WG885-GaKtZI-zBZW3R0lc9A4s0HbxqA3lqH8leXOCe6WO46Z_iTQlALpTR-7oaHqzTegq0KSmEjCFO-jLSrVZnBOQ4ddTvLj4ASsQbj-o6TFUFVZAKSLI3FtWovHw02Gc_D0luFz9TbfaXM-EapEQYajkG0_b_nSCoPq0T9HSyvU4oCxXyQvhwIgzbijR-BheN6a_l6hiqZCw9L1c8MdPRtjpbHtEwWkpQ62s8XdydeJnV5vJYp9ezBbS_vWQ7Nz1siai6epJTdzDkRm-dudVhKzdohwg-FOQ-5gSrvoPS_MF4lZvah3iXY1g4uePO4eNDWGJ74YPybiy',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='mcp_server:google_calendar',
                        args={
                            'action': 'call_tool',
                            'tool_name': 'search_events',
                            'tool_args': {
                                'time_min': '2025-10-23T00:00:00Z',
                                'time_max': '2025-10-23T23:59:59Z',
                                'timezone_str': None,
                                'max_results': 50,
                                'query': None,
                                'calendar_id': 'primary',
                                'next_page_token': None,
                            },
                        },
                        tool_call_id='mcp_0558010cf1416a490068faa102400481a09fa35e84bc26c170',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='mcp_server:google_calendar',
                        content={
                            'output': None,
                            'error': {
                                'code': 500,
                                'message': 'An unknown error occurred while executing the tool.',
                                'type': 'http_error',
                            },
                        },
                        tool_call_id='mcp_0558010cf1416a490068faa102400481a09fa35e84bc26c170',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_0558010cf1416a490068faa102d89481a0b74cca04bcb8f127',
                        signature='gAAAAABo-qEECuiSxfvrR92v1hkqyCTCWyfmpHSaW-vVouk5mOTIFDvaBZdVTFH8-dJfpwEG3MCejRKh9V-I8mrYAjhudVr1ayHo8UYOOU1cfVc6w3wsrkL8hXljjE-amiJhBSjvRc2nwwGtgYpDxOfWTqJkaUvFnMD6MrS4CwMrCBbDOLYZgM1cQbidtrrtpP7D5u42tR6coC_PCOqwPzDN4f0RggrxVxh0038p81VUmlkUeA2jWzRyFpeDGRjXFk84Og73rXAp7EWQv7TmzgVXBjCVwwzJNU8HCZ_gkwh5dvL94QxBx32lEmfOOKcqA3hN3FLwDqXlZ8f7jEqYInnpILQgX5XMdM9OrCyXmDCr_eIy00cjvxnTcXhCnZBOaKCKmTP74yUpGNdLbQcr4BalTiviNYEeCAhJyRo4KnhUZbBoT7MB5NULf-kqhRo1gEGKjWiLdV47PhR7Z8i4BK7zBceganMKpLtzIMW5a6JAujC4Z9FYxcpJZI_CD9NHsPr4SjKgIwv89d6BYo89-xfflF6ZUZBkuDUnL2-Nc9CKgGuKlcDunvYLr38pzA278OFYzh9T42u4SbS8KkSXKjGU3H8LfpMnBEZigriixLt5vj7qnWmZvCFarzxT4U4qqR1ITp5rkO6G9kYvBEfS7wu768mteDBgAajUaeOMQEfjJRErC4wfzbB89YCsXPJz0JE90QZ5LeiP5ZlVezTTaddG9JmiGsBCPckqUb1LWdpvekCfPkePF_uDMVWyJpQ4ZBzQsZx8sHf5spygsiQjlzTiriqwhoTcPuXoONoCr9HeFX1Qy8SGOm87siRPAD7FHJdDxbJwq8tOlMpx8MH1dqEY07lwoxZB0GQ9XbB7QJXfQR_27nkpqBYFkrbqChNJLO2x8gNFClbB0mgYQE1CRy64y6yOrG3CtS53RK5VGrF1GnqwuWdZ452VgShT5nAmPFRlRk1S9px4eMUTAozT0QAYrlHQC7b6I6K3m_Qe3kXGpnn_87i2eGG8mHmXG2FvFChkgf2OU7-LRy_Wl_u-ataICeoBwfngBFMppvUW6tJP009HK7mUE8P1KJntN3ExKLIBhmKhV6ziBpIi1bSTmd8leYqfSaf648c7-sVuDRx7DzxTp19l3fwVFa67GdiagZFs7xaU1HxMnMc3uy5VKWAH_qcv-Mga3VCTtTPpMTjvB95nsLeOFjS2FtpPvaP0N6o5kkkzW7cteWpOHhSX0z7AQA7CqgOCQLfLUc7ltVxnOH4WdHoeZFah_q_Ue6caf0kNo4YsTfbRDdzsW70o8P5Agr-Pgttg19vTDA_eBFur9GDKIRT0vYMWPpykwJBDTgJKOFW6uyNkqNWk_RAAvleE9pAyOoSmgomyrMcnnpdeYHNxeNxvTWFC3mcKSjJIB316wypPvaGTJyaK_pxJScD7CtLrIPkgwPpOsJnDySF6wGe-fGsUMt3zxJrc-S6fp24mYVfTRZbjUsP0fJgLmCohJiAtEg_xvlQ8sPyuLoLdOdossTQ7ufl0CwVn4f_ol4q__gpTvYVaoGsWl3QmHul5zj7OUAn7of6iBfCSlXbrauJvMyNYt4x_dLM8SXTRNPe-ZMDmER9DOw0KJXcUrpl6uw4TphKmUOK6KrxqshujXdN9VDgOwD7eKqIHpvC_6a2R6sS6ZHcebmh2o3bic-Hctomrbv03OQ==',
                        provider_name='openai',
                    ),
                    TextPart(
                        content=IsStr(),
                        id='msg_0558010cf1416a490068faa103e6c481a0930eda4f04bb3f2a',
                    ),
                ],
                usage=RequestUsage(input_tokens=1065, output_tokens=760, details={'reasoning_tokens': 576}),
                model_name='o4-mini-2025-04-16',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 10, 23, 21, 41, 13, tzinfo=timezone.utc),
                },
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )
