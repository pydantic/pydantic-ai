"""Tests for the Amazon Bedrock Knowledge Base backend."""

from __future__ import annotations

import copy
import io
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError, ReadTimeoutError
from typing_extensions import Unpack

from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai_harness import KnowledgeBase
from pydantic_ai_harness.knowledge_base import BedrockKnowledgeBase, Document, MetadataFilter, Passage, normalize_filter

if TYPE_CHECKING:
    from mypy_boto3_bedrock_agent_runtime.type_defs import (
        KnowledgeBaseRetrievalConfigurationTypeDef,
        KnowledgeBaseRetrievalResultTypeDef,
        RetrievalResultContentTypeDef,
        RetrievalResultLocationTypeDef,
        RetrieveRequestTypeDef,
        RetrieveResponseTypeDef,
    )


def _result(
    text: str | None = 'alpha passage',
    *,
    location: RetrievalResultLocationTypeDef | None = None,
    score: float | None = 0.87,
    metadata: dict[str, object] | None = None,
    content: RetrievalResultContentTypeDef | None = None,
) -> KnowledgeBaseRetrievalResultTypeDef:
    if content is None:
        content = {'type': 'TEXT', 'text': text} if text is not None else {'type': 'TEXT'}
    result: KnowledgeBaseRetrievalResultTypeDef = {'content': content}
    if location is not None:
        result['location'] = location
    if score is not None:
        result['score'] = score
    if metadata is not None:
        # The stub types metadata values as mappings (the API's generic "document"); the service returns scalars.
        result['metadata'] = metadata  # pyright: ignore[reportGeneralTypeIssues]
    return result


def _response(*results: KnowledgeBaseRetrievalResultTypeDef) -> RetrieveResponseTypeDef:
    return {
        'guardrailAction': 'NONE',
        'retrievalResults': list(results),
        'ResponseMetadata': {'RequestId': 'request-1', 'HTTPStatusCode': 200, 'HTTPHeaders': {}, 'RetryAttempts': 0},
    }


def _client_error(code: str) -> ClientError:
    return ClientError({'Error': {'Code': code, 'Message': f'{code} happened'}}, 'Retrieve')


@dataclass
class _FakeClient:
    """In-memory `BedrockAgentRuntimeClient`: a canned response, recorded request arguments."""

    response: RetrieveResponseTypeDef = field(default_factory=_response)
    error: Exception | None = None
    calls: list[RetrieveRequestTypeDef] = field(default_factory=list['RetrieveRequestTypeDef'])

    def retrieve(self, **kwargs: Unpack[RetrieveRequestTypeDef]) -> RetrieveResponseTypeDef:
        if self.error is not None:
            raise self.error
        self.calls.append(kwargs)
        return self.response


def _configuration(client: _FakeClient) -> object:
    assert len(client.calls) == 1
    return client.calls[0].get('retrievalConfiguration')


class TestSearch:
    async def test_sends_the_query_limit_and_knowledge_base(self) -> None:
        client = _FakeClient()
        await BedrockKnowledgeBase('KB123', client=client).search('q', limit=7, filter=None)
        assert client.calls == [
            {
                'knowledgeBaseId': 'KB123',
                'retrievalQuery': {'text': 'q'},
                'retrievalConfiguration': {'vectorSearchConfiguration': {'numberOfResults': 7}},
            }
        ]

    async def test_limit_is_capped_at_the_api_maximum(self) -> None:
        client = _FakeClient()
        await BedrockKnowledgeBase('KB123', client=client).search('q', limit=500, filter=None)
        assert _configuration(client) == {'vectorSearchConfiguration': {'numberOfResults': 100}}

    @pytest.mark.parametrize(
        ('filter', 'expected'),
        [
            ({'tenant_id': 'acme'}, {'equals': {'key': 'tenant_id', 'value': 'acme'}}),
            ({'source': ['manuals', 'faq']}, {'in': {'key': 'source', 'value': ['manuals', 'faq']}}),
            (
                {'tenant_id': 'acme', 'year': 2026},
                {
                    'andAll': [
                        {'equals': {'key': 'tenant_id', 'value': 'acme'}},
                        {'equals': {'key': 'year', 'value': 2026}},
                    ]
                },
            ),
        ],
    )
    async def test_neutral_filter_is_translated(self, filter: MetadataFilter, expected: object) -> None:
        client = _FakeClient()
        await BedrockKnowledgeBase('KB123', client=client).search(
            'q', limit=5, filter=normalize_filter(filter), mode='hybrid'
        )
        assert _configuration(client) == {
            'vectorSearchConfiguration': {'numberOfResults': 5, 'overrideSearchType': 'HYBRID', 'filter': expected}
        }

    async def test_empty_filter_sends_none(self) -> None:
        client = _FakeClient()
        await BedrockKnowledgeBase('KB123', client=client).search('q', limit=5, filter=normalize_filter({}))
        assert _configuration(client) == {'vectorSearchConfiguration': {'numberOfResults': 5}}

    async def test_reranking_fetches_wide_and_keeps_the_limit(self) -> None:
        client = _FakeClient()
        arn = 'arn:aws:bedrock:eu-central-1::foundation-model/amazon.rerank-v1:0'
        backend = BedrockKnowledgeBase('KB123', client=client, reranking_model=arn, fetch_results=50)
        await backend.search('q', limit=5, filter=None)
        assert _configuration(client) == {
            'vectorSearchConfiguration': {
                'numberOfResults': 50,
                'rerankingConfiguration': {
                    'type': 'BEDROCK_RERANKING_MODEL',
                    'bedrockRerankingConfiguration': {
                        'modelConfiguration': {'modelArn': arn},
                        'numberOfRerankedResults': 5,
                    },
                },
            }
        }

    async def test_reranking_model_id_is_expanded_with_the_region(self) -> None:
        client = _FakeClient()
        backend = BedrockKnowledgeBase(
            'KB123', client=client, region_name='eu-central-1', reranking_model='amazon.rerank-v1:0'
        )
        await backend.search('q', limit=5, filter=None)
        configuration = _configuration(client)
        assert configuration == {
            'vectorSearchConfiguration': {
                'numberOfResults': 5,
                'rerankingConfiguration': {
                    'type': 'BEDROCK_RERANKING_MODEL',
                    'bedrockRerankingConfiguration': {
                        'modelConfiguration': {
                            'modelArn': 'arn:aws:bedrock:eu-central-1::foundation-model/amazon.rerank-v1:0'
                        },
                        'numberOfRerankedResults': 5,
                    },
                },
            }
        }

    async def test_reranking_model_id_without_a_region_is_rejected(self) -> None:
        backend = BedrockKnowledgeBase('KB123', client=_FakeClient(), reranking_model='amazon.rerank-v1:0')
        with pytest.raises(UserError, match='full model ARN'):
            await backend.search('q', limit=5, filter=None)

    def test_default_client_region_expands_a_reranking_model_id(self) -> None:
        backend = BedrockKnowledgeBase('KB123', region_name='us-east-1', reranking_model='amazon.rerank-v1:0')
        assert backend.reranking_model_arn == 'arn:aws:bedrock:us-east-1::foundation-model/amazon.rerank-v1:0'

    async def test_retrieval_configuration_is_sent_as_is(self) -> None:
        client = _FakeClient()
        configuration: KnowledgeBaseRetrievalConfigurationTypeDef = {
            'vectorSearchConfiguration': {'numberOfResults': 7, 'overrideSearchType': 'HYBRID'}
        }
        await BedrockKnowledgeBase('KB123', client=client, retrieval_configuration=configuration).search(
            'q', limit=5, filter=None
        )
        assert _configuration(client) == configuration

    async def test_filter_is_combined_with_the_configuration_filter_at_one_level(self) -> None:
        client = _FakeClient()
        configuration: KnowledgeBaseRetrievalConfigurationTypeDef = {
            'vectorSearchConfiguration': {
                'numberOfResults': 7,
                'filter': {
                    'andAll': [
                        {'startsWith': {'key': 'source_url', 'value': 'http'}},
                        {'greaterThan': {'key': 'year', 'value': 2020}},
                    ]
                },
            }
        }
        passed = copy.deepcopy(configuration)
        backend = BedrockKnowledgeBase('KB123', client=client, retrieval_configuration=configuration)
        await backend.search('q', limit=5, filter=normalize_filter({'tenant_id': 'acme'}))
        assert _configuration(client) == {
            'vectorSearchConfiguration': {
                'numberOfResults': 7,
                'filter': {
                    'andAll': [
                        {'startsWith': {'key': 'source_url', 'value': 'http'}},
                        {'greaterThan': {'key': 'year', 'value': 2020}},
                        {'equals': {'key': 'tenant_id', 'value': 'acme'}},
                    ]
                },
            }
        }
        assert configuration == passed, 'the configuration the caller passed was changed'

    async def test_filter_is_set_on_a_configuration_without_one(self) -> None:
        client = _FakeClient()
        configuration: KnowledgeBaseRetrievalConfigurationTypeDef = {
            'vectorSearchConfiguration': {'numberOfResults': 7}
        }
        backend = BedrockKnowledgeBase('KB123', client=client, retrieval_configuration=configuration)
        await backend.search('q', limit=5, filter=normalize_filter({'tenant_id': 'acme'}))
        assert _configuration(client) == {
            'vectorSearchConfiguration': {
                'numberOfResults': 7,
                'filter': {'equals': {'key': 'tenant_id', 'value': 'acme'}},
            }
        }

    async def test_filter_without_a_vector_search_configuration_is_rejected(self) -> None:
        backend = BedrockKnowledgeBase(
            'KB123',
            client=_FakeClient(),
            retrieval_configuration={},  # pyright: ignore[reportArgumentType]  # optional key since boto3-stubs 1.43
        )
        with pytest.raises(UserError, match='vectorSearchConfiguration'):
            await backend.search('q', limit=5, filter=normalize_filter({'tenant_id': 'acme'}))


class TestPassages:
    async def test_results_become_passages_without_service_metadata(self) -> None:
        client = _FakeClient(
            response=_response(
                _result(
                    'alpha',
                    location={'type': 'S3', 's3Location': {'uri': 's3://bucket/a.md'}},
                    metadata={'tenant_id': 'acme', 'x-amz-bedrock-kb-chunk-id': 'chunk-1'},
                ),
                _result('beta', score=None),
            )
        )
        passages = await BedrockKnowledgeBase('KB123', client=client).search('q', limit=5, filter=None)
        assert passages == [
            Passage('alpha', source='s3://bucket/a.md', score=0.87, metadata={'tenant_id': 'acme'}),
            Passage('beta'),
        ]

    @pytest.mark.parametrize(
        ('location', 'source'),
        [
            ({'type': 'WEB', 'webLocation': {'url': 'https://docs/beta'}}, 'https://docs/beta'),
            ({'type': 'CONFLUENCE', 'confluenceLocation': {'url': 'https://wiki/page'}}, 'https://wiki/page'),
            ({'type': 'SALESFORCE', 'salesforceLocation': {'url': 'https://sf/record'}}, 'https://sf/record'),
            ({'type': 'SHAREPOINT', 'sharePointLocation': {'url': 'https://sp/doc'}}, 'https://sp/doc'),
            ({'type': 'KENDRA', 'kendraDocumentLocation': {'uri': 'kendra://doc'}}, 'kendra://doc'),
            ({'type': 'CUSTOM', 'customDocumentLocation': {'id': 'doc-7'}}, 'doc-7'),
            ({'type': 'SQL', 'sqlLocation': {'query': 'SELECT 1'}}, 'SELECT 1'),
            ({'type': 'S3'}, None),
        ],
    )
    async def test_every_location_type_yields_its_source(
        self, location: RetrievalResultLocationTypeDef, source: str | None
    ) -> None:
        client = _FakeClient(response=_response(_result('a', location=location)))
        (passage,) = await BedrockKnowledgeBase('KB123', client=client).search('q', limit=5, filter=None)
        assert passage.source == source

    @pytest.mark.parametrize(
        ('content', 'text'),
        [
            (
                {
                    'type': 'ROW',
                    'row': [
                        {'columnName': 'sku', 'columnValue': 'A-1', 'type': 'STRING'},
                        {'columnName': 'stock', 'columnValue': '3', 'type': 'LONG'},
                        {'columnValue': 'ignored'},
                    ],
                },
                'sku: A-1\nstock: 3',
            ),
            ({'type': 'AUDIO', 'audio': {'s3Uri': 's3://b/a.mp3', 'transcription': 'spoken words'}}, 'spoken words'),
            ({'type': 'VIDEO', 'video': {'s3Uri': 's3://b/v.mp4', 'summary': 'what happens'}}, 'what happens'),
            ({'type': 'AUDIO', 'audio': {'s3Uri': 's3://b/a.mp3'}}, '(no text content)'),
            ({'type': 'TEXT'}, '(no text content)'),
        ],
    )
    async def test_non_text_content_is_rendered_as_text(
        self, content: RetrievalResultContentTypeDef, text: str
    ) -> None:
        client = _FakeClient(response=_response(_result(content=content)))
        (passage,) = await BedrockKnowledgeBase('KB123', client=client).search('q', limit=5, filter=None)
        assert passage.text == text


class TestFilterTree:
    @pytest.mark.parametrize(
        ('filter', 'expected'),
        [
            (
                {'field': 'year', 'operator': '>=', 'value': 2024},
                {'greaterThanOrEquals': {'key': 'year', 'value': 2024}},
            ),
            (
                {
                    'operator': 'AND',
                    'conditions': [
                        {'field': 'tenant_id', 'operator': '==', 'value': 'acme'},
                        {
                            'operator': 'AND',
                            'conditions': [
                                {'field': 'year', 'operator': '>', 'value': 2020},
                                {'field': 'year', 'operator': '<', 'value': 2030},
                            ],
                        },
                        {
                            'operator': 'OR',
                            'conditions': [
                                {'field': 'source', 'operator': '!=', 'value': 'drafts'},
                                {'field': 'tag', 'operator': 'not in', 'value': ['old']},
                                {'field': 'year', 'operator': '<=', 'value': 1999},
                            ],
                        },
                    ],
                },
                {
                    'andAll': [
                        {'equals': {'key': 'tenant_id', 'value': 'acme'}},
                        {'greaterThan': {'key': 'year', 'value': 2020}},
                        {'lessThan': {'key': 'year', 'value': 2030}},
                        {
                            'orAll': [
                                {'notEquals': {'key': 'source', 'value': 'drafts'}},
                                {'notIn': {'key': 'tag', 'value': ['old']}},
                                {'lessThanOrEquals': {'key': 'year', 'value': 1999}},
                            ]
                        },
                    ]
                },
            ),
            (
                {'operator': 'OR', 'conditions': [{'field': 'a', 'operator': '==', 'value': 1}]},
                {'equals': {'key': 'a', 'value': 1}},
            ),
        ],
    )
    async def test_tree_is_translated_flat(self, filter: MetadataFilter, expected: object) -> None:
        client = _FakeClient()
        await BedrockKnowledgeBase('KB123', client=client).search('q', limit=5, filter=normalize_filter(filter))
        assert _configuration(client) == {'vectorSearchConfiguration': {'numberOfResults': 5, 'filter': expected}}

    async def test_not_is_rejected(self) -> None:
        client = _FakeClient()
        not_drafts = normalize_filter(
            {'operator': 'NOT', 'conditions': [{'field': 'source', 'operator': '==', 'value': 'drafts'}]}
        )
        with pytest.raises(UserError, match="can't apply NOT"):
            await BedrockKnowledgeBase('KB123', client=client).search('q', limit=5, filter=not_drafts)
        assert client.calls == []


class TestSearchOptions:
    async def test_semantic_mode_is_sent(self) -> None:
        client = _FakeClient()
        await BedrockKnowledgeBase('KB123', client=client).search('q', limit=5, filter=None, mode='semantic')
        assert _configuration(client) == {
            'vectorSearchConfiguration': {'numberOfResults': 5, 'overrideSearchType': 'SEMANTIC'}
        }

    async def test_keyword_mode_is_rejected(self) -> None:
        with pytest.raises(UserError, match="not 'keyword'"):
            await BedrockKnowledgeBase('KB123', client=_FakeClient()).search('q', limit=5, filter=None, mode='keyword')

    async def test_mode_with_a_retrieval_configuration_is_rejected(self) -> None:
        backend = BedrockKnowledgeBase(
            'KB123', client=_FakeClient(), retrieval_configuration={'vectorSearchConfiguration': {'numberOfResults': 3}}
        )
        with pytest.raises(UserError, match='leave the search mode unset'):
            await backend.search('q', limit=5, filter=None, mode='hybrid')

    async def test_score_threshold_drops_weak_passages_and_keeps_unscored_ones(self) -> None:
        client = _FakeClient(
            response=_response(
                _result('strong', score=0.9), _result('weak', score=0.2), _result('unscored', score=None)
            )
        )
        passages = await BedrockKnowledgeBase('KB123', client=client).search(
            'q', limit=5, filter=None, score_threshold=0.5
        )
        assert [passage.text for passage in passages] == ['strong', 'unscored']


class TestErrors:
    @pytest.mark.parametrize(
        'code',
        [
            'ThrottlingException',
            'ServiceQuotaExceededException',
            'InternalServerException',
            'DependencyFailedException',
            'BadGatewayException',
            'ModelNotReadyException',
        ],
    )
    async def test_transient_service_errors_become_model_retry(self, code: str) -> None:
        backend = BedrockKnowledgeBase('KB123', client=_FakeClient(error=_client_error(code)))
        with pytest.raises(ModelRetry, match=f'Knowledge base search failed: .*{code}'):
            await backend.search('q', limit=5, filter=None)

    @pytest.mark.parametrize('code', ['AccessDeniedException', 'ResourceNotFoundException', 'ValidationException'])
    async def test_configuration_errors_propagate(self, code: str) -> None:
        backend = BedrockKnowledgeBase('KB123', client=_FakeClient(error=_client_error(code)))
        with pytest.raises(ClientError, match=code):
            await backend.search('q', limit=5, filter=None)

    @pytest.mark.parametrize(
        'error',
        [
            EndpointConnectionError(endpoint_url='https://bedrock-agent-runtime.eu-central-1.amazonaws.com'),
            ReadTimeoutError(endpoint_url='https://bedrock-agent-runtime.eu-central-1.amazonaws.com'),
        ],
    )
    async def test_network_failures_become_model_retry(self, error: Exception) -> None:
        backend = BedrockKnowledgeBase('KB123', client=_FakeClient(error=error))
        with pytest.raises(ModelRetry, match='Knowledge base search failed'):
            await backend.search('q', limit=5, filter=None)


class TestConfiguration:
    def test_empty_knowledge_base_id_rejected(self) -> None:
        with pytest.raises(ValueError, match='knowledge_base_id must not be empty'):
            BedrockKnowledgeBase('')

    def test_fetch_results_needs_a_reranking_model(self) -> None:
        with pytest.raises(ValueError, match='fetch_results needs reranking_model'):
            BedrockKnowledgeBase('KB123', fetch_results=20)

    @pytest.mark.parametrize('fetch_results', [0, 101])
    def test_fetch_results_out_of_bounds_rejected(self, fetch_results: int) -> None:
        with pytest.raises(ValueError, match='fetch_results must be between 1 and 100'):
            BedrockKnowledgeBase('KB123', reranking_model='arn:x', fetch_results=fetch_results)

    def test_retrieval_configuration_excludes_the_reranking_model(self) -> None:
        with pytest.raises(ValueError, match='retrieval_configuration replaces reranking_model'):
            BedrockKnowledgeBase(
                'KB123', retrieval_configuration={'vectorSearchConfiguration': {}}, reranking_model='arn:x'
            )

    async def test_default_client_requires_a_region(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        for name in ('AWS_DEFAULT_REGION', 'AWS_REGION', 'AWS_PROFILE'):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv('AWS_CONFIG_FILE', str(tmp_path / 'missing-config'))
        monkeypatch.setenv('AWS_SHARED_CREDENTIALS_FILE', str(tmp_path / 'missing-credentials'))
        with pytest.raises(UserError, match='region_name'):
            await BedrockKnowledgeBase('KB123').search('q', limit=5, filter=None)


class TestWithKnowledgeBase:
    async def test_agent_run_applies_the_run_filter(self) -> None:
        client = _FakeClient(response=_response(_result('alpha')))
        agent = Agent(
            'test',
            deps_type=str,
            capabilities=[
                KnowledgeBase(BedrockKnowledgeBase('KB123', client=client), filter=lambda ctx: {'tenant_id': ctx.deps})
            ],
        )
        await agent.run('Search.', deps='acme')
        assert _configuration(client) == {
            'vectorSearchConfiguration': {
                'numberOfResults': 5,
                'filter': {'equals': {'key': 'tenant_id', 'value': 'acme'}},
            }
        }

    def test_search_only_backend_has_no_read_tool(self) -> None:
        toolset = KnowledgeBase[None](BedrockKnowledgeBase('KB123', client=_FakeClient())).get_toolset()
        assert list(toolset.tools) == ['search_knowledge_base']

    def test_from_spec_builds_a_bedrock_backend(self) -> None:
        capability = KnowledgeBase[None].from_spec(
            backend='bedrock',
            knowledge_base_id='KB123',
            region_name='eu-central-1',
            reranking_model='amazon.rerank-v1:0',
            fetch_results=50,
            search_mode='hybrid',
            score_threshold=0.3,
            number_of_results=3,
            filter={'tenant_id': 'acme'},
            show_sources=False,
            description='manuals',
            id='manuals',
        )
        assert capability.backend == BedrockKnowledgeBase(
            'KB123',
            region_name='eu-central-1',
            reranking_model='amazon.rerank-v1:0',
            fetch_results=50,
        )
        assert (capability.search_mode, capability.score_threshold) == ('hybrid', 0.3)
        assert (capability.number_of_results, capability.filter, capability.show_sources) == (
            3,
            {'tenant_id': 'acme'},
            False,
        )
        assert (capability.description, capability.id) == ('manuals', 'manuals')

    def test_from_spec_reads_source_documents(self) -> None:
        capability = KnowledgeBase[None].from_spec(
            backend='bedrock', knowledge_base_id='KB123', region_name='eu-central-1', read_source_documents=True
        )
        assert capability.backend == BedrockKnowledgeBase(
            'KB123', region_name='eu-central-1', read_source_documents=True
        )

    def test_agent_loads_a_filter_tree_from_a_spec_file(self, tmp_path: Path) -> None:
        spec = tmp_path / 'agent.yaml'
        spec.write_text(
            'model: test\n'
            'capabilities:\n'
            '  - KnowledgeBase:\n'
            '      backend: bedrock\n'
            '      knowledge_base_id: KB123\n'
            '      region_name: eu-central-1\n'
            '      search_mode: hybrid\n'
            '      filter:\n'
            '        operator: AND\n'
            '        conditions:\n'
            '          - {field: tenant_id, operator: "==", value: acme}\n'
            '          - operator: OR\n'
            '            conditions:\n'
            '              - {field: year, operator: ">=", value: 2024}\n'
            '              - {field: source, operator: in, value: [manuals, faq]}\n'
        )
        agent = Agent.from_file(spec, custom_capability_types=[KnowledgeBase])
        (capability,) = [c for c in agent.root_capability.capabilities if isinstance(c, KnowledgeBase)]
        assert normalize_filter(capability.filter) == {  # pyright: ignore[reportArgumentType]
            'operator': 'AND',
            'conditions': [
                {'field': 'tenant_id', 'operator': '==', 'value': 'acme'},
                {
                    'operator': 'OR',
                    'conditions': [
                        {'field': 'year', 'operator': '>=', 'value': 2024},
                        {'field': 'source', 'operator': 'in', 'value': ['manuals', 'faq']},
                    ],
                },
            ],
        }

    def test_agent_loads_from_spec_file(self, tmp_path: Path) -> None:
        spec = tmp_path / 'agent.yaml'
        spec.write_text(
            'model: test\n'
            'capabilities:\n'
            '  - KnowledgeBase:\n'
            '      backend: bedrock\n'
            '      knowledge_base_id: KB123\n'
            '      region_name: eu-central-1\n'
            '      filter:\n'
            '        tenant_id: acme\n'
            '        source: [manuals, faq]\n'
        )
        agent = Agent.from_file(spec, custom_capability_types=[KnowledgeBase])
        assert isinstance(agent, Agent)


_URI = 's3://kb-bucket/ingested/acme/manual.md'
_PDF_URI = 's3://kb-bucket/ingested/acme/manual.pdf'


def _s3_result(uri: str, text: str = 'passage') -> KnowledgeBaseRetrievalResultTypeDef:
    return _result(text, location={'type': 'S3', 's3Location': {'uri': uri}})


def _sidecar(attributes: Mapping[str, object]) -> tuple[bytes, str]:
    return json.dumps({'metadataAttributes': attributes}).encode(), 'application/json'


@dataclass
class _FakeS3:
    """In-memory `S3Client`: objects by bucket and key, recorded keys, an optional error."""

    objects: dict[tuple[str, str], tuple[bytes, str | None]] = field(
        default_factory=dict[tuple[str, str], tuple[bytes, 'str | None']]
    )
    error: Exception | None = None
    keys: list[str] = field(default_factory=list[str])

    def get_object(self, *, Bucket: str, Key: str) -> Mapping[str, Any]:
        self.keys.append(Key)
        if self.error is not None:
            raise self.error
        if (Bucket, Key) not in self.objects:
            raise ClientError({'Error': {'Code': 'NoSuchKey', 'Message': 'missing'}}, 'GetObject')
        data, content_type = self.objects[(Bucket, Key)]
        response: dict[str, Any] = {'Body': io.BytesIO(data)}
        if content_type is not None:
            response['ContentType'] = content_type
        return response


def _reader(s3: _FakeS3, *results: KnowledgeBaseRetrievalResultTypeDef, **options: Any) -> BedrockKnowledgeBase:
    return BedrockKnowledgeBase(
        'KB123', client=_FakeClient(response=_response(*results)), s3_client=s3, read_source_documents=True, **options
    )


async def _document_id(backend: BedrockKnowledgeBase) -> str:
    (passage, *_) = await backend.search('q', limit=5, filter=None)
    assert passage.document_id is not None
    return passage.document_id


_ACME = normalize_filter({'tenant_id': 'acme'})


class TestReadSourceDocuments:
    async def test_s3_passages_get_opaque_ids_and_others_none(self) -> None:
        backend = _reader(
            _FakeS3(),
            _s3_result(_URI),
            _result('web', location={'type': 'WEB', 'webLocation': {'url': 'https://example.com'}}),
            _result('s3 without uri', location={'type': 'S3', 's3Location': {}}),
        )
        first, web, no_uri = await backend.search('q', limit=5, filter=None)
        assert first.document_id is not None and first.document_id.startswith('doc_')
        assert 'acme' not in first.document_id and 'kb-bucket' not in first.document_id
        assert (web.document_id, no_uri.document_id) == (None, None)
        again, *_ = await backend.search('q', limit=5, filter=None)
        assert again.document_id == first.document_id

    async def test_without_the_option_passages_have_no_ids_and_no_read_tool(self) -> None:
        backend = BedrockKnowledgeBase('KB123', client=_FakeClient(response=_response(_s3_result(_URI))))
        (passage,) = await backend.search('q', limit=5, filter=None)
        assert passage.document_id is None
        assert list(KnowledgeBase[None](backend).get_toolset().tools) == ['search_knowledge_base']
        assert list(KnowledgeBase[None](_reader(_FakeS3())).get_toolset().tools) == [
            'search_knowledge_base',
            'read_document',
        ]

    async def test_text_source_within_the_filter_is_read_with_its_attributes(self) -> None:
        s3 = _FakeS3(
            {
                ('kb-bucket', 'ingested/acme/manual.md'): (b'# Manual\n\nFull text.', 'text/markdown; charset=utf-8'),
                ('kb-bucket', 'ingested/acme/manual.md.metadata.json'): _sidecar({'tenant_id': 'acme', 'year': 2025}),
            }
        )
        backend = _reader(s3, _s3_result(_URI))
        document = await backend.read(await _document_id(backend), max_chars=100, filter=_ACME)
        assert document is not None
        assert (document.text, document.source, dict(document.metadata)) == (
            '# Manual\n\nFull text.',
            _URI,
            {'tenant_id': 'acme', 'year': 2025},
        )

    async def test_typed_sidecar_values_are_understood(self) -> None:
        sidecar = {
            'tenant_id': {'value': {'type': 'STRING', 'stringValue': 'acme'}, 'includeForEmbedding': True},
            'year': {'value': {'type': 'NUMBER', 'numberValue': 2025}},
            'tags': {'value': {'type': 'STRING_LIST', 'stringListValue': ['a']}},
            'labels': ['a', 'b'],
        }
        s3 = _FakeS3(
            {
                ('kb-bucket', 'ingested/acme/manual.md'): (b'text', 'text/plain'),
                ('kb-bucket', 'ingested/acme/manual.md.metadata.json'): _sidecar(sidecar),
            }
        )
        backend = _reader(s3, _s3_result(_URI))
        document = await backend.read(await _document_id(backend), max_chars=100, filter=_ACME)
        assert document is not None and dict(document.metadata) == {'tenant_id': 'acme', 'year': 2025}

    async def test_source_outside_the_filter_is_never_fetched(self) -> None:
        s3 = _FakeS3(
            {
                ('kb-bucket', 'ingested/acme/manual.md'): (b'secret', 'text/plain'),
                ('kb-bucket', 'ingested/acme/manual.md.metadata.json'): _sidecar({'tenant_id': 'globex'}),
            }
        )
        backend = _reader(s3, _s3_result(_URI))
        assert await backend.read(await _document_id(backend), max_chars=100, filter=_ACME) is None
        assert s3.keys == ['ingested/acme/manual.md.metadata.json']

    @pytest.mark.parametrize('sidecar', [b'not json', b'[1, 2]', b'{"metadataAttributes": "x"}'])
    async def test_unreadable_sidecar_matches_no_filter(self, sidecar: bytes) -> None:
        s3 = _FakeS3(
            {
                ('kb-bucket', 'ingested/acme/manual.md'): (b'text', 'text/plain'),
                ('kb-bucket', 'ingested/acme/manual.md.metadata.json'): (sidecar, 'application/json'),
            }
        )
        backend = _reader(s3, _s3_result(_URI))
        document_id = await _document_id(backend)
        assert await backend.read(document_id, max_chars=100, filter=_ACME) is None
        assert await backend.read(document_id, max_chars=100, filter=None) is not None

    async def test_missing_sidecar_blocks_a_filtered_read_only(self) -> None:
        s3 = _FakeS3({('kb-bucket', 'ingested/acme/manual.md'): (b'text', None)})
        backend = _reader(s3, _s3_result(_URI))
        document_id = await _document_id(backend)
        assert await backend.read(document_id, max_chars=100, filter=_ACME) is None
        document = await backend.read(document_id, max_chars=100, filter=None)
        assert document == Document(document_id, 'text', source=_URI)

    async def test_unknown_id_and_missing_object_are_none(self) -> None:
        s3 = _FakeS3({('kb-bucket', 'ingested/acme/manual.md.metadata.json'): _sidecar({'tenant_id': 'acme'})})
        backend = _reader(s3, _s3_result(_URI))
        assert await backend.read('doc_unknown', max_chars=100, filter=None) is None
        assert await backend.read(await _document_id(backend), max_chars=100, filter=_ACME) is None

    async def test_pdf_needs_a_converter(self) -> None:
        objects = {
            ('kb-bucket', 'ingested/acme/manual.pdf'): (b'%PDF-1.7 binary', 'application/pdf'),
            ('kb-bucket', 'ingested/acme/manual.pdf.metadata.json'): _sidecar({'tenant_id': 'acme'}),
        }
        plain = _reader(_FakeS3(dict(objects)), _s3_result(_PDF_URI))
        assert await plain.read(await _document_id(plain), max_chars=100, filter=_ACME) is None

        seen: list[tuple[bytes, str | None, str]] = []

        def to_text(data: bytes, content_type: str | None, uri: str) -> str | None:
            seen.append((data, content_type, uri))
            return 'extracted text'

        converting = _reader(_FakeS3(dict(objects)), _s3_result(_PDF_URI), convert=to_text)
        document = await converting.read(await _document_id(converting), max_chars=100, filter=_ACME)
        assert document is not None and document.text == 'extracted text'
        assert seen == [(b'%PDF-1.7 binary', 'application/pdf', _PDF_URI)]

        def decline(data: bytes, content_type: str | None, uri: str) -> str | None:
            return None

        declining = _reader(_FakeS3(dict(objects)), _s3_result(_PDF_URI), convert=decline)
        assert await declining.read(await _document_id(declining), max_chars=100, filter=_ACME) is None

    @pytest.mark.parametrize(
        ('key', 'content_type'),
        [('doc.json', 'application/json'), ('doc.md', 'binary/octet-stream'), ('doc.csv', None)],
    )
    async def test_text_is_recognised_by_type_or_suffix(self, key: str, content_type: str | None) -> None:
        uri = f's3://kb-bucket/{key}'
        backend = _reader(_FakeS3({('kb-bucket', key): (b'{"a": 1}', content_type)}), _s3_result(uri))
        document = await backend.read(await _document_id(backend), max_chars=100, filter=None)
        assert document is not None and document.text == '{"a": 1}'

    @pytest.mark.parametrize('code', ['SlowDown', 'InternalError'])
    async def test_transient_s3_errors_become_model_retry(self, code: str) -> None:
        s3 = _FakeS3(error=ClientError({'Error': {'Code': code, 'Message': 'x'}}, 'GetObject'))
        backend = _reader(s3, _s3_result(_URI))
        with pytest.raises(ModelRetry, match='Reading the document failed'):
            await backend.read(await _document_id(backend), max_chars=100, filter=None)

    async def test_network_failures_become_model_retry(self) -> None:
        s3 = _FakeS3(error=EndpointConnectionError(endpoint_url='https://s3.amazonaws.com'))
        backend = _reader(s3, _s3_result(_URI))
        with pytest.raises(ModelRetry, match='Reading the document failed'):
            await backend.read(await _document_id(backend), max_chars=100, filter=None)

    async def test_access_denied_propagates(self) -> None:
        s3 = _FakeS3(error=ClientError({'Error': {'Code': 'AccessDenied', 'Message': 'no'}}, 'GetObject'))
        backend = _reader(s3, _s3_result(_URI))
        with pytest.raises(ClientError, match='AccessDenied'):
            await backend.read(await _document_id(backend), max_chars=100, filter=None)

    async def test_remembered_ids_are_bounded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr('pydantic_ai_harness.knowledge_base._bedrock._MAX_REMEMBERED_DOCUMENTS', 2)
        uris = [f's3://kb-bucket/doc-{index}.md' for index in range(3)]
        objects: dict[tuple[str, str], tuple[bytes, str | None]] = {
            ('kb-bucket', uri.removeprefix('s3://kb-bucket/')): (b'text', 'text/plain') for uri in uris
        }
        backend = _reader(_FakeS3(objects), *(_s3_result(uri) for uri in uris))
        ids = [passage.document_id for passage in await backend.search('q', limit=5, filter=None)]
        assert ids[0] is not None and ids[2] is not None
        assert await backend.read(ids[0], max_chars=100, filter=None) is None
        assert await backend.read(ids[2], max_chars=100, filter=None) is not None

    def test_convert_needs_the_option(self) -> None:
        def to_text(data: bytes, content_type: str | None, uri: str) -> str | None:
            return None  # pragma: no cover

        with pytest.raises(ValueError, match='convert needs read_source_documents=True'):
            BedrockKnowledgeBase('KB123', convert=to_text)

    async def test_default_s3_client_is_built_for_the_region(self, monkeypatch: pytest.MonkeyPatch) -> None:
        built: list[tuple[str | None, str]] = []
        s3 = _FakeS3({('kb-bucket', 'ingested/acme/manual.md'): (b'text', 'text/plain')})

        class _Session:
            def __init__(self, region_name: str | None = None) -> None:
                self.region_name = region_name

            def client(self, service: str, **kwargs: object) -> _FakeS3:
                built.append((self.region_name, service))
                return s3

        monkeypatch.setattr('pydantic_ai_harness.knowledge_base._bedrock.boto3.Session', _Session)
        backend = BedrockKnowledgeBase(
            'KB123',
            region_name='eu-central-1',
            client=_FakeClient(response=_response(_s3_result(_URI))),
            read_source_documents=True,
        )
        document = await backend.read(await _document_id(backend), max_chars=100, filter=None)
        assert document is not None and built == [('eu-central-1', 's3')]

    async def test_agent_reads_the_source_through_read_document(self) -> None:
        s3 = _FakeS3(
            {
                ('kb-bucket', 'ingested/acme/manual.md'): (b'Full manual text.', 'text/markdown'),
                ('kb-bucket', 'ingested/acme/manual.md.metadata.json'): _sidecar({'tenant_id': 'acme'}),
            }
        )
        backend = _reader(s3, _s3_result(_URI))
        toolset = KnowledgeBase[None](backend, filter={'tenant_id': 'acme'}, show_sources=False).get_toolset()
        from pydantic_ai.models.test import TestModel
        from pydantic_ai.tools import RunContext
        from pydantic_ai.usage import RunUsage

        ctx = RunContext[None](deps=None, model=TestModel(), usage=RunUsage())
        found = await toolset.search_knowledge_base(ctx, 'manual')
        text = str(found.return_value)
        assert 's3://' not in text and 'kb-bucket' not in text
        document_id = found.metadata['results'][0]['document_id']
        read = await toolset.read_document(ctx, document_id)
        assert str(read.return_value) == f'Document: {document_id}\n\nFull manual text.'
