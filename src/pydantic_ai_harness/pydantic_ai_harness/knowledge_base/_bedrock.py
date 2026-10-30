"""Amazon Bedrock Knowledge Base backend: searches a knowledge base with the Bedrock `Retrieve` API.

External assumptions (verified 2026-10-30 against botocore 1.42.63 and the API reference,
<https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent-runtime_Retrieve.html>):

- `Retrieve` takes `retrievalConfiguration.vectorSearchConfiguration` with `numberOfResults` (1 to 100),
  `overrideSearchType` (`HYBRID` or `SEMANTIC`), `filter`, `rerankingConfiguration` and
  `implicitFilterConfiguration`. boto3 1.43 adds `managedSearchConfiguration` for managed knowledge
  bases; `retrieval_configuration` passes it through on a boto3 that has it.
- A filter's `andAll` and `orAll` lists need at least two entries, and an `andAll` directly inside an
  `andAll` is rejected with a `ValidationException`; filters are combined at one level here. An
  `orAll` inside an `andAll` is accepted. There is no `not` operator.
- The negative operators (`notEquals` and its list form) match a document that lacks the key (checked live 2026-10-30).
- `overrideSearchType` is `HYBRID` or `SEMANTIC`; there is no keyword-only search and no score
  threshold, so a threshold is applied to the returned scores.
- Filter values and result metadata values are the API's generic "document" type: scalars and lists of
  strings at runtime, which boto3-stubs type as a mapping. `_as_boto_filter` is the one crossover.
- Result metadata keys starting with `x-amz-bedrock-kb-` (`source-uri`, `chunk-id`, `data-source-id`)
  are the service's own and are dropped from passages.
- A reranking model is addressed by ARN (`arn:aws:bedrock:<region>::foundation-model/<model id>`), see
  <https://docs.aws.amazon.com/bedrock/latest/userguide/rerank-supported.html>.
- There is no API to read a whole document back, so this backend only searches.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias

import anyio.to_thread
from typing_extensions import Unpack

from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai_harness.knowledge_base._backend import FilterNode, Passage, SearchMode

try:
    import boto3
    from botocore.config import Config
    from botocore.exceptions import (
        ClientError,
        ConnectionError as BotocoreConnectionError,
        NoRegionError,
        ReadTimeoutError,
    )
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'boto3 is required for BedrockKnowledgeBase. Install it with: pip install "pydantic-ai-slim[bedrock]"'
    ) from _import_error

if TYPE_CHECKING:
    from mypy_boto3_bedrock_agent_runtime.type_defs import (
        KnowledgeBaseRetrievalConfigurationTypeDef,
        KnowledgeBaseRetrievalResultTypeDef,
        KnowledgeBaseVectorSearchConfigurationTypeDef,
        RetrievalFilterTypeDef,
        RetrievalResultContentTypeDef,
        RetrievalResultLocationTypeDef,
        RetrieveRequestTypeDef,
        RetrieveResponseTypeDef,
        VectorSearchBedrockRerankingConfigurationTypeDef,
    )

BEDROCK_MAX_NUMBER_OF_RESULTS = 100
"""Largest `numberOfResults` (and `numberOfRerankedResults`) the `Retrieve` API accepts."""

_USER_AGENT = 'pydantic-ai-harness/knowledge-base'
_SERVICE_METADATA_PREFIX = 'x-amz-bedrock-kb-'
_NO_TEXT = '(no text content)'

# Service states that pass on their own, so the model can search again. Access, resource and
# validation errors are configuration the model cannot correct, and those raise.
_RECOVERABLE_ERROR_CODES = frozenset(
    {
        'ThrottlingException',
        'ServiceQuotaExceededException',
        'InternalServerException',
        'DependencyFailedException',
        'BadGatewayException',
        'ModelNotReadyException',
    }
)


class BedrockAgentRuntimeClient(Protocol):
    """The one call `BedrockKnowledgeBase` makes on a boto3 `bedrock-agent-runtime` client.

    A client from `boto3.client('bedrock-agent-runtime', ...)` satisfies it as-is. Pass one to set a
    profile, credentials or an endpoint explicitly, or to substitute a fake in tests.
    """

    def retrieve(self, **kwargs: Unpack[RetrieveRequestTypeDef]) -> RetrieveResponseTypeDef:
        """Query the knowledge base; the `Retrieve` API."""
        ...  # pragma: no cover


@dataclass
class BedrockKnowledgeBase:
    """Searches an [Amazon Bedrock Knowledge Base](https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base.html) with its `Retrieve` API.

    Chunking, embedding and ingestion stay with the knowledge base. Credentials come from the usual
    boto3 sources (environment, profile, instance role); pass `client` to configure them explicitly.
    The `Retrieve` call runs on a worker thread, since boto3 is synchronous.
    """

    knowledge_base_id: str
    """The knowledge base to search, the id shown in the Bedrock console."""

    region_name: str | None = None
    """The AWS region of the knowledge base; `None` takes it from the ambient AWS configuration."""

    reranking_model: str | None = None
    """A Bedrock reranking model that reorders the retrieved passages, as a model ARN or a model id.

    A bare id such as `'amazon.rerank-v1:0'` resolves to an ARN in `region_name`, or in the default
    client's region. With a `client` you pass, set `region_name` too or give the full ARN.
    """

    fetch_results: int | None = None
    """With `reranking_model`, how many passages the vector search fetches before the reranker keeps the best.

    `None` fetches as many as the search asks for. Fetch wide and let the reranker cut: the knowledge
    base capability's `number_of_results` is how many come back.
    """

    retrieval_configuration: KnowledgeBaseRetrievalConfigurationTypeDef | None = None
    """A complete `retrievalConfiguration` sent as-is, for options the fields above don't cover.

    Use it for implicit filtering, reranker metadata selection, Bedrock-native filters (ranges,
    `startsWith`), or a managed knowledge base's `managedSearchConfiguration` (boto3 1.43 or later).
    It replaces `reranking_model`, `fetch_results` and the search's mode, and its result count wins
    over the search's. A search's metadata filter is still applied, combined with any filter it carries.
    """

    client: BedrockAgentRuntimeClient | None = field(default=None, repr=False)
    """A boto3 `bedrock-agent-runtime` client; when `None`, one is built from `region_name` and the ambient AWS configuration."""

    def __post_init__(self) -> None:
        """Validate configuration against the `Retrieve` API's documented bounds."""
        if not self.knowledge_base_id:
            raise ValueError('knowledge_base_id must not be empty')
        if self.fetch_results is not None:
            if self.reranking_model is None:
                raise ValueError('fetch_results needs reranking_model')
            if not 1 <= self.fetch_results <= BEDROCK_MAX_NUMBER_OF_RESULTS:
                raise ValueError(
                    f'fetch_results must be between 1 and {BEDROCK_MAX_NUMBER_OF_RESULTS}, got {self.fetch_results}'
                )
        if self.retrieval_configuration is not None and self.reranking_model is not None:
            raise ValueError('retrieval_configuration replaces reranking_model; set one or the other.')

    @cached_property
    def _client_and_region(self) -> tuple[BedrockAgentRuntimeClient, str | None]:
        if self.client is not None:
            return self.client, self.region_name
        # A fresh session, so the region and credentials are read when the backend is first used.
        session = boto3.Session(region_name=self.region_name)
        try:
            client = session.client(  # pyright: ignore[reportUnknownMemberType]  # boto3-stubs leave other services' overloads unknown
                'bedrock-agent-runtime', config=Config(user_agent_extra=_USER_AGENT)
            )
        except NoRegionError as exc:
            raise UserError(
                'BedrockKnowledgeBase needs an AWS region: pass `region_name`, set `AWS_DEFAULT_REGION`, '
                'or pass a configured `client`.'
            ) from exc
        return client, client.meta.region_name

    @property
    def reranking_model_arn(self) -> str | None:
        """The reranking model's ARN, resolved from `reranking_model` and the region."""
        model = self.reranking_model
        if model is None or model.startswith('arn:'):
            return model
        _, region = self._client_and_region
        if region is None:
            raise UserError(
                f'reranking_model {model!r} needs a region to resolve in: pass the full model ARN, or set `region_name`.'
            )
        return f'arn:aws:bedrock:{region}::foundation-model/{model}'

    async def search(
        self,
        query: str,
        *,
        limit: int,
        filter: FilterNode | None,
        mode: SearchMode | None = None,
        score_threshold: float | None = None,
    ) -> Sequence[Passage]:
        """Run one `Retrieve` request and turn its results into passages.

        `mode` is `'semantic'` or `'hybrid'` (hybrid needs an OpenSearch Serverless store with a
        filterable text field); `None` lets Bedrock choose. `score_threshold` drops returned passages
        scoring below it, on Bedrock's 0 to 1 scale.
        """
        if mode == 'keyword':
            raise UserError("BedrockKnowledgeBase supports mode='semantic' or 'hybrid', not 'keyword'")
        if mode is not None and self.retrieval_configuration is not None:
            raise UserError('retrieval_configuration sets the search type itself; leave the search mode unset')
        client, _ = self._client_and_region
        call = functools.partial(
            client.retrieve,
            knowledgeBaseId=self.knowledge_base_id,
            retrievalQuery={'text': query},
            retrievalConfiguration=self._configuration(min(limit, BEDROCK_MAX_NUMBER_OF_RESULTS), filter, mode),
        )
        try:
            response = await anyio.to_thread.run_sync(call)
        except ClientError as error:
            if error.response.get('Error', {}).get('Code') not in _RECOVERABLE_ERROR_CODES:
                raise
            raise ModelRetry(f'Knowledge base search failed: {error}') from error
        except (BotocoreConnectionError, ReadTimeoutError) as error:
            raise ModelRetry(f'Knowledge base search failed: {error}') from error
        passages = [_passage(result) for result in response['retrievalResults']]
        if score_threshold is None:
            return passages
        return [passage for passage in passages if passage.score is None or passage.score >= score_threshold]

    def _configuration(
        self, limit: int, filter: FilterNode | None, mode: SearchMode | None
    ) -> KnowledgeBaseRetrievalConfigurationTypeDef:
        native = to_bedrock_filter(filter)
        if self.retrieval_configuration is not None:
            if native is None:
                return self.retrieval_configuration
            # A membership test: `vectorSearchConfiguration` is optional since managed search (boto3 1.43).
            if 'vectorSearchConfiguration' not in self.retrieval_configuration:
                raise UserError(
                    'A metadata filter needs a `vectorSearchConfiguration` in retrieval_configuration to be set on.'
                )
            configured = self.retrieval_configuration['vectorSearchConfiguration']
            existing = configured.get('filter')
            combined = native if existing is None else _all_of([_from_boto_filter(existing), native])
            return {'vectorSearchConfiguration': {**configured, 'filter': _as_boto_filter(combined)}}
        vector: KnowledgeBaseVectorSearchConfigurationTypeDef = {'numberOfResults': limit}
        if mode is not None:
            vector['overrideSearchType'] = 'HYBRID' if mode == 'hybrid' else 'SEMANTIC'
        if native is not None:
            vector['filter'] = _as_boto_filter(native)
        if (arn := self.reranking_model_arn) is not None:
            if self.fetch_results is not None:
                vector['numberOfResults'] = self.fetch_results
            reranking: VectorSearchBedrockRerankingConfigurationTypeDef = {
                'modelConfiguration': {'modelArn': arn},
                'numberOfRerankedResults': limit,
            }
            vector['rerankingConfiguration'] = {
                'type': 'BEDROCK_RERANKING_MODEL',
                'bedrockRerankingConfiguration': reranking,
            }
        return {'vectorSearchConfiguration': vector}


BedrockFilter: TypeAlias = dict[str, Any]
"""A Bedrock `RetrievalFilter` at runtime: one operator key with a `{key, value}` attribute, or `andAll` / `orAll`."""


_OPERATORS = {
    '==': 'equals',
    '!=': 'notEquals',
    '>': 'greaterThan',
    '>=': 'greaterThanOrEquals',
    '<': 'lessThan',
    '<=': 'lessThanOrEquals',
    'in': 'in',
    'not in': 'notIn',
}


def to_bedrock_filter(filter: FilterNode | None) -> BedrockFilter | None:
    """Translate a filter tree: comparisons map one to one, `AND` to `andAll` and `OR` to `orAll`.

    Raises:
        UserError: The filter uses `NOT`, which Bedrock has no operator for. Negate the comparisons
            instead (`!=`, `not in`), which Bedrock evaluates the same way for a missing field.
    """
    return None if filter is None else _translate(filter)


def _translate(node: FilterNode) -> BedrockFilter:
    if 'conditions' not in node:
        return {_OPERATORS[node['operator']]: {'key': node['field'], 'value': node['value']}}
    if node['operator'] == 'NOT':
        raise UserError("BedrockKnowledgeBase can't apply NOT; negate the conditions instead, with '!=' or 'not in'")
    children = [_translate(child) for child in node['conditions']]
    return _all_of(children) if node['operator'] == 'AND' else _combine('orAll', children)


def _combine(key: str, clauses: list[BedrockFilter]) -> BedrockFilter:
    """Join clauses under `key` at one level, folding in clauses already joined the same way.

    Bedrock wants two or more clauses in a list and rejects an `andAll` nested in an `andAll`.
    """
    flat: list[BedrockFilter] = []
    for clause in clauses:
        nested: list[BedrockFilter] | None = clause.get(key)
        flat.extend(nested if nested is not None else [clause])
    return flat[0] if len(flat) == 1 else {key: flat}


def _all_of(clauses: list[BedrockFilter]) -> BedrockFilter:
    return _combine('andAll', clauses)


def _as_boto_filter(filter: BedrockFilter) -> RetrievalFilterTypeDef:
    """Hand a filter to boto3 unchanged: the stubs type its document values as mappings, the service takes scalars."""
    return filter  # pyright: ignore[reportReturnType]


def _from_boto_filter(filter: RetrievalFilterTypeDef) -> BedrockFilter:
    return dict(filter)


def _one_line(text: str) -> str:
    return ' '.join(text.split())


def _passage(result: KnowledgeBaseRetrievalResultTypeDef) -> Passage:
    metadata = result.get('metadata') or {}
    return Passage(
        text=_body(result['content']),
        source=_source(result.get('location')),
        score=result.get('score'),
        metadata={key: value for key, value in metadata.items() if not key.startswith(_SERVICE_METADATA_PREFIX)},
    )


def _source(location: RetrievalResultLocationTypeDef | None) -> str | None:
    """The one location a result reports, as a string, whichever data source kind it came from."""
    if location is None:
        return None
    if (s3 := location.get('s3Location')) is not None:
        return s3.get('uri')
    if (web := location.get('webLocation')) is not None:
        return web.get('url')
    if (confluence := location.get('confluenceLocation')) is not None:
        return confluence.get('url')
    if (salesforce := location.get('salesforceLocation')) is not None:
        return salesforce.get('url')
    if (sharepoint := location.get('sharePointLocation')) is not None:
        return sharepoint.get('url')
    if (kendra := location.get('kendraDocumentLocation')) is not None:
        return kendra.get('uri')
    if (custom := location.get('customDocumentLocation')) is not None:
        return custom.get('id')
    if (sql := location.get('sqlLocation')) is not None:
        return sql.get('query')
    return None


def _body(content: RetrievalResultContentTypeDef) -> str:
    """A passage's text: the chunk text, a structured row as `column: value` lines, or a media transcript or summary."""
    if text := content.get('text'):
        return text
    row = content.get('row') or []
    lines = [
        f'{_one_line(column["columnName"])}: {_one_line(column.get("columnValue", ""))}'
        for column in row
        if 'columnName' in column
    ]
    if lines:
        return '\n'.join(lines)
    audio = content.get('audio')
    video = content.get('video')
    transcription = audio.get('transcription') if audio is not None else None
    summary = video.get('summary') if video is not None else None
    return transcription or summary or _NO_TEXT
