import pytest
from dirty_equals import IsList

from pydantic_ai.embeddings import Embedder, infer_model

from .conftest import try_import

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.vcr,
]

with try_import() as openai_imports_successful:
    from pydantic_ai.embeddings.openai import OpenAIEmbeddingModel
    from pydantic_ai.providers.openai import OpenAIProvider

with try_import() as cohere_imports_successful:
    from pydantic_ai.embeddings.cohere import CohereEmbeddingModel
    from pydantic_ai.providers.cohere import CohereProvider


@pytest.mark.skipif(not openai_imports_successful, reason='OpenAI not installed')
class TestOpenAI:
    async def test_infer_model(self, openai_api_key: str):
        model = infer_model('openai:text-embedding-3-small')
        assert isinstance(model, OpenAIEmbeddingModel)
        assert model.model_name == 'text-embedding-3-small'
        assert model.system == 'openai'
        assert model.base_url == 'https://api.openai.com/v1/'

    async def test_infer_model_azure(self, openai_api_key: str):
        model = infer_model('azure:text-embedding-3-small')
        assert isinstance(model, OpenAIEmbeddingModel)
        assert model.model_name == 'text-embedding-3-small'
        assert model.system == 'azure'
        assert 'azure.com' in model.base_url

    async def test_single(self, openai_api_key: str):
        model = OpenAIEmbeddingModel('text-embedding-3-small', provider=OpenAIProvider(api_key=openai_api_key))
        embedder = Embedder(model)
        embeddings = await embedder.embed('Hello, world!')
        assert embeddings == IsList(-0.019143931567668915, -0.025292053818702698, -0.0017211713129654527, length=1536)

    async def test_bulk(self, openai_api_key: str):
        model = OpenAIEmbeddingModel('text-embedding-3-small', provider=OpenAIProvider(api_key=openai_api_key))
        embedder = Embedder(model)
        embeddings = await embedder.embed(['hello', 'world'])
        assert embeddings == IsList(
            IsList(0.016751619055867195, -0.055799614638090134, 0.005647437181323767, length=1536),
            IsList(-0.010633519850671291, -0.03604777529835701, 0.03019288368523121, length=1536),
            length=2,
        )


@pytest.mark.skipif(not cohere_imports_successful, reason='Cohere not installed')
class TestCohere:
    async def test_infer_model(self, co_api_key: str):
        model = infer_model('cohere:embed-v4.0')
        assert isinstance(model, CohereEmbeddingModel)
        assert model.model_name == 'embed-v4.0'
        assert model.system == 'cohere'
        assert model.base_url == 'https://api.cohere.com'
        assert isinstance(model._provider, CohereProvider)  # type: ignore[reportAttributeAccess]

    async def test_single(self, co_api_key: str):
        model = CohereEmbeddingModel('embed-v4.0', provider=CohereProvider(api_key=co_api_key))
        embedder = Embedder(model)
        embeddings = await embedder.embed('Hello, world!')
        assert embeddings == IsList(-0.016547563, 0.026550192, 0.0044764862, length=1536)

    async def test_bulk(self, co_api_key: str):
        model = CohereEmbeddingModel('embed-v4.0', provider=CohereProvider(api_key=co_api_key))
        embedder = Embedder(model)
        embeddings = await embedder.embed(['hello', 'world'])
        assert embeddings == IsList(
            IsList(0.015943069, 0.013248466, 0.0024139155, length=1536),
            IsList(-0.0060736495, -0.015005487, 0.00033246286, length=1536),
            length=2,
        )
