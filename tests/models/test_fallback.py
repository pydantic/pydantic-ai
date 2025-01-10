from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from pydantic_ai.models import AgentModel, Model
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.tools import ToolDefinition

pytestmark = pytest.mark.anyio


class MockModel(Model):
    def __init__(self, name_str: str, agent_model_mock: Optional[AsyncMock] = None):
        self._name = name_str
        self._agent_model = agent_model_mock if agent_model_mock is not None else AsyncMock()

    async def agent_model(
        self,
        *,
        function_tools: list[ToolDefinition],
        allow_text_result: bool,
        result_tools: list[ToolDefinition],
    ) -> AgentModel:
        return await self._agent_model(
            function_tools=function_tools, allow_text_result=allow_text_result, result_tools=result_tools
        )

    def name(self) -> str:
        return self._name


class ServerError(Exception):
    def __init__(self, status_code: int):
        self.status_code = status_code
        super().__init__(f'Server error with status code {status_code}')


@pytest.fixture
def function_tools() -> list[ToolDefinition]:
    return [MagicMock()]


@pytest.fixture
def result_tools() -> list[ToolDefinition]:
    return [MagicMock()]


async def test_successful_first_model():
    mock_agent_model = AsyncMock()
    mock_result = MagicMock()
    mock_agent_model.return_value = mock_result

    model1 = MockModel('model1', mock_agent_model)
    model2 = MockModel('model2', AsyncMock())

    fallback = FallbackModel(models=[model1, model2])

    result = await fallback.agent_model(function_tools=[], allow_text_result=True, result_tools=[])

    assert result == mock_result
    mock_agent_model.assert_called_once()
    assert fallback.name() == 'Fallback[model1, model2]'


async def test_fallback_to_second_model():
    mock_error_model = AsyncMock()
    mock_error_model.side_effect = ServerError(500)

    mock_success_model = AsyncMock()
    mock_result = MagicMock()
    mock_success_model.return_value = mock_result

    model1 = MockModel('model1', mock_error_model)
    model2 = MockModel('model2', mock_success_model)

    fallback = FallbackModel(models=[model1, model2])

    result = await fallback.agent_model(function_tools=[], allow_text_result=True, result_tools=[])

    assert result == mock_result
    mock_error_model.assert_called_once()
    mock_success_model.assert_called_once()


async def test_non_server_error_raises():
    mock_model = AsyncMock()
    mock_model.side_effect = ValueError('Non-server error')

    model = MockModel('model1', mock_model)
    fallback = FallbackModel(models=[model])

    with pytest.raises(ValueError, match='Non-server error'):
        await fallback.agent_model(function_tools=[], allow_text_result=True, result_tools=[])


async def test_all_models_fail():
    mock_model1 = AsyncMock()
    mock_model1.side_effect = ServerError(500)

    mock_model2 = AsyncMock()
    mock_model2.side_effect = ServerError(502)

    model1 = MockModel('model1', mock_model1)
    model2 = MockModel('model2', mock_model2)

    fallback = FallbackModel(models=[model1, model2])

    with pytest.raises(Exception) as exc_info:
        await fallback.agent_model(function_tools=[], allow_text_result=True, result_tools=[])

    assert 'All models failed with server errors' in str(exc_info.value)
    mock_model1.assert_called_once()
    mock_model2.assert_called_once()


async def test_empty_model_list():
    fallback = FallbackModel(models=[])

    with pytest.raises(Exception) as exc_info:
        await fallback.agent_model(function_tools=[], allow_text_result=True, result_tools=[])

    assert 'All models failed with server errors' in str(exc_info.value)


def test_name_generation():
    model1 = MockModel('model1')
    model2 = MockModel('model2')
    model3 = MockModel('model3')

    fallback = FallbackModel(models=[model1, model2, model3])
    expected_name = 'Fallback[model1, model2, model3]'

    assert fallback.name() == expected_name
