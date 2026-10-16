from __future__ import annotations

import pytest

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from .mock_openai import MockOpenAI, completion_message

with pytest.importorskip("openai"):
    from openai.types.chat.chat_completion_message import ChatCompletionMessage


@pytest.mark.vcr
def test_unrecognized_finish_reason_is_quarantined_to_provider_details() -> None:
    completion = completion_message(ChatCompletionMessage(content="world", role="assistant"))
    completion.choices[0].finish_reason = "new_provider_reason"  # type: ignore[assignment]

    model = OpenAIChatModel(
        "gpt-4o",
        provider=OpenAIProvider(openai_client=MockOpenAI.create_mock(completion)),
    )

    response = model._process_response(completion)

    assert response.finish_reason is None
    assert response.provider_details is not None
    assert response.provider_details["finish_reason"] == "new_provider_reason"
