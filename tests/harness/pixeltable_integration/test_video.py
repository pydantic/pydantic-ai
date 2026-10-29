"""Run the video example with real media/catalog operations and local provider substitutes."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextContent,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from tests.conftest import try_import

with try_import() as imports_successful:
    import pixeltable as pxt
    from pixeltable.functions import openai

    from .support import get_table, tiny_embed

    @pxt.udf  # pyright: ignore[reportUnknownMemberType]
    def transcribe(audio: pxt.Audio, *, model: str) -> dict[str, str]:
        _PROVIDER_CALLS.append('transcribe')
        return {'text': _POLICY}

    @pxt.udf  # pyright: ignore[reportUnknownMemberType]
    def caption(messages: list[dict[str, object]], *, model: str) -> dict[str, object]:
        _PROVIDER_CALLS.append('caption')
        return {'choices': [{'message': {'content': _POLICY}}]}


pytestmark = pytest.mark.skipif(not imports_successful(), reason='pixeltable not installed')

_PROVIDER_CALLS: list[str] = []
_POLICY = 'Meals during business travel are reimbursed up to 35 EUR per day.'


def test_video_example_reuses_ingestion_and_recalls_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    path = Path(__file__).parents[3] / 'src/pydantic_ai_harness/examples/pixeltable_video.py'
    spec = importlib.util.spec_from_file_location('pixeltable_video_example', path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    monkeypatch.setenv('PYDANTIC_AI_MODEL', 'test')
    spec.loader.exec_module(module)

    monkeypatch.setattr(openai, 'transcriptions', transcribe)
    monkeypatch.setattr(openai, 'chat_completions', caption)

    def embedding(*, model: str) -> pxt.Function:
        return tiny_embed

    monkeypatch.setattr(openai.embeddings, 'using', embedding)
    _PROVIDER_CALLS.clear()
    verified: list[bool] = []

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prompts = [
            part
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, UserPromptPart)
        ]
        frames = any(isinstance(part.content, str) and 'saved preference' in part.content for part in prompts)
        returns = [
            part
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        if not returns:
            calls = [
                ToolCallPart(
                    'similarity_search',
                    {
                        'table': 'training.frames' if frames else 'training.segments',
                        'query': _POLICY,
                        'column': 'caption' if frames else 'transcript',
                    },
                    tool_call_id='search',
                )
            ]
            if not frames:
                calls.append(
                    ToolCallPart('write_memory', {'content': 'I prefer visual examples.'}, tool_call_id='remember')
                )
            return ModelResponse(parts=calls)
        search = TypeAdapter(dict[str, object]).validate_python(returns[0].content)
        rows = TypeAdapter(list[dict[str, object]]).validate_python(search['rows'])
        assert rows
        row = rows[0]
        assert row['title'] == 'Expense policy'
        assert row['caption' if frames else 'transcript'] == _POLICY
        assert not {'video', 'audio', 'frame', 'audio_segment'} & row.keys()
        if frames:
            assert TypeAdapter(dict[str, object]).validate_python(row['frame_attrs'])['time'] == 0
            assert any(
                not isinstance(part.content, str)
                and any(
                    isinstance(content, TextContent)
                    and content.content.startswith('<memory>\n')
                    and 'visual examples' in content.content
                    for content in part.content
                )
                for part in prompts
            )
        else:
            assert row['segment_start'] == 0
            assert isinstance(row['segment_end'], (int, float)) and row['segment_end'] > 0
        verified.append(frames)
        return ModelResponse(parts=[TextPart('verified')])

    factory = module.build_agent

    def build_agent() -> Agent[None, str]:
        assert module.DEFAULT_MODEL == 'test'
        return factory(model=FunctionModel(model))

    monkeypatch.setattr(module, 'build_agent', build_agent)
    video_path = Path(__file__).parents[2] / 'assets/pixeltable-policy.mp4'
    pxt.drop_dir('training', force=True, if_not_exists='ignore')
    try:
        module.main(str(video_path))
        first_calls = list(_PROVIDER_CALLS)
        assert set(first_calls) == {'transcribe', 'caption'}
        module.main(str(video_path))
        assert _PROVIDER_CALLS == first_calls
        assert get_table('training.videos').count() == 1
        assert verified == [False, True, False, True]
    finally:
        pxt.drop_dir('training', force=True, if_not_exists='ignore')
