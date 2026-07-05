from datetime import datetime, timezone

from pydantic_ai import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.messages import NativeToolCallPart, NativeToolReturnPart
from pydantic_ai.ui.ag_ui import AGUIAdapter

ts = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

def test_tool_return_part_metadata_dropped_on_agui_roundtrip() -> None:
    original = [
        ModelResponse(
            parts=[
                TextPart(content='hello'),
                ToolCallPart(tool_name='foo', args={'q': 'x'}, tool_call_id='tc-1'),
            ],
            timestamp=ts,
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name='foo',
                    content={'result': 'ok'},
                    tool_call_id='tc-1',
                    metadata={'app_state': {'k': 'v'}, 'attempt': 2},
                    timestamp=ts,
                )
            ],
        ),
    ]

    reloaded = AGUIAdapter.load_messages(AGUIAdapter.dump_messages(original))
    tr = reloaded[1].parts[0]
    assert tr.metadata == original[1].parts[0].metadata, (
        f'metadata lost: reloaded={tr.metadata!r} != original={original[1].parts[0].metadata!r}'
    )
    assert tr.timestamp == original[1].parts[0].timestamp, (
        f'timestamp lost: reloaded={tr.timestamp!r} != original={original[1].parts[0].timestamp!r}'
    )


def test_native_tool_return_part_metadata_dropped_on_agui_roundtrip() -> None:
    original = [
        ModelRequest(parts=[]),
        ModelResponse(
            parts=[
                TextPart(content='hello'),
                NativeToolCallPart(
                    tool_name='web_search', args={'q': 'x'},
                    tool_call_id='nc-1', provider_name='openai',
                ),
                NativeToolReturnPart(
                    tool_name='web_search', content={'result': 'ok'},
                    tool_call_id='nc-1', provider_name='openai',
                    metadata={'caller': 'agent-1', 'attempt': 2},
                    timestamp=ts,
                ),
            ],
            timestamp=ts,
        ),
    ]

    reloaded = AGUIAdapter.load_messages(AGUIAdapter.dump_messages(original))
    ntr = next(p for p in reloaded[0].parts if isinstance(p, NativeToolReturnPart))
    orig_ntr = next(p for p in original[1].parts if isinstance(p, NativeToolReturnPart))
    assert ntr.metadata == orig_ntr.metadata, (
        f'metadata lost: reloaded={ntr.metadata!r} != original={orig_ntr.metadata!r}'
    )
    assert ntr.timestamp == orig_ntr.timestamp, (
        f'timestamp lost: reloaded={ntr.timestamp!r} != original={orig_ntr.timestamp!r}'
    )
