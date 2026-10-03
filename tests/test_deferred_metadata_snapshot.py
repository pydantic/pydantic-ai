from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import DeferredToolRequests


def test_build_results_snapshots_per_call_metadata():
    requests = DeferredToolRequests(calls=[ToolCallPart('tool', {}, tool_call_id='call-1')])
    caller_metadata = {'call-1': {'scope': 'before'}}

    results = requests.build_results(metadata=caller_metadata)
    caller_metadata['call-1']['scope'] = 'after'

    assert results.metadata == {'call-1': {'scope': 'before'}}


def test_remaining_snapshots_per_call_metadata():
    requests = DeferredToolRequests(
        calls=[
            ToolCallPart('resolved', {}, tool_call_id='call-1'),
            ToolCallPart('pending', {}, tool_call_id='call-2'),
        ],
        metadata={
            'call-1': {'scope': 'resolved'},
            'call-2': {'scope': 'before'},
        },
    )

    results = requests.build_results(calls={'call-1': 'done'})
    remaining = requests.remaining(results)
    assert remaining is not None

    requests.metadata['call-2']['scope'] = 'after'

    assert remaining.metadata == {'call-2': {'scope': 'before'}}
