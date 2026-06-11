from pydantic_ai._deferred_capabilities import parse_loaded_capabilities
from pydantic_ai.messages import (
    LoadCapabilityCallPart,
    LoadCapabilityReturnPart,
    ModelRequest,
    ModelResponse,
)

# from pydantic_ai.ui.vercel_ai import VercelAIAdapter
from pydantic_ai.ui.ag_ui import AGUIAdapter

messages = [
    ModelResponse(
        parts=[
            LoadCapabilityCallPart(
                tool_call_id='load-foobar',
                args={'id': 'foobar'},
            )
        ]
    ),
    ModelRequest(
        parts=[
            LoadCapabilityReturnPart(
                tool_call_id='load-foobar',
                content={'instructions': '# Foo Bar'},
            )
        ]
    ),
]

assert parse_loaded_capabilities(messages) == {'foobar'}

# ui_messages = VercelAIAdapter.dump_messages(messages)
# round_tripped = VercelAIAdapter.load_messages(ui_messages)

ui_messages = AGUIAdapter.dump_messages(messages)

print(ui_messages)

print('\n\n\n')

round_tripped = AGUIAdapter.load_messages(ui_messages)

print(round_tripped)

# Expected: {"foobar"}
# Actual: set()
assert parse_loaded_capabilities(round_tripped) == {'foobar'}
