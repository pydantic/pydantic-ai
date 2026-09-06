import pytest
from pydantic import BaseModel
from pydantic_ai.messages import RetryPromptPart
from pydantic_core import ValidationError

def test_tool_call_retry_deduplicates_root_input():
    class MultiErrorModel(BaseModel):
        a: int
        b: int
        c: int

    # Missing fields cause input to be the root dictionary exactly
    invalid_args = {'d': 'some-extra-arg'}
    
    try:
        MultiErrorModel.model_validate(invalid_args)
    except ValidationError as e:
        error = e

    part = RetryPromptPart.from_error(error, tool_name='my_tool')
    msg = part.model_response()
    
    # The root invalid_args object should be included exactly once, even though there are 3 errors
    # In JSON, it appears as "input": { "d": "some-extra-arg" }
    # So we count the occurrences of "input"
    assert msg.count('"input": {') == 1
    # Check that there are 3 validation errors
    assert '3 validation errors:' in msg

