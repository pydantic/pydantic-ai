"""v2 form: provider_details / provider_response_id."""
from pydantic_ai.messages import ModelResponse, TextPart


def trigger():
    r = ModelResponse(parts=[TextPart(content='x')])
    _ = r.provider_details
    _ = r.provider_response_id
    return r


if __name__ == '__main__':
    trigger()
