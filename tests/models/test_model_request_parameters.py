from pydantic import TypeAdapter

from pydantic_ai.models import ModelRequestParameters


def test_model_request_parameters_are_serializable():
    params = ModelRequestParameters(function_tools=[], allow_text_output=False, output_tools=[])
    assert TypeAdapter(ModelRequestParameters).dump_python(params) == {
        'function_tools': [],
        'allow_text_output': False,
        'output_tools': [],
    }
