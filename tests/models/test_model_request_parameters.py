from pydantic import TypeAdapter

from pydantic_ai.models import ModelRequestParameters


def test_model_request_parameters_are_serializable():
    params = ModelRequestParameters(
        function_tools=[], preferred_output_mode=None, allow_text_output=False, output_tools=[], output_object=None
    )
    assert TypeAdapter(ModelRequestParameters).dump_python(params) == {
        'function_tools': [],
        'preferred_output_mode': None,
        'allow_text_output': False,
        'output_tools': [],
        'output_object': None,
    }
