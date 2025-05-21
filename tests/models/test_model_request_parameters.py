from pydantic import TypeAdapter

from pydantic_ai.models import ModelRequestParameters


def test_model_request_parameters_are_serializable():
    params = ModelRequestParameters(
        function_tools=[], output_mode=None, require_tool_use=False, output_tools=[], output_object=None
    )
    assert TypeAdapter(ModelRequestParameters).dump_python(params) == {
        'function_tools': [],
        'preferred_output_mode': None,
        'require_tool_use': False,
        'output_tools': [],
        'output_object': None,
    }
