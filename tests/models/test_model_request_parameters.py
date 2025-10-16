from inline_snapshot import snapshot
from pydantic import TypeAdapter

from pydantic_ai.builtin_tools import ImageGenerationTool, WebSearchTool, WebSearchUserLocation
from pydantic_ai.models import ModelRequestParameters, ToolDefinition

ta = TypeAdapter(ModelRequestParameters)


def test_model_request_parameters_are_serializable():
    params = ModelRequestParameters(
        function_tools=[],
        builtin_tools=[],
        output_mode='text',
        allow_text_output=True,
        output_tools=[],
        output_object=None,
    )
    dumped = ta.dump_python(params)
    assert dumped == snapshot(
        {
            'function_tools': [],
            'builtin_tools': [],
            'output_mode': 'text',
            'output_object': None,
            'output_tools': [],
            'allow_text_output': True,
            'allow_image_output': False,
        }
    )
    assert ta.validate_python(dumped) == params

    params = ModelRequestParameters(
        function_tools=[ToolDefinition(name='test')],
        builtin_tools=[
            WebSearchTool(user_location=WebSearchUserLocation(city='New York', country='US')),
            ImageGenerationTool(size='1024x1024'),
        ],
        output_mode='text',
        allow_text_output=True,
        output_tools=[ToolDefinition(name='final_result')],
        output_object=None,
    )
    dumped = ta.dump_python(params)
    assert dumped == snapshot(
        {
            'function_tools': [
                {
                    'name': 'test',
                    'parameters_json_schema': {'type': 'object', 'properties': {}},
                    'description': None,
                    'outer_typed_dict_key': None,
                    'strict': None,
                    'sequential': False,
                    'kind': 'function',
                    'metadata': None,
                }
            ],
            'builtin_tools': [
                {
                    'kind': 'web_search',
                    'search_context_size': 'medium',
                    'user_location': {'city': 'New York', 'country': 'US'},
                    'blocked_domains': None,
                    'allowed_domains': None,
                    'max_uses': None,
                },
                {
                    'kind': 'image_generation',
                    'background': 'auto',
                    'input_fidelity': None,
                    'moderation': 'auto',
                    'output_compression': 100,
                    'output_format': None,
                    'partial_images': 0,
                    'quality': 'auto',
                    'size': '1024x1024',
                },
            ],
            'output_mode': 'text',
            'output_object': None,
            'output_tools': [
                {
                    'name': 'final_result',
                    'parameters_json_schema': {'type': 'object', 'properties': {}},
                    'description': None,
                    'outer_typed_dict_key': None,
                    'strict': None,
                    'sequential': False,
                    'kind': 'function',
                    'metadata': None,
                }
            ],
            'allow_text_output': True,
            'allow_image_output': False,
        }
    )
    assert ta.validate_python(dumped) == params
