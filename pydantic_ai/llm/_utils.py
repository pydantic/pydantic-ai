import json

from ..messages import FunctionResponse, FunctionValidationError


def function_response_content(m: FunctionResponse) -> str:
    # return f'Response from calling {m.function_name}: {m.content}'
    return m.content


def function_validation_error_content(m: FunctionValidationError) -> str:
    errors_json = json.dumps(m.errors, indent=2)
    return f'Validation error calling {m.function_name}:\n{errors_json}\n\nPlease fix the errors and try again.'
