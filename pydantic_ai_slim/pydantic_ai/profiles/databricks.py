from dataclasses import dataclass

from pydantic_ai.profiles import ModelProfile
from pydantic_ai.profiles.openai import OpenAIModelProfile

_NO_STREAM_OPTIONS_MODELS = {
    'databricks-gpt-oss-120b',
    'databricks-qwen3-next-80b-a3b-instruct',
    'databricks-gpt-oss-20b',
    'databricks-llama-4-maverick',
    'databricks-gemma-3-12b',
}

_NO_TOOL_CALL_REQUIRED = {
    'databricks-gpt-oss-120b',
    'databricks-gpt-oss-20b',
}


@dataclass(kw_only=True)
class DatabricksModelProfile(OpenAIModelProfile):
    """Profile for models used with `DatabricksModel`.

    Inherits all configuration from OpenAIModelProfile.
    """

    databricks_supports_tool_call_required: bool = True
    databricks_stream_options: bool = True


def databricks_model_profile(model_name: str) -> ModelProfile:
    """Fetch databricks model profile based on model name."""
    tool_call_required = True
    stream_options = True
    if model_name in _NO_STREAM_OPTIONS_MODELS:
        stream_options = False
    if model_name in _NO_TOOL_CALL_REQUIRED:
        tool_call_required = False

    return DatabricksModelProfile(
        databricks_stream_options=stream_options, databricks_supports_tool_call_required=tool_call_required
    )
