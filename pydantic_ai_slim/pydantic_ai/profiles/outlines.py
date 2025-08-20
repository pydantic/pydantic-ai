from . import ModelProfile


def outlines_model_profile(model_name: str | None = None) -> ModelProfile:
    """Get the model profile for an Outlines model."""
    return ModelProfile(
        supports_tools=False,
        supports_json_schema_output=True,
        supports_json_object_output=True,
        default_structured_output_mode='native',
    )
