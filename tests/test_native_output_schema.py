from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai.output import NativeOutput
from pydantic_ai.profiles import ModelProfile


class City(BaseModel):
    name: str
    population: int


async def test_native_output_schema_injection_from_profile():
    """
    Test that `native_output_requires_schema_in_instructions=True` in the profile
    causes the schema instructions to be injected, even when using `NativeOutput` implicitly or explicitly.
    """
    profile = ModelProfile(
        supports_json_schema_output=True,
        default_structured_output_mode='native',
        native_output_requires_schema_in_instructions=True,
        prompted_output_template='SCHEMA: {schema}',
    )
    model = TestModel(profile=profile, custom_output_text='{ "name": "Paris", "population": 9000000 }')
    agent = Agent(model, output_type=City)

    await agent.run('Paris')

    params = model.last_model_request_parameters
    assert params
    assert params.prompted_output_instructions is not None
    assert 'SCHEMA:' in params.prompted_output_instructions
    assert 'City' in params.prompted_output_instructions


async def test_native_output_custom_template_override():
    """
    Test that providing a `template` in `NativeOutput` uses that template,
    regardless of the profile setting (even if injection is disabled in profile).
    """
    profile = ModelProfile(
        supports_json_schema_output=True,
        default_structured_output_mode='native',
        native_output_requires_schema_in_instructions=False,  # Disabled in profile
    )
    model = TestModel(profile=profile, custom_output_text='{ "name": "London", "population": 9000000 }')
    agent = Agent(model)

    # Use NativeOutput with explicit template
    await agent.run(
        'London',
        output_type=NativeOutput(City, template='CUSTOM TEMPLATE: {schema}'),
    )

    params = model.last_model_request_parameters
    assert params
    assert params.prompted_output_instructions is not None
    assert 'CUSTOM TEMPLATE:' in params.prompted_output_instructions
    assert 'City' in params.prompted_output_instructions


async def test_native_output_custom_template_precedence():
    """
    Test that providing a `template` in `NativeOutput` takes precedence over the profile default,
    even if injection is enabled in the profile.
    """
    profile = ModelProfile(
        supports_json_schema_output=True,
        default_structured_output_mode='native',
        native_output_requires_schema_in_instructions=True,
        prompted_output_template='DEFAULT SCHEMA: {schema}',
    )
    model = TestModel(profile=profile, custom_output_text='{ "name": "China", "population": 9000000 }')
    agent = Agent(model)

    await agent.run(
        'China',
        output_type=NativeOutput(City, template='OVERRIDE TEMPLATE: {schema}'),
    )

    params = model.last_model_request_parameters
    assert params
    assert params.prompted_output_instructions is not None
    assert 'OVERRIDE TEMPLATE:' in params.prompted_output_instructions
    assert 'DEFAULT SCHEMA:' not in params.prompted_output_instructions


async def test_native_output_no_injection_by_default():
    """
    Test that without the profile setting and without a custom template,
    no instructions are injected for NativeOutput (default behavior).
    """
    profile = ModelProfile(
        supports_json_schema_output=True,
        default_structured_output_mode='native',
        native_output_requires_schema_in_instructions=False,
    )
    model = TestModel(profile=profile, custom_output_text='{ "name": "Tokyo", "population": 9000000 }')
    agent = Agent(model, output_type=City)

    await agent.run('Tokyo')

    params = model.last_model_request_parameters
    assert params
    assert params.prompted_output_instructions is None
