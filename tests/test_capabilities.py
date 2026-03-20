import importlib.util
from dataclasses import dataclass
from pathlib import Path

import pytest

from pydantic_ai._run_context import RunContext
from pydantic_ai.agent import Agent
from pydantic_ai.capabilities import (
    CAPABILITY_TYPES,
    Instructions,
    ModelSettings,
    Thinking,
    Toolset,
    WebSearch,
)
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.capabilities.combined import CombinedCapability
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.settings import ModelSettings as _ModelSettings
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset
from pydantic_ai.toolsets._dynamic import ToolsetFunc

from ._inline_snapshot import snapshot

pytestmark = [
    pytest.mark.anyio,
]


def test_capability_types() -> None:
    assert CAPABILITY_TYPES == snapshot(
        {
            'Instructions': Instructions,
            'ModelSettings': ModelSettings,
            'Thinking': Thinking,
            'WebSearch': WebSearch,
        }
    )


def test_agent_from_spec_basic():
    """Test Agent.from_spec with basic capabilities."""
    agent = Agent.from_spec(
        {
            'model': 'test',
            'capabilities': [
                {'Instructions': 'You are a helpful agent.'},
                'Thinking',
                'WebSearch',
                {'ModelSettings': {'max_tokens': 4096}},
            ],
        }
    )
    assert agent.model is not None


def test_agent_from_spec_no_capabilities():
    """Test Agent.from_spec with no capabilities."""
    agent = Agent.from_spec({'model': 'test'})
    assert agent.model is not None


def test_agent_from_spec_unknown_capability():
    """Test Agent.from_spec with an unknown capability name."""
    with pytest.raises(ValueError, match="Capability 'Unknown' is not in the provided"):
        Agent.from_spec(
            {
                'model': 'test',
                'capabilities': ['Unknown'],
            }
        )


def test_agent_from_spec_bad_args():
    """Test Agent.from_spec with bad arguments for a capability."""
    with pytest.raises(ValueError, match="Failed to instantiate capability 'Instructions'"):
        Agent.from_spec(
            {
                'model': 'test',
                'capabilities': [
                    {'Instructions': {'nonexistent_param': 'value'}},
                ],
            }
        )


@dataclass
class CustomCapability(AbstractCapability[None]):
    greeting: str = 'hello'


def test_agent_from_spec_custom_capability():
    """Test Agent.from_spec with a custom capability type."""
    agent = Agent.from_spec(
        {
            'model': 'test',
            'capabilities': [
                {'CustomCapability': 'world'},
            ],
        },
        custom_capability_types=[CustomCapability],
    )
    assert agent.model is not None


def test_agent_from_spec_with_agent_spec_object():
    """Test Agent.from_spec with an AgentSpec instance."""
    from pydantic_ai._spec import NamedSpec
    from pydantic_ai.agent.spec import AgentSpec

    spec = AgentSpec(
        model='test',
        capabilities=[
            NamedSpec(name='Instructions', arguments=('You are helpful.',)),
            NamedSpec(name='Thinking', arguments=None),
        ],
    )
    agent = Agent.from_spec(spec)
    assert agent.model is not None


def test_agent_from_spec_output_type():
    """Test Agent.from_spec with output_type parameter."""
    from pydantic import BaseModel

    class MyOutput(BaseModel):
        name: str
        value: int

    agent = Agent.from_spec({'model': 'test'}, output_type=MyOutput)
    assert agent.output_type == MyOutput


def test_agent_from_spec_output_schema():
    """Test Agent.from_spec with output_schema in spec."""
    schema = {
        'type': 'object',
        'properties': {
            'name': {'type': 'string'},
            'age': {'type': 'integer'},
        },
        'required': ['name', 'age'],
    }
    agent = Agent.from_spec({'model': 'test', 'output_schema': schema})
    # output_type should be a StructuredDict subclass (dict subclass with JSON schema)
    assert agent.output_type is not str
    assert isinstance(agent.output_type, type) and issubclass(agent.output_type, dict)


def test_agent_from_spec_output_type_takes_precedence():
    """Test that output_type parameter takes precedence over output_schema in spec."""
    from pydantic import BaseModel

    class MyOutput(BaseModel):
        name: str

    schema = {
        'type': 'object',
        'properties': {'name': {'type': 'string'}},
        'required': ['name'],
    }
    agent = Agent.from_spec({'model': 'test', 'output_schema': schema}, output_type=MyOutput)
    assert agent.output_type == MyOutput


def test_agent_from_spec_output_schema_invalid():
    """Test Agent.from_spec with a non-object output_schema raises UserError."""
    from pydantic_ai.exceptions import UserError

    with pytest.raises(UserError, match='Schema must be an object'):
        Agent.from_spec({'model': 'test', 'output_schema': {'type': 'string'}})


async def test_agent_from_spec_output_schema_integration():
    """Test Agent.from_spec with output_schema produces dict output."""
    from pydantic_ai.models.test import TestModel

    schema = {
        'type': 'object',
        'properties': {
            'city': {'type': 'string'},
            'country': {'type': 'string'},
        },
        'required': ['city', 'country'],
    }
    agent = Agent.from_spec({'model': 'test', 'output_schema': schema})
    result = await agent.run(
        'Tell me a city',
        model=TestModel(custom_output_args={'city': 'Paris', 'country': 'France'}),
    )
    assert result.output == {'city': 'Paris', 'country': 'France'}


def test_agent_from_spec_name():
    agent = Agent.from_spec({'model': 'test', 'name': 'my-agent'})
    assert agent.name == 'my-agent'


def test_agent_from_spec_name_override():
    agent = Agent.from_spec({'model': 'test', 'name': 'spec-name'}, name='override-name')
    assert agent.name == 'override-name'


def test_agent_from_spec_description():
    agent = Agent.from_spec({'model': 'test', 'description': 'A helpful agent'})
    assert agent.description == 'A helpful agent'


def test_agent_from_spec_description_override():
    agent = Agent.from_spec({'model': 'test', 'description': 'spec-desc'}, description='override-desc')
    assert agent.description == 'override-desc'


def test_agent_from_spec_instructions():
    agent = Agent.from_spec({'model': 'test', 'instructions': 'Be helpful.'})
    assert 'Be helpful.' in agent._instructions  # pyright: ignore[reportPrivateUsage]


def test_agent_from_spec_instructions_list():
    agent = Agent.from_spec({'model': 'test', 'instructions': ['First.', 'Second.']})
    assert 'First.' in agent._instructions  # pyright: ignore[reportPrivateUsage]
    assert 'Second.' in agent._instructions  # pyright: ignore[reportPrivateUsage]


def test_agent_from_spec_instructions_merged():
    agent = Agent.from_spec(
        {'model': 'test', 'instructions': 'From spec.'},
        instructions='From arg.',
    )
    assert 'From spec.' in agent._instructions  # pyright: ignore[reportPrivateUsage]
    assert 'From arg.' in agent._instructions  # pyright: ignore[reportPrivateUsage]


def test_agent_from_spec_model_settings():
    agent = Agent.from_spec({'model': 'test', 'model_settings': {'temperature': 0.5, 'max_tokens': 100}})
    ms = agent.model_settings
    assert isinstance(ms, dict)
    assert ms.get('temperature') == 0.5  # pyright: ignore[reportUnknownMemberType]
    assert ms.get('max_tokens') == 100  # pyright: ignore[reportUnknownMemberType]


def test_agent_from_spec_model_settings_merged():
    agent = Agent.from_spec(
        {'model': 'test', 'model_settings': {'temperature': 0.5, 'max_tokens': 100}},
        model_settings={'temperature': 0.9},
    )
    ms = agent.model_settings
    assert isinstance(ms, dict)
    assert ms.get('temperature') == 0.9  # pyright: ignore[reportUnknownMemberType]
    assert ms.get('max_tokens') == 100  # pyright: ignore[reportUnknownMemberType]


def test_agent_from_spec_retries():
    agent = Agent.from_spec({'model': 'test', 'retries': 5})
    assert agent._max_tool_retries == 5  # pyright: ignore[reportPrivateUsage]
    assert agent._max_result_retries == 5  # pyright: ignore[reportPrivateUsage]


def test_agent_from_spec_retries_override():
    agent = Agent.from_spec({'model': 'test', 'retries': 5}, retries=2)
    assert agent._max_tool_retries == 2  # pyright: ignore[reportPrivateUsage]
    assert agent._max_result_retries == 2  # pyright: ignore[reportPrivateUsage]


def test_agent_from_spec_output_retries():
    agent = Agent.from_spec({'model': 'test', 'retries': 3, 'output_retries': 10})
    assert agent._max_tool_retries == 3  # pyright: ignore[reportPrivateUsage]
    assert agent._max_result_retries == 10  # pyright: ignore[reportPrivateUsage]


def test_agent_from_spec_end_strategy():
    agent = Agent.from_spec({'model': 'test', 'end_strategy': 'exhaustive'})
    assert agent.end_strategy == 'exhaustive'


def test_agent_from_spec_end_strategy_override():
    agent = Agent.from_spec({'model': 'test', 'end_strategy': 'exhaustive'}, end_strategy='early')
    assert agent.end_strategy == 'early'


def test_agent_from_spec_tool_timeout():
    agent = Agent.from_spec({'model': 'test', 'tool_timeout': 30.0})
    assert agent._tool_timeout == 30.0  # pyright: ignore[reportPrivateUsage]


def test_agent_from_spec_tool_timeout_override():
    agent = Agent.from_spec({'model': 'test', 'tool_timeout': 30.0}, tool_timeout=5.0)
    assert agent._tool_timeout == 5.0  # pyright: ignore[reportPrivateUsage]


def test_agent_from_spec_instrument():
    agent = Agent.from_spec({'model': 'test', 'instrument': True})
    assert agent.instrument is True


def test_agent_from_spec_metadata():
    agent = Agent.from_spec({'model': 'test', 'metadata': {'env': 'prod', 'version': '1.0'}})
    assert agent._metadata == {'env': 'prod', 'version': '1.0'}  # pyright: ignore[reportPrivateUsage]


def test_agent_from_spec_metadata_override():
    agent = Agent.from_spec(
        {'model': 'test', 'metadata': {'env': 'prod'}},
        metadata={'env': 'staging'},
    )
    assert agent._metadata == {'env': 'staging'}  # pyright: ignore[reportPrivateUsage]


def test_agent_from_spec_model_override():
    agent = Agent.from_spec({'model': 'test'}, model='test')
    assert agent.model is not None


def test_agent_from_spec_capabilities_merged():
    @dataclass
    class ExtraCap(AbstractCapability[None]):
        pass

    agent = Agent.from_spec(
        {
            'model': 'test',
            'capabilities': [{'Instructions': 'From spec.'}],
        },
        capabilities=[ExtraCap()],
    )
    # Should have both the Instructions capability from spec and ExtraCap from arg
    children = agent.root_capability.capabilities
    assert any(isinstance(c, Instructions) for c in children)
    assert any(isinstance(c, ExtraCap) for c in children)


def test_model_json_schema_with_capabilities():
    from pydantic_ai.agent.spec import AgentSpec

    schema = AgentSpec.model_json_schema_with_capabilities()

    assert schema['type'] == 'object'
    assert 'model' in schema['properties']
    assert 'capabilities' in schema['properties']
    assert '$schema' in schema['properties']

    # Capabilities should be a list with anyOf containing default types
    cap_schema = schema['properties']['capabilities']
    assert cap_schema['type'] == 'array'
    any_of = cap_schema['items']['anyOf']

    # Collect all capability names referenced in the schema (both const literals and object keys)
    capability_names: set[str] = set()
    for entry in any_of:
        if 'const' in entry:
            capability_names.add(entry['const'])
        elif '$ref' in entry:
            # Extract the name from refs like '#/$defs/spec_Instructions'
            ref = entry['$ref']
            ref_name = ref.rsplit('/', 1)[-1]
            for prefix in ('spec_', 'short_spec_'):
                if ref_name.startswith(prefix):
                    capability_names.add(ref_name[len(prefix) :])

    assert capability_names == {'Instructions', 'ModelSettings', 'Thinking', 'WebSearch'}


def test_model_json_schema_with_custom_capabilities():
    from pydantic_ai.agent.spec import AgentSpec

    schema = AgentSpec.model_json_schema_with_capabilities(
        custom_capability_types=[CustomCapability],
    )

    any_of = schema['properties']['capabilities']['items']['anyOf']

    capability_names: set[str] = set()
    for entry in any_of:
        if 'const' in entry:
            capability_names.add(entry['const'])
        elif '$ref' in entry:
            ref = entry['$ref']
            ref_name = ref.rsplit('/', 1)[-1]
            for prefix in ('spec_', 'short_spec_'):
                if ref_name.startswith(prefix):
                    capability_names.add(ref_name[len(prefix) :])

    assert 'CustomCapability' in capability_names
    # Default capabilities should still be present
    assert 'Thinking' in capability_names
    assert 'WebSearch' in capability_names


def test_save_schema(tmp_path: str):
    from pathlib import Path

    from pydantic_ai.agent.spec import AgentSpec

    schema_path = Path(tmp_path) / 'agent_spec.schema.json'
    AgentSpec._save_schema(schema_path)  # pyright: ignore[reportPrivateUsage]

    assert schema_path.exists()
    import json

    schema = json.loads(schema_path.read_text(encoding='utf-8'))
    assert schema['type'] == 'object'
    assert 'model' in schema['properties']
    assert 'capabilities' in schema['properties']

    # Calling again should not rewrite if content matches
    mtime = schema_path.stat().st_mtime
    AgentSpec._save_schema(schema_path)  # pyright: ignore[reportPrivateUsage]
    assert schema_path.stat().st_mtime == mtime


def test_from_file_yaml(tmp_path: str):
    from pathlib import Path

    from pydantic_ai.agent.spec import AgentSpec

    spec_path = Path(tmp_path) / 'agent.yaml'
    spec_path.write_text('model: test\nname: my-agent\ninstructions: Be helpful\n', encoding='utf-8')
    spec = AgentSpec.from_file(spec_path)
    assert spec.model == 'test'
    assert spec.name == 'my-agent'
    assert spec.instructions == 'Be helpful'


def test_from_file_json(tmp_path: str):
    from pathlib import Path

    from pydantic_ai.agent.spec import AgentSpec

    spec_path = Path(tmp_path) / 'agent.json'
    spec_path.write_text('{"model": "test", "name": "my-agent"}', encoding='utf-8')
    spec = AgentSpec.from_file(spec_path)
    assert spec.model == 'test'
    assert spec.name == 'my-agent'


def test_from_file_with_schema_field(tmp_path: str):
    """$schema field in the file should be accepted and not cause validation errors."""
    from pathlib import Path

    from pydantic_ai.agent.spec import AgentSpec

    spec_path = Path(tmp_path) / 'agent.yaml'
    spec_path.write_text('model: test\n', encoding='utf-8')

    # YAML with $schema comment (ignored by yaml parser)
    spec_with_schema = Path(tmp_path) / 'agent_with_schema.json'
    spec_with_schema.write_text('{"$schema": "./agent_schema.json", "model": "test"}', encoding='utf-8')
    spec = AgentSpec.from_file(spec_with_schema)
    assert spec.model == 'test'
    assert spec.json_schema_path == './agent_schema.json'


def test_to_file_yaml(tmp_path: str):
    from pathlib import Path

    from pydantic_ai.agent.spec import AgentSpec

    spec = AgentSpec(model='test', name='my-agent', instructions='Be helpful')
    spec_path = Path(tmp_path) / 'agent.yaml'
    spec.to_file(spec_path)

    content = spec_path.read_text(encoding='utf-8')
    # Should start with yaml-language-server schema comment
    assert content.startswith('# yaml-language-server: $schema=')
    assert 'model: test' in content
    assert 'name: my-agent' in content

    # Schema file should be generated
    schema_path = Path(tmp_path) / 'agent_schema.json'
    assert schema_path.exists()


def test_to_file_json(tmp_path: str):
    import json
    from pathlib import Path

    from pydantic_ai.agent.spec import AgentSpec

    spec = AgentSpec(model='test', name='my-agent')
    spec_path = Path(tmp_path) / 'agent.json'
    spec.to_file(spec_path)

    data = json.loads(spec_path.read_text(encoding='utf-8'))
    assert data['$schema'] == 'agent_schema.json'
    assert data['model'] == 'test'
    assert data['name'] == 'my-agent'

    # Schema file should be generated
    schema_path = Path(tmp_path) / 'agent_schema.json'
    assert schema_path.exists()


def test_to_file_no_schema(tmp_path: str):
    from pathlib import Path

    from pydantic_ai.agent.spec import AgentSpec

    spec = AgentSpec(model='test')
    spec_path = Path(tmp_path) / 'agent.yaml'
    spec.to_file(spec_path, schema_path=None)

    content = spec_path.read_text(encoding='utf-8')
    assert '# yaml-language-server' not in content

    # No schema file should be generated
    schema_path = Path(tmp_path) / 'agent_schema.json'
    assert not schema_path.exists()


def test_to_file_roundtrip_yaml(tmp_path: str):
    from pathlib import Path

    from pydantic_ai.agent.spec import AgentSpec

    spec = AgentSpec(model='test', name='roundtrip', instructions=['Be helpful', 'Be concise'])
    spec_path = Path(tmp_path) / 'agent.yaml'
    spec.to_file(spec_path)

    loaded = AgentSpec.from_file(spec_path)
    assert loaded.model == 'test'
    assert loaded.name == 'roundtrip'
    assert loaded.instructions == ['Be helpful', 'Be concise']


def test_to_file_roundtrip_json(tmp_path: str):
    from pathlib import Path

    from pydantic_ai.agent.spec import AgentSpec

    spec = AgentSpec(model='test', name='roundtrip', retries=3)
    spec_path = Path(tmp_path) / 'agent.json'
    spec.to_file(spec_path)

    loaded = AgentSpec.from_file(spec_path)
    assert loaded.model == 'test'
    assert loaded.name == 'roundtrip'
    assert loaded.retries == 3


@dataclass
class ToolsetFuncCapability(AbstractCapability[None]):
    """A capability that returns a ToolsetFunc instead of an AbstractToolset."""

    def get_toolset(self) -> ToolsetFunc[None]:
        def make_toolset(ctx: RunContext[None]) -> AbstractToolset[None]:
            toolset = FunctionToolset[None]()

            @toolset.tool_plain
            def greet(name: str) -> str:
                """Greet someone by name."""
                return f'Hello, {name}!'

            return toolset

        return make_toolset


async def test_capability_returning_toolset_func():
    """Test that a capability returning a ToolsetFunc works with an agent."""
    from pydantic_ai.models.test import TestModel

    agent = Agent(
        TestModel(),
        capabilities=[ToolsetFuncCapability()],
    )
    result = await agent.run('Greet Alice')

    tool_calls = [
        part
        for msg in result.all_messages()
        if isinstance(msg, ModelResponse)
        for part in msg.parts
        if isinstance(part, ToolCallPart)
    ]
    assert len(tool_calls) == 1
    assert tool_calls[0].tool_name == 'greet'

    tool_returns = [
        part
        for msg in result.all_messages()
        if isinstance(msg, ModelRequest)
        for part in msg.parts
        if isinstance(part, ToolReturnPart)
    ]
    assert len(tool_returns) == 1
    assert isinstance(tool_returns[0].content, str)
    assert tool_returns[0].content.startswith('Hello, ')


async def test_capability_returning_toolset_func_combined():
    """Test that a ToolsetFunc capability works alongside other capabilities via CombinedCapability."""
    from pydantic_ai.models.test import TestModel

    agent = Agent(
        TestModel(),
        capabilities=[
            Instructions('You are a helpful greeter.'),
            ToolsetFuncCapability(),
        ],
    )
    result = await agent.run('Greet Bob')

    tool_returns = [
        part
        for msg in result.all_messages()
        if isinstance(msg, ModelRequest)
        for part in msg.parts
        if isinstance(part, ToolReturnPart)
    ]
    assert len(tool_returns) == 1
    assert isinstance(tool_returns[0].content, str)
    assert tool_returns[0].content.startswith('Hello, ')


def test_model_settings_from_spec_positional():
    """ModelSettings.from_spec with a single positional dict arg."""
    cap = ModelSettings.from_spec({'max_tokens': 4096, 'temperature': 0.5})
    assert cap.settings == {'max_tokens': 4096, 'temperature': 0.5}


def test_model_settings_from_spec_kwargs():
    """ModelSettings.from_spec with keyword arguments."""
    cap = ModelSettings.from_spec(max_tokens=100)
    assert cap.settings == {'max_tokens': 100}


def test_model_settings_callable_get_model_settings():
    """Callable ModelSettings returns the callable from get_model_settings for resolution in the chain."""

    def dynamic_settings(ctx: RunContext[None]) -> _ModelSettings:
        return _ModelSettings(temperature=0.9)

    cap = ModelSettings(settings=dynamic_settings)

    # get_model_settings returns the callable directly — resolution happens in the agent's settings chain
    result = cap.get_model_settings()
    assert callable(result)
    assert result is dynamic_settings


async def test_model_settings_static_before_model_request():
    """Static ModelSettings passes through before_model_request without modification."""
    from pydantic_ai.capabilities.abstract import BeforeModelRequestContext
    from pydantic_ai.models import ModelRequestParameters
    from pydantic_ai.models.test import TestModel
    from pydantic_ai.result import RunUsage

    cap = ModelSettings(settings=_ModelSettings(max_tokens=200))

    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage(), prompt=None, messages=[])
    input_settings = _ModelSettings(temperature=0.5)
    result = await cap.before_model_request(
        ctx,
        BeforeModelRequestContext(
            messages=[],
            model_settings=input_settings,
            model_request_parameters=ModelRequestParameters(),
        ),
    )
    # Static settings are handled by get_model_settings, not before_model_request
    assert result.model_settings is input_settings


def test_abstract_capability_get_model_settings_default():
    """AbstractCapability.get_model_settings() returns None by default."""

    @dataclass
    class PlainCap(AbstractCapability[None]):
        pass

    cap = PlainCap()
    assert cap.get_model_settings() is None


def test_combined_capability_get_model_settings_merge():
    """CombinedCapability.get_model_settings() merges settings from all sub-capabilities."""
    caps = CombinedCapability(
        capabilities=[
            ModelSettings(settings=_ModelSettings(max_tokens=100)),
            ModelSettings(settings=_ModelSettings(temperature=0.5)),
        ]
    )
    merged = caps.get_model_settings()
    assert merged is not None
    assert not callable(merged)
    assert merged.get('max_tokens') == 100
    assert merged.get('temperature') == 0.5


def test_combined_capability_get_model_settings_none():
    """CombinedCapability.get_model_settings() returns None when no capabilities provide settings."""
    caps = CombinedCapability(capabilities=[Instructions('hello')])
    assert caps.get_model_settings() is None


def test_toolset_capability_get_toolset():
    """Toolset capability returns its toolset."""
    ts = FunctionToolset[None]()
    cap = Toolset(toolset=ts)
    assert cap.get_toolset() is ts


async def test_toolset_capability_in_agent():
    """A Toolset capability's tools are available to the agent."""
    from pydantic_ai.models.test import TestModel

    ts = FunctionToolset[None]()

    @ts.tool_plain
    def greet(name: str) -> str:
        """Greet someone by name."""
        return f'Hello, {name}!'

    agent = Agent(TestModel(), capabilities=[Toolset(toolset=ts)])
    result = await agent.run('Greet Alice')

    tool_returns = [
        part
        for msg in result.all_messages()
        if isinstance(msg, ModelRequest)
        for part in msg.parts
        if isinstance(part, ToolReturnPart)
    ]
    assert len(tool_returns) == 1
    assert isinstance(tool_returns[0].content, str)
    assert tool_returns[0].content.startswith('Hello, ')


def test_infer_fmt_explicit():
    """_infer_fmt returns the explicit fmt when provided."""
    from pydantic_ai.agent.spec import _infer_fmt  # pyright: ignore[reportPrivateUsage]

    assert _infer_fmt(Path('agent.txt'), 'json') == 'json'
    assert _infer_fmt(Path('agent.txt'), 'yaml') == 'yaml'


def test_infer_fmt_unknown_extension():
    """_infer_fmt raises ValueError for unknown extension without explicit fmt."""
    from pydantic_ai.agent.spec import _infer_fmt  # pyright: ignore[reportPrivateUsage]

    with pytest.raises(ValueError, match="Could not infer format for filename 'agent.txt'"):
        _infer_fmt(Path('agent.txt'), None)


def test_invalid_custom_capability_type():
    """Passing a non-AbstractCapability subclass to model_json_schema_with_capabilities raises ValueError."""
    from pydantic_ai.agent.spec import AgentSpec

    with pytest.raises(ValueError, match='must be subclasses of AbstractCapability'):
        AgentSpec.model_json_schema_with_capabilities(
            custom_capability_types=[str],  # type: ignore[list-item]
        )


def test_to_file_with_path_schema_path(tmp_path: str):
    """to_file works when schema_path is passed as a relative Path (not str), triggering the non-str branch."""
    from pydantic_ai.agent.spec import AgentSpec

    spec = AgentSpec(model='test', name='path-schema')
    spec_path = Path(tmp_path) / 'agent.yaml'
    # Pass a relative Path (not str) to exercise the isinstance(schema_path, str) == False branch
    schema_path = Path('custom_schema.json')
    spec.to_file(spec_path, schema_path=schema_path)

    resolved_schema = Path(tmp_path) / 'custom_schema.json'
    assert resolved_schema.exists()
    content = spec_path.read_text(encoding='utf-8')
    assert 'model: test' in content


@pytest.mark.skipif(not importlib.util.find_spec('rich'), reason='CLI deps not installed')
def test_load_agent_yaml(tmp_path: str):
    """load_agent loads an agent from a YAML spec file."""
    from pydantic_ai._cli import load_agent

    spec_path = Path(tmp_path) / 'agent.yaml'
    spec_path.write_text('model: test\nname: yaml-agent\n', encoding='utf-8')
    agent = load_agent(str(spec_path))
    assert agent is not None
    assert agent.name == 'yaml-agent'


@pytest.mark.skipif(not importlib.util.find_spec('rich'), reason='CLI deps not installed')
def test_load_agent_json(tmp_path: str):
    """load_agent loads an agent from a JSON spec file."""
    from pydantic_ai._cli import load_agent

    spec_path = Path(tmp_path) / 'agent.json'
    spec_path.write_text('{"model": "test", "name": "json-agent"}', encoding='utf-8')
    agent = load_agent(str(spec_path))
    assert agent is not None
    assert agent.name == 'json-agent'


@pytest.mark.skipif(not importlib.util.find_spec('rich'), reason='CLI deps not installed')
def test_load_agent_missing_file(tmp_path: str):
    """load_agent returns None for a non-existent spec file."""
    from pydantic_ai._cli import load_agent

    agent = load_agent(str(Path(tmp_path) / 'nonexistent.yaml'))
    assert agent is None


async def test_thinking_capability_applies_settings():
    """Thinking capability's model settings are applied to the model request."""

    agent = Agent('test', capabilities=[Thinking()])
    result = await agent.run('hi')
    # The agent ran successfully — verify that the Thinking settings were included
    # by checking that the capability produces non-None model settings
    cap_settings = agent.root_capability.get_model_settings()
    assert cap_settings is not None
    assert not callable(cap_settings)
    assert cap_settings.get('anthropic_thinking') == {'type': 'adaptive'}
    assert cap_settings.get('openai_reasoning_effort') == 'high'
    # Verify the run itself succeeds
    assert result.output is not None


def test_thinking_from_spec_rejects_args():
    """Thinking.from_spec raises TypeError when given arguments."""
    with pytest.raises(TypeError, match='does not accept arguments'):
        Thinking.from_spec('extra')

    with pytest.raises(TypeError, match='does not accept arguments'):
        Thinking.from_spec(budget=100)
