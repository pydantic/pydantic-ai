"""The public worker context contract, independent of a particular engine."""

from __future__ import annotations

import json

import pytest
from pydantic import TypeAdapter

from pydantic_ai import RunContext
from pydantic_ai._run_context import AnchoredEvidence
from pydantic_ai.durable_exec import SerializedRunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RunUsage, UsageLimits
from pydantic_ai.workspaces import WorkspaceRef


def test_serialized_context_rehydrates_json_and_guards_omitted_fields() -> None:
    source = RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(requests=2),
        usage_limits=UsageLimits(request_limit=7),
        loaded_capability_ids={'loaded'},
        discovered_tool_names={'discovered'},
        _anchored_evidence=AnchoredEvidence(discovered_tool_names=frozenset({'anchored'})),
    )
    projected = SerializedRunContext.serialize_run_context(source)
    projected['workspace_ref'] = WorkspaceRef(provider='local', id='example')
    wire = json.loads(TypeAdapter(dict[str, object]).dump_json(projected))

    assert isinstance(wire['usage'], dict)
    assert isinstance(wire['loaded_capability_ids'], list)
    restored = SerializedRunContext(deps=None, **wire)
    assert restored.usage == source.usage
    assert restored.usage_limits == source.usage_limits
    assert restored.loaded_capability_ids == {'loaded'}
    assert restored.discovered_tool_names == {'discovered'}
    assert restored._anchored_evidence == source._anchored_evidence  # pyright: ignore[reportPrivateUsage]
    assert restored.workspace_ref == WorkspaceRef(provider='local', id='example')
    assert restored.agent is None
    with pytest.raises(UserError, match="'model' is not available"):
        _ = restored.model
    with pytest.raises(UserError, match="'validation_context' is not available"):
        _ = restored.validation_context


def test_serialized_context_uses_availability_snapshots_and_old_payload_fallback() -> None:
    source = RunContext(deps=None, model=TestModel(), usage=RunUsage())
    projected = SerializedRunContext.serialize_run_context(source)
    projected['available_tool_names'] = ['visible']
    projected['active_capability_ids'] = ['active']
    projected['_deferred_capability_ids'] = ['deferred']
    restored = SerializedRunContext(deps=None, **projected)
    assert restored.is_tool_available('visible')
    assert restored.is_tool_available(ToolDefinition(name='owned', capability_id='active'))
    assert restored._deferred_capability_ids == {'deferred'}  # pyright: ignore[reportPrivateUsage]

    for name in ('available_tool_names', 'active_capability_ids', '_deferred_capability_ids', '_anchored_evidence'):
        projected.pop(name)
    older = SerializedRunContext(deps=None, **projected)
    assert older.available_tool_names == set()
    assert older._anchored_evidence == AnchoredEvidence()  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(UserError, match="'capabilities' is not available"):
        _ = older.active_capability_ids
