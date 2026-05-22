"""Probe OpenAI Responses `tool_search` namespace replay behavior.

Run from the repository root:

    OPENAI_API_KEY=... uv run python scripts/openai_tool_search_namespace_probe.py

Optional:

    OPENAI_MODEL=gpt-5.4 uv run python scripts/openai_tool_search_namespace_probe.py
    uv run python scripts/openai_tool_search_namespace_probe.py --model gpt-5.4 --quick
    uv run python scripts/openai_tool_search_namespace_probe.py --pydantic-ai-only

The probe answers:

* For flat `defer_loading=True` functions, does OpenAI accept replay with
  `namespace == function name`?
* Does it reject missing or random namespaces?
* Does that hold when `tool_search` is client-executed?
* For an explicit namespace wrapper, does only the explicit namespace work?
* What namespaces does OpenAI return in observed hosted/client tool-search runs?
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections.abc import Iterable
from typing import Any

from openai import AsyncOpenAI, BadRequestError


FunctionTool = dict[str, Any]
ResponseItem = dict[str, Any]


def _policy_tool(name: str, description: str, *, defer_loading: bool = True) -> FunctionTool:
    return {
        'type': 'function',
        'name': name,
        'description': description,
        'parameters': {
            'type': 'object',
            'properties': {'order_id': {'type': 'string'}},
            'required': ['order_id'],
            'additionalProperties': False,
        },
        'strict': True,
        'defer_loading': defer_loading,
    }


REFUND_TOOL = _policy_tool('lookup_refund_policy', 'Look up the refund policy for an order.')
EXCHANGE_TOOL = _policy_tool('lookup_exchange_policy', 'Look up the exchange policy for an order.')
FLAT_TOOLS: list[FunctionTool] = [REFUND_TOOL, EXCHANGE_TOOL]

HOSTED_TOOL_SEARCH: dict[str, Any] = {'type': 'tool_search'}
CLIENT_TOOL_SEARCH: dict[str, Any] = {
    'type': 'tool_search',
    'execution': 'client',
    'description': 'Search for relevant tools.',
    'parameters': {
        'type': 'object',
        'properties': {'queries': {'type': 'array', 'items': {'type': 'string'}}},
        'required': ['queries'],
        'additionalProperties': False,
    },
}

NAMESPACE_TOOL: dict[str, Any] = {
    'type': 'namespace',
    'name': 'refunds',
    'description': 'Refund and exchange policy tools.',
    'tools': FLAT_TOOLS,
}


def _dump_item(item: Any) -> ResponseItem:
    if hasattr(item, 'model_dump'):
        return item.model_dump(mode='json', exclude_none=True)
    assert isinstance(item, dict)
    return item


def _print_json(label: str, value: Any) -> None:
    print(f'\n{label}')
    print(json.dumps(value, indent=2, sort_keys=True))


def _summarize_output_items(items: Iterable[Any]) -> list[ResponseItem]:
    summary: list[ResponseItem] = []
    for item in items:
        data = _dump_item(item)
        compact = {
            key: data[key]
            for key in (
                'type',
                'name',
                'namespace',
                'call_id',
                'arguments',
                'execution',
                'status',
            )
            if key in data
        }
        if data.get('type') == 'tool_search_output':
            compact['tools'] = [
                {
                    key: tool[key]
                    for key in ('type', 'name', 'defer_loading')
                    if isinstance(tool, dict) and key in tool
                }
                for tool in data.get('tools', [])
            ]
        summary.append(compact)
    return summary


def _synthetic_tool_search_items(
    *,
    tools: list[dict[str, Any]],
    function_name: str,
    namespace: str | None,
    call_id_prefix: str,
) -> list[ResponseItem]:
    function_call: ResponseItem = {
        'type': 'function_call',
        'name': function_name,
        'arguments': '{"order_id":"order-123"}',
        'call_id': f'{call_id_prefix}_function',
    }
    if namespace is not None:
        function_call['namespace'] = namespace

    return [
        {'role': 'user', 'content': 'Previous turn: find and call the relevant policy tool.'},
        {
            'type': 'tool_search_call',
            'execution': 'client',
            'arguments': {'queries': ['refunds']},
            'call_id': f'{call_id_prefix}_search',
            'status': 'completed',
        },
        {
            'type': 'tool_search_output',
            'execution': 'client',
            'call_id': f'{call_id_prefix}_search',
            'status': 'completed',
            'tools': tools,
        },
        function_call,
        {
            'type': 'function_call_output',
            'call_id': f'{call_id_prefix}_function',
            'output': 'order-123: policy probe output',
        },
        {'role': 'user', 'content': 'Use the prior tool result to answer in one short sentence.'},
    ]


async def _expect_accepts(
    client: AsyncOpenAI,
    *,
    model: str,
    label: str,
    input_items: list[ResponseItem],
    tools: list[dict[str, Any]],
) -> bool:
    try:
        response = await client.responses.create(
            model=model,
            input=input_items,
            instructions='Use prior tool results only. Do not call another tool.',
            tools=tools,
            tool_choice='none',
            store=False,
        )
    except BadRequestError as e:
        print(f'\n[FAIL: rejected unexpectedly] {label}')
        print(getattr(e, 'message', str(e)))
        return False

    print(f'\n[PASS: accepted] {label}')
    print(f'output_text={response.output_text!r}')
    return True


async def _expect_continuation_tool_call(
    client: AsyncOpenAI,
    *,
    model: str,
    label: str,
    input_items: list[ResponseItem],
    tools: list[dict[str, Any]],
    expected_function_name: str,
) -> bool:
    try:
        response = await client.responses.create(
            model=model,
            input=input_items,
            instructions=(
                f'You must call `{expected_function_name}` for order order-456. '
                'Do not answer in text before calling the tool.'
            ),
            tools=tools,
            tool_choice='required',
            parallel_tool_calls=False,
            store=False,
        )
    except BadRequestError as e:
        print(f'\n[FAIL: continuation rejected unexpectedly] {label}')
        print(getattr(e, 'message', str(e)))
        return False

    items = [_dump_item(item) for item in response.output]
    _print_json(f'continuation output: {label}', _summarize_output_items(items))
    function_calls = [item for item in items if item.get('type') == 'function_call']
    if not function_calls:
        print(f'\n[FAIL: no continuation function_call] {label}')
        return False

    expected_calls = [call for call in function_calls if call.get('name') == expected_function_name]
    if not expected_calls:
        print(f'\n[FAIL: wrong continuation function_call] {label}')
        return False

    print(f'\n[PASS: continuation called expected tool] {label}')
    return True


async def _expect_rejects(
    client: AsyncOpenAI,
    *,
    model: str,
    label: str,
    input_items: list[ResponseItem],
    tools: list[dict[str, Any]],
) -> bool:
    try:
        await client.responses.create(
            model=model,
            input=input_items,
            instructions='Use prior tool results only. Do not call another tool.',
            tools=tools,
            tool_choice='none',
            store=False,
        )
    except BadRequestError as e:
        print(f'\n[PASS: rejected] {label}')
        print(getattr(e, 'message', str(e)))
        return True

    print(f'\n[FAIL: accepted unexpectedly] {label}')
    return False


async def _run_synthetic_replay_matrix(client: AsyncOpenAI, model: str) -> None:
    print('\n=== Synthetic client-executed replay: flat deferred function tools ===')
    flat_request_tools = [CLIENT_TOOL_SEARCH, *FLAT_TOOLS]

    for function_name in ('lookup_refund_policy', 'lookup_exchange_policy'):
        await _expect_rejects(
            client,
            model=model,
            label=f'flat {function_name}: missing namespace',
            input_items=_synthetic_tool_search_items(
                tools=FLAT_TOOLS,
                function_name=function_name,
                namespace=None,
                call_id_prefix=f'flat_missing_{function_name}',
            ),
            tools=flat_request_tools,
        )
        await _expect_accepts(
            client,
            model=model,
            label=f'flat {function_name}: namespace == function name',
            input_items=_synthetic_tool_search_items(
                tools=FLAT_TOOLS,
                function_name=function_name,
                namespace=function_name,
                call_id_prefix=f'flat_correct_{function_name}',
            ),
            tools=flat_request_tools,
        )
        await _expect_rejects(
            client,
            model=model,
            label=f'flat {function_name}: random namespace',
            input_items=_synthetic_tool_search_items(
                tools=FLAT_TOOLS,
                function_name=function_name,
                namespace='definitely_wrong_namespace',
                call_id_prefix=f'flat_wrong_{function_name}',
            ),
            tools=flat_request_tools,
        )

    print('\n=== Synthetic replay continuation: does namespace value affect later tool calls? ===')
    await _expect_continuation_tool_call(
        client,
        model=model,
        label='flat replay with namespace == function name',
        input_items=_synthetic_tool_search_items(
            tools=FLAT_TOOLS,
            function_name='lookup_refund_policy',
            namespace='lookup_refund_policy',
            call_id_prefix='flat_continue_correct',
        ),
        tools=flat_request_tools,
        expected_function_name='lookup_refund_policy',
    )
    await _expect_continuation_tool_call(
        client,
        model=model,
        label='flat replay with random namespace',
        input_items=_synthetic_tool_search_items(
            tools=FLAT_TOOLS,
            function_name='lookup_refund_policy',
            namespace='definitely_wrong_namespace',
            call_id_prefix='flat_continue_wrong',
        ),
        tools=flat_request_tools,
        expected_function_name='lookup_refund_policy',
    )

    print('\n=== Synthetic client-executed replay: explicit namespace tool ===')
    namespaced_request_tools = [CLIENT_TOOL_SEARCH, NAMESPACE_TOOL]
    namespace_output_tools = [NAMESPACE_TOOL]

    await _expect_rejects(
        client,
        model=model,
        label='explicit NamespaceTool: missing namespace',
        input_items=_synthetic_tool_search_items(
            tools=namespace_output_tools,
            function_name='lookup_refund_policy',
            namespace=None,
            call_id_prefix='namespace_missing',
        ),
        tools=namespaced_request_tools,
    )
    await _expect_accepts(
        client,
        model=model,
        label='explicit NamespaceTool: namespace == refunds',
        input_items=_synthetic_tool_search_items(
            tools=namespace_output_tools,
            function_name='lookup_refund_policy',
            namespace='refunds',
            call_id_prefix='namespace_correct',
        ),
        tools=namespaced_request_tools,
    )
    await _expect_rejects(
        client,
        model=model,
        label='explicit NamespaceTool: namespace == function name',
        input_items=_synthetic_tool_search_items(
            tools=namespace_output_tools,
            function_name='lookup_refund_policy',
            namespace='lookup_refund_policy',
            call_id_prefix='namespace_function_name',
        ),
        tools=namespaced_request_tools,
    )


async def _observe_hosted_tool_search(client: AsyncOpenAI, model: str) -> None:
    print('\n=== Observed hosted tool_search output: flat deferred function tools ===')
    for function_name in ('lookup_refund_policy', 'lookup_exchange_policy'):
        response = await client.responses.create(
            model=model,
            input=f'Use tool search, then call {function_name} for order order-123.',
            instructions='Call the requested policy function. Do not answer in text before calling it.',
            tools=[*FLAT_TOOLS, HOSTED_TOOL_SEARCH],
            tool_choice='auto',
            parallel_tool_calls=False,
            store=False,
        )
        _print_json(f'hosted observed output for {function_name}', _summarize_output_items(response.output))


async def _observe_client_tool_search(client: AsyncOpenAI, model: str) -> None:
    print('\n=== Observed client-executed tool_search output: flat deferred function tools ===')
    first = await client.responses.create(
        model=model,
        input='Use tool search, then call lookup_refund_policy for order order-123.',
        instructions='Search for the requested policy tool before calling it.',
        tools=[CLIENT_TOOL_SEARCH, *FLAT_TOOLS],
        tool_choice='auto',
        parallel_tool_calls=False,
        store=False,
    )
    first_items = [_dump_item(item) for item in first.output]
    _print_json('client first response output', _summarize_output_items(first_items))

    search_calls = [item for item in first_items if item.get('type') == 'tool_search_call']
    if not search_calls:
        print('\n[SKIP] client observation: first response did not emit a tool_search_call')
        return

    search_call = search_calls[0]
    second_input = [
        *first_items,
        {
            'type': 'tool_search_output',
            'execution': 'client',
            'call_id': search_call['call_id'],
            'status': 'completed',
            'tools': FLAT_TOOLS,
        },
    ]
    second = await client.responses.create(
        model=model,
        input=second_input,
        instructions='Call lookup_refund_policy for order order-123 after using the search result.',
        tools=[CLIENT_TOOL_SEARCH, *FLAT_TOOLS],
        tool_choice='auto',
        parallel_tool_calls=False,
        store=False,
    )
    _print_json('client second response output', _summarize_output_items(second.output))


def _summarize_pydantic_messages(messages: list[Any]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for message in messages:
        parts: list[dict[str, Any]] = []
        for part in getattr(message, 'parts', []):
            item: dict[str, Any] = {'part': type(part).__name__}
            for attr in ('tool_name', 'args', 'content', 'tool_call_id', 'provider_name', 'provider_details'):
                if hasattr(part, attr):
                    value = getattr(part, attr)
                    if value is not None:
                        item[attr] = value
            parts.append(item)
        summary.append({'message': type(message).__name__, 'parts': parts})
    return summary


async def _observe_pydantic_ai_deferred_capability(model: str) -> None:
    """Run the exact Pydantic AI shape: one deferred capability owns two tools."""
    from pydantic_ai import Agent, FunctionToolset
    from pydantic_ai.capabilities.capability import Capability
    from pydantic_ai.messages import ModelResponse, ToolCallPart
    from pydantic_ai.models.openai import OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider

    print('\n=== Pydantic AI deferred capability: two tools in one capability ===')

    toolset = FunctionToolset[None]()

    @toolset.tool_plain
    def lookup_refund_policy(order_id: str) -> str:
        """Look up the refund policy for an order."""
        return f'{order_id}: refund allowed for 30 days'

    @toolset.tool_plain
    def lookup_exchange_policy(order_id: str) -> str:
        """Look up the exchange policy for an order."""
        return f'{order_id}: exchange allowed for 14 days'

    capability = Capability[None](
        id='refunds',
        description='Refund and exchange policy tools.',
        instructions='Use the requested policy tool before answering policy questions.',
        toolset=toolset,
        defer_loading=True,
    )

    agent: Agent[None, str] = Agent(
        model=OpenAIResponsesModel(model, provider=OpenAIProvider()),
        capabilities=[capability],
    )

    for tool_name, order_id in (
        ('lookup_refund_policy', 'order-123'),
        ('lookup_exchange_policy', 'order-456'),
    ):
        result = await agent.run(
            (
                f'Load the refunds capability, then call `{tool_name}` for {order_id}. '
                'Do not answer until after the tool has been called.'
            ),
            model_settings={'openai_store': False},
        )
        messages = result.all_messages()
        tool_calls = [
            part
            for message in messages
            if isinstance(message, ModelResponse)
            for part in message.parts
            if isinstance(part, ToolCallPart) and part.tool_name in {'lookup_refund_policy', 'lookup_exchange_policy'}
        ]

        print(f'\nPydantic AI run requesting {tool_name}')
        if not tool_calls:
            print('[NO POLICY TOOL CALL FOUND]')
            _print_json('message summary', _summarize_pydantic_messages(messages))
            continue

        for call in tool_calls:
            namespace = (call.provider_details or {}).get('namespace')
            print(
                json.dumps(
                    {
                        'tool_name': call.tool_name,
                        'namespace': namespace,
                        'provider_name': call.provider_name,
                        'args': call.args,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=os.getenv('OPENAI_MODEL', 'gpt-5.4'))
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run only synthetic replay validation; skips model-choice observation requests.',
    )
    parser.add_argument(
        '--pydantic-ai',
        action='store_true',
        help='Also run the end-to-end Pydantic AI deferred-capability namespace probe.',
    )
    parser.add_argument(
        '--pydantic-ai-only',
        action='store_true',
        help='Run only the Pydantic AI deferred-capability namespace probe.',
    )
    args = parser.parse_args()

    if not os.getenv('OPENAI_API_KEY'):
        raise SystemExit('OPENAI_API_KEY is not set.')

    client = AsyncOpenAI()
    print(f'Using model: {args.model}')

    if args.pydantic_ai_only:
        await _observe_pydantic_ai_deferred_capability(args.model)
        return

    await _run_synthetic_replay_matrix(client, args.model)
    if not args.quick:
        await _observe_hosted_tool_search(client, args.model)
        await _observe_client_tool_search(client, args.model)
    if args.pydantic_ai:
        await _observe_pydantic_ai_deferred_capability(args.model)


if __name__ == '__main__':
    asyncio.run(main())
