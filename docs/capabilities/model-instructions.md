---
description: "Send each model the instructions it needs, matched by model name or computed from the run context, with the Pydantic AI ModelInstructions capability."
---

# Model Instructions

[`ModelInstructions`][pydantic_ai.capabilities.ModelInstructions] is a [capability](overview.md) that adds instructions depending on the model that handles each request. Some models need a sentence others don't, to counter something they tend to do: pass a mapping from model name patterns to text, and each model gets the text its pattern maps to.

```python {title="model_instructions.py"}
from pydantic_ai import Agent, ModelMessage, ModelRequest, ModelResponse, TextPart
from pydantic_ai.capabilities import ModelFallback, ModelInstructions
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.models.function import AgentInfo, FunctionModel


def overloaded(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    raise ModelAPIError(model_name='gpt-6-luna', message='Overloaded')


def answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart('Paris.')])


# These deterministic responses stand in for a GPT model that's down and a Claude model.
gpt = FunctionModel(overloaded, model_name='gpt-6-luna')
claude = FunctionModel(answer, model_name='claude-opus-5-5')

agent = Agent(
    gpt,
    instructions='You are a geography tutor.',
    capabilities=[
        ModelInstructions(
            {
                'gpt-6*': "Don't spawn sub-agents unless the user asks.",
                'claude-*': 'Keep tool calls to one per turn.',
            }
        ),
        ModelFallback(claude),
    ],
)

result = agent.run_sync('What is the capital of France?')
request = result.all_messages()[0]
assert isinstance(request, ModelRequest)
print(request.instructions)
"""
You are a geography tutor.

Keep tool calls to one per turn.
"""
```

The GPT model was sent its own sentence, and when it failed, [`ModelFallback`](model-fallback.md#instructions-for-each-model) resolved the instructions again, so the Claude model that answered got its own.

## Matching models

The patterns are [glob patterns](https://docs.python.org/3/library/fnmatch.html), matched case-sensitively, and the first one that matches wins, in the mapping's order:

- A pattern without a `:` is matched against the model name, so `'gpt-6*'` matches a GPT-6 model whichever route it takes: `openai:gpt-6-luna`, `azure:gpt-6-luna`, or `gateway/openai:gpt-6-luna`. For a name that starts with its vendor, like OpenRouter's `openai/gpt-6-luna`, it's also matched against the part after the last `/`. For other naming schemes, such as Bedrock's `us.anthropic.claude-...`, start the pattern with `*`.
- A pattern with a `:` is matched against the model ID, `<provider>:<model name>`, so `'azure:gpt-6*'` matches GPT-6 only on Azure. The provider is the one serving the model: `openai` for OpenAI and the Gateway's OpenAI route, `openrouter` for any model on OpenRouter.
- `'*'` matches every model, so it goes last, for the text every other model gets.

A model that no pattern matches gets no text from the capability.

To decide from the model's [profile](../models/overview.md#inspecting-a-models-profile), its behavioral traits rather than its name, or from the run's dependencies, pass a function instead. It takes the [`RunContext`][pydantic_ai.tools.RunContext], whose `model` is the model handling the request, and returns the text or `None`, like an [instruction function](../agent.md#instructions); it may be sync or async:

```python {title="model_instructions_function.py"}
from pydantic_ai import Agent, ModelRequest, RunContext
from pydantic_ai.capabilities import ModelInstructions
from pydantic_ai.models import Model
from pydantic_ai.models.test import TestModel


def delegation(ctx: RunContext[str]) -> str | None:
    if isinstance(ctx.model, Model) and ctx.model.profile.get('delegates_unprompted', False):
        return f"Don't spawn sub-agents unless {ctx.deps} asks."
    return None


agent = Agent(deps_type=str, capabilities=[ModelInstructions(delegation)])

result = agent.run_sync('Hi!', deps='the user', model=TestModel(profile={'delegates_unprompted': True}))
request = result.all_messages()[0]
assert isinstance(request, ModelRequest)
print(request.instructions)
#> Don't spawn sub-agents unless the user asks.
```

## The instruction block

The text is sent as one [named block](../agent.md#appending-instruction-changes), `capability:model_instructions:text`, after the instructions that come before the capability: the agent's own and those of capabilities listed earlier. It never replaces them. To use several, give each its own `id`, which names its block `capability:<id>:text`; two that share an id and differ raise a [`UserError`][pydantic_ai.exceptions.UserError].

A mapping's text depends only on the model, so it's cached with the agent's literal instructions and stays the same on every request to that model. A function can read anything in the run context, so its text is [dynamic](../agent.md#instructions) and goes after the cached part.

When a conversation moves to another model, because a request fell back or a later run continues the history on another model, the change is appended with [`on_change='append'`][pydantic_ai.messages.InstructionPart.on_change]: the new model sees the earlier model's text, then its own as a change at that point. Models tracked the change more faithfully this way, and were harder to talk out of the new text with a `<system>` lookalike, than when the instructions were rewritten as if the conversation had started on the new model. For a line only some models need, with nothing for the others, a rewritten block does better: see [Instructions for specific models](../agent.md#model-specific-instructions). A known cost is listed under [Where no rendering we tried does well](../message-history.md).

## Fallback and realtime models

A [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] picks its model inside the model call, so there's no one model to match: the capability sends no text and warns once. Use the [`ModelFallback`](model-fallback.md) capability instead, which resolves the instructions for each model it tries.

[`SelectModel`](select-model.md) chooses the model before the request, so its choice is the model that's matched. On a [realtime model](../realtime/overview.md), the mapping is matched against its name the same way.

## Agent specs

In an [agent spec](../agent-spec.md), the capability takes the mapping:

```yaml
capabilities:
  - ModelInstructions:
      instructions:
        gpt-6*: Don't spawn sub-agents unless the user asks.
        claude-*: Keep tool calls to one per turn.
```

A function can only be passed in code.
