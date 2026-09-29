# Goal

Keep an unattended agent working until a caller-defined completion check accepts its final answer. Use `Goal` when nobody is available to answer follow-up questions and a successful run must satisfy a concrete condition.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/goal/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://pydantic.dev/docs/ai/harness/#version-policy).

## Define completion

Supply an explicit `goal` and an async `verify(ctx, output)` callable. The verifier returns `None` to accept the output, or a nonempty string describing what remains. It receives the processed output unchanged, including structured outputs, and can inspect dependencies and message history through `RunContext`.

Output functions run before this check. A rejection can call them again on a later attempt. Keep output functions and verifiers read-only or idempotent; perform irreversible commits only after `agent.run()` returns an accepted result. `Goal` does not roll back tool calls or output-function side effects.

Define completion using evidence you trust. A model declaring success is not proof that files were written, tests passed, or an external operation completed. The capability does not guess completion from punctuation or classify natural-language questions.

```python
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.goal import Goal


class Summary(BaseModel):
    items: list[str]


async def verify(ctx: RunContext[int], output: object) -> str | None:
    if not isinstance(output, Summary) or len(output.items) != ctx.deps:
        return f'Provide exactly {ctx.deps} summary items.'
    return None


agent = Agent(
    TestModel(custom_output_args={'items': ['First finding', 'Second finding']}),
    deps_type=int,
    output_type=Summary,
    retries={'output': 3},
    capabilities=[Goal(goal='Produce a two-item summary.', verify=verify)],
)
result = agent.run_sync('Summarize the findings.', deps=2)
print(result.output.items)
#> ['First finding', 'Second finding']
```

This example uses a local test model so it runs without credentials. Replace it with your production model and a verifier appropriate to the task. For blocking checks, offload the work rather than blocking the agent's event loop.

## Bound continuation

When a final output fails verification, `Goal` raises `ModelRetry` through the public output hook. Core sends the goal and the verifier's explanation back to the model, preserving the conversation and allowing further tool calls. There is no second agent loop or persisted goal object.

Set `Agent(retries={'output': n})` to bound these continuations. Goal rejections share the output retry budget with other output validation failures; they do not reserve independent rounds. Exhaustion raises `UnexpectedModelBehavior` rather than returning an unverified answer. Verifier exceptions propagate, and an empty explanation raises `UserError`.

`UsageLimits` remain authoritative. Reaching a usage limit raises `UsageLimitExceeded`; `Goal` does not bypass the limit with an extra tool-less grace request. Token consumption cannot in general be predicted before a response. Applications needing a final wrap-up should budget for it explicitly rather than rely on a request after exhaustion.

The instructions ask the model to make reasonable decisions, but do not authorize bypassing approvals or inventing credentials. Decide whether an accurately reported blocker satisfies your completion condition. If not, the verifier should reject it and the bounded run will eventually fail rather than claim success.

## Interactive runs and composition

Set `headless=False` to retain the goal instructions without running the verifier. This lets an interactive run ask a question and return control to its user. Mode is explicit; `Goal` does not infer it from the terminal or interface.

Multiple `Goal` instances contribute independent instructions and completion checks. Every check must accept. A capability instance keeps no per-run retry counter and can be reused across runs. Callable verifiers are not serializable in agent specifications.

Use `run()` or `run_sync()` when rejected output must stay private. Partial streamed outputs are not checked. Core does not support output retries through `run_stream()`; a rejected final streamed output raises `UnexpectedModelBehavior`, and the caller may already have received partial content. Deferred tool approvals are not completed answers and remain the host's responsibility.

`Goal` evaluates processed output with outermost ordering inside instrumentation, like `OutputGuardrail`. Other output-changing capabilities at the same ordering position can still run after it; arrange their ordering so verification sees the output you intend to accept. This capability does not replace verification tools, task planning, or a live trajectory judge: those can produce evidence or steer work before completion.

## Telemetry

Each completion check emits a `goal.verify` span through `ctx.tracer`. `goal.met` records whether the check accepted. `goal.description` and, for a rejection, `goal.gap` are recorded only when `trace_include_content` is enabled. Verifier exceptions are recorded by the span. Interactive runs and partial outputs emit no goal spans. Core instrumentation already records the resulting model requests and retry flow.
