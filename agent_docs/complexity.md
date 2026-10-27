# Cognitive Complexity

> How to fix a function that `scripts/check_complexity.py` reports, in a way that fits this repo

**When to check**: When pre-commit or CI reports a function's cognitive complexity, or before you add a branch to a function that is already over the limit

`scripts/check_complexity.py` scores functions with [complexipy](https://github.com/rohaquinlop/complexipy) and compares each one with its version before the change. The limit is `max-complexity-allowed` in `[tool.complexipy]` in `pyproject.toml`, and the check covers the packages listed in `paths` there. A change fails when it adds a function over the limit, takes a function over the limit, or makes a function that is already over the limit more complex. Functions already over the limit pass until a change makes them worse. A function moved under the same name is compared with the function it replaces. A renamed function passes when its score is unchanged, and counts as new when it also grew. So does a function moved in from outside the checked packages.

Pre-commit runs the check on the staged changes, and CI runs it on the pull request. To check what your uncommitted edits add, or what the whole branch adds:

```bash
uv run scripts/check_complexity.py --base HEAD
uv run scripts/check_complexity.py --base "$(git merge-base HEAD main)"
```

## Reading a report

```text
pydantic_ai_slim/pydantic_ai/agent/__init__.py:4443: _PreparedAgentRun::open 31 -> 35: already over the limit, and this change made it worse (agent_docs/complexity.md#existing-debt)
    C003 at line 4512, about -12: move the block into a private method, or a function next to its caller (agent_docs/complexity.md#extract-a-method)
    C003 at line 4470, about -9: move the block into a private method, or a function next to its caller (agent_docs/complexity.md#extract-a-method)
    C005 at line 4459, about -2: name the compound condition as a predicate method (agent_docs/complexity.md#extract-a-method)
    pick the case whose shape matches the function, and apply its fix (agent_docs/complexity.md#cases)
```

- The first line names the function, its score before and after the change, and the [failure](#failures).
- Each line that starts with a rule ID, such as `C003`, is a complexipy suggestion, pointed at a [fix](#fixes). The reduction is complexipy's estimate.
- The last line points at the [cases](#cases). Each case pairs a shape with the fix that fits it in this repo. Pick the case whose **Shape** describes the function, and apply its fix.

## Failures

### New function

The function did not exist before the change. Split it before it lands: apply the fix for its [case](#cases), starting with complexipy's suggestions.

### Crossed the limit

The function was within the limit before the change. Extract the block you added, or apply the cheapest suggestion. One extraction is usually enough, because most functions that cross the limit land at 16 to 18.

### Existing debt

The function was already over the limit, and the change made it worse. Keep the pull request on its own task. Do not restructure the whole function. Put the logic you added in a method, so the function grows by the call alone. When your addition is one more branch of a type or key dispatch, turn the dispatch into `match` once, as [Dispatch on type](#dispatch-on-type) describes. Each later `case` then adds nothing.

## Cases

### part-ladder

**Shape**: maps message history to or from a wire format, a UI protocol or OTel. It loops over messages, branches on `ModelRequest` and `ModelResponse`, loops over parts, and runs an `isinstance` ladder over part types, with ladders over native tools or content items inside.

**Fix**:

- Turn each ladder into `match part:` with one `case` per part type and `case _: assert_never(part)`.
- Map each part type in its own method. When request-wide state crosses parts, keep it on a private, data-only context dataclass, one per request, as `OpenAIChatModel._MapModelResponseContext` in `models/openai.py` does.
- Move a whole-request pass that runs after the loop into a module function, as `place_system_messages_before_generation` in `models/anthropic/_message_passes.py` does.
- Move logic every provider repeats into a `Model.prepare_messages` pass, as `_drop_superseded_request_scoped_prompts` in `models/__init__.py` does.

### stream-state-machine

**Shape**: an async generator or event handler over provider, realtime or UI events. It runs an `isinstance` ladder over the event union, keeps state across events, and yields events through `_parts_manager`.

**Fix**:

- Keep per-stream state on dataclass fields with `field(init=False)`, not in locals or closures.
- Handle each event type in a method that returns an `Iterable` of events, as `OpenAIStreamedResponse._map_part_delta` in `models/openai.py` does.
- Dispatch in a sync method with one `match`, and forward its result once in the generator. An async generator cannot `yield from`, so each branch that delegates inside it costs a loop.
- Keep a `case _:` fallback over SDK event unions. Those unions grow with the SDK, so `assert_never` is wrong there.

### orchestration

**Shape**: the run loop, tool execution, the graph scheduler, lifecycle hooks or a receive loop. It is an async generator or context manager with `try`/`finally`, task groups, cancellation or a `yield` that receives a value.

**Fix**: move the pure steps out: predicates, accumulators and reducers. Keep every `yield`, cleanup and cancellation path in the original function, because their order is part of the behavior. `_segment_by_barriers` in `_tool_execution.py` and `_GraphIterator._handle_execution_request` in `pydantic_graph/graph_builder.py` are such steps. Review a change here for behavior even when the diff looks mechanical.

### closure-hosted

**Shape**: at least half the score sits in a nested `def`, closure or inner generator. complexipy scores a nested function as part of the function that defines it.

**Fix**: hoist the nested function to a method or a module function, and pass the state it captured as arguments. Where an API takes a callback, pass the bound method.

### history-pass

**Shape**: a pass over `list[ModelMessage]` that drops, merges, reorders, repairs or synthesizes parts. It nests a loop over messages and a loop over parts, and keeps sets of IDs.

**Fix**: split the pass into an index phase, then a rebuild with a comprehension and `dataclasses.replace`. Skip unwanted items with `continue` guards. When the pass does two jobs, make it two passes in the pipeline. `_drop_compaction_parts` and the pipeline `_clean_message_history` in `messages.py` follow this shape.

### tree-walk

**Shape**: a walk over a tree, such as a JSON schema, an HTML document, a type annotation or a graph, with a branch per node kind and inline accounting.

**Fix**: separate the traversal from one handler method per node kind, as `JsonSchemaTransformer.walk` in `_json_schema.py` does with the handlers its subclasses override. Move repeated budget checks into one method that raises, as `_HTMLMarkupBudget._count` in `common_tools/web_fetch.py` does. A walk that enforces a resource bound must keep its behavior exactly.

### vocab-translator

**Shape**: maps one cross-provider setting to one provider's request parameters: native tools, `tool_choice`, thinking, cache settings, model settings or realtime session config.

**Fix**: `match` over the closed union and end in `case _: assert_never(...)`. Build each variant that needs a nested object in its own method, as `AnthropicModel._map_web_search_tool` in `models/anthropic/__init__.py` does. Normalize the input first and map it second, as `CohereModel._get_tool_choice` in `models/cohere.py` does.

### precedence-resolution

**Shape**: resolves many inputs across layers, such as run, override, spec, agent, capability and default, or a constructor that normalizes many optional parameters.

**Fix**: resolve each layer or phase in its own named function, as `_merge_retries_with_spec` and `_layer_model_settings` in `agent/__init__.py` do. Replace a chain of ternaries over the same layers with one resolver.

### check-chain

**Shape**: a long, flat run of independent checks: validation that raises, predicates that return `False`, or projections such as `if value is not None: out[key] = value`.

**Fix**: group the checks into named predicates along the sections the code already has, as `SpanNode._attribute_matches` in `pydantic_evals/otel/span_tree.py` does. Build `provider_details` and similar projections in one function that the streaming and non-streaming paths share, as `_map_provider_details` in `models/openai.py` does.

### name-dispatch

**Shape**: a flat `if`/`elif` chain on a string or literal key, such as a provider name, a model prefix, a command or a key binding, or an `except` ladder over SDK error classes.

**Fix**: `match key:` with one `case 'a' | 'b':` per target, ending in `case _:` that raises or calls `assert_never`, as `WorkspaceCall.execute` in `durable_exec/_workspace.py` does. For an error ladder, map each error class in its own function, as `_map_api_errors` and `_map_status_error` in `models/openai.py` do.

### presentation

**Shape**: terminal UI and reports: Rich tables and panels, menus, frames, diff rows and prompt loops.

**Fix**: build each row, card or frame in its own method, and use guard clauses. `task_row` in `pydantic_clai2/runtime/tasks.py` is such a builder.

### algorithm

**Shape**: a self-contained algorithm whose branches are its edge cases: parsing, bounds checks, retry timing, protocol handling.

**Fix**: use [guard clauses](#guard-clauses), and name the conditions as predicates, as `_check_domain` in `_ssrf.py` does. A function that enforces a security bound, such as those in `_ssrf.py`, must keep its behavior exactly.

## Fixes

How complexipy counts, which decides which fix pays off:

- `if`, `for`, `while`, `except`, a ternary and each comprehension generator cost 1, plus 1 per level of nesting.
- `elif` and `else` cost 1 each, and each run of boolean operators costs 1.
- A `match` costs 1 plus its nesting once, however many cases it has.
- A `try` body and a `with` body add no nesting. An `except` body does.
- A nested `def`, closure or lambda counts toward the function that defines it, one level deeper.

### Guard clauses

C001, C002. Return early for the cases you can rule out, and skip unwanted items with `continue` at the top of a loop. The main path then runs at a lower nesting level, and every branch on it costs less.

### Extract a method

C003, C005.

- Make the block a private method on the class, or a private function next to its only caller in the same module.
- Never extract it as a nested `def`. complexipy counts a nested `def` toward the enclosing function, one level deeper.
- A local variable does not simplify a compound condition: `ok = a and b` costs what `if a and b:` costs. Name the condition as a predicate method instead.

### Dispatch on type

C004. A `match` costs 1 however many cases it has. An `if`/`elif isinstance` chain costs 1 per branch. Dispatch with `match part:`, one `case TextPart():` per type, and `case _: assert_never(part)`. Keep each case to a call to the method that handles that type. `UIEventStream.handle_part_start` in `ui/_event_stream.py` dispatches to its `handle_*` methods this way; it is an async generator, so each of its cases costs one delegation loop. Ruff's `C901` still counts each `case`, so a large dispatcher can need `# noqa: C901`.

### Merge nested ifs

C007. Merge nested `if` statements with no code between them into `if a and b:`, as [Code Simplification](code-simplification.md) asks.

### Flatten nested try

C011. Move the inner `try` into its own method, so its `except` bodies are not nested inside the outer handler.

## Ignoring a function

Prefer a fix. When a function has to stay complex, mark its `def` line, or the comment line directly above the function and its decorators, with a reason in parentheses:

```python
def _map_messages(self, messages: list[ModelMessage]) -> list[Message]:  # complexipy: ignore (one branch per content type the API accepts)
```

The check fails an ignore without a reason. CI annotates each new ignore so reviewers see it. complexipy drops an ignored function entirely, so its later growth goes unreported.
