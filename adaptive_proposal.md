# AdaptiveModel Design Proposal

## Overview

`AdaptiveModel` is a new model type that provides full control over model selection at runtime. Unlike `FallbackModel` which tries models sequentially, `AdaptiveModel` allows custom logic to select the next model based on rich context including attempts, exceptions, and agent dependencies.

## Core API

```python
class AdaptiveModel[StateT](Model):
    def __init__(
        self,
        selector: Callable[[AdaptiveContext[StateT]], Model]
                  | Callable[[AdaptiveContext[StateT]], Awaitable[Model]],
        *,
        state: StateT | None = None,
        on_attempt_failed: (
            Callable[[StateT, Model, Exception, datetime, timedelta], bool]
            | Callable[[StateT, Model, Exception, datetime, timedelta], Awaitable[bool]]
            | None
        ) = None,
        on_attempt_succeeded: (
            Callable[[StateT, Model, ModelResponse, datetime, timedelta], None]
            | Callable[[StateT, Model, ModelResponse, datetime, timedelta], Awaitable[None]]
            | None
        ) = None,
    ):
        """
        Args:
            selector: Sync or async function that selects the next model to try.
                     Must return a Model or raise an exception to stop.
                     The selector manages its own pool of models (via closure, state, etc.).
            state: State object passed to selector. If None, an empty dict is used.
                   Reuse the same AdaptiveModel instance to share state across runs.
                   Create new instances for isolated state.
            on_attempt_failed: Optional sync or async hook called after each failed attempt.
                              Receives (state, model, exception, timestamp, duration).
                              Must return bool: True to continue, False to stop and re-raise.
            on_attempt_succeeded: Optional sync or async hook called after successful attempt.
                                 Receives (state, model, response, timestamp, duration).
        """

# Usage - state is managed at the model level, NOT passed through Agent
# Models are managed by the selector (via closure, state, etc.)
models = [Model('openai:gpt-4o'), Model('anthropic:claude-3-5-sonnet')]

def my_selector(ctx: AdaptiveContext[MyState]) -> Model:
    # Selector has full control over model selection
    return models[0]  # Can use models from closure, state, or anywhere

adaptive = AdaptiveModel(
    selector=my_selector,
    state=MyState(),  # State lives in the model
    on_attempt_failed=my_failure_handler,  # Optional lifecycle hooks
    on_attempt_succeeded=my_success_handler,
)
agent = Agent(adaptive)

# Just run normally - no changes to Agent API
result = await agent.run('query')
```

## Lifecycle Hooks

AdaptiveModel provides two optional lifecycle hooks for clean separation of concerns:

### `on_attempt_failed`

Called after each failed model attempt. Can be sync or async. **Must return bool**:
- `True`: Continue trying (call selector for next model)
- `False`: Stop immediately (re-raise the exception)

Use for:
- Conditional retry logic (e.g., only retry on specific exceptions)
- Throttling detection and tracking
- Error rate monitoring
- Circuit breaker patterns
- Failure analytics

```python
# Sync hook
def on_failure(state: MyState, model: Model, exception: Exception, timestamp: datetime, duration: timedelta) -> bool:
    """Called when a model attempt fails. Returns True to continue, False to stop."""
    if 'throttl' in str(exception).lower():
        state.throttled_models[id(model)] = timestamp
        return True  # Continue trying with other models

    # For non-retryable errors, stop immediately
    if isinstance(exception, ValueError):
        return False

    return True  # Default: continue

# Async hook
async def on_failure_async(state: MyState, model: Model, exception: Exception, timestamp: datetime, duration: timedelta) -> bool:
    """Called when a model attempt fails - can do async operations."""
    if 'throttl' in str(exception).lower():
        state.throttled_models[id(model)] = timestamp
        # Log to external service
        await log_service.record_throttle(model.model_name, timestamp)
        return True  # Continue trying

    return False  # Stop on other errors
```

### `on_attempt_succeeded`

Called after a successful model attempt. Can be sync or async. Use for:
- Quality tracking and metrics
- Quota deduction
- Cost tracking
- Response caching
- Performance monitoring

```python
# Sync hook
def on_success(state: MyState, model: Model, response: ModelResponse, timestamp: datetime, duration: timedelta):
    """Called when a model attempt succeeds."""
    # Track quality
    quality_score = evaluate_response(response)
    state.quality_history[model.model_name].append(quality_score)

    # Deduct quota
    tokens_used = response.usage.total_tokens if response.usage else 0
    state.quota_remaining -= tokens_used

# Async hook
async def on_success_async(state: MyState, model: Model, response: ModelResponse, timestamp: datetime, duration: timedelta):
    """Called when a model attempt succeeds - can do async operations."""
    # Store metrics in database
    await metrics_db.insert({
        'model': model.model_name,
        'duration': duration.total_seconds(),
        'tokens': response.usage.total_tokens if response.usage else 0,
        'timestamp': timestamp
    })
```

**Note:** For streaming requests, `on_attempt_succeeded` is called when the stream starts, not when it completes.

## Execution Flow

The `AdaptiveModel` automatically handles fallback and retry by:

1. Calling `selector(context)` to get the next model to try
2. Attempting the request with that model
3. If the request **succeeds**:
   - Call `on_attempt_succeeded` hook (if provided)
   - Return the result
4. If the request **fails**:
   - Recording the attempt (model + exception) in `context.attempts`
   - Call `on_attempt_failed` hook (if provided)
     - If hook returns `False`, re-raise the exception immediately
     - If hook returns `True` (or hook not provided), continue to step 5
5. Calling `selector(context)` again with updated context
   - If selector returns a `Model`, goto step 2 (retry/fallback)
   - If selector raises an exception, stop and raise `FallbackExceptionGroup`

The selector has **full control** over retry/fallback logic:
- Return the **same model** that failed → retry with same model
- Return a **different model** → fallback to another model
- Raise an exception → stop trying
- Use `await asyncio.sleep()` for backoff/waiting (selector can be async)

Lifecycle hooks provide **clean separation of concerns**:
- `on_attempt_failed`: Decides **whether to continue** (True/False) + side effects
- `selector`: Decides **which model to use next**
- `on_attempt_succeeded`: Handles **success side effects** (metrics, tracking, etc.)

## Context

```python
@dataclass
class AdaptiveContext[StateT]:
    """Context provided to the selector function."""

    state: StateT  # User-defined state object
    attempts: list[Attempt]  # History of attempts in this request
    messages: list[ModelMessage]  # The original request
    model_settings: ModelSettings | None
    model_request_parameters: ModelRequestParameters

@dataclass
class Attempt:
    """Record of a single attempt."""
    model: Model
    exception: Exception | None
    timestamp: datetime
    duration: timedelta
```

## State Management

State is managed at the `AdaptiveModel` level - it's part of the model instance, not passed through the Agent. This keeps the Agent API unchanged and makes state management explicit and simple.

### Defining State

State can be any Python type - typically a dataclass for complex state or a simple dict for basic cases:

```python
@dataclass
class AdaptiveState:
    """State for adaptive model selection."""
    throttled_models: dict[int, float] = field(default_factory=dict)
    model_call_counts: dict[int, int] = field(default_factory=dict)

# State is part of the model instance
adaptive = AdaptiveModel(
    models=[gpt35, gpt4, claude],
    selector=my_selector,
    state=AdaptiveState()  # Pass state here
)
```

### Concurrency Considerations

⚠️ **Important:** State objects are **not automatically thread-safe or async-safe**.

For **concurrent async requests** (e.g., FastAPI, multiple simultaneous requests):

**✅ Safe: Per-Request State (Recommended)**
```python
@app.post("/query")
async def handle_query(query: str):
    # Create fresh model instance per request - completely safe
    adaptive = AdaptiveModel(
        models=[...],
        selector=my_selector,
        state=MyState()  # Fresh state per request
    )
    agent = Agent(adaptive)
    result = await agent.run(query)
    return result.output
```

**⚠️ Requires Synchronization: Shared State**

If you need shared state across concurrent requests, use locks:

```python
import asyncio

# Global shared state
global_state = LoadBalanceState()
state_lock = asyncio.Lock()

async def locked_selector(ctx: AdaptiveContext[LoadBalanceState]) -> Model | None:
    async with state_lock:
        # Only one request can access state at a time
        model = min(ctx.models, key=lambda m: ctx.state.call_counts.get(id(m), 0))
        ctx.state.call_counts[id(model)] = ctx.state.call_counts.get(id(model), 0) + 1
        return model

# Reused across all requests
adaptive = AdaptiveModel(
    models=[...],
    selector=locked_selector,
    state=global_state
)

@app.post("/query")
async def handle_query(query: str):
    agent = Agent(adaptive)  # Reuses same adaptive instance
    result = await agent.run(query)
    return result.output
```

**Alternative: Lock in Hooks**
```python
state_lock = asyncio.Lock()

async def on_failure_locked(state: MyState, model: Model, exception: Exception, timestamp: datetime, duration: timedelta) -> bool:
    async with state_lock:
        # Thread-safe state update
        if 'throttl' in str(exception).lower():
            state.throttled_models[id(model)] = timestamp
            return True  # Continue trying
    return False  # Stop on other errors

adaptive = AdaptiveModel(
    selector=my_selector,
    state=global_state,
    on_attempt_failed=on_failure_locked  # Hook handles locking
)
```

**When to Use Locks:**
- ✅ Load balancing across requests (shared call counts)
- ✅ Global throttling (shared throttle timers)
- ✅ System-wide quotas (shared quota tracking)
- ❌ Per-request retry logic (use per-request state instead)

### Per-Run State (Isolated)

For state that should be **isolated per run**, create a new `AdaptiveModel` instance for each run:

```python
def selector(ctx: AdaptiveContext[AdaptiveState]) -> Model | None:
    # Access state - unique to this model instance
    if not ctx.attempts:
        return ctx.models[0]
    return None

# Create new model instance for each run
def run_with_fresh_state(query: str):
    adaptive = AdaptiveModel(
        models=[...],
        selector=selector,
        state=AdaptiveState()  # Fresh state
    )
    agent = Agent(adaptive)
    return agent.run_sync(query)

result1 = run_with_fresh_state('query 1')  # Fresh state
result2 = run_with_fresh_state('query 2')  # Fresh state
```

### Global State (Shared)

For state that should be **shared across runs**, reuse the same `AdaptiveModel` instance:

```python
# Global model with shared state
adaptive = AdaptiveModel(
    models=[...],
    selector=selector,
    state=AdaptiveState()  # Shared state
)
agent = Agent(adaptive)

# All runs share the same model instance and state
result1 = await agent.run('query 1')  # Uses shared state
result2 = await agent.run('query 2')  # Same state persists

# State persists: throttling, call counts, etc. are maintained
```

### Per-User State (API Sessions)

For multi-user applications, create a model instance per user:

```python
# Model instance per user
user_models: dict[str, AdaptiveModel] = {}

async def api_endpoint(user_id: str, query: str):
    # Get or create model for this user
    if user_id not in user_models:
        user_models[user_id] = AdaptiveModel(
            models=[...],
            selector=selector,
            state=AdaptiveState(user_tier='premium')
        )

    agent = Agent(user_models[user_id])
    result = await agent.run(query)
    return result.output
```

**Key Differences:**

| Pattern | Model Lifetime | Use Case |
|---------|---------------|----------|
| **Per-Run** | New instance per run | Retry logic within one request |
| **Global** | Single instance | Throttling, load balancing, system-wide quotas |
| **Per-User** | Instance per user | User-specific quotas, preferences, history |

## Use Cases

### 1. Throttling with Timeout (Global State)

Handle rate limiting by timing out models for 30 seconds after throttling errors.

⚠️ **Concurrency:** This example uses shared state across concurrent requests with proper locking.

```python
import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta

# Models managed by selector (via closure)
models = [primary, secondary, tertiary]

@dataclass
class ThrottleState:
    throttled_models: dict[int, datetime] = field(default_factory=dict)  # model_id -> timestamp
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

async def on_throttle_error(
    state: ThrottleState,
    model: Model,
    exception: Exception,
    timestamp: datetime,
    duration: timedelta
) -> bool:
    """Hook: Record throttled models and decide whether to continue."""
    if 'throttl' in str(exception).lower():
        async with state.lock:
            state.throttled_models[id(model)] = timestamp
        return True  # Continue trying other models
    return False  # Stop on non-throttle errors

async def throttle_aware_selector(ctx: AdaptiveContext[ThrottleState]) -> Model:
    """Selector: Choose first available non-throttled model."""
    async with ctx.state.lock:
        now = datetime.now()
        # Find first available model
        for model in models:
            model_id = id(model)
            if model_id in ctx.state.throttled_models:
                if (now - ctx.state.throttled_models[model_id]).total_seconds() < 30:
                    continue
                del ctx.state.throttled_models[model_id]
            return model

        # All throttled - wait for soonest available
        if ctx.state.throttled_models and len(ctx.attempts) < 10:
            soonest = min(ctx.state.throttled_models.items(), key=lambda x: x[1])
            wait_time = 30 - (now - soonest[1]).total_seconds()
            if wait_time > 0:
                # Release lock during sleep
                pass
        else:
            raise RuntimeError("All models throttled")

    # Sleep outside lock
    if wait_time > 0:
        await asyncio.sleep(wait_time)
        async with ctx.state.lock:
            del ctx.state.throttled_models[soonest[0]]
            return next(m for m in models if id(m) == soonest[0])

    raise RuntimeError("All models throttled")

adaptive = AdaptiveModel(
    selector=throttle_aware_selector,
    state=ThrottleState(),
    on_attempt_failed=on_throttle_error,
)

# Reuse same model instance - state persists across runs
agent = Agent(adaptive)

result1 = await agent.run('query 1')  # Uses shared throttle state
result2 = await agent.run('query 2')  # Same state persists
```

**Benefits of using hook:**
- ✅ Hook decides whether to continue based on exception type
- ✅ Selector focuses on selection logic only
- ✅ Cleaner separation of concerns
- ✅ Lock ensures thread-safe state updates

### 2. Load Balancing (Global State)

Distribute requests evenly across multiple accounts/instances.

⚠️ **Concurrency:** This example uses shared state across concurrent requests with proper locking.

```python
# Models managed by selector (via closure)
models = [
    OpenAIChatModel('gpt-4o', api_key=key1),
    OpenAIChatModel('gpt-4o', api_key=key2),
]

@dataclass
class LoadBalanceState:
    call_counts: dict[int, int] = field(default_factory=dict)  # model_id -> count
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

async def round_robin_selector(ctx: AdaptiveContext[LoadBalanceState]) -> Model:
    async with ctx.state.lock:
        if not ctx.attempts:
            # Use least-used model
            model = min(models, key=lambda m: ctx.state.call_counts.get(id(m), 0))
            ctx.state.call_counts[id(model)] = ctx.state.call_counts.get(id(model), 0) + 1
            return model

        # On retry, try next least-used
        failed = {id(a.model) for a in ctx.attempts}
        available = [m for m in models if id(m) not in failed]
        if available:
            return min(available, key=lambda m: ctx.state.call_counts.get(id(m), 0))
        raise RuntimeError("All models failed")

# Load balance across accounts
adaptive = AdaptiveModel(
    selector=round_robin_selector,
    state=LoadBalanceState()  # Shared state in model
)

# Reuse same model instance - state persists across runs
agent = Agent(adaptive)

result1 = await agent.run('query 1')  # Uses account 1
result2 = await agent.run('query 2')  # Uses account 2
result3 = await agent.run('query 3')  # Uses account 1
```

### 3. User Tier-Based Selection (Per-User State)

Route to different models based on user subscription level.

⚠️ **Concurrency:** If the same user makes concurrent requests, add locking. Otherwise, per-user state is isolated.

```python
# Models managed by selector (via closure)
models = [gpt35, gpt4mini, gpt4]

@dataclass
class UserState:
    tier: str  # 'free', 'pro', 'enterprise'
    monthly_tokens_used: int
    monthly_token_limit: int
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)  # For concurrent requests from same user

async def tier_based_selector(ctx: AdaptiveContext[UserState]) -> Model:
    async with ctx.state.lock:
        user = ctx.state

        if not ctx.attempts:
            if user.tier == 'enterprise':
                return next((m for m in models if 'gpt-4o' in m.model_name), models[0])
            elif user.tier == 'pro':
                # Check usage limits
                if user.monthly_tokens_used < user.monthly_token_limit * 0.9:
                    return next((m for m in models if 'gpt-4o-mini' in m.model_name), models[0])
                return next((m for m in models if 'gpt-3.5' in m.model_name), models[0])
            else:
                return next((m for m in models if 'gpt-3.5' in m.model_name), models[0])

        # Retry with next model
        tried = {id(a.model) for a in ctx.attempts}
        available = [m for m in models if id(m) not in tried]
        if available:
            return available[0]
        raise RuntimeError("All models failed")

# Per-user model instances in API endpoint
user_models: dict[str, AdaptiveModel] = {}

async def api_endpoint(user_id: str, query: str):
    if user_id not in user_models:
        user_models[user_id] = AdaptiveModel(
            selector=tier_based_selector,
            state=UserState(tier='pro', monthly_tokens_used=80000, monthly_token_limit=100000)
        )

    agent = Agent(user_models[user_id])
    result = await agent.run(query)
    return result.output
```

### 4. Cost-Optimized with Quality Fallback (Global State)

Start with cheap models, upgrade if quality is insufficient.

⚠️ **Concurrency:** This example uses shared state across concurrent requests with proper locking.

```python
# Models managed by selector (via closure), sorted by cost
models = [gpt35, gpt4mini, gpt4]

@dataclass
class QualityState:
    quality_threshold: float = 0.8
    model_history: dict[str, list[float]] = field(default_factory=dict)  # model -> quality scores
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

async def cost_quality_selector(ctx: AdaptiveContext[QualityState]) -> Model:
    async with ctx.state.lock:
        if not ctx.attempts:
            # Check if cheap model historically meets quality threshold
            cheap_model = models[0]
            avg_quality = (
                sum(ctx.state.model_history.get(cheap_model.model_name, [1.0])) /
                len(ctx.state.model_history.get(cheap_model.model_name, [1.0]))
            )

            if avg_quality >= ctx.state.quality_threshold:
                return cheap_model
            # Quality too low, start with better model
            return models[1] if len(models) > 1 else cheap_model

        # Upgrade on failure
        tried = {id(a.model) for a in ctx.attempts}
        available = [m for m in models if id(m) not in tried]
        if available:
            return available[0]
        raise RuntimeError("All models failed")

async def on_success_track_quality(
    state: QualityState,
    model: Model,
    response: ModelResponse,
    timestamp: datetime,
    duration: timedelta
):
    """Hook: Track quality metrics."""
    quality_score = await evaluate_response_quality(response)
    async with state.lock:
        if model.model_name not in state.model_history:
            state.model_history[model.model_name] = []
        state.model_history[model.model_name].append(quality_score)
        # Keep only last 100 scores
        state.model_history[model.model_name] = state.model_history[model.model_name][-100:]

adaptive = AdaptiveModel(
    selector=cost_quality_selector,
    state=QualityState(),
    on_attempt_succeeded=on_success_track_quality  # Automatic quality tracking
)

# Reuse same model instance - quality history persists
agent = Agent(adaptive)

result = await agent.run('query')
```

### 5. Smart Retry with Exponential Backoff (Per-Run State)

Retry same model with backoff for transient errors, fallback for permanent errors.

```python
import asyncio
from typing import Any

# Models managed by selector (via closure)
models = [primary, backup]

def on_failure_check(state: Any, model: Model, exception: Exception, timestamp: datetime, duration: timedelta) -> bool:
    """Decide whether to retry based on exception type."""
    # Only retry on transient errors (5xx)
    is_transient = (
        hasattr(exception, 'status_code') and
        500 <= exception.status_code < 600
    )
    return is_transient  # True = continue, False = stop

async def exponential_backoff_selector(ctx: AdaptiveContext[Any]) -> Model:
    # No state needed - retry logic is per-run only
    if not ctx.attempts:
        return models[0]

    last = ctx.attempts[-1]
    is_transient = (
        last.exception and
        hasattr(last.exception, 'status_code') and
        500 <= last.exception.status_code < 600
    )

    if is_transient and len(ctx.attempts) <= 5:
        # Exponential backoff: 1s, 2s, 4s, 8s, 16s
        await asyncio.sleep(2 ** (len(ctx.attempts) - 1))
        return last.model  # Retry same model

    # Try different model
    tried = {id(a.model) for a in ctx.attempts}
    available = [m for m in models if id(m) not in tried]
    if available:
        return available[0]
    raise RuntimeError("All models failed")

adaptive = AdaptiveModel(
    selector=exponential_backoff_selector,
    on_attempt_failed=on_failure_check  # Stop on non-transient errors
    # No state parameter - defaults to empty dict
)

agent = Agent(adaptive)
# Each run has independent retry logic based on ctx.attempts
result = await agent.run('query')
```

### 6. Context Length-Based Model Upgrade (Per-Run State)

Automatically upgrade to a long-context model when conversation exceeds a threshold.

```python
from typing import Any

# Models sorted by context length capability
models = [
    OpenAIChatModel('gpt-4o-mini'),           # 128K context
    OpenAIChatModel('gpt-4o'),                # 128K context
    AnthropicModel('claude-3-7-sonnet-latest'),  # 1M context
]

def context_aware_selector(ctx: AdaptiveContext[Any]) -> Model:
    """Upgrade to long-context model when conversation gets large."""
    # No state needed - decision based on current message count only

    message_count = len(ctx.messages)

    if not ctx.attempts:
        # First attempt - choose based on context size
        if message_count > 50:
            # Use long-context model for large conversations
            return next((m for m in models if 'claude-3-7-sonnet' in m.model_name), models[0])
        else:
            # Use standard model for normal conversations
            return next((m for m in models if 'gpt-4o-mini' in m.model_name), models[0])

    # On retry, try next available model
    tried = {id(a.model) for a in ctx.attempts}
    available = [m for m in models if id(m) not in tried]
    if available:
        return available[0]
    raise RuntimeError("All models failed")

adaptive = AdaptiveModel(
    selector=context_aware_selector
    # No state parameter - decision based on message count only
)

agent = Agent(adaptive)

# Short conversation uses cheap model
result1 = agent.run_sync('Hello')  # Uses gpt-4o-mini

# Long conversation automatically upgrades
messages = result1.new_messages()
for i in range(60):
    result = agent.run_sync(f'Message {i}', message_history=messages)
    messages = result.new_messages()
# Automatically switched to claude-3-7-sonnet due to message count
```

### 7. Self-Directed Model Selection via Tool (Per-Run State)

Agent dynamically selects its own model for the next cycle based on task complexity.

```python
from pydantic import BaseModel
from typing import Any, Literal

# Models managed by selector (via closure)
models = [
    OpenAIChatModel('gpt-4o'),           # Default balanced
    OpenAIChatModel('gpt-5-mini'),       # Fast, simple tasks
    OpenAIChatModel('gpt-5', model_settings=ModelSettings(reasoning_level='medium')),  # Complex tasks
]

class NextModelHint(BaseModel):
    """Tool for agent to request specific model characteristics for next cycle."""
    reasoning_level: Literal['minimal', 'medium', 'high']
    complexity: Literal['simple', 'complex']

def self_directed_selector(ctx: AdaptiveContext[Any]) -> Model:
    """Select model based on agent's own hint from previous cycle."""
    # No state needed - decision based on message history only

    # Look for hint in last response
    if ctx.messages and isinstance(last_msg := ctx.messages[-1], ModelResponse):
        for part in last_msg.parts:
            if isinstance(part, ToolCallPart) and part.tool_name == 'next_model_hint':
                args = part.args_as_dict()

                # Agent requested high reasoning for complex task
                if args.get('reasoning_level') == 'high' and args.get('complexity') == 'complex':
                    return next((m for m in models if 'gpt-5' in m.model_name), models[0])

                # Agent requested minimal reasoning for simple task
                if args.get('reasoning_level') == 'minimal':
                    return next((m for m in models if 'gpt-5-mini' in m.model_name), models[1])

    # Default to balanced model
    return models[0]

adaptive = AdaptiveModel(
    selector=self_directed_selector
    # No state parameter - decision based on message history only
)

agent = Agent(adaptive)

@agent.tool
def next_model_hint(ctx: RunContext, reasoning_level: str, complexity: str) -> str:
    """Request specific model for next cycle. Always call with other tools."""
    return f"Will use {reasoning_level} reasoning, {complexity} model next cycle"

# Agent decides its own model per cycle:
# Cycle 1: "I need to generate complex SQL" → calls next_model_hint(reasoning_level='high', complexity='complex')
# Cycle 2: Uses gpt-5 with medium reasoning for SQL generation
# Cycle 3: "Just checking if embeddings exist" → calls next_model_hint(reasoning_level='minimal', complexity='simple')
# Cycle 4: Uses gpt-5-mini for fast simple check
```

**Key Benefits:**
- Agent self-optimizes based on task complexity it discovers
- No external orchestration needed
- Works within standard ReAct flow
- Hint tool runs in parallel with other tools (no extra cycle)

### 8. Account Quota Management (Global State)

Rotate across accounts based on remaining quota.

⚠️ **Concurrency:** This example uses shared state across concurrent requests with proper locking.

```python
from datetime import datetime

# Models managed by selector (via closure)
models = [
    OpenAIChatModel('gpt-4o', api_key=key1),
    OpenAIChatModel('gpt-4o', api_key=key2),
    OpenAIChatModel('gpt-4o', api_key=key3),
]

@dataclass
class AccountPool:
    accounts: dict[str, dict]  # account_id -> {api_key, quota_remaining, quota_limit, reset_time}
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def get_available_account(self) -> str | None:
        async with self.lock:
            now = datetime.now()
            for account_id, info in self.accounts.items():
                if now >= info['reset_time']:
                    info['quota_remaining'] = info['quota_limit']
                if info['quota_remaining'] > 0:
                    return account_id
            return None

async def quota_rotation_selector(ctx: AdaptiveContext[AccountPool]) -> Model:
    tried = {id(a.model) for a in ctx.attempts}

    account_id = await ctx.state.get_available_account()
    if not account_id:
        raise RuntimeError("No accounts with available quota")

    # Find model for this account that hasn't been tried
    async with ctx.state.lock:
        model = next(
            (m for m in models
             if hasattr(m, 'api_key') and m.api_key == ctx.state.accounts[account_id]['api_key']
             and id(m) not in tried),
            None
        )
        if model:
            return model
        raise RuntimeError("All accounts exhausted")

async def on_success_deduct_quota(
    state: AccountPool,
    model: Model,
    response: ModelResponse,
    timestamp: datetime,
    duration: timedelta
):
    """Hook: Deduct tokens from quota."""
    tokens_used = response.usage.total_tokens if response.usage else 0
    async with state.lock:
        for account_id, info in state.accounts.items():
            if hasattr(model, 'api_key') and model.api_key == info['api_key']:
                info['quota_remaining'] -= tokens_used
                break

adaptive = AdaptiveModel(
    selector=quota_rotation_selector,
    state=AccountPool(accounts={
        'account1': {'api_key': key1, 'quota_remaining': 100000, 'quota_limit': 100000, 'reset_time': datetime(2025, 12, 1)},
        'account2': {'api_key': key2, 'quota_remaining': 50000, 'quota_limit': 100000, 'reset_time': datetime(2025, 12, 1)},
    }),
    on_attempt_succeeded=on_success_deduct_quota  # Automatic quota deduction
)

# Reuse same model instance - quotas tracked across all requests
agent = Agent(adaptive)

result = await agent.run('query')
# State persists: quotas are tracked and deducted automatically
```

## Key Benefits

1. **Full Control**: Custom logic for model selection based on any criteria
2. **Stateful Logic**: Maintain state across calls (throttle timers, usage counts, quality metrics)
3. **Smart Waiting**: Wait for throttled models instead of giving up
4. **Load Distribution**: Balance across identical models on different accounts
5. **User-Aware**: Access agent dependencies for user-specific routing
6. **Cost Optimization**: Dynamic model selection based on cost, quality, and usage
7. **Complex Retry**: Implement exponential backoff, circuit breakers, etc.

## State Pattern Summary

State is managed at the `AdaptiveModel` level, not passed through the Agent. This provides three distinct usage patterns based on model instance lifetime:

### When to Use Each Pattern

**Per-Run State (Isolated):**
- ✅ Retry logic within a single request
- ✅ Context-based decisions (message count, complexity)
- ✅ No cross-request coordination needed
- Example: Exponential backoff, context length upgrades

**Global State (Shared):**
- ✅ System-wide throttling and rate limiting
- ✅ Load balancing across accounts
- ✅ Cross-request metrics and quality tracking
- Example: Throttle timers, call counts, quotas

**Per-User State (Session):**
- ✅ User-specific quotas and limits
- ✅ User tier-based routing
- ✅ Per-user preferences and history
- Example: API endpoints with user sessions

### Implementation Pattern

```python
# 1. Define state type
@dataclass
class MyState:
    field1: str
    field2: int

# 2. Create adaptive model with state
adaptive = AdaptiveModel(
    models=[...],
    selector=my_selector,
    state=MyState()  # State lives in model
)

# 3. Use the model - no changes to Agent API
agent = Agent(adaptive)

# Per-run: create new model instance each time
def run_isolated(query: str):
    adaptive = AdaptiveModel(models=[...], selector=my_selector, state=MyState())
    agent = Agent(adaptive)
    return agent.run_sync(query)

# Global: reuse same model instance
global_adaptive = AdaptiveModel(models=[...], selector=my_selector, state=MyState())
global_agent = Agent(global_adaptive)
result1 = await global_agent.run('query1')  # Shared state
result2 = await global_agent.run('query2')  # Same state persists

# Per-user: model instance per user
user_models: dict[str, AdaptiveModel] = {}
if user_id not in user_models:
    user_models[user_id] = AdaptiveModel(models=[...], selector=my_selector, state=MyState())
agent = Agent(user_models[user_id])
result = await agent.run('query')
```

## Comparison with FallbackModel

| Feature | FallbackModel | AdaptiveModel |
|---------|--------------|---------------|
| Model Selection | Sequential | Custom logic |
| State Management | None | Per-run, global, or per-user |
| Wait/Retry Logic | Not supported | Full control (async sleep) |
| Load Balancing | Not supported | Supported |
| Context Access | None | Messages, attempts, state |
| Complexity | Simple | Flexible |

`AdaptiveModel` complements `FallbackModel` by supporting sophisticated routing scenarios while `FallbackModel` remains the simpler choice for basic sequential fallback.
