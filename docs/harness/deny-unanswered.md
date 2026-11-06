---
title: Deny Unanswered
description: "Deny approval-required calls and unanswered AskUser questions with readable results, so an unattended run finishes instead of pausing on deferred calls."
---

# Deny Unanswered

`DenyUnanswered` denies approval-required calls and unanswered `AskUser` questions with readable results, so a run nobody attends finishes instead of pausing on `DeferredToolRequests`.

Deferred calls exist for a caller who answers them in a second run. In an unattended run nobody ever will: approvals wait on a person who is not there, questions wait on a user who left. `DenyUnanswered` ends the wait inside the run:

- A held approval is denied with `message` (`ToolDenied`).
- A deferred `ask_user_question` call is answered as cancelled through `ask_user_result`, so the model sees `AskUser`'s standard decline text.

Any other deferred call stays deferred for its own handler. Without `DenyUnanswered`, nothing changes.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/deny_unanswered/)

## Usage

```python
from pydantic_ai_harness import AskUser, DenyUnanswered

agent = Agent(
    model,
    capabilities=[
        AskUser(),  # defer every question; nobody is listening
        DenyUnanswered(),
    ],
)
```

Use `message` to change the denial text for approvals:

```python
DenyUnanswered(message='No one is on duty, so the request was not approved.')
```

## Forwarding to children

`SubAgents` delegations and `DynamicWorkflow` child runs carry the parent run's `DenyUnanswered` into the child's capabilities when it is present, so a delegate that asks a question or requests approval is denied inside the child run too, and the delegation finishes instead of returning a pending child. The handler runs innermost, after every handler that may ask the user, so a present human still wins.

## Telemetry

Emits nothing into telemetry.
