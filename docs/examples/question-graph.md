# Question Graph

Example of a graph for asking and evaluating questions.

Demonstrates:

* [`pydantic_graph`](../graph.md)

## Running the Example

With [dependencies installed and environment variables set](./setup.md#usage), run:

```bash
python/uv-run -m pydantic_ai_examples.question_graph
```

## Example Code

```snippet {path="/examples/pydantic_ai_examples/question_graph.py"}```

The mermaid diagram generated in this example looks like this:

```mermaid
---
title: question_graph
---
stateDiagram-v2
  [*] --> Ask
  Ask --> Answer: ask the question
  Answer --> Evaluate: answer the question
  Evaluate --> Congratulate
  Evaluate --> Castigate
  Congratulate --> [*]: success
  Castigate --> Ask: try again
```
