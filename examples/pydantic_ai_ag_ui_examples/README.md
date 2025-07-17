# PydanticAI AG-UI Examples

This example uses PydanticAI agents with the [AG-UI Dojo](https://github.com/ag-ui-protocol/ag-ui/tree/main/typescript-sdk/apps/dojo) example app.

## Prerequisites

1. An [OpenAI API key](https://help.openai.com/en/articles/4936850-where-do-i-find-my-openai-api-key)
2. A clone of the [AG-UI repository](https://github.com/ag-ui-protocol/ag-ui)

## Running

1. Run the Pydantic AI AG-UI example backend

    1. Install the `pydantic-ai-examples` package

        ```shell
        pip/uv-add pydantic-ai-examples
        ```

    2. Run the example AG-UI app

        ```shell
        python/uv-run -m pydantic_ai_ag_ui_examples.dojo_server
        ```

2. Run the AG-UI Dojo example frontend
    1. Move to the cloned AG-UI repository directory
    2. In the `typescript-sdk/integrations/pydantic-ai` directory, copy `.env-sample` to `.env`
    3. Open it in your editor and set `OPENAI_API_KEY` to a valid OpenAI key
    4. Open a terminal in the `typescript-sdk` directory
    5. Run the Dojo app following the [official instructions](https://github.com/ag-ui-protocol/ag-ui/tree/main/typescript-sdk/apps/dojo#development-setup)

3. Finally visit <http://localhost:3000/pydantic-ai>

## Feature Demos

### [Agentic Chat](http://localhost:3000/pydantic-ai/feature/agentic_chat)

This demonstrates a basic agent interaction including PydanticAI server side
tools and AG-UI client side tools.

#### Agent Tools

- `time` - PydanticAI tool to check the current time for a time zone
- `background` - AG-UI tool to set the background color of the client window

#### Agent Prompts

```text
What is the time in New York?
```

```text
Change the background to blue
```

A complex example which mixes both AG-UI and PydanticAI tools:

```text
Perform the following steps, waiting for the response of each step before continuing:
1. Get the time
2. Set the background to red
3. Get the time
4. Report how long the background set took by diffing the two times
```

### [Agentic Generative UI](http://localhost:3000/pydantic-ai/feature/agentic_generative_ui)

Demonstrates a long running task where the agent sends updates to the frontend
to let the user know what's happening.

#### Plan Prompts

```text
Create a plan for breakfast and execute it
```

### [Human in the Loop](http://localhost:3000/pydantic-ai/feature/human_in_the_loop)

Demonstrates simple human in the loop workflow where the agent comes up with a
plan and the user can approve it using checkboxes.

#### Task Planning Tools

- `generate_task_steps` - AG-UI tool to generate and confirm steps

#### Task Planning Prompt

```text
Generate a list of steps for cleaning a car for me to review
```

### [Predictive State Updates](http://localhost:3000/pydantic-ai/feature/predictive_state_updates)

Demonstrates how to use the predictive state updates feature to update the state
of the UI based on agent responses, including user interaction via user
confirmation.

#### Story Tools

- `write_document` - AG-UI tool to write the document to a window
- `document_predict_state` - PydanticAI tool that enables document state
  prediction for the `write_document` tool

This also shows how to use custom instructions based on shared state information.

#### Story Example

Starting document text

```markdown
Bruce was a good dog,
```

Agent prompt

```text
Help me complete my story about bruce the dog, is should be no longer than a sentence.
```

### [Shared State](http://localhost:3000/pydantic-ai/feature/shared_state)

Demonstrates how to use the shared state between the UI and the agent.

State sent to the agent is detected by a function based instruction. This then
validates the data using a custom pydantic model before using to create the
instructions for the agent to follow and send to the client using a AG-UI tool.

#### Recipe Tools

- `display_recipe` - AG-UI tool to display the recipe in a graphical format

#### Recipe Example

1. Customise the basic settings of your recipe
2. Click `Improve with AI`

### [Tool Based Generative UI](http://localhost:3000/pydantic-ai/feature/tool_based_generative_ui)

Demonstrates customised rendering for tool output with used confirmation.

#### Haiku Tools

- `generate_haiku` - AG-UI tool to display a haiku in English and Japanese

#### Haiku Prompt

```text
Generate a haiku about formula 1
```
