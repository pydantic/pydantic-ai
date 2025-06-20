# PydanticAI

Implementation of the AG-UI protocol for PydanticAI.

## Prerequisites

This example uses a PydanticAI agent using an OpenAI model and the AG-UI dojo.

1. An [OpenAI API key](https://help.openai.com/en/articles/4936850-where-do-i-find-my-openai-api-key)
2. A clone of this repository
3. A clone of the [AG-UI protocol repository](https://github.com/ag-ui-protocol/ag-ui)

## Running

To run this integration you need to:

1. Make a copy of `jobs-agent/.env.local-example` as `.env`
2. Open it in your editor and set `OPENAI_API_KEY` to a valid OpenAI key
3. Open terminal in the root directory of this repository clone
4. Install the required modules and run the server

    ```shell
    cd jobs-agent
    just install-deps
    source .venv/bin/activate
    python -m examples.pydantic_ai_ag_ui_examples.dojo_server
    ```

5. Open another terminal in root directory of the `ag-ui` repository clone
6. Start the integration ag-ui dojo:

    ```shell
    cd typescript-sdk
    pnpm install && pnpm run dev
    ```

7. Finally visit [http://localhost:3000/pydantic-ai](http://localhost:3000/pydantic-ai)

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
of the UI based on agent responses, including user interaction via git aconfirmation.

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
