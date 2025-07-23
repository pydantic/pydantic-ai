# Agent User Interaction (AG-UI)

Example of using Pydantic AI agents with the [AG-UI Dojo](https://github.com/ag-ui-protocol/ag-ui/tree/main/typescript-sdk/apps/dojo) example app.

See the [AG-UI docs](../ag-ui.md) for more information about the AG-UI integration.

Demonstrates:

- [AG-UI](../ag-ui.md)
- [Tools](../tools.md)

## Prerequisites

- An [OpenAI API key](https://help.openai.com/en/articles/4936850-where-do-i-find-my-openai-api-key)

## Running the Example

With [dependencies installed and environment variables set](./index.md#usage)
you will need two command line windows.

### Pydantic AI AG-UI backend

Setup your OpenAI API Key

```bash
export OPENAI_API_KEY=<your api key>
```

Start the Pydantic AI AG-UI example backend.

```bash
python/uv-run -m pydantic_ai_examples.ag_ui
```

### AG-UI Dojo example frontend

Next run the AG-UI Dojo example frontend.

1. Clone the [AG-UI repository](https://github.com/ag-ui-protocol/ag-ui)

    ```shell
    git clone https://github.com/ag-ui-protocol/ag-ui.git
    ```

2. Change into to the `ag-ui/typescript-sdk` directory

    ```shell
    cd ag-ui/typescript-sdk
    ```

3. Run the Dojo app following the [official instructions](https://github.com/ag-ui-protocol/ag-ui/tree/main/typescript-sdk/apps/dojo#development-setup)
4. Visit <http://localhost:3000/pydantic-ai>
5. Select View `Pydantic AI` from the sidebar

## Feature Examples

### Agentic Chat

This demonstrates a basic agent interaction including Pydantic AI server side
tools and AG-UI client side tools.

View the [Agentic Chat example](http://localhost:3000/pydantic-ai/feature/agentic_chat).

#### Agent Tools

- `time` - Pydantic AI tool to check the current time for a time zone
- `background` - AG-UI tool to set the background color of the client window

#### Agent Prompts

```text
What is the time in New York?
```

```text
Change the background to blue
```

A complex example which mixes both AG-UI and Pydantic AI tools:

```text
Perform the following steps, waiting for the response of each step before continuing:
1. Get the time
2. Set the background to red
3. Get the time
4. Report how long the background set took by diffing the two times
```

#### Agentic Chat - Code

```snippet {path="/examples/pydantic_ai_examples/ag_ui/api/agentic_chat.py"}```

### Agentic Generative UI

Demonstrates a long running task where the agent sends updates to the frontend
to let the user know what's happening.

View the [Agentic Generative UI example](http://localhost:3000/pydantic-ai/feature/agentic_generative_ui).

#### Plan Prompts

```text
Create a plan for breakfast and execute it
```

#### Agentic Generative UI - Code

```snippet {path="/examples/pydantic_ai_examples/ag_ui/api/agentic_generative_ui.py"}```

### Human in the Loop

Demonstrates simple human in the loop workflow where the agent comes up with a
plan and the user can approve it using checkboxes.

#### Task Planning Tools

- `generate_task_steps` - AG-UI tool to generate and confirm steps

#### Task Planning Prompt

```text
Generate a list of steps for cleaning a car for me to review
```

#### Human in the Loop - Code

```snippet {path="/examples/pydantic_ai_examples/ag_ui/api/human_in_the_loop.py"}```

### Predictive State Updates

Demonstrates how to use the predictive state updates feature to update the state
of the UI based on agent responses, including user interaction via user
confirmation.

View the [Predictive State Updates example](http://localhost:3000/pydantic-ai/feature/predictive_state_updates).

#### Story Tools

- `write_document` - AG-UI tool to write the document to a window
- `document_predict_state` - Pydantic AI tool that enables document state
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

#### Predictive State Updates - Code

```snippet {path="/examples/pydantic_ai_examples/ag_ui/api/predictive_state_updates.py"}```

### Shared State

Demonstrates how to use the shared state between the UI and the agent.

State sent to the agent is detected by a function based instruction. This then
validates the data using a custom pydantic model before using to create the
instructions for the agent to follow and send to the client using a AG-UI tool.

View the [Shared State example](http://localhost:3000/pydantic-ai/feature/shared_state).

#### Recipe Tools

- `display_recipe` - AG-UI tool to display the recipe in a graphical format

#### Recipe Example

1. Customise the basic settings of your recipe
2. Click `Improve with AI`

#### Shared State - Code

```snippet {path="/examples/pydantic_ai_examples/ag_ui/api/shared_state.py"}```

### Tool Based Generative UI

Demonstrates customised rendering for tool output with used confirmation.

View the [Tool Based Generative UI example](http://localhost:3000/pydantic-ai/feature/tool_based_generative_ui).

#### Haiku Tools

- `generate_haiku` - AG-UI tool to display a haiku in English and Japanese

#### Haiku Prompt

```text
Generate a haiku about formula 1
```

#### Tool Based Generative UI - Code

```snippet {path="/examples/pydantic_ai_examples/ag_ui/api/tool_based_generative_ui.py"}```
