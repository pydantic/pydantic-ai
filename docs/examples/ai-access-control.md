# AI Access Control

Example of implementing access control security for AI agents using PydanticAI and Permit.io

Demonstrates:

- [tools](../tools.md)
- [agent dependencies](../dependencies.md)
- [structured `result_type`](../results.md#structured-result-validation)
- [result validation](../results.md#result-validators-functions)
- Permit.io [AI Access Control](https://www.permit.io/ai-access-control)

In this `Financial Advisor` agent example, the user can ask for financial advice, and the agent will use the `validate_financial_query` and `validate_financial_response` tools to check permissions and validate responses.
The raw user query is transformed into a `FinancialQuery` object that includes user context (such as their permission tier).
This structured data allows us to carry out permission checks before the AI generates a response, and the output is strictly validated with `FinancialResponse`.

```mermaid
graph TD
    USER[User Query] -->|Request| VALIDATE[Validate Financial Query]
    VALIDATE -->|Permission Check| PERMIT[Permit.io]
    PERMIT -->|Decision| VALIDATE
    VALIDATE -->|If Approved| AI[AI Processing]
    AI -->|Generate Response| VALIDATE_RESP[Validate Financial Response]
    VALIDATE_RESP -->|Permission Check| PERMIT
    PERMIT -->|Decision| VALIDATE_RESP
    VALIDATE_RESP -->|Add Disclaimer if Needed| RESPONSE[Return Structured Response]
    RESPONSE -->|Final Response| USER
```

## Running the Example

### Setup Access Control Configuration
To run this example, you need a Permit.io API key and a running Permit authorization microservice

- Get a free API key [here](https://app.permit.io) set it as an environment variable with the key `PERMIT_KEY`
- Run a local Policy Decision Point (PDP):

```bash
docker pull permitio/pdp-v2:latest
docker run -it \\
    -p 7766:7000 \\
    --env PDP_API_KEY=<YOUR_API_KEY> \\
    --env PDP_DEBUG=True \\
    permitio/pdp-v2:latest
```

Run the following command to set up our Permit.io financial advisor policy configuration:

```bash
python/uv-run -m pydantic_ai_examples.ai-access-control-config
```

### Run the Example

With [dependencies installed and environment variables set](./index.md#usage), run the following command to run the example:

```bash
python/uv-run -m pydantic_ai_examples.ai-access-control
```

## Example code

```python {title="pydantic_ai_examples/ai-access-control.py"}
#! examples/pydantic_ai_examples/ai-access-control.py
```

## Configuration code

The following script will set the following fine-grained Attribute-Based Access Control (ABAC) policy in Permit.io

1. **User Tiers**:
   - `opted_in_user`: Users who have explicitly consented to AI advice
   - `premium_user`: Users with access to all features and advanced financial advice
2. **Resources and Actions**:
   - `financial_advice`: Controls who can receive investment recommendations
     - Attributes: `is_ai_generated` (boolean)
     - Actions: `receive`
   - `financial_response`: Controls content and disclaimers in responses
     - Attributes: `contains_advice` (boolean)
     - Actions: `requires_disclaimer`
3. **Decision Logic**:
   - Only `opted_in_user` can receive AI-generated advice
   - AI generated responses require disclaimers for regulatory compliance

```python {title="pydantic_ai_examples/ai-access-control-config.py"}
#! examples/pydantic_ai_examples/ai-access-control-config.py
```

## Demo

Watch how PydanticAI ensures secure financial advice:

![AI Access Control Live Demo](https://github.com/user-attachments/assets/e97c0a9e-d0a0-45f8-8da9-710e539d42f3)


## Further Reading

With PydanticAI's tools and structured response system with Permit.io's permission checks, AI developers can implement fine-grained AI Access Control with minimal effort.

In the following blog post, we cover more in-depth topics in AI Access Control using PydanticAI and Permit.io: https://www.permit.io/blog/ai-agents-access-control-with-pydantic-ai
