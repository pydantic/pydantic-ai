# Fine-Grained Permissions for AI Agents

This example demonstrates implementing a secure financial advisor agent using PydanticAI with four distinct access control perimeters. It shows how to build a production-ready AI system that enforces certification requirements, regulatory compliance, and proper risk management.

Demonstrates:

- [dynamic system prompt](../agents.md#system-prompts)
- [structured `result_type`](../results.md#structured-result-validation)
- [result validation](../results.md#result-validators-functions)
- [tools](../tools.md)

## Access Control Perimeters Overview

### Prompt Filtering

The first perimeter to secure our financial advisor agent focuses on checking user permissions for receiving AI advice before they reach the AI model. This ensures that users can only make requests within their permission scope. By implementing this perimeter, we can check if users opt-in for AI finance advice.

In our financial advisor example, this perimeter:

- Checks user opt-in status for AI-generated advice
- Classifies query intent to apply appropriate permissions

```bash
@financial_agent.tool
async def validate_financial_query(
ctx: RunContext[PermitDeps],
query: FinancialQuery,
) -> Dict[str, bool]:
```

### Data Protection

The second perimeter manages access to the knowledge and data sources that the AI agent can reference. This layer ensures that sensitive information is only accessible to authorized users and that the AI model respects data classification levels.

In our financial advisor example, this perimeter:

- Controls access to sensitive financial knowledge base and documentation
- User clearance level verification

```bash
@financial_agent.tool
async def access_financial_knowledge(
ctx: RunContext[PermitDeps],
query: FinancialQuery
) -> List[str]:
```

### Secure External Access

The third perimeter protects interactions with external systems and APIs. This layer ensures that the AI agent can only perform authorized operations on external resources and maintains proper security boundaries. It prevents unauthorized system access and ensures proper authentication and authorization for external interactions.

In our financial advisor example, this perimeter:

- Controls Portfolio operation authorization
- Manages External API access control

```bash
@financial_agent.tool
async def check_action_permissions(
ctx: RunContext[PermitDeps],
action: str,
context: UserContext
) -> bool:
```

### Response Enforcement

The final perimeter validates and enforces security policies on AI-generated responses before they reach users. This important layer makes sure that responses meet compliance requirements, contain necessary disclaimers, and don't leak sensitive information. It acts as the last line of defense in maintaining security and compliance.

In our financial advisor example, this perimeter:

- Ensures compliance with financial regulations
- Manages risk disclosures and disclaimers
- Prevents unauthorized advice distribution

```bash
@financial_agent.tool
async def validate_financial_response(
ctx: RunContext[PermitDeps],
response: FinancialResponse,
context: UserContext
) -> Dict[str, bool]:
```

## Running the Example

First, configure your Permit.io environment (you can get a [free API key here](https://app.permit.io)): by setting the required environment variables:

```bash
# Required environment variables
export PERMIT_KEY='your-api-key'  # Your Permit.io API key
export PDP_URL='http://localhost:7766'  # Your PDP URL (default: http://localhost:7766)
```

The code will automatically load these environment variables:

```bash
import os
from dotenv import load_dotenv


# Load environment variables from .env file if present
load_dotenv()

# Get Permit.io configuration from environment
PERMIT_KEY = os.environ['PERMIT_KEY']
PDP_URL = os.environ.get('PDP_URL', 'http://localhost:7766')

# Initialize Permit client with environment configuration
permit = Permit(
    token=PERMIT_KEY,
    pdp=PDP_URL,
)
```

### Setup Steps

1. First, run the configuration script to set up required resources and roles in Permit.io:

```bash
python/uv-run -m pydantic_ai_examples.secure-ai-config
```

The configuration script sets up a complete ABAC (Attribute-Based Access Control) model including:

#### Resources and Attributes

- `financial_advice`: AI-generated advice with risk levels
- `financial_document`: Documents with classification levels and clearance requirements
- `financial_response`: Response content with advice detection
- `portfolio`: Investment portfolios with value tiers

#### Condition Sets

- Document Clearance Check: Validates user's clearance level
- AI Advice Opt-in Check: Verifies user consent

#### User Sets

- Opted-in Users: Users who consented to AI advice
- High Clearance Users: Users with elevated access

#### Resource Sets

- Confidential Documents: High-security financial documents
- AI Finance Advice: Financial advice with AI

#### Roles and Permissions

- Resources: financial_advice, financial_document, financial_response, portfolio
- Roles: restricted_user, premium_user
- Permissions: Various access levels for each role

## Policy Table

Here is how our permission table looks like. With our ABAC policy implementation in Permit.io we defined granular permissions for different user roles:

![permit policy](https://hackmd.io/_uploads/ryOKcyxqJx.png)

## Demo

A video demonstration of the Financial Advisor AI Agent showcasing:

<a href="https://imgur.com/MFNGtDV"><img src="https://i.imgur.com/MFNGtDV.gif" title="source: imgur.com" /></a>

You can try the demo yourself by following the setup instructions below.

## Running the Example

With [dependencies installed and environment variables set](./index.md#usage), run:

```bash
python/uv-run -m pydantic_ai_examples.secure-ai-config
```

To set up our Permit.io configuration.

```bash
python/uv-run -m pydantic_ai_examples.secure-ai-agent
```

To run our app.

## Example Code

python code that runs the app

```python {title="secure_ai_agent.py"}
#! examples/pydantic_ai_examples/secure-ai-agent.py
```

python config file for permission configuration.

```python {title="secure-ai-config.py"}
#! examples/pydantic_ai_examples/secure-ai-config.py
```

## Further Reading

To learn more about the concepts used in this implementation, check out these helpful resources:

1. [What is Fine-Grained Authorization (FGA)?](https://www.permit.io/blog/what-is-fine-grained-authorization-fga) - Learn about the fundamentals of FGA and how it differs from traditional access control methods.

2. [How to Implement Attribute-Based Access Control (ABAC) Authorization?](https://www.permit.io/blog/how-to-implement-abac) - Understand how to design and implement ABAC systems effectively.

3. [RBAC VS ABAC: Choosing the Right Authorization Policy Model](https://www.permit.io/blog/rbac-vs-abac) - Explore how RBAC complements ABAC in modern authorization systems.
