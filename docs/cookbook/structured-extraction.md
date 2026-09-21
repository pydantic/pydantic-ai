---
title: Extract structured data from text
description: Turn an unstructured business document into validated application data.
---

# Extract structured data from text

Pass a Pydantic model as `output_type` when the application needs validated data rather than a prose response.

```bash
pip/uv-add "pydantic-ai-slim[openai]"
export OPENAI_API_KEY=your-api-key
```

```python {dunder_name="not_main"}
import asyncio
from datetime import date
from decimal import Decimal

from pydantic import BaseModel

from pydantic_ai import Agent


class LineItem(BaseModel):
    description: str
    quantity: int
    unit_price: Decimal


class Invoice(BaseModel):
    invoice_number: str
    invoice_date: date
    customer: str
    items: list[LineItem]
    total: Decimal


agent = Agent('openai:gpt-5.6-sol', output_type=Invoice)


async def main() -> None:
    result = await agent.run(
        """
        ACME Hosting — Invoice INV-2048
        Issued: 2026-09-18
        Bill to: Northstar Labs

        Managed database, 2 × $120.00
        Object storage, 1 × $35.50

        Amount due: $275.50
        """
    )
    print(result.output.invoice_number, result.output.total)
    #> INV-2048 275.50


if __name__ == '__main__':
    asyncio.run(main())
```

Pydantic AI supplies the model with the output schema and validates the response. The rest of the application receives an `Invoice`, including parsed `date` and `Decimal` values, without maintaining a separate extraction prompt or JSON parser.

## Related

See [Structured output](../output.md#structured-output) for output modes, unions, and schema customization.
