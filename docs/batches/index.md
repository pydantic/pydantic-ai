# Batch Processing

Pydantic AI supports batch processing through provider-specific implementations that allow you to process multiple requests cost-effectively with only one request.

!!! info "About batch processing"
    *Batch processing is very useful when you want to process large datasets and make multiple calls to an LLM provider. You don't need to process them with loops - just create a batch, save `batch_id` somewhere and forget about it until it's ready. Alternatively, it's also useful if you're willing to wait up to 24 hours for a response.*

## OpenAI Batch API

The OpenAI Batch API provides 50% cost savings compared to default API calls with a 24-hour processing window. Batch jobs require at least 2 requests.

[Learn more about OpenAI Batch API →](openai.md)

## Key Benefits

- **Cost Savings**: Up to 50% reduction in API costs
- **Bulk Processing**: Handle hundreds or thousands of requests efficiently
- **Async Processing**: Submit jobs and retrieve results when ready
- **Tool Support**: Full support for tools
- **Structured Output**: All output modes (native, tool, prompted) supported

!!! warning "Important: Tool Usage with Batch API"
    When using tools with batch processing, the AI model returns tool call **requests** rather than executing tools automatically. You need to:

    1. **Check batch results** for `tool_calls` instead of direct responses
    2. **Execute the tools manually** in your application code
    3. **Submit follow-up requests** with tool responses to get final AI answers

    This differs from the regular Agent API which handles tool execution automatically.

## Use Cases

Batch processing is ideal for:

- **Data Analysis**: Processing large datasets with LLM analysis
- **Content Generation**: Bulk content creation and transformation
- **Evaluation**: Running evaluations across multiple test cases
- **A/B Testing**: Comparing different prompts or models
- **Bulk Translation**: Translating multiple documents
- **Report Generation**: Creating reports from structured data

## Available Providers

Currently supported batch processing providers:

- **OpenAI**: batch API support with tools and structured outputs

Additional providers may be added in future releases.
