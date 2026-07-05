# DashScope Vision

Image understanding with Alibaba Cloud Model Studio (DashScope) using a Qwen VL model, then structured output with a text model.

Demonstrates:

- [`alibaba:`](../models/openai.md#alibaba-cloud-model-studio-dashscope) provider prefix (recommended over generic `openai:` + `OPENAI_BASE_URL`)
- [Image input](../input.md#image-input) with [`BinaryContent`][pydantic_ai.BinaryContent]
- A **two-step** pattern: vision model for pixels, text model for [`output_type`](../output.md#structured-output)
- Optional split between a **vision** model (`qwen-vl-plus`) and a **text** model (`qwen-plus`)

## Running the Example

With [dependencies installed](./setup.md#usage), set an API key and run:

```bash
export DASHSCOPE_API_KEY='your-key'   # or ALIBABA_API_KEY
uv run -m pydantic_ai_examples.dashscope_vision
```

China region endpoint (optional — this example reads `ALIBABA_BASE_URL`; in your own code pass `base_url` to [`AlibabaProvider`][pydantic_ai.providers.alibaba.AlibabaProvider] as shown in [DashScope docs](../models/openai.md#alibaba-cloud-model-studio-dashscope)):

```bash
export ALIBABA_BASE_URL='https://dashscope.aliyuncs.com/compatible-mode/v1'
uv run -m pydantic_ai_examples.dashscope_vision
```

Override models:

```bash
PYDANTIC_AI_VISION_MODEL=alibaba:qwen-vl-max \
PYDANTIC_AI_TEXT_MODEL=alibaba:qwen-max \
uv run -m pydantic_ai_examples.dashscope_vision
```

## Example Code

```snippet {path="/examples/pydantic_ai_examples/dashscope_vision.py"}```
