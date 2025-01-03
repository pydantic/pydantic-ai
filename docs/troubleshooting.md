# Troubleshooting

Below are suggestions on how to fix some common errors you might encounter while using PydanticAI. If the issue you're experiencing is not listed below or addressed in the documentation, please feel free to ask in the [Pydantic Slack](help.md) or create an issue on [GitHub](https://github.com/pydantic/pydantic-ai/issues).

## Jupyter Notebook Errors

### `RuntimeError: This event loop is already running`

This error is caused by conflicts between the event loops in Jupyter notebook and PydanticAI's. One way to manage these conflicts is by using [`nest-asyncio`](https://pypi.org/project/nest-asyncio/). Namely, before you execute any agent runs, do the following:
```python {test="skip"}
import nest_asyncio

nest_asyncio.apply()
```
Note: This fix also applies to Google Colab.

## API Key Configuration

### `UserError: API key must be provided or set in the [MODEL]_API_KEY environment variable`

If you're running into issues with setting the API key for your model, visit the [Models](models.md) page to learn more about how to set an environment variable and/or pass in an `api_key` argument.
