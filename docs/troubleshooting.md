# Troubleshooting

Common/known issues are shown below with associated solutions.

## Jupyter Notebooks

If you're running `pydantic-ai` in a jupyter notebook, you might consider using [`nest-asyncio`](https://pypi.org/project/nest-asyncio/)
to manage conflicts between event loops that occur between Jupyter's event loops and `pydantic-ai`'s.

Before you execute any agent runs, do the following:

```python {test="skip" lint="skip"}
import nest_asyncio

nest_asyncio.apply()
```
