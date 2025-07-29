from dataclasses import dataclass

from devtools import debug

from pydantic_ai import Agent, ModelRetry, RunContext

try:
    import datasets
    import duckdb
    import pandas as pd
except ImportError as e:
    raise ImportError(
        'Please install both duckdb and pandas.\n'
        '- pip: `pip install duckdb pandas\n'
        '- uv: `uv pip install duckdb pandas'
    ) from e


@dataclass
class AnalystAgentDeps:
    output: dict[str, pd.DataFrame]


analyst_agent = Agent(
    'openai:gpt-4o',
    deps_type=AnalystAgentDeps,
    instructions='You are a data analyst and your job is to analyze the data according to the user request.',
)


@analyst_agent.tool
def load_dataset(
    ctx: RunContext[AnalystAgentDeps],
    path: str,
    split: str = 'train',
) -> str:
    """Load the `split` of dataset `dataset_name` from huggingface.

    Args:
        ctx: Pydantic AI agent RunContext
        path: name of the dataset in the form of `<user_name>/<dataset_name>`
        split: load the split of the dataset (default: "train")
    """
    builder = datasets.load_dataset_builder(path)  # pyright: ignore[reportUnknownMemberType]
    splits: dict[str, datasets.SplitInfo] = builder.info.splits or {}  # pyright: ignore[reportUnknownMemberType]
    if split not in splits:
        raise ModelRetry(
            f'{split} is not valid for dataset {path}. Valid splits are {",".join(splits.keys())}'
        )

    builder.download_and_prepare()  # pyright: ignore[reportUnknownMemberType]
    dataset = builder.as_dataset(split=split)
    assert isinstance(dataset, datasets.Dataset)
    dataframe = dataset.to_pandas()
    assert isinstance(dataframe, pd.DataFrame)
    ref = f'Out[{len(ctx.deps.output) + 1}]'
    ctx.deps.output[ref] = dataframe
    output = [f'Loaded the dataset as `{ref}`.']
    if dataset.info.description:
        output.append(f'Description: {dataset.info.description}')
    if dataset.info.features:
        output.append(f'Features: {dataset.info.features!r}')
    return '\n'.join(output)


@analyst_agent.tool
def run_duckdb(ctx: RunContext[AnalystAgentDeps], dataset: str, sql: str) -> str:
    """Run DuckDB SQL query on the DataFrame.

    Note that the virtual table name used in DuckDB SQL must be `dataset`.

    Args:
        ctx: Pydantic AI agent RunContext
        dataset: reference string to the DataFrame
        sql: the query to be executed using DuckDB
    """
    data = ctx.deps.output[dataset]
    result = duckdb.query_df(df=data, virtual_table_name='dataset', sql_query=sql)
    ref = f'Out[{len(ctx.deps.output) + 1}]'
    ctx.deps.output[ref] = result.df()  # pyright: ignore[reportUnknownMemberType]
    return f'Executed SQL, result is `{ref}`'


@analyst_agent.tool
def display(ctx: RunContext[AnalystAgentDeps], name: str) -> str:
    """Display the dataframe at most 5 rows."""
    dataset = ctx.deps.output[name]
    return dataset.head().to_string()  # pyright: ignore[reportUnknownMemberType]


if __name__ == '__main__':
    deps = AnalystAgentDeps(output={})
    result = analyst_agent.run_sync(
        user_prompt='Count how many negative comments are there in the dataset `cornell-movie-review-data/rotten_tomatoes`',
        deps=deps,
    )
    debug(result.output)
