"""Read-only Pixeltable catalog tools for a Pydantic AI agent.

Unstored computed columns rerun their functions on every read, so projections and
filters naming them are rejected. Version handles (`'dir.tbl:3'`) expose old rows
and columns, so paths containing `:` are refused.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Mapping
from datetime import date, datetime
from typing import NoReturn

import pixeltable as pxt
from pixeltable.catalog.table_metadata import ColumnMetadata, TableMetadata
from pixeltable.exprs.expr import Expr
from typing_extensions import TypedDict

from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai_harness.pixeltable._types import column_base

_MEDIA = frozenset({'Image', 'Video', 'Audio', 'Document'})
ALL_TABLES = '*'


WhereValue = str | int | float | bool | None
"""A `query_table` equality-filter value; timestamps, dates, and UUIDs are ISO strings."""


class TableNames(TypedDict):
    """Result of `list_tables`."""

    tables: list[str]


class RowsResult(TypedDict):
    """Result of `query_table` and `similarity_search`."""

    table: str
    rows: list[dict[str, object]]
    truncated: bool


class ColumnDescription(TypedDict):
    """One column in a `describe_table` result."""

    name: str
    type: str
    is_computed: bool
    is_stored: bool
    """`False` for a computed column that reruns its function (possibly a model call) on every read."""


class IndexDescription(TypedDict):
    """One index in a `describe_table` result."""

    name: str
    index_type: str
    columns: list[str]


class TableDescription(TypedDict):
    """Result of `describe_table`."""

    table: str
    kind: str
    comment: str | None
    columns: list[ColumnDescription]
    indexes: list[IndexDescription]


# Accepted Python value types per column base type for `where` equality filters.
_WHERE_VALUE_TYPES: dict[str, type | tuple[type, ...]] = {
    'String': str,
    'Int': int,
    'Float': (int, float),
    'Bool': bool,
    'Timestamp': str,
    'Date': str,
    'UUID': str,
}

# JSON has no timestamp, date, or UUID; the model sends ISO strings, which must be parsed to match.
_WHERE_PARSERS: dict[str, Callable[[str], object]] = {
    'Timestamp': datetime.fromisoformat,
    'Date': date.fromisoformat,
    'UUID': uuid.UUID,
}


def _computed_on_read(info: ColumnMetadata) -> bool:
    # An unstored computed column reruns its function, possibly a paid or side-effecting UDF, on every read.
    return info['is_computed'] and not info['is_stored']


def _norm(path: str) -> str:
    # Pixeltable folds identifiers to lowercase, so 'HR.Handbook' names the table listed as 'hr/handbook'.
    return path.replace('/', '.').lower()


def _allowed(path: str, tables: list[str]) -> bool:
    # A version handle ('tbl:3') reads rows and columns since deleted or dropped; refuse it.
    if ':' in path:
        return False
    if tables == [ALL_TABLES]:
        return True
    npath = _norm(path)
    return any(npath == _norm(entry) or npath.startswith(f'{_norm(entry)}.') for entry in tables)


def _is_media_type(type_: str) -> bool:
    return column_base(type_) in _MEDIA


def _is_skipped_type(type_: str) -> bool:
    return column_base(type_) in {'Array', 'Binary'}


def _cell(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _cell(item) for key, item in value.items()}  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
    if isinstance(value, (list, tuple)):
        return [_cell(item) for item in value]  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
    isoformat = getattr(value, 'isoformat', None)
    if callable(isoformat):
        return isoformat()
    return str(value)


def _retry_or_raise(exc: Exception, prefix: str = '') -> NoReturn:
    if not isinstance(exc, pxt.Error):
        raise ModelRetry(f'{prefix}{exc}') from exc
    if isinstance(exc, (pxt.RequestError, pxt.NotFoundError, pxt.ConcurrencyError)) or (
        isinstance(exc, pxt.ExternalServiceError) and exc.is_retryable
    ):
        raise ModelRetry(f'{prefix}{exc.message}') from exc
    raise exc


def _row(row: Mapping[str, object]) -> dict[str, object]:
    return {key: _cell(value) for key, value in row.items()}


def _clip(value: object, budget: int) -> object:
    """`value` if its JSON fits in `budget` characters; else a string cut to end in `...`, or None."""
    if len(json.dumps(value, ensure_ascii=False)) <= budget:
        return value
    if not isinstance(value, str):
        return None  # a cut dict or number would read as real data; `truncated` flags the gap
    text = value
    keep = budget - 5  # quotes plus '...'
    # Escapes make the JSON longer than the text; shrink in proportion until it fits.
    while keep > 0 and (size := len(json.dumps(text[:keep], ensure_ascii=False))) + 3 > budget:
        keep = min(keep - 1, keep * (budget - 3) // size)
    return f'{text[:keep]}...' if keep > 0 else '...'


def _bounded_payload(table: str, rows: list[dict[str, object]], *, has_more: bool, max_chars: int) -> RowsResult:
    """Keep leading rows while the JSON fits in `max_chars`.

    The first row that does not fit has its cells cut to share the remaining room (short cells
    first, so they stay whole), so one oversized value cannot empty the result. Rows after it
    are dropped.
    """
    kept: list[dict[str, object]] = []
    truncated = has_more
    used = len(json.dumps(RowsResult(table=table, rows=[], truncated=False), ensure_ascii=False))
    for row in rows:
        separator = 2 if kept else 0
        size = len(json.dumps(row, ensure_ascii=False)) + separator
        if used + size <= max_chars:
            kept.append(row)
            used += size
            continue
        truncated = True
        sizes = {key: len(json.dumps(value, ensure_ascii=False)) for key, value in row.items()}
        room = max_chars - used - separator - (len(json.dumps(dict.fromkeys(row, 0), ensure_ascii=False)) - len(row))
        budgets: dict[str, int] = {}
        for index, key in enumerate(sorted(sizes, key=sizes.__getitem__)):
            budgets[key] = min(sizes[key], room // (len(sizes) - index))
            room -= budgets[key]
        clipped = {key: _clip(value, budgets[key]) for key, value in row.items()}
        if used + separator + len(json.dumps(clipped, ensure_ascii=False)) <= max_chars:
            kept.append(clipped)
        break
    return RowsResult(table=table, rows=kept, truncated=truncated)


class PixeltableToolset(FunctionToolset[AgentDepsT]):
    """List, describe, query, and similarity-search Pixeltable tables."""

    def __init__(self, *, tables: list[str], max_rows: int, max_chars: int, id: str = 'pixeltable') -> None:
        super().__init__(id=id)
        if not tables:
            raise ValueError("tables must be a non-empty allowlist; pass ['*'] to allow the whole catalog")
        self._tables = tables
        self._max_rows = max_rows
        self._max_chars = max_chars
        self.add_function(self.list_tables, name='list_tables')
        self.add_function(self.describe_table, name='describe_table')
        self.add_function(self.query_table, name='query_table')
        self.add_function(self.similarity_search, name='similarity_search')

    def list_tables(self) -> TableNames:
        """List the allowed Pixeltable table and view paths."""
        try:
            tables = pxt.list_tables()
        except pxt.Error as exc:
            _retry_or_raise(exc)
        return TableNames(tables=sorted(_norm(path) for path in tables if _allowed(path, self._tables)))

    def describe_table(self, table: str) -> TableDescription:
        """Describe a table's columns and indexes.

        Args:
            table: Pixeltable table path (for example `my_app.doc_chunks`).

        Returns:
            Kind, comment, columns, and indexes.
        """
        t = self._open_table(table)
        metadata = self._metadata(t)
        return TableDescription(
            table=_norm(metadata['path']),
            kind=metadata['kind'],
            comment=metadata['comment'],
            columns=[
                ColumnDescription(
                    name=info['name'], type=info['type_'], is_computed=info['is_computed'], is_stored=info['is_stored']
                )
                for info in metadata['columns'].values()
            ],
            indexes=[
                IndexDescription(name=info['name'], index_type=info['index_type'], columns=info['columns'])
                for info in metadata['indexes'].values()
            ],
        )

    def query_table(
        self,
        table: str,
        columns: list[str] | None = None,
        where: dict[str, WhereValue] | None = None,
        limit: int = 10,
    ) -> RowsResult:
        """Select rows from a table. `where` is equality only (`{"status": "open"}`).

        Args:
            table: Pixeltable table path.
            columns: Columns to return. Omit to skip media, array, and binary
                columns. Explicit media, array, binary, and unstored computed columns are rejected.
            where: Equality filters mapping column name to value. Media, array, and
                binary columns reject non-null filters; `None` matches null rows.
            limit: Maximum rows to return, capped by the capability.

        Returns:
            Matching rows, with `truncated` set when the row or character cap applied.
        """
        t = self._open_table(table)
        metadata = self._metadata(t)
        try:
            query = self._project(t, columns, metadata)
        except (pxt.Error, TypeError, ValueError) as exc:
            _retry_or_raise(exc)
        query = self._where(query, t, where, metadata)
        return self._collect(table, query, limit)

    def similarity_search(
        self,
        table: str,
        query: str,
        column: str,
        columns: list[str] | None = None,
        limit: int = 5,
        idx: str | None = None,
    ) -> RowsResult:
        """Nearest-neighbor search on a column that has an embedding index.

        Args:
            table: Pixeltable table path.
            query: Text to embed and search for.
            column: Column with an embedding index.
            columns: Extra columns to return with the match and score. Indexed media is
                omitted unless explicitly requested. Explicit media, array, binary, and unstored
                computed columns are rejected.
            limit: Maximum rows to return, capped by the capability.
            idx: Embedding index name. Required when the column has more than one index.

        Returns:
            Rows ordered by similarity, each with a `score` field. If the table has a column by
            that name, the field takes `similarity_` prefixes until it is free.
        """
        if not query.strip():
            raise ModelRetry('similarity_search query must be non-empty')
        t = self._open_table(table)
        metadata = self._metadata(t)
        column_md = metadata['columns']
        if column not in column_md:
            raise ModelRetry(f'Unknown column {column!r} on {_norm(metadata["path"])!r}. Call describe_table.')
        selected = self._default_columns(metadata) if columns is None else list(columns)
        if column not in selected and not (
            _is_skipped_type(column_md[column]['type_']) or _is_media_type(column_md[column]['type_'])
        ):
            selected = [column, *[name for name in selected if name != column]]
        # A real column named `score` must not be shadowed by the similarity score.
        score_name = 'score'
        while score_name in column_md:
            score_name = f'similarity_{score_name}'
        try:
            search = t[column].similarity(string=query, idx=idx)  # pyright: ignore[reportUnknownMemberType]
            query_obj = self._project(t, selected, metadata, **{score_name: search}).order_by(search, asc=False)
            assert isinstance(query_obj, pxt.Query)
            return self._collect(table, query_obj, limit)
        except pxt.Error as exc:
            _retry_or_raise(exc)

    def _metadata(self, t: pxt.Table) -> TableMetadata:
        try:
            return t.get_metadata()
        except pxt.Error as exc:
            _retry_or_raise(exc)

    def _open_table(self, table: str) -> pxt.Table:
        if not _allowed(table, self._tables):
            raise ModelRetry(f'Table {table!r} is not in the Pixeltable allowlist. Call list_tables.')
        # No handle cache: resolving the name on every call keeps a moved or dropped table
        # from being served under an allowlisted name.
        try:
            t = pxt.get_table(table)
        except pxt.Error as exc:
            _retry_or_raise(exc, f'Cannot open table {table!r}: ')
        assert t is not None  # if_not_exists='error' raises instead of returning None
        return t

    def _default_columns(self, metadata: TableMetadata) -> list[str]:
        columns: list[str] = []
        for name, info in metadata['columns'].items():
            type_ = info['type_']
            if _is_media_type(type_) or _is_skipped_type(type_) or _computed_on_read(info):
                continue
            columns.append(name)
        return columns

    def _project(self, t: pxt.Table, columns: list[str] | None, metadata: TableMetadata, **extra: Expr) -> pxt.Query:
        path = _norm(metadata['path'])
        column_md = metadata['columns']
        if columns is None:
            names = self._default_columns(metadata)
        else:
            names = list(dict.fromkeys(columns))
            if not names:
                raise ModelRetry('query_table needs at least one column')
        items: list[Expr] = []
        named: dict[str, Expr] = dict(extra)
        for name in names:
            if name in named:
                continue
            if name not in column_md:
                raise ModelRetry(f'Unknown column {name!r} on {path!r}. Call describe_table.')
            if _computed_on_read(column_md[name]):
                raise ModelRetry(f'Column {name!r} is computed on read; these tools do not run it.')
            type_ = column_md[name]['type_']
            if _is_skipped_type(type_):
                raise ModelRetry(f'Column {name!r} has an array or binary type; choose a stored scalar column.')
            ref = t[name]
            if _is_media_type(type_):
                raise ModelRetry(
                    f'Media column {name!r} exposes a local file path; use a safe application URL instead.'
                )
            else:
                items.append(ref)
        if not items and not named:
            raise ModelRetry('No selectable columns. Name a non-array column, or describe_table.')
        return t.select(*items, **named)

    def _where(
        self, query: pxt.Query, t: pxt.Table, where: dict[str, WhereValue] | None, metadata: TableMetadata
    ) -> pxt.Query:
        if not where:
            return query
        column_md = metadata['columns']
        pred: Expr | None = None
        for name, raw_value in where.items():
            if name not in column_md:
                raise ModelRetry(f'Unknown column {name!r} in where. Call describe_table.')
            info = column_md[name]
            type_ = info['type_']
            if _computed_on_read(info):
                raise ModelRetry(f'Column {name!r} is computed on read; these tools do not run it.')
            if raw_value is not None and (_is_skipped_type(type_) or _is_media_type(type_)):
                raise ModelRetry(f'Column {name!r} does not support equality filters. Call describe_table.')
            base = column_base(type_)
            expected = _WHERE_VALUE_TYPES.get(base)
            if (
                raw_value is not None
                and expected is not None
                and (not isinstance(raw_value, expected) or (isinstance(raw_value, bool) and base != 'Bool'))
            ):
                raise ModelRetry(f'where value for {name!r} must be {base}, got {type(raw_value).__name__}')
            value: object = raw_value
            parse = _WHERE_PARSERS.get(base)
            if parse is not None and isinstance(raw_value, str):
                try:
                    value = parse(raw_value)
                except ValueError as exc:
                    raise ModelRetry(f'where value for {name!r} must be an ISO-format {base}: {exc}') from exc
            try:
                clause = t[name] == value
                pred = clause if pred is None else pred & clause
            except (pxt.Error, TypeError, ValueError) as exc:
                _retry_or_raise(exc, f'Cannot filter {name!r} by {value!r}: ')
        if pred is None:  # pragma: no cover - `where` is non-empty here
            return query
        try:
            filtered = query.where(pred)
            assert isinstance(filtered, pxt.Query)
            return filtered
        except (pxt.Error, TypeError, ValueError) as exc:
            _retry_or_raise(exc)

    def _collect(self, table: str, query: pxt.Query, limit: int) -> RowsResult:
        if limit < 1:
            raise ModelRetry('limit must be at least 1')
        n = min(limit, self._max_rows)
        try:
            limited = query.limit(n + 1)
            assert isinstance(limited, pxt.Query)
            fetched = list(limited.collect())
        except pxt.Error as exc:
            _retry_or_raise(exc)
        has_more = len(fetched) > n
        rows = [_row(dict(row)) for row in fetched[:n]]
        return _bounded_payload(_norm(table), rows, has_more=has_more, max_chars=self._max_chars)
