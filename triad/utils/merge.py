"""Three-way merge utilities for :class:`pyarrow.Table`.

The module implements a column-wise three-way merge of Arrow tables that are
identified by a declared unique, non-null business key. It is meant for the
common offline-cleaning workflow where ``base`` is the shared baseline and
``local`` / ``incoming`` are two independently edited snapshots of it.

The merge is deliberately conservative:

* a row deleted on one side and left untouched on the other is deleted,
* a column changed on one side only keeps that change,
* a column changed identically on both sides keeps the shared change,
* a row deleted on one side and modified on the other is a row-level conflict,
* a column changed differently on both sides is a column-level conflict,
* a key added on both sides with different contents is a row-level conflict.

Nothing is silently overwritten. Conflicts retain the key, the column (or the
whole row) and the values / presence from all three sides. A usable merged
table is only produced once every conflict has been resolved by the caller.

Examples:
    >>> result = merge_tables(base, local, incoming, key="id")
    >>> result.conflicts
    [...]
    >>> merged = result.resolve({0: "local", 1: "incoming"})
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple, Union

import pyarrow as pa

CONFLICT_VALUE = "value"
"""A non-key column was changed differently by ``local`` and ``incoming``."""

CONFLICT_DELETE_MODIFY = "delete_modify"
"""One side deleted a row while the other side modified it."""

CONFLICT_ADD_ADD = "add_add"
"""Both sides added the same key with different contents."""

_BASE = "base"
_LOCAL = "local"
_INCOMING = "incoming"
_SIDES = (_BASE, _LOCAL, _INCOMING)
_DELETE_CHOICES = frozenset({"delete", "drop", "remove"})

KeyColumns = Union[str, Sequence[str]]
Decisions = Union[
    Mapping[Any, Any],
    Sequence[Any],
    Callable[["MergeConflict"], Any],
    str,
    None,
]


class MergeError(Exception):
    """Base class for three-way merge errors."""


class MergeSchemaError(MergeError, ValueError):
    """Raised when the three tables don't share the same schema."""


class MergeKeyError(MergeError, ValueError):
    """Raised when the business key is invalid, null or not unique."""


class MergeConflictError(MergeError):
    """Raised when unresolved conflicts prevent producing a merged table."""

    def __init__(
        self,
        message: str,
        conflicts: Optional[Sequence["MergeConflict"]] = None,
    ):
        super().__init__(message)
        self.conflicts: List["MergeConflict"] = list(conflicts or [])


@dataclass(frozen=True)
class MergeConflict:
    """A single unresolved three-way merge conflict.

    For a column-level conflict :attr:`column` is the column name and
    :attr:`base`, :attr:`local`, :attr:`incoming` are the individual cell
    values. For a row-level conflict :attr:`column` is ``None`` and the three
    values are complete row dictionaries (``None`` when that side doesn't
    contain the row).

    The ``*_exists`` flags describe row presence. They are what distinguishes
    "the row is absent on this side" from "the row exists and holds ``NULL``".
    """

    key: Tuple[Any, ...]
    kind: str
    column: Optional[str]
    base: Any
    local: Any
    incoming: Any
    base_exists: bool
    local_exists: bool
    incoming_exists: bool
    key_columns: Tuple[str, ...] = ()
    index: int = -1

    def __hash__(self) -> int:  # pragma: no cover - only for convenient lookups
        return hash(self.index)

    @property
    def key_dict(self) -> Dict[str, Any]:
        """The business key as a column name to value mapping."""
        return dict(zip(self.key_columns, self.key))

    @property
    def key_value(self) -> Any:
        """The single key value, or the key tuple for a composite key."""
        if len(self.key) == 1:
            return self.key[0]
        return self.key

    @property
    def scope(self) -> str:
        """``"column"`` for column conflicts, ``"row"`` for row conflicts."""
        return "column" if self.column is not None else "row"

    @property
    def is_column_conflict(self) -> bool:
        """Whether this is a column-level conflict."""
        return self.column is not None

    @property
    def is_row_conflict(self) -> bool:
        """Whether this is a row-level conflict."""
        return self.column is None

    @property
    def base_value(self) -> Any:
        """Alias of :attr:`base`."""
        return self.base

    @property
    def local_value(self) -> Any:
        """Alias of :attr:`local`."""
        return self.local

    @property
    def incoming_value(self) -> Any:
        """Alias of :attr:`incoming`."""
        return self.incoming

    @property
    def values(self) -> Tuple[Any, Any, Any]:
        """The ``(base, local, incoming)`` values."""
        return (self.base, self.local, self.incoming)

    @property
    def existence(self) -> Tuple[bool, bool, bool]:
        """The ``(base, local, incoming)`` row existence flags."""
        return (self.base_exists, self.local_exists, self.incoming_exists)

    @property
    def deleted_sides(self) -> Tuple[str, ...]:
        """Names of the sides that don't contain the row."""
        sides = (self.base_exists, self.local_exists, self.incoming_exists)
        return tuple(side for side, exists in zip(_SIDES, sides) if not exists)


@dataclass
class _TableIndex:
    rows: List[Dict[str, Any]]
    keys: List[Tuple[Any, ...]]
    frozen: List[Any]
    by_key: Dict[Any, int]


@dataclass
class _RowEntry:
    key: Tuple[Any, ...]
    cells: List[Optional[Any]]
    col_conflicts: Dict[int, int]
    row_conflict: Optional[int] = None
    candidates: Tuple[Optional[int], Optional[int], Optional[int]] = (
        None,
        None,
        None,
    )


class MergeResult:
    """The outcome of :func:`merge_tables`.

    ``conflicts`` lists every unresolved conflict. ``table`` (and its alias
    ``merged_table``) raises :class:`MergeConflictError` while conflicts are
    outstanding. Call :meth:`resolve` with the caller's choices to obtain the
    final :class:`pyarrow.Table`.
    """

    def __init__(
        self,
        schema: pa.Schema,
        entries: List[_RowEntry],
        conflicts: List[MergeConflict],
        source_arrays: List[List[pa.Array]],
        key_columns: Tuple[str, ...],
    ):
        self._schema = schema
        self._entries = entries
        self._conflicts = conflicts
        self._source_arrays = source_arrays
        self._key_columns = key_columns
        self._clean_table: Optional[pa.Table] = None

    @property
    def schema(self) -> pa.Schema:
        """The schema of the merged table (the base table's schema)."""
        return self._schema

    @property
    def key_columns(self) -> Tuple[str, ...]:
        """The declared business key columns."""
        return self._key_columns

    @property
    def conflicts(self) -> List[MergeConflict]:
        """All unresolved conflicts, in deterministic order."""
        return self._conflicts

    @property
    def has_conflicts(self) -> bool:
        """Whether any conflict is still unresolved."""
        return len(self._conflicts) > 0

    @property
    def is_clean(self) -> bool:
        """Whether the merge completed without conflicts."""
        return len(self._conflicts) == 0

    @property
    def table(self) -> pa.Table:
        """The merged table, raising if conflicts are not resolved yet."""
        if self._conflicts:
            raise MergeConflictError(
                f"{len(self._conflicts)} unresolved merge conflict(s); "
                "call resolve() with a choice for each conflict",
                self._conflicts,
            )
        if self._clean_table is None:
            self._clean_table = self._materialize({})
        return self._clean_table

    @property
    def merged_table(self) -> pa.Table:
        """Alias of :attr:`table`."""
        return self.table

    @property
    def merged(self) -> pa.Table:
        """Alias of :attr:`table`."""
        return self.table

    def __len__(self) -> int:
        return self.table.num_rows

    def __getitem__(self, item: Any) -> Any:
        return self.table[item]

    def __getattr__(self, name: str) -> Any:
        # Delegate the rest of the pyarrow.Table API so a conflict-free result
        # can be used wherever a table is expected. Conflicts still raise.
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            table = object.__getattribute__(self, "table")
        except AttributeError:  # pragma: no cover - defensive
            raise AttributeError(name)
        return getattr(table, name)

    def to_table(self) -> pa.Table:
        """Return the merged table, raising if conflicts are unresolved."""
        return self.table

    def to_batches(self, max_chunksize: Optional[int] = None) -> List[pa.RecordBatch]:
        """Return the merged table as Arrow record batches."""
        return self.table.to_batches(max_chunksize=max_chunksize)

    def resolve(
        self,
        decisions: Decisions = None,
        *,
        default: Any = None,
    ) -> pa.Table:
        """Resolve every conflict and return the merged table.

        :param decisions: how each conflict is resolved. Accepted forms are:

            * ``None`` - legal only when there are no conflicts or ``default``
              covers them,
            * a single choice such as ``"local"`` applied to every conflict,
            * a callable called with each :class:`MergeConflict`,
            * a sequence of choices parallel to :attr:`conflicts`,
            * a mapping keyed by conflict index, by :class:`MergeConflict`, or
              by ``(key, column)`` (``column`` is ``None`` for row conflicts).

          A choice is usually ``"base"``, ``"local"``, ``"incoming"``, or
          ``"delete"``. For a column conflict ``None`` means ``NULL`` and any
          other value is used verbatim. For a row conflict a mapping or a
          sequence of values defines an explicit row.
        :param default: an optional choice used for every conflict not present
            in ``decisions``.
        :return: the merged :class:`pyarrow.Table`
        """
        if not self._conflicts and decisions is None and default is None:
            return self.table
        resolved = self._collect_decisions(decisions, default)
        missing = [c for c in self._conflicts if c.index not in resolved]
        if missing:
            raise MergeConflictError(
                f"{len(missing)} merge conflict(s) have no decision",
                missing,
            )
        return self._materialize(resolved)

    def resolve_all(self, choice: Any = "local") -> pa.Table:
        """Resolve every conflict with the same choice."""
        return self.resolve(default=choice)

    def _collect_decisions(self, decisions: Decisions, default: Any) -> Dict[int, Any]:
        resolved: Dict[int, Any] = {}
        if isinstance(decisions, str):
            if default is not None:
                raise MergeError("can't set both a string decision and a default")
            default = decisions
            decisions = None
        if default is not None:
            for conflict in self._conflicts:
                resolved[conflict.index] = self._apply_callable(default, conflict)
        if decisions is None:
            return resolved
        if callable(decisions):
            for conflict in self._conflicts:
                resolved[conflict.index] = self._apply_callable(decisions, conflict)
            return resolved
        if isinstance(decisions, Mapping):
            for key, choice in decisions.items():
                resolved[self._match_decision_key(key)] = choice
            return resolved
        if isinstance(decisions, Sequence) and not isinstance(
            decisions, (str, bytes, bytearray)
        ):
            choices = list(decisions)
            if len(choices) != len(self._conflicts):
                raise MergeError(
                    f"expected {len(self._conflicts)} decisions, got {len(choices)}"
                )
            for conflict, choice in zip(self._conflicts, choices):
                resolved[conflict.index] = choice
            return resolved
        raise MergeError(f"invalid decisions: {decisions!r}")

    @staticmethod
    def _apply_callable(func: Callable[..., Any], conflict: MergeConflict) -> Any:
        if not callable(func):
            return func
        try:
            signature = inspect.signature(func)
        except (TypeError, ValueError):  # pragma: no cover - builtins etc.
            return func(conflict)
        positional = 0
        for parameter in signature.parameters.values():
            if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
                positional = 2
                break
            if parameter.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                positional += 1
        if positional >= 2:
            return func(conflict, conflict.index)
        return func(conflict)

    def _match_decision_key(self, key: Any) -> int:
        if isinstance(key, int) and not isinstance(key, bool):
            if 0 <= key < len(self._conflicts):
                return key
            raise MergeError(f"conflict index out of range: {key}")
        if isinstance(key, MergeConflict):
            for conflict in self._conflicts:
                if conflict is key or conflict == key:
                    return conflict.index
            raise MergeError(f"unknown conflict: {key!r}")
        if isinstance(key, tuple) and len(key) == 2:
            raw_key, column = key
            return self._find_conflict(raw_key, column)
        raise MergeError(f"invalid conflict key: {key!r}")

    def _find_conflict(self, raw_key: Any, column: Any) -> int:
        key = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        for conflict in self._conflicts:
            if conflict.key == key and conflict.column == column:
                return conflict.index
        raise MergeError(f"no conflict for key {raw_key!r} and column {column!r}")

    def _materialize(self, resolved: Dict[int, Any]) -> pa.Table:
        rows: List[List[Any]] = []
        for entry in self._entries:
            if entry.row_conflict is not None:
                cells = self._resolve_row(entry, resolved[entry.row_conflict])
                if cells is not None:
                    rows.append(cells)
            else:
                rows.append(self._resolve_cells(entry, resolved))
        return self._build_table(rows)

    def _resolve_cells(self, entry: _RowEntry, resolved: Dict[int, Any]) -> List[Any]:
        cells: List[Any] = []
        for column, cell in enumerate(entry.cells):
            if cell is not None:
                cells.append(cell)
            else:
                conflict = self._conflicts[entry.col_conflicts[column]]
                cells.append(
                    self._resolve_column(entry, column, resolved[conflict.index])
                )
        return cells

    def _resolve_column(self, entry: _RowEntry, column: int, choice: Any) -> Any:
        name = self._schema.field(column).name
        if isinstance(choice, str) and choice in _SIDES:
            side = _SIDES.index(choice)
            index = entry.candidates[side]
            if index is None:  # pragma: no cover - column conflicts always have rows
                raise MergeError(
                    f"side {choice!r} doesn't contain the row for column {name!r}"
                )
            return ("cell", side, index)
        if isinstance(choice, str) and choice in _DELETE_CHOICES:
            raise MergeError(f"can't delete column {name!r}")
        return ("value", choice)

    def _resolve_row(self, entry: _RowEntry, choice: Any) -> Optional[List[Any]]:
        if choice is None or (isinstance(choice, str) and choice in _DELETE_CHOICES):
            return None
        if isinstance(choice, str):
            if choice not in _SIDES:
                raise MergeError(f"invalid row decision: {choice!r}")
            side = _SIDES.index(choice)
            index = entry.candidates[side]
            if index is None:
                return None
            return [("cell", side, index) for _ in self._schema]
        if isinstance(choice, Mapping):
            names = [field.name for field in self._schema]
            missing = [name for name in names if name not in choice]
            if missing:
                raise MergeError(f"explicit row is missing columns: {missing}")
            return [("value", choice[name]) for name in names]
        if isinstance(choice, Sequence) and not isinstance(
            choice, (str, bytes, bytearray)
        ):
            values = list(choice)
            if len(values) != len(self._schema):
                raise MergeError(
                    f"explicit row has {len(values)} values, "
                    f"expected {len(self._schema)}"
                )
            return [("value", value) for value in values]
        raise MergeError(f"invalid row decision: {choice!r}")

    def _build_table(self, rows: List[List[Any]]) -> pa.Table:
        if not rows:
            arrays = [pa.array([], type=field.type) for field in self._schema]
            return pa.Table.from_arrays(arrays, schema=self._schema)
        arrays = [
            self._build_column(rows, column) for column in range(len(self._schema))
        ]
        return pa.Table.from_arrays(arrays, schema=self._schema)

    def _build_column(self, rows: List[List[Any]], column: int) -> pa.Array:
        field_type = self._schema.field(column).type
        pieces: List[pa.Array] = []
        side: Optional[int] = None
        indices: List[int] = []

        def flush() -> None:
            if side is not None:
                index = pa.array(indices, type=pa.int64())
                pieces.append(self._source_arrays[side][column].take(index))

        for cells in rows:
            cell = cells[column]
            if cell[0] == "cell" and cell[1] == side:
                indices.append(cell[2])
                continue
            flush()
            if cell[0] == "cell":
                side = cell[1]
                indices = [cell[2]]
            else:
                side = None
                indices = []
                if cell[1] is None and not self._schema.field(column).nullable:
                    raise MergeError(
                        "can't set NULL to non-nullable column "
                        f"{self._schema.field(column).name!r}"
                    )
                pieces.append(pa.array([cell[1]], type=field_type))
        flush()
        if len(pieces) == 1:
            return pieces[0]
        return pa.concat_arrays(pieces)


def merge_tables(
    base: pa.Table,
    local: pa.Table,
    incoming: pa.Table,
    keys: Optional[KeyColumns] = None,
    *,
    key: Optional[KeyColumns] = None,
    on: Optional[KeyColumns] = None,
    sort_by_key: bool = False,
) -> MergeResult:
    """Three-way merge Arrow tables by a unique, non-null business key.

    ``base`` is the common baseline, ``local`` and ``incoming`` are two
    independent edits of it. Inputs don't need to be sorted.

    Because identity comes from the key, a side that changes a key column is
    seen as deleting the old key and adding the new one; the change is never
    applied to key columns of an existing row.

    The result is a :class:`MergeResult`. If it has no conflicts,
    ``result.table`` (or ``result.resolve()``) is the merged
    :class:`pyarrow.Table`. Otherwise ``result.conflicts`` describes every
    conflict and :meth:`MergeResult.resolve` must be called with the caller's
    choices before a usable table is returned.

    :param base: the baseline table
    :param local: one edited snapshot of ``base``
    :param incoming: another edited snapshot of ``base``
    :param keys: column name or names forming the business key
    :param key: alias of ``keys``
    :param on: alias of ``keys``
    :param sort_by_key: if True, sort the merged rows by key; by default rows
        keep ``base`` order and new rows are appended in ``local`` order then
        ``incoming``-only order
    :return: a :class:`MergeResult`
    :raises MergeSchemaError: if the three schemas differ
    :raises MergeKeyError: if the key is missing, null, or duplicated
    """
    key_columns = _resolve_key_columns(keys, key, on)
    _validate_tables(base, local, incoming)
    _validate_key_columns(base.schema, key_columns)
    schema = base.schema
    key_positions = {schema.get_field_index(name) for name in key_columns}
    base_index = _index_table(base, key_columns, "base")
    local_index = _index_table(local, key_columns, "local")
    incoming_index = _index_table(incoming, key_columns, "incoming")
    entries: List[_RowEntry] = []
    conflicts: List[MergeConflict] = []
    for position, key_value in enumerate(base_index.keys):
        local_position = local_index.by_key.get(base_index.frozen[position])
        incoming_position = incoming_index.by_key.get(base_index.frozen[position])
        _merge_base_row(
            position,
            key_value,
            local_position,
            incoming_position,
            key_columns,
            key_positions,
            base_index,
            local_index,
            incoming_index,
            entries,
            conflicts,
        )
    _merge_additions(
        key_columns,
        len(schema),
        base_index,
        local_index,
        incoming_index,
        entries,
        conflicts,
    )
    if sort_by_key:
        entries.sort(key=lambda entry: _sort_key(entry.key))
    source_arrays = [
        [table.column(i).combine_chunks() for i in range(table.num_columns)]
        for table in (base, local, incoming)
    ]
    return MergeResult(schema, entries, conflicts, source_arrays, key_columns)


def merge_arrow_tables(
    base: pa.Table,
    local: pa.Table,
    incoming: pa.Table,
    keys: Optional[KeyColumns] = None,
    **kwargs: Any,
) -> MergeResult:
    """Alias of :func:`merge_tables`."""
    return merge_tables(base, local, incoming, keys, **kwargs)


def merge_pa_tables(
    base: pa.Table,
    local: pa.Table,
    incoming: pa.Table,
    keys: Optional[KeyColumns] = None,
    **kwargs: Any,
) -> MergeResult:
    """Alias of :func:`merge_tables`."""
    return merge_tables(base, local, incoming, keys, **kwargs)


def three_way_merge(
    base: pa.Table,
    local: pa.Table,
    incoming: pa.Table,
    keys: Optional[KeyColumns] = None,
    **kwargs: Any,
) -> MergeResult:
    """Alias of :func:`merge_tables`."""
    return merge_tables(base, local, incoming, keys, **kwargs)


merge_three_way = three_way_merge


def resolve_merge_conflicts(
    result: MergeResult,
    decisions: Decisions = None,
    *,
    default: Any = None,
) -> pa.Table:
    """Resolve a :class:`MergeResult` and return its merged table."""
    return result.resolve(decisions, default=default)


resolve_conflicts = resolve_merge_conflicts
TableMergeResult = MergeResult
TableMergeConflict = MergeConflict


def _resolve_key_columns(
    keys: Optional[KeyColumns],
    key: Optional[KeyColumns],
    on: Optional[KeyColumns],
) -> Tuple[str, ...]:
    supplied = [value for value in (keys, key, on) if value is not None]
    if len(supplied) > 1:
        raise MergeKeyError("only one of keys, key and on can be specified")
    if not supplied:
        raise MergeKeyError("business key columns must be specified")
    value = supplied[0]
    columns = [value] if isinstance(value, str) else list(value)
    if not columns:
        raise MergeKeyError("business key can't be empty")
    if any(not isinstance(column, str) for column in columns):
        raise MergeKeyError(f"invalid business key columns: {value!r}")
    if len(set(columns)) != len(columns):
        raise MergeKeyError(f"duplicate business key columns: {columns!r}")
    return tuple(columns)


def _validate_tables(base: pa.Table, local: pa.Table, incoming: pa.Table) -> None:
    for name, table in (
        (_BASE, base),
        (_LOCAL, local),
        (_INCOMING, incoming),
    ):
        if not isinstance(table, pa.Table):
            raise TypeError(f"{name} must be a pyarrow.Table")
    _validate_schema(base.schema, local.schema, _LOCAL)
    _validate_schema(base.schema, incoming.schema, _INCOMING)


def _validate_schema(base: pa.Schema, other: pa.Schema, name: str) -> None:
    if len(base) != len(other):
        raise MergeSchemaError(f"{name} has {len(other)} columns, base has {len(base)}")
    if len(set(base.names)) != len(base.names):
        raise MergeSchemaError(f"base has duplicated column names: {base.names!r}")
    for left, right in zip(base, other):
        if left.name != right.name:
            raise MergeSchemaError(
                f"column name mismatch: base has {left.name!r}, "
                f"{name} has {right.name!r}"
            )
        if not left.type.equals(right.type):
            raise MergeSchemaError(
                f"column {left.name!r} type mismatch: base has {left.type}, "
                f"{name} has {right.type}"
            )


def _validate_key_columns(schema: pa.Schema, key_columns: Tuple[str, ...]) -> None:
    for column in key_columns:
        if column not in schema.names:
            raise MergeKeyError(f"key column {column!r} is not in the schema")


def _index_table(
    table: pa.Table, key_columns: Tuple[str, ...], name: str
) -> _TableIndex:
    rows = table.to_pylist()
    keys = [tuple(row[column] for column in key_columns) for row in rows]
    frozen: List[Any] = []
    by_key: Dict[Any, int] = {}
    for position, key in enumerate(keys):
        if any(value is None for value in key):
            raise MergeKeyError(f"{name} has a null business key: {key!r}")
        token = _freeze(key)
        if token in by_key:
            raise MergeKeyError(f"{name} has a duplicated business key: {key!r}")
        by_key[token] = position
        frozen.append(token)
    return _TableIndex(rows, keys, frozen, by_key)


def _merge_base_row(  # noqa: C901
    base_position: int,
    key: Tuple[Any, ...],
    local_position: Optional[int],
    incoming_position: Optional[int],
    key_columns: Tuple[str, ...],
    key_positions: set,
    base_index: _TableIndex,
    local_index: _TableIndex,
    incoming_index: _TableIndex,
    entries: List[_RowEntry],
    conflicts: List[MergeConflict],
) -> None:
    base_row = base_index.rows[base_position]
    if local_position is None and incoming_position is None:
        return
    if local_position is None:
        if _rows_equal(base_row, incoming_index.rows[incoming_position], base_row):
            return
        _append_row_conflict(
            key,
            key_columns,
            CONFLICT_DELETE_MODIFY,
            base_row,
            None,
            incoming_index.rows[incoming_position],
            True,
            False,
            True,
            (base_position, None, incoming_position),
            entries,
            conflicts,
        )
        return
    if incoming_position is None:
        if _rows_equal(base_row, local_index.rows[local_position], base_row):
            return
        _append_row_conflict(
            key,
            key_columns,
            CONFLICT_DELETE_MODIFY,
            base_row,
            local_index.rows[local_position],
            None,
            True,
            True,
            False,
            (base_position, local_position, None),
            entries,
            conflicts,
        )
        return
    _merge_present_row(
        key,
        key_columns,
        key_positions,
        (base_position, local_position, incoming_position),
        base_row,
        local_index.rows[local_position],
        incoming_index.rows[incoming_position],
        entries,
        conflicts,
    )


def _merge_present_row(  # noqa: C901
    key: Tuple[Any, ...],
    key_columns: Tuple[str, ...],
    key_positions: set,
    candidates: Tuple[int, int, int],
    base_row: Dict[str, Any],
    local_row: Dict[str, Any],
    incoming_row: Dict[str, Any],
    entries: List[_RowEntry],
    conflicts: List[MergeConflict],
) -> None:
    cells: List[Optional[Any]] = []
    col_conflicts: Dict[int, int] = {}
    for column, name in enumerate(base_row):
        if column in key_positions:
            cells.append(("cell", 0, candidates[0]))
            continue
        base_value = base_row[name]
        local_value = local_row[name]
        incoming_value = incoming_row[name]
        local_same = _value_equal(base_value, local_value)
        incoming_same = _value_equal(base_value, incoming_value)
        if local_same and incoming_same:
            cells.append(("cell", 0, candidates[0]))
        elif local_same:
            cells.append(("cell", 2, candidates[2]))
        elif incoming_same:
            cells.append(("cell", 1, candidates[1]))
        elif _value_equal(local_value, incoming_value):
            cells.append(("cell", 1, candidates[1]))
        else:
            conflict = MergeConflict(
                key=key,
                kind=CONFLICT_VALUE,
                column=name,
                base=base_value,
                local=local_value,
                incoming=incoming_value,
                base_exists=True,
                local_exists=True,
                incoming_exists=True,
                key_columns=key_columns,
                index=len(conflicts),
            )
            conflicts.append(conflict)
            cells.append(None)
            col_conflicts[column] = conflict.index
    entries.append(
        _RowEntry(
            key=key,
            cells=cells,
            col_conflicts=col_conflicts,
            candidates=candidates,
        )
    )


def _append_row_conflict(
    key: Tuple[Any, ...],
    key_columns: Tuple[str, ...],
    kind: str,
    base_row: Optional[Dict[str, Any]],
    local_row: Optional[Dict[str, Any]],
    incoming_row: Optional[Dict[str, Any]],
    base_exists: bool,
    local_exists: bool,
    incoming_exists: bool,
    candidates: Tuple[Optional[int], Optional[int], Optional[int]],
    entries: List[_RowEntry],
    conflicts: List[MergeConflict],
) -> None:
    conflict = MergeConflict(
        key=key,
        kind=kind,
        column=None,
        base=base_row,
        local=local_row,
        incoming=incoming_row,
        base_exists=base_exists,
        local_exists=local_exists,
        incoming_exists=incoming_exists,
        key_columns=key_columns,
        index=len(conflicts),
    )
    conflicts.append(conflict)
    entries.append(
        _RowEntry(
            key=key,
            cells=[],
            col_conflicts={},
            row_conflict=conflict.index,
            candidates=candidates,
        )
    )


def _merge_additions(
    key_columns: Tuple[str, ...],
    num_columns: int,
    base_index: _TableIndex,
    local_index: _TableIndex,
    incoming_index: _TableIndex,
    entries: List[_RowEntry],
    conflicts: List[MergeConflict],
) -> None:
    for local_position, key in enumerate(local_index.keys):
        if local_index.frozen[local_position] in base_index.by_key:
            continue
        incoming_position = incoming_index.by_key.get(
            local_index.frozen[local_position]
        )
        if incoming_position is None:
            entries.append(_whole_row_entry(key, 1, local_position, num_columns))
        elif _rows_equal(
            local_index.rows[local_position],
            incoming_index.rows[incoming_position],
            local_index.rows[local_position],
        ):
            entries.append(_whole_row_entry(key, 1, local_position, num_columns))
        else:
            _append_row_conflict(
                key,
                key_columns,
                CONFLICT_ADD_ADD,
                None,
                local_index.rows[local_position],
                incoming_index.rows[incoming_position],
                False,
                True,
                True,
                (None, local_position, incoming_position),
                entries,
                conflicts,
            )
    for incoming_position, key in enumerate(incoming_index.keys):
        token = incoming_index.frozen[incoming_position]
        if token in base_index.by_key or token in local_index.by_key:
            continue
        entries.append(_whole_row_entry(key, 2, incoming_position, num_columns))


def _whole_row_entry(
    key: Tuple[Any, ...], side: int, position: int, num_columns: int
) -> _RowEntry:
    return _RowEntry(
        key=key,
        cells=[("cell", side, position) for _ in range(num_columns)],
        col_conflicts={},
        candidates=tuple(position if index == side else None for index in range(3)),  # type: ignore[arg-type]
    )


def _rows_equal(
    left: Dict[str, Any], right: Dict[str, Any], template: Dict[str, Any]
) -> bool:
    return all(_value_equal(left[name], right[name]) for name in template)


def _value_equal(left: Any, right: Any) -> bool:
    if left is right:
        return True
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, float) and isinstance(right, float):
        if math.isnan(left) and math.isnan(right):
            return True
        return left == right
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not (isinstance(left, Mapping) and isinstance(right, Mapping)):
            return False
        if left.keys() != right.keys():
            return False
        return all(_value_equal(left[name], right[name]) for name in left)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if not (isinstance(left, (list, tuple)) and isinstance(right, (list, tuple))):
            return False
        if len(left) != len(right):
            return False
        return all(_value_equal(x, y) for x, y in zip(left, right))
    try:
        return bool(left == right)
    except Exception:  # pragma: no cover - defensive for exotic scalar types
        return False


def _freeze(value: Any) -> Any:
    if value is None:
        return ("null",)
    if isinstance(value, float) and math.isnan(value):
        return ("nan",)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, Decimal):
        return ("decimal", value)
    if isinstance(value, (int, float, str, bytes)):
        return ("scalar", value)
    if isinstance(value, (datetime, date)):
        return ("datetime", value.isoformat())
    if isinstance(value, Mapping):
        return (
            "map",
            tuple((name, _freeze(item)) for name, item in value.items()),
        )
    if isinstance(value, (list, tuple)):
        return ("seq", tuple(_freeze(item) for item in value))
    return ("object", repr(value))


def _sort_key(key: Tuple[Any, ...]) -> Tuple[Any, ...]:
    return tuple(_value_sort_key(value) for value in key)


def _value_sort_key(value: Any) -> Tuple[Any, ...]:
    if value is None:
        return (0, "")
    if isinstance(value, bool):
        return (1, int(value))
    if isinstance(value, float):
        if math.isnan(value):
            return (2, float("inf"), 0)
        return (2, value, 0)
    if isinstance(value, (int, Decimal)):
        return (2, value, 0)
    if isinstance(value, str):
        return (3, value)
    if isinstance(value, bytes):
        return (4, value)
    if isinstance(value, (datetime, date)):
        return (5, value.isoformat())
    if isinstance(value, Mapping):
        return (6, tuple((name, _value_sort_key(item)) for name, item in value.items()))
    if isinstance(value, (list, tuple)):
        return (7, tuple(_value_sort_key(item) for item in value))
    return (8, repr(value))
