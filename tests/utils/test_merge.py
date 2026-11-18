from decimal import Decimal

import pyarrow as pa
import pytest
from pytest import raises

from triad.utils.merge import (
    CONFLICT_ADD_ADD,
    CONFLICT_DELETE_MODIFY,
    CONFLICT_VALUE,
    MergeConflictError,
    MergeError,
    MergeKeyError,
    MergeSchemaError,
    merge_arrow_tables,
    merge_pa_tables,
    merge_tables,
    merge_three_way,
    resolve_merge_conflicts,
    three_way_merge,
)
from triad.utils.pyarrow import merge_tables as merge_tables_from_pyarrow


SCHEMA = pa.schema(
    [
        pa.field("id", pa.int32(), nullable=False),
        pa.field("name", pa.string()),
        pa.field("note", pa.string()),
        pa.field("amount", pa.decimal128(6, 2)),
    ]
)


def _table(rows):
    return pa.Table.from_pylist(rows, schema=SCHEMA)


def _row(id, name="name", note="note", amount=Decimal("1.00")):
    return {"id": id, "name": name, "note": note, "amount": amount}


def test_empty_tables():
    empty = _table([])
    result = merge_tables(empty, empty, empty, key="id")
    assert not result.has_conflicts
    assert result.conflicts == []
    assert result.table.num_rows == 0
    assert result.table.schema.equals(SCHEMA, check_metadata=True)
    assert set(result.table.column_names) == {"id", "name", "note", "amount"}


def test_one_sided_and_shared_changes_merge_by_column():
    base = _table([_row(1, name="a", note="n1")])
    local = _table([_row(1, name="A", note="n1")])
    incoming = _table([_row(1, name="a", note="n2")])
    result = merge_tables(base, local, incoming, key="id")
    assert result.conflicts == []
    assert result.table.to_pylist() == [_row(1, name="A", note="n2")]

    incoming = _table([_row(1, name="A", note="n1")])
    result = merge_tables(base, local, incoming, key="id")
    assert result.conflicts == []
    assert result.table.to_pylist() == [_row(1, name="A", note="n1")]


def test_same_column_different_changes_is_conflict():
    base = _table([_row(1, note="n0")])
    local = _table([_row(1, note="local")])
    incoming = _table([_row(1, note="incoming")])
    result = merge_tables(base, local, incoming, key="id")
    assert len(result.conflicts) == 1
    conflict = result.conflicts[0]
    assert conflict.kind == CONFLICT_VALUE
    assert conflict.key == (1,)
    assert conflict.column == "note"
    assert conflict.scope == "column"
    assert conflict.values == ("n0", "local", "incoming")
    assert conflict.existence == (True, True, True)
    assert conflict.key_dict == {"id": 1}
    with raises(MergeConflictError) as e:
        result.table
    assert len(e.value.conflicts) == 1


def test_resolve_choices_and_schema_preservation():
    base = _table([_row(1, name="b", note="n0")])
    local = _table([_row(1, name="l", note="l0")])
    incoming = _table([_row(1, name="i", note="n0")])
    result = merge_tables(base, local, incoming, key="id")
    assert result.conflicts[0].column == "name"
    assert result.resolve({0: "base"}).to_pylist() == [_row(1, name="b", note="l0")]
    assert result.resolve({0: "local"}).to_pylist() == [_row(1, name="l", note="l0")]
    assert result.resolve({0: "incoming"}).to_pylist() == [_row(1, name="i", note="l0")]
    resolved = result.resolve({0: "local"})
    assert resolved.schema.equals(SCHEMA, check_metadata=True)


def test_unsorted_input_is_supported_and_base_order_is_kept():
    base = _table([_row(3), _row(1), _row(2)])
    local = _table([_row(2, name="x"), _row(3), _row(1)])
    incoming = _table([_row(1), _row(3, note="y"), _row(2)])
    result = merge_tables(base, local, incoming, key="id")
    assert result.conflicts == []
    assert [r["id"] for r in result.table.to_pylist()] == [3, 1, 2]
    assert result.table.column("name").to_pylist() == ["name", "name", "x"]
    assert result.table.column("note").to_pylist() == ["y", "note", "note"]


def test_sort_by_key_orders_base_rows_and_additions():
    base = _table([_row(3), _row(1)])
    local = _table([_row(3), _row(1), _row(2, name="added")])
    result = merge_tables(base, local, base, key="id", sort_by_key=True)
    assert result.conflicts == []
    assert [row["id"] for row in result.table.to_pylist()] == [1, 2, 3]


def test_delete_then_unchanged_deletes_and_both_delete_deletes():
    base = _table([_row(1)])
    unchanged = _table([_row(1)])
    result = merge_tables(base, _table([]), unchanged, key="id")
    assert result.conflicts == []
    assert result.table.num_rows == 0
    result = merge_tables(base, _table([]), _table([]), key="id")
    assert result.conflicts == []
    assert result.table.num_rows == 0


@pytest.mark.parametrize("delete_side", ["local", "incoming"])
def test_delete_vs_modify_is_row_conflict(delete_side):
    base = _table([_row(1, name="a")])
    modified = _table([_row(1, name="b")])
    empty = _table([])
    if delete_side == "local":
        local, incoming = empty, modified
    else:
        local, incoming = modified, empty
    result = merge_tables(base, local, incoming, key="id")
    assert len(result.conflicts) == 1
    conflict = result.conflicts[0]
    assert conflict.kind == CONFLICT_DELETE_MODIFY
    assert conflict.column is None
    assert conflict.scope == "row"
    assert conflict.existence == (
        True,
        delete_side != "local",
        delete_side != "incoming",
    )
    assert conflict.deleted_sides == (delete_side,)
    assert conflict.base == _row(1, name="a")
    assert conflict.key == (1,)

    kept = "incoming" if delete_side == "local" else "local"
    assert result.resolve({0: kept}).to_pylist() == [_row(1, name="b")]
    assert result.resolve({0: "delete"}).num_rows == 0
    assert result.resolve({0: delete_side}).num_rows == 0
    assert result.resolve({0: "base"}).to_pylist() == [_row(1, name="a")]


def test_different_same_key_additions_is_row_conflict():
    empty = _table([])
    local = _table([_row(7, name="l")])
    incoming = _table([_row(7, name="i")])
    result = merge_tables(empty, local, incoming, key="id")
    assert len(result.conflicts) == 1
    conflict = result.conflicts[0]
    assert conflict.kind == CONFLICT_ADD_ADD
    assert conflict.column is None
    assert conflict.base is None
    assert conflict.existence == (False, True, True)
    assert conflict.local == _row(7, name="l")
    assert conflict.incoming == _row(7, name="i")
    assert result.resolve({0: "local"}).to_pylist() == [_row(7, name="l")]
    assert result.resolve({0: "incoming"}).to_pylist() == [_row(7, name="i")]
    assert result.resolve({0: "delete"}).num_rows == 0


def test_changed_key_is_seen_as_delete_and_add():
    base = _table([_row(1, name="a"), _row(2, name="b")])
    local = _table([_row(10, name="a"), _row(2, name="b")])
    incoming = _table([_row(1, name="a"), _row(2, name="B")])
    result = merge_tables(base, local, incoming, key="id")
    assert result.conflicts == []
    assert result.table.to_pylist() == [
        _row(2, name="B"),
        _row(10, name="a"),
    ]


def test_identical_one_sided_and_incoming_only_additions():
    empty = _table([])
    both_add = _table([_row(1, name="x")])
    result = merge_tables(empty, both_add, both_add, key="id")
    assert result.conflicts == []
    assert result.table.to_pylist() == [_row(1, name="x")]

    local = _table([_row(1, name="l")])
    incoming = _table([_row(2, name="i")])
    result = merge_tables(empty, local, incoming, key="id")
    assert result.conflicts == []
    assert [r["id"] for r in result.table.to_pylist()] == [1, 2]


def test_null_value_differs_from_missing_row():
    base = _table([_row(1, note="n")])
    # local nulls the note: this is a value change, not a deletion
    local = _table([_row(1, note=None)])
    incoming = _table([_row(1, note="n")])
    result = merge_tables(base, local, incoming, key="id")
    assert result.conflicts == []
    assert result.table.num_rows == 1
    assert result.table.to_pylist() == [_row(1, note=None)]

    # null vs a real value modified by the other side is a column conflict
    incoming = _table([_row(1, note="x")])
    result = merge_tables(base, local, incoming, key="id")
    assert len(result.conflicts) == 1
    assert result.conflicts[0].values == ("n", None, "x")
    assert result.resolve({0: "local"}).to_pylist() == [_row(1, note=None)]


def test_nested_columns_are_atomic_values():
    schema = pa.schema(
        [
            pa.field("id", pa.int32(), nullable=False),
            pa.field("s", pa.struct([("x", pa.int64()), ("y", pa.string())])),
            pa.field("l", pa.list_(pa.int64())),
        ]
    )

    def table(rows):
        return pa.Table.from_pylist(rows, schema=schema)

    def row(id, s, lst):
        return {"id": id, "s": s, "l": lst}

    base = table([row(1, {"x": 1, "y": "a"}, [1, 2])])
    # local edits the struct, incoming edits the list: both merge
    local = table([row(1, {"x": 2, "y": "a"}, [1, 2])])
    incoming = table([row(1, {"x": 1, "y": "a"}, [1, 3])])
    result = merge_tables(base, local, incoming, key="id")
    assert result.conflicts == []
    assert result.table.to_pylist() == [row(1, {"x": 2, "y": "a"}, [1, 3])]

    # conflicting nested edits surface as one column conflict for the column
    incoming = table([row(1, {"x": 3, "y": "b"}, [1, 2])])
    result = merge_tables(base, local, incoming, key="id")
    assert len(result.conflicts) == 1
    conflict = result.conflicts[0]
    assert conflict.column == "s"
    assert conflict.values == (
        {"x": 1, "y": "a"},
        {"x": 2, "y": "a"},
        {"x": 3, "y": "b"},
    )


def test_nested_types_are_preserved():
    schema = pa.schema(
        [
            pa.field("id", pa.int32(), nullable=False, metadata={b"k": b"v"}),
            pa.field("d", pa.dictionary(pa.int32(), pa.string())),
            pa.field("m", pa.map_(pa.string(), pa.int64())),
        ],
        metadata={b"top": b"meta"},
    )
    rows = [
        {
            "id": 1,
            "d": "a",
            "m": [("k", 1)],
        }
    ]
    base = pa.Table.from_pylist(rows, schema=schema)
    local = pa.Table.from_pylist(rows, schema=schema)
    incoming = pa.Table.from_pylist(rows, schema=schema)
    result = merge_tables(base, local, incoming, key="id")
    assert result.conflicts == []
    assert result.table.schema.equals(schema, check_metadata=True)
    assert result.table.schema.field("id").metadata == {b"k": b"v"}


def test_conflict_decision_forms():
    base = _table([_row(1, note="n0")])
    local = _table([_row(1, note="l")])
    incoming = _table([_row(1, note="i")])
    result = merge_tables(base, local, incoming, key="id")
    conflict = result.conflicts[0]
    expected_local = [_row(1, note="l")]
    expected_incoming = [_row(1, note="i")]
    expected_base = [_row(1, note="n0")]
    assert result.resolve({conflict: "local"}).to_pylist() == expected_local
    assert result.resolve({0: "incoming"}).to_pylist() == expected_incoming
    assert result.resolve({((1,), "note"): "base"}).to_pylist() == expected_base
    assert result.resolve({(1, "note"): "local"}).to_pylist() == expected_local
    assert result.resolve(["incoming"]).to_pylist() == expected_incoming
    assert result.resolve(lambda c: "base").to_pylist() == expected_base
    assert result.resolve("incoming").to_pylist() == expected_incoming
    assert result.resolve(default="local").to_pylist() == expected_local
    assert result.resolve_all("base").to_pylist() == expected_base
    assert result.resolve({0: "explicit"}).to_pylist() == [_row(1, note="explicit")]
    assert result.resolve({0: None}).to_pylist() == [_row(1, note=None)]
    assert resolve_merge_conflicts(result, {0: "local"}).to_pylist() == expected_local
    assert (
        result.resolve(lambda c, i: "base" if i == c.index else "local").to_pylist()
        == expected_base
    )


def test_two_argument_callable_gets_conflict_and_index():
    base = _table([_row(1, note="n0"), _row(2, note="n0")])
    local = _table([_row(1, note="l"), _row(2, note="l")])
    incoming = _table([_row(1, note="i"), _row(2, note="i")])
    result = merge_tables(base, local, incoming, key="id")
    assert [c.index for c in result.conflicts] == [0, 1]

    def pick(conflict, index):
        assert conflict.index == index
        return "base" if index == 0 else "incoming"

    resolved = result.resolve(pick)
    assert resolved.to_pylist() == [_row(1, note="n0"), _row(2, note="i")]


def test_string_and_default_decisions_cannot_be_combined():
    base = _table([_row(1, note="n0")])
    local = _table([_row(1, note="l")])
    incoming = _table([_row(1, note="i")])
    result = merge_tables(base, local, incoming, key="id")
    with raises(MergeError):
        result.resolve("local", default="incoming")


def test_column_conflict_cannot_be_deleted():
    base = _table([_row(1, note="n0")])
    local = _table([_row(1, note="l")])
    incoming = _table([_row(1, note="i")])
    result = merge_tables(base, local, incoming, key="id")
    with raises(MergeError):
        result.resolve({0: "delete"})


def test_null_into_non_nullable_column_is_rejected():
    schema = pa.schema(
        [
            pa.field("id", pa.int64(), nullable=False),
            pa.field("v", pa.int64(), nullable=False),
        ]
    )

    def table(rows):
        return pa.Table.from_pylist(rows, schema=schema)

    base = table([{"id": 1, "v": 1}])
    result = merge_tables(
        base, table([{"id": 1, "v": 2}]), table([{"id": 1, "v": 3}]), key="id"
    )
    with raises(MergeError):
        result.resolve({0: None})
    assert result.resolve({0: "local"}).to_pylist() == [{"id": 1, "v": 2}]


def test_explicit_values_keep_nested_and_dictionary_types():
    schema = pa.schema(
        [
            pa.field("id", pa.int64(), nullable=False),
            pa.field("d", pa.dictionary(pa.int32(), pa.string())),
            pa.field("m", pa.map_(pa.string(), pa.int64())),
        ]
    )

    def table(rows):
        return pa.Table.from_pylist(rows, schema=schema)

    base = table([{"id": 1, "d": "a", "m": [("k", 1)]}])
    local = table([{"id": 1, "d": "b", "m": [("k", 1)]}])
    incoming = table([{"id": 1, "d": "c", "m": [("k", 1)]}])
    result = merge_tables(base, local, incoming, key="id")
    resolved = result.resolve({0: "explicit"})
    assert resolved.to_pylist() == [{"id": 1, "d": "explicit", "m": [("k", 1)]}]
    assert resolved.schema.equals(schema, check_metadata=True)

    deleted = merge_tables(base, table([]), incoming, key="id")
    assert deleted.conflicts[0].kind == CONFLICT_DELETE_MODIFY
    explicit = deleted.resolve({0: {"id": 1, "d": "z", "m": [("x", 2)]}})
    assert explicit.to_pylist() == [{"id": 1, "d": "z", "m": [("x", 2)]}]
    assert explicit.schema.equals(schema, check_metadata=True)


def test_unresolved_or_bad_decisions_raise():
    base = _table([_row(1, note="n0")])
    local = _table([_row(1, note="l")])
    incoming = _table([_row(1, note="i")])
    result = merge_tables(base, local, incoming, key="id")
    with raises(MergeConflictError):
        result.resolve()
    with raises(MergeError):
        result.resolve(["local", "incoming"])
    with raises(MergeError):
        result.resolve({5: "local"})
    with raises(MergeError):
        result.resolve({((99,), "note"): "local"})
    with raises(MergeError):
        result.resolve({0: "delete"})
    with raises(MergeError):
        result.resolve(3)


def test_row_level_explicit_decisions():
    base = _table([_row(1, name="a")])
    modified = _table([_row(1, name="b")])
    result = merge_tables(base, _table([]), modified, key="id")
    assert result.resolve({0: _row(1, name="z")}).to_pylist() == [_row(1, name="z")]
    assert result.resolve({0: [1, "z", "note", Decimal("1.00")]}).to_pylist() == [
        _row(1, name="z")
    ]
    assert result.resolve({0: None}).num_rows == 0
    with raises(MergeError):
        result.resolve({0: {"id": 1}})
    with raises(MergeError):
        result.resolve({0: [1, 2]})


def test_empty_schema_and_exotic_types():
    empty_schema = pa.schema([])
    empty = pa.Table.from_arrays([], schema=empty_schema)
    with raises(MergeKeyError):
        merge_tables(empty, empty, empty, key="id")


def test_key_validation():
    empty = _table([])
    base = _table([_row(1)])
    with raises(MergeKeyError):
        merge_tables(base, base, base)
    with raises(MergeKeyError):
        merge_tables(base, base, base, key="missing")
    with raises(MergeKeyError):
        merge_tables(base, base, base, key=["id"], keys="id")
    with raises(MergeKeyError):
        merge_tables(base, base, base, key=[])
    null_key = _table([_row(1)][:0] + [dict(_row(1), id=None)])
    with raises(MergeKeyError):
        merge_tables(null_key, empty, empty, key="id")
    duplicate = _table([_row(1), _row(1)])
    with raises(MergeKeyError):
        merge_tables(duplicate, empty, empty, key="id")


def test_schema_validation():
    base = _table([_row(1)])
    wrong_type = pa.Table.from_pylist(
        [{"id": "1", "name": "", "note": "", "amount": Decimal("1.00")}],
        schema=pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("name", pa.string()),
                pa.field("note", pa.string()),
                pa.field("amount", pa.decimal128(6, 2)),
            ]
        ),
    )
    with raises(MergeSchemaError):
        merge_tables(base, wrong_type, base, key="id")
    wrong_order = pa.schema(
        [
            pa.field("name", pa.string()),
            pa.field("id", pa.int32(), nullable=False),
            pa.field("note", pa.string()),
            pa.field("amount", pa.decimal128(6, 2)),
        ]
    )
    reordered = pa.Table.from_pylist(
        [{"id": 1, "name": "n", "note": "x", "amount": Decimal("1.00")}],
        schema=wrong_order,
    )
    with raises(MergeSchemaError):
        merge_tables(base, reordered, base, key="id")
    duplicate_names = pa.schema(
        [pa.field("id", pa.int32()), pa.field("id", pa.string())]
    )
    duplicated = pa.Table.from_arrays(
        [pa.array([1]), pa.array(["x"])], schema=duplicate_names
    )
    with raises(MergeSchemaError):
        merge_tables(duplicated, duplicated, duplicated, key="id")
    with raises(TypeError):
        merge_tables([], base, base, key="id")


def test_composite_key_and_struct_key():
    schema = pa.schema(
        [
            pa.field("a", pa.string(), nullable=False),
            pa.field("b", pa.int32(), nullable=False),
            pa.field("v", pa.string()),
        ]
    )

    def table(rows):
        return pa.Table.from_pylist(rows, schema=schema)

    base = table([{"a": "x", "b": 1, "v": "b"}])
    local = table([{"a": "x", "b": 1, "v": "l"}])
    incoming = table([{"a": "x", "b": 1, "v": "i"}])
    result = merge_tables(base, local, incoming, keys=["a", "b"])
    assert result.conflicts[0].key == ("x", 1)
    assert result.conflicts[0].key_dict == {"a": "x", "b": 1}
    assert result.conflicts[0].key_value == ("x", 1)

    struct_schema = pa.schema(
        [
            pa.field("k", pa.struct([("a", pa.int64())]), nullable=False),
            pa.field("v", pa.string()),
        ]
    )
    struct_table = pa.Table.from_pylist(
        [{"k": {"a": 1}, "v": "x"}], schema=struct_schema
    )
    result = merge_tables(struct_table, struct_table, struct_table, key="k")
    assert result.table.to_pylist() == [{"k": {"a": 1}, "v": "x"}]


def test_nan_is_treated_as_equal():
    schema = pa.schema(
        [
            pa.field("id", pa.int64(), nullable=False),
            pa.field("f", pa.float64()),
        ]
    )
    nan = float("nan")

    def table(rows):
        return pa.Table.from_pylist(rows, schema=schema)

    base = table([{"id": 1, "f": nan}])
    local = table([{"id": 1, "f": nan}])
    incoming = table([{"id": 1, "f": 2.0}])
    result = merge_tables(base, local, incoming, key="id")
    assert result.conflicts == []
    assert result.table.column("f").to_pylist() == [2.0]


def test_aliases_and_module_exports():
    base = _table([_row(1)])
    for func, kwargs in [
        (merge_tables, {"key": "id"}),
        (merge_arrow_tables, {"key": "id"}),
        (merge_pa_tables, {"key": "id"}),
        (three_way_merge, {"key": "id"}),
        (merge_three_way, {"on": "id"}),
        (merge_tables_from_pyarrow, {"keys": "id"}),
    ]:
        result = func(base, base, base, **kwargs)
        assert result.table.to_pylist() == [_row(1)]


def test_result_delegates_to_resolved_table():
    base = _table([_row(1), _row(2)])
    result = merge_tables(base, base, base, key="id")
    assert result.num_rows == 2
    assert result.column_names == ["id", "name", "note", "amount"]
    assert len(result.to_batches()) == 1
    assert result.slice(0, 1).to_pylist() == [_row(1)]
    base = _table([_row(1)])
    conflict = merge_tables(
        base,
        _table([_row(1, note="l")]),
        _table([_row(1, note="i")]),
        key="id",
    )
    with raises(MergeConflictError):
        conflict.num_rows
