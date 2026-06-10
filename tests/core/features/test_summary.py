import pandas as pd
import pytest

from src.core.features.summary import GlobalSummary, SummaryBuilder


class FakeDB:
    def __init__(self, frames):
        self._frames = iter(frames)
        self.calls = []

    def execute_query(self, query, params=None):
        self.calls.append((query, params))
        return next(self._frames)


# --- GlobalSummary --------------------------------------------------------


def test_global_summary_to_dataframe_layout():
    summary = GlobalSummary(
        total_distinct_keys=110_706,
        total_key_occurrences=3_892_388_715,
        total_distinct_tags=192_821_586,
        total_tag_occurrences=3_892_388_715,
    )
    df = summary.to_dataframe()
    assert list(df.columns) == ["metric", "value"]
    assert list(zip(df["metric"], df["value"])) == [
        ("total_distinct_keys", 110_706),
        ("total_key_occurrences", 3_892_388_715),
        ("total_distinct_tags", 192_821_586),
        ("total_tag_occurrences", 3_892_388_715),
    ]


def test_global_summary_to_dict():
    summary = GlobalSummary(1, 2, 3, 4)
    assert summary.to_dict() == {
        "total_distinct_keys": 1,
        "total_key_occurrences": 2,
        "total_distinct_tags": 3,
        "total_tag_occurrences": 4,
    }


# --- SummaryBuilder -------------------------------------------------------


def test_summary_builder_reads_both_aggregates():
    keys_df = pd.DataFrame(
        {"total_distinct_keys": [3], "total_key_occurrences": [60]}
    )
    tags_df = pd.DataFrame(
        {"total_distinct_tags": [5], "total_tag_occurrences": [200]}
    )
    db = FakeDB([keys_df, tags_df])

    summary = SummaryBuilder(db).build()

    assert summary == GlobalSummary(
        total_distinct_keys=3,
        total_key_occurrences=60,
        total_distinct_tags=5,
        total_tag_occurrences=200,
    )


def test_summary_builder_calls_db_with_paramless_queries():
    db = FakeDB(
        [
            pd.DataFrame({"total_distinct_keys": [1], "total_key_occurrences": [1]}),
            pd.DataFrame({"total_distinct_tags": [1], "total_tag_occurrences": [1]}),
        ]
    )

    SummaryBuilder(db).build()

    # Both calls carry params=() (no interpolation possible).
    assert all(params == () for _, params in db.calls)
    # And they were the two aggregate queries.
    sqls = [sql for sql, _ in db.calls]
    assert any("FROM keys" in s for s in sqls)
    assert any("FROM tags" in s for s in sqls)


def test_summary_builder_raises_on_empty_result():
    db = FakeDB(
        [
            pd.DataFrame(columns=["total_distinct_keys", "total_key_occurrences"]),
            pd.DataFrame(columns=["total_distinct_tags", "total_tag_occurrences"]),
        ]
    )
    with pytest.raises(RuntimeError):
        SummaryBuilder(db).build()


def test_summary_builder_treats_null_sum_as_zero():
    # SUM() over an empty table returns NULL -> NaN. The builder must coerce
    # that to 0 so partial / freshly-imported taginfo DBs are not a crash.
    keys_df = pd.DataFrame(
        {"total_distinct_keys": [0], "total_key_occurrences": [float("nan")]}
    )
    tags_df = pd.DataFrame(
        {"total_distinct_tags": [0], "total_tag_occurrences": [float("nan")]}
    )
    db = FakeDB([keys_df, tags_df])

    summary = SummaryBuilder(db).build()

    assert summary.total_distinct_keys == 0
    assert summary.total_key_occurrences == 0
    assert summary.total_distinct_tags == 0
    assert summary.total_tag_occurrences == 0
