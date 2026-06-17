"""Count rows Pipeline B (standardize-first, then filter) produces.

Pipeline B:
  1. Group raw rows by their standardized (key, value) form.
  2. Sum count_all within each group.
  3. Keep groups whose merged count >= 500.

For comparison, Pipeline A (filter-first) yields 224,123 rows.

The standardization is pushed down into SQLite via a GROUP BY
expression. The expression must mirror the logic in
src/core/features/standardize.py::_normalize_column:
  - LOWER(TRIM(col)) when not empty, else 'none'
"""
import sqlite3

DB = "/Volumes/Seagate M3/taginfo.sqlite"
PIPELINE_A_ROWS = 224_123
PIPELINE_A_OCCURRENCES = 3_350_015_993
THRESHOLD = 500

# Mirror of standardize._normalize_column in SQL.
# Empty-after-trim strings become the missing-value token 'none'.
# Then key|value is joined with the '|' delimiter.
STANDARDIZED_FEATURE_SQL = (
    "CASE WHEN LOWER(TRIM(key))   = '' THEN 'none' "
    "ELSE LOWER(TRIM(key)) END"
    " || '|' || "
    "CASE WHEN LOWER(TRIM(value)) = '' THEN 'none' "
    "ELSE LOWER(TRIM(value)) END"
)


def count_pipeline_b_rows() -> tuple[int, int]:
    """Return (distinct_rows, total_occurrences) Pipeline B produces.

    The inner subquery is the same as before but selects SUM(count_all)
    so the outer query can sum the merged counts of the surviving groups.
    """
    with sqlite3.connect(DB) as conn:
        row = conn.execute(
            f"SELECT COUNT(*), COALESCE(SUM(s), 0) FROM ("
            f"  SELECT SUM(count_all) AS s FROM tags "
            f"  GROUP BY {STANDARDIZED_FEATURE_SQL} "
            f"  HAVING SUM(count_all) >= ?"
            f")",
            (THRESHOLD,),
        ).fetchone()
    return int(row[0]), int(row[1])


def main() -> None:
    n_rows, n_occ = count_pipeline_b_rows()
    delta_rows = n_rows - PIPELINE_A_ROWS
    delta_occ = n_occ - PIPELINE_A_OCCURRENCES
    print(f"Pipeline A rows:                {PIPELINE_A_ROWS:,}")
    print(f"Pipeline B rows:                {n_rows:,}")
    print(f"Delta rows:                     {delta_rows:+,}")
    print(f"Pipeline A occurrences:         {PIPELINE_A_OCCURRENCES:,}")
    print(f"Pipeline B occurrences:         {n_occ:,}")
    print(f"Delta occurrences:              {delta_occ:+,}")


if __name__ == "__main__":
    main()
