import pandas as pd

from src.core.features.render import render_profile_markdown


def _profile():
    return pd.DataFrame(
        {
            "base_key": ["addr", "landuse", "natural"],
            "cluster_count": [3000, 500, 200],
            "total_count_all": [1_500_000_000, 200_000_000, 50_000_000],
            "representative_medoids": [
                "addr:street|hauptstraße; addr:city|berlin; addr:postcode|10115",
                "landuse|farmland; landuse|forest; landuse|residential",
                "natural|water; natural|wood; natural|wetland",
            ],
        }
    )


# --- output type and content -------------------------------------------


def test_returns_string():
    out = render_profile_markdown(_profile())
    assert isinstance(out, str)


def test_includes_a_header_row():
    out = render_profile_markdown(_profile())
    assert "| base_key |" in out


def test_includes_every_base_key():
    out = render_profile_markdown(_profile())
    assert "| addr |" in out
    assert "| landuse |" in out
    assert "| natural |" in out


def test_includes_representative_medoids_column():
    out = render_profile_markdown(_profile())
    # Pipes inside medoid cells are escaped so the Markdown table stays
    # parseable. The "key|value" pair shows up as "key\|value".
    assert "addr:street\\|hauptstraße" in out
    assert "landuse\\|farmland" in out
    assert "natural\\|water" in out


def test_includes_formatted_counts():
    out = render_profile_markdown(_profile())
    # Big numbers should be human-readable (with thousands separators or
    # at least not be raw floats).
    assert "1,500,000,000" in out or "1500000000" in out
    assert "200,000,000" in out or "200000000" in out


def test_pipe_in_medoid_escaped_or_unambiguous():
    # The pipe character in 'landuse|farmland' is also the table column
    # separator in Markdown. The function must escape or work around it
    # so the rendered table does not break.
    out = render_profile_markdown(_profile())
    lines = [ln for ln in out.splitlines() if ln.startswith("| landuse")]
    assert lines, "expected a landuse row in the table"
    # Even with escaped pipes inside the medoid cell, the row should
    # have exactly 4 top-level columns when parsed on un-escaped pipes.
    # We test this by replacing escaped pipes with a placeholder, then
    # splitting.
    unescaped = lines[0].replace("\\|", "")
    cells = [c.strip() for c in unescaped.split("|")]
    # leading + 4 + trailing = 6 cells.
    assert len(cells) == 6


# --- empty input -------------------------------------------------------


def test_empty_profile_returns_empty_string():
    df = pd.DataFrame(
        columns=["base_key", "cluster_count", "total_count_all", "representative_medoids"]
    )
    out = render_profile_markdown(df)
    assert out == ""
