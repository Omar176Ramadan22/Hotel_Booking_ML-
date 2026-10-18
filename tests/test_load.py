"""Tests for src/data/load.py.

Most tests use a tiny hand-made table, so they check the cleaning LOGIC and do
not depend on H2.csv. The last test checks the real file and is skipped when
the file is not available.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.data.load import H2_ROWS, clean, load_clean, load_raw, standardize_text

DATA = Path(__file__).resolve().parents[1] / "data" / "H2.csv"


def make_bookings() -> pd.DataFrame:
    """Five fake bookings, each with a different problem.

    Rows 0 and 2 become exact duplicates after stripping spaces,
    row 3 has zero guests and row 4 has an impossible ADR.
    """
    return pd.DataFrame({
        "Meal":     ["BB   ", "HB", "BB   ", "SC", "BB"],
        "Agent":    ["   9", "NULL", "   9", " null ", "7"],
        "Company":  ["  NULL", "  NULL", "  NULL", "40", "NULL"],
        "Adults":   [2, 1, 2, 0, 2],
        "Children": [0, None, 0, 0, 0],
        "Babies":   [0, 0, 0, 0, 0],
        "ADR":      [100.0, 80.0, 100.0, 50.0, 5400.0],
    })


# ---- row-level rules -------------------------------------------------------
def test_strips_padding_spaces():
    out = standardize_text(make_bookings())
    assert out["Meal"].tolist() == ["BB", "HB", "BB", "SC", "BB"]


def test_null_markers_become_missing_in_any_case():
    out = standardize_text(make_bookings())
    assert out["Agent"].isna().tolist() == [False, True, False, True, False]
    assert out["Company"].isna().sum() == 4


def test_numeric_ids_match_text_ids():
    from_form = standardize_text({"Agent": 9, "Company": 40.0})
    from_csv = standardize_text({"Agent": "   9", "Company": "40"})
    assert from_form.loc[0, "Agent"] == from_csv.loc[0, "Agent"] == "9"
    assert from_form.loc[0, "Company"] == "40"


def test_single_new_booking_is_supported():
    booking = {"Meal": "HB  ", "Agent": "NULL", "LeadTime": 30}
    out = standardize_text(booking)
    assert len(out) == 1
    assert out.loc[0, "Meal"] == "HB"
    assert pd.isna(out.loc[0, "Agent"])
    assert out.loc[0, "LeadTime"] == 30  # numbers are left alone


def test_input_is_not_modified():
    df = make_bookings()
    before = df.copy()
    standardize_text(df)
    clean(df)
    pd.testing.assert_frame_equal(df, before)


# ---- dataset-level rules ---------------------------------------------------
def test_clean_removes_duplicates_outliers_and_empty_bookings():
    out, log = clean(make_bookings())
    assert len(out) == 2                      # 5 - 1 duplicate - 1 ADR - 1 zero guests
    assert out["ADR"].max() <= 1000
    assert not out.duplicated().any()
    removed = {row["step"]: row["rows_removed"] for row in log}
    assert removed["drop exact duplicate rows"] == 1
    assert removed["drop ADR > 1000"] == 1
    assert removed["drop bookings with zero guests"] == 1


def test_each_rule_can_be_switched_off():
    out, _ = clean(make_bookings(), drop_duplicates=False,
                   adr_max=None, drop_zero_guests=False)
    assert len(out) == 5


def test_clean_skips_rules_when_columns_are_absent():
    df = make_bookings().drop(columns=["ADR", "Adults", "Children", "Babies"])
    out, _ = clean(df)
    assert len(out) == 4  # only the duplicate is removed


def test_load_raw_reports_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"LeadTime": [1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_raw(path)


# ---- the real H2 file --------------------------------------------------------
@pytest.mark.skipif(not DATA.exists(), reason="data/H2.csv not available")
def test_real_h2_file():
    raw = load_raw(DATA, expected_rows=H2_ROWS)
    assert raw.shape == (H2_ROWS, 31)
    df, _ = load_clean(DATA, expected_rows=H2_ROWS)
    assert df["Agent"].isna().sum() > 0
    assert not df.duplicated().any()
