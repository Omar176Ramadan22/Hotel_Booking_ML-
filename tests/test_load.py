from pathlib import Path

from src.data.load import load_clean, load_raw

DATA = Path(__file__).resolve().parents[1] / "data" / "H2.csv"


def test_raw_shape_matches_spec():
    assert load_raw(DATA).shape == (79_330, 31)


def test_clean_removes_padding_null_and_duplicates():
    df, _ = load_clean(DATA)
    assert df["Meal"].str.contains(" ").sum() == 0
    assert (df["Agent"] == "NULL").sum() == 0
    assert df["Agent"].isna().sum() > 0
    assert df.duplicated().sum() == 0
    assert df["ADR"].max() <= 1000
