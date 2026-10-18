"""Load and clean the City Hotel (H2) dataset.

Everything here is *task-independent* cleaning that is safe to do before the
train/test split, because each step only looks at one row at a time (or at exact
copies of a row) and never learns statistics from the data:

1. Strip padding whitespace from text columns   ('HB       ' -> 'HB').
2. Turn the text "NULL" into a real missing value (Agent, Company).
3. Drop exact duplicate rows                       (prevents train/test overlap).
4. Drop invalid records                            (ADR outlier, zero-guest bookings).

Things that DO learn from the data (imputation, encoding, scaling) are NOT done
here. They belong in the preprocessing pipeline and are fitted on training data only.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# The 31 source labels exactly as in Section 4 of the project specification.
EXPECTED_COLUMNS = [
    "IsCanceled", "LeadTime", "ArrivalDateYear", "ArrivalDateMonth",
    "ArrivalDateWeekNumber", "ArrivalDateDayOfMonth", "StaysInWeekendNights",
    "StaysInWeekNights", "Adults", "Children", "Babies", "Meal", "Country",
    "MarketSegment", "DistributionChannel", "IsRepeatedGuest",
    "PreviousCancellations", "PreviousBookingsNotCanceled", "ReservedRoomType",
    "AssignedRoomType", "BookingChanges", "DepositType", "Agent", "Company",
    "DaysInWaitingList", "CustomerType", "ADR", "RequiredCarParkingSpaces",
    "TotalOfSpecialRequests", "ReservationStatus", "ReservationStatusDate",
]
EXPECTED_ROWS = 79_330

# Identifier columns: numbers that are names, not quantities. Keep them as text.
ID_COLUMNS = ["Agent", "Company"]
MISSING_TOKENS = ["NULL", ""]


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read H2.csv exactly as published and verify its schema."""
    df = pd.read_csv(path, dtype={c: str for c in ID_COLUMNS})
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"H2.csv is missing columns: {sorted(missing_cols)}")
    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS:,} rows, found {len(df):,}")
    return df[EXPECTED_COLUMNS]


def standardize_text(df: pd.DataFrame) -> pd.DataFrame:
    """Strip padding spaces and convert the 'NULL' text marker to NaN."""
    df = df.copy()
    text_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in text_cols:
        df[col] = df[col].str.strip().replace(MISSING_TOKENS, pd.NA)
    return df


def clean(
    df: pd.DataFrame,
    drop_duplicates: bool = True,
    adr_max: float | None = 1000,
    drop_zero_guests: bool = True,
) -> tuple[pd.DataFrame, list[dict]]:
    """Apply the cleaning rules and return (clean_df, log of what was removed)."""
    log: list[dict] = []

    def record(step: str, before: int, after: int) -> None:
        log.append({"step": step, "rows_before": before, "rows_after": after,
                    "rows_removed": before - after})

    n = len(df)
    df = standardize_text(df)
    record("strip whitespace + NULL -> NaN", n, len(df))

    if drop_duplicates:
        n = len(df)
        df = df.drop_duplicates()
        record("drop exact duplicate rows", n, len(df))

    if adr_max is not None:
        n = len(df)
        df = df[df["ADR"] <= adr_max]
        record(f"drop ADR > {adr_max}", n, len(df))

    if drop_zero_guests:
        n = len(df)
        guests = df[["Adults", "Children", "Babies"]].fillna(0).sum(axis=1)
        df = df[guests > 0]
        record("drop bookings with zero guests", n, len(df))

    return df.reset_index(drop=True), log


def load_clean(path: str | Path, **clean_options) -> tuple[pd.DataFrame, list[dict]]:
    """One call used by every task: read, verify, clean."""
    return clean(load_raw(path), **clean_options)
