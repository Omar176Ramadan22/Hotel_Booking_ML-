"""Load and clean hotel booking data.

The module has two kinds of rules:

ROW-LEVEL rules -> `standardize_text`
    Look at one booking at a time. They must run on the training data AND on
    every new booking at prediction time, so the model always sees values in
    the same format ('HB       ' -> 'HB', 'NULL' -> missing, Agent 9 -> '9').

DATASET-LEVEL rules -> `load_raw`, `clean`
    Only make sense for a whole training file: checking the file's shape,
    removing duplicate rows, removing invalid records. They never run on a
    single new booking.

None of these rules learn anything from the data (no means, counts or
scalers), so they are safe to apply before the train/test split. Anything
that learns from data belongs in the preprocessing pipeline, fitted on the
training set only.
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
H2_ROWS = 79_330  # size of the published City Hotel file

# Identifier columns: numbers that are names, not quantities. Keep them as text.
ID_COLUMNS = ["Agent", "Company"]
# Text markers that mean "no value" (compared case-insensitively after stripping).
MISSING_TOKENS = {"null", "none", "nan", "na", "n/a", ""}
GUEST_COLUMNS = ["Adults", "Children", "Babies"]
NIGHT_COLUMNS = ["StaysInWeekendNights", "StaysInWeekNights"]
# Columns where the category 'Undefined' is a missing value, not a real group.
UNDEFINED_COLUMNS = ["MarketSegment", "DistributionChannel"]


# --------------------------------------------------------------------------
# Row-level rules: training data AND new bookings
# --------------------------------------------------------------------------
def standardize_text(data: pd.DataFrame | dict | list[dict]) -> pd.DataFrame:
    """Put text values in one consistent format.

    Accepts a DataFrame, one booking as a dict, or a list of dicts, so the same
    function serves the training file and the prediction interface. Columns
    that are not present are simply skipped.
    """
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        df = data.copy()

    # IDs may arrive as numbers (9, 9.0) from a form or as text ('  9') from
    # the CSV; turn both into the same text value.
    for col in ID_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map(_id_to_text)

    text_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in text_cols:
        df[col] = df[col].map(_clean_text)

    for col in UNDEFINED_COLUMNS:
        if col in df.columns:
            df[col] = df[col].mask(df[col] == "Undefined", pd.NA)
    return df


def _clean_text(value):
    if not isinstance(value, str):
        return value
    value = value.strip()
    return pd.NA if value.lower() in MISSING_TOKENS else value


def _id_to_text(value):
    if pd.isna(value):
        return pd.NA
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return _clean_text(str(value))


# --------------------------------------------------------------------------
# Dataset-level rules: training file only
# --------------------------------------------------------------------------
def load_raw(
    path: str | Path,
    required_columns: list[str] | None = None,
    expected_rows: int | None = None,
) -> pd.DataFrame:
    """Read a booking CSV and check that it has the columns we rely on.

    `expected_rows` is optional: pass `H2_ROWS` to verify the official file;
    leave it as None to load any other file with the same schema.
    """
    required = EXPECTED_COLUMNS if required_columns is None else required_columns
    df = pd.read_csv(path, dtype={c: str for c in ID_COLUMNS})

    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"File is missing columns: {missing_cols}")
    if expected_rows is not None and len(df) != expected_rows:
        raise ValueError(f"Expected {expected_rows:,} rows, found {len(df):,}")
    return df[required]


def clean(
    df: pd.DataFrame,
    drop_duplicates: bool = True,
    adr_max: float | None = 1000,
    drop_zero_guests: bool = True,
    drop_zero_nights: bool = True,
    babies_max: int | None = 3,
) -> tuple[pd.DataFrame, list[dict]]:
    """Apply the dataset-level rules and return (clean_df, log of removals).

    Each rule is skipped when the columns it needs are absent, so the function
    works on any subset of the schema. The input DataFrame is not modified.
    """
    log: list[dict] = []

    def record(step: str, before: int, after: int) -> None:
        log.append({"step": step, "rows_before": before, "rows_after": after,
                    "rows_removed": before - after})

    n = len(df)
    df = standardize_text(df)
    record("strip whitespace + missing markers -> NaN", n, len(df))

    if drop_duplicates:
        n = len(df)
        df = df.drop_duplicates()
        record("drop exact duplicate rows", n, len(df))

    if adr_max is not None and "ADR" in df.columns:
        n = len(df)
        df = df[df["ADR"].isna() | (df["ADR"] <= adr_max)]
        record(f"drop ADR > {adr_max}", n, len(df))

    present_guests = [c for c in GUEST_COLUMNS if c in df.columns]
    if drop_zero_guests and present_guests:
        n = len(df)
        guests = df[present_guests].fillna(0).sum(axis=1)
        df = df[guests > 0]
        record("drop bookings with zero guests", n, len(df))

    # ADR is revenue divided by nights, so a stay of zero nights has no real
    # daily rate (all of them have ADR = 0 in H2).
    present_nights = [c for c in NIGHT_COLUMNS if c in df.columns]
    if drop_zero_nights and present_nights:
        n = len(df)
        nights = df[present_nights].fillna(0).sum(axis=1)
        df = df[nights > 0]
        record("drop bookings with zero nights", n, len(df))

    if babies_max is not None and "Babies" in df.columns:
        n = len(df)
        df = df[df["Babies"].isna() | (df["Babies"] <= babies_max)]
        record(f"drop Babies > {babies_max}", n, len(df))

    return df.reset_index(drop=True), log


def load_clean(
    path: str | Path,
    expected_rows: int | None = None,
    **clean_options,
) -> tuple[pd.DataFrame, list[dict]]:
    """Read, check and clean a training file in one call."""
    return clean(load_raw(path, expected_rows=expected_rows), **clean_options)
