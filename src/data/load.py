from __future__ import annotations
from pathlib import Path
import pandas as pd
from src.data.rules import GUEST_COLUMNS, NIGHT_COLUMNS, positive_total

EXPECTED_COLUMNS = [
    "IsCanceled",
    "LeadTime",
    "ArrivalDateYear",
    "ArrivalDateMonth",
    "ArrivalDateWeekNumber",
    "ArrivalDateDayOfMonth",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
    "Children",
    "Babies",
    "Meal",
    "Country",
    "MarketSegment",
    "DistributionChannel",
    "IsRepeatedGuest",
    "PreviousCancellations",
    "PreviousBookingsNotCanceled",
    "ReservedRoomType",
    "AssignedRoomType",
    "BookingChanges",
    "DepositType",
    "Agent",
    "Company",
    "DaysInWaitingList",
    "CustomerType",
    "ADR",
    "RequiredCarParkingSpaces",
    "TotalOfSpecialRequests",
    "ReservationStatus",
    "ReservationStatusDate",
]
H2_ROWS = 79330
ID_COLUMNS = ["Agent", "Company"]
MISSING_TOKENS = {"null", "none", "nan", "na", "n/a", ""}


def standardize_text(data: pd.DataFrame | dict | list[dict]) -> pd.DataFrame:
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        df = data.copy()
    # Normalize ID spellings such as 9 and 9.0.
    for col in ID_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map(_id_to_text)
    text_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in text_cols:
        df[col] = df[col].map(_clean_text)
    # Keep identifier dtypes stable even when a whole column is missing.
    for col in ID_COLUMNS:
        if col in df:
            df[col] = df[col].astype(object).where(df[col].notna(), pd.NA)
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


def load_raw(
    path: str | Path,
    required_columns: list[str] | None = None,
    expected_rows: int | None = None,
) -> pd.DataFrame:
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
    adr_min: float | None = 0,
) -> tuple[pd.DataFrame, list[dict]]:
    log: list[dict] = []

    def record(step: str, before: int, after: int) -> None:
        log.append(
            {
                "step": step,
                "rows_before": before,
                "rows_after": after,
                "rows_removed": before - after,
            }
        )

    # Cleaning uses fixed rules, not fitted statistics.
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
    if adr_min is not None and "ADR" in df.columns:
        n = len(df)
        df = df[df["ADR"].isna() | (df["ADR"] >= adr_min)]
        record(f"drop ADR < {adr_min}", n, len(df))
    present_guests = [c for c in GUEST_COLUMNS if c in df.columns]
    if drop_zero_guests and present_guests:
        n = len(df)
        df = df[positive_total(df, GUEST_COLUMNS)]
        record("drop bookings with zero guests", n, len(df))
    present_nights = [c for c in NIGHT_COLUMNS if c in df.columns]
    if drop_zero_nights and present_nights:
        n = len(df)
        df = df[positive_total(df, NIGHT_COLUMNS)]
        record("drop bookings with zero nights", n, len(df))
    return (df.reset_index(drop=True), log)


def load_clean(
    path: str | Path, expected_rows: int | None = None, **clean_options
) -> tuple[pd.DataFrame, list[dict]]:
    return clean(load_raw(path, expected_rows=expected_rows), **clean_options)
