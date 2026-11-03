import json
from datetime import date
from pathlib import Path

import pandas as pd

GUEST_COLUMNS = ["Adults", "Children", "Babies"]
NIGHT_COLUMNS = ["StaysInWeekendNights", "StaysInWeekNights"]
MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


def positive_total(data, columns):
    present = [column for column in columns if column in data.columns]
    if not present:
        return pd.Series(True, index=data.index)
    return data[present].fillna(0).sum(axis=1) > 0


def arrival_date(row):
    if row["ArrivalDateMonth"] not in MONTHS:
        raise ValueError("ArrivalDateMonth must be a full English month name.")
    try:
        return date(
            int(row["ArrivalDateYear"]),
            MONTHS.index(row["ArrivalDateMonth"]) + 1,
            int(row["ArrivalDateDayOfMonth"]),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("The arrival date is invalid.") from exc


def arrival_week(day):
    # H2 weeks start on Sunday and week one includes January 1.
    offset = (date(day.year, 1, 1).weekday() + 1) % 7
    return ((day - date(day.year, 1, 1)).days + offset) // 7 + 1


def write_training_limits(artifact, data, numeric_columns):
    numeric = {}
    for column in numeric_columns:
        if column in data:
            values = pd.to_numeric(data[column], errors="coerce").dropna()
            if len(values):
                numeric[column] = {
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
    dates = [arrival_date(row) for _, row in data.iterrows()]
    totals = {}
    for name, columns in (("guests", GUEST_COLUMNS), ("nights", NIGHT_COLUMNS)):
        values = data[columns].fillna(0).sum(axis=1)
        totals[name] = {"min": float(values.min()), "max": float(values.max())}
    limits = {
        "training_rows": len(data),
        "numeric": numeric,
        "totals": totals,
        "arrival_date": {"min": min(dates).isoformat(), "max": max(dates).isoformat()},
    }
    path = Path(artifact).with_suffix(".bounds.json")
    path.write_text(json.dumps(limits, indent=2) + "\n", encoding="utf-8")
    return limits


def validate_training_limits(data, limits):
    for column, bounds in limits["numeric"].items():
        if column not in data:
            continue
        outside = (data[column] < bounds["min"]) | (data[column] > bounds["max"])
        if outside.any():
            raise ValueError(
                f"{column} is outside the training range [{bounds['min']:g}, {bounds['max']:g}]."
            )
    for name, columns in (("guests", GUEST_COLUMNS), ("nights", NIGHT_COLUMNS)):
        values = data[columns].fillna(0).sum(axis=1)
        bounds = limits["totals"][name]
        if ((values < bounds["min"]) | (values > bounds["max"])).any():
            raise ValueError(
                f"Total {name} is outside the training range [{bounds['min']:g}, {bounds['max']:g}]."
            )
    minimum = date.fromisoformat(limits["arrival_date"]["min"])
    maximum = date.fromisoformat(limits["arrival_date"]["max"])
    for _, row in data.iterrows():
        if not minimum <= arrival_date(row) <= maximum:
            raise ValueError(
                f"Arrival date is outside the training range [{minimum}, {maximum}]."
            )
