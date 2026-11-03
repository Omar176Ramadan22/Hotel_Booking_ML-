import json
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.data.rules import (
    GUEST_COLUMNS,
    NIGHT_COLUMNS,
    MONTHS,
    positive_total,
    arrival_date,
    arrival_week,
    validate_training_limits,
)
from src.features.columns import feature_columns
from src.regression.models import MODELS

ARTIFACT_DIR = Path("artifacts")
CATEGORY_FIELDS = (
    "Country", "Meal", "MarketSegment", "DistributionChannel",
    "ReservedRoomType", "DepositType", "CustomerType",
)
EXAMPLE_BOOKING = {
    "LeadTime": 30,
    "ArrivalDateYear": 2016,
    "ArrivalDateMonth": "July",
    "ArrivalDateWeekNumber": 29,
    "ArrivalDateDayOfMonth": 15,
    "StaysInWeekendNights": 1,
    "StaysInWeekNights": 3,
    "Adults": 2,
    "Children": 0,
    "Babies": 0,
    "Meal": "BB",
    "Country": "PRT",
    "MarketSegment": "Online TA",
    "DistributionChannel": "TA/TO",
    "IsRepeatedGuest": 0,
    "PreviousCancellations": 0,
    "PreviousBookingsNotCanceled": 0,
    "ReservedRoomType": "A",
    "DepositType": "No Deposit",
    "CustomerType": "Transient",
    "Agent": "9",
    "Company": None,
}


def validate_bookings(bookings):
    if isinstance(bookings, dict):
        bookings = [bookings]
    if (
        not isinstance(bookings, list)
        or not bookings
        or (not all((isinstance(b, dict) for b in bookings)))
    ):
        raise ValueError(
            "Supply one booking object or a non-empty list of booking objects."
        )
    numeric, categorical = feature_columns("regression")
    numeric = [name for name in numeric if name not in {"HasAgent", "HasCompany"}]
    required = set(numeric + categorical) - {"Children", "Country"}
    for number, booking in enumerate(bookings, 1):
        missing = required - booking.keys()
        if missing:
            raise ValueError(
                f"Booking {number}: missing fields: {', '.join(sorted(missing))}"
            )
        if any((isinstance(value, (dict, list)) for value in booking.values())):
            raise ValueError(
                f"Booking {number}: fields must contain individual values."
            )
    data = pd.DataFrame(bookings)
    if "Children" not in data:
        data["Children"] = 0
    data["Children"] = data["Children"].fillna(0)
    for name in numeric:
        values = pd.to_numeric(data[name], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float)).all() or (values < 0).any():
            raise ValueError(f"{name} must contain finite non-negative numbers.")
        if (values % 1 != 0).any():
            raise ValueError(f"{name} must contain whole numbers.")
        data[name] = values.astype(int)
    for name in categorical:
        if name not in data:
            continue
        if any(
            (
                not pd.isna(value) and (not isinstance(value, str))
                for value in data[name]
            )
        ):
            raise ValueError(f"{name} must contain text values.")
        data[name] = data[name].map(
            lambda value: (value.strip().upper() if name in CATEGORY_FIELDS else value.strip())
            if isinstance(value, str) else value
        )
    if not data["IsRepeatedGuest"].isin([0, 1]).all():
        raise ValueError("IsRepeatedGuest must be 0 or 1.")
    if not data["ArrivalDateWeekNumber"].between(1, 53).all():
        raise ValueError("ArrivalDateWeekNumber must be between 1 and 53.")
    if not positive_total(data, GUEST_COLUMNS).all():
        raise ValueError("Each booking must have at least one guest.")
    if not positive_total(data, NIGHT_COLUMNS).all():
        raise ValueError("Each booking must have at least one night.")
    for _, row in data.iterrows():
        day = arrival_date(row)
        expected = arrival_week(day)
        if int(row["ArrivalDateWeekNumber"]) != expected:
            raise ValueError(
                f"ArrivalDateWeekNumber must be {expected} for {day}; H2 weeks start Sunday and week 1 contains January 1."
            )
    return data


def normalize_categories(data, pipeline):
    data = data.copy()
    transformer = pipeline.named_steps["preprocess"].named_steps["encode"]
    columns = next(columns for name, _, columns in transformer.transformers_ if name == "cat")
    encoder = transformer.named_transformers_["cat"].named_steps["encode"]
    for name, seen in zip(columns, encoder.categories_):
        if name not in CATEGORY_FIELDS or name not in data:
            continue
        options = {value.strip().upper(): value for value in seen if isinstance(value, str)}
        for value in data[name]:
            if name == "Country" and pd.isna(value):
                continue
            if value not in options:
                raise ValueError(f"Unknown {name}: {value!r}. Valid options: {', '.join(sorted(options))}")
        # Preserve the fitted encoder's spelling so frozen model weights remain valid.
        data[name] = data[name].map(lambda value: options[value] if not pd.isna(value) else value)
    return data


def predict_bookings(bookings, model_name="ridge", stage="tuned"):
    if model_name not in MODELS or stage not in {"baseline", "tuned"}:
        raise ValueError("Select a supported model and baseline or tuned stage.")
    data = validate_bookings(bookings)
    artifact = ARTIFACT_DIR / f"regression_{model_name}_{stage}.joblib"
    if not artifact.exists():
        raise FileNotFoundError(f"Missing {artifact}. Run train or tune first.")
    bounds_path = artifact.with_suffix(".bounds.json")
    if not bounds_path.exists():
        raise FileNotFoundError(
            f"Missing training bounds: {bounds_path}. Restore the matching bounds file."
        )
    limits = json.loads(bounds_path.read_text(encoding="utf-8"))
    validate_training_limits(data, limits)
    pipeline = joblib.load(artifact)
    data = normalize_categories(data, pipeline)
    predictions = np.asarray(pipeline.predict(data), dtype=float)
    if not np.isfinite(predictions).all():
        raise ValueError("The model returned a non-finite ADR prediction.")
    negative = np.flatnonzero(predictions < 0)
    if len(negative):
        rows = ", ".join(str(int(index) + 1) for index in negative)
        warnings.warn(
            f"Negative ADR prediction for booking(s) {rows} clipped to 0; review these bookings.",
            UserWarning,
            stacklevel=2,
        )
        predictions = np.maximum(predictions, 0)
    return predictions


def prompt_booking():
    booking = {}
    for field, default in EXAMPLE_BOOKING.items():
        value = input(f"{field} [{('' if default is None else default)}]: ").strip()
        booking[field] = value if value else default
    return booking


def run_predict(model_name="ridge", input_path=None, stage="tuned"):
    if input_path is None:
        bookings = prompt_booking()
    else:
        try:
            bookings = json.loads(Path(input_path).read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON input: {exc.msg}") from exc
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter("always", UserWarning)
        predictions = predict_bookings(bookings, model_name, stage)
    output = {
        "model": model_name,
        "stage": stage,
        "predictions": [{"ADR": float(value)} for value in predictions],
    }
    if notices:
        output["warnings"] = [str(notice.message) for notice in notices]
    print(json.dumps(output, indent=2))
    return output
