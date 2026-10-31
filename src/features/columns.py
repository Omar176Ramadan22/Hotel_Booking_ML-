"""Which columns each task is allowed to use (the leakage decisions, in one place).

PREDICTION POINT (both supervised tasks): the moment the booking is created.
Only information that exists at that moment may be a feature. Everything that is
set later in the booking lifecycle is excluded, and the reasons are written next
to each group so they can be copied into the report's Leakage Audit section.

Columns that are not listed in a task's feature lists are DROPPED by the
preprocessing pipeline. That is deliberate: an unknown or new column can never
leak in by accident.
"""

from __future__ import annotations

TARGETS = {"regression": "ADR", "classification": "IsCanceled"}

# --------------------------------------------------------------------------
# Never allowed, for any supervised task
# --------------------------------------------------------------------------
# Outcome of the booking. ReservationStatus ('Canceled', 'Check-Out', 'No-Show')
# is a re-encoding of IsCanceled; ReservationStatusDate is when that outcome
# happened. Both are DIRECT leakage for classification and unavailable at
# booking time for regression.
OUTCOME_COLUMNS = ["IsCanceled", "ReservationStatus", "ReservationStatusDate"]

# Values that are only fixed AFTER the booking is made (room assigned at/near
# check-in, amendments accumulate, waiting list is resolved later).
# AssignedRoomType is also INDIRECT leakage for ADR: the room actually given is
# tied to the price actually charged. Use ReservedRoomType instead.
POST_BOOKING_COLUMNS = ["AssignedRoomType", "BookingChanges", "DaysInWaitingList"]

# Borderline: customers can add parking / special requests after booking, so the
# value at booking time is not what the file stores. Excluded by default;
# `include_borderline=True` lets us run one documented ablation with them.
BORDERLINE_COLUMNS = ["RequiredCarParkingSpaces", "TotalOfSpecialRequests"]

# Raw identifier columns are replaced by HasAgent / HasCompany (and optionally
# AgentTop) by BookingFeatureBuilder, so the raw IDs never reach the encoder.
RAW_ID_COLUMNS = ["Agent", "Company"]

# --------------------------------------------------------------------------
# Allowed features (known when the booking is created)
# --------------------------------------------------------------------------
_COMMON_NUMERIC = [
    "LeadTime", "ArrivalDateYear", "ArrivalDateWeekNumber", "ArrivalDateDayOfMonth",
    "StaysInWeekendNights", "StaysInWeekNights", "Adults", "Children", "Babies",
    "IsRepeatedGuest", "PreviousCancellations", "PreviousBookingsNotCanceled",
    "HasAgent", "HasCompany",
]
_COMMON_CATEGORICAL = [
    "ArrivalDateMonth", "Meal", "Country", "MarketSegment", "DistributionChannel",
    "ReservedRoomType", "DepositType", "CustomerType",
]

# ADR is the target for regression, so it cannot be an input there. For
# cancellation it is the price quoted at booking time, which is known at the
# prediction point. (Documented decision: drop it here if the team disagrees.)
_NUMERIC_BY_TASK = {
    "regression": _COMMON_NUMERIC,
    "classification": _COMMON_NUMERIC + ["ADR"],
}


def feature_columns(
    task: str, top_n_agents: int = 0, include_borderline: bool = False
) -> tuple[list[str], list[str]]:
    """Return (numeric_columns, categorical_columns) allowed for `task`."""
    if task not in TARGETS:
        raise ValueError(f"task must be one of {sorted(TARGETS)}, got '{task}'")
    numeric = list(_NUMERIC_BY_TASK[task])
    categorical = list(_COMMON_CATEGORICAL)
    if top_n_agents > 0:
        categorical.append("AgentTop")
    if include_borderline:
        numeric += BORDERLINE_COLUMNS
    return numeric, categorical


def forbidden_columns(task: str) -> set[str]:
    """Columns that must never appear in the model input for `task`.

    (The borderline columns are not listed: they are excluded by default but may
    be switched on for the ablation experiment.)
    """
    forbidden = set(OUTCOME_COLUMNS) | set(POST_BOOKING_COLUMNS) | set(RAW_ID_COLUMNS)
    forbidden.add(TARGETS[task])
    return forbidden
