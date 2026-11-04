from __future__ import annotations

TARGETS = {"regression": "ADR", "classification": "IsCanceled"}
# Booking outcomes are unavailable when a new booking is priced.
OUTCOME_COLUMNS = ["IsCanceled", "ReservationStatus", "ReservationStatusDate"]
# These fields can change later; assigned rooms may reveal the final price.
POST_BOOKING_COLUMNS = ["AssignedRoomType", "BookingChanges", "DaysInWaitingList"]
# Parking and requests may be added later; exclude them by default.
BORDERLINE_COLUMNS = ["RequiredCarParkingSpaces", "TotalOfSpecialRequests"]
# Replace raw identifiers with presence flags and optional top-agent categories.
RAW_ID_COLUMNS = ["Agent", "Company"]
_COMMON_NUMERIC = [
    "LeadTime",
    "ArrivalDateYear",
    "ArrivalDateWeekNumber",
    "ArrivalDateDayOfMonth",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
    "Children",
    "Babies",
    "IsRepeatedGuest",
    "PreviousCancellations",
    "PreviousBookingsNotCanceled",
    "HasAgent",
    "HasCompany",
]
_COMMON_CATEGORICAL = [
    "ArrivalDateMonth",
    "Meal",
    "Country",
    "MarketSegment",
    "DistributionChannel",
    "ReservedRoomType",
    "DepositType",
    "CustomerType",
]
# ADR is a regression target, but a quoted price can be a classification input.
_NUMERIC_BY_TASK = {
    "regression": _COMMON_NUMERIC,
    "classification": _COMMON_NUMERIC + ["ADR"],
}


def feature_columns(
    task: str, top_n_agents: int = 0, include_borderline: bool = False
) -> tuple[list[str], list[str]]:
    if task not in TARGETS:
        raise ValueError(f"task must be one of {sorted(TARGETS)}, got '{task}'")
    numeric = list(_NUMERIC_BY_TASK[task])
    categorical = list(_COMMON_CATEGORICAL)
    if top_n_agents > 0:
        categorical.append("AgentTop")
    if include_borderline:
        numeric += BORDERLINE_COLUMNS
    return (numeric, categorical)


def forbidden_columns(task: str) -> set[str]:
    forbidden = set(OUTCOME_COLUMNS) | set(POST_BOOKING_COLUMNS) | set(RAW_ID_COLUMNS)
    forbidden.add(TARGETS[task])
    return forbidden
