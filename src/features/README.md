# Features

* `columns.py` - **the leakage decisions**: which columns each task may use and why
  (prediction point = the moment the booking is created). Anything not listed is
  dropped by the pipeline, so a new column can never leak in by accident.
* `booking_features.py` - `BookingFeatureBuilder`, a sklearn transformer that runs
  inside the pipeline: creates `HasAgent` / `HasCompany` *before* dropping the raw
  IDs, optional `AgentTop`, and fixed-rule fills (`Children` -> 0, `Country` -> `Unknown`).

Excluded for both tasks: `IsCanceled`/`ReservationStatus`/`ReservationStatusDate`
(outcome), `AssignedRoomType`, `BookingChanges`, `DaysInWaitingList` (set after
booking). Borderline (off by default): `RequiredCarParkingSpaces`, `TotalOfSpecialRequests`.
