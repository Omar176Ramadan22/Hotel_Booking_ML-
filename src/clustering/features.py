# features.py
# Which columns we use for clustering.
# Clustering has no target, so we don't split into train/test.

# Columns we remove before clustering
EXCLUDED_COLUMNS = [
    "IsCanceled",             # the label itself
    "ReservationStatus",      # same info as IsCanceled
    "ReservationStatusDate",  # date of the outcome
    "AssignedRoomType",       # decided at check-in
    "BookingChanges",         # changes happen after booking
    "DaysInWaitingList",      # known after booking
    "DepositType",            # Non Refund almost always cancels
    "Agent",                  # ID number, not a real value
    "Company",                # ID number, mostly empty
    "Country",                # too many categories
]