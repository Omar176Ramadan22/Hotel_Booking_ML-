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

# Number columns (will be scaled)
NUMERIC_COLUMNS = [
    "LeadTime",                # days between booking and arrival
    "TotalNights",             # weekend + week nights
    "TotalGuests",             # adults + children + babies
    "ADR",                     # price per night
    "TotalOfSpecialRequests",
]

# 0/1 columns (we leave them as 0/1, no scaling)
FLAG_COLUMNS = [
    "IsRepeatedGuest",
    "HasKids",                 # 1 if children or babies > 0
    "HadPreviousCancellation", # 1 if PreviousCancellations > 0
]

# Text columns (will be one-hot encoded)
CATEGORICAL_COLUMNS = [
    "ArrivalDateMonth",
    "Meal",
    "MarketSegment",
    "DistributionChannel",
    "CustomerType",
    "ReservedRoomType",
]