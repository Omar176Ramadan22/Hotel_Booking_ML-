from src.data.load import load_clean
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

def load_cluster_data(path="data/H2.csv"):
    # load and clean the data the same way as the rest of the team
    df, log = load_clean(path)

    # 4 rows have no value for Children, treat them as 0
    df["Children"] = df["Children"].fillna(0)

    # make the new columns
    df["TotalNights"] = df["StaysInWeekendNights"] + df["StaysInWeekNights"]
    df["TotalGuests"] = df["Adults"] + df["Children"] + df["Babies"]
    df["HasKids"] = ((df["Children"] + df["Babies"]) > 0).astype(int)
    df["HadPreviousCancellation"] = (df["PreviousCancellations"] > 0).astype(int)

    # keep only the columns we chose
    columns = NUMERIC_COLUMNS + FLAG_COLUMNS + CATEGORICAL_COLUMNS
    return df[columns]