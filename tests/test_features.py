"""Tests for src/features (train/test split and preprocessing pipeline).

Most tests use a small fake table built from the real column names, so they
check the LOGIC. The last test uses H2.csv and is skipped when it is absent.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.load import EXPECTED_COLUMNS, load_clean, standardize_text
from src.features.pipeline import (
    LEAKAGE_COLUMNS, build_preprocessor, split_xy, task_features,
)
from src.features.split import cv_folds, split_train_test

DATA = Path(__file__).resolve().parents[1] / "data" / "H2.csv"
TASKS = ["regression", "classification", "clustering"]


def make_bookings(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """`n` random but valid bookings with all 31 columns, about 30% cancelled."""
    rng = np.random.default_rng(seed)
    pick = lambda values: rng.choice(values, n)  # noqa: E731
    canceled = (rng.random(n) < 0.3).astype(int)
    df = pd.DataFrame({
        "IsCanceled": canceled,
        "LeadTime": rng.integers(0, 300, n),
        "ArrivalDateYear": pick([2015, 2016, 2017]),
        "ArrivalDateMonth": pick(["January", "July", "December"]),
        "ArrivalDateWeekNumber": rng.integers(1, 53, n),
        "ArrivalDateDayOfMonth": rng.integers(1, 29, n),
        "StaysInWeekendNights": rng.integers(0, 3, n),
        "StaysInWeekNights": rng.integers(1, 5, n),
        "Adults": rng.integers(1, 3, n),
        "Children": rng.integers(0, 2, n).astype(float),
        "Babies": np.zeros(n, dtype=int),
        "Meal": pick(["BB", "HB", "SC"]),
        "Country": pick(["PRT", "FRA", "DEU"]),
        "MarketSegment": pick(["Online TA", "Groups", "Direct"]),
        "DistributionChannel": pick(["TA/TO", "Direct"]),
        "IsRepeatedGuest": pick([0, 1]),
        "PreviousCancellations": rng.integers(0, 2, n),
        "PreviousBookingsNotCanceled": rng.integers(0, 2, n),
        "ReservedRoomType": pick(["A", "D"]),
        "AssignedRoomType": pick(["A", "D"]),
        "BookingChanges": rng.integers(0, 3, n),
        "DepositType": pick(["No Deposit", "Non Refund"]),
        "Agent": pick(["9", "7", "NULL"]),
        "Company": pick(["NULL", "NULL", "40"]),
        "DaysInWaitingList": rng.integers(0, 5, n),
        "CustomerType": pick(["Transient", "Group"]),
        "ADR": rng.uniform(50, 200, n).round(2),
        "RequiredCarParkingSpaces": pick([0, 1]),
        "TotalOfSpecialRequests": rng.integers(0, 3, n),
        "ReservationStatus": np.where(canceled == 1, "Canceled", "Check-Out"),
        "ReservationStatusDate": "2016-01-01",
    })
    assert list(df.columns) == EXPECTED_COLUMNS
    return standardize_text(df)


def model_inputs(df: pd.DataFrame, task: str) -> pd.DataFrame:
    return df if task == "clustering" else split_xy(df, task)[0]


# ---- split -----------------------------------------------------------------
def test_split_is_80_20_and_disjoint():
    df = make_bookings()
    train, test = split_train_test(df)
    assert len(train) == 160 and len(test) == 40
    assert set(train.index).isdisjoint(test.index)


def test_split_keeps_cancellation_rate():
    df = make_bookings(1000)
    train, test = split_train_test(df)
    assert abs(train["IsCanceled"].mean() - test["IsCanceled"].mean()) < 0.01


def test_split_is_reproducible():
    df = make_bookings()
    a, _ = split_train_test(df)
    b, _ = split_train_test(df)
    c, _ = split_train_test(df, random_state=7)
    assert a.index.equals(b.index)
    assert not a.index.equals(c.index)


def test_cv_folds_match_task():
    assert type(cv_folds("classification")).__name__ == "StratifiedKFold"
    assert type(cv_folds("regression")).__name__ == "KFold"
    with pytest.raises(ValueError):
        cv_folds("clustering")


# ---- leakage ---------------------------------------------------------------
@pytest.mark.parametrize("task", TASKS)
def test_no_leaky_column_or_target_is_used(task):
    df = make_bookings()
    prep = build_preprocessor(task).fit(model_inputs(df, task))
    out_names = list(prep.get_feature_names_out())
    forbidden = LEAKAGE_COLUMNS + ["IsCanceled"] + (["ADR"] if task == "regression" else [])
    for col in forbidden:
        assert not any(name == col or name.startswith(col + "_") for name in out_names), col


def test_adr_is_input_only_where_allowed():
    assert "ADR" not in task_features("regression")["numeric"]
    assert "ADR" in task_features("classification")["numeric"]


def test_output_does_not_change_when_leaky_columns_change():
    """Proof that the pipeline really ignores the leaky columns."""
    df = make_bookings()
    prep = build_preprocessor("classification").fit(split_xy(df, "classification")[0])
    flipped = df.copy()
    flipped["ReservationStatus"] = "Check-Out"
    flipped["AssignedRoomType"] = "K"
    flipped["BookingChanges"] = 99
    a = prep.transform(split_xy(df, "classification")[0])
    b = prep.transform(split_xy(flipped, "classification")[0])
    np.testing.assert_array_equal(a, b)


# ---- fit on train only -----------------------------------------------------
def test_scaler_learns_from_training_rows_only():
    df = make_bookings()
    train, test = split_train_test(df)
    prep = build_preprocessor("classification").fit(split_xy(train, "classification")[0])
    scaler = prep.named_steps["columns"].named_transformers_["numeric"].named_steps["scale"]
    lead_idx = task_features("classification")["numeric"].index("LeadTime")
    assert scaler.mean_[lead_idx] == pytest.approx(train["LeadTime"].mean())
    assert scaler.mean_[lead_idx] != pytest.approx(df["LeadTime"].mean())


# ---- missing values and new bookings ---------------------------------------
def test_missing_agent_becomes_none_category():
    df = make_bookings()
    prep = build_preprocessor("classification", min_category_count=1)
    out = pd.DataFrame(
        prep.fit_transform(split_xy(df, "classification")[0]),
        columns=prep.get_feature_names_out(),
    )
    no_agent = df["Agent"].isna().to_numpy()
    assert (out.loc[no_agent, "Agent_none"] == 1).all()
    assert (out.loc[~no_agent, "Agent_none"] == 0).all()


def test_single_new_booking_with_unseen_values():
    df = make_bookings()
    prep = build_preprocessor("classification").fit(split_xy(df, "classification")[0])
    booking = df.iloc[0].to_dict()
    booking.update({"Country": "XYZ", "Agent": 12345, "Meal": "FB", "Children": None})
    out = prep.transform(split_xy(standardize_text(booking), "classification")[0])
    assert out.shape == (1, len(prep.get_feature_names_out()))
    assert not np.isnan(out).any()


def test_unknown_task_is_rejected():
    with pytest.raises(ValueError):
        build_preprocessor("forecasting")


# ---- real file -------------------------------------------------------------
@pytest.mark.skipif(not DATA.exists(), reason="H2.csv not available")
def test_real_file_split_and_pipeline():
    df, _ = load_clean(DATA)
    train, test = split_train_test(df)
    assert (len(train), len(test)) == (42_432, 10_608)
    assert train["IsCanceled"].mean() == pytest.approx(test["IsCanceled"].mean(), abs=0.001)
    for task in TASKS:
        prep = build_preprocessor(task).fit(model_inputs(train, task))
        out = prep.transform(model_inputs(test, task))
        assert out.shape[0] == len(test)
        assert not np.isnan(out).any()
