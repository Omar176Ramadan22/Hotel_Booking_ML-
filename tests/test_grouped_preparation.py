"""Model-free checks of the shared regression data foundation."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.model_selection import train_test_split

from src.data.groups import regression_group_ids, regression_group_keys
from src.data.load import load_clean, standardize_text
from src.data.preparation import load_config, prepare_regression_data
from src.data.split import grouped_cv_folds, make_split
from src.features.columns import feature_columns
from test_preprocessing import make_bookings

ROOT = Path(__file__).resolve().parents[1]


def test_groups_ignore_target_outcomes_ids_and_borderline():
    df = pd.concat([make_bookings().iloc[[0]]] * 3, ignore_index=True)
    df.loc[1, ["ADR", "IsCanceled", "BookingChanges", "DaysInWaitingList",
               "RequiredCarParkingSpaces", "TotalOfSpecialRequests"]] = [999, 1, 9, 9, 3, 5]
    df.loc[1, ["Agent", "Company", "ReservationStatus", "AssignedRoomType"]] = ["555", "NULL", "Canceled", "Z"]
    df.loc[2, "LeadTime"] += 1
    ids = regression_group_ids(df)
    assert ids.iloc[0] == ids.iloc[1] != ids.iloc[2]
    assert "ADR" not in sum(feature_columns("regression"), [])


def test_missing_text_numeric_normalization_and_no_mutation():
    df = pd.concat([make_bookings().iloc[[0]]] * 2, ignore_index=True)
    df["Children"] = [np.nan, 0]
    df["Country"] = [pd.NA, " Unknown "]
    df["Agent"] = [9.0, " 9 "]
    df["Company"] = [" NuLl ", None]
    df["LeadTime"] = ["1.0", 1]
    before = df.copy(deep=True)
    ids = regression_group_ids(df)
    assert ids.nunique() == 1
    pd.testing.assert_frame_equal(df, before)
    assert regression_group_ids(df.assign(Country=[None, "null"])).nunique() == 1
    # Genuine categorical case differences are preserved by the existing policy.
    assert regression_group_ids(df.assign(Meal=["BB", "bb"])).nunique() == 2


def test_group_determinism_order_subset_and_index_independence():
    df = make_bookings()
    ids = regression_group_ids(df)
    pd.testing.assert_series_equal(ids, regression_group_ids(df))
    pd.testing.assert_series_equal(ids.iloc[::-1], regression_group_ids(df.iloc[::-1]))
    assert ids.iloc[[4]].iloc[0] == regression_group_ids(df.iloc[[4]].reset_index(drop=True)).iloc[0]
    assert ids.nunique() == regression_group_keys(df).nunique()


def test_groups_reject_missing_required_and_invalid_numeric():
    df = make_bookings()
    with pytest.raises(ValueError, match="LeadTime"):
        regression_group_ids(df.drop(columns="LeadTime"))
    with pytest.raises(ValueError, match="finite"):
        regression_group_ids(df.assign(LeadTime=np.inf))
    with pytest.raises(ValueError, match="Invalid numeric"):
        regression_group_ids(df.assign(LeadTime="bad"))


def test_group_keys_do_not_collide_on_sentinels_delimiters_or_long_numbers():
    df = pd.concat([make_bookings().iloc[[0]]] * 2, ignore_index=True)
    assert regression_group_ids(df.assign(Meal=[None, "null-value"])).nunique() == 2
    assert regression_group_ids(df.assign(Meal=['A|B', 'A'], Country=['C', 'B|C'])).nunique() == 2
    assert regression_group_ids(df.assign(LeadTime=[
        '123456789012345678901234567891', '123456789012345678901234567892',
    ])).nunique() == 2


def test_split_alignment_reproducibility_and_group_separation():
    df = pd.concat([make_bookings()] * 3, ignore_index=True)
    df.index = np.arange(len(df)) * 7 + 100
    before = df.copy()
    a = make_split(df, "regression")
    b = make_split(df, "regression")
    for left, right in zip(a, b):
        if isinstance(left, pd.DataFrame):
            pd.testing.assert_frame_equal(left, right)
        else:
            pd.testing.assert_series_equal(left, right)
    for X, y in [(a[0], a[2]), (a[1], a[3])]:
        assert X.index.equals(y.index)
        pd.testing.assert_series_equal(y, df.loc[X.index, "ADR"])
        assert "ADR" not in X and "regression_group" not in X
    assert set(a[0].index).isdisjoint(a[1].index)
    assert set(regression_group_ids(a[0])).isdisjoint(regression_group_ids(a[1]))
    assert set(a[0].index) | set(a[1].index) == set(df.index)
    assert not a[0].equals(make_split(df, "regression", random_state=17)[0])
    pd.testing.assert_frame_equal(df, before)


def test_classification_matches_original_stratified_split_exactly():
    df = make_bookings()
    df.loc[0, "ADR"] = 0
    actual = make_split(df, "classification")
    expected = train_test_split(df.drop(columns="IsCanceled"), df.IsCanceled,
                                test_size=.2, random_state=42, stratify=df.IsCanceled)
    for a, b in zip(actual, expected):
        if isinstance(a, pd.DataFrame):
            pd.testing.assert_frame_equal(a, b)
        else:
            pd.testing.assert_series_equal(a, b)


def test_group_cv_folds_partition_only_training_and_are_reusable():
    train, test, _, _ = make_split(pd.concat([make_bookings()] * 3), "regression")
    groups = regression_group_ids(train)
    for n in (3, 5):
        folds = grouped_cv_folds(train, groups, n)
        repeated = grouped_cv_folds(train, groups, n)
        for (tr, va), (tr2, va2) in zip(folds, repeated):
            np.testing.assert_array_equal(tr, tr2)
            np.testing.assert_array_equal(va, va2)
            assert set(groups.iloc[tr]).isdisjoint(groups.iloc[va])
            assert set(groups.iloc[va]).isdisjoint(regression_group_ids(test))
        np.testing.assert_array_equal(np.sort(np.concatenate([v for _, v in folds])), np.arange(len(train)))
    with pytest.raises(ValueError, match="alignment"):
        grouped_cv_folds(train, groups.iloc[::-1])
    with pytest.raises(ValueError, match="at least"):
        grouped_cv_folds(train, groups, 1000)


@pytest.mark.parametrize("section,key,value", [
    (None, "random_state", True), (None, "random_state", -1),
    ("regression", "test_size", 1), ("regression", "test_size", "0.2"),
    ("regression", "paid_bookings_only", "yes"),
    ("regression", "split_strategy", "random"),
    ("cv", "n_splits", 1), ("tuning", "cv_folds", 3.5),
    ("cleaning", "drop_duplicates", "true"), ("cleaning", "adr_min", 1001),
])
def test_invalid_config(tmp_path, section, key, value):
    cfg = load_config(ROOT / "config.yaml")
    (cfg if section is None else cfg[section])[key] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_duplicate_and_missing_configuration_keys(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("random_state: 42\nrandom_state: 7\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        load_config(path)
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="configuration"):
        load_config(path)


def test_real_config_drives_population_split_and_cv(tmp_path):
    p = prepare_regression_data(ROOT / "config.yaml")
    assert (len(p.X_train), len(p.X_test)) == (41867, 10453)
    assert p.groups_train.index.equals(p.y_train.index)
    assert p.groups_test.index.equals(p.y_test.index)
    assert set(p.groups_train).isdisjoint(p.groups_test)
    assert len(p.baseline_cv) == 5 and len(p.tuning_cv) == 3
    cfg = load_config(ROOT / "config.yaml")
    cfg["dataset"]["path"] = str(ROOT / "data/H2.csv")
    cfg["random_state"] = 17
    cfg["regression"].update(test_size=.3, paid_bookings_only=False)
    cfg["cv"]["n_splits"] = 4
    cfg["tuning"]["cv_folds"] = 2
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    q = prepare_regression_data(path)
    assert len(q.X_train) + len(q.X_test) == 53042
    assert (pd.concat([q.y_train, q.y_test]) == 0).any()
    assert .28 < len(q.X_test) / 53042 < .32
    assert len(q.baseline_cv) == 4 and len(q.tuning_cv) == 2
