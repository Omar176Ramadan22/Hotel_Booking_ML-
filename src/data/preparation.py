"""Configuration-driven, model-free shared regression preparation."""

from dataclasses import dataclass
from pathlib import Path
import math

import pandas as pd
import yaml

from src.data.groups import regression_group_ids
from src.data.load import load_clean
from src.data.split import grouped_cv_folds, make_split


class UniqueKeyLoader(yaml.SafeLoader):
    """Reject duplicate YAML keys rather than silently overwriting settings."""


def _unique_mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"Duplicate configuration key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def load_config(path="config.yaml"):
    with Path(path).open(encoding="utf-8") as stream:
        cfg = yaml.load(stream, Loader=UniqueKeyLoader)
    if not isinstance(cfg, dict):
        raise ValueError("Configuration must be a mapping")
    try:
        seed = cfg["random_state"]
        size = cfg["regression"]["test_size"]
        paid = cfg["regression"]["paid_bookings_only"]
        strategy = cfg["regression"]["split_strategy"]
        folds = [cfg["cv"]["n_splits"], cfg["tuning"]["cv_folds"]]
        dataset_path = cfg["dataset"]["path"]
        cleaning = cfg["cleaning"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Missing or malformed preparation configuration: {exc}") from exc
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("random_state must be an integer in [0, 2**32)")
    if type(size) not in (int, float) or not math.isfinite(size) or not 0 < size < 1:
        raise ValueError("regression.test_size must be a fraction between 0 and 1")
    if type(paid) is not bool:
        raise ValueError("regression.paid_bookings_only must be boolean")
    if strategy != "group_shuffle":
        raise ValueError("regression.split_strategy must be group_shuffle")
    if any(type(n) is not int or n < 2 for n in folds):
        raise ValueError("CV fold counts must be integers >= 2")
    if not isinstance(dataset_path, str) or not dataset_path:
        raise ValueError("dataset.path must be a nonempty string")
    if not isinstance(cleaning, dict):
        raise ValueError("cleaning must be a mapping")
    allowed = {"drop_duplicates", "drop_zero_guests", "drop_zero_nights", "adr_min", "adr_max"}
    if set(cleaning) - allowed:
        raise ValueError(f"Unknown cleaning options: {sorted(set(cleaning) - allowed)}")
    for key in allowed - {"adr_min", "adr_max"}:
        if key in cleaning and type(cleaning[key]) is not bool:
            raise ValueError(f"cleaning.{key} must be boolean")
    for key in ("adr_min", "adr_max"):
        value = cleaning.get(key)
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
            raise ValueError(f"cleaning.{key} must be finite numeric or null")
    low, high = cleaning.get("adr_min", 0), cleaning.get("adr_max", 1000)
    if low is not None and high is not None and low > high:
        raise ValueError("cleaning.adr_min must be <= cleaning.adr_max")
    return cfg


@dataclass
class RegressionPreparation:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    groups_train: pd.Series
    groups_test: pd.Series
    baseline_cv: tuple
    tuning_cv: tuple
    cleaning_log: list


def prepare_regression_data(config_path="config.yaml") -> RegressionPreparation:
    """Read central settings, clean once, split once, and freeze training CV folds."""
    path = Path(config_path).resolve()
    cfg = load_config(path)
    data_path = Path(cfg["dataset"]["path"])
    if not data_path.is_absolute():
        data_path = path.parent / data_path
    df, log = load_clean(data_path, **cfg["cleaning"])
    options = cfg["regression"]
    X_train, X_test, y_train, y_test = make_split(
        df, "regression", test_size=options["test_size"],
        random_state=cfg["random_state"], paid_bookings_only=options["paid_bookings_only"],
    )
    groups_train, groups_test = regression_group_ids(X_train), regression_group_ids(X_test)
    return RegressionPreparation(
        X_train, X_test, y_train, y_test, groups_train, groups_test,
        grouped_cv_folds(X_train, groups_train, cfg["cv"]["n_splits"]),
        grouped_cv_folds(X_train, groups_train, cfg["tuning"]["cv_folds"]), log,
    )
