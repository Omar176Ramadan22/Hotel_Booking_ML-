from pathlib import Path
from types import SimpleNamespace
from sklearn.model_selection import GroupShuffleSplit

from src.data.groups import regression_group_ids
from src.data.load import load_clean
from src.data.preparation import load_config, validate_config
from src.data.split import grouped_cv_folds, regression_population
from src.features.columns import TARGETS


def prepare_training_data(config_path="config.yaml", config=None):
    path = Path(config_path).resolve()
    cfg = load_config(path) if config is None else validate_config(config)
    data_path = Path(cfg["dataset"]["path"])
    if not data_path.is_absolute():
        data_path = path.parent / data_path
    data, log = load_clean(data_path, **cfg["cleaning"])
    options = cfg["regression"]
    if options["paid_bookings_only"]:
        data = regression_population(data)
    target = TARGETS["regression"]
    features = data.drop(columns=target)
    groups = regression_group_ids(features)
    # Recover the original membership without constructing held-out features or labels.
    training, _ = next(GroupShuffleSplit(
        n_splits=1, test_size=options["test_size"], random_state=cfg["random_state"],
    ).split(features, groups=groups))
    X_train = features.iloc[training]
    y_train = data[target].iloc[training]
    groups_train = groups.iloc[training]
    return SimpleNamespace(
        X_train=X_train, y_train=y_train, groups_train=groups_train, cleaning_log=log,
        baseline_cv=grouped_cv_folds(X_train, groups_train, cfg["cv"]["n_splits"]),
        tuning_cv=grouped_cv_folds(X_train, groups_train, cfg["tuning"].get("cv_folds", 3)),
    )
