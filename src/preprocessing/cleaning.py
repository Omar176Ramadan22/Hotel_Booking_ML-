# %% [markdown]
# # Two-layer cleaning
# **Layer 1 (generic):** trim whitespace, standardise missing markers, drop duplicates.
# **Layer 2 (dataset-specific):** rules chosen by the dataset description file (YAML).
#
# Design rule: everything here is *stateless* - no statistic is learned from the data
# (no means, medians, quantiles). So it is safe to run on the full file BEFORE the
# train/test split. Anything data-dependent (imputation, encoding, scaling) belongs in
# the sklearn Pipeline and is fitted on the training split only.

# %% Cell 1 - imports and constants
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import yaml

log = logging.getLogger("cleaning")
DEFAULT_MISSING_MARKERS = ["", "na", "n/a", "nan", "null", "none"]


# %% Cell 2 - Layer 1: generic functions (work on ANY dataframe)
def trim_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing spaces from every text column (e.g. 'HB       ' -> 'HB')."""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_string_dtype(out[col]) or out[col].dtype == object:
            out[col] = out[col].astype("string").str.strip().astype(object)
    return out


def standardize_missing(df: pd.DataFrame, markers=DEFAULT_MISSING_MARKERS) -> pd.DataFrame:
    """Turn marker strings such as 'NULL' or '' into real NaN (case-insensitive)."""
    markers = {m.lower() for m in markers}
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object or pd.api.types.is_string_dtype(out[col]):
            s = out[col].astype("string")
            out[col] = out[col].where(~s.str.lower().isin(markers), np.nan)
    return out


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows (keeps the first occurrence)."""
    return df.drop_duplicates(keep="first").reset_index(drop=True)


def generic_clean(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Run layer 1 according to the `generic:` block of the description file."""
    report = {"rows_in": len(df)}
    if cfg.get("trim_whitespace", True):
        df = trim_whitespace(df)
    before = int(df.isna().sum().sum())
    df = standardize_missing(df, cfg.get("missing_markers", DEFAULT_MISSING_MARKERS))
    report["missing_markers_converted"] = int(df.isna().sum().sum()) - before
    if cfg.get("drop_duplicates", True):
        n = len(df)
        df = remove_duplicates(df)
        report["duplicates_removed"] = n - len(df)
    report["rows_out"] = len(df)
    return df, report


# %% Cell 3 - Layer 2: dataset-specific rule registry
RULES: dict[str, Callable[..., pd.DataFrame]] = {}


def rule(name: str):
    """Decorator: register a cleaning rule so the description file can call it by name."""
    def deco(fn):
        RULES[name] = fn
        return fn
    return deco


@rule("undefined_to_missing")
def undefined_to_missing(df, columns, markers=("Undefined",), **_):
    out = df.copy()
    for col in columns:
        out[col] = out[col].where(~out[col].isin(markers), np.nan)
    return out


@rule("cast_types")
def cast_types(df, numeric=(), dates=(), **_):
    out = df.copy()
    for col in numeric:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in dates:
        out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


@rule("adr_outliers")
def adr_outliers(df, column="ADR", min_value=0.0, max_value=1000.0, action="drop", **_):
    """Fixed domain limits. Quantile/IQR fences would be learned from data = leakage-prone."""
    s = df[column]
    bad = (s < min_value) | (s > max_value)
    if action == "drop":
        return df.loc[~bad].reset_index(drop=True)
    out = df.copy()                                   # action == "clip"
    out[column] = s.clip(min_value, max_value)
    return out


@rule("invalid_guest_count")
def invalid_guest_count(df, columns, **_):
    total = df[list(columns)].fillna(0).sum(axis=1)
    return df.loc[total > 0].reset_index(drop=True)


@rule("zero_night_stays")
def zero_night_stays(df, columns, **_):
    total = df[list(columns)].fillna(0).sum(axis=1)
    return df.loc[total > 0].reset_index(drop=True)


def dataset_clean(df: pd.DataFrame, rules_cfg: list[dict]) -> tuple[pd.DataFrame, list[dict]]:
    """Run layer 2: apply, in order, every rule listed (and enabled) in the description file."""
    steps = []
    for spec in rules_cfg:
        spec = dict(spec)
        name = spec.pop("name")
        if not spec.pop("enabled", True):
            continue
        if name not in RULES:
            raise KeyError(f"Unknown cleaning rule '{name}'. Registered: {sorted(RULES)}")
        n = len(df)
        df = RULES[name](df, **spec)
        steps.append({"rule": name, "rows_removed": n - len(df)})
    return df, steps


# %% Cell 4 - Orchestrator: description file -> both layers
def load_description(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def clean(df: pd.DataFrame, description: dict) -> tuple[pd.DataFrame, dict]:
    df, generic_report = generic_clean(df, description.get("generic", {}))
    df, rule_steps = dataset_clean(df, description.get("dataset_rules", []))
    report = {"dataset": description.get("dataset"),
              "generic": generic_report,
              "dataset_rules": rule_steps,
              "final_rows": len(df)}
    return df, report


def clean_from_description(
    description_path: str | Path,
    project_root: str | Path | None = None,
    df: pd.DataFrame | None = None,
    save: bool = True,
) -> tuple[pd.DataFrame, dict]:
    root = Path(project_root) if project_root is not None else Path.cwd()
    description = Path(description_path)
    if not description.is_absolute():
        description = root / description
    desc = load_description(description)

    if df is None:
        raw_path = Path(desc["raw_path"])
        if not raw_path.is_absolute():
            raw_path = root / raw_path
        raw = pd.read_csv(raw_path)
    else:
        raw = df.copy()

    cleaned, report = clean(raw, desc)
    if save:
        clean_path = Path(desc["clean_path"])
        report_path = Path(desc["report_path"])
        if not clean_path.is_absolute():
            clean_path = root / clean_path
        if not report_path.is_absolute():
            report_path = root / report_path
        clean_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned.to_csv(clean_path, index=False)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return cleaned, report


# %% Cell 5 - Run it
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cleaned, report = clean_from_description("config/datasets/h2.yaml")
    print(json.dumps(report, indent=2))
    print(cleaned.shape)
