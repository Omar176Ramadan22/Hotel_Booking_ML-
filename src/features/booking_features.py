"""Row-level feature building that runs INSIDE the sklearn pipeline.

Why a transformer and not a cleaning step?
    The same code must run on the training data and on a single new booking at
    prediction time. Putting it in the pipeline guarantees both see identical
    treatment, and the saved pipeline carries it along.

Order of operations (this matters):
    1. HasAgent / HasCompany are created FIRST, from the raw ID columns.
    2. Only then are the raw Agent / Company columns dropped.
    If the flags were created after the drop they would be all zeros.

Fixed-rule fills (Children -> 0, Country -> 'Unknown') learn nothing from the
data, so they are safe anywhere. The only fitted statistic here is the list of
most frequent agents (when top_n_agents > 0); it is learned in `fit`, i.e. from
training data only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from src.data.load import standardize_text


class BookingFeatureBuilder(BaseEstimator, TransformerMixin):
    """Create HasAgent / HasCompany (+ optional AgentTop) and apply fixed fills.

    Parameters
    ----------
    top_n_agents : int, default 0
        0  -> baseline: raw Agent ID is dropped, only the HasAgent flag is kept.
        N>0 -> also keep an `AgentTop` categorical column holding the N most
               frequent agents seen in training, 'Other' for the rest and
               'None' when no agent was recorded.
    """

    def __init__(self, top_n_agents: int = 0):
        self.top_n_agents = top_n_agents

    # -- fit: the only learned thing is the list of top agents --------------
    def fit(self, X, y=None):
        self.top_agents_: list[str] = []
        if self.top_n_agents > 0:
            agents = standardize_text(X)["Agent"].dropna()
            self.top_agents_ = agents.value_counts().head(self.top_n_agents).index.tolist()
        return self

    # -- transform: identical for training rows and new bookings -------------
    def transform(self, X) -> pd.DataFrame:
        check_is_fitted(self, "top_agents_")
        df = standardize_text(X)  # copy; same text rules as the cleaning step

        # A new booking from a form may omit these fields entirely.
        for col in ("Agent", "Company", "Children", "Country"):
            if col not in df.columns:
                df[col] = pd.NA

        # 1) flags BEFORE dropping the raw IDs
        df["HasAgent"] = df["Agent"].notna().astype(int)
        df["HasCompany"] = df["Company"].notna().astype(int)

        # 2) optional top-N agent grouping (uses the list learned in fit)
        if self.top_n_agents > 0:
            agent = df["Agent"].astype(object)
            df["AgentTop"] = np.where(
                agent.isna(), "None",
                np.where(agent.isin(self.top_agents_), agent, "Other"),
            )

        # 3) now the raw IDs can go
        df = df.drop(columns=["Agent", "Company"])

        # 4) fixed-rule fills (no statistics learned)
        df["Children"] = pd.to_numeric(df["Children"], errors="coerce").fillna(0)
        df["Country"] = df["Country"].fillna("Unknown")

        # sklearn imputers/encoders expect np.nan, not pd.NA, in text columns
        text_cols = df.select_dtypes(include=["object", "string"]).columns
        for col in text_cols:
            df[col] = df[col].astype(object).where(df[col].notna(), np.nan)
        return df
