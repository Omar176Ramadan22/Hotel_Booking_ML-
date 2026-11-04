from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from src.data.load import standardize_text


class BookingFeatureBuilder(BaseEstimator, TransformerMixin):

    def __init__(self, top_n_agents: int = 0):
        self.top_n_agents = top_n_agents

    def fit(self, X, y=None):
        # Learn top agents within each training fold.
        self.top_agents_: list[str] = []
        if self.top_n_agents > 0:
            agents = standardize_text(X)["Agent"].dropna()
            self.top_agents_ = (
                agents.value_counts().head(self.top_n_agents).index.tolist()
            )
        return self

    def transform(self, X) -> pd.DataFrame:
        check_is_fitted(self, "top_agents_")
        df = standardize_text(X)
        # New bookings may omit these optional fields.
        for col in ("Agent", "Company", "Children", "Country"):
            if col not in df.columns:
                df[col] = pd.NA
        # Create presence flags before dropping raw IDs.
        df["HasAgent"] = df["Agent"].notna().astype(int)
        df["HasCompany"] = df["Company"].notna().astype(int)
        if self.top_n_agents > 0:
            agent = df["Agent"].astype(object)
            df["AgentTop"] = np.where(
                agent.isna(),
                "None",
                np.where(agent.isin(self.top_agents_), agent, "Other"),
            )
        df = df.drop(columns=["Agent", "Company"])
        # Use the same fixed defaults for training and prediction.
        df["Children"] = pd.to_numeric(df["Children"], errors="coerce").fillna(0)
        df["Country"] = df["Country"].fillna("Unknown")
        # sklearn expects np.nan rather than pd.NA in categorical inputs.
        text_cols = df.select_dtypes(include=["object", "string"]).columns
        for col in text_cols:
            df[col] = df[col].astype(object).where(df[col].notna(), np.nan)
        return df
