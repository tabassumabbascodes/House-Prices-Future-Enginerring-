
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class HouseFeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X["TotalSF"] = (
            X["TotalBsmtSF"].fillna(0)
            + X["1stFlrSF"].fillna(0)
            + X["2ndFlrSF"].fillna(0)
        )

        X["TotalBathrooms"] = (
            X["FullBath"].fillna(0)
            + 0.5 * X["HalfBath"].fillna(0)
            + X["BsmtFullBath"].fillna(0)
            + 0.5 * X["BsmtHalfBath"].fillna(0)
        )

        X["HouseAgeAtSale"] = (
            X["YrSold"] - X["YearBuilt"]
        ).clip(lower=0)

        X["YearsSinceRemodel"] = (
            X["YrSold"] - X["YearRemodAdd"]
        ).clip(lower=0)

        X["IsRemodeled"] = (
            X["YearRemodAdd"] > X["YearBuilt"]
        ).astype(int)

        porch_cols = [
            "WoodDeckSF", "OpenPorchSF", "EnclosedPorch",
            "3SsnPorch", "ScreenPorch"
        ]

        X["TotalPorchSF"] = X[porch_cols].fillna(0).sum(axis=1)

        return X
