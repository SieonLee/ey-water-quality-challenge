from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


SEED = 42

TARGETS = [
    "Total Alkalinity",
    "Electrical Conductance",
    "Dissolved Reactive Phosphorus",
]

FEATURES = [
    "nir",
    "green",
    "swir16",
    "swir22",
    "NDMI",
    "MNDWI",
    "NDWI_green",
    "NDWI_swir",
    "landsat_missing",
    "pet",
    "pet_log",
    "Latitude",
    "Longitude",
    "geo_cluster",
    "month",
    "dayofyear",
    "quarter",
    "month_sin",
    "month_cos",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["month"] = df["Sample Date"].dt.month.astype("Int64")
    df["dayofyear"] = df["Sample Date"].dt.dayofyear.astype("Int64")
    df["quarter"] = df["Sample Date"].dt.quarter.astype("Int64")
    df["month_sin"] = np.sin(2 * np.pi * df["month"].fillna(0) / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"].fillna(0) / 12)

    nir = df.get("nir", pd.Series(np.nan, index=df.index))
    green = df.get("green", pd.Series(np.nan, index=df.index))
    swir16 = df.get("swir16", pd.Series(np.nan, index=df.index))

    df["NDWI_green"] = (green - nir) / (green + nir + 1e-8)
    df["NDWI_swir"] = (nir - swir16) / (nir + swir16 + 1e-8)
    df["landsat_missing"] = nir.isna().astype(int)

    if "pet" in df.columns:
        df["pet_log"] = np.log1p(df["pet"].clip(lower=0))

    return df


def add_geo_clusters(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_clusters: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = train_df.copy()
    test_df = test_df.copy()

    km = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
    km.fit(train_df[["Latitude", "Longitude"]].to_numpy())

    train_df["geo_cluster"] = km.predict(train_df[["Latitude", "Longitude"]].to_numpy())
    test_df["geo_cluster"] = km.predict(test_df[["Latitude", "Longitude"]].to_numpy())
    return train_df, test_df


def get_available_features(df: pd.DataFrame) -> list[str]:
    return [feature for feature in FEATURES if feature in df.columns]

