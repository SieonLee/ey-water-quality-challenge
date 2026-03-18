from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = [
    "water_quality_training_dataset.csv",
    "landsat_features_training.csv",
    "terraclimate_features_training.csv",
    "submission_template.csv",
    "landsat_features_validation.csv",
    "terraclimate_features_validation.csv",
]


def _candidate_data_roots() -> list[Path]:
    env_data_dir = os.environ.get("EY_WATER_QUALITY_DATA_DIR")
    candidates = [
        PROJECT_ROOT,
        PROJECT_ROOT.parent / "Snowflake Notebooks Package",
    ]
    if env_data_dir:
        candidates.insert(0, Path(env_data_dir).expanduser())
    return candidates


def resolve_data_root() -> Path:
    for candidate in _candidate_data_roots():
        if all((candidate / name).exists() for name in REQUIRED_FILES):
            return candidate

    searched = "\n".join(f"- {path}" for path in _candidate_data_roots())
    raise FileNotFoundError(
        "Could not find the required CSV files. Checked:\n"
        f"{searched}\n\n"
        "Set EY_WATER_QUALITY_DATA_DIR to the folder containing the challenge CSV files."
    )


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(resolve_data_root() / name)


def parse_sample_date(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Sample Date"] = pd.to_datetime(
        df["Sample Date"],
        format="mixed",
        dayfirst=True,
        errors="coerce",
    )
    return df


def load_raw_datasets() -> dict[str, pd.DataFrame]:
    dataset_names = {
        "train_target": "water_quality_training_dataset.csv",
        "landsat_train": "landsat_features_training.csv",
        "terraclimate_train": "terraclimate_features_training.csv",
        "test_template": "submission_template.csv",
        "landsat_test": "landsat_features_validation.csv",
        "terraclimate_test": "terraclimate_features_validation.csv",
    }

    datasets = {key: parse_sample_date(read_csv(path)) for key, path in dataset_names.items()}
    return datasets


def build_model_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    datasets = load_raw_datasets()

    train_full = datasets["train_target"].merge(
        datasets["landsat_train"],
        on=["Latitude", "Longitude", "Sample Date"],
        how="left",
    )
    train_full = train_full.merge(
        datasets["terraclimate_train"],
        on=["Latitude", "Longitude", "Sample Date"],
        how="left",
    )

    test_full = datasets["test_template"].merge(
        datasets["landsat_test"],
        on=["Latitude", "Longitude", "Sample Date"],
        how="left",
    )
    test_full = test_full.merge(
        datasets["terraclimate_test"],
        on=["Latitude", "Longitude", "Sample Date"],
        how="left",
    )

    return train_full, test_full
