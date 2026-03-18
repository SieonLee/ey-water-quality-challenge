from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from data_loading import PROJECT_ROOT, build_model_tables, load_raw_datasets
from features import SEED, TARGETS, add_features, add_geo_clusters, get_available_features


np.random.seed(SEED)

BLEND_W = {
    "Total Alkalinity": 0.12,
    "Electrical Conductance": 0.20,
    "Dissolved Reactive Phosphorus": 0.35,
}


def make_extratrees() -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=2000,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=2,
        max_features="sqrt",
        bootstrap=True,
        random_state=SEED,
        n_jobs=-1,
    )


def make_quantile(q: float) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="quantile",
        quantile=q,
        learning_rate=0.06,
        max_depth=6,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=0.2,
        max_bins=255,
        random_state=SEED,
    )


def get_quantile_for_target(target: str) -> float:
    if target == "Dissolved Reactive Phosphorus":
        return 0.85
    if target == "Electrical Conductance":
        return 0.80
    return 0.75


def build_preprocess(available_features: list[str]) -> ColumnTransformer:
    categorical_features = [feature for feature in ["geo_cluster"] if feature in available_features]
    numeric_features = [feature for feature in available_features if feature not in categorical_features]

    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            ("cat", SimpleImputer(strategy="most_frequent"), categorical_features),
        ],
        remainder="drop",
    )


def fit_and_predict() -> pd.DataFrame:
    train_full, test_full = build_model_tables()

    train_processed = add_features(train_full)
    test_processed = add_features(test_full)
    train_processed, test_processed = add_geo_clusters(train_processed, test_processed)

    available_features = get_available_features(train_processed)
    preprocess = build_preprocess(available_features)

    x_train = train_processed[available_features]
    x_test = test_processed[available_features]
    predictions: dict[str, np.ndarray] = {}

    print("=" * 80)
    print("Final Attempt: ExtraTrees(bootstrap) + target-wise Quantile blending")
    print("=" * 80)

    for target in TARGETS:
        weight = BLEND_W[target]
        quantile = get_quantile_for_target(target)
        y_train = train_processed[target].astype(float).to_numpy()

        print(f"\nTarget: {target}")
        print(f"  blend weight (quantile): {weight:.2f}")
        print(f"  quantile: {quantile:.2f}")

        extra_trees = Pipeline([("prep", preprocess), ("model", make_extratrees())])
        extra_trees.fit(x_train, y_train)
        pred_extra_trees = extra_trees.predict(x_test)

        quantile_model = Pipeline([("prep", preprocess), ("model", make_quantile(quantile))])
        quantile_model.fit(x_train, y_train)
        pred_quantile = quantile_model.predict(x_test)

        prediction = (1 - weight) * pred_extra_trees + weight * pred_quantile
        prediction = np.maximum(prediction, 0)

        predictions[target] = prediction

        print(f"  Train mean: {y_train.mean():.2f}")
        print(f"  Pred  mean: {prediction.mean():.2f}")
        print(f"  Pred   std: {prediction.std():.2f} (train std: {y_train.std():.2f})")

    submission = load_raw_datasets()["test_template"].copy()
    for target in TARGETS:
        submission[target] = predictions[target]

    return submission


def main() -> None:
    output_dir = PROJECT_ROOT / "outputs" / "submissions"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "submission_final_aggressive_phos.csv"

    submission = fit_and_predict()
    submission.to_csv(output_path, index=False)
    print(f"\nSaved submission to: {output_path}")


if __name__ == "__main__":
    main()

