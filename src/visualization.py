from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from data_loading import PROJECT_ROOT, build_model_tables
from features import TARGETS, add_features


sns.set_theme(style="whitegrid")


def save_target_distributions(train_df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, target in zip(axes, TARGETS):
        sns.histplot(train_df[target], kde=True, ax=ax, color="#2b6cb0")
        ax.set_title(target)
        ax.set_xlabel("Value")
    fig.tight_layout()
    fig.savefig(output_dir / "target_distributions.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_missingness(train_df: pd.DataFrame, output_dir: Path) -> None:
    missing_pct = train_df.isna().mean().sort_values(ascending=False).head(15) * 100
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=missing_pct.values, y=missing_pct.index, ax=ax, color="#dd6b20")
    ax.set_title("Top Missing Features")
    ax.set_xlabel("Missing Percentage")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(output_dir / "missingness_top15.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_sampling_locations(train_df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(
        data=train_df,
        x="Longitude",
        y="Latitude",
        hue="Electrical Conductance",
        palette="viridis",
        s=22,
        alpha=0.7,
        ax=ax,
    )
    ax.set_title("Sampling Locations")
    fig.tight_layout()
    fig.savefig(output_dir / "sampling_locations.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_correlation_heatmap(train_df: pd.DataFrame, output_dir: Path) -> None:
    columns = [column for column in ["nir", "green", "swir16", "pet", "NDMI", "MNDWI"] if column in train_df.columns]
    corr_df = train_df[columns + TARGETS].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_df, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Feature/Target Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(output_dir / "correlation_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    output_dir = PROJECT_ROOT / "outputs" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_full, _ = build_model_tables()
    train_processed = add_features(train_full)

    save_target_distributions(train_processed, output_dir)
    save_missingness(train_processed, output_dir)
    save_sampling_locations(train_processed, output_dir)
    save_correlation_heatmap(train_processed, output_dir)

    print(f"Saved figures to: {output_dir}")


if __name__ == "__main__":
    main()

