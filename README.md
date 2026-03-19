# EY Water Quality Challenge

End-to-end machine learning project for predicting water quality indicators using Landsat, TerraClimate, geospatial feature engineering, and ensemble regression models.

This repository documents my approach to the EY water quality prediction challenge. The project combines remote sensing features from Landsat, climate variables from TerraClimate, and geospatial-temporal feature engineering to predict three water quality targets:

- Total Alkalinity
- Electrical Conductance
- Dissolved Reactive Phosphorus

The final modeling approach blends `ExtraTreesRegressor` with a quantile-based `HistGradientBoostingRegressor`, with a more aggressive blend for phosphorus to better recover upper-tail behavior.

## Why This Project Matters

This project shows a full applied machine learning workflow on messy environmental data, from data integration and feature engineering through model design and submission generation.

From a hiring perspective, it demonstrates:

- end-to-end ownership of a real prediction problem
- practical use of geospatial, temporal, and climate features
- modeling choices driven by data characteristics instead of generic defaults
- ensemble design for skewed targets and noisy observations
- strong fit for applied ML, analytics, and environmental or geospatial data roles

## Recruiter Snapshot

- **Problem:** predict three water quality indicators from satellite, climate, spatial, and date-based features
- **Data:** Landsat remote sensing features, TerraClimate variables, coordinates, and sampling dates
- **Approach:** feature engineering plus target-specific ensemble regression
- **Key modeling choice:** blend `ExtraTreesRegressor` with quantile `HistGradientBoostingRegressor`
- **Why it stands out:** connects remote sensing style data with practical tabular ML decisions

## Business / Modeling Framing

The dataset is heterogeneous, partially missing, and skewed across targets, especially for phosphorus. Rather than treating this as a one-model benchmark problem, I built a compact ensemble designed to handle nonlinear interactions, missingness, and upper-tail recovery more effectively.

## Overview

This project focuses on a practical tabular modeling workflow for environmental prediction. Rather than relying on a single off-the-shelf model, I built a target-specific ensemble that combines:

- spectral information from Landsat
- climate variables from TerraClimate
- temporal seasonality features
- lightweight spatial context from coordinate clustering

The final solution was designed to be simple, interpretable, and effective on heterogeneous tabular data with missing values and skewed targets.

## Project Structure

```text
.
├── README.md
├── requirements.txt
├── data/
│   └── README.md
├── docs/
│   └── methodology.md
├── notebooks/
│   └── 01_eda.ipynb
├── outputs/
│   ├── figures/
│   └── submissions/
└── src/
    ├── data_loading.py
    ├── features.py
    ├── predict.py
    ├── train.py
    └── visualization.py
```

## Problem Framing

The task is to predict water quality indicators using tabular environmental features aligned by location and sampling date. I focused on building a practical tabular ensemble that balances:

- nonlinear feature interactions
- robustness to missing satellite and climate values
- sensitivity to high-value target regions, especially phosphorus

## Data Sources

The workflow uses the following challenge files:

- `water_quality_training_dataset.csv`
- `landsat_features_training.csv`
- `terraclimate_features_training.csv`
- `submission_template.csv`
- `landsat_features_validation.csv`
- `terraclimate_features_validation.csv`

The raw CSV files are intentionally not versioned in GitHub. See [data/README.md](data/README.md) for details.

## Workflow

### 1. Data Integration

Training and validation features are merged on:

- `Latitude`
- `Longitude`
- `Sample Date`

### 2. Feature Engineering

Engineered features include:

- calendar features: month, quarter, day-of-year
- cyclical seasonal encoding: `month_sin`, `month_cos`
- derived water indices: `NDWI_green`, `NDWI_swir`
- missingness indicator for Landsat availability
- `pet_log` as a stabilized climate feature
- geospatial clusters from latitude and longitude via `KMeans`

### 3. Modeling

For each target:

- fit an `ExtraTreesRegressor` as a strong nonlinear baseline
- fit a quantile `HistGradientBoostingRegressor` to better capture high-end values
- blend both predictions with target-specific weights

This was especially useful for `Dissolved Reactive Phosphorus`, where underprediction of the upper tail was a key bottleneck.

## Exploratory Data Analysis

The EDA notebook is in [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb), and the notebook includes rendered outputs so the visual analysis is visible directly on GitHub.

The EDA focuses on:

- target distributions
- missing-value profiles
- geographic sampling locations
- feature correlations
- simple target-vs-feature visual diagnostics

Key observations from EDA:

- `Dissolved Reactive Phosphorus` is more skewed than the other targets, making tail prediction more important.
- Several remote sensing features contain meaningful missingness, so explicit imputation and missingness flags were important.
- Sampling locations are geographically structured rather than uniformly distributed, which motivated the `geo_cluster` feature.
- Feature-target relationships appear nonlinear, which supports using tree-based ensemble models.

You can also generate exportable figures directly with the script:

```bash
python src/visualization.py
```

## Feature Engineering

To improve predictive performance, I engineered a compact set of temporal, spectral, climate, and spatial features on top of the merged raw inputs.

Main feature groups:

- temporal features: `month`, `dayofyear`, `quarter`
- cyclical seasonality: `month_sin`, `month_cos`
- water-related indices: `NDWI_green`, `NDWI_swir`
- missingness signal: `landsat_missing`
- transformed climate feature: `pet_log`
- location-aware feature: `geo_cluster`

The goal was not to maximize feature count, but to add a small number of useful features that reflect seasonality, water-related spectral behavior, missing-data patterns, and geographic structure.

## Modeling Approach

The final solution uses a target-specific blended ensemble.

For each target:

- train an `ExtraTreesRegressor` as the main nonlinear baseline
- train a quantile `HistGradientBoostingRegressor` for better upper-tail recovery
- blend the two predictions with target-specific weights

Target-specific blend weights:

- `Total Alkalinity`: `0.12`
- `Electrical Conductance`: `0.20`
- `Dissolved Reactive Phosphorus`: `0.35`

Target-specific quantiles:

- `Total Alkalinity`: `0.75`
- `Electrical Conductance`: `0.80`
- `Dissolved Reactive Phosphorus`: `0.85`

This setup worked well because `ExtraTrees` handled general nonlinear tabular structure effectively, while the quantile model helped reduce underprediction in higher-value regions.

## Reproducibility

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the training and generate a submission:

```bash
python src/train.py
```

This writes a submission file to:

```text
outputs/submissions/submission_final_aggressive_phos.csv
```

## Repository Highlights

- [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb): EDA with rendered charts and diagnostic visuals
- [src/train.py](src/train.py): final training and submission-generation pipeline
- [src/features.py](src/features.py): reusable feature engineering utilities
- [docs/methodology.md](docs/methodology.md): concise methodology summary

## Key Takeaways

- Tree ensembles worked well on heterogeneous environmental tabular data.
- Target-specific blending helped more than using a single global ensemble rule.
- Simple geospatial clustering added useful local context without requiring a full spatial model.

## Notes

- Raw challenge data may be subject to competition terms and should be reviewed before public upload.
- If you want to keep this repo public, keep the CSVs excluded and include only code, documentation, and generated example figures.
