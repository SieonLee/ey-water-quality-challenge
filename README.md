# EY Water Quality Challenge

This repository documents my end-to-end approach for the EY water quality prediction challenge. The project combines remote sensing features from Landsat, climate variables from TerraClimate, and geospatial-temporal feature engineering to predict three water quality targets:

- Total Alkalinity
- Electrical Conductance
- Dissolved Reactive Phosphorus

The final modeling approach blends `ExtraTreesRegressor` with a quantile-based `HistGradientBoostingRegressor`, with a more aggressive blend for phosphorus to better recover upper-tail behavior.

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

## Approach

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

The EDA notebook is in [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb). It covers:

- target distributions
- missing-value profiles
- geographic sampling locations
- feature correlations
- simple target-vs-feature visual diagnostics

You can also generate exportable figures directly with the script:

```bash
python src/visualization.py
```

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

## Key Takeaways

- Tree ensembles worked well on heterogeneous environmental tabular data.
- Target-specific blending helped more than using a single global ensemble rule.
- Simple geospatial clustering added useful local context without requiring a full spatial model.

## Notes

- Raw challenge data may be subject to competition terms and should be reviewed before public upload.
- If you want to keep this repo public, keep the CSVs excluded and include only code, documentation, and generated example figures.
