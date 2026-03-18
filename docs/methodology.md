# Methodology

## Objective

Predict three continuous water quality targets from remote sensing and climate covariates:

- Total Alkalinity
- Electrical Conductance
- Dissolved Reactive Phosphorus

## Pipeline Summary

1. Merge target labels with Landsat and TerraClimate features on location and sample date.
2. Parse dates and engineer temporal seasonality features.
3. Create simple water-index derivatives from spectral bands.
4. Add a missingness flag for Landsat availability.
5. Log-transform `pet` to reduce skew.
6. Cluster coordinates into regional groups using `KMeans`.
7. Train target-specific blended regressors.

## Why This Model Family

`ExtraTreesRegressor` provides:

- strong nonlinear performance on mixed tabular data
- low sensitivity to scaling
- resilience to moderately noisy engineered features

Quantile `HistGradientBoostingRegressor` provides:

- asymmetric prediction support for high-value tails
- complementary behavior when ExtraTrees regresses too hard toward the mean

## Blending Strategy

Each target uses a different blend weight and quantile:

- Total Alkalinity: mild quantile support
- Electrical Conductance: moderate quantile support
- Dissolved Reactive Phosphorus: strongest quantile support

This target-wise strategy reflects that the phosphorus target was the hardest to fit well with a mean-oriented ensemble alone.

## Suggested Future Improvements

- cross-validated blend-weight tuning
- log-transform experiments for skewed targets
- spatial cross-validation to stress-test geographic generalization
- feature importance and SHAP-style diagnostics
- residual analysis by season and region

