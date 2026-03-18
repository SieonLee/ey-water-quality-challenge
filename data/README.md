# Data Notes

This folder documents the expected raw input files for the project.

The repository code expects the following CSV files to exist in the project root:

- `water_quality_training_dataset.csv`
- `landsat_features_training.csv`
- `terraclimate_features_training.csv`
- `submission_template.csv`
- `landsat_features_validation.csv`
- `terraclimate_features_validation.csv`

These files are excluded from version control by default because challenge or competition datasets may have sharing restrictions.

If you want to reproduce the results locally, place the CSV files in the repository root and run:

```bash
python src/train.py
```

