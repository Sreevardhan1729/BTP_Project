# Project Pipeline

This repository contains a machine learning pipeline.

## Dataset
The pipeline utilizes the Politifact dataset. Raw data files (`politifact_train.csv`, `politifact_test.csv`) are expected in the `data/raw/` directory. The dataset includes a 'news' column for text and a 'label' column for the target variable.

## How to Run
To execute the entire pipeline, run the following command:

```bash
python scripts/run_pipeline.py --config configs/config-run.yaml
```
