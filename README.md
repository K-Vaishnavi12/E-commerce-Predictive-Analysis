# Will They Buy? E-Commerce Conversion Prediction

Solution for the **Summer Analytics 2026 - Week 2 Hackathon** (E-Commerce Conversion
Prediction). Given user behaviour and session data for an online store, predict
whether a user will convert (make a purchase) or not. Evaluation metric is **F1
score** on a held-out private test set.

## What this project does

- Loads the train / public_test / private_test CSVs.
- Does a quick EDA (target balance, missing values, conversion rate by category,
  income floor at 12000, etc.).
- Builds 37 features: missingness flags, log-scaled income/time, engagement
  ratios, discount interactions, age/income buckets, campaign bucket.
- Trains an ensemble of three gradient-boosted models with 5-fold stratified CV
  repeated across 3 seeds:
  - LightGBM
  - XGBoost (with `enable_categorical=True`)
  - CatBoost
- Equal-weight blend (1/3 each) of the three models.
- Picks the F1-optimal probability threshold from out-of-fold predictions
  (lands around `t = 0.27`).
- Refits on `train + public_test` for the final private prediction.
- Writes `submission.csv` in the required `User_ID, Converted` format.

Final scores (full out-of-fold on train + public):

| Metric | Value |
|---|---|
| OOF AUC | ~0.730 |
| OOF F1  | ~0.566 |
| Public test F1 | ~0.547 |

## Tech stack

- **Language**: Python 3.11
- **Notebook**: Jupyter (`BuildSubmission.ipynb`)
- **Data**: `pandas`, `numpy`
- **Modelling**: `scikit-learn` (CV, metrics), `lightgbm`, `xgboost`, `catboost`

Tested versions:

```
python       3.11.9
pandas       2.3.3
numpy        2.4.0
scikit-learn 1.8.0
lightgbm     4.6.0
xgboost      3.2.0
catboost     1.2.10
```

## Project structure

```
.
|-- BuildSubmission.ipynb      <- main notebook (EDA + features + models + submission)
|-- pipeline_final.py          <- script form of the same pipeline (faster to run)
|-- pipeline.py / pipeline_v2.py <- earlier iterations, kept for reference
|-- eda.py                     <- standalone EDA script
|-- threshold_study.py         <- F1-vs-threshold sweep
|-- test_cb.py                 <- catboost-only sanity check
|-- train.csv                  <- 10 000 rows, labelled
|-- public_test.csv            <- 3 000 rows, labelled (used for validation + refit)
|-- private_test.csv           <- 3 000 rows, unlabelled (the target of the submission)
|-- sample_submission.csv      <- format reference
|-- submission.csv             <- final predictions (User_ID, Converted)
|-- submission_proba.csv       <- raw probabilities (diagnostic only, do not upload)
|-- cb_probs.npz               <- cached catboost probs from test_cb.py
`-- README.md
```

## How to run

### 1. Install Python 3.11

Get it from <https://www.python.org/downloads/> if you don't have it.

### 2. Install the dependencies

From the project folder:

```bash
pip install pandas numpy scikit-learn lightgbm xgboost catboost jupyter nbconvert ipykernel
```

If you'd rather pin to the tested versions:

```bash
pip install pandas==2.3.3 numpy==2.4.0 scikit-learn==1.8.0 lightgbm==4.6.0 xgboost==3.2.0 catboost==1.2.10 jupyter nbconvert ipykernel
```

### 3. Make sure the data files are in place

The four CSVs (`train.csv`, `public_test.csv`, `private_test.csv`,
`sample_submission.csv`) need to be in the same folder as the notebook.

### 4a. Run the notebook (recommended)

Open it in Jupyter / VS Code and run the cells top to bottom:

```bash
jupyter notebook BuildSubmission.ipynb
```

Or execute it headlessly from the terminal (writes outputs back into the
notebook):

```bash
jupyter nbconvert --to notebook --execute BuildSubmission.ipynb --inplace --ExecutePreprocessor.timeout=2400
```

Expect roughly **20-30 minutes** of total runtime end-to-end (CV pass + refit
pass across 3 models x 3 seeds x 5 folds).

### 4b. Run the script instead (faster)

If you just want to regenerate `submission.csv` without the notebook overhead:

```bash
python pipeline_final.py
```

This takes about **5 minutes** and produces the same `submission.csv`.

### 5. Check the output

After running, you should see a fresh `submission.csv` in the folder:

- 3000 rows (one per `User_ID` in `private_test.csv`)
- 2 columns exactly: `User_ID, Converted`
- `Converted` is binary (0 or 1)
- ~51% predicted positives at the chosen F1-optimal threshold

The last cell of the notebook runs an `assert`-based format check against
`sample_submission.csv` to confirm everything is correct.

## Notes

- The pipeline uses tree-based models, so missing values in `Age`, `Income`,
  and `Time_On_Site` are passed through as-is (no imputation needed for the
  model itself, only for derived features).
- `Browser_Version`, `Campaign_Code`, and `City_Tier` are kept numeric.
  `Device_Type` and `Traffic_Source` are the only true categoricals.
- The threshold (~0.27) is below 0.5 because the data is imbalanced (31%
  positive) and we're optimising F1, which prefers recall over precision here.
- Public test labels are used both for monitoring and for the final refit -
  this is allowed by the problem statement ("Participants may use this dataset
  for validation and experimentation.").
