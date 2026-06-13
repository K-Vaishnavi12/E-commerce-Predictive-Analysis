"""
E-Commerce Conversion Prediction - Summer Analytics 2026 Hackathon (Week 2)

Pipeline:
  1. Load data
  2. Preprocess + feature engineering
  3. Stratified K-fold LightGBM + XGBoost with OOF F1 threshold tuning
  4. Validate on public_test
  5. Refit on train + public_test, predict private_test, write submission.csv
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

DATA_DIR = Path(".")
RANDOM_STATE = 42
N_FOLDS = 5

# -------------------- 1. LOAD --------------------
train = pd.read_csv(DATA_DIR / "train.csv")
public = pd.read_csv(DATA_DIR / "public_test.csv")
private = pd.read_csv(DATA_DIR / "private_test.csv")

print(f"train={train.shape}, public={public.shape}, private={private.shape}")
print(f"train target rate: {train['Converted'].mean():.4f}")
print(f"public target rate: {public['Converted'].mean():.4f}")


# -------------------- 2. FEATURE ENGINEERING --------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Missingness flags (signal often correlates with conversion)
    df["age_missing"] = df["Age"].isna().astype(int)
    df["income_missing"] = df["Income"].isna().astype(int)
    df["time_missing"] = df["Time_On_Site"].isna().astype(int)
    df["n_missing"] = df[["Age", "Income", "Time_On_Site"]].isna().sum(axis=1)

    # Income floor flag (12000 looks like a clipped value)
    df["income_floor"] = (df["Income"] == 12000.0).astype(int)

    # Log-scale features for skewed columns
    df["log_income"] = np.log1p(df["Income"].fillna(df["Income"].median()))
    df["log_time"] = np.log1p(df["Time_On_Site"].fillna(df["Time_On_Site"].median()))

    # Engagement ratios / interactions
    df["products_per_page"] = df["Products_Viewed"] / (df["Pages_Viewed"] + 1)
    df["pages_minus_products"] = df["Pages_Viewed"] - df["Products_Viewed"]
    df["time_per_page"] = df["Time_On_Site"].fillna(df["Time_On_Site"].median()) / (
        df["Pages_Viewed"] + 1
    )
    df["time_per_product"] = df["Time_On_Site"].fillna(df["Time_On_Site"].median()) / (
        df["Products_Viewed"] + 1
    )
    df["engagement"] = df["Pages_Viewed"] + df["Products_Viewed"]
    df["engagement_x_discount"] = df["engagement"] * df["Discount_Seen"]
    df["prev_x_discount"] = df["Previous_Purchases"] * df["Discount_Seen"]

    # Age buckets (LGBM doesn't need this but it can help linear models / interpretation)
    df["age_bucket"] = pd.cut(
        df["Age"].fillna(-1),
        bins=[-2, 0, 25, 35, 45, 55, 100],
        labels=[-1, 0, 1, 2, 3, 4],
    ).astype(int)

    # Income buckets
    df["income_bucket"] = pd.qcut(
        df["Income"].fillna(df["Income"].median()), q=10, labels=False, duplicates="drop"
    )

    # Campaign code grouping (high cardinality -> bucket by thousand)
    df["Campaign_Bucket"] = (df["Campaign_Code"] // 1000).astype(int)

    return df


def encode_categoricals(
    train_df: pd.DataFrame, *other_dfs: pd.DataFrame, cat_cols: list[str]
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """Convert string categoricals to pandas Categorical with shared categories
    (LightGBM consumes Categorical dtype natively)."""
    train_df = train_df.copy()
    others = [df.copy() for df in other_dfs]

    for col in cat_cols:
        # union of categories across all splits to keep encoding consistent
        all_vals = pd.concat([train_df[col]] + [d[col] for d in others]).astype(str)
        cats = pd.Index(all_vals.dropna().unique())
        train_df[col] = pd.Categorical(train_df[col].astype(str), categories=cats)
        for d in others:
            d[col] = pd.Categorical(d[col].astype(str), categories=cats)

    return train_df, others


train = add_features(train)
public = add_features(public)
private = add_features(private)

CAT_COLS = ["Device_Type", "Traffic_Source"]
train, (public, private) = encode_categoricals(train, public, private, cat_cols=CAT_COLS)

DROP = ["User_ID", "Converted"]
FEATURES = [c for c in train.columns if c not in DROP]
print(f"\n{len(FEATURES)} features:\n{FEATURES}")

X = train[FEATURES]
y = train["Converted"].astype(int)
X_pub = public[FEATURES]
y_pub = public["Converted"].astype(int)
X_prv = private[FEATURES]


# -------------------- 3. CV TRAINING --------------------
def best_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[float, float]:
    """Find probability threshold that maximises F1."""
    thresholds = np.linspace(0.05, 0.95, 181)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        f1 = f1_score(y_true, (y_proba >= t).astype(int))
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t, best_f1


def train_lgbm_oof(
    X: pd.DataFrame,
    y: pd.Series,
    X_holdout: pd.DataFrame,
    X_test: pd.DataFrame,
    n_folds: int = N_FOLDS,
    seed: int = RANDOM_STATE,
):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(X))
    holdout_preds = np.zeros(len(X_holdout))
    test_preds = np.zeros(len(X_test))

    params = dict(
        objective="binary",
        metric="binary_logloss",
        learning_rate=0.03,
        num_leaves=63,
        min_data_in_leaf=40,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=5,
        lambda_l2=1.0,
        verbose=-1,
        seed=seed,
    )

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        dtr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=CAT_COLS)
        dva = lgb.Dataset(X_va, label=y_va, categorical_feature=CAT_COLS, reference=dtr)

        model = lgb.train(
            params,
            dtr,
            num_boost_round=4000,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)],
        )

        oof[va_idx] = model.predict(X_va, num_iteration=model.best_iteration)
        holdout_preds += model.predict(X_holdout, num_iteration=model.best_iteration) / n_folds
        test_preds += model.predict(X_test, num_iteration=model.best_iteration) / n_folds

        fold_auc = roc_auc_score(y_va, oof[va_idx])
        print(f"  [LGBM fold {fold + 1}/{n_folds}] best_iter={model.best_iteration:4d}  val_auc={fold_auc:.5f}")

    return oof, holdout_preds, test_preds


def train_xgb_oof(
    X: pd.DataFrame,
    y: pd.Series,
    X_holdout: pd.DataFrame,
    X_test: pd.DataFrame,
    n_folds: int = N_FOLDS,
    seed: int = RANDOM_STATE,
):
    """XGBoost with native categorical support (enable_categorical=True)."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(X))
    holdout_preds = np.zeros(len(X_holdout))
    test_preds = np.zeros(len(X_test))

    params = dict(
        objective="binary:logistic",
        eval_metric="logloss",
        eta=0.03,
        max_depth=6,
        min_child_weight=5,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        tree_method="hist",
        enable_categorical=True,
        random_state=seed,
        verbosity=0,
    )

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        dtr = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
        dva = xgb.DMatrix(X_va, label=y_va, enable_categorical=True)
        dho = xgb.DMatrix(X_holdout, enable_categorical=True)
        dts = xgb.DMatrix(X_test, enable_categorical=True)

        model = xgb.train(
            params,
            dtr,
            num_boost_round=4000,
            evals=[(dva, "val")],
            early_stopping_rounds=150,
            verbose_eval=False,
        )

        oof[va_idx] = model.predict(dva, iteration_range=(0, model.best_iteration + 1))
        holdout_preds += model.predict(dho, iteration_range=(0, model.best_iteration + 1)) / n_folds
        test_preds += model.predict(dts, iteration_range=(0, model.best_iteration + 1)) / n_folds

        fold_auc = roc_auc_score(y_va, oof[va_idx])
        print(f"  [XGB  fold {fold + 1}/{n_folds}] best_iter={model.best_iteration:4d}  val_auc={fold_auc:.5f}")

    return oof, holdout_preds, test_preds


t0 = time.time()
print("\n--- LightGBM ---")
oof_lgb, pub_lgb, prv_lgb = train_lgbm_oof(X, y, X_pub, X_prv)
print("\n--- XGBoost ---")
oof_xgb, pub_xgb, prv_xgb = train_xgb_oof(X, y, X_pub, X_prv)
print(f"\nCV training time: {time.time() - t0:.1f}s")

# -------------------- 4. ENSEMBLE + THRESHOLD --------------------
oof = 0.5 * oof_lgb + 0.5 * oof_xgb
pub_proba = 0.5 * pub_lgb + 0.5 * pub_xgb
prv_proba = 0.5 * prv_lgb + 0.5 * prv_xgb

oof_auc = roc_auc_score(y, oof)
print(f"\nOOF AUC (ensemble): {oof_auc:.5f}")
print(f"OOF AUC (LGBM):     {roc_auc_score(y, oof_lgb):.5f}")
print(f"OOF AUC (XGB):      {roc_auc_score(y, oof_xgb):.5f}")

# Find threshold maximising OOF F1
best_t, best_f1 = best_threshold(y.values, oof)
print(f"\nBest OOF threshold = {best_t:.3f}  ->  OOF F1 = {best_f1:.5f}")

# Validate on public_test
pub_pred = (pub_proba >= best_t).astype(int)
pub_f1 = f1_score(y_pub, pub_pred)
pub_auc = roc_auc_score(y_pub, pub_proba)
print(f"\nPublic_test  AUC = {pub_auc:.5f}")
print(f"Public_test  F1  = {pub_f1:.5f}  (threshold={best_t})")
print("\nClassification report on public_test:")
print(classification_report(y_pub, pub_pred, digits=4))

# Sanity check: also tune threshold on public to see ceiling
pub_t, pub_best_f1 = best_threshold(y_pub.values, pub_proba)
print(f"(Public-best threshold {pub_t:.3f} would give F1 {pub_best_f1:.5f})")


# -------------------- 5. REFIT ON TRAIN+PUBLIC, PREDICT PRIVATE --------------------
print("\n--- Refitting on train + public_test ---")
X_full = pd.concat([X, X_pub], ignore_index=True)
y_full = pd.concat([y, y_pub], ignore_index=True)

# Reuse same CV scheme on the combined data to train final ensemble
oof_full_lgb, _, prv_full_lgb = train_lgbm_oof(X_full, y_full, X_full.iloc[:1], X_prv)
oof_full_xgb, _, prv_full_xgb = train_xgb_oof(X_full, y_full, X_full.iloc[:1], X_prv)

prv_proba_final = 0.5 * prv_full_lgb + 0.5 * prv_full_xgb
oof_full = 0.5 * oof_full_lgb + 0.5 * oof_full_xgb
final_t, final_oof_f1 = best_threshold(y_full.values, oof_full)
print(f"\nFinal OOF (train+public) F1 = {final_oof_f1:.5f}  at threshold {final_t:.3f}")
print(f"Final OOF (train+public) AUC = {roc_auc_score(y_full, oof_full):.5f}")

# Use the threshold tuned on full OOF
final_preds = (prv_proba_final >= final_t).astype(int)
print(f"\nPrivate predictions: positives = {final_preds.sum()} / {len(final_preds)}  "
      f"({final_preds.mean():.3%})")

submission = pd.DataFrame(
    {"User_ID": private["User_ID"].astype(int), "Converted": final_preds.astype(int)}
)
submission.to_csv(DATA_DIR / "submission.csv", index=False)
print(f"\nWrote submission.csv  shape={submission.shape}")
print(submission.head())

# Save probabilities too for reference / possible later threshold-tuning
pd.DataFrame(
    {
        "User_ID": private["User_ID"].astype(int),
        "proba": prv_proba_final,
    }
).to_csv(DATA_DIR / "submission_proba.csv", index=False)
print("Wrote submission_proba.csv (probabilities for reference, not for upload)")
