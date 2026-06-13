"""
Final pipeline: LightGBM + XGBoost + CatBoost ensemble with multi-seed averaging,
F1-threshold tuning, and refit on train + public_test.

Outputs: submission.csv (and submission_proba.csv for diagnostics).
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

DATA_DIR = Path(".")
N_FOLDS = 5
SEEDS = [42, 7, 2024]


# -------------------- 1. LOAD --------------------
train = pd.read_csv(DATA_DIR / "train.csv")
public = pd.read_csv(DATA_DIR / "public_test.csv")
private = pd.read_csv(DATA_DIR / "private_test.csv")
print(f"train={train.shape}, public={public.shape}, private={private.shape}")
print(f"train target rate: {train['Converted'].mean():.4f}, public: {public['Converted'].mean():.4f}")


# -------------------- 2. FEATURE ENGINEERING --------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Missingness flags
    df["age_missing"] = df["Age"].isna().astype(int)
    df["income_missing"] = df["Income"].isna().astype(int)
    df["time_missing"] = df["Time_On_Site"].isna().astype(int)
    df["n_missing"] = df[["Age", "Income", "Time_On_Site"]].isna().sum(axis=1)

    df["income_floor"] = (df["Income"] == 12000.0).astype(int)
    income_med = df["Income"].median()
    time_med = df["Time_On_Site"].median()
    age_med = df["Age"].median()

    df["log_income"] = np.log1p(df["Income"].fillna(income_med))
    df["log_time"] = np.log1p(df["Time_On_Site"].fillna(time_med))

    pages = df["Pages_Viewed"].astype(float)
    products = df["Products_Viewed"].astype(float)
    time_filled = df["Time_On_Site"].fillna(time_med).astype(float)
    age_filled = df["Age"].fillna(age_med).astype(float)
    inc_filled = df["Income"].fillna(income_med).astype(float)
    prev = df["Previous_Purchases"].astype(float)
    disc = df["Discount_Seen"].astype(float)

    df["products_per_page"] = products / (pages + 1)
    df["pages_minus_products"] = pages - products
    df["time_per_page"] = time_filled / (pages + 1)
    df["time_per_product"] = time_filled / (products + 1)
    df["engagement"] = pages + products
    df["engagement_x_discount"] = df["engagement"] * disc
    df["prev_x_discount"] = prev * disc
    df["pages_x_discount"] = pages * disc
    df["products_x_discount"] = products * disc
    df["log_time_x_discount"] = df["log_time"] * disc
    df["age_x_income"] = age_filled * np.log1p(inc_filled) / 100
    df["prev_per_age"] = prev / (age_filled + 1)
    df["high_engagement"] = ((pages > 20) & (products > 20)).astype(int)
    df["low_engagement"] = ((pages < 5) & (products < 5)).astype(int)
    df["time_outlier"] = (time_filled > 100).astype(int)

    df["age_bucket"] = pd.cut(
        df["Age"].fillna(-1),
        bins=[-2, 0, 25, 35, 45, 55, 100],
        labels=[-1, 0, 1, 2, 3, 4],
    ).astype(int)
    df["income_bucket"] = pd.qcut(
        df["Income"].fillna(income_med), q=10, labels=False, duplicates="drop"
    )
    df["Campaign_Bucket"] = (df["Campaign_Code"] // 1000).astype(int)
    return df


def encode_categoricals(train_df, *other_dfs, cat_cols):
    train_df = train_df.copy()
    others = [df.copy() for df in other_dfs]
    for col in cat_cols:
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
print(f"# features: {len(FEATURES)}")

X = train[FEATURES]
y = train["Converted"].astype(int)
X_pub = public[FEATURES]
y_pub = public["Converted"].astype(int)
X_prv = private[FEATURES]


# -------------------- helpers --------------------
def best_threshold(y_true, y_proba):
    thresholds = np.linspace(0.05, 0.95, 361)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        f1 = f1_score(y_true, (y_proba >= t).astype(int))
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t, best_f1


def lgb_oof(X, y, X_pub, X_prv, seed):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    oof = np.zeros(len(X)); pub = np.zeros(len(X_pub)); prv = np.zeros(len(X_prv))
    params = dict(objective="binary", metric="binary_logloss",
                  learning_rate=0.025, num_leaves=47, min_data_in_leaf=50,
                  feature_fraction=0.8, bagging_fraction=0.85, bagging_freq=5,
                  lambda_l1=0.5, lambda_l2=2.0, verbose=-1, seed=seed)
    for tr, va in skf.split(X, y):
        dtr = lgb.Dataset(X.iloc[tr], label=y.iloc[tr], categorical_feature=CAT_COLS)
        dva = lgb.Dataset(X.iloc[va], label=y.iloc[va], categorical_feature=CAT_COLS, reference=dtr)
        m = lgb.train(params, dtr, num_boost_round=6000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
        oof[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
        pub += m.predict(X_pub, num_iteration=m.best_iteration) / N_FOLDS
        prv += m.predict(X_prv, num_iteration=m.best_iteration) / N_FOLDS
    return oof, pub, prv


def xgb_oof(X, y, X_pub, X_prv, seed):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    oof = np.zeros(len(X)); pub = np.zeros(len(X_pub)); prv = np.zeros(len(X_prv))
    params = dict(objective="binary:logistic", eval_metric="logloss",
                  eta=0.025, max_depth=5, min_child_weight=8,
                  subsample=0.85, colsample_bytree=0.8,
                  reg_alpha=0.5, reg_lambda=2.0,
                  tree_method="hist", enable_categorical=True,
                  random_state=seed, verbosity=0)
    for tr, va in skf.split(X, y):
        dtr = xgb.DMatrix(X.iloc[tr], label=y.iloc[tr], enable_categorical=True)
        dva = xgb.DMatrix(X.iloc[va], label=y.iloc[va], enable_categorical=True)
        dpb = xgb.DMatrix(X_pub, enable_categorical=True)
        dpv = xgb.DMatrix(X_prv, enable_categorical=True)
        m = xgb.train(params, dtr, num_boost_round=6000, evals=[(dva, "v")],
                      early_stopping_rounds=200, verbose_eval=False)
        it = (0, m.best_iteration + 1)
        oof[va] = m.predict(dva, iteration_range=it)
        pub += m.predict(dpb, iteration_range=it) / N_FOLDS
        prv += m.predict(dpv, iteration_range=it) / N_FOLDS
    return oof, pub, prv


def cat_oof(X, y, X_pub, X_prv, seed):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    oof = np.zeros(len(X)); pub = np.zeros(len(X_pub)); prv = np.zeros(len(X_prv))
    cat_idx = [X.columns.get_loc(c) for c in CAT_COLS]

    def to_cb(df):
        df = df.copy()
        for c in CAT_COLS:
            df[c] = df[c].astype(str)
        return df

    Xc, Xcp, Xcr = to_cb(X), to_cb(X_pub), to_cb(X_prv)
    for tr, va in skf.split(Xc, y):
        train_pool = Pool(Xc.iloc[tr], y.iloc[tr], cat_features=cat_idx)
        val_pool = Pool(Xc.iloc[va], y.iloc[va], cat_features=cat_idx)
        m = CatBoostClassifier(
            iterations=4000, learning_rate=0.03, depth=6,
            l2_leaf_reg=5.0, random_seed=seed,
            eval_metric="Logloss", verbose=False,
            early_stopping_rounds=200,
        )
        m.fit(train_pool, eval_set=val_pool)
        oof[va] = m.predict_proba(val_pool)[:, 1]
        pub += m.predict_proba(Pool(Xcp, cat_features=cat_idx))[:, 1] / N_FOLDS
        prv += m.predict_proba(Pool(Xcr, cat_features=cat_idx))[:, 1] / N_FOLDS
    return oof, pub, prv


# -------------------- 3. CV TRAINING (multi-seed) --------------------
t0 = time.time()
results = {"lgb": None, "xgb": None, "cat": None}

for name, fn in [("lgb", lgb_oof), ("xgb", xgb_oof), ("cat", cat_oof)]:
    print(f"\n=== {name.upper()} multi-seed ===")
    oof_acc = np.zeros(len(X)); pub_acc = np.zeros(len(X_pub)); prv_acc = np.zeros(len(X_prv))
    for s in SEEDS:
        oo, pp, rr = fn(X, y, X_pub, X_prv, seed=s)
        oof_acc += oo / len(SEEDS); pub_acc += pp / len(SEEDS); prv_acc += rr / len(SEEDS)
        print(f"  seed={s}  oof_auc={roc_auc_score(y, oo):.4f}  pub_auc={roc_auc_score(y_pub, pp):.4f}")
    results[name] = (oof_acc, pub_acc, prv_acc)

print(f"\nTotal CV training time: {time.time() - t0:.1f}s")

# -------------------- 4. ENSEMBLE --------------------
oof_lgb, pub_lgb, prv_lgb = results["lgb"]
oof_xgb, pub_xgb, prv_xgb = results["xgb"]
oof_cat, pub_cat, prv_cat = results["cat"]

# Equal-weight ensemble (CatBoost is slightly stronger but equal weights are more robust)
W = (1/3, 1/3, 1/3)
oof = W[0] * oof_lgb + W[1] * oof_xgb + W[2] * oof_cat
pub_proba = W[0] * pub_lgb + W[1] * pub_xgb + W[2] * pub_cat
prv_proba = W[0] * prv_lgb + W[1] * prv_xgb + W[2] * prv_cat

print(f"\nOOF AUC  LGB={roc_auc_score(y, oof_lgb):.5f}  XGB={roc_auc_score(y, oof_xgb):.5f}  CAT={roc_auc_score(y, oof_cat):.5f}  ENS={roc_auc_score(y, oof):.5f}")
print(f"PUB AUC  LGB={roc_auc_score(y_pub, pub_lgb):.5f}  XGB={roc_auc_score(y_pub, pub_xgb):.5f}  CAT={roc_auc_score(y_pub, pub_cat):.5f}  ENS={roc_auc_score(y_pub, pub_proba):.5f}")

oof_t, oof_f1 = best_threshold(y.values, oof)
pub_t, pub_f1 = best_threshold(y_pub.values, pub_proba)
print(f"\nOOF best threshold = {oof_t:.3f}  ->  OOF F1 = {oof_f1:.5f}")
print(f"PUB best threshold = {pub_t:.3f}  ->  PUB F1 = {pub_f1:.5f}")

# Threshold sweep
print(f"\n{'thr':>5} {'oof_f1':>8} {'pub_f1':>8} {'oof_pos%':>9} {'pub_pos%':>9}")
for t in np.arange(0.20, 0.42, 0.02):
    yo = (oof >= t).astype(int); yp = (pub_proba >= t).astype(int)
    print(f"{t:5.3f} {f1_score(y, yo):8.4f} {f1_score(y_pub, yp):8.4f} {yo.mean():9.3%} {yp.mean():9.3%}")

# -------------------- 5. REFIT ON TRAIN+PUBLIC, PREDICT PRIVATE --------------------
print("\n=== Refit on train + public_test ===")
X_full = pd.concat([X, X_pub], ignore_index=True)
y_full = pd.concat([y, y_pub], ignore_index=True)
final_lgb_prv = np.zeros(len(X_prv)); final_xgb_prv = np.zeros(len(X_prv)); final_cat_prv = np.zeros(len(X_prv))
final_lgb_oof = np.zeros(len(X_full)); final_xgb_oof = np.zeros(len(X_full)); final_cat_oof = np.zeros(len(X_full))

for s in SEEDS:
    oo, _, rr = lgb_oof(X_full, y_full, X_full.iloc[:1], X_prv, seed=s)
    final_lgb_oof += oo / len(SEEDS); final_lgb_prv += rr / len(SEEDS)
    oo, _, rr = xgb_oof(X_full, y_full, X_full.iloc[:1], X_prv, seed=s)
    final_xgb_oof += oo / len(SEEDS); final_xgb_prv += rr / len(SEEDS)
    oo, _, rr = cat_oof(X_full, y_full, X_full.iloc[:1], X_prv, seed=s)
    final_cat_oof += oo / len(SEEDS); final_cat_prv += rr / len(SEEDS)

oof_full = W[0] * final_lgb_oof + W[1] * final_xgb_oof + W[2] * final_cat_oof
prv_final = W[0] * final_lgb_prv + W[1] * final_xgb_prv + W[2] * final_cat_prv
final_t, final_f1 = best_threshold(y_full.values, oof_full)
print(f"\nFull OOF AUC = {roc_auc_score(y_full, oof_full):.5f}")
print(f"Full OOF F1  = {final_f1:.5f}  at threshold {final_t:.3f}")

# Sanity: show how the chosen threshold behaves on train_only (more honest estimate of real F1)
print(f"\nTrain-only OOF F1 at threshold {final_t:.3f}: {f1_score(y, (oof >= final_t).astype(int)):.5f}")
print(f"Public_test    F1 at threshold {final_t:.3f}: {f1_score(y_pub, (pub_proba >= final_t).astype(int)):.5f}")

preds = (prv_final >= final_t).astype(int)
print(f"\nPrivate predictions: positives = {preds.sum()} / {len(preds)} ({preds.mean():.3%})")

submission = pd.DataFrame({"User_ID": private["User_ID"].astype(int),
                           "Converted": preds.astype(int)})
submission.to_csv(DATA_DIR / "submission.csv", index=False)
print(f"\nWrote submission.csv shape={submission.shape}")
print(submission.head())

pd.DataFrame({"User_ID": private["User_ID"].astype(int),
              "proba": prv_final}).to_csv(DATA_DIR / "submission_proba.csv", index=False)
print("Wrote submission_proba.csv (probabilities for diagnostics)")
