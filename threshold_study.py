"""Quick threshold sweep to study precision/recall trade-off."""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

# Load probabilities saved by the pipeline
proba = pd.read_csv("submission_proba.csv")["proba"].values

# Re-run a CV to get OOF probs
import lightgbm as lgb, xgboost as xgb
from sklearn.model_selection import StratifiedKFold

# Reload data
exec(open("pipeline.py", encoding="utf-8").read().split("# -------------------- 3.")[0])

# Quickly retrain just for OOF probabilities
y = train["Converted"].astype(int)
X = train[FEATURES]
y_pub = public["Converted"].astype(int)
X_pub = public[FEATURES]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(X))
pub = np.zeros(len(X_pub))

params = dict(objective="binary", metric="binary_logloss", learning_rate=0.03,
              num_leaves=63, min_data_in_leaf=40, feature_fraction=0.85,
              bagging_fraction=0.85, bagging_freq=5, lambda_l2=1.0,
              verbose=-1, seed=42)
for tr, va in skf.split(X, y):
    dtr = lgb.Dataset(X.iloc[tr], label=y.iloc[tr], categorical_feature=CAT_COLS)
    dva = lgb.Dataset(X.iloc[va], label=y.iloc[va], categorical_feature=CAT_COLS, reference=dtr)
    m = lgb.train(params, dtr, num_boost_round=4000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
    oof[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
    pub += m.predict(X_pub, num_iteration=m.best_iteration) / 5

print(f"OOF AUC: {roc_auc_score(y, oof):.4f}")
print(f"PUB AUC: {roc_auc_score(y_pub, pub):.4f}")

print(f"\n{'thr':>5} {'oof_f1':>8} {'oof_p':>7} {'oof_r':>7} {'pub_f1':>8} {'pub_p':>7} {'pub_r':>7} {'oof_pos%':>8} {'pub_pos%':>8}")
for t in np.arange(0.20, 0.55, 0.025):
    yo = (oof >= t).astype(int)
    yp = (pub >= t).astype(int)
    print(f"{t:5.3f} "
          f"{f1_score(y, yo):8.4f} {precision_score(y, yo, zero_division=0):7.4f} {recall_score(y, yo):7.4f} "
          f"{f1_score(y_pub, yp):8.4f} {precision_score(y_pub, yp, zero_division=0):7.4f} {recall_score(y_pub, yp):7.4f} "
          f"{yo.mean():8.3%} {yp.mean():8.3%}")
