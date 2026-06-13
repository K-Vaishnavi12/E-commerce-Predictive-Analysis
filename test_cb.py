"""Compare adding CatBoost to the ensemble."""
import warnings, time
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier, Pool

warnings.filterwarnings("ignore")

# Reload preprocessing from pipeline_v2
import importlib.util
spec = importlib.util.spec_from_file_location("pv2", "pipeline_v2.py")
# We don't want to execute the full pipeline; copy add_features here
exec(open("pipeline_v2.py").read().split("def best_threshold")[0])

# After running the file's preprocessing block, X, y, X_pub, X_prv, FEATURES, CAT_COLS are in scope
print("Setup loaded. Now training CatBoost only.")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(X)); pub = np.zeros(len(X_pub)); prv = np.zeros(len(X_prv))

# CatBoost wants int indexes for cat features
cat_idx = [X.columns.get_loc(c) for c in CAT_COLS]

# Convert pandas Categorical to plain string for CatBoost
def to_cb(df):
    df = df.copy()
    for c in CAT_COLS:
        df[c] = df[c].astype(str)
    return df

Xc = to_cb(X); Xcp = to_cb(X_pub); Xcr = to_cb(X_prv)

for fold, (tr, va) in enumerate(skf.split(Xc, y)):
    train_pool = Pool(Xc.iloc[tr], y.iloc[tr], cat_features=cat_idx)
    val_pool = Pool(Xc.iloc[va], y.iloc[va], cat_features=cat_idx)
    m = CatBoostClassifier(
        iterations=4000, learning_rate=0.03, depth=6,
        l2_leaf_reg=5.0, random_seed=42,
        eval_metric="Logloss", verbose=False,
        early_stopping_rounds=200,
    )
    m.fit(train_pool, eval_set=val_pool)
    oof[va] = m.predict_proba(val_pool)[:, 1]
    pub += m.predict_proba(Pool(Xcp, cat_features=cat_idx))[:, 1] / 5
    prv += m.predict_proba(Pool(Xcr, cat_features=cat_idx))[:, 1] / 5
    print(f"fold {fold + 1} best={m.best_iteration_}  val_auc={roc_auc_score(y.iloc[va], oof[va]):.4f}")

print(f"\nCB OOF AUC: {roc_auc_score(y, oof):.5f}")
print(f"CB PUB AUC: {roc_auc_score(y_pub, pub):.5f}")

# Threshold sweep
def sweep(name, ot, op, yt, yp):
    print(f"\n{name}: best OOF threshold scan")
    for t in np.arange(0.22, 0.42, 0.02):
        f_o = f1_score(yt, (ot >= t).astype(int))
        f_p = f1_score(yp, (op >= t).astype(int))
        print(f"  t={t:.2f}  oof_f1={f_o:.4f}  pub_f1={f_p:.4f}")

sweep("CatBoost", oof, pub, y, y_pub)

# Save CB probs for ensemble
np.savez("cb_probs.npz", oof=oof, pub=pub, prv=prv)
print("Saved cb_probs.npz")
