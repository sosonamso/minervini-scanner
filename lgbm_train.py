"""
LightGBM Walk-Forward 학습
- lgbm_raw.csv 로드
- Walk-Forward 4 Fold 검증
- 최종 모델 저장
"""
import os, pickle, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

FOLDS = [
    ("2021-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
    ("2021-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
    ("2021-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
    ("2021-01-01", "2025-12-31", "2026-01-01", "2026-12-31"),
]

LGB_PARAMS = {
    "objective":        "binary",
    "metric":           "auc",
    "boosting_type":    "gbdt",
    "num_leaves":       31,
    "learning_rate":    0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq":     5,
    "min_child_samples":20,
    "verbose":          -1,
    "n_jobs":           -1,
}


def get_feat_cols(df):
    skip = {"ticker", "date", "label", "entry"}
    return [c for c in df.columns if c not in skip]


if __name__ == "__main__":
    # 데이터 로드
    df = pd.read_csv("lgbm_raw.csv", encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"로드: {len(df)}건 (양성:{(df.label==1).sum()} 음성:{(df.label==0).sum()})")
    print(f"날짜: {df.date.min().date()} ~ {df.date.max().date()}")

    FEAT_COLS = get_feat_cols(df)
    print(f"피처 수: {len(FEAT_COLS)}")

    # ── Walk-Forward Validation ────────────────────────
    fold_aucs = []
    feat_imp_list = []

    print(f"\n{'='*55}")
    print("[ LightGBM Walk-Forward Validation ]")
    print(f"{'='*55}")

    for i, (tr_start, tr_end, te_start, te_end) in enumerate(FOLDS):
        tr = df[(df.date >= tr_start) & (df.date <= tr_end)]
        te = df[(df.date >= te_start) & (df.date <= te_end)]

        if len(tr) < 50 or len(te) < 20:
            print(f"Fold {i+1}: 데이터 부족 스킵 (train:{len(tr)} test:{len(te)})")
            continue

        X_tr = tr[FEAT_COLS].values.astype(np.float32)
        y_tr = tr["label"].values
        X_te = te[FEAT_COLS].values.astype(np.float32)
        y_te = te["label"].values

        print(f"\nFold {i+1}: train {tr_start[:4]}~{tr_end[:4]} ({len(tr)}건) | test {te_start[:4]} ({len(te)}건)")
        print(f"  양성비율 - train:{y_tr.mean():.2f} test:{y_te.mean():.2f}")

        dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=FEAT_COLS)
        dvalid = lgb.Dataset(X_te, label=y_te, reference=dtrain)

        callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
        model = lgb.train(
            LGB_PARAMS,
            dtrain,
            num_boost_round=500,
            valid_sets=[dvalid],
            callbacks=callbacks,
        )

        pred = model.predict(X_te)
        auc  = roc_auc_score(y_te, pred)
        fold_aucs.append(auc)
        print(f"  → Fold {i+1} AUC: {auc:.4f} (best iter: {model.best_iteration})")

        # 피처 중요도 수집
        imp = pd.Series(model.feature_importance(importance_type="gain"),
                        index=FEAT_COLS)
        feat_imp_list.append(imp)

    print(f"\n평균 AUC: {np.mean(fold_aucs):.4f} (±{np.std(fold_aucs):.4f})")
    print(f"Fold별: {[round(a,4) for a in fold_aucs]}")

    # 피처 중요도 상위 20개
    if feat_imp_list:
        avg_imp = pd.concat(feat_imp_list, axis=1).mean(axis=1).sort_values(ascending=False)
        print(f"\n피처 중요도 Top 20:")
        for feat, val in avg_imp.head(20).items():
            print(f"  {feat:30s} {val:.1f}")

    # ── 최종 모델 학습 (전체 데이터) ──────────────────
    print("\n최종 모델 학습 중...")
    X_all = df[FEAT_COLS].values.astype(np.float32)
    y_all = df["label"].values

    # 마지막 10% validation (Early Stopping용)
    n_val    = int(len(df) * 0.1)
    X_tr_f   = X_all[:-n_val]; y_tr_f = y_all[:-n_val]
    X_val_f  = X_all[-n_val:]; y_val_f = y_all[-n_val:]

    dtrain_f = lgb.Dataset(X_tr_f, label=y_tr_f, feature_name=FEAT_COLS)
    dvalid_f = lgb.Dataset(X_val_f, label=y_val_f, reference=dtrain_f)

    callbacks_f = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(50)]
    final_model = lgb.train(
        LGB_PARAMS,
        dtrain_f,
        num_boost_round=1000,
        valid_sets=[dvalid_f],
        callbacks=callbacks_f,
    )

    val_pred = final_model.predict(X_val_f)
    val_auc  = roc_auc_score(y_val_f, val_pred)
    print(f"최종 모델 Val AUC: {val_auc:.4f}")

    # 저장
    final_model.save_model("model_lgbm.txt")
    with open("feat_cols_lgbm.pkl", "wb") as f:
        pickle.dump(FEAT_COLS, f)

    print("저장 완료: model_lgbm.txt / feat_cols_lgbm.pkl")
    print(f"\nWalk-Forward 평균 AUC: {np.mean(fold_aucs):.4f}")
