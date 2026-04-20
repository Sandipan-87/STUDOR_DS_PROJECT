"""
tests/final_audit.py
PathAI Engine — Pre-Submission Diagnostic Audit

Checks:
  1. Overfitting       — Train vs Test AUC / Recall delta for the Task 2 model.
  2. Feature leakage   — Verify final_result was NOT used to compute engagement_score.
  3. Data consistency  — Confirm Task 3 returner students are real IDs in studentInfo.csv.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

PARQUET      = ROOT / "outputs" / "scored_features.parquet"
STUDENT_INFO = ROOT / "data" / "studentInfo.csv"

PASS     = "[PASS]"
FAIL     = "[FAIL]"
WARN     = "[WARN]"

# Thresholds for overfitting check
MAX_AUC_DELTA    = 0.10   # train-test AUC gap > 0.10 → overfitting
MAX_RECALL_DELTA = 0.15   # train-test Recall gap > 0.15 → overfitting

# Task 2 constants (must match task2_predictive_model.py exactly)
PREDICTION_WEEK    = 6
DECISION_THRESHOLD = 0.35
RANDOM_STATE       = 42
PASS_LABELS        = {"Pass", "Distinction"}
TRAIN_PRES_T3      = {"2013B", "2013J"}
TEST_PRES_T3       = {"2014B", "2014J"}

FEATURE_COLS = [
    "mean_score", "min_score", "w6_score",
    "score_trend", "score_decline", "score_cv",
    "weeks_active", "mean_forum", "mean_diversity", "total_freq",
]

results: list[tuple[str, str, str]] = []   # (check_name, status, detail)


def record(name: str, ok: bool, detail: str, warn: bool = False) -> None:
    status = WARN if (warn and not ok) else (PASS if ok else FAIL)
    results.append((name, status, detail))
    icon = "OK " if ok else ("!! " if not warn else "?? ")
    print(f"  {status}  {name}")
    print(f"         {detail}")


# ---------------------------------------------------------------------------
# Helpers — replicate Task 2 data prep (must match task2_predictive_model.py)
# ---------------------------------------------------------------------------

def _ols_slope(weeks: np.ndarray, scores: np.ndarray) -> float:
    return float(np.polyfit(weeks, scores, 1)[0]) if len(weeks) >= 2 else 0.0


def _build_task2_dataset() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(PARQUET)
    df = df[df["week_number"] <= PREDICTION_WEEK].copy()

    def _agg(g: pd.DataFrame) -> pd.Series:
        g = g.sort_values("week_number")
        scores = g["engagement_score"].values.astype(float)
        weeks  = g["week_number"].values.astype(float)
        return pd.Series({
            "mean_score":     scores.mean(),
            "min_score":      scores.min(),
            "w6_score":       scores[-1],
            "score_trend":    _ols_slope(weeks, scores),
            "score_decline":  float(scores[0] - scores[-1]),
            "score_cv":       scores.std() / (scores.mean() + 1e-9),
            "weeks_active":   float(len(scores)),
            "mean_forum":     g["forum_clicks"].mean(),
            "mean_diversity": g["diversity"].mean(),
            "total_freq":     float(g["freq"].sum()),
        })

    grouped = (
        df.groupby(
            ["id_student", "code_module", "code_presentation",
             "final_result", "label"],
            observed=True,
        )
        .apply(_agg, include_groups=False)
        .reset_index()
    )

    X = grouped[FEATURE_COLS].astype("float32")
    y = grouped["label"].astype(int)
    return X, y


def _train_task2_model(X_train, y_train) -> XGBClassifier:
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    spw   = round((n_neg / n_pos) * 1.5, 3)
    model = XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, eval_metric="aucpr",
        random_state=RANDOM_STATE, verbosity=0,
    )
    model.fit(X_train, y_train)
    return model


# ===========================================================================
# Audit Check 1: Overfitting
# ===========================================================================

def check_overfitting() -> None:
    print("\n[1] Overfitting Audit (Task 2 — Train vs Test metrics)")
    print("    Re-training with identical hyperparameters and same random split...")

    X, y = _build_task2_dataset()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    model = _train_task2_model(X_tr, y_tr)

    # Train-set metrics
    tr_proba = model.predict_proba(X_tr)[:, 1]
    tr_preds  = (tr_proba >= DECISION_THRESHOLD).astype(int)
    tr_auc    = roc_auc_score(y_tr, tr_proba)
    tr_recall = recall_score(y_tr, tr_preds, zero_division=0)

    # Test-set metrics
    te_proba = model.predict_proba(X_te)[:, 1]
    te_preds  = (te_proba >= DECISION_THRESHOLD).astype(int)
    te_auc    = roc_auc_score(y_te, te_proba)
    te_recall = recall_score(y_te, te_preds, zero_division=0)

    auc_delta    = tr_auc    - te_auc
    recall_delta = tr_recall - te_recall

    print(f"    {'Metric':<16} {'Train':>8} {'Test':>8} {'Delta':>8} {'Threshold':>10}")
    print(f"    {'-'*16} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
    print(f"    {'ROC-AUC':<16} {tr_auc:>8.4f} {te_auc:>8.4f} "
          f"{auc_delta:>+8.4f} {MAX_AUC_DELTA:>10.2f}")
    print(f"    {'Recall':<16} {tr_recall:>8.4f} {te_recall:>8.4f} "
          f"{recall_delta:>+8.4f} {MAX_RECALL_DELTA:>10.2f}")

    auc_ok    = auc_delta    <= MAX_AUC_DELTA
    recall_ok = recall_delta <= MAX_RECALL_DELTA

    record(
        "ROC-AUC gap (train - test)",
        auc_ok,
        f"Delta = {auc_delta:+.4f}  (max allowed {MAX_AUC_DELTA})",
    )
    record(
        "Recall gap (train - test)",
        recall_ok,
        f"Delta = {recall_delta:+.4f}  (max allowed {MAX_RECALL_DELTA})",
    )


# ===========================================================================
# Audit Check 2: Feature Leakage
# ===========================================================================

def check_leakage() -> None:
    print("\n[2] Feature Leakage Audit (Task 1 engagement_score independence)")

    df = pd.read_parquet(PARQUET)

    # 2a — Code inspection: final_result must not appear as an input in FE / scoring.
    #      We strip everything after 'if __name__' from scoring.py because that block
    #      contains display-only groupby("final_result") calls, which are not leaks.
    fe_src  = (ROOT / "src" / "feature_engineering.py").read_text(encoding="utf-8")
    sc_src_full = (ROOT / "src" / "scoring.py").read_text(encoding="utf-8")
    # Crop to the compute logic — everything before the __main__ guard
    sc_src = sc_src_full.split('if __name__')[0]

    import re
    bad_pattern = re.compile(
        r"groupby.*final_result|final_result.*groupby|"
        r"transform.*final_result|final_result.*transform|"
        r"pivot.*final_result|final_result.*pivot",
        re.IGNORECASE
    )
    fe_bad = bool(bad_pattern.search(fe_src))
    sc_bad = bool(bad_pattern.search(sc_src))

    record(
        "final_result NOT in FE groupby/transform",
        not fe_bad,
        "feature_engineering.py: no leaking groupby/transform patterns" if not fe_bad
        else "ALERT: final_result appears in groupby/transform in feature_engineering.py",
    )
    record(
        "final_result NOT in scoring compute logic",
        not sc_bad,
        "scoring.py (compute path only): no leaking patterns" if not sc_bad
        else "ALERT: final_result appears in compute logic of scoring.py",
    )

    # 2b — Statistical: engagement_score AUC predicting binary label
    #      If leakage, AUC would be ~1.0; legitimate correlation gives 0.6-0.8
    from sklearn.metrics import roc_auc_score as roc
    df_agg = df.groupby("id_student").agg(
        mean_score=("engagement_score", "mean"),
        label=("label", "first"),
    ).dropna()

    solo_auc = roc(df_agg["label"], df_agg["mean_score"])
    # Legitimate range: 0.55-0.82 (engaged students tend to pass, but not perfectly)
    # Leakage signal: > 0.90 would be suspicious
    auc_ok = solo_auc < 0.90
    record(
        "engagement_score solo AUC < 0.90",
        auc_ok,
        f"Single-feature AUC = {solo_auc:.4f}  (leakage threshold: 0.90)",
        warn=True,
    )

    # 2c — Within-group variance: std of engagement_score per final_result
    #      If std is near 0 for any group, the feature was constructed from the label
    group_std = df.groupby("final_result")["engagement_score"].std().round(3)
    all_nonzero = (group_std > 1.0).all()
    std_detail  = "  |  ".join(f"{k}: std={v}" for k, v in group_std.items())
    record(
        "engagement_score has variance in all outcome groups",
        all_nonzero,
        std_detail,
    )

    # 2d — Overlap check: IQR of Pass and Withdrawn distributions must overlap.
    #      We use IQR (not a fixed threshold) because the score distribution is
    #      left-skewed (mean ~16, std ~11) — most students score below 30 regardless
    #      of outcome, which is expected not suspicious.
    pass_q1, pass_q3 = df[df["final_result"] == "Pass"]["engagement_score"].quantile([0.25, 0.75])
    with_q1, with_q3 = df[df["final_result"] == "Withdrawn"]["engagement_score"].quantile([0.25, 0.75])
    # IQR overlap exists if the ranges [q1,q3] for the two groups intersect
    iqr_overlap = (pass_q1 <= with_q3) and (with_q1 <= pass_q3)
    record(
        "Score distributions have IQR overlap (no perfect separation)",
        iqr_overlap,
        f"Pass IQR: [{pass_q1:.1f}, {pass_q3:.1f}]  |  "
        f"Withdrawn IQR: [{with_q1:.1f}, {with_q3:.1f}]  |  "
        f"Overlap: {iqr_overlap}",
    )


# ===========================================================================
# Audit Check 3: Data Consistency
# ===========================================================================

def check_data_consistency() -> None:
    print("\n[3] Data Consistency Audit (Task 3 — 1,653 returner students)")

    si      = pd.read_csv(STUDENT_INFO)
    scored  = pd.read_parquet(PARQUET)

    si_train = si[si["code_presentation"].isin(TRAIN_PRES_T3)]
    si_test  = si[si["code_presentation"].isin(TEST_PRES_T3)]

    # Replicate EXACTLY the Task 3 eligibility logic
    scored_train = scored[scored["code_presentation"].isin(TRAIN_PRES_T3)]
    matrix_index = set(scored_train["id_student"].unique())

    eligible = (
        si_test["id_student"][
            si_test["id_student"].isin(si_train["id_student"]) &
            si_test["id_student"].isin(matrix_index)
        ]
        .unique()
    )
    n_eligible = len(eligible)

    # 3a — Count matches reported count
    count_ok = 1600 <= n_eligible <= 1700   # allow slight difference from run variation
    record(
        "Task 3 returner count in expected range [1600, 1700]",
        count_ok,
        f"Computed: {n_eligible:,}  (Task 3 reported: 1,653)",
    )

    # 3b — All eligible IDs exist in studentInfo
    si_all_ids  = set(si["id_student"])
    phantom_ids = set(eligible) - si_all_ids
    no_phantoms = len(phantom_ids) == 0
    record(
        "All returner IDs exist in studentInfo.csv",
        no_phantoms,
        f"Phantom IDs: {len(phantom_ids)}  (must be 0)",
    )

    # 3c — All eligible IDs actually appear in 2013 scored data
    scored_2013_ids = set(scored_train["id_student"].unique())
    missing_2013 = set(eligible) - scored_2013_ids
    in_2013_ok   = len(missing_2013) == 0
    record(
        "All returner IDs present in 2013 behavioral matrix",
        in_2013_ok,
        f"Missing from 2013 scored data: {len(missing_2013)}  (must be 0)",
    )

    # 3d — All eligible IDs appear in 2014 studentInfo (ground-truth enrollment)
    si_2014_ids   = set(si_test["id_student"])
    missing_2014  = set(eligible) - si_2014_ids
    in_2014_ok    = len(missing_2014) == 0
    record(
        "All returner IDs have 2014 enrollment (ground truth)",
        in_2014_ok,
        f"Missing from 2014 studentInfo: {len(missing_2014)}  (must be 0)",
    )

    # 3e — No NaN engagement scores in the parquet
    null_scores = int(scored["engagement_score"].isna().sum())
    record(
        "No NaN engagement scores in parquet",
        null_scores == 0,
        f"NaN count: {null_scores}  (must be 0)",
    )


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    DIV = "=" * 60

    print(DIV)
    print("  PathAI Engine — Final Pre-Submission Audit")
    print(DIV)

    check_overfitting()
    check_leakage()
    check_data_consistency()

    # Summary
    passed  = sum(1 for _, s, _ in results if s == PASS)
    failed  = sum(1 for _, s, _ in results if s == FAIL)
    warned  = sum(1 for _, s, _ in results if s == WARN)
    total   = len(results)

    print(f"\n{DIV}")
    print(f"  AUDIT SUMMARY  —  {total} checks total")
    print(DIV)
    print(f"  {PASS}  {passed} checks passed")
    if warned:
        print(f"  {WARN}  {warned} warnings (acceptable anomalies)")
    if failed:
        print(f"  {FAIL}  {failed} checks FAILED  ← action required")

    print()
    if failed == 0:
        print("  REPOSITORY IS CLEAN — safe to submit.")
    else:
        print("  ACTION REQUIRED — fix the FAIL items before submission.")
    print(DIV)

    sys.exit(0 if failed == 0 else 1)
