"""
feature_engineering.py
PathAI Engine — Dynamic feature construction (per student × week).

Six features, recalculated every week:
  F1  freq          Total clicks that week.
  F2  diversity     Unique activity categories accessed that week.
  F3  prox_activity Binary flag: 1 if a TMA/CMA deadline falls within 7 days.
  F4  trend_slope   % change in clicks vs. previous week.
  F5  forum_clicks  Clicks on collaborative activity types (forum, wiki, etc.).
  F6  ema_freq      3-week Exponential Moving Average of freq (α=0.5, span=3).
                    Smooths single-week spikes; used in scoring instead of raw freq.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Constants


FORUM_ACTIVITY_TYPES = frozenset([
    "forumng", "oucollaborate", "ouwiki", "ouelluminate", "repeatactivity",
])

PROXIMITY_WINDOW_DAYS = 7
EPSILON = 1e-3

GROUP_KEYS_FULL   = ["code_module", "code_presentation", "id_student", "week_number"]
GROUP_KEYS_STUD   = ["code_module", "code_presentation", "id_student"]
GROUP_KEYS_MODULE = ["code_module", "code_presentation", "week_number"]



# Feature helpers


def _compute_freq(weekly_clicks: pd.DataFrame) -> pd.DataFrame:
    """F1: Total clicks per student per week."""
    return (
        weekly_clicks
        .groupby(GROUP_KEYS_FULL, observed=True)["sum_click"]
        .sum()
        .reset_index()
        .rename(columns={"sum_click": "freq"})
    )


def _compute_ema_freq(features: pd.DataFrame) -> pd.DataFrame:
    """
    F6: 3-week Exponential Moving Average of weekly click freq.

    Uses span=3 (α = 2/(3+1) = 0.5) so each week's EMA is 50 % current
    clicks + 50 % recent history. More robust than raw freq because it
    dampens single-week cramming spikes while preserving trend momentum.
    EWM is applied on observed weeks only (gaps are treated sequentially).
    """
    EMA_SPAN = 3

    def _ewm(s: pd.Series) -> pd.Series:
        return s.ewm(span=EMA_SPAN, adjust=False).mean()

    features = features.sort_values(GROUP_KEYS_FULL).copy()
    features["ema_freq"] = (
        features
        .groupby(GROUP_KEYS_STUD, observed=True)["freq"]
        .transform(_ewm)
        .round(2)
        .astype("float32")
    )
    return features


def _compute_diversity(weekly_clicks: pd.DataFrame) -> pd.DataFrame:
    """F2: Count of distinct activity types accessed per student per week."""
    div = (
        weekly_clicks
        .groupby(GROUP_KEYS_FULL, observed=True)["activity_type"]
        .nunique()
        .reset_index()
        .rename(columns={"activity_type": "diversity"})
    )
    div["diversity"] = div["diversity"].astype("int8")
    return div


def _compute_forum_clicks(weekly_clicks: pd.DataFrame) -> pd.DataFrame:
    """F5: Sum of clicks on collaborative/social activity types."""
    mask = weekly_clicks["activity_type"].isin(FORUM_ACTIVITY_TYPES)
    if not mask.any():
        logger.warning("No rows matched FORUM_ACTIVITY_TYPES.")

    forum = (
        weekly_clicks[mask]
        .groupby(GROUP_KEYS_FULL, observed=True)["sum_click"]
        .sum()
        .reset_index()
        .rename(columns={"sum_click": "forum_clicks"})
    )
    forum["forum_clicks"] = forum["forum_clicks"].astype("int32")
    return forum


def _build_proximity_flag(assessments: pd.DataFrame) -> pd.DataFrame:
    """
    F3: Assessment proximity flag.

    Returns a module-level DataFrame with prox_activity = 1 for any
    (module, presentation, week) where a TMA/CMA deadline falls within
    PROXIMITY_WINDOW_DAYS days after that week begins.
    """
    rows = []
    for _, row in assessments.dropna(subset=["date"]).iterrows():
        deadline_day = int(row["date"])
        for day_offset in range(PROXIMITY_WINDOW_DAYS + 1):
            day = deadline_day - day_offset
            if day < 0:
                continue
            rows.append({
                "code_module":       row["code_module"],
                "code_presentation": row["code_presentation"],
                "week_number":       int(day // 7) + 1,
            })

    if not rows:
        logger.warning("No assessment dates found — prox_activity will be all 0.")
        return pd.DataFrame(columns=["code_module", "code_presentation",
                                     "week_number", "prox_activity"])

    prox_df = pd.DataFrame(rows).drop_duplicates().assign(prox_activity=1)
    prox_df["prox_activity"] = prox_df["prox_activity"].astype("int8")
    logger.info("  ↳ Proximity flag covers %d (module, week) pairs", len(prox_df))
    return prox_df


def _compute_trend_slope(features: pd.DataFrame) -> pd.DataFrame:
    """
    F4: Week-over-week % change in total clicks.

    Formula: ((freq_t − freq_{t−1}) / (freq_{t−1} + ε)) × 100
    First week for each student is set to 0.0.
    """
    features = features.sort_values(GROUP_KEYS_FULL).copy()
    eps = EPSILON

    def _pct_change_safe(s: pd.Series) -> pd.Series:
        shifted = s.shift(1)
        return ((s - shifted) / (shifted + eps)) * 100.0

    features["trend_slope"] = (
        features
        .groupby(GROUP_KEYS_STUD, observed=True)["freq"]
        .transform(_pct_change_safe)
        .fillna(0.0)
        .astype("float32")
    )
    return features



# Public API


def compute_features(
    weekly_clicks: pd.DataFrame,
    assessments: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build all six dynamic engagement features for every
    (id_student, code_module, code_presentation, week_number).

    Parameters
    ----------
    weekly_clicks : output of data_loader.load_weekly_click_base()
    assessments   : output of data_loader.load_assessments()

    Returns
    -------
    pd.DataFrame with columns:
        code_module, code_presentation, id_student, week_number,
        freq, ema_freq, diversity, forum_clicks, prox_activity, trend_slope
    """
    logger.info("─" * 60)
    logger.info("PathAI Engine — Feature Engineering")
    logger.info("─" * 60)

    logger.info("[F1] Weekly Interaction Frequency …")
    features = _compute_freq(weekly_clicks)

    logger.info("[F6] 3-Week EMA of Frequency …")
    features = _compute_ema_freq(features)

    logger.info("[F2] Activity Diversity …")
    diversity = _compute_diversity(weekly_clicks)
    features = features.merge(diversity, on=GROUP_KEYS_FULL, how="left")

    logger.info("[F5] Forum Engagement …")
    forum = _compute_forum_clicks(weekly_clicks)
    features = features.merge(forum, on=GROUP_KEYS_FULL, how="left")
    features["forum_clicks"] = features["forum_clicks"].fillna(0).astype("int32")

    logger.info("[F3] Assessment Proximity Activity …")
    prox = _build_proximity_flag(assessments)
    features = features.merge(prox, on=GROUP_KEYS_MODULE, how="left")
    features["prox_activity"] = features["prox_activity"].fillna(0).astype("int8")

    logger.info("[F4] Weekly Trend Slope …")
    features = _compute_trend_slope(features)

    features = features.reset_index(drop=True)
    logger.info("─" * 60)
    logger.info("Feature matrix ready  →  %s rows × %d cols",
                f"{len(features):,}", len(features.columns))
    logger.info("─" * 60)
    return features


def attach_outcome_label(
    features: pd.DataFrame,
    student_info: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join binary `label` and `final_result` onto the feature matrix.
    label = 1 → Withdrawn/Fail;  label = 0 → Pass/Distinction.
    """
    label_df = (
        student_info[["code_module", "code_presentation", "id_student",
                      "label", "final_result"]]
        .drop_duplicates(subset=["code_module", "code_presentation", "id_student"])
    )
    return features.merge(
        label_df,
        on=["code_module", "code_presentation", "id_student"],
        how="left",
    )



# Standalone smoke test


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader import build_master_pipeline

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    wc, si, assessments, sa = build_master_pipeline()
    features = compute_features(wc, assessments)
    features = attach_outcome_label(features, si)

    DIVIDER = "=" * 60
    print(f"\n{DIVIDER}\n  PathAI — Feature Matrix: First 5 Rows\n{DIVIDER}")
    print(features.head(5).to_string(index=False))
    print(f"\n{DIVIDER}\n  Null Counts\n{DIVIDER}")
    print(features.isnull().sum().to_string())
    print(f"\n{DIVIDER}\n  Feature Summary Statistics\n{DIVIDER}")
    feat_cols = ["freq", "ema_freq", "diversity", "prox_activity",
                 "trend_slope", "forum_clicks"]
    print(features[feat_cols].describe().round(3).to_string())
