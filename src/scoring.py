"""
scoring.py
PathAI Engine — Engagement scoring algorithm (0–100 per student × week).

Pipeline:
  1. Winsorise trend_slope at p5/p95 (removes ε-explosion artefacts).
  2. Min-Max normalise each feature within its (module, presentation) group.
  3. Compute weighted composite score.
  4. Global Min-Max rescale → strict [0, 100] bounds.

Weights (sum = 1.0):
  freq          0.30  Strongest individual predictor (Herodotou 2019).
  forum_clicks  0.25  Most resilient to last-minute gaming (Tempelaar 2015).
  diversity     0.20  Separates deep engagement from click-farming.
  trend_slope   0.15  Trajectory signal; clean after winsorisation.
  prox_activity 0.10  Binary context flag — lowest per-student discriminance.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# Configuration


WEIGHTS: dict[str, float] = {
    "freq_norm":      0.30,
    "forum_norm":     0.25,
    "diversity_norm": 0.20,
    "trend_norm":     0.15,
    "prox_norm":      0.10,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"

TREND_LOWER_PCT = 0.05
TREND_UPPER_PCT = 0.95
MINMAX_EPSILON  = 1e-9



# Winsorise trend_slope


def winsorise_trend_slope(
    features: pd.DataFrame,
    lower_pct: float = TREND_LOWER_PCT,
    upper_pct: float = TREND_UPPER_PCT,
) -> tuple[pd.DataFrame, float, float]:
    """
    Clip trend_slope at global [lower_pct, upper_pct] percentiles.
    Uses global (not per-module) percentiles because ε-explosion artefacts
    appear uniformly across all modules.
    """
    lo = float(features["trend_slope"].quantile(lower_pct))
    hi = float(features["trend_slope"].quantile(upper_pct))
    raw_min = float(features["trend_slope"].min())
    raw_max = float(features["trend_slope"].max())

    df = features.copy()
    df["trend_slope"] = df["trend_slope"].clip(lo, hi)

    logger.info(
        "Winsorised trend_slope: raw [%.1f, %.1f] → clipped [%.1f, %.1f]  (p%d/p%d)",
        raw_min, raw_max, lo, hi,
        int(lower_pct * 100), int(upper_pct * 100),
    )
    return df, lo, hi



#Per-module Min-Max normalisation


def _minmax_transform(series: pd.Series) -> pd.Series:
    """Scale a Series to [0, 1]. Returns 0.0 when range == 0 (constant feature)."""
    mn, mx = series.min(), series.max()
    return (series - mn) / (mx - mn + MINMAX_EPSILON)


def normalise_features(features: pd.DataFrame) -> pd.DataFrame:
    """
    Apply per-(module, presentation) Min-Max normalisation to all five raw
    features, producing five `*_norm` columns in [0, 1].
    """
    df = features.copy()
    group_keys = ["code_module", "code_presentation"]

    mapping = {
        "freq":          "freq_norm",
        "diversity":     "diversity_norm",
        "prox_activity": "prox_norm",
        "trend_slope":   "trend_norm",
        "forum_clicks":  "forum_norm",
    }

    for raw_col, norm_col in mapping.items():
        df[norm_col] = (
            df.groupby(group_keys, observed=True)[raw_col]
            .transform(_minmax_transform)
            .astype("float32")
        )

    logger.info("Per-module Min-Max normalisation complete.")
    return df



#Weighted composite → global rescale


def compute_engagement_score(features: pd.DataFrame) -> pd.DataFrame:
    """
    Full scoring pipeline: winsorise → normalise → weighted composite
    → global Min-Max rescale to strict [0, 100].

    The global rescale (Step 4) is necessary because the weighted average of
    five independently normalised features empirically peaks at ~0.86, not 1.0.
    The linear rescale preserves all relative rankings.

    Parameters
    ----------
    features : output of feature_engineering.compute_features()

    Returns
    -------
    pd.DataFrame with additional columns:
        freq_norm, diversity_norm, prox_norm, trend_norm, forum_norm,
        engagement_score  [0, 100]
    """
    logger.info("=" * 60)
    logger.info("PathAI Engine — Engagement Scoring")
    logger.info("=" * 60)
    logger.info("Weights: %s", WEIGHTS)

    df, _, _ = winsorise_trend_slope(features)
    df = normalise_features(df)

    composite = (
        WEIGHTS["freq_norm"]      * df["freq_norm"]      +
        WEIGHTS["forum_norm"]     * df["forum_norm"]     +
        WEIGHTS["diversity_norm"] * df["diversity_norm"] +
        WEIGHTS["trend_norm"]     * df["trend_norm"]     +
        WEIGHTS["prox_norm"]      * df["prox_norm"]
    )
    logger.info("Composite pre-rescale: max = %.4f", float(composite.max()))

    # Global rescale → [0, 100]
    c_min, c_max = float(composite.min()), float(composite.max())
    df["engagement_score"] = (
        (composite - c_min) / (c_max - c_min + MINMAX_EPSILON) * 100.0
    ).round(2).astype("float32")

    logger.info(
        "Scoring complete. Score stats — mean: %.1f  std: %.1f  min: %.1f  max: %.1f",
        df["engagement_score"].mean(), df["engagement_score"].std(),
        df["engagement_score"].min(), df["engagement_score"].max(),
    )
    logger.info("=" * 60)
    return df



# Persistence


def save_scores(
    scored_df: pd.DataFrame,
    output_dir: str | Path = "./outputs",
    filename: str = "scored_features.parquet",
) -> Path:
    """Save scored DataFrame to Parquet. Creates output_dir if absent."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    df_save = scored_df.copy()
    for col in df_save.select_dtypes("category").columns:
        df_save[col] = df_save[col].astype(str)

    df_save.to_parquet(out_path, index=False)
    logger.info("Scores saved → %s  (%.1f MB)", out_path,
                out_path.stat().st_size / 1e6)
    return out_path



# Standalone smoke test


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))

    from data_loader import build_master_pipeline
    from feature_engineering import compute_features, attach_outcome_label

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    wc, si, assessments, _ = build_master_pipeline()
    features = compute_features(wc, assessments)
    features = attach_outcome_label(features, si)
    scored   = compute_engagement_score(features)

    save_scores(scored)

    cols = ["id_student", "code_module", "week_number",
            "freq_norm", "forum_norm", "diversity_norm",
            "trend_norm", "prox_norm", "engagement_score", "final_result"]
    print("\n=== Scored DataFrame — First 5 Rows ===")
    print(scored[cols].head(5).to_string(index=False))

    print("\n=== Score Distribution by Outcome ===")
    print(scored.groupby("final_result")["engagement_score"]
          .describe().round(1).to_string())
