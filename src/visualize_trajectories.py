"""
visualize_trajectories.py
PathAI Engine — Week-by-week engagement trajectory plot for Task 1.

Identifies three representative student archetypes from scored data and
plots their engagement_score trajectory on a single chart.

Archetypes:
  Steady Engager  — High sustained score, Pass/Distinction outcome.
  Early Dropout   — Engaged early, collapses and withdraws.
  Late Recoverer  — Low start, aggressive late-semester recovery.

Output: ./outputs/task1_trajectories.png
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import build_master_pipeline
from feature_engineering import compute_features, attach_outcome_label
from scoring import compute_engagement_score, save_scores

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# Design tokens


ARCHETYPE_STYLES = {
    "Steady Engager": {"color": "#2563EB", "marker": "o", "linewidth": 2.6, "markersize": 7},
    "Early Dropout":  {"color": "#DC2626", "marker": "s", "linewidth": 2.6, "markersize": 7},
    "Late Recoverer": {"color": "#16A34A", "marker": "^", "linewidth": 2.6, "markersize": 8},
}

OUTPUT_DIR  = Path("./outputs")
OUTPUT_FILE = OUTPUT_DIR / "task1_trajectories.png"
CACHE_FILE  = OUTPUT_DIR / "scored_features.parquet"



# Data loading


def load_scored_data() -> pd.DataFrame:
    """Return scored DataFrame, using parquet cache when available."""
    if CACHE_FILE.exists():
        logger.info("Loading scored data from cache: %s", CACHE_FILE)
        return pd.read_parquet(CACHE_FILE)

    logger.info("Cache not found — running full pipeline …")
    wc, si, assessments, _ = build_master_pipeline()
    features = compute_features(wc, assessments)
    features = attach_outcome_label(features, si)
    scored   = compute_engagement_score(features)
    save_scores(scored, output_dir=OUTPUT_DIR)
    return scored



# Archetype selection


def _build_student_summary(scored: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-student statistics for archetype selection."""
    df = scored.sort_values(["id_student", "week_number"])

    base = (
        df.groupby(["id_student", "code_module", "code_presentation", "final_result"],
                   observed=True)
        .agg(
            mean_score=("engagement_score", "mean"),
            std_score=("engagement_score", "std"),
            num_weeks=("week_number", "count"),
            last_week=("week_number", "max"),
        )
        .reset_index()
    )
    base["std_score"] = base["std_score"].fillna(0.0)
    base["cv"] = base["std_score"] / (base["mean_score"] + 1e-9)

    early = (
        df[df["week_number"] <= 4]
        .groupby("id_student")["engagement_score"].mean()
        .rename("early_score").reset_index()
    )
    last4_mask = (
        df.groupby("id_student")["week_number"].transform("max") - df["week_number"]
    ) < 4
    late = (
        df[last4_mask]
        .groupby("id_student")["engagement_score"].mean()
        .rename("late_score").reset_index()
    )

    summary = base.merge(early, on="id_student", how="left").merge(late, on="id_student", how="left")
    summary["early_score"]    = summary["early_score"].fillna(summary["mean_score"])
    summary["late_score"]     = summary["late_score"].fillna(summary["mean_score"])
    summary["recovery"]       = summary["late_score"]  - summary["early_score"]
    summary["dropout_signal"] = summary["early_score"] - summary["late_score"]
    return summary


def _pick_best(candidates: pd.DataFrame, sort_col: str,
               ascending: bool = False) -> Optional[int]:
    """Return id_student of the top candidate, or None if empty."""
    if candidates.empty:
        return None
    return int(candidates.sort_values(sort_col, ascending=ascending).iloc[0]["id_student"])


def find_archetypes(summary: pd.DataFrame) -> dict[str, int]:
    """
    Select one student ID per archetype using progressively relaxed criteria.
    Thresholds are calibrated to the empirical score distribution (mean ~16, max 100).
    """
    archetypes: dict[str, int] = {}

    # Steady Engager — top-decile mean score, low volatility, long participation
    for min_weeks, min_mean, max_cv in [(15, 16.5, 0.65), (12, 15.0, 0.70), (10, 14.0, 0.75)]:
        cands = summary[
            summary["final_result"].isin(["Pass", "Distinction"]) &
            (summary["num_weeks"] >= min_weeks) &
            (summary["mean_score"] >= min_mean) &
            (summary["cv"] <= max_cv)
        ]
        sid = _pick_best(cands, "mean_score")
        if sid:
            archetypes["Steady Engager"] = sid
            _log_archetype("Steady Engager", sid, summary)
            break

    # Early Dropout — above-median early engagement, stops before Week 14
    for min_early, max_last, min_drop in [(10, 14, 8), (7, 18, 5), (4, 22, 3)]:
        cands = summary[
            (summary["final_result"] == "Withdrawn") &
            (summary["early_score"] >= min_early) &
            (summary["last_week"] <= max_last) &
            (summary["dropout_signal"] >= min_drop)
        ]
        sid = _pick_best(cands, "dropout_signal")
        if sid:
            archetypes["Early Dropout"] = sid
            _log_archetype("Early Dropout", sid, summary)
            break

    # Late Recoverer — low early score, meaningful late uptick
    for min_weeks, max_early, min_recovery in [(14, 8, 8), (12, 10, 6), (10, 12, 4)]:
        cands = summary[
            summary["final_result"].isin(["Pass", "Distinction"]) &
            (summary["num_weeks"] >= min_weeks) &
            (summary["early_score"] <= max_early) &
            (summary["recovery"] >= min_recovery)
        ]
        if "Steady Engager" in archetypes:
            cands = cands[cands["id_student"] != archetypes["Steady Engager"]]
        sid = _pick_best(cands, "recovery")
        if sid:
            archetypes["Late Recoverer"] = sid
            _log_archetype("Late Recoverer", sid, summary)
            break

    if len(archetypes) < 3:
        missing = {"Steady Engager", "Early Dropout", "Late Recoverer"} - set(archetypes)
        logger.warning("Could not find archetype(s): %s", missing)

    return archetypes


def _log_archetype(name: str, sid: int, summary: pd.DataFrame) -> None:
    row = summary[summary["id_student"] == sid].iloc[0]
    logger.info("%-16s → id_student=%-8d  mean=%.1f  weeks=%d  cv=%.2f",
                name, sid, row["mean_score"], row["num_weeks"], row["cv"])



# Plot

def plot_trajectories(
    scored: pd.DataFrame,
    archetypes: dict[str, int],
    summary: pd.DataFrame,
) -> None:
    """Render and save the three-archetype engagement trajectory chart."""
    fig, ax = plt.subplots(figsize=(14, 7.5))
    fig.patch.set_facecolor("#FAFAF8")
    ax.set_facecolor("#FAFAF8")

    # Engagement zone bands
    ax.axhspan(  0, 33, color="#FFEAEA", alpha=0.55, zorder=0)
    ax.axhspan( 33, 66, color="#FFF9E6", alpha=0.55, zorder=0)
    ax.axhspan( 66, 100, color="#EAFAF1", alpha=0.55, zorder=0)

    for y_mid, label, color in [(16, "At Risk", "#C0392B"),
                                 (49, "Developing", "#D68910"),
                                 (83, "Engaged", "#1A7A4A")]:
        ax.text(0.995, y_mid / 100, label, transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=9, color=color,
                fontweight="semibold", alpha=0.75)

    for y in (33, 66):
        ax.axhline(y, color="#CCCCCC", linewidth=0.8, linestyle="--", zorder=1)

    all_weeks = []
    annotation_targets = []

    for archetype, sid in archetypes.items():
        style = ARCHETYPE_STYLES[archetype]
        student_rows = (
            scored[scored["id_student"] == sid]
            .sort_values("week_number")[["week_number", "engagement_score"]]
        )
        weeks  = student_rows["week_number"].values
        scores = student_rows["engagement_score"].values
        all_weeks.extend(weeks.tolist())

        row    = summary[summary["id_student"] == sid].iloc[0]
        label  = (f"{archetype}  (id={sid})\n"
                  f"Module {row['code_module']} · {row['code_presentation']} · "
                  f"{row['final_result']} · μ={row['mean_score']:.0f}")

        ax.plot(weeks, scores, label=label,
                color=style["color"], marker=style["marker"],
                linestyle="-", linewidth=style["linewidth"],
                markersize=style["markersize"], zorder=4, alpha=0.92)
        ax.fill_between(weeks, scores, alpha=0.08, color=style["color"], zorder=2)

        # Annotation for each archetype's defining moment
        if archetype == "Steady Engager":
            idx = scores.argmax()
            annotation_targets.append((weeks[idx], scores[idx],
                                        "Peak sustained\nengagement",
                                        style["color"], (-5, 18)))
        elif archetype == "Early Dropout":
            annotation_targets.append((weeks[-1], scores[-1],
                                        f"Last active\n(Week {weeks[-1]})",
                                        style["color"], (10, -22)))
        elif archetype == "Late Recoverer":
            mid = len(weeks) // 2
            if mid < len(weeks):
                idx = np.argmax(scores[mid:]) + mid
                annotation_targets.append((weeks[idx], scores[idx],
                                            "Late recovery\nspike",
                                            style["color"], (10, 12)))

    for (wx, wy, text, color, xytext) in annotation_targets:
        ax.annotate(text, xy=(wx, wy),
                    xytext=(wx + xytext[0], wy + xytext[1]),
                    fontsize=8.5, color=color, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.4,
                                    connectionstyle="arc3,rad=0.2"),
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                              edgecolor=color, alpha=0.85, linewidth=1),
                    zorder=6)

    # Axes
    x_min = max(1, min(all_weeks) - 0.5)
    x_max = max(all_weeks) + 0.5
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-2, 102)
    ax.set_xlabel("Week Number", fontsize=12, labelpad=8, color="#333333")
    ax.set_ylabel("Engagement Score (0–100)", fontsize=12, labelpad=8, color="#333333")
    ax.set_title("Student Engagement Trajectories — PathAI Engine  (Task 1: OULAD)",
                 fontsize=14, fontweight="bold", pad=18, color="#111111")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.tick_params(axis="both", labelsize=9.5, color="#888888", labelcolor="#444444")
    for spine in ax.spines.values():
        spine.set_edgecolor("#DDDDDD")
        spine.set_linewidth(0.8)
    ax.grid(axis="both", color="#E5E5E5", linewidth=0.6, linestyle="-", zorder=0)

    legend = ax.legend(loc="upper right", fontsize=8.8, framealpha=0.95,
                        edgecolor="#CCCCCC", facecolor="white",
                        borderpad=1.0, labelspacing=0.9, handlelength=2.5,
                        title="Archetypes", title_fontsize=9)
    legend.get_title().set_fontweight("bold")

    fig.text(0.01, 0.01,
             "PathAI Engine · Studor DS Team · Engagement score: weighted combination of "
             "frequency, forum activity, diversity, trend slope, and assessment proximity.",
             fontsize=7.5, color="#999999", style="italic")

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FILE, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    logger.info("Plot saved → %s", OUTPUT_FILE.resolve())
    plt.close(fig)



# Entry point


if __name__ == "__main__":
    scored  = load_scored_data()
    summary = _build_student_summary(scored)
    logger.info("Summary covers %s students.", f"{len(summary):,}")

    archetypes = find_archetypes(summary)

    if len(archetypes) < 3:
        raise RuntimeError(
            f"Only {len(archetypes)}/3 archetypes found: {list(archetypes.keys())}."
        )

    print("\n" + "=" * 60)
    print("  Archetype Student Profiles")
    print("=" * 60)
    profile_rows = summary[summary["id_student"].isin(archetypes.values())].copy()
    profile_rows["archetype"] = profile_rows["id_student"].map(
        {v: k for k, v in archetypes.items()}
    )
    cols = ["archetype", "id_student", "final_result", "code_module",
            "code_presentation", "num_weeks", "mean_score", "early_score",
            "late_score", "cv", "recovery"]
    print(profile_rows[cols].to_string(index=False))

    plot_trajectories(scored, archetypes, summary)
    print(f"\nPlot saved to: {OUTPUT_FILE.resolve()}")
