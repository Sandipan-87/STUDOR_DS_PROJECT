"""
data_loader.py
PathAI Engine — OULAD data loading and aggregation pipeline.

Memory strategy for studentVle (453 MB):
  - Downcast int64 → int16/int32 at read time (~65 % memory reduction).
  - Use 'category' dtype for low-cardinality string columns.
  - Drop id_site and date columns immediately after use.
  - Aggregate to (student, module, week, activity_type) before returning.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# Dtype maps

STUDENT_VLE_DTYPES = {
    "code_module":       "category",
    "code_presentation": "category",
    "id_student":        "int32",
    "id_site":           "int16",   # max ~2,000
    "date":              "int16",   # range −25 … 269
    "sum_click":         "int16",   # max ~1,000
}

VLE_DTYPES = {
    "id_site":           "int16",
    "code_module":       "category",
    "code_presentation": "category",
    "activity_type":     "category",
}

STUDENT_INFO_DTYPES = {
    "id_student":        "int32",
    "code_module":       "category",
    "code_presentation": "category",
}

STUDENT_INFO_KEEP_COLS = [
    "code_module", "code_presentation", "id_student",
    "gender", "region", "highest_education", "imd_band",
    "age_band", "num_of_prev_attempts", "studied_credits",
    "disability", "final_result",
]


# Internal helper

def _load_csv(
    filename: str,
    dtype: Optional[dict] = None,
    usecols: Optional[list] = None,
    **kwargs,
) -> pd.DataFrame:
    """Read a CSV from DATA_DIR and log its row count and memory footprint."""
    path = DATA_DIR / filename
    logger.info("Loading  %-35s", path.name)
    df = pd.read_csv(path, dtype=dtype, usecols=usecols, **kwargs)
    mem_mb = df.memory_usage(deep=True).sum() / 1e6
    logger.info("  ↳ %s rows × %d cols  |  %.1f MB in memory",
                f"{len(df):,}", len(df.columns), mem_mb)
    return df


# Public loaders

def load_vle() -> pd.DataFrame:
    """Load VLE activity-type lookup table."""
    return _load_csv("vle.csv", dtype=VLE_DTYPES)


def load_student_info() -> pd.DataFrame:
    """
    Load student demographics and outcomes.
    Adds binary column `label`: 1 = Withdrawn/Fail, 0 = Pass/Distinction.
    """
    df = _load_csv("studentInfo.csv", dtype=STUDENT_INFO_DTYPES,
                   usecols=STUDENT_INFO_KEEP_COLS)
    df["label"] = df["final_result"].isin(["Withdrawn", "Fail"]).astype("int8")
    return df


def load_assessments() -> pd.DataFrame:
    """Load TMA/CMA deadline schedule. Exam rows are excluded to prevent leakage."""
    df = _load_csv(
        "assessments.csv",
        dtype={
            "id_assessment":     "int32",
            "code_module":       "category",
            "code_presentation": "category",
            "weight":            "float32",
            "date":              "float32",
        },
    )
    df = df[df["assessment_type"] != "Exam"].copy()
    logger.info("  ↳ Excluded Exam rows; %d TMA/CMA deadlines remain", len(df))
    return df


def load_student_assessment() -> pd.DataFrame:
    """Load per-student assessment submission records."""
    return _load_csv(
        "studentAssessment.csv",
        dtype={
            "id_assessment":  "int32",
            "id_student":     "int32",
            "date_submitted": "float32",
            "is_banked":      "int8",
            "score":          "float32",
        },
    )


def load_student_registration() -> pd.DataFrame:
    """Load student registration and unregistration dates."""
    return _load_csv(
        "studentRegistration.csv",
        dtype={
            "id_student":          "int32",
            "code_module":         "category",
            "code_presentation":   "category",
            "date_registration":   "float32",
            "date_unregistration": "float32",
        },
    )


def load_courses() -> pd.DataFrame:
    """Load course-length metadata."""
    return _load_csv(
        "courses.csv",
        dtype={
            "code_module":                "category",
            "code_presentation":          "category",
            "module_presentation_length": "int16",
        },
    )



# Core heavy loader


def load_weekly_click_base() -> pd.DataFrame:
    """
    Load studentVle.csv and return weekly aggregated clicks per student.

    Week convention: week_number = (date // 7) + 1  (1-indexed, date ≥ 0).
    Pre-module rows (date < 0) are excluded.

    Returns
    -------
    pd.DataFrame
        Columns: code_module, code_presentation, id_student,
                 week_number, activity_type, sum_click
    """
    # Load VLE lookup first (small, stays in RAM)
    vle_lookup = (
        load_vle()[["id_site", "activity_type"]]
        .drop_duplicates(subset="id_site")
        .set_index("id_site")
    )

    svle = _load_csv("studentVle.csv", dtype=STUDENT_VLE_DTYPES)

    # Drop pre-module activity (date < 0)
    n_before = len(svle)
    svle = svle[svle["date"] >= 0].copy()
    logger.info("  ↳ Dropped %s pre-module rows; %s remain",
                f"{n_before - len(svle):,}", f"{len(svle):,}")

    svle["week_number"] = ((svle["date"] // 7) + 1).astype("int8")

    # Attach activity_type from VLE lookup
    svle = svle.merge(vle_lookup, on="id_site", how="left")
    svle["activity_type"] = (
        svle["activity_type"].cat.add_categories("unknown").fillna("unknown")
    )

    svle.drop(columns=["id_site", "date"], inplace=True)

    # Aggregate to (student, module, week, activity_type)
    weekly_clicks = (
        svle.groupby(
            ["code_module", "code_presentation", "id_student",
             "week_number", "activity_type"],
            observed=True,
        )["sum_click"]
        .sum()
        .reset_index()
    )
    weekly_clicks["sum_click"] = weekly_clicks["sum_click"].astype("int32")

    logger.info(
        "Weekly click base ready  →  %s rows | %s unique students | weeks %d–%d",
        f"{len(weekly_clicks):,}",
        f"{weekly_clicks['id_student'].nunique():,}",
        weekly_clicks["week_number"].min(),
        weekly_clicks["week_number"].max(),
    )
    return weekly_clicks


# Master pipeline


def build_master_pipeline() -> tuple[pd.DataFrame, pd.DataFrame,
                                     pd.DataFrame, pd.DataFrame]:
    """
    Load and return all four tables needed by feature_engineering.py.

    Returns: (weekly_clicks, student_info, assessments, student_assess)
    """
    logger.info("=" * 60)
    logger.info("PathAI Engine — Data Loading Pipeline")
    logger.info("=" * 60)

    weekly_clicks  = load_weekly_click_base()
    student_info   = load_student_info()
    assessments    = load_assessments()
    student_assess = load_student_assessment()

    logger.info("=" * 60)
    logger.info("All tables loaded successfully.")
    logger.info("=" * 60)

    return weekly_clicks, student_info, assessments, student_assess


# Standalone smoke test

if __name__ == "__main__":
    wc, si, ass, sa = build_master_pipeline()
    print("\nweekly_clicks head:\n", wc.head().to_string())
    print("\nstudent_info head:\n", si.head().to_string())
    print("\nassessments head:\n", ass.head().to_string())
    print("\nstudent_assess head:\n", sa.head().to_string())
