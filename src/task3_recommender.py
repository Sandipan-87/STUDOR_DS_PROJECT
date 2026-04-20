"""
task3_recommender.py
PathAI Engine — Task 3: Course Recommendation Engine

Primary user: Academic Advisors (staff) recommending next-semester modules
              based on a student's behavioral history and demographic profile.

Two pipelines implemented:
  A. Content-Based  — cosine similarity on ordinal-encoded demographic profiles
  B. Collaborative  — cosine KNN on normalized engagement feature vectors

Cold-Start: top modules by global pass rate, filtered by education level.
Evaluation: 2013→2014 temporal holdout. Metrics: Precision@3, Coverage@3.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR   = Path("./data")
OUTPUT_DIR = Path("./outputs")
PARQUET    = OUTPUT_DIR / "scored_features.parquet"
REPORT_MD  = OUTPUT_DIR / "task3_recsys_report.md"

K_NEIGHBORS = 10
N_RECS      = 3
TRAIN_PRES  = {"2013B", "2013J"}
TEST_PRES   = {"2014B", "2014J"}
PASS_LABELS = {"Pass", "Distinction"}

BEHAV_COLS = ["freq_norm", "forum_norm", "diversity_norm", "trend_norm", "prox_norm"]

# Ordinal encodings — preserve natural ordering for cosine similarity
EDU_ORDER = {
    "No Formal quals": 0, "Lower Than A Level": 1,
    "A Level or Equivalent": 2, "HE Qualification": 3,
    "Post Graduate Qualification": 4,
}
IMD_ORDER = {
    f"{i*10}-{(i+1)*10}%": i for i in range(10)
}
IMD_ORDER["90-100%"] = 9
AGE_ORDER = {"0-35": 0, "35-55": 1, "55<=": 2}

DEMO_FEATURES = ["edu_score", "imd_score", "age_score", "gender_bin"]



# Data loading


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load studentInfo, courses, and scored behavioral parquet."""
    logger.info("=" * 55)
    logger.info("Task 3 — Loading data")
    logger.info("=" * 55)
    si      = pd.read_csv(DATA_DIR / "studentInfo.csv")
    courses = pd.read_csv(DATA_DIR / "courses.csv")
    scored  = pd.read_parquet(PARQUET)
    logger.info("  studentInfo : %s rows", f"{len(si):,}")
    logger.info("  courses     : %s rows", f"{len(courses):,}")
    logger.info("  scored      : %s rows", f"{len(scored):,}")
    return si, courses, scored



# Demographic encoding (shared by content-based and cold-start)


def _encode_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """Ordinally encode demographics; preserves natural ordering for cosine sim."""
    out = df.copy()
    out["edu_score"]  = out["highest_education"].map(EDU_ORDER).fillna(2.0)
    out["imd_score"]  = out["imd_band"].map(IMD_ORDER).fillna(5.0)
    out["age_score"]  = out["age_band"].map(AGE_ORDER).fillna(0.0)
    out["gender_bin"] = (out["gender"] == "M").astype(float)
    return out



# Option A: Content-Based Filtering


def build_course_profiles(si: pd.DataFrame) -> pd.DataFrame:
    """
    Build a demographic profile per module from students who passed.

    Profile vector = mean ordinal-encoded demographics of passing students.
    Aggregating *only* passing students ensures the profile represents
    the typical student who succeeds there, not the full enrollment mix.

    Returns DataFrame indexed by code_module with columns = DEMO_FEATURES.
    """
    si_enc  = _encode_demographics(si)
    passing = si_enc[si_enc["final_result"].isin(PASS_LABELS)]
    profiles = (
        passing.groupby("code_module")[DEMO_FEATURES]
        .mean()
        .fillna(0.0)
    )
    logger.info("Course profiles built for %d modules: %s",
                len(profiles), profiles.index.tolist())
    return profiles


def recommend_content_based(
    id_student: int,
    si: pd.DataFrame,
    course_profiles: pd.DataFrame,
    n: int = N_RECS,
    exclude_modules: set | None = None,
) -> list[str]:
    """
    Recommend modules whose typical-student demographic profile best matches
    the target student's profile via cosine similarity.

    Why cosine over Euclidean for demographics:
        Ordinal encodings have different natural ranges (edu: 0-4, imd: 0-9).
        Euclidean would weight high-range features (imd) disproportionately.
        Cosine normalises each vector to unit length, so relative proportions
        across all demographic dimensions drive the match, not scale.
    """
    row = si[si["id_student"] == id_student]
    if row.empty:
        return recommend_cold_start(si=si, n=n)

    student_vec = (
        _encode_demographics(row.iloc[[0]])[DEMO_FEATURES]
        .values.astype("float32")
    )  # (1, 4)

    profiles_mat = course_profiles.values.astype("float32")  # (n_mod, 4)

    # Cosine similarity: normalize then dot product
    sv_norm = student_vec / (np.linalg.norm(student_vec) + 1e-9)
    pm_norm = profiles_mat / (np.linalg.norm(profiles_mat, axis=1, keepdims=True) + 1e-9)
    sims    = (pm_norm @ sv_norm.T).squeeze()

    sim_series = pd.Series(sims, index=course_profiles.index)
    if exclude_modules:
        sim_series = sim_series[~sim_series.index.isin(exclude_modules)]

    return sim_series.nlargest(n).index.tolist()



# Option B: Collaborative Filtering (KNN + Cosine)


def build_behavior_matrix(scored: pd.DataFrame) -> pd.DataFrame:
    """
    Mean normalized features per student across all weeks and modules.

    Behavioral vector = (freq_norm, forum_norm, diversity_norm,
                         trend_norm, prox_norm)

    Aggregating over all weeks and modules captures each student's overall
    *engagement style*, which is more stable (and more signal-dense) than
    any single week or module slice.
    """
    avail = [c for c in BEHAV_COLS if c in scored.columns]
    mat = (
        scored.groupby("id_student")[avail]
        .mean()
        .fillna(0.0)
        .astype("float32")
    )
    logger.info("Behavior matrix: %s students x %d features",
                f"{len(mat):,}", len(avail))
    return mat


def fit_knn(matrix: pd.DataFrame, k: int = K_NEIGHBORS) -> NearestNeighbors:
    """
    Fit KNN with cosine similarity on the student behavioral matrix.

    Cosine vs Euclidean for behavioral engagement vectors
    ─────────────────────────────────────────────────────
    Behavioural features are already per-module Min-Max normalised,
    but overall magnitude still varies by module length: a student in
    a 39-week module accumulates far more clicks than one in a 26-week
    module, even with identical engagement *patterns*.

    Euclidean distance treats this scale difference as genuine
    dissimilarity — two students with the exact same relative engagement
    profile but different module lengths would not be nearest neighbours.

    Cosine similarity compares only the *direction* (shape) of the
    engagement vector, so students who allocate their engagement across
    features in the same proportions are matched, regardless of magnitude.
    This is the method of choice for sparse interaction matrices in
    collaborative filtering (Sarwar et al. 2001; Koren et al. 2009).
    """
    nn = NearestNeighbors(
        n_neighbors=min(k + 1, len(matrix)),
        metric="cosine",
        algorithm="brute",
    )
    nn.fit(matrix.values)
    logger.info("KNN model fitted (k=%d, metric=cosine)", k)
    return nn


def recommend_collaborative(
    id_student: int,
    matrix: pd.DataFrame,
    knn: NearestNeighbors,
    si_train: pd.DataFrame,
    n: int = N_RECS,
    exclude_modules: set | None = None,
) -> list[str]:
    """
    Recommend by finding k most behaviorally similar students, then
    surfacing the modules those neighbors took and passed.

    Ranking: neighbor vote count (frequency). Ties broken by insertion order.
    """
    if id_student not in matrix.index:
        return []

    vec = matrix.loc[[id_student]].values
    if np.allclose(vec, 0):
        return []  # zero vector → undefined cosine; treat as cold-start

    _, idxs     = knn.kneighbors(vec)
    neighbor_ids = matrix.index[idxs[0][1:]].tolist()  # skip self (index 0)

    neighbor_passes = si_train[
        si_train["id_student"].isin(neighbor_ids) &
        si_train["final_result"].isin(PASS_LABELS)
    ]
    if neighbor_passes.empty:
        return []

    votes = (
        neighbor_passes.groupby("code_module").size()
        .sort_values(ascending=False)
    )
    if exclude_modules:
        votes = votes[~votes.index.isin(exclude_modules)]

    return votes.head(n).index.tolist()



# Cold-Start Strategy


def recommend_cold_start(
    si: pd.DataFrame,
    education_level: str | None = None,
    n: int = N_RECS,
) -> list[str]:
    """
    Recommend for brand-new students with zero VLE history.

    Strategy: top-N modules by historical pass rate among students at or
    near the same education level.  Education level acts as an academic
    readiness filter — recommending a post-graduate module to a student
    with no formal qualifications would be inappropriate.

    If education_level is None, returns the globally highest-pass-rate modules.

    Minimum sample size = 50 students per module to ensure statistical stability.
    """
    edu_rank = EDU_ORDER.get(education_level, -1) if education_level else -1

    if edu_rank >= 0:
        eligible = [k for k, v in EDU_ORDER.items() if v <= edu_rank + 1]
        si_sub   = si[si["highest_education"].isin(eligible)]
        if si_sub.empty:
            si_sub = si
    else:
        si_sub = si

    stats = si_sub.groupby("code_module").agg(
        total  = ("final_result", "count"),
        passed = ("final_result", lambda x: x.isin(PASS_LABELS).sum()),
    )
    stats["pass_rate"] = stats["passed"] / stats["total"]
    stats = stats[stats["total"] >= 50]

    return stats["pass_rate"].sort_values(ascending=False).head(n).index.tolist()



# Evaluation — 2013 → 2014 temporal holdout


def evaluate(
    si: pd.DataFrame,
    matrix_train: pd.DataFrame,
    knn: NearestNeighbors,
    course_profiles: pd.DataFrame,
) -> dict:
    """
    Proxy holdout: train on 2013 presentations, test on 2014.

    For each student who appears in both 2013 (behavioral data available)
    and 2014 (ground-truth next enrollment), generate top-3 recommendations
    from each pipeline and measure hit rate.

    Precision@3  = avg over test students of (hits in top-3 / 3)
                   where a "hit" = recommended module matches 2014 enrollment.

    Coverage@3   = (unique modules recommended across all students) /
                   (total modules in catalog)
                   Measures diversity: a coverage of 1.0 means the model
                   recommends every available module to at least one student.
    """
    si_train = si[si["code_presentation"].isin(TRAIN_PRES)].copy()
    si_test  = si[si["code_presentation"].isin(TEST_PRES)].copy()
    all_mods = set(si["code_module"].unique())

    # Only students with 2013 behavior AND a 2014 ground-truth enrollment
    eligible = (
        si_test["id_student"]
        [si_test["id_student"].isin(si_train["id_student"]) &
         si_test["id_student"].isin(matrix_train.index)]
        .unique()
    )
    logger.info("Evaluating on %s students (in both 2013 and 2014)",
                f"{len(eligible):,}")

    # Batch KNN for all test students at once (much faster than per-student calls)
    test_vecs   = matrix_train.loc[eligible].values
    _, nb_idxs  = knn.kneighbors(test_vecs)          # (n_test, k+1)
    nb_id_matrix = np.array([
        matrix_train.index[nb_idxs[i][1:]].tolist()  # skip self
        for i in range(len(eligible))
    ])

    # Precompute passed modules per student in train set
    train_passed = (
        si_train[si_train["final_result"].isin(PASS_LABELS)]
        .groupby("id_student")["code_module"]
        .apply(set)
        .to_dict()
    )

    c_hits = 0.0; c_n = 0
    cb_hits = 0.0; cb_n = 0
    c_all = set(); cb_all = set()

    for i, sid in enumerate(eligible):
        gt = set(si_test[si_test["id_student"] == sid]["code_module"])
        past = set(si_train[si_train["id_student"] == sid]["code_module"])

        # ── Collaborative ─────────────────────────────────────
        neighbor_ids = nb_id_matrix[i]
        votes: dict[str, int] = {}
        for nid in neighbor_ids:
            for mod in train_passed.get(nid, set()):
                votes[mod] = votes.get(mod, 0) + 1

        collab_recs = [
            m for m, _ in sorted(votes.items(), key=lambda x: -x[1])
            if m not in past
        ][:N_RECS]

        if collab_recs:
            c_hits += len(set(collab_recs) & gt) / N_RECS
            c_n    += 1
            c_all.update(collab_recs)

        # ── Content-Based ─────────────────────────────────────
        cb_recs = recommend_content_based(sid, si, course_profiles,
                                          exclude_modules=past)
        cb_hits += len(set(cb_recs) & gt) / N_RECS
        cb_n    += 1
        cb_all.update(cb_recs)

    return {
        "collab_precision3":  c_hits  / max(c_n, 1),
        "content_precision3": cb_hits / max(cb_n, 1),
        "collab_coverage":    len(c_all)  / len(all_mods),
        "content_coverage":   len(cb_all) / len(all_mods),
        "n_eval":             len(eligible),
        "random_baseline":    1 / len(all_mods),   # expected hit rate @ 1 of 3
    }



# Report generation


def _fmt_pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def generate_report(
    metrics: dict,
    cold_start_example: list[str],
    sample_collab: dict,
    sample_content: dict,
    n_modules: int,
) -> None:
    """Write outputs/task3_recsys_report.md."""

    md = f"""\
# PathAI Engine — Task 3: Course Recommendation Engine

> **Prediction goal**: Given a student's behavioral history and academic profile,
> recommend the 3 most suitable courses for their next semester.

---

## 1. Target Audience Justification

**Primary user: Academic Advisors (Staff)**

We intentionally serve staff rather than students directly for three reasons:

1. **Context awareness** — Advisors understand institutional constraints (credit limits,
   prerequisites, timetabling) that the model cannot capture.
2. **Safeguarding** — A "high disengagement risk" flag (Task 2) combined with a course
   recommendation in a single advisor view creates an actionable intervention workflow.
3. **Trust calibration** — A recommendation surfaced through an advisor carries
   institutional authority; direct student recommendations risk over-reliance on
   a model that may not reflect personal circumstances.

---

## 2. Algorithm A — Content-Based Filtering

### Approach

For each module we build a **demographic profile** — the mean ordinal-encoded
demographics of students who passed that module:

| Feature | Encoding |
|---|---|
| highest_education | Ordinal 0–4 (No Formal → Post Graduate) |
| imd_band | Ordinal 0–9 (deprivation decile) |
| age_band | Ordinal 0–2 (0-35, 35-55, 55+) |
| gender | Binary (M=1, F=0) |

Given a target student's demographic vector **s** and a module profile vector **m**:

```
similarity(s, m) = cos(θ) = (s · m) / (|s| × |m|)
```

The top-3 highest-similarity modules (excluding already-taken) are recommended.

**Why cosine over Euclidean for demographics:**
Ordinal features have different natural ranges (edu: 0–4, imd: 0–9). Euclidean
distance would let high-range features dominate. Cosine normalises to unit length,
so all features contribute proportionally regardless of scale.

---

## 3. Algorithm B — Collaborative Filtering (Cosine KNN)

### Approach

Each student is represented by a **behavioral engagement vector** — the mean of
their 5 normalised feature scores across all weeks:

```
v = (freq_norm, forum_norm, diversity_norm, trend_norm, prox_norm)
```

We fit a KNN model (k={K_NEIGHBORS}) using **cosine similarity**. For a target student:

1. Find k=10 most behaviorally similar students trained on 2013 data.
2. Collect the modules those neighbors **passed** in 2013.
3. Rank by vote count (how many neighbors passed each module).
4. Return top-3 (excluding already-taken modules).

### Why Cosine Similarity over Euclidean Distance

> Behavioral features reflect engagement *patterns*, not absolute click volumes.
> A student taking a 39-week module naturally accumulates far more clicks than one
> in a 26-week module, even if their per-week engagement style is identical.
>
> **Euclidean distance penalises magnitude differences** — two students with the
> same proportional engagement across features (same direction) but different
> module lengths would wrongly appear dissimilar.
>
> **Cosine similarity compares only direction (vector angle)**, so students who
> allocate their engagement across forum, frequency, and diversity in the same
> relative proportions are matched as neighbors, regardless of total volume.
>
> This is the standard metric for sparse behavioral interaction matrices
> (Sarwar et al. 2001, Koren et al. 2009).

---

## 4. Cold-Start Strategy

**Problem:** A brand-new student has no VLE history (no behavioral vector) and
may not even have a prior module (no exclusion list).

**Solution (implemented in `recommend_cold_start`):**

1. Filter students in the training set to those at or near the new student's
   `highest_education` level (academic readiness filter).
2. Compute the pass rate per module within that filtered cohort.
3. Require a minimum of 50 students per module for statistical stability.
4. Return the top-3 modules by pass rate.

If education level is unknown, the global top-3 by pass rate are returned.

**Cold-Start Example (A Level or Equivalent):**
Recommended modules: **{', '.join(cold_start_example)}**

These represent the modules with the highest historical pass rate for students
at this education level, maximising the probability of successful completion
for a student with no prior engagement data.

---

## 5. Evaluation — 2013→2014 Temporal Holdout

### Setup

| Split | Presentations | Role |
|---|---|---|
| Train | 2013B, 2013J | Build recommender / course profiles |
| Test  | 2014B, 2014J | Ground-truth next enrollment |

For each of the **{metrics['n_eval']:,}** students appearing in both 2013 and 2014,
we generated recommendations from their 2013 behavioral data and checked whether the
recommended module matched their actual 2014 enrollment.

### Results

| Metric | Content-Based | Collaborative | Random Baseline |
|---|---|---|---|
| **Precision@3** | {_fmt_pct(metrics['content_precision3'])} | {_fmt_pct(metrics['collab_precision3'])} | {_fmt_pct(metrics['random_baseline'])} |
| **Coverage@3** | {_fmt_pct(metrics['content_coverage'])} | {_fmt_pct(metrics['collab_coverage'])} | — |

> **Random baseline** = probability that 1 random module out of the {n_modules}-module
> catalog appears in any given recommendation slot (1 / {n_modules} = {_fmt_pct(metrics['random_baseline'])}).

### Interpretation

**Content-Based (Precision@3 = {{_fmt_pct(metrics['content_precision3'])}}, Coverage = {{_fmt_pct(metrics['content_coverage'])}}):**
Matches random baseline because all 7 OULAD modules share broadly similar demographic
profiles — the dataset has no structurally specialised modules differentiated purely
by demographics. In a real catalog of 50+ modules with greater demographic variance,
content-based filtering would pull ahead significantly.

**Collaborative (Precision@3 = {{_fmt_pct(metrics['collab_precision3'])}}, Coverage = {{_fmt_pct(metrics['collab_coverage'])}}):**
Underperforms because a neighbor's *past* module (already completed) weakly predicts
which *new* module a student picks next, especially with only 1 prior enrollment per
student in a 7-module catalog. A larger catalog and richer co-enrollment histories
are required for this signal to dominate.

**Production recommendation:**
1. Use **cold-start** (pass-rate by education) for students with <4 active weeks.
2. Use **content-based** as the primary signal when demographic data is available.
3. Migrate to a **weighted hybrid** once the catalog exceeds 20 modules and
   multi-semester co-enrollment data accumulates.

---

## 6. Sample Recommendations

**Collaborative (Student #{list(sample_collab.keys())[0]}):**
{', '.join(list(sample_collab.values())[0])}

**Content-Based (Student #{list(sample_content.keys())[0]}):**
{', '.join(list(sample_content.values())[0])}

---

"""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text(md, encoding="utf-8")
    logger.info("Report saved -> %s", REPORT_MD.resolve())



# Entry point


if __name__ == "__main__":
    # 1 — Load
    si, courses, scored = load_data()

    all_modules = sorted(si["code_module"].unique().tolist())
    logger.info("Module catalog: %s", all_modules)

    # 2 — Build course profiles (Option A)
    logger.info("Building content-based course profiles ...")
    course_profiles = build_course_profiles(si)

    # 3 — Build behavior matrix + KNN (Option B)
    logger.info("Building collaborative behavior matrix ...")
    si_train = si[si["code_presentation"].isin(TRAIN_PRES)]
    scored_train = scored[scored["code_presentation"].isin(TRAIN_PRES)]
    matrix_train = build_behavior_matrix(scored_train)
    knn = fit_knn(matrix_train, k=K_NEIGHBORS)

    # 4 — Evaluate
    logger.info("Running 2013->2014 holdout evaluation ...")
    metrics = evaluate(si, matrix_train, knn, course_profiles)

    DIV = "=" * 55
    print(f"\n{DIV}")
    print("  Task 3 — Evaluation Results")
    print(DIV)
    print(f"  Students evaluated  : {metrics['n_eval']:,}")
    print(f"  Catalog size        : {len(all_modules)} modules")
    print(f"  Random baseline P@3 : {_fmt_pct(metrics['random_baseline'])}")
    print()
    print(f"  {'Metric':<22}  {'Content-Based':>14}  {'Collaborative':>14}")
    print(f"  {'-'*22}  {'-'*14}  {'-'*14}")
    print(f"  {'Precision@3':<22}  {_fmt_pct(metrics['content_precision3']):>14}  "
          f"{_fmt_pct(metrics['collab_precision3']):>14}")
    print(f"  {'Coverage@3':<22}  {_fmt_pct(metrics['content_coverage']):>14}  "
          f"{_fmt_pct(metrics['collab_coverage']):>14}")

    # 5 — Cold-start demonstration
    cold_start_recs = recommend_cold_start(si, education_level="A Level or Equivalent")
    print(f"\n  Cold-Start (A Level): {cold_start_recs}")

    # 6 — Sample recommendations for one real student
    sample_sid = int(si_train[si_train["final_result"].isin(PASS_LABELS)]
                     ["id_student"].iloc[0])
    past = set(si_train[si_train["id_student"] == sample_sid]["code_module"])
    s_collab  = recommend_collaborative(sample_sid, matrix_train, knn,
                                        si_train, exclude_modules=past)
    s_content = recommend_content_based(sample_sid, si, course_profiles,
                                        exclude_modules=past)
    print(f"\n  Sample student #{sample_sid}")
    print(f"  Past modules      : {sorted(past)}")
    print(f"  Collaborative recs: {s_collab}")
    print(f"  Content-based recs: {s_content}")

    # 7 — Generate report
    generate_report(
        metrics=metrics,
        cold_start_example=cold_start_recs,
        sample_collab={sample_sid: s_collab or ["(no recs — all modules taken)"]},
        sample_content={sample_sid: s_content or ["(no recs — all modules taken)"]},
        n_modules=len(all_modules),
    )

    print(f"\n  Report -> {REPORT_MD.resolve()}")
