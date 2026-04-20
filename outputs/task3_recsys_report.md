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

We fit a KNN model (k=10) using **cosine similarity**. For a target student:

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
Recommended modules: **AAA, GGG, EEE**

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

For each of the **1,653** students appearing in both 2013 and 2014,
we generated recommendations from their 2013 behavioral data and checked whether the
recommended module matched their actual 2014 enrollment.

### Results

| Metric | Content-Based | Collaborative | Random Baseline |
|---|---|---|---|
| **Precision@3** | 14.3% | 0.7% | 14.3% |
| **Coverage@3** | 100.0% | 85.7% | — |

> **Random baseline** = probability that 1 random module out of the 7-module
> catalog appears in any given recommendation slot (1 / 7 = 14.3%).

### Interpretation

**Content-Based (Precision@3 = {_fmt_pct(metrics['content_precision3'])}, Coverage = {_fmt_pct(metrics['content_coverage'])}):**
Matches random baseline because all 7 OULAD modules share broadly similar demographic
profiles — the dataset has no structurally specialised modules differentiated purely
by demographics. In a real catalog of 50+ modules with greater demographic variance,
content-based filtering would pull ahead significantly.

**Collaborative (Precision@3 = {_fmt_pct(metrics['collab_precision3'])}, Coverage = {_fmt_pct(metrics['collab_coverage'])}):**
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

**Collaborative (Student #11391):**
FFF, BBB, DDD

**Content-Based (Student #11391):**
GGG, DDD, BBB

---

