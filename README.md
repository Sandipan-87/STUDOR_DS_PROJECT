# PathAI Engine — Student Engagement & Risk Intelligence Platform

**PathAI Engine** is a production-grade behavioral analytics pipeline built on the Open University Learning Analytics Dataset (OULAD). It transforms raw clickstream data into a dynamic student engagement score, a Week-6 disengagement early-warning system, and a personalized course recommendation engine — designed to give university advisors actionable intelligence before it is too late.

---

## Project Structure

```
STUDOR_DS/
│
├── data/                          # Raw OULAD CSV files (not committed — see setup)
│   ├── studentInfo.csv
│   ├── studentVle.csv
│   ├── vle.csv
│   ├── assessments.csv
│   ├── studentAssessment.csv
│   └── courses.csv
│
├── src/                           # Production pipeline modules
│   ├── data_loader.py             # Task 1 — memory-efficient OULAD ingestion
│   ├── feature_engineering.py     # Task 1 — 6 behavioral features (incl. EMA)
│   ├── scoring.py                 # Task 1 — weighted engagement score [0–100]
│   ├── visualize_trajectories.py  # Task 1 — archetype trajectory plot
│   ├── task2_predictive_model.py  # Task 2 — XGBoost disengagement classifier
│   └── task3_recommender.py       # Task 3 — content-based + collaborative recs
│
├── tests/
│   └── final_audit.py             # Pre-submission: overfitting + leakage audit
│
├── outputs/                       # Generated artifacts (auto-created at runtime)
│   ├── scored_features.parquet    # Full behavioral feature matrix (all weeks)
│   ├── task1_trajectories.png     # Archetype engagement trajectory plot
│   ├── task2_staff_alert_ui.md    # Advisor-facing at-risk student notification
│   └── task3_recsys_report.md     # Recommendation engine methodology report
│
├── requirements.txt               # Pinned dependencies
└── README.md
```

---

## Quickstart — Full Reproduction

### Prerequisites

- Python **3.11** or 3.12
- Git
- ~2 GB free disk space (for dataset + outputs)

---

### Step 1 — Clone the repository

```bash
git clone https://github.com/Sandipan-87/STUDOR_DS_PROJECT.git
cd STUDOR_DS
```

---

### Step 2 — Create and activate a virtual environment

**macOS / Linux:**
```bash
python3.11 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

---

### Step 3 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> All versions are pinned. Exact reproduction is guaranteed on Python 3.11+.

---

### Step 4 — Download the OULAD dataset

1. Go to [Kaggle — OULAD Dataset](https://www.kaggle.com/datasets/anlgrbz/student-demographics-online-education-dataoulad)
2. Download and unzip the archive.
3. Place **all CSV files** directly inside the `data/` folder:

```
data/
├── assessments.csv
├── courses.csv
├── studentAssessment.csv
├── studentInfo.csv
├── studentVle.csv
└── vle.csv
```

> **Alternative dataset:** If OULAD is unavailable, the [xAPI-Edu-Data dataset](https://www.kaggle.com/datasets/aljarah/xAPI-Edu-Data) on Kaggle is an accepted substitute as noted in the project brief.

---

### Step 5 — Run the pipeline (in order)

Each script is fully self-contained. Run them in the exact sequence below:

```bash
# Task 1 — Data ingestion and feature engineering
python src/data_loader.py
python src/feature_engineering.py
python src/scoring.py

# Task 1 — Generate archetype trajectory visualization
python src/visualize_trajectories.py

# Task 2 — Predictive disengagement model + staff alert UI
python src/task2_predictive_model.py

# Task 3 — Course recommendation engine + evaluation report
python src/task3_recommender.py
```

**Expected runtime:** ~4–6 minutes end-to-end on a standard laptop.

**Outputs generated:**

| File | Description |
|---|---|
| `outputs/scored_features.parquet` | Full behavioral feature matrix (579K rows × 18 cols) |
| `outputs/task1_trajectories.png` | Week-over-week engagement plot for 3 student archetypes |
| `outputs/task2_staff_alert_ui.md` | Advisor notification mockup for highest-risk student |
| `outputs/task3_recsys_report.md` | Rec engine methodology, evaluation results, and cold-start doc |

---

### Step 6 — Run the pre-submission audit

```bash
python tests/final_audit.py
```

This runs **12 automated checks** covering:
- **Overfitting** — Train vs. Test AUC and Recall delta for the Task 2 XGBoost model
- **Feature leakage** — Confirms `final_result` labels were never used to compute `engagement_score` (statistical + code-path inspection)
- **Data consistency** — Validates all Task 3 evaluation student IDs against `studentInfo.csv`

**Expected output:**
```
  AUDIT SUMMARY  —  12 checks total
  [PASS]  12 checks passed

  REPOSITORY IS CLEAN — safe to submit.
```

---

## Task Summaries

### Task 1 — Behavioral Scoring Framework

Engineers 6 features from VLE clickstream data:

| Feature | Signal |
|---|---|
| `freq` | Total weekly clicks |
| `ema_freq` | 3-week Exponential Moving Average of clicks (momentum) |
| `diversity` | Unique activity types accessed per week |
| `forum_clicks` | Collaboration and discussion activity |
| `prox_activity` | Elevated engagement within 2 weeks of an assessment |
| `trend_slope` | Week-over-week click trajectory (winsorized at p5/p95) |

Engagement score is a weighted composite normalized to **[0, 100]** using per-module Min-Max scaling + global linear rescale.

### Task 2 — Predictive Disengagement Model

XGBoost binary classifier trained **strictly on Week ≤ 6 data** (zero leakage).  
Optimized for **Recall** (FN cost >> FP cost in an early-warning system):

| Metric | Value |
|---|---|
| ROC-AUC | 0.734 |
| Recall | **0.948** |
| Precision | 0.500 |
| F2 (β=2) | 0.804 |

Staff alert triggered at **35% risk threshold** with plain-English risk factor explanations and 3 recommended advisor actions.

### Task 3 — Course Recommendation Engine

Two recommendation approaches + cold-start, evaluated via 2013→2014 temporal holdout (1,653 students):

| Approach | Precision@3 | Coverage@3 |
|---|---|---|
| Content-Based (demographic cosine) | 14.3% | 100% |
| Collaborative KNN (behavioral cosine) | 0.7% | 85.7% |
| Random baseline | 14.3% | — |

Both pipelines use **cosine similarity** (justified over Euclidean in code docstrings). Cold-start recommends top modules by historical pass rate, filtered by the student's education level.

---

## Design Decisions & Limitations

- **Why Recall over Precision?** Missing a true dropout (FN) has ~6× the cost of a false alarm (FP). The asymmetric `scale_pos_weight` and 0.35 decision threshold encode this business reality explicitly.
- **Why cosine similarity for collaborative filtering?** Engagement features are already Min-Max normalised but differ in magnitude by module length. Cosine corrects for this by comparing vector direction only.
- **Collaborative filtering limitation:** With only 7 OULAD modules, the KNN approach has limited catalog to differentiate between. In production with 50+ modules and multi-semester histories, collaborative would dominate.
- **What I'd do differently with more time:** Train on a rolling temporal window (not a static 2013 split), add SHAP waterfall plots per student to the staff alert, and build the hybrid recommendation model.

---

## Repository

Built for the **Studor DS Screening Assessment** by the PathAI Engine team.  
Dataset: [OULAD — Open University Learning Analytics Dataset](https://analyse.kmi.open.ac.uk/open_dataset)
