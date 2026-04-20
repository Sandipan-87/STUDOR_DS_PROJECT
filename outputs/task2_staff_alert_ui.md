# PathAI — At-Risk Student Alert

> **Prediction Point**: End of Week 6 &nbsp;|&nbsp; Generated: 20 Apr 2026, 11:35

---

## Student Overview

| Field | Value |
|---|---|
| Student ID | `629394` |
| Module | **FFF** |
| Risk Level | HIGH |
| Disengagement Probability | **98%** |
| Engagement Score at Week 6 | 8.9 / 100 |
| Weeks Active (of 6) | 1 |

> A score below **30 / 100** places a student in the High Disengagement Risk Zone.
> This alert fires at Week 6 — early enough for a meaningful intervention.

---

## Why PathAI Raised This Flag

### Risk Factor 1 — Active Weeks out of 6
**Model value: 1.0**

Students active in fewer than 4 of the first 6 weeks have a 3× higher withdrawal rate. Even one zero-activity week in the first fortnight is a significant early warning signal.

### Risk Factor 2 — Total Platform Clicks (Weeks 1–6)
**Model value: 24.00**

Raw click volume captures the quantity of platform interaction. Below ~150 total clicks in six weeks signals insufficient exposure to course material to pass assessments.

---

## Recommended Advisor Actions

| # | Action | Timeline |
|---|---|---|
| 1 | **Send a personalised check-in email** — acknowledge their effort; ask if they need support or are facing any barriers | Within 48 hours |
| 2 | **Schedule a 15-min virtual meeting** — review their assessment timeline and current study plan together | This week |
| 3 | **Connect to learning support services** — share links to academic skills centre, peer study groups, or tutoring | Within 1 week |

---

## Model Transparency

| Metric | Value |
|---|---|
| Model | XGBoost Binary Classifier |
| Training window | Weeks 1–6 (zero data leakage) |
| Alert threshold | Probability ≥ 35% |
| ROC-AUC | 0.734 |
| Recall (test set) | 0.948 |
| Precision (test set) | 0.500 |

> *PathAI uses only VLE clickstream and engagement data.*
> *No demographic information (age, gender, location) is used in this prediction.*

---

