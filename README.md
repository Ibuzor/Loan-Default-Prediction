# Home Equity Loan Default Prediction

A binary classification model that predicts home equity loan default risk,
built with ECOA (Equal Credit Opportunity Act) compliance and interpretability
as first-class requirements rather than an afterthought.

## Problem

A bank's consumer credit department wants to replace manual underwriting of
home equity applications with a model-driven process. This has to be achieved without inheriting the biases and inconsistency of human judgment, and while keeping every rejection
traceable and defensible under ECOA.

## Approach

- **Data**: HMEQ (Home Equity) dataset — 5,960 loan records, ~20% default rate
- **Models compared**: Logistic Regression (baseline), Decision Tree, Random
  Forest, Explainable Boosting Machine (EBM) and XGBoost (with monotonic constraints on key features for
  ECOA-defensibility)
- **Evaluation**: precision/recall/F1/ROC-AUC prioritized over raw accuracy,
  reflecting the cost asymmetry between missed defaulters (false negatives)
  and wrongly rejected reliable borrowers (false positives)
- **Missingness**: DEBTINC and VALUE missingness is treated as signal (MNAR),
  not noise. Binary missingness indicators are retained as model features
- **Interpretability**: SHAP (TreeSHAP) for feature-level explanations
- **Bonus**: a RAG-based Adverse Action Notice Generator that drafts
  ECOA-compliant rejection reasons from SHAP output, grounded in Regulation B
  (12 CFR Part 1002 Appendix C) reason codes

## Result

XGBoost (with monotonic constraints) is the recommended model, chosen for
recall performance under the cost-asymmetry framing established above.

## Repo structure

- `Loan_Default_Prediction.ipynb` — full analysis: EDA, modeling, evaluation,
  SHAP interpretability, and the adverse action notice generator
- `hmeq.csv` — dataset
- `requirements.txt` — Python dependencies

## Running it

```bash
pip install -r requirements.txt
jupyter notebook Loan_Default_Prediction.ipynb
```

## Limitations

A full proxy-variable or disparate-impact bias audit is out of scope for this
version and so therefore flagged as an open limitation rather than treated as solved.

## Context

Built as a capstone project for MIT Professional Education's Applied AI & Data
Science program.

## Graded Results

**Final score: 94/100 (Grade Excellent)** - top 10% cutoff was 97

| Assessment | Score |
|---|---|
| Milestone Submission | 20/20 |
| Live Presentation | 34/40 |
| Final Submission | 40/40 |

**Instructor feedback (Final Submission):** "You have done an excellent job on the
overall assignment. I really liked the technical accuracy, and the explanation
stood out."

**Instructor feedback (Live Presentation):** "Good structure and flow, decent
understanding of the solution, and positive notes on the fairness/ECOA angle
and the GenAI adverse-action-notice feature. Tighten time
management so the analysis section doesn't run long."