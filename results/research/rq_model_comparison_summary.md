# Central Research Question — Model Comparison

**Question:** How does the proposed SHAP-guided Multinomial Logistic Regression (MLR) model compare with baseline Naive Bayes (NB) and Support Vector Machine (SVM) models in multiclass SMS classification?

## Summary

- Highest overall macro F1 on the held-out test set: **SVM** (0.9635)
- Proposed MLR macro F1: **0.9425**
- MLR smishing F1: **0.9275**

The proposed MLR model provides direct class probabilities and serves as the primary explainable model via SHAP. Baseline NB and SVM are included for performance comparison under the same preprocessing and feature pipeline.

## Model Comparison (Test Set)

| Model | Accuracy | Macro F1 | Ham F1 | Spam F1 | Smishing F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| NB | 0.9229 | 0.9222 | 0.9721 | 0.8856 | 0.9088 |
| SVM | 0.9639 | 0.9635 | 0.9934 | 0.9478 | 0.9494 |
| MLR | 0.9434 | 0.9425 | 0.9891 | 0.9110 | 0.9275 |

## Related Reports

- `rq1_explainability_summary.md` — SHAP explainability for smishing
- `rq2_performance_and_hypothesis_summary.md` — per-class metrics and H01
