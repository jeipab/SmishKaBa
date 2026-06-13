# Central Research Question — Model Comparison

**Question:** How does the proposed SHAP-guided Multinomial Logistic Regression (MLR) model compare with baseline Naive Bayes (NB) and Support Vector Machine (SVM) models in multiclass SMS classification?

## Summary

- Highest overall macro F1 on the held-out test set: **SVM** (0.9640)
- Proposed MLR macro F1: **0.9462**
- MLR smishing F1: **0.9335**

The proposed MLR model provides direct class probabilities and serves as the primary explainable model via SHAP. Baseline NB and SVM are included for performance comparison under the same preprocessing and feature pipeline.

## Model Comparison (Test Set)

| Model | Accuracy | Macro F1 | Ham F1 | Spam F1 | Smishing F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| NB | 0.9214 | 0.9200 | 0.9635 | 0.8787 | 0.9179 |
| SVM | 0.9644 | 0.9640 | 0.9985 | 0.9445 | 0.9491 |
| MLR | 0.9469 | 0.9462 | 0.9869 | 0.9180 | 0.9335 |

## Related Reports

- `rq1_explainability_summary.md` — SHAP explainability for smishing
- `rq2_performance_and_hypothesis_summary.md` — per-class metrics and H01
