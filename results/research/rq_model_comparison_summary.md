# Central Research Question — Model Comparison

**Question:** How does the proposed SHAP-guided Multinomial Logistic Regression (MLR) model compare with baseline Naive Bayes (NB) and Support Vector Machine (SVM) models in multiclass SMS classification?

## Summary

- Highest overall macro F1 on the held-out test set: **SVM** (0.9635)
- Proposed MLR macro F1: **0.9425**
- MLR smishing F1: **0.9275**

The proposed MLR model provides direct class probabilities and serves as the primary explainable model via SHAP. Baseline NB and SVM are included for performance comparison under the same preprocessing and feature pipeline.

## Model Comparison (Test Set)

| Model | Macro Precision | Macro Recall | Macro F1 | Ham P | Ham R | Ham F1 | Spam P | Spam R | Spam F1 | Smishing P | Smishing R | Smishing F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NB | 0.9226 | 0.9223 | 0.9222 | 0.9706 | 0.9735 | 0.9721 | 0.9075 | 0.8647 | 0.8856 | 0.8897 | 0.9288 | 0.9088 |
| SVM | 0.9642 | 0.9636 | 0.9635 | 0.9869 | 1.0000 | 0.9934 | 0.9728 | 0.9240 | 0.9478 | 0.9327 | 0.9667 | 0.9494 |
| MLR | 0.9435 | 0.9428 | 0.9425 | 0.9784 | 1.0000 | 0.9891 | 0.9460 | 0.8784 | 0.9110 | 0.9061 | 0.9500 | 0.9275 |

## Related Reports

- `rq1_explainability_summary.md` — SHAP explainability for smishing
- `rq2_performance_and_hypothesis_summary.md` — per-class metrics and H01
