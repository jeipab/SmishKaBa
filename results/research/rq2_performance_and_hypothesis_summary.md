# Research Question 2 — Model Performance

**Subquestion:** Is there a significant difference between the baseline NB model, baseline SVM model, and proposed MLR model in terms of class-specific performance for ham, spam, and smishing?

## Overall Model Ranking

Models ranked by macro F1, then weighted F1, then macro recall:

| Rank | Model | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | SVM | 0.9642 | 0.9636 | 0.9635 | 0.9638 |
| 2 | MLR | 0.9435 | 0.9428 | 0.9425 | 0.9430 |
| 3 | NB | 0.9226 | 0.9223 | 0.9222 | 0.9227 |

## Class-Specific Performance

### Ham

| Model | Precision | Recall | F1-Score |
| --- | ---: | ---: | ---: |
| NB | 0.9706 | 0.9735 | 0.9721 |
| SVM | 0.9869 | 1.0000 | 0.9934 |
| MLR | 0.9784 | 1.0000 | 0.9891 |

### Spam

| Model | Precision | Recall | F1-Score |
| --- | ---: | ---: | ---: |
| NB | 0.9075 | 0.8647 | 0.8856 |
| SVM | 0.9728 | 0.9240 | 0.9478 |
| MLR | 0.9460 | 0.8784 | 0.9110 |

### Smishing

| Model | Precision | Recall | F1-Score |
| --- | ---: | ---: | ---: |
| NB | 0.8897 | 0.9288 | 0.9088 |
| SVM | 0.9327 | 0.9667 | 0.9494 |
| MLR | 0.9061 | 0.9500 | 0.9275 |

## Best Model per Metric

| Metric | Best Model |
| --- | --- |
| Macro Precision | SVM |
| Macro Recall | SVM |
| Macro F1 | SVM |
| Ham Precision | SVM |
| Ham Recall | SVM |
| Ham F1 | SVM |
| Spam Precision | SVM |
| Spam Recall | SVM |
| Spam F1 | SVM |
| Smishing Precision | SVM |
| Smishing Recall | SVM |
| Smishing F1 | SVM |

## Hypothesis Testing (H01)

**Conclusion:** H01 is rejected at alpha=0.05: at least one pairwise McNemar test found a significant difference in overall classification performance (NB vs SVM, NB vs MLR, SVM vs MLR).

Pairwise **McNemar tests** were run on overall correct/incorrect predictions for the same test set (alpha = 0.05).

| Comparison | p-value | Significant | Result |
| --- | ---: | --- | --- |
| NB vs SVM | 0.0000 | Yes | SVM performs better |
| NB vs MLR | 0.0000 | Yes | MLR performs better |
| SVM vs MLR | 0.0000 | Yes | SVM performs better |

## Source Files

- `results\model_comparison.csv`
- `results\evaluation_summary.json`
- `results\statistical_tests\model_pair_comparisons.csv`
- `results\statistical_tests\hypothesis_test_summary.json`
