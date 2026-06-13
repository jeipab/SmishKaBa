# Research Question 2 — Model Performance

**Subquestion:** Is there a significant difference between the baseline NB model, baseline SVM model, and proposed MLR model in terms of class-specific performance for ham, spam, and smishing?

## Overall Model Ranking

Models ranked by macro F1, then weighted F1, then accuracy:

| Rank | Model | Accuracy | Macro F1 | Weighted F1 |
| ---: | --- | ---: | ---: | ---: |
| 1 | SVM | 0.9644 | 0.9640 | 0.9644 |
| 2 | MLR | 0.9469 | 0.9462 | 0.9466 |
| 3 | NB | 0.9214 | 0.9200 | 0.9205 |

## Class-Specific Performance

### Ham

| Model | Precision | Recall | F1-Score |
| --- | ---: | ---: | ---: |
| NB | 0.9373 | 0.9912 | 0.9635 |
| SVM | 0.9971 | 1.0000 | 0.9985 |
| MLR | 0.9755 | 0.9985 | 0.9869 |

### Spam

| Model | Precision | Recall | F1-Score |
| --- | ---: | ---: | ---: |
| NB | 0.9118 | 0.8480 | 0.8787 |
| SVM | 0.9726 | 0.9179 | 0.9445 |
| MLR | 0.9438 | 0.8936 | 0.9180 |

### Smishing

| Model | Precision | Recall | F1-Score |
| --- | ---: | ---: | ---: |
| NB | 0.9130 | 0.9227 | 0.9179 |
| SVM | 0.9252 | 0.9742 | 0.9491 |
| MLR | 0.9205 | 0.9470 | 0.9335 |

## Best Model per Metric

| Metric | Best Model |
| --- | --- |
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
